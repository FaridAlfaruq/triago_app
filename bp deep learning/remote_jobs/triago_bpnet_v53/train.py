"""
Remote GPU Training: TriaGO_BPNet v5.3 Precision
Trained on Kaggle GPU using clean dataset_7ch.npz.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Model Architecture: TriaGO_BPNet v5.3 Precision (374,850 params)
# ==============================================================================

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, filters: int = 32):
        super().__init__()
        branch_filters = filters // 4
        self.b1 = nn.Conv1d(in_channels, branch_filters, kernel_size=1, padding=0)
        self.b2 = nn.Conv1d(in_channels, branch_filters, kernel_size=3, padding=1)
        self.b3 = nn.Conv1d(in_channels, branch_filters, kernel_size=7, padding=3)
        self.b4 = nn.Conv1d(in_channels, branch_filters, kernel_size=15, padding=7)
        self.act = nn.SiLU()
        self.bn = nn.BatchNorm1d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.act(self.b1(x))
        b2 = self.act(self.b2(x))
        b3 = self.act(self.b3(x))
        b4 = self.act(self.b4(x))
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.bn(out)


class TriaGoBPNet(nn.Module):
    def __init__(self, in_channels: int = 7):
        super().__init__()
        # 1. Stem
        self.stem_conv = nn.Conv1d(in_channels, 64, kernel_size=11, padding=5)
        self.stem_act = nn.SiLU()
        self.stem_bn = nn.BatchNorm1d(64)

        # 2. Inceptions
        self.inc1 = InceptionBlock1D(64, filters=64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout1d(0.15)

        self.inc2 = InceptionBlock1D(64, filters=128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout1d(0.20)

        self.inc3 = InceptionBlock1D(128, filters=128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 3. BiGRU
        self.bigru = nn.GRU(128, 64, batch_first=True, bidirectional=True)

        # 4. MHSA
        self.mhsa = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(128)

        # 5. Dual Global Context Pooling
        self.context_drop = nn.Dropout(0.25)

        # 6. Regressor Heads
        self.sbp_dense = nn.Linear(256, 80)
        self.sbp_act = nn.SiLU()
        self.sbp_bn = nn.BatchNorm1d(80)
        self.sbp_drop = nn.Dropout(0.20)
        self.sbp_out = nn.Linear(80, 1)

        self.dbp_dense = nn.Linear(256, 80)
        self.dbp_act = nn.SiLU()
        self.dbp_bn = nn.BatchNorm1d(80)
        self.dbp_drop = nn.Dropout(0.20)
        self.dbp_out = nn.Linear(80, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] == 1250 and x.shape[2] == 7:
            x = x.transpose(1, 2)

        x = self.stem_bn(self.stem_act(self.stem_conv(x)))
        x = self.drop1(self.pool1(self.inc1(x)))
        x = self.drop2(self.pool2(self.inc2(x)))
        x = self.pool3(self.inc3(x))

        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)

        attn_out, _ = self.mhsa(gru_out, gru_out, gru_out)
        x = self.ln(gru_out + attn_out)

        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        context = torch.cat([avg_pool, max_pool], dim=1)
        context = self.context_drop(context)

        s = self.sbp_drop(self.sbp_bn(self.sbp_act(self.sbp_dense(context))))
        sbp_pred = self.sbp_out(s).squeeze(-1)

        d = self.dbp_drop(self.dbp_bn(self.dbp_act(self.dbp_dense(context))))
        dbp_pred = self.dbp_out(d).squeeze(-1)

        return sbp_pred, dbp_pred


# ==============================================================================
# Physics & BHS Loss
# ==============================================================================

class PhysicsBHSLoss(nn.Module):
    def __init__(self, sbp_mean: float, sbp_std: float, dbp_mean: float, dbp_std: float):
        super().__init__()
        self.sbp_mean = sbp_mean
        self.sbp_std = sbp_std
        self.dbp_mean = dbp_mean
        self.dbp_std = dbp_std
        self.huber_sbp = nn.HuberLoss(delta=1.0)
        self.huber_dbp = nn.HuberLoss(delta=0.8)

    def _soft_exceedance(self, err_abs: torch.Tensor, threshold: float, k: float) -> torch.Tensor:
        return torch.sigmoid(k * (err_abs - threshold)).mean()

    def forward(
        self,
        pred_s_z: torch.Tensor,
        pred_d_z: torch.Tensor,
        true_s_z: torch.Tensor,
        true_d_z: torch.Tensor,
        k: float = 1.0,
    ) -> torch.Tensor:
        l_sbp = self.huber_sbp(pred_s_z, true_s_z)
        l_dbp = self.huber_dbp(pred_d_z, true_d_z)

        s_true = true_s_z * self.sbp_std + self.sbp_mean
        d_true = true_d_z * self.dbp_std + self.dbp_mean
        s_pred = pred_s_z * self.sbp_std + self.sbp_mean
        d_pred = pred_d_z * self.dbp_std + self.dbp_mean

        map_true = d_true + (s_true - d_true) / 3.0
        map_pred = d_pred + (s_pred - d_pred) / 3.0
        l_map = torch.abs(map_pred - map_true).mean() / self.sbp_std

        pp_true = s_true - d_true
        pp_pred = s_pred - d_pred
        l_pp = torch.abs(pp_pred - pp_true).mean() / self.sbp_std

        sbp_err_abs = torch.abs(s_pred - s_true)
        dbp_err_abs = torch.abs(d_pred - d_true)

        l_sbp_bhs = 0.8 * self._soft_exceedance(sbp_err_abs, 10.0, k) + 0.6 * self._soft_exceedance(sbp_err_abs, 15.0, k)
        l_dbp_bhs = 1.0 * self._soft_exceedance(dbp_err_abs, 5.0, k) + 1.2 * self._soft_exceedance(dbp_err_abs, 14.0, k)

        total_loss = (
            (1.40 * l_sbp)
            + (1.00 * l_dbp)
            + (0.10 * l_map)
            + (0.10 * l_pp)
            + (0.60 * l_sbp_bhs)
            + (0.60 * l_dbp_bhs)
        )
        return total_loss


# ==============================================================================
# Helper: Locate Dataset File Dynamically
# ==============================================================================

def find_dataset_file() -> Path:
    candidates = [
        Path("/kaggle/input/triago-clean-dataset-7ch/dataset_7ch.npz"),
        Path("./dataset_7ch.npz"),
        Path("../dataset_7ch.npz"),
        Path(__file__).resolve().parent / "dataset_7ch.npz",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Search /kaggle/input recursively
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        found = list(kaggle_input.glob("**/dataset_7ch.npz"))
        if found:
            return found[0]

    raise FileNotFoundError("dataset_7ch.npz could not be located!")


def prepare_dataloaders(
    dataset_path: Path,
    test_size: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, float]]:
    data = np.load(dataset_path)
    X = data["X"]
    y_sbp = data["y_sbp"]
    y_dbp = data["y_dbp"]
    groups = data["groups"]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(X, groups=groups))

    sbp_mean = float(y_sbp[train_idx].mean())
    sbp_std = float(y_sbp[train_idx].std())
    dbp_mean = float(y_dbp[train_idx].mean())
    dbp_std = float(y_dbp[train_idx].std())

    y_sbp_train_z = (y_sbp[train_idx] - sbp_mean) / sbp_std
    y_dbp_train_z = (y_dbp[train_idx] - dbp_mean) / dbp_std
    y_sbp_val_z = (y_sbp[val_idx] - sbp_mean) / sbp_std
    y_dbp_val_z = (y_dbp[val_idx] - dbp_mean) / dbp_std

    train_ds = TensorDataset(
        torch.from_numpy(X[train_idx]).float(),
        torch.from_numpy(y_sbp_train_z).float(),
        torch.from_numpy(y_dbp_train_z).float(),
        torch.from_numpy(y_sbp[train_idx]).float(),
        torch.from_numpy(y_dbp[train_idx]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X[val_idx]).float(),
        torch.from_numpy(y_sbp_val_z).float(),
        torch.from_numpy(y_dbp_val_z).float(),
        torch.from_numpy(y_sbp[val_idx]).float(),
        torch.from_numpy(y_dbp[val_idx]).float(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    stats = {
        "sbp_mean": sbp_mean,
        "sbp_std": sbp_std,
        "dbp_mean": dbp_mean,
        "dbp_std": dbp_std,
        "train_len": len(train_idx),
        "val_len": len(val_idx),
        "train_patients": len(np.unique(groups[train_idx])),
        "val_patients": len(np.unique(groups[val_idx])),
    }
    return train_loader, val_loader, stats


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: PhysicsBHSLoss,
    device: torch.device,
    k: float,
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_s_true, all_s_pred = [], []
    all_d_true, all_d_pred = [], []

    with torch.no_grad():
        for bx, by_sz, by_dz, by_s, by_d in loader:
            bx = bx.to(device)
            by_sz = by_sz.to(device)
            by_dz = by_dz.to(device)

            pred_sz, pred_dz = model(bx)
            loss = criterion(pred_sz, pred_dz, by_sz, by_dz, k=k)
            total_loss += loss.item() * len(bx)

            pred_s = pred_sz.cpu().numpy() * criterion.sbp_std + criterion.sbp_mean
            pred_d = pred_dz.cpu().numpy() * criterion.dbp_std + criterion.dbp_mean

            all_s_true.append(by_s.numpy())
            all_s_pred.append(pred_s)
            all_d_true.append(by_d.numpy())
            all_d_pred.append(pred_d)

    mean_loss = total_loss / len(loader.dataset)
    s_true = np.concatenate(all_s_true)
    s_pred = np.concatenate(all_s_pred)
    d_true = np.concatenate(all_d_true)
    d_pred = np.concatenate(all_d_pred)

    sbp_mae = float(np.mean(np.abs(s_pred - s_true)))
    dbp_mae = float(np.mean(np.abs(d_pred - d_true)))
    sbp_rmse = float(np.sqrt(np.mean((s_pred - s_true) ** 2)))
    dbp_rmse = float(np.sqrt(np.mean((d_pred - d_true) ** 2)))

    return mean_loss, sbp_mae, dbp_mae, sbp_rmse, dbp_rmse, s_true, s_pred, d_true, d_pred


def main():
    parser = argparse.ArgumentParser(description="TriaGO_BPNet v5.3 GPU Training")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    dataset_path = find_dataset_file()
    print(f"Dataset: {dataset_path}")

    train_loader, val_loader, stats = prepare_dataloaders(
        dataset_path, test_size=0.2, batch_size=args.batch_size, seed=42
    )

    model = TriaGoBPNet(in_channels=7).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: TriaGO_BPNet_v53_Precision ({total_params:,} parameters)")

    criterion = PhysicsBHSLoss(
        sbp_mean=stats["sbp_mean"],
        sbp_std=stats["sbp_std"],
        dbp_mean=stats["dbp_mean"],
        dbp_std=stats["dbp_std"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    os.makedirs("output", exist_ok=True)
    history_file = open("output/training_history.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(history_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_sbp_mae", "val_dbp_mae", "val_sbp_rmse", "val_dbp_rmse"])

    best_val_loss = float("inf")
    best_metrics = {}

    print(f"Training started: {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        k = min(3.0, 0.5 + (2.5 * (epoch - 1) / 35.0))

        model.train()
        train_loss = 0.0
        for bx, by_sz, by_dz, _, _ in train_loader:
            bx = bx.to(device)
            by_sz = by_sz.to(device)
            by_dz = by_dz.to(device)

            optimizer.zero_grad()
            pred_sz, pred_dz = model(bx)
            loss = criterion(pred_sz, pred_dz, by_sz, by_dz, k=k)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(bx)

        train_loss /= len(train_loader.dataset)
        val_loss, s_mae, d_mae, s_rmse, d_rmse, st, sp, dt, dp = evaluate(
            model, val_loader, criterion, device, k=k
        )
        scheduler.step(val_loss)

        writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{s_mae:.2f}", f"{d_mae:.2f}", f"{s_rmse:.2f}", f"{d_rmse:.2f}"])
        history_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "output/model_best.pt")
            best_metrics = {
                "best_epoch": epoch,
                "val_loss": round(val_loss, 4),
                "val_sbp_mae": round(s_mae, 2),
                "val_dbp_mae": round(d_mae, 2),
                "val_sbp_rmse": round(s_rmse, 2),
                "val_dbp_rmse": round(d_rmse, 2),
                "sbp_sd": round(float(np.std(sp - st)), 2),
                "dbp_sd": round(float(np.std(dp - dt)), 2),
            }

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | SBP MAE: {s_mae:.2f} | DBP MAE: {d_mae:.2f}")

    history_file.close()

    # BHS Standard percentages
    s_err = np.abs(sp - st)
    d_err = np.abs(dp - dt)
    bhs_sbp = {
        "le_5": round(float(np.mean(s_err <= 5.0) * 100), 1),
        "le_10": round(float(np.mean(s_err <= 10.0) * 100), 1),
        "le_15": round(float(np.mean(s_err <= 15.0) * 100), 1),
    }
    bhs_dbp = {
        "le_5": round(float(np.mean(d_err <= 5.0) * 100), 1),
        "le_10": round(float(np.mean(d_err <= 10.0) * 100), 1),
        "le_15": round(float(np.mean(d_err <= 15.0) * 100), 1),
    }

    aami_sbp = bool(best_metrics["val_sbp_mae"] <= 5.0 and best_metrics["sbp_sd"] <= 8.0)
    aami_dbp = bool(best_metrics["val_dbp_mae"] <= 5.0 and best_metrics["dbp_sd"] <= 8.0)

    summary = {
        "model_name": "TriaGO_BPNet_v53_Precision",
        "total_parameters": total_params,
        "epochs_trained": args.epochs,
        "train_segments": stats["train_len"],
        "val_segments": stats["val_len"],
        "train_patients": stats["train_patients"],
        "val_patients": stats["val_patients"],
        **best_metrics,
        "bhs_sbp": bhs_sbp,
        "bhs_dbp": bhs_dbp,
        "aami_sbp_pass": aami_sbp,
        "aami_dbp_pass": aami_dbp,
    }

    with open("output/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- FINAL EVALUATION SUMMARY ---")
    print(f"Best Epoch: {best_metrics['best_epoch']} | Val Loss: {best_metrics['val_loss']}")
    print(f"SBP: MAE={best_metrics['val_sbp_mae']} mmHg, RMSE={best_metrics['val_sbp_rmse']} mmHg, SD={best_metrics['sbp_sd']}")
    print(f"DBP: MAE={best_metrics['val_dbp_mae']} mmHg, RMSE={best_metrics['val_dbp_rmse']} mmHg, SD={best_metrics['dbp_sd']}")
    print(f"BHS SBP: <=5: {bhs_sbp['le_5']}%, <=10: {bhs_sbp['le_10']}%, <=15: {bhs_sbp['le_15']}%")
    print(f"BHS DBP: <=5: {bhs_dbp['le_5']}%, <=10: {bhs_dbp['le_10']}%, <=15: {bhs_dbp['le_15']}%")
    print(f"AAMI: SBP {'PASS' if aami_sbp else 'FAIL'} | DBP {'PASS' if aami_dbp else 'FAIL'}")


if __name__ == "__main__":
    main()
