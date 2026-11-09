"""
Kaggle GPU Training on PulseDB v2.0 (VitalDB + MIMIC) Benchmark
Target: BHS Grade A for DBP (<=5 >=60%, <=10 >=85%, <=15 >=95%)
        BHS Grade B for SBP (<=5 >=50%, <=10 >=75%, <=15 >=90%)
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
# Architecture: RegularizedBPNet (Depthwise Separable Multi-Scale CNN-BiGRU)
# ==============================================================================

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding: int):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class InceptionBlockDS1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        b = out_ch // 4
        self.b1 = nn.Conv1d(in_ch, b, kernel_size=1, bias=False)
        self.b2 = DepthwiseSeparableConv1d(in_ch, b, kernel_size=3, padding=1)
        self.b3 = DepthwiseSeparableConv1d(in_ch, b, kernel_size=7, padding=3)
        self.b4 = DepthwiseSeparableConv1d(in_ch, b, kernel_size=15, padding=7)
        self.act = nn.SiLU()
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.bn(self.act(out))


class RegularizedBPNet(nn.Module):
    def __init__(self, in_channels: int = 7):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.20),
        )

        # Inception Stages
        self.inc1 = InceptionBlockDS1D(32, 48)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout1d(0.25)

        self.inc2 = InceptionBlockDS1D(48, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout1d(0.30)

        # BiGRU
        self.gru = nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.gru_drop = nn.Dropout(0.30)

        # MHSA
        self.mhsa = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True, dropout=0.30)
        self.ln = nn.LayerNorm(64)

        # Pooling
        self.context_drop = nn.Dropout(0.35)

        # Dual Heads
        self.sbp_dense = nn.Linear(128, 32)
        self.sbp_act = nn.SiLU()
        self.sbp_bn = nn.BatchNorm1d(32)
        self.sbp_drop = nn.Dropout(0.30)
        self.sbp_out = nn.Linear(32, 1)

        self.dbp_dense = nn.Linear(128, 32)
        self.dbp_act = nn.SiLU()
        self.dbp_bn = nn.BatchNorm1d(32)
        self.dbp_drop = nn.Dropout(0.30)
        self.dbp_out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] == 1250 and x.shape[2] == 7:
            x = x.transpose(1, 2)

        x = self.stem(x)
        x = self.drop1(self.pool1(self.inc1(x)))
        x = self.drop2(self.pool2(self.inc2(x)))

        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        gru_out = self.gru_drop(gru_out)

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
# Augmentations & Physics Loss
# ==============================================================================

class MultimodalAugmenter:
    @staticmethod
    def augment(batch_x: torch.Tensor) -> torch.Tensor:
        out = batch_x.clone()
        # 1. PTT-preserving circular time shift
        shift = random.randint(-40, 40)
        out = torch.roll(out, shifts=shift, dims=1)

        # 2. Amplitude jitter (+/- 5%)
        scale = 1.0 + (random.random() - 0.5) * 0.10
        out = out * scale

        # 3. Respiratory baseline wander
        if random.random() < 0.5:
            t = torch.linspace(0, 10, out.shape[1], device=out.device)
            freq = 0.2 + random.random() * 0.15
            drift = 0.05 * torch.sin(2 * np.pi * freq * t).unsqueeze(0).unsqueeze(-1)
            out[:, :, 0] += drift[:, :, 0]
            out[:, :, 1] += drift[:, :, 0]

        return out


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
# Helpers
# ==============================================================================

def find_dataset_file() -> Path:
    candidates = [
        Path("/kaggle/input/pulsedb-vital-mimic-7ch/dataset_7ch.npz"),
        Path("./dataset_7ch.npz"),
        Path("../dataset_7ch.npz"),
        Path(__file__).resolve().parent / "dataset_7ch.npz",
        Path(__file__).resolve().parents[2] / "pulsedb_benchmark_7ch.npz",
    ]
    for c in candidates:
        if c.exists():
            return c

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        found = list(kaggle_input.glob("**/dataset_7ch.npz"))
        if found:
            return found[0]

    raise FileNotFoundError("dataset_7ch.npz could not be located!")


def get_bhs_grade(p5: float, p10: float, p15: float) -> str:
    if p5 >= 60.0 and p10 >= 85.0 and p15 >= 95.0:
        return "Grade A"
    elif p5 >= 50.0 and p10 >= 75.0 and p15 >= 90.0:
        return "Grade B"
    elif p5 >= 40.0 and p10 >= 65.0 and p15 >= 85.0:
        return "Grade C"
    else:
        return "Grade D"


def prepare_dataloaders(
    dataset_path: Path,
    test_size: float = 0.2,
    batch_size: int = 64,
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
    parser = argparse.ArgumentParser(description="PulseDB v2.0 Benchmark GPU Training")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
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

    model = RegularizedBPNet(in_channels=7).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: RegularizedBPNet ({total_params:,} parameters)")
    print(f"Split: Train {stats['train_len']} segs ({stats['train_patients']} pts) | Val {stats['val_len']} segs ({stats['val_patients']} pts)")

    criterion = PhysicsBHSLoss(
        sbp_mean=stats["sbp_mean"],
        sbp_std=stats["sbp_std"],
        dbp_mean=stats["dbp_mean"],
        dbp_std=stats["dbp_std"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    os.makedirs("output", exist_ok=True)
    history_file = open("output/training_history.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(history_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_sbp_mae", "val_dbp_mae", "val_sbp_rmse", "val_dbp_rmse"])

    best_val_loss = float("inf")
    best_metrics = {}

    print(f"Training started: {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        k = min(3.0, 0.5 + (2.5 * (epoch - 1) / 30.0))

        model.train()
        train_loss = 0.0
        for bx, by_sz, by_dz, _, _ in train_loader:
            bx = MultimodalAugmenter.augment(bx).to(device)
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
        scheduler.step()

        writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{s_mae:.2f}", f"{d_mae:.2f}", f"{s_rmse:.2f}", f"{d_rmse:.2f}"])
        history_file.flush()

        s_err = np.abs(sp - st)
        d_err = np.abs(dp - dt)
        sbp_p5 = round(float(np.mean(s_err <= 5.0) * 100), 1)
        sbp_p10 = round(float(np.mean(s_err <= 10.0) * 100), 1)
        sbp_p15 = round(float(np.mean(s_err <= 15.0) * 100), 1)
        dbp_p5 = round(float(np.mean(d_err <= 5.0) * 100), 1)
        dbp_p10 = round(float(np.mean(d_err <= 10.0) * 100), 1)
        dbp_p15 = round(float(np.mean(d_err <= 15.0) * 100), 1)

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
                "bhs_sbp": {"le_5": sbp_p5, "le_10": sbp_p10, "le_15": sbp_p15, "grade": get_bhs_grade(sbp_p5, sbp_p10, sbp_p15)},
                "bhs_dbp": {"le_5": dbp_p5, "le_10": dbp_p10, "le_15": dbp_p15, "grade": get_bhs_grade(dbp_p5, dbp_p10, dbp_p15)},
            }

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | SBP MAE: {s_mae:.2f} (BHS: {sbp_p5}%/{sbp_p10}%) | DBP MAE: {d_mae:.2f} (BHS: {dbp_p5}%/{dbp_p10}%)")

    history_file.close()

    aami_sbp = bool(best_metrics["val_sbp_mae"] <= 5.0 and best_metrics["sbp_sd"] <= 8.0)
    aami_dbp = bool(best_metrics["val_dbp_mae"] <= 5.0 and best_metrics["dbp_sd"] <= 8.0)

    summary = {
        "model_name": "RegularizedBPNet_PulseDB_v2",
        "total_parameters": total_params,
        "epochs_trained": args.epochs,
        "train_segments": stats["train_len"],
        "val_segments": stats["val_len"],
        "train_patients": stats["train_patients"],
        "val_patients": stats["val_patients"],
        **best_metrics,
        "aami_sbp_pass": aami_sbp,
        "aami_dbp_pass": aami_dbp,
    }

    with open("output/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ FINAL BHS & AAMI RESULTS ================")
    print(f"Best Epoch: {best_metrics['best_epoch']} | Val Loss: {best_metrics['val_loss']}")
    print(f"SBP Error: MAE={best_metrics['val_sbp_mae']} mmHg, RMSE={best_metrics['val_sbp_rmse']} mmHg, SD={best_metrics['sbp_sd']}")
    print(f"DBP Error: MAE={best_metrics['val_dbp_mae']} mmHg, RMSE={best_metrics['val_dbp_rmse']} mmHg, SD={best_metrics['dbp_sd']}")
    print(f"BHS SBP: <=5: {best_metrics['bhs_sbp']['le_5']}%, <=10: {best_metrics['bhs_sbp']['le_10']}%, <=15: {best_metrics['bhs_sbp']['le_15']}% -> {best_metrics['bhs_sbp']['grade']}")
    print(f"BHS DBP: <=5: {best_metrics['bhs_dbp']['le_5']}%, <=10: {best_metrics['bhs_dbp']['le_10']}%, <=15: {best_metrics['bhs_dbp']['le_15']}% -> {best_metrics['bhs_dbp']['grade']}")
    print(f"AAMI Compliance: SBP {'PASS' if aami_sbp else 'FAIL'} | DBP {'PASS' if aami_dbp else 'FAIL'}")


if __name__ == "__main__":
    main()
