"""
Stage 3: CNN-BiGRU Deep Learning Model (TriaGO_BPNet v5.3 Precision)
Multimodal Blood Pressure Estimation using 7-Channel Input.

Ponytail principle: minimal, clear PyTorch implementation, stdlib first.
Visualization invariant: plt.show() only, NEVER plt.savefig().
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
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
# Model Architecture: TriaGO_BPNet v5.3 Precision
# ==============================================================================

class InceptionBlock1D(nn.Module):
    """Multi-scale 1D Inception block with 4 parallel kernel branches."""
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
    """
    TriaGO_BPNet v5.3 Precision:
    Input (B, 1250, 7) -> Stem -> 3x Inception -> BiGRU -> MHSA -> Dual Pool -> Dual Head
    Total parameter count: exactly 374,850.
    """
    def __init__(self, in_channels: int = 7):
        super().__init__()
        # 1. Stem Layer Unified 7-Channel
        self.stem_conv = nn.Conv1d(in_channels, 64, kernel_size=11, padding=5)
        self.stem_act = nn.SiLU()
        self.stem_bn = nn.BatchNorm1d(64)

        # 2. Multi-Scale Inception Blocks
        self.inc1 = InceptionBlock1D(64, filters=64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout1d(0.15)

        self.inc2 = InceptionBlock1D(64, filters=128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout1d(0.20)

        self.inc3 = InceptionBlock1D(128, filters=128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 3. BiGRU Dynamic Recurrent Features
        self.bigru = nn.GRU(128, 64, batch_first=True, bidirectional=True)

        # 4. Multi-Head Self-Attention (MHSA)
        self.mhsa = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(128)

        # 5. Global Context Aggregation (AvgPool + MaxPool = 256)
        self.context_drop = nn.Dropout(0.25)

        # 6. Task-Specific Regressor Heads
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
        # Shape check: (B, 1250, 7) -> (B, 7, 1250)
        if x.shape[1] == 1250 and x.shape[2] == 7:
            x = x.transpose(1, 2)

        # 1. Stem
        x = self.stem_bn(self.stem_act(self.stem_conv(x)))

        # 2. Inception Blocks
        x = self.drop1(self.pool1(self.inc1(x)))
        x = self.drop2(self.pool2(self.inc2(x)))
        x = self.pool3(self.inc3(x))

        # 3. BiGRU: (B, 128, 156) -> (B, 156, 128)
        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)

        # 4. MHSA + Residual + LayerNorm
        attn_out, _ = self.mhsa(gru_out, gru_out, gru_out)
        x = self.ln(gru_out + attn_out)

        # 5. Global Context Pooling
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        context = torch.cat([avg_pool, max_pool], dim=1)
        context = self.context_drop(context)

        # 6. Regressor Heads
        s = self.sbp_drop(self.sbp_bn(self.sbp_act(self.sbp_dense(context))))
        sbp_pred = self.sbp_out(s).squeeze(-1)

        d = self.dbp_drop(self.dbp_bn(self.dbp_act(self.dbp_dense(context))))
        dbp_pred = self.dbp_out(d).squeeze(-1)

        return sbp_pred, dbp_pred


# ==============================================================================
# Physics-Informed & BHS Loss
# ==============================================================================

class PhysicsBHSLoss(nn.Module):
    """Loss module matching TriaGo v5.3 surgical BHS + MAP/PP physics constraints."""
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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Huber loss on normalized z-scores
        l_sbp = self.huber_sbp(pred_s_z, true_s_z)
        l_dbp = self.huber_dbp(pred_d_z, true_d_z)

        # 2. Reconstructed targets in mmHg
        s_true = true_s_z * self.sbp_std + self.sbp_mean
        d_true = true_d_z * self.dbp_std + self.dbp_mean
        s_pred = pred_s_z * self.sbp_std + self.sbp_mean
        d_pred = pred_d_z * self.dbp_std + self.dbp_mean

        # 3. Physics MAP & Pulse Pressure
        map_true = d_true + (s_true - d_true) / 3.0
        map_pred = d_pred + (s_pred - d_pred) / 3.0
        l_map = torch.abs(map_pred - map_true).mean() / self.sbp_std

        pp_true = s_true - d_true
        pp_pred = s_pred - d_pred
        l_pp = torch.abs(pp_pred - pp_true).mean() / self.sbp_std

        # 4. Surgical BHS exceedance penalties
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

        metrics = {
            "loss": total_loss.item(),
            "l_sbp": l_sbp.item(),
            "l_dbp": l_dbp.item(),
            "l_map": l_map.item(),
            "l_pp": l_pp.item(),
        }
        return total_loss, metrics


# ==============================================================================
# Pipeline & Training Routines
# ==============================================================================

def prepare_dataloaders(
    dataset_path: Path,
    test_size: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, float]]:
    data = np.load(dataset_path)
    X = data["X"]  # (N, 1250, 7)
    y_sbp = data["y_sbp"]
    y_dbp = data["y_dbp"]
    groups = data["groups"]

    # Leak-free patient-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(X, groups=groups))

    # Fit target scaler ONLY on train fold
    sbp_mean = float(y_sbp[train_idx].mean())
    sbp_std = float(y_sbp[train_idx].std())
    dbp_mean = float(y_dbp[train_idx].mean())
    dbp_std = float(y_dbp[train_idx].std())

    # Standardize targets
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
            loss, _ = criterion(pred_sz, pred_dz, by_sz, by_dz, k=k)
            total_loss += loss.item() * len(bx)

            # Reconstruct to mmHg
            pred_s = pred_sz.cpu().numpy() * criterion.sbp_std + criterion.sbp_mean
            pred_d = pred_dz.cpu().numpy() * criterion.dbp_std + criterion.dbp_mean

            all_s_true.append(by_s.numpy())
            all_s_pred.append(pred_s)
            all_d_true.append(by_d.numpy())
            all_d_pred.append(pred_d)

    n_samples = len(loader.dataset)
    mean_loss = total_loss / n_samples
    s_true = np.concatenate(all_s_true)
    s_pred = np.concatenate(all_s_pred)
    d_true = np.concatenate(all_d_true)
    d_pred = np.concatenate(all_d_pred)

    sbp_mae = float(np.mean(np.abs(s_pred - s_true)))
    dbp_mae = float(np.mean(np.abs(d_pred - d_true)))
    sbp_rmse = float(np.sqrt(np.mean((s_pred - s_true) ** 2)))
    dbp_rmse = float(np.sqrt(np.mean((d_pred - d_true) ** 2)))

    return mean_loss, sbp_mae, dbp_mae, sbp_rmse, dbp_rmse, s_true, s_pred, d_true, d_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: PhysicsBHSLoss,
    epochs: int = 40,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, list], dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {"train_loss": [], "val_loss": [], "val_sbp_mae": [], "val_dbp_mae": []}
    best_val_loss = float("inf")
    best_metrics = {}

    for epoch in range(epochs):
        # Sharpness ramp: k from 0.5 to 3.0 over 35 epochs
        k = min(3.0, 0.5 + (2.5 * epoch / 35.0))

        model.train()
        train_loss = 0.0
        for bx, by_sz, by_dz, _, _ in train_loader:
            bx = bx.to(device)
            by_sz = by_sz.to(device)
            by_dz = by_dz.to(device)

            optimizer.zero_grad()
            pred_sz, pred_dz = model(bx)
            loss, _ = criterion(pred_sz, pred_dz, by_sz, by_dz, k=k)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(bx)

        train_loss /= len(train_loader.dataset)
        val_loss, s_mae, d_mae, s_rmse, d_rmse, st, sp, dt, dp = evaluate(
            model, val_loader, criterion, device, k=k
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_sbp_mae"].append(s_mae)
        history["val_dbp_mae"].append(d_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "sbp_mae": s_mae,
                "dbp_mae": d_mae,
                "sbp_rmse": s_rmse,
                "dbp_rmse": d_rmse,
                "s_true": st,
                "s_pred": sp,
                "d_true": dt,
                "d_pred": dp,
            }

    return history, best_metrics


def visualize_results(history: dict, best: dict) -> None:
    """Render interactive evaluation curves. Invariant: NEVER call plt.savefig."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss", color="tab:blue", lw=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="tab:orange", lw=2)
    axes[0].set_title("Physics BHS Loss Progression", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. SBP Scatter
    st, sp = best["s_true"], best["s_pred"]
    axes[1].scatter(st, sp, color="tab:red", alpha=0.6, edgecolors="none")
    lims_s = [min(st.min(), sp.min()) - 5, max(st.max(), sp.max()) + 5]
    axes[1].plot(lims_s, lims_s, "k--", alpha=0.7, label="Ideal")
    axes[1].set_title(f"SBP: MAE={best['sbp_mae']:.2f} mmHg, RMSE={best['sbp_rmse']:.2f}", fontweight="bold")
    axes[1].set_xlabel("True SBP (mmHg)")
    axes[1].set_ylabel("Estimated SBP (mmHg)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. DBP Scatter
    dt, dp = best["d_true"], best["d_pred"]
    axes[2].scatter(dt, dp, color="tab:blue", alpha=0.6, edgecolors="none")
    lims_d = [min(dt.min(), dp.min()) - 5, max(dt.max(), dp.max()) + 5]
    axes[2].plot(lims_d, lims_d, "k--", alpha=0.7, label="Ideal")
    axes[2].set_title(f"DBP: MAE={best['dbp_mae']:.2f} mmHg, RMSE={best['dbp_rmse']:.2f}", fontweight="bold")
    axes[2].set_xlabel("True DBP (mmHg)")
    axes[2].set_ylabel("Estimated DBP (mmHg)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# CLI & Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3: TriaGo CNN-BiGRU Deep Learning Model")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--no-show", action="store_true", help="Headless mode (suppress plt.show)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "dataset_7ch.npz"

    # 1. DataLoaders
    train_loader, val_loader, stats = prepare_dataloaders(
        dataset_path, test_size=0.2, batch_size=args.batch_size, seed=42
    )

    # 2. Model & Loss instantiation
    model = TriaGoBPNet(in_channels=7).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = PhysicsBHSLoss(
        sbp_mean=stats["sbp_mean"],
        sbp_std=stats["sbp_std"],
        dbp_mean=stats["dbp_mean"],
        dbp_std=stats["dbp_std"],
    )

    # 3. Training & Validation
    history, best = train_model(
        model, train_loader, val_loader, criterion, epochs=args.epochs, lr=args.lr, device=device
    )

    # 4. AAMI standard check (MAE <= 5.0 mmHg, SD <= 8.0 mmHg)
    s_err = best["s_pred"] - best["s_true"]
    d_err = best["d_pred"] - best["d_true"]
    s_sd = float(np.std(s_err))
    d_sd = float(np.std(d_err))
    aami_sbp = bool(best["sbp_mae"] <= 5.0 and s_sd <= 8.0)
    aami_dbp = bool(best["dbp_mae"] <= 5.0 and d_sd <= 8.0)

    # 5. Summary Contract (< 1 KB)
    summary_path = base_dir / "stage3_summary.json"
    summary_data = {
        "model_name": "TriaGO_BPNet_v53_Precision",
        "total_parameters": total_params,
        "train_segments": stats["train_len"],
        "val_segments": stats["val_len"],
        "train_patients": stats["train_patients"],
        "val_patients": stats["val_patients"],
        "best_epoch": best["epoch"],
        "val_sbp_mae": round(best["sbp_mae"], 2),
        "val_dbp_mae": round(best["dbp_mae"], 2),
        "val_sbp_rmse": round(best["sbp_rmse"], 2),
        "val_dbp_rmse": round(best["dbp_rmse"], 2),
        "aami_sbp_pass": aami_sbp,
        "aami_dbp_pass": aami_dbp,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 6. Concise Stdout (<= 15 lines)
    print(f"Model: {summary_data['model_name']} ({total_params:,} params)")
    print(f"Split: Train {stats['train_len']} segs ({stats['train_patients']} pts) | Val {stats['val_len']} segs ({stats['val_patients']} pts)")
    print(f"Results Epoch {best['epoch']}: SBP MAE={best['sbp_mae']:.2f} mmHg (SD={s_sd:.2f}) | DBP MAE={best['dbp_mae']:.2f} mmHg (SD={d_sd:.2f})")
    print(f"AAMI Compliance: SBP={'PASS' if aami_sbp else 'FAIL'} | DBP={'PASS' if aami_dbp else 'FAIL'}")
    print(f"Contract: {summary_path.name} ({summary_path.stat().st_size} bytes)")

    # 7. Interactive Visualization
    if not args.no_show:
        visualize_results(history, best)


if __name__ == "__main__":
    main()
