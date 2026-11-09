"""
Stage 3 Regularized: Compact Multi-Scale CNN-BiGRU for Multimodal BP Estimation.
Anti-overfitting architecture (<80k params) with depthwise separable convs,
PTT-preserving data augmentation, and early stopping.

Ponytail principle: minimal, clear PyTorch implementation, stdlib first.
Visualization invariant: plt.show() only, NEVER plt.savefig().
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
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
# Model Architecture: Compact RegularizedBPNet (~52k parameters)
# ==============================================================================

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1D convolution for spatial/temporal decoupling."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding: int):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class InceptionBlockDS1D(nn.Module):
    """Lightweight 1D Inception block with depthwise-separable multi-scale branches."""
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
    """
    Compact regularized multimodal network:
    Input (B, 1250, 7) -> Stem -> 2x DS-Inception -> BiGRU -> MHSA -> Dual Pool -> Dual Head
    Parameter count: ~52,370 params (86.0% reduction from 374,850 baseline).
    """
    def __init__(self, in_channels: int = 7):
        super().__init__()
        # 1. Stem Layer (7 channels -> 32 channels)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.20),
        )

        # 2. Depthwise Separable Multi-Scale Stages
        self.inc1 = InceptionBlockDS1D(32, 48)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout1d(0.25)

        self.inc2 = InceptionBlockDS1D(48, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout1d(0.30)

        # 3. Compact BiGRU Temporal Encoder
        self.gru = nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.gru_drop = nn.Dropout(0.30)

        # 4. Multi-Head Self-Attention with Residual & LayerNorm
        self.mhsa = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True, dropout=0.30)
        self.ln = nn.LayerNorm(64)

        # 5. Dual Context Aggregation (AvgPool + MaxPool = 128)
        self.context_drop = nn.Dropout(0.35)

        # 6. Compact Regressor Heads
        self.sbp_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.30),
            nn.Linear(32, 1),
        )
        self.dbp_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.30),
            nn.Linear(32, 1),
        )

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

        avg_p = x.mean(dim=1)
        max_p = x.max(dim=1).values
        ctx = self.context_drop(torch.cat([avg_p, max_p], dim=1))

        sbp_pred = self.sbp_head(ctx).squeeze(-1)
        dbp_pred = self.dbp_head(ctx).squeeze(-1)
        return sbp_pred, dbp_pred


# ==============================================================================
# Multimodal Signal Augmentation (Training Only)
# ==============================================================================

class MultimodalAugmenter:
    """Anti-overfitting data augmentation for 7-channel ECG+PPG segments.
    Preserves inter-channel PTT temporal alignment while introducing:
    1. Synchronous circular time-shift (+/- 40 samples / ~320ms)
    2. Random amplitude scaling / jitter (+/- 5%)
    3. Respiratory baseline wander jitter on ECG and PPG (0.15-0.35 Hz)
    """
    def __init__(
        self,
        shift_max: int = 40,
        jitter_range: float = 0.05,
        wander_amp: float = 0.04,
        p_aug: float = 0.5,
    ):
        self.shift_max = shift_max
        self.jitter_range = jitter_range
        self.wander_amp = wander_amp
        self.p_aug = p_aug

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p_aug <= 0:
            return x

        B, T, C = x.shape
        device = x.device
        out = x.clone()

        # 1. Synchronous Time Shift across all channels (preserves PTT delay)
        if self.shift_max > 0 and random.random() < self.p_aug:
            shift = random.randint(-self.shift_max, self.shift_max)
            out = torch.roll(out, shifts=shift, dims=1)

        # 2. Amplitude Jitter (+/- 5%)
        if self.jitter_range > 0 and random.random() < self.p_aug:
            jitter = 1.0 + (torch.rand(B, 1, C, device=device) * (2 * self.jitter_range) - self.jitter_range)
            out = out * jitter

        # 3. Baseline Wander (respiratory sinusoidal drift on ECG [0] & PPG [1])
        if self.wander_amp > 0 and random.random() < self.p_aug:
            t = torch.linspace(0, 10.0, steps=T, device=device).view(1, T, 1)
            freq = 0.15 + torch.rand(B, 1, 1, device=device) * 0.20
            phase = torch.rand(B, 1, 1, device=device) * (2 * np.pi)
            amp = (torch.rand(B, 1, 1, device=device) * 2 - 1) * self.wander_amp
            wander = amp * torch.sin(2 * np.pi * freq * t + phase)
            out[:, :, 0:2] = out[:, :, 0:2] + wander

        return out


# ==============================================================================
# Physics-Informed & BHS Loss
# ==============================================================================

class PhysicsBHSLoss(nn.Module):
    """Loss module matching surgical BHS + MAP/PP physics constraints."""
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
            loss, _ = criterion(pred_sz, pred_dz, by_sz, by_dz, k=k)
            total_loss += loss.item() * len(bx)

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
    scheduler_type: str = "cosine",
    patience: int = 7,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, list], dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    augmenter = MultimodalAugmenter(shift_max=40, jitter_range=0.05, wander_amp=0.04, p_aug=0.5)

    history = {"train_loss": [], "val_loss": [], "val_sbp_mae": [], "val_dbp_mae": []}
    best_val_loss = float("inf")
    best_metrics = {}
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        k = min(3.0, 0.5 + (2.5 * epoch / max(1, epochs - 5)))

        model.train()
        train_loss = 0.0
        for bx, by_sz, by_dz, _, _ in train_loader:
            bx = bx.to(device)
            by_sz = by_sz.to(device)
            by_dz = by_dz.to(device)

            bx_aug = augmenter(bx)

            optimizer.zero_grad()
            pred_sz, pred_dz = model(bx_aug)
            loss, _ = criterion(pred_sz, pred_dz, by_sz, by_dz, k=k)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(bx)

        train_loss /= len(train_loader.dataset)
        val_loss, s_mae, d_mae, s_rmse, d_rmse, st, sp, dt, dp = evaluate(
            model, val_loader, criterion, device, k=k
        )

        if scheduler_type == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_sbp_mae"].append(s_mae)
        history["val_dbp_mae"].append(d_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
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
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_metrics


def visualize_results(history: dict, best: dict) -> None:
    """Render interactive evaluation curves. Invariant: NEVER call plt.savefig."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="tab:blue", lw=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="tab:orange", lw=2)
    axes[0].set_title("Physics BHS Loss Progression", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    st, sp = best["s_true"], best["s_pred"]
    axes[1].scatter(st, sp, color="tab:red", alpha=0.6, edgecolors="none")
    lims_s = [min(st.min(), sp.min()) - 5, max(st.max(), sp.max()) + 5]
    axes[1].plot(lims_s, lims_s, "k--", alpha=0.7, label="Ideal")
    axes[1].set_title(f"SBP: MAE={best['sbp_mae']:.2f} mmHg, RMSE={best['sbp_rmse']:.2f}", fontweight="bold")
    axes[1].set_xlabel("True SBP (mmHg)")
    axes[1].set_ylabel("Estimated SBP (mmHg)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

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
    parser = argparse.ArgumentParser(description="Stage 3 Regularized: Multimodal Blood Pressure Estimation")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"], help="Scheduler type")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (default: 7)")
    parser.add_argument("--no-show", action="store_true", help="Headless mode (suppress plt.show)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "dataset_7ch.npz"

    train_loader, val_loader, stats = prepare_dataloaders(
        dataset_path, test_size=0.2, batch_size=args.batch_size, seed=42
    )

    model = RegularizedBPNet(in_channels=7).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = PhysicsBHSLoss(
        sbp_mean=stats["sbp_mean"],
        sbp_std=stats["sbp_std"],
        dbp_mean=stats["dbp_mean"],
        dbp_std=stats["dbp_std"],
    )

    history, best = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        epochs=args.epochs,
        lr=args.lr,
        scheduler_type=args.scheduler,
        patience=args.patience,
        device=device,
    )

    s_err = best["s_pred"] - best["s_true"]
    d_err = best["d_pred"] - best["d_true"]
    s_sd = float(np.std(s_err))
    d_sd = float(np.std(d_err))
    aami_sbp = bool(best["sbp_mae"] <= 5.0 and s_sd <= 8.0)
    aami_dbp = bool(best["dbp_mae"] <= 5.0 and d_sd <= 8.0)

    summary_path = base_dir / "stage3_regularized_summary.json"
    summary_data = {
        "model_name": "RegularizedBPNet_v1",
        "total_parameters": total_params,
        "baseline_parameters": 374850,
        "parameter_reduction_pct": round((374850 - total_params) / 374850 * 100, 2),
        "train_segments": stats["train_len"],
        "val_segments": stats["val_len"],
        "train_patients": stats["train_patients"],
        "val_patients": stats["val_patients"],
        "best_epoch": best["epoch"],
        "val_loss": round(best["val_loss"], 4),
        "val_sbp_mae": round(best["sbp_mae"], 2),
        "val_dbp_mae": round(best["dbp_mae"], 2),
        "val_sbp_rmse": round(best["sbp_rmse"], 2),
        "val_dbp_rmse": round(best["dbp_rmse"], 2),
        "aami_sbp_pass": aami_sbp,
        "aami_dbp_pass": aami_dbp,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Concise console summary (<= 15 lines)
    print(f"Model: {summary_data['model_name']} ({total_params:,} params, -{summary_data['parameter_reduction_pct']}%)")
    print(f"Split: Train {stats['train_len']} segs ({stats['train_patients']} pts) | Val {stats['val_len']} segs ({stats['val_patients']} pts)")
    print(f"Best Epoch: {best['epoch']} | Val Loss: {best['val_loss']:.4f}")
    print(f"Metrics: SBP MAE={best['sbp_mae']:.2f} mmHg (SD={s_sd:.2f}) | DBP MAE={best['dbp_mae']:.2f} mmHg (SD={d_sd:.2f})")
    print(f"AAMI Compliance: SBP={'PASS' if aami_sbp else 'FAIL'} | DBP={'PASS' if aami_dbp else 'FAIL'}")
    print(f"Contract: {summary_path.name} ({summary_path.stat().st_size} bytes)")

    if not args.no_show:
        visualize_results(history, best)


if __name__ == "__main__":
    main()
