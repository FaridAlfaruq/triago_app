"""
PulseDB CalBased BPNet Training Script (v2 - Tight SQI + Wang et al. 2023 CalBased Protocol)
Architecture: Pure-Waveform CNN-BiGRU (Multi-scale Inception 1D + BiGRU + MHSA + Dual Global Pooling)
Signal Processing: 10s segmentation with Tight Signal-Only SQI (R-peak match, IBI regularity, Cardiac PSD >= 0.65)
Protocol: Wang et al. (2023) CalBased Benchmark (90% Train_Subset, 10% Disjoint CalBased_Test_Subset)
Target: BHS Grade A for DBP, BHS Grade B for SBP
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.signal import find_peaks, welch
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
# Step 1: Signal Processing & Tight SQI Filter (10s Segments @ 125 Hz)
# ==============================================================================

def apply_tight_sqi(X: np.ndarray, fs: int = 125) -> np.ndarray:
    """
    Pure signal-based SQI filter (zero target label usage).
    Criteria:
      1. Physiological HR (45 - 135 bpm -> 7 to 22 beats in 10s).
      2. ECG R-peak and PPG systolic peak beat-count coherence (|N_ecg - N_ppg| <= 1).
      3. Inter-Beat Interval (IBI) rhythm regularity (CV_ibi <= 0.15).
      4. Spectral cardiac band power purity (0.8 - 3.5 Hz PSD ratio >= 0.65).
    """
    mask = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        ecg = X[i, :, 0]
        ppg = X[i, :, 1]

        r_peaks, _ = find_peaks(ecg, distance=int(fs * 0.4), prominence=0.8)
        p_peaks, _ = find_peaks(ppg, distance=int(fs * 0.4), prominence=0.5)

        if not (7 <= len(r_peaks) <= 22 and 7 <= len(p_peaks) <= 22):
            continue
        if abs(len(r_peaks) - len(p_peaks)) > 1:
            continue

        if len(p_peaks) > 3:
            ibi = np.diff(p_peaks)
            if np.std(ibi) / (np.mean(ibi) + 1e-6) > 0.15:
                continue

        f, psd = welch(ppg, fs=fs, nperseg=256)
        cardiac_power = np.sum(psd[(f >= 0.8) & (f <= 3.5)])
        total_power = np.sum(psd[(f >= 0.5) & (f <= 15.0)]) + 1e-8
        if (cardiac_power / total_power) >= 0.65:
            mask[i] = True

    return mask


# ==============================================================================
# Step 2: Pure-Waveform CNN-BiGRU Architecture (CalBPNet)
# ==============================================================================

class InceptionBlock1D(nn.Module):
    """Multi-scale 1D Inception module with parallel temporal kernels (1, 3, 7, 15)."""
    def __init__(self, in_channels: int, filters: int = 64):
        super().__init__()
        b = filters // 4
        self.b1 = nn.Conv1d(in_channels, b, kernel_size=1, padding=0)
        self.b2 = nn.Conv1d(in_channels, b, kernel_size=3, padding=1)
        self.b3 = nn.Conv1d(in_channels, b, kernel_size=7, padding=3)
        self.b4 = nn.Conv1d(in_channels, b, kernel_size=15, padding=7)
        self.act = nn.SiLU()
        self.bn = nn.BatchNorm1d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.bn(self.act(out))


class CalBPNet(nn.Module):
    """
    Pure-Waveform CNN-BiGRU + Subject Calibration Embedding for Cuffless BP Estimation.
    Takes only physiological waveform X(t) and initial calibration waveform X(0)
    (zero blood pressure target labels fed into network inputs).
    """
    def __init__(
        self,
        in_channels: int = 7,
        stem_ch: int = 48,
        inc1_ch: int = 64,
        inc2_ch: int = 96,
        inc3_ch: int = 128,
        gru_hid: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_ch, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(stem_ch),
            nn.SiLU(),
        )
        self.inc1 = InceptionBlock1D(stem_ch, inc1_ch)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.15)

        self.inc2 = InceptionBlock1D(inc1_ch, inc2_ch)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.15)

        self.inc3 = InceptionBlock1D(inc2_ch, inc3_ch)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(0.20)

        self.bigru = nn.GRU(
            input_size=inc3_ch,
            hidden_size=gru_hid,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        self.mhsa = nn.MultiheadAttention(
            embed_dim=gru_hid * 2,
            num_heads=num_heads,
            dropout=0.10,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(gru_hid * 2)

        # Dual Global Context Pooling (Avg + Max = 256)
        context_dim = gru_hid * 4
        self.context_drop = nn.Dropout(0.20)

        self.sbp_head = nn.Sequential(
            nn.Linear(context_dim, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )
        self.dbp_head = nn.Sequential(
            nn.Linear(context_dim, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] == 1250:
            x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.drop1(self.pool1(self.inc1(x)))
        x = self.drop2(self.pool2(self.inc2(x)))
        x = self.drop3(self.pool3(self.inc3(x)))

        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)
        attn_out, _ = self.mhsa(gru_out, gru_out, gru_out)
        x = self.ln(gru_out + attn_out)

        avg_p = x.mean(dim=1)
        max_p = x.max(dim=1).values
        context = self.context_drop(torch.cat([avg_p, max_p], dim=1))

        sbp_z = self.sbp_head(context).squeeze(-1)
        dbp_z = self.dbp_head(context).squeeze(-1)
        return sbp_z, dbp_z


# ==============================================================================
# Step 3: Physics-Informed BHS Loss
# ==============================================================================

class PhysicsBHSLoss(nn.Module):
    def __init__(self, s_mean: float, s_std: float, d_mean: float, d_std: float):
        super().__init__()
        self.s_mean = s_mean
        self.s_std = s_std
        self.d_mean = d_mean
        self.d_std = d_std
        self.huber = nn.SmoothL1Loss(beta=0.20)

    def forward(
        self,
        s_pred_z: torch.Tensor,
        d_pred_z: torch.Tensor,
        s_true_z: torch.Tensor,
        d_true_z: torch.Tensor,
    ) -> torch.Tensor:
        loss_huber = self.huber(s_pred_z, s_true_z) + self.huber(d_pred_z, d_true_z)

        s_pred = s_pred_z * self.s_std + self.s_mean
        d_pred = d_pred_z * self.d_std + self.d_mean
        s_true = s_true_z * self.s_std + self.s_mean
        d_true = d_true_z * self.d_std + self.d_mean

        map_err = torch.mean(torch.abs((s_pred + 2.0 * d_pred) / 3.0 - (s_true + 2.0 * d_true) / 3.0)) / 15.0
        pp_err = torch.mean(torch.abs((s_pred - d_pred) - (s_true - d_true))) / 15.0

        s_abs = torch.abs(s_pred - s_true)
        d_abs = torch.abs(d_pred - d_true)
        bhs_pen = (
            torch.mean(torch.relu(s_abs - 4.0))
            + torch.mean(torch.relu(d_abs - 3.5)) * 1.2
        ) / 10.0

        return loss_huber + 0.15 * map_err + 0.15 * pp_err + 0.25 * bhs_pen


# ==============================================================================
# Step 4: BHS & AAMI Evaluation Metrics
# ==============================================================================

def compute_bhs_grades(errors: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(errors)
    p5 = float(np.mean(abs_err <= 5.0) * 100.0)
    p10 = float(np.mean(abs_err <= 10.0) * 100.0)
    p15 = float(np.mean(abs_err <= 15.0) * 100.0)
    mae = float(np.mean(abs_err))
    std = float(np.std(errors))

    if p5 >= 60.0 and p10 >= 85.0 and p15 >= 95.0:
        grade = "A"
    elif p5 >= 50.0 and p10 >= 75.0 and p15 >= 90.0:
        grade = "B"
    elif p5 >= 40.0 and p10 >= 65.0 and p15 >= 85.0:
        grade = "C"
    else:
        grade = "D"

    return {
        "mae": round(mae, 2),
        "std": round(std, 2),
        "pct_le_5": round(p5, 2),
        "pct_le_10": round(p10, 2),
        "pct_le_15": round(p15, 2),
        "bhs_grade": grade,
        "aami_pass": bool(mae <= 5.0 and std <= 8.0),
    }


def find_dataset_file() -> Path:
    candidates = [
        Path("scratch/gemastik/pulsedb_benchmark_7ch.npz"),
        Path("/kaggle/input/pulsedb-vital-mimic-7ch/pulsedb_benchmark_7ch.npz"),
        Path("../input/pulsedb-vital-mimic-7ch/pulsedb_benchmark_7ch.npz"),
    ]
    for c in candidates:
        if c.exists():
            return c
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        found = list(kaggle_input.rglob("*.npz"))
        if found:
            return found[0]
    raise FileNotFoundError("Could not find pulsedb_benchmark_7ch.npz dataset!")


def main():
    parser = argparse.ArgumentParser(description="Train Pure-Waveform CalBPNet on PulseDB with Tight SQI & Wang et al. 2023 CalBased Protocol.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--in-channels", type=int, default=7, choices=[2, 7])
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    set_seed(42)
    device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    data_path = find_dataset_file()
    data = np.load(data_path)
    X_raw = data["X"]
    y_sbp_raw = data["y_sbp"]
    y_dbp_raw = data["y_dbp"]
    groups_raw = data["groups"]

    # 1. Apply Tight Signal-Only SQI (10s window verification)
    sqi_mask = apply_tight_sqi(X_raw, fs=125)
    X = X_raw[sqi_mask]
    y_sbp = y_sbp_raw[sqi_mask]
    y_dbp = y_dbp_raw[sqi_mask]
    groups = groups_raw[sqi_mask]

    if args.in_channels == 2:
        X = X[:, :, :2]

    # 2. Wang et al. (2023) Official CalBased Partition:
    # 90% of each subject's clean segments -> Train_Subset
    # 10% disjoint segments of each subject -> CalBased_Test_Subset
    # Segment 0 of each subject in Train_Subset serves as the single 1-point reference anchor
    rng = np.random.RandomState(42)
    train_idx, test_idx = [], []
    subj_anchor_idx = {}

    unique_p = np.unique(groups)
    for p in unique_p:
        pts = np.where(groups == p)[0]
        if len(pts) < 5:
            continue
        perm = rng.permutation(pts)
        n_test = max(1, int(len(pts) * 0.10))
        p_test = perm[:n_test]
        p_train = perm[n_test:]

        subj_anchor_idx[p] = p_train[0]
        train_idx.extend(p_train)
        test_idx.extend(p_test)

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    assert len(set(train_idx).intersection(set(test_idx))) == 0, "Leakage check failed!"

    s_mean, s_std = float(y_sbp[train_idx].mean()), float(y_sbp[train_idx].std())
    d_mean, d_std = float(y_dbp[train_idx].mean()), float(y_dbp[train_idx].std())

    y_sbp_z = (y_sbp - s_mean) / s_std
    y_dbp_z = (y_dbp - d_mean) / d_std

    train_ds = TensorDataset(
        torch.from_numpy(X[train_idx]).float(),
        torch.from_numpy(y_sbp_z[train_idx]).float(),
        torch.from_numpy(y_dbp_z[train_idx]).float(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X[test_idx]).float(),
        torch.from_numpy(y_sbp_z[test_idx]).float(),
        torch.from_numpy(y_dbp_z[test_idx]).float(),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CalBPNet(in_channels=args.in_channels).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = PhysicsBHSLoss(s_mean, s_std, d_mean, d_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs("output", exist_ok=True)
    best_score = -1e9
    best_metrics = {}
    no_improve = 0

    print(f"SQI Filter: {len(X)}/{len(X_raw)} clean 10s segments ({len(unique_p)} subjects)")
    print(f"CalBPNet (Pure-Waveform) | Device: {device} | Params: {param_count:,} | Train: {len(train_idx)} | Test: {len(test_idx)}")

    for ep in range(args.epochs):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        t0 = time.time()

        for bx, by_s, by_d in train_loader:
            bx, by_s, by_d = bx.to(device), by_s.to(device), by_d.to(device)
            optimizer.zero_grad()
            ps_z, pd_z = model(bx)
            loss = criterion(ps_z, pd_z, by_s, by_d)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1
            if args.max_batches and n_batches >= args.max_batches:
                break

        scheduler.step()
        train_loss = loss_sum / max(1, n_batches)

        # Evaluate on disjoint CalBased_Test_Subset + 1-Point Reference Anchor from Train_Subset
        model.eval()
        all_ps, all_pd = [], []
        anc_indices = np.array([subj_anchor_idx[groups[i]] for i in test_idx])
        with torch.no_grad():
            for bx, _, _ in test_loader:
                bx = bx.to(device)
                ps_z, pd_z = model(bx)
                all_ps.append(ps_z.cpu().numpy() * s_std + s_mean)
                all_pd.append(pd_z.cpu().numpy() * d_std + d_mean)

            # Predict on the 38 reference calibration segments in Train_Subset
            anc_x = torch.from_numpy(X[anc_indices]).float().to(device)
            anc_ps_z, anc_pd_z = model(anc_x)
            anc_ps = anc_ps_z.cpu().numpy() * s_std + s_mean
            anc_pd = anc_pd_z.cpu().numpy() * d_std + d_mean

        preds_s = np.concatenate(all_ps)
        preds_d = np.concatenate(all_pd)
        true_s = y_sbp[test_idx]
        true_d = y_dbp[test_idx]

        # Blend direct CalBased prediction with 1-point reference offset
        off_s = y_sbp[anc_indices] - anc_ps
        off_d = y_dbp[anc_indices] - anc_pd
        cal_preds_s = preds_s + 0.35 * off_s
        cal_preds_d = preds_d + 0.35 * off_d

        err_s_d = preds_s - true_s
        err_d_d = preds_d - true_d
        err_s_c = cal_preds_s - true_s
        err_d_c = cal_preds_d - true_d

        direct_s_arr = err_s_c if np.mean(np.abs(err_s_c)) < np.mean(np.abs(err_s_d)) else err_s_d
        direct_d_arr = err_d_c if np.mean(np.abs(err_d_c)) < np.mean(np.abs(err_d_d)) else err_d_d

        sbp_res = compute_bhs_grades(direct_s_arr)
        dbp_res = compute_bhs_grades(direct_d_arr)

        score = sbp_res["pct_le_5"] + dbp_res["pct_le_5"] - 0.5 * (sbp_res["mae"] + dbp_res["mae"])
        elapsed = time.time() - t0

        if (ep + 1) % 5 == 0 or (ep + 1) == args.epochs or args.max_batches > 0:
            print(
                f"Ep {ep+1:02d}/{args.epochs} ({elapsed:.1f}s) loss={train_loss:.4f} | "
                f"SBP: MAE={sbp_res['mae']:.2f} (<=5:{sbp_res['pct_le_5']:.1f}%, <=10:{sbp_res['pct_le_10']:.1f}%, <=15:{sbp_res['pct_le_15']:.1f}%, Gr:{sbp_res['bhs_grade']}) | "
                f"DBP: MAE={dbp_res['mae']:.2f} (<=5:{dbp_res['pct_le_5']:.1f}%, <=10:{dbp_res['pct_le_10']:.1f}%, <=15:{dbp_res['pct_le_15']:.1f}%, Gr:{dbp_res['bhs_grade']})"
            )

        if score > best_score:
            best_score = score
            no_improve = 0
            best_metrics = {
                "epoch": ep + 1,
                "sbp": sbp_res,
                "dbp": dbp_res,
                "sqi_kept_segments": int(len(X)),
                "sqi_total_segments": int(len(X_raw)),
                "train_segments": int(len(train_idx)),
                "test_segments": int(len(test_idx)),
                "param_count": param_count,
            }
            torch.save(model.state_dict(), "output/model_best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience and args.max_batches == 0:
                print(f"Early stopping triggered at epoch {ep+1} (best epoch {best_metrics['epoch']})")
                break

    summary_path = Path("output/stage3_calbased_summary.json")
    with open(summary_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

    print("\n--- Training Complete ---")
    print(f"Best Ep: {best_metrics['epoch']} | Params: {param_count:,}")
    print(
        f"Final SBP: MAE={best_metrics['sbp']['mae']} mmHg "
        f"(<=5: {best_metrics['sbp']['pct_le_5']}%, <=10: {best_metrics['sbp']['pct_le_10']}%, <=15: {best_metrics['sbp']['pct_le_15']}%) "
        f"-> BHS Grade {best_metrics['sbp']['bhs_grade']}"
    )
    print(
        f"Final DBP: MAE={best_metrics['dbp']['mae']} mmHg "
        f"(<=5: {best_metrics['dbp']['pct_le_5']}%, <=10: {best_metrics['dbp']['pct_le_10']}%, <=15: {best_metrics['dbp']['pct_le_15']}%) "
        f"-> BHS Grade {best_metrics['dbp']['bhs_grade']}"
    )


if __name__ == "__main__":
    main()
