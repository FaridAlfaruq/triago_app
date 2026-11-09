"""Stage 2: 7-Channel Multimodal Feature Construction for Deep Learning (CNN-BiGRU).

Reuses loading, synchronization, and SQI from stage1.py (Ponytail principle: zero code duplication).
Output: X [N, 1250, 7], y_sbp [N], y_dbp [N], groups [N] saved to dataset_7ch.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
if "DISPLAY" not in os.environ and sys.platform != "win32":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert

# Reuse verified pipeline functions from stage1
from stage1 import (
    evaluate_segment_sqi,
    load_metadata,
    load_raw_signals,
    synchronize_signals,
)


def filter_physiological_targets(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Filter non-physiological target outliers (e.g. inverted SBP < DBP)."""
    valid = (
        (meta_df["SBP"] >= 70.0)
        & (meta_df["SBP"] <= 200.0)
        & (meta_df["DBP"] >= 40.0)
        & (meta_df["DBP"] <= 120.0)
        & (meta_df["SBP"] > meta_df["DBP"])
    )
    return meta_df[valid].copy().reset_index(drop=True)


def extract_7channel_features(ecg_seg: np.ndarray, ppg_seg: np.ndarray, fs: float = 125.0) -> np.ndarray:
    """Construct 7 physiological channels for CNN-BiGRU [1250, 7]."""
    # 1. Z-score normalization
    ecg_norm = (ecg_seg - np.mean(ecg_seg)) / (np.std(ecg_seg) + 1e-8)
    ppg_norm = (ppg_seg - np.mean(ppg_seg)) / (np.std(ppg_seg) + 1e-8)

    # 2. VPG (1st derivative)
    vpg = np.gradient(ppg_norm, 1.0 / fs)
    vpg_norm = (vpg - np.mean(vpg)) / (np.std(vpg) + 1e-8)

    # 3. APG (2nd derivative)
    apg = np.gradient(vpg_norm, 1.0 / fs)
    apg_norm = (apg - np.mean(apg)) / (np.std(apg) + 1e-8)

    # 4. Temporal Cross-Interaction
    corr = ecg_norm * ppg_norm
    corr_norm = (corr - np.mean(corr)) / (np.std(corr) + 1e-8)

    # 5. Hilbert Instantaneous Amplitude Envelopes
    env_ecg = np.abs(hilbert(ecg_norm))
    env_ecg_norm = (env_ecg - np.mean(env_ecg)) / (np.std(env_ecg) + 1e-8)

    env_ppg = np.abs(hilbert(ppg_norm))
    env_ppg_norm = (env_ppg - np.mean(env_ppg)) / (np.std(env_ppg) + 1e-8)

    return np.stack([
        ecg_norm,
        ppg_norm,
        vpg_norm,
        apg_norm,
        corr_norm,
        env_ecg_norm,
        env_ppg_norm,
    ], axis=-1).astype(np.float32)


def plot_7channel_sample(feat_7ch: np.ndarray, fs: float = 125.0, title_suffix: str = "", show: bool = True) -> None:
    """Render interactive 7-channel feature representation for review."""
    t_sec = np.arange(len(feat_7ch)) / fs
    labels = [
        ("Ch 1: Normalized ECG", "#2ca02c"),
        ("Ch 2: Normalized PPG", "#d62728"),
        ("Ch 3: VPG (PPG 1st Deriv)", "#9467bd"),
        ("Ch 4: APG (PPG 2nd Deriv)", "#8c564b"),
        ("Ch 5: Cross-Interaction (ECG * PPG)", "#e377c2"),
        ("Ch 6: ECG Hilbert Envelope", "#ff7f0e"),
        ("Ch 7: PPG Hilbert Envelope", "#17becf"),
    ]

    fig, axes = plt.subplots(7, 1, figsize=(14, 11), sharex=True)
    fig.canvas.manager.set_window_title(f"Figure: 7-Channel Deep Learning Input - {title_suffix}")

    for idx, (label, color) in enumerate(labels):
        axes[idx].plot(t_sec, feat_7ch[:, idx], color=color, lw=1.1)
        axes[idx].set_ylabel(label, fontsize=9, fontweight="bold")
        axes[idx].grid(True, linestyle="--", alpha=0.4)
        axes[idx].tick_params(labelsize=8)

    axes[-1].set_xlabel("Time in 10-Second Segment (s)", fontsize=11, fontweight="bold")
    fig.suptitle(f"7-Channel Input Tensor Representation [1250 x 7] ({title_suffix})", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_7channel_dataset(
    data_dir: Path,
    window_sec: float = 10.0,
    stride_sec: float = 2.0,
    fs: float = 125.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process all subjects, segment, SQI audit, and assemble 7-channel dataset."""
    raw_meta = load_metadata(data_dir)
    meta_df = filter_physiological_targets(raw_meta)
    win_len = int(window_sec * fs)
    stride_len = int(stride_sec * fs)

    X_list, y_sbp_list, y_dbp_list, groups_list = [], [], [], []

    for _, row in meta_df.iterrows():
        sid = row["subject_id"]
        pid = row["patient_id"]
        sbp = float(row["SBP"])
        dbp = float(row["DBP"])

        ecg_df, ppg_df = load_raw_signals(sid, data_dir)
        sync_data = synchronize_signals(ecg_df, ppg_df, fs_target=fs)

        ecg_clean = sync_data["ecg_filt"]
        ppg_clean = sync_data["ppg_filt"]

        for s_idx in range(0, len(ecg_clean) - win_len + 1, stride_len):
            e_seg = ecg_clean[s_idx : s_idx + win_len]
            p_seg = ppg_clean[s_idx : s_idx + win_len]

            is_valid, _, _ = evaluate_segment_sqi(e_seg, p_seg, fs=fs)
            if is_valid:
                feat = extract_7channel_features(e_seg, p_seg, fs=fs)
                X_list.append(feat)
                y_sbp_list.append(sbp)
                y_dbp_list.append(dbp)
                groups_list.append(pid)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_sbp_list, dtype=np.float32),
        np.array(y_dbp_list, dtype=np.float32),
        np.array(groups_list),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: 7-Channel Feature Construction for Deep Learning")
    parser.add_argument("--no-show", action="store_true", help="Run headlessly without opening GUI plot window")
    parser.add_argument("--preview-idx", type=int, default=0, help="Index of valid segment to preview")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Data"

    print("=== Stage 2: 7-Channel Input Tensor Construction ===")
    X, y_sbp, y_dbp, groups = build_7channel_dataset(data_dir, window_sec=10.0, stride_sec=2.0, fs=125.0)

    # Save compact NPZ artifact for Stage 3 deep learning models
    npz_path = base_dir / "dataset_7ch.npz"
    np.savez_compressed(npz_path, X=X, y_sbp=y_sbp, y_dbp=y_dbp, groups=groups)

    # Save summary contract (< 1 KB)
    unique_patients = len(np.unique(groups))
    summary = {
        "task_id": "STAGE2-7CHANNEL-FEATURE-EXTRACTION",
        "tensor_shape": list(X.shape),
        "target_shapes": {"y_sbp": list(y_sbp.shape), "y_dbp": list(y_dbp.shape)},
        "channels": [
            "ECG_Norm", "PPG_Norm", "VPG_Norm", "APG_Norm",
            "Cross_Interaction_Norm", "ECG_Hilbert_Env", "PPG_Hilbert_Env"
        ],
        "total_valid_segments": int(X.shape[0]),
        "unique_patients_covered": unique_patients,
        "dataset_npz_bytes": int(npz_path.stat().st_size),
        "target_stats": {
            "sbp_mean": round(float(np.mean(y_sbp)), 2),
            "sbp_std": round(float(np.std(y_sbp)), 2),
            "dbp_mean": round(float(np.mean(y_dbp)), 2),
            "dbp_std": round(float(np.std(y_dbp)), 2),
        },
    }

    summary_path = base_dir / "stage2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Input Tensor Shape: {X.shape} [Segments, TimeSteps(10s@125Hz), Channels]")
    print(f"Total Valid Segments: {len(X)} across {unique_patients} Unique Patients")
    print(f"SBP: mean {summary['target_stats']['sbp_mean']} mmHg | DBP: mean {summary['target_stats']['dbp_mean']} mmHg")
    print(f"NPZ Tensor File: {npz_path.name} ({npz_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Summary JSON: {summary_path.name} ({summary_path.stat().st_size} bytes)")

    if len(X) > 0:
        pid = groups[args.preview_idx]
        plot_7channel_sample(
            X[args.preview_idx],
            fs=125.0,
            title_suffix=f"Segment #{args.preview_idx} (Patient {pid}, SBP {y_sbp[args.preview_idx]:.0f}/{y_dbp[args.preview_idx]:.0f})",
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
