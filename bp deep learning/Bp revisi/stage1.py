"""Stage 1: Load, Preprocess, Downsample (125Hz), Invert PPG, Synchronize, Segment, and SQI Audit.

Ponytail principle: Minimal code, standard library + numpy + scipy + pandas + matplotlib.
Zero disk plot saves: Interactive plt.show() only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
if "DISPLAY" not in os.environ and sys.platform != "win32":
    matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, iirnotch, sosfiltfilt, welch
from scipy.stats import kurtosis, skew


def butter_bandpass(data: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter using second-order sections (SOS)."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data)


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Load target.csv containing patient IDs and BP labels."""
    target_path = data_dir / "metadata" / "target.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"target.csv not found at {target_path}")
    return pd.read_csv(target_path)


def load_raw_signals(subject_id: str, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw ECG and PPG recordings."""
    ecg_path = data_dir / "ECG" / f"{subject_id}_ecg.csv"
    ppg_path = data_dir / "PPG" / f"{subject_id}_ppg.csv"
    if not ecg_path.exists() or not ppg_path.exists():
        raise FileNotFoundError(f"Missing ECG or PPG for {subject_id}")
    return pd.read_csv(ecg_path), pd.read_csv(ppg_path)


def synchronize_signals(ecg_df: pd.DataFrame, ppg_df: pd.DataFrame, fs_target: float = 125.0) -> Dict[str, Any]:
    """Synchronize time base and sampling rate between ECG and PPG on uniform 125Hz grid."""
    t_ecg_raw = ecg_df["Time (s)"].to_numpy()
    sig_ecg_raw = ecg_df["ECG_Raw (V)"].to_numpy()

    t_ppg_raw = ppg_df["Time (s)"].to_numpy()
    sig_ppg_raw = ppg_df["PPG_IR"].to_numpy()

    t_start = max(float(t_ecg_raw[0]), float(t_ppg_raw[0]))
    t_end = min(float(t_ecg_raw[-1]), float(t_ppg_raw[-1]))
    t_sync = np.arange(t_start, t_end, 1.0 / fs_target)

    # Uniform interpolation onto 125 Hz grid
    ecg_sync = np.interp(t_sync, t_ecg_raw, sig_ecg_raw)
    ppg_sync = np.interp(t_sync, t_ppg_raw, sig_ppg_raw)

    # Invert MAX30105 optical absorption
    ppg_inv = -ppg_sync

    # Filter both signals
    ecg_filt = butter_bandpass(ecg_sync, low=0.5, high=40.0, fs=fs_target, order=4)
    ppg_filt = butter_bandpass(ppg_inv, low=0.5, high=10.0, fs=fs_target, order=4)

    return {
        "t_sync": t_sync,
        "fs": fs_target,
        "t_ecg_raw": t_ecg_raw,
        "sig_ecg_raw": sig_ecg_raw,
        "ecg_sync": ecg_sync,
        "ecg_filt": ecg_filt,
        "t_ppg_raw": t_ppg_raw,
        "sig_ppg_raw": sig_ppg_raw,
        "ppg_sync": ppg_sync,
        "ppg_inv": ppg_inv,
        "ppg_filt": ppg_filt,
    }


def evaluate_segment_sqi(ecg_seg: np.ndarray, ppg_seg: np.ndarray, fs: float = 125.0) -> Tuple[bool, str, Dict[str, float]]:
    """Multimodal Signal Quality Index (SQI) assessment for 10-second segments:
    
    1. Checks for NaNs, Infs, and sensor flatline.
    2. Pan-Tompkins QRS energy operator for robust ECG R-peak detection.
    3. Prominence peak detection for PPG pulsatile waves.
    4. Physiological Heart Rate bounds: [40, 180] BPM.
    5. Dual-channel HR concordance: |HR_ecg - HR_ppg| <= 6.0 BPM.
    6. Morphological criteria: PPG Skewness > -0.15, Kurtosis >= 1.8.
    7. Spectral SNR: >= 50% relative power in cardiac band (0.5 - 8.0 Hz).
    """
    if np.isnan(ecg_seg).any() or np.isnan(ppg_seg).any():
        return False, "NaN/Inf Present", {}
    if np.std(ecg_seg) < 1e-4 or np.std(ppg_seg) < 1e-4:
        return False, "Flatline Signal", {}

    # 1. ECG QRS Detection via Energy Operator (derivative^2 + MWI)
    d_ecg = np.diff(ecg_seg, prepend=ecg_seg[0])
    mwi_ecg = np.convolve(d_ecg**2, np.ones(18) / 18, mode="same")
    ecg_peaks, _ = find_peaks(mwi_ecg, distance=int(0.4 * fs), prominence=0.3 * np.std(mwi_ecg))

    # 2. PPG Systolic Peak Detection
    ppg_peaks, _ = find_peaks(ppg_seg, distance=int(0.4 * fs), prominence=0.25 * np.std(ppg_seg))

    if len(ecg_peaks) < 4:
        return False, f"ECG Peaks Insufficient ({len(ecg_peaks)})", {}
    if len(ppg_peaks) < 4:
        return False, f"PPG Peaks Insufficient ({len(ppg_peaks)})", {}

    # 3. Heart Rate Calculation & Concordance
    hr_ecg = 60.0 / np.mean(np.diff(ecg_peaks) / fs)
    hr_ppg = 60.0 / np.mean(np.diff(ppg_peaks) / fs)

    if not (40.0 <= hr_ecg <= 180.0) or not (40.0 <= hr_ppg <= 180.0):
        return False, f"HR Out of Bounds (ECG:{hr_ecg:.0f}, PPG:{hr_ppg:.0f})", {}

    hr_diff = abs(hr_ecg - hr_ppg)
    if hr_diff > 6.0:
        return False, f"HR Discordance (Diff:{hr_diff:.1f} bpm)", {}

    # 4. Statistical Morphological Metrics
    s_val = float(skew(ppg_seg))
    k_val = float(kurtosis(ppg_seg, fisher=False))
    if s_val <= -0.15:
        return False, f"Invalid PPG Skewness ({s_val:.2f})", {}
    if k_val < 1.8:
        return False, f"Invalid PPG Kurtosis ({k_val:.2f})", {}

    # 5. Spectral SNR (0.5 - 8.0 Hz Band Power)
    freqs, psd = welch(ppg_seg, fs=fs, nperseg=int(2 * fs))
    p_band = np.sum(psd[(freqs >= 0.5) & (freqs <= 8.0)])
    p_total = np.sum(psd) + 1e-9
    snr_ratio = float(p_band / p_total)
    if snr_ratio < 0.50:
        return False, f"Low Spectral SNR ({snr_ratio*100:.1f}%)", {}

    metrics = {
        "hr_ecg": round(float(hr_ecg), 1),
        "hr_ppg": round(float(hr_ppg), 1),
        "hr_diff": round(float(hr_diff), 1),
        "skewness": round(s_val, 2),
        "kurtosis": round(k_val, 2),
        "spectral_snr": round(snr_ratio, 3),
        "n_ecg_peaks": len(ecg_peaks),
        "n_ppg_peaks": len(ppg_peaks),
    }
    return True, "PASSED_SQI", metrics


def segment_with_sqi(sync_data: Dict[str, Any], window_sec: float = 10.0, stride_sec: float = 2.0) -> List[Dict[str, Any]]:
    """Segment synchronized signals with overlapping sliding window and evaluate SQI."""
    fs = sync_data["fs"]
    win_len = int(window_sec * fs)
    stride_len = int(stride_sec * fs)

    t_sync = sync_data["t_sync"]
    ecg_filt = sync_data["ecg_filt"]
    ppg_filt = sync_data["ppg_filt"]

    segments = []
    total_len = len(t_sync)
    for start_idx in range(0, total_len - win_len + 1, stride_len):
        end_idx = start_idx + win_len
        t_seg = t_sync[start_idx:end_idx]
        e_seg = ecg_filt[start_idx:end_idx]
        p_seg = ppg_filt[start_idx:end_idx]

        is_valid, reason, metrics = evaluate_segment_sqi(e_seg, p_seg, fs=fs)
        segments.append({
            "start_time": round(float(t_seg[0]), 2),
            "end_time": round(float(t_seg[-1]), 2),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "is_valid": is_valid,
            "reason": reason,
            "metrics": metrics,
        })
    return segments


def plot_all_stages(
    sync_data: Dict[str, Any],
    segments: List[Dict[str, Any]],
    subject_id: str,
    zoom_start: float = 10.0,
    zoom_dur: float = 10.0,
    show: bool = True,
) -> None:
    """Render Figures 1, 2, and 3 (Segmentation & SQI audit) interactively with zero disk saving."""
    zoom_end = zoom_start + zoom_dur

    t_sync = sync_data["t_sync"]
    m_sync = (t_sync >= zoom_start) & (t_sync <= zoom_end)
    t_win = t_sync[m_sync]

    t_ecg_raw = sync_data["t_ecg_raw"]
    sig_ecg_raw = sync_data["sig_ecg_raw"]
    m_ecg_raw = (t_ecg_raw >= zoom_start) & (t_ecg_raw <= zoom_end)

    t_ppg_raw = sync_data["t_ppg_raw"]
    sig_ppg_raw = sync_data["sig_ppg_raw"]
    m_ppg_raw = (t_ppg_raw >= zoom_start) & (t_ppg_raw <= zoom_end)

    ecg_sync_win = sync_data["ecg_sync"][m_sync]
    ecg_filt_win = sync_data["ecg_filt"][m_sync]
    ppg_inv_win = sync_data["ppg_inv"][m_sync]
    ppg_filt_win = sync_data["ppg_filt"][m_sync]

    # =========================================================================
    # FIGURE 1: ECG Processing (2x2 Grid)
    # [0, 0]: ECG raw data (~256 Hz)
    # [0, 1]: Downsampled data (125 Hz)
    # [1, 0]: Downsampled data (125 Hz)
    # [1, 1]: After filtering (0.5 - 40 Hz)
    # =========================================================================
    fig1, ax1 = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig1.canvas.manager.set_window_title(f"Figure 1: ECG Downsampling & Filtering - {subject_id}")

    ax1[0, 0].plot(t_ecg_raw[m_ecg_raw], sig_ecg_raw[m_ecg_raw], color="#1f77b4", lw=1.0)
    ax1[0, 0].set_title(f"ECG Raw Data (~256 Hz) - [0, 0]", fontweight="bold", fontsize=11)
    ax1[0, 0].set_ylabel("Amplitude (V)")
    ax1[0, 0].grid(True, linestyle="--", alpha=0.5)

    ax1[0, 1].plot(t_win, ecg_sync_win, color="#ff7f0e", lw=1.0)
    ax1[0, 1].set_title(f"ECG Downsampled to 125 Hz - [0, 1]", fontweight="bold", fontsize=11)
    ax1[0, 1].set_ylabel("Amplitude (V)")
    ax1[0, 1].grid(True, linestyle="--", alpha=0.5)

    ax1[1, 0].plot(t_win, ecg_sync_win, color="#ff7f0e", lw=1.0)
    ax1[1, 0].set_title(f"ECG Downsampled to 125 Hz - [1, 0]", fontweight="bold", fontsize=11)
    ax1[1, 0].set_xlabel("Time (s)")
    ax1[1, 0].set_ylabel("Amplitude (V)")
    ax1[1, 0].grid(True, linestyle="--", alpha=0.5)

    ax1[1, 1].plot(t_win, ecg_filt_win, color="#2ca02c", lw=1.2)
    ax1[1, 1].set_title(f"ECG Filtered (0.5 - 40.0 Hz) - [1, 1]", fontweight="bold", fontsize=11)
    ax1[1, 1].set_xlabel("Time (s)")
    ax1[1, 1].set_ylabel("Amplitude (V)")
    ax1[1, 1].grid(True, linestyle="--", alpha=0.5)

    fig1.suptitle(f"Figure 1: ECG Downsampling & Filtering ({subject_id})", fontsize=13, fontweight="bold")
    fig1.tight_layout()

    # =========================================================================
    # FIGURE 2: PPG Processing (2x2 Grid)
    # [0, 0]: Raw PPG signal
    # [0, 1]: Inverted signal (x = -x)
    # [1, 0]: Inverted signal (x = -x)
    # [1, 1]: Filtered signal (0.5 - 10 Hz)
    # =========================================================================
    fig2, ax2 = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig2.canvas.manager.set_window_title(f"Figure 2: PPG Inversion & Filtering - {subject_id}")

    ax2[0, 0].plot(t_ppg_raw[m_ppg_raw], sig_ppg_raw[m_ppg_raw], color="#d62728", lw=1.0)
    ax2[0, 0].set_title(f"Raw PPG Signal (MAX30105 IR) - [0, 0]", fontweight="bold", fontsize=11)
    ax2[0, 0].set_ylabel("Raw Counts")
    ax2[0, 0].grid(True, linestyle="--", alpha=0.5)

    ax2[0, 1].plot(t_win, ppg_inv_win, color="#9467bd", lw=1.0)
    ax2[0, 1].set_title(f"Inverted PPG Signal (x = -x) - [0, 1]", fontweight="bold", fontsize=11)
    ax2[0, 1].set_ylabel("Inverted Counts")
    ax2[0, 1].grid(True, linestyle="--", alpha=0.5)

    ax2[1, 0].plot(t_win, ppg_inv_win, color="#9467bd", lw=1.0)
    ax2[1, 0].set_title(f"Inverted PPG Signal (x = -x) - [1, 0]", fontweight="bold", fontsize=11)
    ax2[1, 0].set_xlabel("Time (s)")
    ax2[1, 0].set_ylabel("Inverted Counts")
    ax2[1, 0].grid(True, linestyle="--", alpha=0.5)

    ax2[1, 1].plot(t_win, ppg_filt_win, color="#17becf", lw=1.2)
    ax2[1, 1].set_title(f"Filtered PPG Signal (0.5 - 10.0 Hz) - [1, 1]", fontweight="bold", fontsize=11)
    ax2[1, 1].set_xlabel("Time (s)")
    ax2[1, 1].set_ylabel("Filtered Amplitude")
    ax2[1, 1].grid(True, linestyle="--", alpha=0.5)

    fig2.suptitle(f"Figure 2: PPG Inversion & Filtering ({subject_id})", fontsize=13, fontweight="bold")
    fig2.tight_layout()

    # =========================================================================
    # FIGURE 3: Segmentation & SQI Signal Quality Audit
    # Full timeline showing passed vs rejected segments & zoomed verified sample
    # =========================================================================
    valid_count = sum(1 for s in segments if s["is_valid"])
    total_count = len(segments)
    pass_pct = (valid_count / total_count * 100.0) if total_count > 0 else 0.0

    fig3, (ax_full_e, ax_full_p, ax_zoom) = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1, 1, 1.2]})
    fig3.canvas.manager.set_window_title(f"Figure 3: Segmentation & SQI Audit - {subject_id}")

    # Panel 1: Full ECG with SQI segment status
    ax_full_e.plot(t_sync, sync_data["ecg_filt"], color="#2ca02c", lw=0.8, label="Filtered ECG")
    ax_full_e.set_title(f"Full Recording Segmentation & SQI ({subject_id}): {valid_count}/{total_count} Segments Passed ({pass_pct:.1f}%)", fontweight="bold", fontsize=12)
    ax_full_e.set_ylabel("ECG (V)")
    ax_full_e.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Full PPG with SQI segment status
    ax_full_p.plot(t_sync, sync_data["ppg_filt"], color="#d62728", lw=0.8, label="Filtered PPG")
    ax_full_p.set_ylabel("PPG Amplitude")
    ax_full_p.set_xlabel("Time (s)")
    ax_full_p.grid(True, linestyle="--", alpha=0.4)

    # Color overlay for valid vs rejected segments
    for seg in segments:
        color = "#2ca02c" if seg["is_valid"] else "#d62728"
        alpha = 0.12 if seg["is_valid"] else 0.18
        ax_full_e.axvspan(seg["start_time"], seg["end_time"], color=color, alpha=alpha)
        ax_full_p.axvspan(seg["start_time"], seg["end_time"], color=color, alpha=alpha)

    # Panel 3: Zoomed verified segment
    ax_zoom.plot(t_win, ecg_filt_win / np.std(ecg_filt_win), color="#2ca02c", lw=1.2, label="Normalized ECG")
    ax_zoom.plot(t_win, ppg_filt_win / np.std(ppg_filt_win), color="#d62728", lw=1.2, label="Normalized Inverted PPG")
    ax_zoom.set_title(f"Zoomed 10-Second Segment ({zoom_start:.1f}s - {zoom_end:.1f}s) for Deep Learning Input", fontweight="bold", fontsize=11)
    ax_zoom.set_xlabel("Synchronized Shared Time Base (s)")
    ax_zoom.set_ylabel("Z-Normalized Units")
    ax_zoom.grid(True, linestyle="--", alpha=0.5)
    ax_zoom.legend(loc="upper right")

    fig3.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Synchronization, Segmentation (10s), and SQI Quality Gating")
    parser.add_argument("--subject", type=str, default="subject001", help="Subject ID to process")
    parser.add_argument("--zoom-start", type=float, default=10.0, help="Start time in seconds for 10s window")
    parser.add_argument("--window-sec", type=float, default=10.0, help="Segmentation window duration in seconds (default: 10.0)")
    parser.add_argument("--stride-sec", type=float, default=2.0, help="Sliding window stride in seconds (default: 2.0)")
    parser.add_argument("--no-show", action="store_true", help="Run headlessly without opening GUI plot windows")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Data"

    meta_df = load_metadata(data_dir)
    ecg_df, ppg_df = load_raw_signals(args.subject, data_dir)

    # 1. Synchronize signals on uniform 125 Hz grid
    sync_data = synchronize_signals(ecg_df, ppg_df, fs_target=125.0)

    # 2. Segment and evaluate Signal Quality Index (SQI)
    segments = segment_with_sqi(sync_data, window_sec=args.window_sec, stride_sec=args.stride_sec)
    valid_segments = [s for s in segments if s["is_valid"]]
    pass_pct = round(len(valid_segments) / len(segments) * 100.0, 1) if segments else 0.0

    # 3. Export contract JSON (< 1 KB)
    summary = {
        "task_id": "STAGE1-SEGMENTATION-SQI",
        "subject_id": args.subject,
        "sampling_rate_hz": sync_data["fs"],
        "segment_config": {
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "window_samples": int(args.window_sec * sync_data["fs"]),
        },
        "sqi_results": {
            "total_segments": len(segments),
            "passed_segments": len(valid_segments),
            "pass_rate_pct": pass_pct,
            "rejected_segments": len(segments) - len(valid_segments),
        },
        "sample_valid_metrics": valid_segments[0]["metrics"] if valid_segments else {},
    }

    summary_path = base_dir / "stage1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"=== Stage 1: Multimodal Synchronization, Segmentation & SQI Audit ===")
    print(f"Subject: {args.subject} | Grid: 125.0 Hz | Total Segments (10s): {len(segments)}")
    print(f"SQI Pass Rate: {len(valid_segments)} / {len(segments)} ({pass_pct}%) Passed Multi-Factor Filter")
    if valid_segments:
        m = valid_segments[0]["metrics"]
        print(f"Sample Valid Metrics: HR_ECG={m['hr_ecg']} bpm, HR_PPG={m['hr_ppg']} bpm, Diff={m['hr_diff']} bpm, SNR={m['spectral_snr']*100:.1f}%")
    print(f"Summary Contract: {summary_path.name} ({summary_path.stat().st_size} bytes)")

    plot_all_stages(
        sync_data=sync_data,
        segments=segments,
        subject_id=args.subject,
        zoom_start=args.zoom_start,
        zoom_dur=args.window_sec,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
