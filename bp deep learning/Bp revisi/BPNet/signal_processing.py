"""
signal_processing.py
Signal quality assessment (SQI) and preprocessing pipeline for PulseDB.

Stages:
1. Raw Continuous Signals (PulseDB v2.0 @ 125 Hz).
2. Continuous Zero-Phase Butterworth Bandpass Filtering (Linear detrend + SOS zero-phase).
   ECG: 0.5-40.0 Hz | PPG: 0.5-8.0 Hz (no segment boundary transients).
3. 10-Second Window Segmentation (1250 samples @ 125 Hz).
4. 4-Stage Tight SQI Screening (Pass vs Fail).
5. 10s Zoom Case Studies (Passed vs Failed Segment).
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, sosfiltfilt, detrend, find_peaks, welch, hilbert

from load import resolve_pulsedb_root, load_raw_subject, list_available_subjects


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardize signal to zero mean and unit variance."""
    return (x - np.mean(x)) / (np.std(x) + eps)


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float = 125.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter using Second-Order Sections (SOS).
    Applies linear detrending prior to filtering to eliminate DC step transients,
    with padlen to prevent edge ringing.
    """
    data_detrend = detrend(data, type="linear")
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    pad = min(len(data) - 1, int(fs * 2))
    return sosfiltfilt(sos, data_detrend, padlen=pad)


def extract_7ch_segment(ecg_clean: np.ndarray, ppg_clean: np.ndarray) -> np.ndarray:
    """
    Extract CalBPNet 7-channel waveform tensor of shape (1250, 7):
    Channels: [ECG, PPG, VPG, APG, ECG*PPG Corr, ECG Envelope, PPG Envelope]
    """
    ecg_norm = zscore(ecg_clean)
    ppg_norm = zscore(ppg_clean)
    vpg_norm = zscore(np.gradient(ppg_norm))
    apg_norm = zscore(np.gradient(vpg_norm))
    corr_norm = zscore(ecg_norm * ppg_norm)
    ecg_env_norm = zscore(np.abs(hilbert(ecg_norm)))
    ppg_env_norm = zscore(np.abs(hilbert(ppg_norm)))

    return np.column_stack([
        ecg_norm, ppg_norm, vpg_norm, apg_norm, corr_norm, ecg_env_norm, ppg_env_norm
    ]).astype(np.float32)


def compute_cardiac_psd_ratio(
    ppg_signal: np.ndarray,
    fs: int = 125,
    cardiac_band: Tuple[float, float] = (0.8, 3.5),
    total_band: Tuple[float, float] = (0.5, 15.0),
) -> float:
    """Calculate spectral cardiac power purity ratio using Welch periodogram."""
    f, psd = welch(ppg_signal, fs=fs, nperseg=256)
    cardiac_power = np.sum(psd[(f >= cardiac_band[0]) & (f <= cardiac_band[1])])
    total_power = np.sum(psd[(f >= total_band[0]) & (f <= total_band[1])]) + 1e-8
    return float(cardiac_power / total_power)


def evaluate_segment_sqi(
    ecg_signal: np.ndarray,
    ppg_signal: np.ndarray,
    fs: int = 125,
    min_bpm: float = 45.0,
    max_bpm: float = 135.0,
    max_beat_mismatch: int = 1,
    max_ibi_cv: float = 0.15,
    min_psd_ratio: float = 0.65,
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate 4-stage Tight SQI on a single 10-second segment."""
    duration_s = len(ecg_signal) / fs
    min_beats = int(min_bpm * duration_s / 60.0)
    max_beats = int(max_bpm * duration_s / 60.0)

    min_peak_distance = int(fs * 0.4)
    r_peaks, _ = find_peaks(ecg_signal, distance=min_peak_distance, prominence=0.8)
    p_peaks, _ = find_peaks(ppg_signal, distance=min_peak_distance, prominence=0.5)

    n_r = len(r_peaks)
    n_p = len(p_peaks)

    # Stage 1: Heart Rate (45 - 135 bpm)
    stage1 = (min_beats <= n_r <= max_beats) and (min_beats <= n_p <= max_beats)

    # Stage 2: Dual-Sensor Peak Coherence (|N_ecg - N_ppg| <= 1)
    stage2 = stage1 and (abs(n_r - n_p) <= max_beat_mismatch)

    # Stage 3: Rhythm Regularity (IBI CV <= 0.15)
    ibi_cv = 0.0
    stage3 = stage2
    if stage2 and n_p > 3:
        ibi = np.diff(p_peaks)
        ibi_cv = float(np.std(ibi) / (np.mean(ibi) + 1e-6))
        stage3 = (ibi_cv <= max_ibi_cv)

    # Stage 4: Spectral Power Purity (PSD Ratio >= 0.65)
    psd_ratio = compute_cardiac_psd_ratio(ppg_signal, fs=fs)
    stage4 = stage3 and (psd_ratio >= min_psd_ratio)

    details = {
        "ecg_beats": n_r,
        "ppg_beats": n_p,
        "r_peaks": r_peaks,
        "p_peaks": p_peaks,
        "ibi_cv": round(ibi_cv, 4),
        "psd_ratio": round(psd_ratio, 4),
        "stage1_hr_pass": stage1,
        "stage2_coherence_pass": stage2,
        "stage3_regularity_pass": stage3,
        "stage4_spectral_pass": stage4,
    }
    return stage4, details


def get_failure_reason(details: Dict[str, Any]) -> str:
    """Generate concise failure explanation."""
    reasons = []
    if not details.get("stage1_hr_pass", True):
        reasons.append(f"HR Out of Range (ECG:{details['ecg_beats']}, PPG:{details['ppg_beats']} beats)")
    elif not details.get("stage2_coherence_pass", True):
        reasons.append(f"Beat Mismatch (|ECG {details['ecg_beats']} - PPG {details['ppg_beats']}| > 1)")
    elif not details.get("stage3_regularity_pass", True):
        reasons.append(f"Irregular Rhythm (IBI CV {details['ibi_cv']:.3f} > 0.15)")
    elif not details.get("stage4_spectral_pass", True):
        reasons.append(f"Low Cardiac PSD ({details['psd_ratio']:.2f} < 0.65)")
    return "; ".join(reasons) if reasons else "Quality Check Failed"


def preprocess_subject_signals(
    ecg_raw: np.ndarray,
    ppg_raw: np.ndarray,
    sbp_raw: np.ndarray,
    dbp_raw: np.ndarray,
    fs: int = 125,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess recording:
    1. Continuous zero-phase bandpass filtering (eliminates segment seam transients).
    2. Slices into 10-second segments.
    3. Tight SQI evaluation and 7-channel CalBPNet feature extraction.
    """
    n_segs = len(ecg_raw)
    ecg_cont = ecg_raw.flatten()
    ppg_cont = ppg_raw.flatten()

    # Continuous filtering across recording
    ecg_filt_cont = butter_bandpass_filter(ecg_cont, 0.5, 40.0, fs=fs)
    ppg_filt_cont = butter_bandpass_filter(ppg_cont, 0.5, 8.0, fs=fs)

    feat_list = []
    clean_indices = []

    for i in range(n_segs):
        seg_ecg = ecg_filt_cont[i * 1250 : (i + 1) * 1250]
        seg_ppg = ppg_filt_cont[i * 1250 : (i + 1) * 1250]

        is_clean, _ = evaluate_segment_sqi(seg_ecg, seg_ppg, fs=fs)
        if is_clean:
            ch7 = extract_7ch_segment(seg_ecg, seg_ppg)
            feat_list.append(ch7)
            clean_indices.append(i)

    sqi_mask = np.zeros(n_segs, dtype=bool)
    sqi_mask[clean_indices] = True

    if len(feat_list) == 0:
        return np.empty((0, 1250, 7), dtype=np.float32), np.empty(0), np.empty(0), sqi_mask

    return np.array(feat_list, dtype=np.float32), sbp_raw[sqi_mask], dbp_raw[sqi_mask], sqi_mask


def plot_preprocessing_pipeline(
    subject_id: str = "p000005",
    max_segments: int = 10,
    source: str = "PulseDB_Vital",
    fs: int = 125,
) -> None:
    """
    Visualize 4-stage pipeline with simple titles and clean legends:
    - Row 0: Raw Signals (ECG & PPG @ 125 Hz).
    - Row 1: Zero-Phase Filtered Signals.
    - Row 2: 10s Window Segmentation & SQI Screening Map (Green: Pass, Red: Fail).
    - Row 3: Zoomed 10s Segments (Left: Passed Segment, Right: Failed Segment).
    """
    print(f"\nLoading raw PulseDB subject '{subject_id}' ({source})...")
    ecg_raw, ppg_raw, abp_raw, sbp, dbp = load_raw_subject(
        subject_id=subject_id,
        max_segments=max_segments,
        source=source,
    )

    n_segs = len(ecg_raw)
    total_samples = n_segs * 1250
    total_seconds = total_samples / fs
    time_continuous = np.linspace(0, total_seconds, total_samples)

    # 1. Continuous Raw Signals
    ecg_raw_cont = ecg_raw.flatten()
    ppg_raw_cont = ppg_raw.flatten()

    # 2. Continuous Zero-Phase Filtering (Prevents segment boundary transients)
    ecg_filt_cont = butter_bandpass_filter(ecg_raw_cont, 0.5, 40.0, fs=fs)
    ppg_filt_cont = butter_bandpass_filter(ppg_raw_cont, 0.5, 8.0, fs=fs)

    # 3. Clean slice into 10s segments
    ecg_filt_list = [ecg_filt_cont[i * 1250 : (i + 1) * 1250] for i in range(n_segs)]
    ppg_filt_list = [ppg_filt_cont[i * 1250 : (i + 1) * 1250] for i in range(n_segs)]

    # 4. SQI Evaluation
    sqi_results = []
    for i in range(n_segs):
        ok, details = evaluate_segment_sqi(ecg_filt_list[i], ppg_filt_list[i], fs=fs)
        sqi_results.append((ok, details))

    pass_indices = [i for i in range(n_segs) if sqi_results[i][0]]
    fail_indices = [i for i in range(n_segs) if not sqi_results[i][0]]
    n_pass = len(pass_indices)
    pass_pct = (n_pass / n_segs) * 100.0

    p_idx = pass_indices[0] if pass_indices else 0
    f_idx = fail_indices[0] if fail_indices else (1 if n_segs > 1 else 0)

    # --------------------------------------------------------------------------
    # PRINT DETAILED EXPLANATION TO TERMINAL CONSOLE
    # --------------------------------------------------------------------------
    print("=" * 65)
    print("PULSEDB SIGNAL PROCESSING & SQI REPORT")
    print("=" * 65)
    print(f"Subject ID         : {subject_id} ({source})")
    print(f"Sampling Rate      : {fs} Hz")
    print(f"Total Duration     : {total_seconds:.0f} seconds ({n_segs} segments of 10s)")
    print(f"Filtering Method   : Continuous linear detrend + Zero-phase SOS")
    print(f"Reference BP       : SBP {sbp.mean():.1f} mmHg, DBP {dbp.mean():.1f} mmHg")
    print("-" * 65)
    print("SQI SCREENING PER 10s SEGMENT:")
    for i in range(n_segs):
        ok, d = sqi_results[i]
        status = "PASS" if ok else "FAIL"
        reason = "" if ok else f" -> Reason: {get_failure_reason(d)}"
        print(f"  Segment {i:2d} ({i*10:3d}-{(i+1)*10:3d}s): {status:4s} | Beats=(ECG:{d['ecg_beats']:2d}, PPG:{d['ppg_beats']:2d}) | CV={d['ibi_cv']:.3f} | PSD={d['psd_ratio']:.2f}{reason}")
    print("-" * 65)
    print(f"SUMMARY: {n_pass} / {n_segs} Segments Kept ({pass_pct:.1f}% Clean)")
    print(f"  Passed Example : Segment {p_idx} (HR: {sqi_results[p_idx][1]['ecg_beats']*6} bpm, PSD: {sqi_results[p_idx][1]['psd_ratio']:.2f})")
    print(f"  Failed Example : Segment {f_idx} ({get_failure_reason(sqi_results[f_idx][1])})")
    print("=" * 65 + "\n")

    # --------------------------------------------------------------------------
    # SIMPLE PLOT (Clean Titles, Simple Legends, Zero Text Clutter)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(13, 10))
    plt.subplots_adjust(hspace=0.45, wspace=0.22)

    # ROW 0: RAW SIGNALS
    ax = axes[0, 0]
    ax.plot(time_continuous, ecg_raw_cont, color="#1f77b4", linewidth=0.8, label="ECG")
    ax.set_title("Raw ECG", fontsize=11, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    ax.plot(time_continuous, ppg_raw_cont, color="#2ca02c", linewidth=0.8, label="PPG")
    ax.set_title("Raw PPG", fontsize=11, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    # ROW 1: FILTERED SIGNALS
    ax = axes[1, 0]
    ax.plot(time_continuous, ecg_filt_cont, color="#1f77b4", linewidth=0.8, label="ECG")
    ax.set_title("Filtered ECG", fontsize=11, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    ax.plot(time_continuous, ppg_filt_cont, color="#2ca02c", linewidth=0.8, label="PPG")
    ax.set_title("Filtered PPG", fontsize=11, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    # ROW 2: SEGMENTATION & SQI MAP
    pass_patch = mpatches.Patch(color="lightgreen", alpha=0.5, label="Pass")
    fail_patch = mpatches.Patch(color="#ff9999", alpha=0.5, label="Fail")

    for col_idx, (sig_cont, name, col) in enumerate([(ecg_filt_cont, "ECG", "#1f77b4"), (ppg_filt_cont, "PPG", "#2ca02c")]):
        ax = axes[2, col_idx]
        ax.plot(time_continuous, sig_cont, color=col, linewidth=0.7)

        for i in range(n_segs):
            t_start = i * 10.0
            t_end = (i + 1) * 10.0
            is_ok = sqi_results[i][0]
            bg_color = "lightgreen" if is_ok else "#ff9999"
            ax.axvspan(t_start, t_end, color=bg_color, alpha=0.35)
            ax.axvline(t_start, color="gray", linestyle="--", linewidth=0.6)

        ax.set_title(f"Segmentation - {name}", fontsize=11, weight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(handles=[pass_patch, fail_patch], loc="upper right", fontsize=8)

    # ROW 3: ZOOMED 10s SEGMENTS (Pass vs Fail) - Simple Legends, No Dot Labels
    time_10s = np.linspace(0, 10.0, 1250)

    # 3A: PASSED SEGMENT
    ax = axes[3, 0]
    p_ecg = ecg_filt_list[p_idx]
    p_ppg = ppg_filt_list[p_idx]
    p_ok, p_det = sqi_results[p_idx]
    p_r = p_det["r_peaks"]
    p_p = p_det["p_peaks"]

    ax.plot(time_10s, p_ecg, color="#1f77b4", linewidth=1.1, label="ECG")
    ax.plot(time_10s, p_ppg, color="#2ca02c", linewidth=1.1, label="PPG")
    if len(p_r) > 0:
        ax.scatter(time_10s[p_r], p_ecg[p_r], color="red", s=30, zorder=5)
    if len(p_p) > 0:
        ax.scatter(time_10s[p_p], p_ppg[p_p], color="crimson", s=30, zorder=5)

    ax.set_title(f"Segment {p_idx} (Passed)", fontsize=11, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    # 3B: FAILED SEGMENT
    ax = axes[3, 1]
    f_ecg = ecg_filt_list[f_idx]
    f_ppg = ppg_filt_list[f_idx]
    f_ok, f_det = sqi_results[f_idx]
    f_r = f_det["r_peaks"]
    f_p = f_det["p_peaks"]

    ax.plot(time_10s, f_ecg, color="#1f77b4", linewidth=1.1, label="ECG")
    ax.plot(time_10s, f_ppg, color="#2ca02c", linewidth=1.1, label="PPG")
    if len(f_r) > 0:
        ax.scatter(time_10s[f_r], f_ecg[f_r], color="red", s=30, zorder=5)
    if len(f_p) > 0:
        ax.scatter(time_10s[f_p], f_ppg[f_p], color="crimson", s=30, zorder=5)

    status_tag = "Failed" if not f_ok else "Passed"
    ax.set_title(f"Segment {f_idx} ({status_tag})", fontsize=11, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"PulseDB Preprocessing Pipeline (Subject: {subject_id})", fontsize=13, weight="bold")
    print("Displaying full signal processing pipeline via plt.show()...")
    plt.tight_layout()
    plt.show()


# Backward compatibility alias
plot_preprocessing_comparison = plot_preprocessing_pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PulseDB signal processing & Tight SQI pipeline.")
    parser.add_argument("--subject", type=str, default="p000005", help="Subject ID (default: p000005)")
    parser.add_argument("--segments", type=int, default=10, help="Number of 10s segments (default: 10)")
    parser.add_argument("--source", type=str, default="PulseDB_Vital", choices=["PulseDB_Vital", "PulseDB_MIMIC"])
    parser.add_argument("--no-plot", action="store_true", help="Print summary without opening plot window")
    args = parser.parse_args()

    if args.no_plot:
        ecg_raw, ppg_raw, abp_raw, sbp, dbp = load_raw_subject(
            subject_id=args.subject,
            max_segments=args.segments,
            source=args.source,
        )
        X_clean, sbp_clean, dbp_clean, mask = preprocess_subject_signals(
            ecg_raw, ppg_raw, sbp, dbp, fs=125
        )
        print(f"Tight SQI Clean: {len(X_clean)} / {len(ecg_raw)} segments kept ({len(X_clean)/len(ecg_raw)*100:.1f}%).")
    else:
        plot_preprocessing_pipeline(
            subject_id=args.subject,
            max_segments=args.segments,
            source=args.source,
        )
