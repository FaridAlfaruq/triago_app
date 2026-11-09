"""ecg_deriv.py - ECG-Derived Respiration (EDR) implementation based on:

https://github.com/hiikariri/respiratory-rate-estimation/blob/main/respiration_rate.py

Implements single-lead ECG R-peak amplitude tracking with PCHIP interpolation
and estimation via Welch PSD or breath peak counting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, find_peaks, welch

STAGE_DIR = Path(__file__).resolve().parent
TRIAGO_DIR = Path(r"C:\Users\Adyty\Documents\Farid ITS\TriaGo")
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))
if str(TRIAGO_DIR) not in sys.path:
    sys.path.insert(0, str(TRIAGO_DIR))

import stage2
from processing_data.processing_data import ECGProcessor

# Defaults from hiikariri/respiratory-rate-estimation
EDR_LOWCUT = 5.0
EDR_HIGHCUT = 40.0
MIN_BREATH_INTERVAL = 1.17  # seconds (~51 bpm max)
HEIGHT_FRAC = 0.35
PROMINENCE_FRAC = 0.075
RESP_BAND = (10.0 / 60.0, 30.0 / 60.0)  # 10 to 30 rpm
WINDOW = 32.0  # seconds
FS = 125.0


def compute_edr(
    ecg: np.ndarray,
    fs: float = FS,
    r_peaks: np.ndarray | None = None,
    lowcut: float = EDR_LOWCUT,
    highcut: float = EDR_HIGHCUT,
    interp: str = "pchip",
    normalize: bool = True,
):
    """Extract respiration waveform from ECG R-peak amplitude modulation."""
    ecg = np.asarray(ecg, dtype=float)
    dt = 1.0 / fs

    # 1. Bandpass filter 5-40 Hz for QRS amplitude extraction
    nyq = 0.5 * fs
    b, a = butter(2, [lowcut / nyq, highcut / nyq], btype="bandpass")
    filtered = filtfilt(b, a, ecg)

    # 2. Fiducial R-peaks
    if r_peaks is None:
        ecg_proc = ECGProcessor(target_fs=int(fs))
        r_peaks, _ = ecg_proc.detect_r_peaks(ecg, fs=fs)
    r_peaks = np.unique(np.asarray(r_peaks, dtype=int))
    r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(ecg))]

    if len(r_peaks) < 4:
        raise ValueError(f"Need >= 4 R peaks; detected {len(r_peaks)}")

    r_loc = r_peaks * dt
    r_amp = filtered[r_peaks]

    # 3. Shape-preserving PCHIP interpolation to uniform grid
    interpolator = PchipInterpolator(r_loc, r_amp)
    edr_time = np.arange(r_loc[0], r_loc[-1], dt)
    edr = interpolator(edr_time)

    # 4. Z-score normalization
    if normalize:
        std = np.std(edr)
        edr = (edr - np.mean(edr)) / (std if std > 1e-9 else 1.0)

    return edr_time, edr, r_peaks


def count_breath_peaks(edr_window: np.ndarray, fs: float = FS):
    """Detect breath peaks via height, prominence, and distance thresholds."""
    if len(edr_window) < 3:
        return np.array([], dtype=int)

    height = HEIGHT_FRAC * np.mean(np.abs(edr_window))
    prominence = PROMINENCE_FRAC * np.ptp(edr_window)
    distance = max(1, int(round((MIN_BREATH_INTERVAL * 0.75) * fs)))

    peaks, _ = find_peaks(
        edr_window,
        distance=distance,
        height=height,
        prominence=prominence,
    )
    return peaks


def psd_rate(edr_window: np.ndarray, fs: float = FS, band: tuple = RESP_BAND) -> float:
    """Calculate dominant respiration frequency via Welch PSD."""
    x = edr_window - np.mean(edr_window)
    n = len(x)
    if n < 8:
        return float("nan")

    f, pxx = welch(x, fs=fs, nperseg=n)
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return float("nan")

    f_peak = f[mask][np.argmax(pxx[mask])]
    return float(f_peak * 60.0)


def respiration_rate_from_edr(
    edr_time: np.ndarray,
    edr: np.ndarray,
    fs: float = FS,
    window: float = WINDOW,
    method: str = "psd",
):
    """Calculate respiratory rate across sliding/non-overlapping windows."""
    t_start, t_end = edr_time[0], edr_time[-1]
    window_rates = []

    s = t_start
    while s < t_end - 1e-9:
        e = min(s + window, t_end)
        duration = e - s
        if duration >= 10.0:
            mask = (edr_time >= s) & (edr_time < e)
            seg = edr[mask]

            if method == "psd":
                rate = psd_rate(seg, fs)
            elif method == "peak":
                pks = count_breath_peaks(seg, fs)
                rate = len(pks) * 60.0 / duration
            else:
                raise ValueError(f"Unknown method {method}")

            if np.isfinite(rate):
                window_rates.append(rate)
        s += window

    overall = float(np.mean(window_rates)) if window_rates else float("nan")
    return window_rates, overall


def benchmark_ecg_deriv(case_id: int = 1):
    """Run hiikariri EDR on SQI-passed segments from VitalDB Case 1 and plot comparison."""
    df = stage2.load_vital(case_id)
    good_segs, _ = stage2.segment_and_process(df)
    ecg_proc = ECGProcessor(target_fs=int(FS))

    records = []
    sample_edr = None

    for seg in good_segs:
        r_peaks, _ = ecg_proc.detect_r_peaks(seg["ecg_raw"], fs=FS)
        edr_time, edr, peaks = compute_edr(seg["ecg_raw"], fs=FS, r_peaks=r_peaks)

        _, rate_psd = respiration_rate_from_edr(edr_time, edr, fs=FS, window=WINDOW, method="psd")
        _, rate_peak = respiration_rate_from_edr(edr_time, edr, fs=FS, window=WINDOW, method="peak")

        ref = seg["stats"]["rr_mean"]
        records.append({
            "seg_idx": seg["seg_idx"],
            "ref_rr": ref,
            "est_psd": rate_psd,
            "est_peak": rate_peak,
            "err_psd": abs(rate_psd - ref),
            "err_peak": abs(rate_peak - ref),
        })

        if sample_edr is None:
            sample_edr = {
                "time": edr_time,
                "edr": edr,
                "r_peaks": peaks,
                "ecg": seg["ecg_raw"],
                "ref": ref,
                "psd": rate_psd,
                "peak": rate_peak,
            }

    df_out = pd.DataFrame(records)

    # Print summary metrics
    mae_psd = df_out["err_psd"].mean()
    mae_peak = df_out["err_peak"].mean()
    cp2_psd = (df_out["err_psd"] <= 2.0).mean() * 100.0

    print("=" * 65)
    print("hiikariri/respiratory-rate-estimation Benchmark on VitalDB Case 1")
    print("=" * 65)
    print(f"Total Segments Tested : {len(df_out)}")
    print(f"PSD Method MAE        : {mae_psd:.2f} rpm (Within +/-2 rpm: {cp2_psd:.1f}%)")
    print(f"Peak Counting MAE     : {mae_peak:.2f} rpm")
    print("=" * 65)

    # Visualize 2 columns using plt.show()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    # Col 1: Time Series & Extracted Waveform
    axes[0, 0].plot(df_out["seg_idx"], df_out["ref_rr"], "g-o", label="Reference (VENT_RR)", markersize=3)
    axes[0, 0].plot(df_out["seg_idx"], df_out["est_psd"], "m-s", label=f"hiikariri PSD (MAE: {mae_psd:.2f})", markersize=3, alpha=0.8)
    axes[0, 0].set_title("Tracking: hiikariri PSD Method vs VENT_RR")
    axes[0, 0].set_ylabel("RR (rpm)")
    axes[0, 0].set_xlabel("Segment Index")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    t_edr = sample_edr["time"] - sample_edr["time"][0]
    axes[1, 0].plot(t_edr, sample_edr["edr"], color="purple", lw=1.2)
    axes[1, 0].set_title(f"Sample Segment #0 EDR Waveform (PSD: {sample_edr['psd']:.1f} rpm | Ref: {sample_edr['ref']:.1f} rpm)")
    axes[1, 0].set_ylabel("Z-scored Amplitude")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True, alpha=0.3)

    # Col 2: Correlation Scatter & Error Histogram
    axes[0, 1].scatter(df_out["ref_rr"], df_out["est_psd"], color="purple", alpha=0.7)
    r_min, r_max = min(df_out["ref_rr"].min(), df_out["est_psd"].min()) - 2, max(df_out["ref_rr"].max(), df_out["est_psd"].max()) + 2
    axes[0, 1].plot([r_min, r_max], [r_min, r_max], "k--", label="Identity (y=x)")
    axes[0, 1].plot([r_min, r_max], [r_min + 2, r_max + 2], "r:", label="+/- 2 rpm")
    axes[0, 1].plot([r_min, r_max], [r_min - 2, r_max - 2], "r:")
    axes[0, 1].set_title("Scatter: hiikariri PSD Method vs Reference")
    axes[0, 1].set_xlabel("VENT_RR (rpm)")
    axes[0, 1].set_ylabel("Estimated RR (rpm)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].hist(df_out["err_psd"], bins=15, color="purple", edgecolor="black", alpha=0.7)
    axes[1, 1].axvline(mae_psd, color="red", linestyle="--", label=f"MAE: {mae_psd:.2f} rpm")
    axes[1, 1].set_title("Absolute Error Distribution (PSD Method)")
    axes[1, 1].set_xlabel("Absolute Error (rpm)")
    axes[1, 1].set_ylabel("Segment Count")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("ECG-Derived Respiration (hiikariri/respiratory-rate-estimation) Evaluation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    benchmark_ecg_deriv(case_id=1)
