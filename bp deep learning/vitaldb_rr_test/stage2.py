import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import vitaldb

VITAL_DIR = Path(__file__).resolve().parents[2] / "temp estimation" / "vital_files"
TRACKS = ["SNUADC/ECG_II", "SNUADC/PLETH", "Solar8000/VENT_RR"]
FS_RAW = 500.0
FS_DS = 125.0
WIN_SEC = 60.0
WIN_SAMPLES = int(WIN_SEC * FS_DS)


def load_vital(case_id: int = 1):
    """Load local vital file trimmed strictly to valid VENT_RR (> 0)."""
    path = VITAL_DIR / f"{case_id}.vital"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    vf = vitaldb.VitalFile(str(path))
    df = vf.to_pandas(TRACKS, interval=1.0 / FS_DS)

    valid_mask = df["Solar8000/VENT_RR"].notna() & (df["Solar8000/VENT_RR"] > 0)
    valid_idx = df.index[valid_mask]
    df_valid = df.loc[valid_idx[0] : valid_idx[-1]].copy()
    df_valid["Solar8000/VENT_RR"] = df_valid["Solar8000/VENT_RR"].ffill()
    return df_valid.reset_index(drop=True)


def bandpass_transient_free(x: np.ndarray, low: float = 0.5, high: float = 10.0, fs: float = FS_DS, pad_sec: float = 2.0):
    """Butterworth 0.5-10 Hz bandpass with DC detrending and reflection padding to kill edge transients."""
    pad_len = int(pad_sec * fs)
    b, a = signal.butter(2, [low, high], btype="bandpass", fs=fs)

    # Detrend DC offset and pad boundary to absorb filter settling response
    x_detrend = signal.detrend(x)
    x_padded = np.pad(x_detrend, pad_len, mode="reflect")
    x_filt = signal.filtfilt(b, a, x_padded)
    return x_filt[pad_len:-pad_len]


def evaluate_sqi(ecg_raw: np.ndarray, ppg_raw: np.ndarray, rr: np.ndarray, fs: float = FS_DS):
    """Tighter SQI: enforces physiological QRS voltage, kurtosis bounds, baseline wander limit, pulse rhythm, and RR stability."""
    reasons = []

    # 1. ECG Rules: QRS amplitude, kurtosis bounds (rejects flatline & delta-spikes), baseline wander power
    ecg_ptp = float(np.ptp(ecg_raw))
    ecg_kurt = float(stats.kurtosis(ecg_raw))
    f_ecg, p_ecg = signal.welch(ecg_raw - np.mean(ecg_raw), fs=fs, nperseg=500)
    bw_ratio = float(np.sum(p_ecg[f_ecg < 0.5]) / (np.sum(p_ecg) + 1e-9))

    ecg_pass = True
    if not (0.30 <= ecg_ptp <= 4.5):
        ecg_pass = False
        reasons.append(f"ECG PTP {ecg_ptp:.1f}mV out of [0.3, 4.5]")
    if not (4.5 <= ecg_kurt <= 40.0):
        ecg_pass = False
        reasons.append(f"ECG Kurt {ecg_kurt:.1f} out of [4.5, 40.0]")
    if bw_ratio >= 0.55:
        ecg_pass = False
        reasons.append(f"ECG Baseline Wander {bw_ratio:.2f} >= 0.55")

    # 2. PPG Rules: Dynamic range, physiological pulse rate (45-135 bpm), and interval regularity CV
    ppg_ptp = float(np.ptp(ppg_raw))
    pks, _ = signal.find_peaks(ppg_raw - np.mean(ppg_raw), distance=int(0.4 * fs), prominence=0.15 * ppg_ptp)
    ppg_hr = len(pks)
    ibi = np.diff(pks) / fs if len(pks) >= 2 else np.array([])
    ibi_cv = float(np.std(ibi) / (np.mean(ibi) + 1e-9)) if len(ibi) >= 5 else 1.0

    ppg_pass = True
    if not (8.0 <= ppg_ptp <= 105.0):
        ppg_pass = False
        reasons.append(f"PPG PTP {ppg_ptp:.1f} out of [8, 105]")
    if not (45 <= ppg_hr <= 135):
        ppg_pass = False
        reasons.append(f"PPG Heart Rate {ppg_hr} out of [45, 135]")
    if ibi_cv >= 0.35:
        ppg_pass = False
        reasons.append(f"PPG Irregularity CV {ibi_cv:.2f} >= 0.35")

    # 3. Ground-truth Ventilator RR SQI: Physiological breathing rate
    rr_pass = bool(np.all((rr >= 6.0) & (rr <= 35.0)))
    if not rr_pass:
        reasons.append(f"VENT_RR [{np.min(rr):.1f}, {np.max(rr):.1f}] out of [6, 35]")

    is_good = bool(ecg_pass and ppg_pass and rr_pass)

    return is_good, {
        "ecg_ptp": round(ecg_ptp, 2),
        "ecg_kurt": round(ecg_kurt, 1),
        "bw_ratio": round(bw_ratio, 2),
        "ppg_peaks": ppg_hr,
        "ibi_cv": round(ibi_cv, 2),
        "rr_mean": round(float(np.mean(rr)), 1),
        "rejection_reason": "; ".join(reasons) if reasons else "Passed",
    }


def segment_and_process(df):
    """Segment data into 60s windows, categorize into good (passed) and bad (failed) segments."""
    n_segs = len(df) // WIN_SAMPLES
    good_segments = []
    bad_segments = []

    for i in range(n_segs):
        seg = df.iloc[i * WIN_SAMPLES : (i + 1) * WIN_SAMPLES]
        ecg = seg["SNUADC/ECG_II"].to_numpy(dtype=float)
        ppg = seg["SNUADC/PLETH"].to_numpy(dtype=float)
        rr = seg["Solar8000/VENT_RR"].to_numpy(dtype=float)

        is_good, stats_dict = evaluate_sqi(ecg, ppg, rr)
        seg_data = {
            "seg_idx": i,
            "ecg_raw": ecg,
            "ppg_raw": ppg,
            "rr": rr,
            "stats": stats_dict,
            "is_good": is_good,
        }

        if is_good:
            seg_data["ecg_filt"] = bandpass_transient_free(ecg)
            seg_data["ppg_filt"] = bandpass_transient_free(ppg)
            good_segments.append(seg_data)
        else:
            bad_segments.append(seg_data)

    return good_segments, bad_segments


def plot_before_after(passed_seg: dict, failed_seg: dict | None = None):
    """Plot 4 rows: Rows 0-2 = Before/After DSP; Row 3 = Passed vs Failed SQI segment comparison."""
    t = np.arange(WIN_SAMPLES) / FS_DS
    st_p = passed_seg["stats"]

    fig, axes = plt.subplots(4, 2, figsize=(13, 9), sharex=True)

    # Row 0: ECG (Before vs After) for Passed Segment
    axes[0, 0].plot(t, passed_seg["ecg_raw"], color="navy")
    axes[0, 0].set_title(f"Before DSP: Raw ECG (PTP={st_p['ecg_ptp']} mV, Kurt={st_p['ecg_kurt']})")
    axes[0, 0].set_ylabel("ECG II (mV)")

    axes[0, 1].plot(t, passed_seg["ecg_filt"], color="royalblue")
    axes[0, 1].set_title("After DSP: 0.5-10 Hz Bandpass ECG (Transient-Free)")

    # Row 1: PPG (Before vs After) for Passed Segment
    axes[1, 0].plot(t, passed_seg["ppg_raw"], color="crimson")
    axes[1, 0].set_title(f"Before DSP: Raw PPG (Pulse Peaks={st_p['ppg_peaks']}, IBI_CV={st_p['ibi_cv']})")
    axes[1, 0].set_ylabel("PPG (PLETH)")

    axes[1, 1].plot(t, passed_seg["ppg_filt"], color="indianred")
    axes[1, 1].set_title("After DSP: 0.5-10 Hz Bandpass PPG (Baseline Removed)")

    # Row 2: VENT_RR for Passed Segment
    axes[2, 0].plot(t, passed_seg["rr"], color="green")
    axes[2, 0].set_title(f"VENT_RR (Raw Ground Truth: {st_p['rr_mean']} rpm)")
    axes[2, 0].set_ylabel("VENT_RR")

    axes[2, 1].plot(t, passed_seg["rr"], color="limegreen")
    axes[2, 1].set_title(f"VENT_RR (Aligned Ground Truth: {st_p['rr_mean']} rpm)")

    # Row 3: SQI Comparison - Segment that Passed SQI vs Segment that Failed SQI
    axes[3, 0].plot(t, passed_seg["ecg_raw"], color="darkgreen", lw=1.1)
    axes[3, 0].set_title(f"SQI PASSED (Seg #{passed_seg['seg_idx']}): PTP={st_p['ecg_ptp']} mV, Kurt={st_p['ecg_kurt']}, HR={st_p['ppg_peaks']} bpm", color="darkgreen", fontweight="bold")
    axes[3, 0].set_ylabel("ECG II (mV)")
    axes[3, 0].set_xlabel("Time (s)")

    if failed_seg is not None:
        st_f = failed_seg["stats"]
        axes[3, 1].plot(t, failed_seg["ecg_raw"], color="firebrick", lw=1.1)
        axes[3, 1].set_title(f"SQI REJECTED (Seg #{failed_seg['seg_idx']}): {st_f['rejection_reason']}", color="firebrick", fontweight="bold", fontsize=9)
        axes[3, 1].set_xlabel("Time (s)")
    else:
        axes[3, 1].text(0.5, 0.5, "All segments passed SQI", ha="center", va="center")
        axes[3, 1].set_xlabel("Time (s)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.suptitle("DSP Pipeline & Calibrated Tight SQI Validation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_vital(1)
    good_segs, bad_segs = segment_and_process(df)
    print(f"Total 60s segments: {len(df) // WIN_SAMPLES} | Passed SQI: {len(good_segs)} | Failed SQI: {len(bad_segs)}")
    if good_segs:
        failed_sample = bad_segs[0] if bad_segs else None
        plot_before_after(good_segs[0], failed_sample)
