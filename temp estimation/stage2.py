"""
stage2.py - Stage 2: Digital Signal Processing (DSP), Bandpass Filtering, 10s Segmentation & Perfusion Index
Author: Lead Orchestrator Pipeline
Task:
1. Downsample PPG from 500 Hz to 125 Hz with edge-reflection padding (eliminates initial sample discontinuity)
2. Unified detrending and filtering:
   - AC track: 4th-order Butterworth bandpass (0.5 - 10.0 Hz) with 5.0s warmup lead-in
   - DC track: 4th-order Butterworth lowpass (< 0.5 Hz) for optical baseline
3. Strict Signal Quality Index (SQI) rejecting variance & amplitude spikes (catches Case 5 Window 1501)
4. Compute Perfusion Index strictly as:
   PI (%) = (AC / DC) * 100% where DC is mean of lowpass baseline (<0.5 Hz)
   - Zero artificial DC shifting (if DC <= 0, PI is invalid / NaN)
   - Zero immediate clipping: preserves original unclipped PI
5. Clean 2-column comparative visualization (Before vs After) with legends at top-right
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats

# Import load_case_data from Stage 1
from stage1 import load_case_data, VITAL_DIR

# -------------------------------------------------------------------------
# 1. Downsampling Helper (500 Hz -> 125 Hz with Edge Reflection Padding)
# -------------------------------------------------------------------------
def downsample_signal(x, t, fs_in=500.0, fs_out=125.0):
    """
    Downsamples signal x and time array t from fs_in to fs_out using polyphase decimation.
    Fixes initial edge artifact:
    1. Subtracts initial DC offset (x[0]) to prevent step jump against FIR zero-padding.
    2. Applies reflection boundary padding.
    3. Resamples, strips padding, and restores offset.
    """
    decim_factor = int(round(fs_in / fs_out))
    offset = float(x[0]) if len(x) > 0 else 0.0
    x_zero = x - offset

    pad_len = decim_factor * 16  # 64 samples boundary padding
    x_padded = np.pad(x_zero, pad_len, mode="reflect")

    x_ds_padded = scipy.signal.resample_poly(x_padded, up=1, down=decim_factor)
    ds_pad_len = pad_len // decim_factor

    x_ds = x_ds_padded[ds_pad_len:-ds_pad_len] + offset
    t_ds = t[::decim_factor][:len(x_ds)]
    return x_ds, t_ds

# -------------------------------------------------------------------------
# 2. AC Bandpass (0.5 - 10.0 Hz) and DC Lowpass (< 0.5 Hz) Filtering
# -------------------------------------------------------------------------
def bandpass_ppg(x, fs=125.0, lowcut=0.5, highcut=10.0, order=4, warmup_sec=5.0):
    """
    Zero-phase 4th-order Butterworth bandpass filter (0.5 to 10.0 Hz) for pulsatile AC extraction.
    Startup transient eliminated via warmup_sec reflection lead-in buffer.
    """
    sos = scipy.signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    if warmup_sec > 0 and len(x) > int(warmup_sec * fs):
        pad_pts = int(round(warmup_sec * fs))
        offset = float(x[0])
        x_zero = x - offset
        x_pad = np.pad(x_zero, pad_pts, mode="reflect") + offset
        filtered_pad = scipy.signal.sosfiltfilt(sos, x_pad)
        return filtered_pad[pad_pts:-pad_pts]
    else:
        return scipy.signal.sosfiltfilt(sos, x)

def lowpass_ppg(x, fs=125.0, cutoff=0.5, order=4, warmup_sec=5.0):
    """
    Zero-phase 4th-order Butterworth lowpass filter (< 0.5 Hz) for optical DC baseline extraction.
    Captures non-pulsatile background absorption without cardiac modulation.
    """
    sos = scipy.signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    if warmup_sec > 0 and len(x) > int(warmup_sec * fs):
        pad_pts = int(round(warmup_sec * fs))
        offset = float(x[0])
        x_zero = x - offset
        x_pad = np.pad(x_zero, pad_pts, mode="reflect") + offset
        filtered_pad = scipy.signal.sosfiltfilt(sos, x_pad)
        return filtered_pad[pad_pts:-pad_pts]
    else:
        return scipy.signal.sosfiltfilt(sos, x)

# -------------------------------------------------------------------------
# 3. Strict Signal Quality Index (SQI)
# -------------------------------------------------------------------------
def evaluate_sqi(w_bp, dc, min_skew=0.1, kurt_range=(2.0, 6.0), std_range=(0.5, 15.0), max_ac=60.0):
    """
    Strict SQI evaluation for 10s PPG window:
    - Skewness >= 0.1: True systolic steep rise & diastolic decay (catches symmetric motion noise).
    - Kurtosis in [2.0, 6.0]: Physiological pulse peakedness.
    - Standard Deviation in [0.5, 15.0]: Discards flatline noise (<0.5) and violent motion (>15.0).
    - AC <= 60.0: Discards unphysiological amplitude surges (e.g. Case 5 Window 1501 with AC=100.32).
    - DC > 0: Valid optical transmission baseline.
    """
    sk = float(scipy.stats.skew(w_bp))
    kt = float(scipy.stats.kurtosis(w_bp, fisher=False))
    sd = float(np.std(w_bp))
    ac = float(np.percentile(w_bp, 95) - np.percentile(w_bp, 5))

    passed = bool(
        (sk >= min_skew) and
        (kurt_range[0] <= kt <= kurt_range[1]) and
        (std_range[0] <= sd <= std_range[1]) and
        (ac <= max_ac) and
        (dc > 0.0)
    )
    return passed, sk, kt, sd, ac

# -------------------------------------------------------------------------
# 4. Strict Perfusion Index (PI) Calculation
# -------------------------------------------------------------------------
def compute_pi_window(w_bp, w_dc):
    """
    Computes Perfusion Index strictly per definition:
    PI (%) = (AC / DC) * 100%
    - AC: 95th percentile minus 5th percentile of bandpass filtered PPG (0.5 - 10.0 Hz).
    - DC: Mean of low-pass filtered baseline (< 0.5 Hz).
    - Zero artificial DC shifts: if DC <= 0, PI is NaN.
    - Zero immediate clipping: preserves true raw PI.
    """
    dc = float(np.mean(w_dc))
    ac = float(np.percentile(w_bp, 95) - np.percentile(w_bp, 5))
    
    if dc > 0.0:
        pi_original = (ac / dc) * 100.0
    else:
        pi_original = np.nan
        
    return ac, dc, pi_original

# -------------------------------------------------------------------------
# 5. Complete DSP Pipeline on a Case
# -------------------------------------------------------------------------
def process_case_dsp(caseid, window_sec=10.0, lowcut=0.5, highcut=10.0, warmup_sec=5.0):
    """
    Executes Stage 2 DSP pipeline for a single case:
    1. Loads proc_data valid segments from Stage 1.
    2. Uses real lead-in samples (warmup_sec) from raw_data to eliminate filter startup transients.
    3. Downsamples to 125 Hz with edge reflection padding.
    4. Separates signal into AC track (0.5-10.0 Hz bandpass) and DC track (< 0.5 Hz lowpass).
    5. Evaluates 10s windows with strict SQI and computes true PI = (AC / DC) * 100%.
    """
    raw_data, proc_data = load_case_data(caseid)
    if proc_data["n_segments"] == 0:
        return {"caseid": caseid, "status": "NO_VALID_SEGMENTS"}

    fs_raw = raw_data["fs_ppg"]  # 500 Hz
    fs_target = 125.0
    win_len = int(window_sec * fs_target)

    t_ppg_full = raw_data["t_ppg"]
    ppg_full = raw_data["ppg"]

    all_epochs = []
    passed_example = None
    failed_example = None
    first_seg_diag = None

    for s_idx, seg in enumerate(proc_data["segments"], start=1):
        t_start = seg["t_start_sec"]
        t_end = seg["t_end_sec"]
        seg_dur = t_end - t_start
        if seg_dur < window_sec:
            continue

        # Extract with lead-in & lead-out warmup samples from raw_data
        t_w_start = max(0.0, t_start - warmup_sec)
        t_w_end = min(float(t_ppg_full[-1]), t_end + warmup_sec)

        mask_warm = (t_ppg_full >= t_w_start) & (t_ppg_full <= t_w_end)
        p_raw_warm = ppg_full[mask_warm]
        t_raw_warm = t_ppg_full[mask_warm]

        actual_lead_sec = t_start - t_w_start
        if actual_lead_sec < warmup_sec:
            needed_pad_raw = int(round((warmup_sec - actual_lead_sec) * fs_raw))
            offset_raw = float(p_raw_warm[0])
            p_raw_warm = np.pad(p_raw_warm - offset_raw, (needed_pad_raw, 0), mode="reflect") + offset_raw
            t_lead_pad = np.linspace(t_start - warmup_sec, t_w_start, needed_pad_raw, endpoint=False)
            t_raw_warm = np.concatenate([t_lead_pad, t_raw_warm])

        # Step 1: Downsample 500 -> 125 Hz with edge reflection
        p_ds_warm, t_ds_warm = downsample_signal(p_raw_warm, t_raw_warm, fs_in=fs_raw, fs_out=fs_target)

        # Step 2: Separate AC (0.5-10 Hz bandpass) and DC (< 0.5 Hz lowpass)
        sos_bp = scipy.signal.butter(4, [lowcut, highcut], btype="bandpass", fs=fs_target, output="sos")
        p_bp_warm = scipy.signal.sosfiltfilt(sos_bp, p_ds_warm)

        sos_lp = scipy.signal.butter(4, lowcut, btype="lowpass", fs=fs_target, output="sos")
        p_lp_warm = scipy.signal.sosfiltfilt(sos_lp, p_ds_warm)

        # Step 3: Trim lead-in and lead-out buffers back to [t_start, t_end]
        mask_seg = (t_ds_warm >= t_start) & (t_ds_warm <= t_end)
        p_ds = p_ds_warm[mask_seg]
        t_ds = t_ds_warm[mask_seg]
        p_bp = p_bp_warm[mask_seg]
        p_lp = p_lp_warm[mask_seg]

        # Store diagnostic snippet from first segment
        if first_seg_diag is None:
            dur_pts = min(3750, len(p_ds))
            p_bp_unbuf = scipy.signal.sosfiltfilt(sos_bp, p_ds[:dur_pts])
            first_seg_diag = {
                "t_raw": seg["t_ppg"][:dur_pts * 4],
                "p_raw": seg["ppg"][:dur_pts * 4],
                "t_ds": t_ds[:dur_pts],
                "p_ds": p_ds[:dur_pts],
                "p_bp": p_bp[:dur_pts],
                "p_lp": p_lp[:dur_pts],
                "p_bp_unbuf": p_bp_unbuf
            }

        # Step 4: 10s Windowing & Strict SQI Gating
        n_windows = len(p_ds) // win_len
        for i in range(n_windows):
            st = i * win_len
            en = st + win_len
            w_raw = p_ds[st:en]
            w_bp = p_bp[st:en]
            w_lp = p_lp[st:en]
            t_win = t_ds[st:en]
            t_center = float(t_win[len(t_win) // 2])

            ac, dc, pi_val = compute_pi_window(w_bp, w_lp)
            passed, sk, kt, sd, _ = evaluate_sqi(w_bp, dc)

            # Core body temperature synchronized at window center
            t_bt = seg["t_bt"]
            bt_vals = seg["bt"]
            bt_diffs = np.abs(t_bt - t_center)
            min_bt_idx = np.argmin(bt_diffs)
            t_core_gt = float(bt_vals[min_bt_idx]) if bt_diffs[min_bt_idx] <= 15.0 else np.nan

            epoch_data = {
                "seg_id": s_idx,
                "window_idx": i,
                "t_center_sec": round(t_center, 2),
                "t_center_min": round(t_center / 60.0, 2),
                "sqi_passed": passed,
                "skew": round(sk, 3),
                "kurt": round(kt, 3),
                "std": round(sd, 3),
                "ac": round(ac, 3),
                "dc": round(dc, 3),
                "pi": round(pi_val, 2) if passed and not np.isnan(pi_val) else np.nan,
                "t_core": round(t_core_gt, 2)
            }

            if passed:
                if passed_example is None:
                    passed_example = {
                        "t_rel": t_win - t_win[0],
                        "w_raw": w_raw,
                        "w_bp": w_bp,
                        "w_lp": w_lp,
                        "skew": sk,
                        "kurt": kt,
                        "ac": ac,
                        "dc": dc,
                        "pi": pi_val
                    }
            else:
                if failed_example is None and (sk < 0.1 or kt < 2.0 or kt > 6.0 or sd > 15.0 or ac > 60.0):
                    failed_example = {
                        "t_rel": t_win - t_win[0],
                        "w_raw": w_raw,
                        "w_bp": w_bp,
                        "w_lp": w_lp,
                        "skew": sk,
                        "kurt": kt,
                        "std": sd,
                        "ac": ac,
                        "dc": dc,
                        "pi": pi_val
                    }

            all_epochs.append(epoch_data)

    df_epochs = pd.DataFrame(all_epochs)
    n_total = len(df_epochs)
    n_passed = int(df_epochs["sqi_passed"].sum())
    n_rejected = n_total - n_passed
    pass_pct = (n_passed / n_total * 100.0) if n_total > 0 else 0.0

    passed_df = df_epochs[df_epochs["sqi_passed"]]
    pi_vals = passed_df["pi"].dropna().values
    pi_mean = float(np.mean(pi_vals)) if len(pi_vals) > 0 else 0.0
    pi_min = float(np.min(pi_vals)) if len(pi_vals) > 0 else 0.0
    pi_max = float(np.max(pi_vals)) if len(pi_vals) > 0 else 0.0
    ac_mean = float(passed_df["ac"].mean()) if len(passed_df) > 0 else 0.0
    dc_mean = float(passed_df["dc"].mean()) if len(passed_df) > 0 else 0.0

    return {
        "caseid": caseid,
        "status": "OK",
        "n_total_epochs": n_total,
        "n_passed_epochs": n_passed,
        "n_rejected_epochs": n_rejected,
        "pass_pct": round(pass_pct, 2),
        "valid_duration_min": round(n_passed * 10.0 / 60.0, 2),
        "ac_mean": round(ac_mean, 2),
        "dc_mean": round(dc_mean, 2),
        "pi_stats": {
            "mean": round(pi_mean, 2),
            "min": round(pi_min, 2),
            "max": round(pi_max, 2)
        },
        "df_epochs": df_epochs,
        "diag": first_seg_diag,
        "passed_example": passed_example,
        "failed_example": failed_example
    }

# -------------------------------------------------------------------------
# 6. Clean 2-Column Comparative Visualization (Before vs After)
# -------------------------------------------------------------------------
def visualize_dsp(caseid=1):
    """
    Renders clean 6-panel (3 rows x 2 columns) comparative figure:
    - Row 1: Downsampling (Before: 500 Hz vs After: 125 Hz)
    - Row 2: AC & DC Separation (Before: Raw Signal vs After: DC Baseline <0.5Hz & AC Bandpass 0.5-10Hz)
    - Row 3: Quality Gating (Passed 10s Segment vs Rejected 10s Segment)
    All legends placed at top-right (loc='upper right').
    Only displays via plt.show() - does not save to disk.
    """
    print(f"\nProcessing DSP visualization for Case {caseid}...")
    res = process_case_dsp(caseid, lowcut=0.5, highcut=10.0)
    if res.get("status") != "OK":
        print(f"Error processing Case {caseid}: {res.get('status')}")
        return

    diag = res["diag"]
    passed_ex = res["passed_example"]
    failed_ex = res["failed_example"]

    # 5s window for downsampling close-up
    n_raw_5s = 2500
    n_ds_5s = 625
    t_5s_raw = diag["t_raw"][:n_raw_5s] - diag["t_raw"][0]
    p_5s_raw = diag["p_raw"][:n_raw_5s]
    t_5s_ds = diag["t_ds"][:n_ds_5s] - diag["t_ds"][0]
    p_5s_ds = diag["p_ds"][:n_ds_5s]

    # 10s window to show AC & DC separation and startup transient
    n_10s = min(1250, len(diag["p_bp"]))
    t_10s = diag["t_ds"][:n_10s] - diag["t_ds"][0]
    p_10s_raw = diag["p_ds"][:n_10s]
    p_10s_bp = diag["p_bp"][:n_10s]
    p_10s_lp = diag["p_lp"][:n_10s]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9.5))
    fig.canvas.manager.set_window_title(f"Stage 2 DSP - VitalDB Case {caseid}")

    # =========================================================================
    # ROW 1: Downsampling (500 Hz -> 125 Hz)
    # =========================================================================
    ax1 = axes[0, 0]
    ax1.plot(t_5s_raw, p_5s_raw, color="#7f7f7f", linewidth=0.8, alpha=0.7, label="Waveform (500 Hz)")
    ax1.plot(t_5s_raw[::25], p_5s_raw[::25], "o", color="#1f77b4", markersize=3, label="Sample Points (500 Hz)")
    ax1.set_title("Before: Raw PPG (500 Hz)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = axes[0, 1]
    ax2.plot(t_5s_ds, p_5s_ds, color="#1f77b4", linewidth=1.2, label="Waveform (125 Hz)")
    ax2.plot(t_5s_ds[::6], p_5s_ds[::6], "s", color="#ff7f0e", markersize=3.5, label="Sample Points (125 Hz)")
    ax2.set_title("After: Downsampled PPG (125 Hz)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8)

    # =========================================================================
    # ROW 2: AC & DC Component Separation
    # =========================================================================
    ax3 = axes[1, 0]
    ax3.plot(t_10s, p_10s_raw, color="#1f77b4", linewidth=1.0, label="Downsampled Raw PPG")
    ax3.plot(t_10s, p_10s_lp, color="#d62728", linestyle="--", linewidth=1.5, label="DC Baseline (< 0.5 Hz)")
    ax3.set_title("Before: Raw PPG with Lowpass DC Baseline", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax3.set_xlabel("Time (s)", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=8)

    ax4 = axes[1, 1]
    ax4.plot(t_10s, p_10s_bp, color="#2ca02c", linewidth=1.2, label="AC Pulse (0.5 - 10.0 Hz)")
    ax4.axhline(0.0, color="#7f7f7f", linestyle=":", linewidth=0.8, label="Zero Line")
    ax4.set_title("After: Isolated Pulsatile AC Component", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Filtered Amp (a.u.)", fontsize=9)
    ax4.set_xlabel("Time (s)", fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", fontsize=8)

    # =========================================================================
    # ROW 3: 10s Window Quality Gating (Passed vs Rejected)
    # =========================================================================
    ax5 = axes[2, 0]
    if passed_ex is not None:
        ax5.plot(passed_ex["t_rel"], passed_ex["w_bp"], color="#2ca02c", linewidth=1.2, label="Filtered AC Pulse")
        ax5.axhline(np.percentile(passed_ex["w_bp"], 95), color="#ff7f0e", linestyle="--", linewidth=0.9, label=f"AC 95%: {np.percentile(passed_ex['w_bp'], 95):.1f}")
        ax5.axhline(np.percentile(passed_ex["w_bp"], 5), color="#ff7f0e", linestyle=":", linewidth=0.9, label=f"AC 5%: {np.percentile(passed_ex['w_bp'], 5):.1f}")
        ax5.set_title(f"Passed 10s Window (AC={passed_ex['ac']:.1f}, DC={passed_ex['dc']:.1f} | PI={passed_ex['pi']:.1f}%)", fontsize=10, fontweight="bold", color="#1b5e20")
    else:
        ax5.text(0.5, 0.5, "No Passed Segment", ha="center", va="center")
    ax5.set_ylabel("Filtered Amp", fontsize=9)
    ax5.set_xlabel("Time (s)", fontsize=9)
    ax5.grid(True, linestyle="--", alpha=0.4)
    ax5.legend(loc="upper right", fontsize=8)

    ax6 = axes[2, 1]
    if failed_ex is not None:
        ax6.plot(failed_ex["t_rel"], failed_ex["w_bp"], color="#d62728", linewidth=1.2, label="Corrupted / Anomaly Wave")
        ax6.set_title(f"Rejected 10s Window (SQI Discard: S={failed_ex['skew']:.2f}, K={failed_ex['kurt']:.1f}, Std={failed_ex['std']:.1f})", fontsize=10, fontweight="bold", color="#b71c1c")
    else:
        ax6.text(0.5, 0.5, "100% Passed (Zero Corrupted Segments)", ha="center", va="center")
    ax6.set_ylabel("Filtered Amp", fontsize=9)
    ax6.set_xlabel("Time (s)", fontsize=9)
    ax6.grid(True, linestyle="--", alpha=0.4)
    ax6.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    print("Displaying clean 2-column comparative figure (plt.show()). Close plot window to continue.")
    plt.show()

# -------------------------------------------------------------------------
# 7. Main CLI Function
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 2: Digital Signal Processing, Bandpass Filtering & Perfusion Index")
    parser.add_argument("--case", type=int, default=1, help="Case ID to visualize (1 to 10, default: 1)")
    parser.add_argument("--audit-only", action="store_true", help="Run DSP audit summary table only, do not display plot")
    parser.add_argument("--highcut", type=float, default=10.0, help="High-cutoff frequency in Hz (default: 10.0)")
    args = parser.parse_args()

    print("=" * 110)
    print(f"STAGE 2: DIGITAL SIGNAL PROCESSING (PI = AC / DC * 100% | DC < 0.5 Hz, AC 0.5 - {args.highcut} Hz)")
    print("=" * 110)

    results = []
    case_ids = list(range(1, 11))

    for cid in case_ids:
        print(f"Processing DSP for Case {cid:2d}/10 ... ", end="", flush=True)
        res = process_case_dsp(cid, lowcut=0.5, highcut=args.highcut)
        if res.get("status") == "OK":
            p = res["pi_stats"]
            results.append({
                "caseid": cid,
                "total_epochs": res["n_total_epochs"],
                "passed_epochs": res["n_passed_epochs"],
                "rejected_epochs": res["n_rejected_epochs"],
                "pass_pct": res["pass_pct"],
                "valid_dur_min": res["valid_duration_min"],
                "ac_mean": res["ac_mean"],
                "dc_mean": res["dc_mean"],
                "pi_mean": p["mean"],
                "pi_min": p["min"],
                "pi_max": p["max"],
                "status": "OK"
            })
            print(f"OK | Epochs: {res['n_total_epochs']:>4} | Passed: {res['n_passed_epochs']:>4} ({res['pass_pct']:5.1f}%) | AC: {res['ac_mean']:4.1f} | DC: {res['dc_mean']:4.1f} | PI Mean: {p['mean']:5.1f}% (Range: {p['min']:4.1f}-{p['max']:5.1f}%)")
        else:
            print(f"ERROR: {res.get('status')}")

    # Save summary JSON
    out_json = os.path.join(os.path.dirname(__file__), "stage2_dsp_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed DSP audit JSON to: {out_json}")

    # Summary Table with unclipped PI = AC/DC * 100%
    df_summary = pd.DataFrame(results)
    df_table = df_summary[["caseid", "total_epochs", "passed_epochs", "rejected_epochs", "pass_pct", "valid_dur_min", "ac_mean", "dc_mean", "pi_mean", "pi_min", "pi_max"]]
    df_table.columns = ["Case", "Total 10s", "Pass 10s", "Reject 10s", "Pass (%)", "Valid (m)", "AC Mean", "DC Mean", "PI Mean (%)", "PI Min (%)", "PI Max (%)"]

    total_valid_epochs = df_table["Pass 10s"].sum()
    total_valid_hours = (total_valid_epochs * 10.0) / 3600.0

    print("\n" + "=" * 110)
    print(f"STAGE 2 DSP COHORT AUDIT TABLE (10 CASES - STRICT UNCLIPPED PI = AC/DC * 100%)")
    print("=" * 110)
    print(df_table.to_string(index=False))
    print("-" * 110)
    print(f"Total High-Quality Passed 10s Epochs: {total_valid_epochs:,} ({total_valid_hours:.2f} hours)")
    print("=" * 110)

    # Visualization
    if not args.audit_only:
        visualize_dsp(caseid=args.case)

if __name__ == "__main__":
    main()
