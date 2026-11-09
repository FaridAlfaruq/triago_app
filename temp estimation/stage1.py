"""
stage1.py - Stage 1: Raw Cohort Verification, Data Quality Audit, and Signal Visualization
Author: Lead Orchestrator Pipeline
Task: Verify 10 VitalDB cases (1 to 10), extract statistics, identify invalid/missing data,
      separate unmolested 'raw_data' from physiologically gated 'proc_data', and display
      interactive signal plots using matplotlib.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vitaldb

VITAL_DIR = os.path.join(os.path.dirname(__file__), "vital_files")

def load_case_data(caseid, min_temp=33.0, max_temp=45.0, min_seg_dur_sec=60.0, max_gap_sec=30.0):
    """
    Loads raw VitalDB signals and splits into two distinct variables:
    1. raw_data: Original untouched recordings (full time series, NaNs included).
    2. proc_data: Data ready for downstream processing, strictly restricted to valid
       core temperature (min_temp <= BT <= max_temp) and synchronized PPG segments.

    Parameters:
        caseid (int): VitalDB case identifier (1 to 10).
        min_temp (float): Physiological lower threshold in deg C (default 33.0).
        max_temp (float): Physiological upper threshold in deg C (default 45.0).
        min_seg_dur_sec (float): Minimum duration for a valid epoch in seconds (default 60s).
        max_gap_sec (float): Max allowed gap in valid BT before splitting segments (default 30s).

    Returns:
        tuple: (raw_data, proc_data)
    """
    filepath = os.path.join(VITAL_DIR, f"{caseid}.vital")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Case file not found: {filepath}")

    vf = vitaldb.VitalFile(filepath)
    available_tracks = vf.get_track_names()
    if not ("SNUADC/PLETH" in available_tracks and "Solar8000/BT" in available_tracks):
        raise ValueError(f"Case {caseid} missing required tracks. Found: {available_tracks}")

    # 1. Raw PPG (500 Hz)
    df_ppg = vf.to_pandas(["SNUADC/PLETH"], interval=1/500.0)
    ppg_raw = df_ppg["SNUADC/PLETH"].values
    t_ppg_raw = np.arange(len(ppg_raw)) / 500.0

    # 2. Raw BT (Solar8000/BT at 2.0s native interval)
    df_bt = vf.to_pandas(["Solar8000/BT"], interval=2.0)
    bt_raw = df_bt["Solar8000/BT"].values
    t_bt_raw = np.arange(len(bt_raw)) * 2.0

    raw_data = {
        "caseid": caseid,
        "ppg": ppg_raw,
        "t_ppg": t_ppg_raw,
        "fs_ppg": 500.0,
        "bt": bt_raw,
        "t_bt": t_bt_raw,
        "fs_bt": 0.5,
        "duration_sec": float(t_ppg_raw[-1]) if len(t_ppg_raw) > 0 else 0.0,
        "duration_min": float(t_ppg_raw[-1] / 60.0) if len(t_ppg_raw) > 0 else 0.0
    }

    # 3. Gating logic for proc_data
    valid_bt_mask = (~np.isnan(bt_raw)) & (bt_raw >= min_temp) & (bt_raw <= max_temp)
    valid_indices = np.where(valid_bt_mask)[0]

    if len(valid_indices) == 0:
        proc_data = {
            "caseid": caseid,
            "bt": np.array([], dtype=float),
            "t_bt": np.array([], dtype=float),
            "ppg": np.array([], dtype=float),
            "t_ppg": np.array([], dtype=float),
            "segments": [],
            "fs_ppg": 500.0,
            "fs_bt": 0.5,
            "valid_bt_mask": valid_bt_mask,
            "total_valid_sec": 0.0,
            "total_valid_min": 0.0,
            "total_valid_hours": 0.0,
            "n_segments": 0
        }
        return raw_data, proc_data

    # Identify contiguous segments
    t_valid = t_bt_raw[valid_indices]
    splits = np.where(np.diff(t_valid) > max_gap_sec)[0]
    seg_starts = np.insert(splits + 1, 0, 0)
    seg_ends = np.append(splits, len(valid_indices) - 1)

    segments = []
    for s_idx, (st_i, en_i) in enumerate(zip(seg_starts, seg_ends), start=1):
        idx_slice = valid_indices[st_i:en_i + 1]
        t_start = t_bt_raw[idx_slice[0]]
        t_end = t_bt_raw[idx_slice[-1]]
        dur_sec = t_end - t_start

        if dur_sec < min_seg_dur_sec:
            continue

        # Extract synchronized PPG slice
        ppg_mask = (t_ppg_raw >= t_start) & (t_ppg_raw <= t_end)
        seg_ppg = ppg_raw[ppg_mask]
        seg_t_ppg = t_ppg_raw[ppg_mask]
        seg_bt = bt_raw[idx_slice]
        seg_t_bt = t_bt_raw[idx_slice]

        segments.append({
            "seg_id": s_idx,
            "t_start_sec": float(t_start),
            "t_end_sec": float(t_end),
            "duration_sec": float(dur_sec),
            "duration_min": float(dur_sec / 60.0),
            "ppg": seg_ppg,
            "t_ppg": seg_t_ppg,
            "bt": seg_bt,
            "t_bt": seg_t_bt,
            "ppg_nan_pct": float(np.isnan(seg_ppg).sum() / len(seg_ppg) * 100.0) if len(seg_ppg) > 0 else 0.0
        })

    # Assemble concatenated proc_data arrays
    if len(segments) > 0:
        ppg_proc = np.concatenate([s["ppg"] for s in segments])
        t_ppg_proc = np.concatenate([s["t_ppg"] for s in segments])
        bt_proc = np.concatenate([s["bt"] for s in segments])
        t_bt_proc = np.concatenate([s["t_bt"] for s in segments])
        total_valid_sec = float(sum(s["duration_sec"] for s in segments))
    else:
        ppg_proc = np.array([], dtype=float)
        t_ppg_proc = np.array([], dtype=float)
        bt_proc = np.array([], dtype=float)
        t_bt_proc = np.array([], dtype=float)
        total_valid_sec = 0.0

    proc_data = {
        "caseid": caseid,
        "bt": bt_proc,
        "t_bt": t_bt_proc,
        "ppg": ppg_proc,
        "t_ppg": t_ppg_proc,
        "segments": segments,
        "fs_ppg": 500.0,
        "fs_bt": 0.5,
        "valid_bt_mask": valid_bt_mask,
        "total_valid_sec": total_valid_sec,
        "total_valid_min": total_valid_sec / 60.0,
        "total_valid_hours": total_valid_sec / 3600.0,
        "n_segments": len(segments)
    }

    return raw_data, proc_data

def audit_case(caseid):
    """
    Audits a single case file, extracting quality and physiological statistics
    from both raw_data and proc_data.
    """
    try:
        raw_data, proc_data = load_case_data(caseid)
    except FileNotFoundError:
        return {"caseid": caseid, "status": "FILE_NOT_FOUND"}
    except ValueError as ve:
        return {"caseid": caseid, "status": "MISSING_MANDATORY_TRACKS", "error": str(ve)}

    filepath = os.path.join(VITAL_DIR, f"{caseid}.vital")
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

    ppg_raw = raw_data["ppg"]
    n_ppg_total = len(ppg_raw)
    n_ppg_nan = int(np.isnan(ppg_raw).sum())
    pct_ppg_nan = (n_ppg_nan / n_ppg_total) * 100.0 if n_ppg_total > 0 else 100.0

    ppg_valid = ppg_raw[~np.isnan(ppg_raw)]
    if len(ppg_valid) > 0:
        ppg_min = float(np.min(ppg_valid))
        ppg_max = float(np.max(ppg_valid))
        ppg_mean = float(np.mean(ppg_valid))
        ppg_std = float(np.std(ppg_valid))
    else:
        ppg_min = ppg_max = ppg_mean = ppg_std = 0.0

    bt_raw = raw_data["bt"]
    n_bt_total = len(bt_raw)
    n_bt_nan = int(np.isnan(bt_raw).sum())
    pct_bt_nan = (n_bt_nan / n_bt_total) * 100.0 if n_bt_total > 0 else 100.0

    bt_proc = proc_data["bt"]
    n_bt_proc = len(bt_proc)

    if n_bt_proc > 0:
        bt_min = float(np.min(bt_proc))
        bt_max = float(np.max(bt_proc))
        bt_mean = float(np.mean(bt_proc))
        bt_median = float(np.median(bt_proc))
        bt_std = float(np.std(bt_proc))
    else:
        bt_min = bt_max = bt_mean = bt_median = bt_std = 0.0

    n_hypo_extreme = int(np.sum((~np.isnan(bt_raw)) & (bt_raw < 33.0)))
    n_hyper_extreme = int(np.sum((~np.isnan(bt_raw)) & (bt_raw > 45.0)))
    valid_recorded_bt = n_bt_total - n_bt_nan
    pct_physio_valid = (n_bt_proc / valid_recorded_bt) * 100.0 if valid_recorded_bt > 0 else 0.0

    if proc_data["total_valid_hours"] >= 0.5 and pct_ppg_nan < 20.0:
        verdict = "USABLE_VALID"
    elif proc_data["total_valid_hours"] > 0.15 and pct_ppg_nan < 40.0:
        verdict = "USABLE_LIMITED"
    else:
        verdict = "INSUFFICIENT_DATA"

    return {
        "caseid": caseid,
        "file_size_mb": round(file_size_mb, 2),
        "status": "OK",
        "verdict": verdict,
        "n_segments": proc_data["n_segments"],
        "raw_stats": {
            "duration_min": round(raw_data["duration_min"], 2),
            "ppg_total_samples": n_ppg_total,
            "ppg_nan_pct": round(pct_ppg_nan, 2),
            "ppg_min": round(ppg_min, 2),
            "ppg_max": round(ppg_max, 2),
            "ppg_mean": round(ppg_mean, 2),
            "ppg_std": round(ppg_std, 2),
            "bt_total_samples": n_bt_total,
            "bt_nan_pct": round(pct_bt_nan, 2),
            "bt_out_of_bounds_low (<33C)": n_hypo_extreme,
            "bt_out_of_bounds_high (>45C)": n_hyper_extreme
        },
        "proc_stats": {
            "valid_hours": round(proc_data["total_valid_hours"], 2),
            "valid_duration_min": round(proc_data["total_valid_min"], 2),
            "ppg_proc_samples": len(proc_data["ppg"]),
            "bt_proc_samples": n_bt_proc,
            "bt_min_c": round(bt_min, 2),
            "bt_max_c": round(bt_max, 2),
            "bt_mean_c": round(bt_mean, 2),
            "bt_median_c": round(bt_median, 2),
            "bt_std_c": round(bt_std, 2),
            "physio_valid_pct": round(pct_physio_valid, 2)
        }
    }

def visualize_case(caseid=1, closeup_dur_sec=10.0):
    """
    Displays an interactive matplotlib figure showing:
    1. Full raw PPG vs valid proc_data PPG epochs.
    2. 10-second close-up of clean PPG within proc_data.
    3. Core body temperature comparing raw recordings (with transients) vs proc_data (valid physiological range).
    NOTE: Only displays using plt.show(), does not save or download figure.
    """
    print(f"\nLoading Case {caseid} signals (separating raw_data and proc_data)...")
    raw_data, proc_data = load_case_data(caseid)

    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title(f"Stage 1 Signal Audit - VitalDB Case {caseid}")

    # Subplot 1: Raw vs Proc PPG
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(raw_data["t_ppg"] / 60.0, raw_data["ppg"], color="#b0bec5", linewidth=0.5, rasterized=True, label="raw_data: Unfiltered Full Recording")
    
    # Overlay valid proc_data segments
    for idx, seg in enumerate(proc_data["segments"]):
        lbl = "proc_data: Valid T_core Gated Epochs" if idx == 0 else None
        ax1.plot(seg["t_ppg"] / 60.0, seg["ppg"], color="#1f77b4", linewidth=0.6, rasterized=True, label=lbl)

    # 10s close-up epoch inside the first proc_data segment
    if len(proc_data["segments"]) > 0:
        first_seg = proc_data["segments"][0]
        closeup_t_start = first_seg["t_start_sec"] + min(300.0, first_seg["duration_sec"] / 3.0)
        ax1.axvspan(closeup_t_start / 60.0, (closeup_t_start + closeup_dur_sec) / 60.0, color="#ff7f0e", alpha=0.45, label="10s Close-Up Window")
    else:
        closeup_t_start = 0.0

    ax1.set_title(f"Case {caseid}: Raw vs Processed Photoplethysmogram (SNUADC/PLETH @ 500 Hz)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax1.set_xlabel("Time (minutes)", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    # Subplot 2: 10s Close-Up of PPG
    ax2 = fig.add_subplot(3, 1, 2)
    idx_start = int(closeup_t_start * 500.0)
    idx_end = idx_start + int(closeup_dur_sec * 500.0)
    if idx_end <= len(raw_data["ppg"]):
        t_closeup = np.arange(idx_end - idx_start) / 500.0
        ppg_closeup = raw_data["ppg"][idx_start:idx_end]
        ax2.plot(t_closeup, ppg_closeup, color="#ff7f0e", linewidth=1.2)
        ax2.set_title(f"Case {caseid}: 10-Second Close-Up from proc_data (at t = {closeup_t_start/60.0:.1f} min)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Amplitude", fontsize=9)
    ax2.set_xlabel("Time (seconds within window)", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Core Body Temperature
    ax3 = fig.add_subplot(3, 1, 3)
    bt_raw = raw_data["bt"]
    t_bt_raw = raw_data["t_bt"] / 60.0
    invalid_mask = (~np.isnan(bt_raw)) & ((bt_raw < 33.0) | (bt_raw > 45.0))

    if np.any(invalid_mask):
        ax3.plot(t_bt_raw[invalid_mask], bt_raw[invalid_mask], "x", color="#d62728", markersize=4.0, alpha=0.7, label="raw_data: Room-Air / Probe Transient (< 33.0 °C)")

    ax3.plot(proc_data["t_bt"] / 60.0, proc_data["bt"], "o-", color="#2ca02c", markersize=2.5, linewidth=0.8, label="proc_data: Valid Core Temp (33.0 - 45.0 °C)")
    ax3.axhline(33.0, color="#d62728", linestyle=":", linewidth=1.0, label="Physiological Cutoff (33.0 °C)")
    ax3.set_title(f"Case {caseid}: Esophageal Core Temperature (raw_data vs proc_data)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Temperature (°C)", fontsize=9)
    ax3.set_xlabel("Time (minutes)", fontsize=9)
    ax3.set_ylim([max(0.0, np.nanmin(bt_raw) - 2.0) if len(bt_raw[~np.isnan(bt_raw)]) > 0 else 0, 42.0])
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    print("Displaying figure via matplotlib (plt.show()). Close plot window to continue.")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Stage 1: VitalDB Cohort Verification, Audit & Signal Visualization")
    parser.add_argument("--case", type=int, default=1, help="Case ID to visualize (1 to 10, default: 1)")
    parser.add_argument("--audit-only", action="store_true", help="Run audit summary table only, do not display plot")
    args = parser.parse_args()

    print("=" * 85)
    print("STAGE 1: VITALDB 10-CASE COHORT VERIFICATION AND RAW VS PROC QUALITY AUDIT")
    print("=" * 85)
    
    results = []
    case_ids = list(range(1, 11))
    
    print(f"\nScanning directory: {VITAL_DIR}")
    for cid in case_ids:
        print(f"Auditing Case {cid:2d}/10 ... ", end="", flush=True)
        res = audit_case(cid)
        results.append(res)
        if res.get("status") == "OK":
            p = res["proc_stats"]
            r = res["raw_stats"]
            print(f"OK | Verdict: {res['verdict']:<14} | Segs: {res['n_segments']} | Raw Dur: {r['duration_min']:5.1f}m -> Valid: {p['valid_duration_min']:5.1f}m ({p['valid_hours']:4.1f}h) | BT: {p['bt_min_c']:4.1f}-{p['bt_max_c']:4.1f} C")
        else:
            print(f"ERROR: {res.get('status')}")

    # Save complete JSON audit
    out_json = os.path.join(os.path.dirname(__file__), "stage1_audit_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed audit JSON to: {out_json}")

    # Build Summary Table
    rows = []
    total_valid_hours = 0.0
    for res in results:
        if res.get("status") == "OK":
            p = res["proc_stats"]
            r = res["raw_stats"]
            total_valid_hours += p["valid_hours"]
            rows.append({
                "Case": res["caseid"],
                "Size (MB)": res["file_size_mb"],
                "Raw Dur (m)": r["duration_min"],
                "Raw PPG NaNs (%)": r["ppg_nan_pct"],
                "Valid Dur (m)": p["valid_duration_min"],
                "Valid BT (h)": p["valid_hours"],
                "Valid Segs": res["n_segments"],
                "BT Min (°C)": p["bt_min_c"],
                "BT Max (°C)": p["bt_max_c"],
                "BT Mean (°C)": p["bt_mean_c"],
                "Physio Valid (%)": p["physio_valid_pct"],
                "Verdict": res["verdict"]
            })
    
    df_summary = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("COHORT AUDIT SUMMARY TABLE: RAW_DATA VS PROC_DATA (10 CASES)")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("-" * 90)
    print(f"Total Physiologically Valid Core Temperature: {total_valid_hours:.2f} hours across {len(rows)} cases.")
    print("=" * 90)

    # Visualization
    if not args.audit_only:
        visualize_case(caseid=args.case)

if __name__ == "__main__":
    main()
