#!/usr/bin/env python3
"""
test_triago_external_data.py
Evaluation of Transfer Learning CalBPNet model on external TriaGo primary dataset:
Path: C:\\Users\\Adyty\\Documents\\Farid ITS\\TriaGo\\data_primer (Data1.csv - Data21.csv)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import torch
import torch.nn as nn

# Include local modules
WORKSPACE_ROOT = Path("C:/Users/Adyty/Documents/Farid ITS/Tugas Akhir")
sys.path.append(str(WORKSPACE_ROOT / "scratch/gemastik"))
sys.path.append(str(WORKSPACE_ROOT / "scratch/gemastik/Bp revisi/BPNet"))
sys.path.append(str(WORKSPACE_ROOT / "scratch/gemastik/remote_jobs/pulsedb_calbased"))

from checklist_eval import CalBPNet
from signal_processing import extract_7ch_segment, butter_bandpass_filter, evaluate_segment_sqi

# PulseDB Normalization Constants used during model training
PDB_SBP_MEAN = 118.2714
PDB_SBP_STD = 17.9963
PDB_DBP_MEAN = 64.2229
PDB_DBP_STD = 12.2722


def compute_bhs_grades(errors: np.ndarray) -> Dict[str, Any]:
    abs_err = np.abs(errors)
    p5 = float(np.mean(abs_err <= 5.0) * 100.0)
    p10 = float(np.mean(abs_err <= 10.0) * 100.0)
    p15 = float(np.mean(abs_err <= 15.0) * 100.0)
    mae = float(np.mean(abs_err))
    std = float(np.std(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_bias = float(np.mean(errors))

    if p5 >= 60.0 and p10 >= 85.0 and p15 >= 95.0:
        grade = "A"
    elif p5 >= 50.0 and p10 >= 75.0 and p15 >= 90.0:
        grade = "B"
    elif p5 >= 40.0 and p10 >= 65.0 and p15 >= 85.0:
        grade = "C"
    else:
        grade = "D"

    aami_pass = bool(abs(mean_bias) <= 5.0 and std <= 8.0)

    return {
        "mae": round(mae, 2),
        "std": round(std, 2),
        "rmse": round(rmse, 2),
        "bias": round(mean_bias, 2),
        "pct_le_5": round(p5, 2),
        "pct_le_10": round(p10, 2),
        "pct_le_15": round(p15, 2),
        "bhs_grade": grade,
        "aami_pass": aami_pass,
    }


def main():
    triago_dir = Path(r"C:\Users\Adyty\Documents\Farid ITS\TriaGo\data_primer")
    if not triago_dir.exists():
        print(f"Error: Directory not found: {triago_dir}")
        sys.exit(1)

    weights_path = WORKSPACE_ROOT / "scratch/gemastik/Bp revisi/model_best.pt"
    if not weights_path.exists():
        print(f"Error: Model weights not found: {weights_path}")
        sys.exit(1)

    # 1. Load Model
    model = CalBPNet()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    files = sorted(triago_dir.glob("Data*.csv"), key=lambda x: int("".join(c for c in x.stem if c.isdigit())))
    
    extracted_X = []
    extracted_s = []
    extracted_d = []
    extracted_subj = []
    extracted_seg_idx = []

    total_files = len(files)
    total_raw_segs = 0
    passed_sqi_segs = 0

    # 2. Extract and Preprocess 7-Channel Segments
    for f in files:
        subj_id = f.stem
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            
            raw_ecg = df["ECG"].to_numpy(dtype=float)
            raw_ppg = df["PPG_IR"].to_numpy(dtype=float)
            gt_s = float(df["SBP_Ground_Truth"].dropna().iloc[0])
            gt_d = float(df["DBP_Ground_Truth"].dropna().iloc[0])

            # Resample 400 Hz -> 125 Hz (up 5, down 16)
            ecg_125 = resample_poly(raw_ecg, 5, 16)
            ppg_125 = resample_poly(raw_ppg, 5, 16)

            # Invert PPG if photodiode polarity produces negative skewness
            ppg_zero = ppg_125 - np.mean(ppg_125)
            if np.mean(ppg_zero ** 3) < 0:
                ppg_125 = -ppg_125

            # Continuous Zero-Phase Butterworth Filtering (eliminates seam transients)
            ecg_filt = butter_bandpass_filter(ecg_125, 0.5, 40.0, fs=125)
            ppg_filt = butter_bandpass_filter(ppg_125, 0.5, 8.0, fs=125)

            n_segs = len(ecg_filt) // 1250
            subj_clean_count = 0

            for i in range(n_segs):
                total_raw_segs += 1
                seg_ecg = ecg_filt[i * 1250 : (i + 1) * 1250]
                seg_ppg = ppg_filt[i * 1250 : (i + 1) * 1250]

                is_clean, _ = evaluate_segment_sqi(seg_ecg, seg_ppg, fs=125)
                if is_clean:
                    passed_sqi_segs += 1
                    ch7 = extract_7ch_segment(seg_ecg, seg_ppg)
                    extracted_X.append(ch7)
                    extracted_s.append(gt_s)
                    extracted_d.append(gt_d)
                    extracted_subj.append(subj_id)
                    extracted_seg_idx.append(subj_clean_count)
                    subj_clean_count += 1

        except Exception as e:
            print(f"Warning: Failed processing {f.name}: {e}")

    X_all = np.array(extracted_X, dtype=np.float32)
    ys_all = np.array(extracted_s, dtype=np.float32)
    yd_all = np.array(extracted_d, dtype=np.float32)
    subjs = np.array(extracted_subj)
    seg_indices = np.array(extracted_seg_idx)

    n_samples = len(X_all)
    unique_subjs = np.unique(subjs)

    # 3. Model Inference (Zero-Shot)
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_all).float()
        ps_z, pd_z = model(x_tensor)
        pred_s_raw = (ps_z.squeeze(-1).numpy() * PDB_SBP_STD) + PDB_SBP_MEAN
        pred_d_raw = (pd_z.squeeze(-1).numpy() * PDB_DBP_STD) + PDB_DBP_MEAN

    # 4. Compute Uncalibrated Metrics
    err_s_uncal = pred_s_raw - ys_all
    err_d_uncal = pred_d_raw - yd_all
    stats_s_uncal = compute_bhs_grades(err_s_uncal)
    stats_d_uncal = compute_bhs_grades(err_d_uncal)

    # 5. Compute 1-Point Calibrated Metrics (Disjoint Evaluation on t >= 1)
    cal_pred_s = np.zeros_like(pred_s_raw)
    cal_pred_d = np.zeros_like(pred_d_raw)
    eval_mask = np.zeros(n_samples, dtype=bool)
    n_cal_subjs = 0

    for s in unique_subjs:
        idx = np.where(subjs == s)[0]
        if len(idx) > 1:
            anc_idx = idx[0]  # First chronological segment as anchor
            eval_idx = idx[1:] # Evaluate strictly on subsequent segments
            eval_mask[eval_idx] = True
            n_cal_subjs += 1

            off_s = ys_all[anc_idx] - pred_s_raw[anc_idx]
            off_d = yd_all[anc_idx] - pred_d_raw[anc_idx]

            cal_pred_s[eval_idx] = pred_s_raw[eval_idx] + off_s
            cal_pred_d[eval_idx] = pred_d_raw[eval_idx] + off_d

    n_cal_eval = int(np.sum(eval_mask))
    if n_cal_eval > 0:
        err_s_cal = cal_pred_s[eval_mask] - ys_all[eval_mask]
        err_d_cal = cal_pred_d[eval_mask] - yd_all[eval_mask]
        stats_s_cal = compute_bhs_grades(err_s_cal)
        stats_d_cal = compute_bhs_grades(err_d_cal)
    else:
        stats_s_cal = None
        stats_d_cal = None

    # 6. Save JSON Summary
    out_json = WORKSPACE_ROOT / "scratch/gemastik/triago_external_test_results.json"
    results = {
        "dataset_path": str(triago_dir),
        "total_subjects": total_files,
        "clean_subjects": len(unique_subjs),
        "total_segments_created": total_raw_segs,
        "clean_segments_sqi": n_samples,
        "uncalibrated": {"sbp": stats_s_uncal, "dbp": stats_d_uncal},
        "calibrated_1pt": {
            "eval_segments": n_cal_eval,
            "multi_segment_subjects": n_cal_subjs,
            "sbp": stats_s_cal,
            "dbp": stats_d_cal,
        },
    }
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # 7. Print Output (Suppressed <= 15 lines)
    print("=" * 65)
    print(f"TRIA GO EXTERNAL BENCHMARK: {n_samples} segs from {len(unique_subjs)}/{total_files} subjects")
    print(f"Uncalibrated SBP : MAE={stats_s_uncal['mae']} mmHg (STD {stats_s_uncal['std']}), Bias={stats_s_uncal['bias']:+.2f}, BHS: {stats_s_uncal['bhs_grade']}, AAMI: {'PASS' if stats_s_uncal['aami_pass'] else 'FAIL'}")
    print(f"Uncalibrated DBP : MAE={stats_d_uncal['mae']} mmHg (STD {stats_d_uncal['std']}), Bias={stats_d_uncal['bias']:+.2f}, BHS: {stats_d_uncal['bhs_grade']}, AAMI: {'PASS' if stats_d_uncal['aami_pass'] else 'FAIL'}")
    if stats_s_cal:
        print(f"Calibrated 1-Pt  : Evaluated on {n_cal_eval} disjoint segments ({n_cal_subjs} multi-segment subjects)")
        print(f"Calibrated SBP   : MAE={stats_s_cal['mae']} mmHg (STD {stats_s_cal['std']}), Bias={stats_s_cal['bias']:+.2f}, BHS: {stats_s_cal['bhs_grade']}, AAMI: {'PASS' if stats_s_cal['aami_pass'] else 'FAIL'}")
        print(f"Calibrated DBP   : MAE={stats_d_cal['mae']} mmHg (STD {stats_d_cal['std']}), Bias={stats_d_cal['bias']:+.2f}, BHS: {stats_d_cal['bhs_grade']}, AAMI: {'PASS' if stats_d_cal['aami_pass'] else 'FAIL'}")
    print(f"Saved results to: {out_json.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
