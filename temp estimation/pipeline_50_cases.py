"""
pipeline_50_cases.py - End-to-End Pipeline on 50 VitalDB Surgical Cases
Author: Alex (Lead Orchestrator)
Master Reference: scratch/temp estimation/PRD_VitalDB_Temp_Validation.md

Evaluates:
1. Scale up from 10 cases to 50 real VitalDB surgical cases.
2. Full Pipeline Execution:
   - Cases 1-10: Sourced from local validated Stage 3 pipeline.
   - Cases 11-60+: Streamed via vitaldb API (SNUADC/PLETH & Solar8000/BT).
   - Stage 1: Continuous valid core temp segments (33.0 - 45.0 C, >= 60s).
   - Stage 2: 125 Hz DSP, zero-phase Butterworth BPF (0.5-10 Hz) & LPF (<0.5 Hz), strict SQI gating, unclipped PI = AC/DC * 100%.
   - Stage 3: Dynamic Stolwijk-Hardy ODE (tau=180s, PI_eff = 0.10 * PI), MLX90614 NETD noise (0.05 C), 17-bit quantization (0.02 C).
   - Stage 4: Strict patient split (35 Train cases, 15 Test cases), TRF curve_fit optimization, unseen generalization.
3. Metrics: Pooled cohort & Macro-averaged patient metrics, per-case pass/fail audits.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from scipy.optimize import curve_fit
import concurrent.futures
import vitaldb

from stage3 import process_case_stage3

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_50_PARQUET = os.path.join(CURRENT_DIR, "cohort50_epochs.parquet")
SUMMARY_50_JSON = os.path.join(CURRENT_DIR, "cohort50_evaluation_summary.json")

TARGET_CASES_TOTAL = 50
TRAIN_RATIO = 0.70  # 35 Train cases, 15 Test cases

A_TRUE = 0.574
B_TRUE = 0.850
ALPHA_TRUE = 0.800
TAU_TRUE = 180.0
PI_SCALE = 0.10

def process_streamed_case(caseid: int, target_fs: float = 125.0):
    """
    Streams and processes a VitalDB case (caseid >= 11) from Stage 1 through Stage 3.
    """
    try:
        vals = vitaldb.load_case(caseid, ["SNUADC/PLETH", "Solar8000/BT"], interval=1.0 / target_fs)
        if vals is None or vals.shape[0] < target_fs * 180:
            return None
        ppg = vals[:, 0]
        bt = vals[:, 1]
    except Exception:
        return None

    # Physiological gating on core temp (at least 30 valid samples = 60s)
    bt_valid_mask = np.isfinite(bt) & (bt >= 33.0) & (bt <= 45.0)
    if np.sum(bt_valid_mask) < 30:
        return None

    valid_idx = np.where(bt_valid_mask)[0]
    bt_interp = np.interp(np.arange(len(bt)), valid_idx, bt[valid_idx])

    # Find contiguous segments of valid core temp (gaps <= 30s allowed)
    dt_gap_samples = int(30.0 * target_fs)
    diff_idx = np.diff(valid_idx)
    break_pts = np.where(diff_idx > dt_gap_samples)[0]

    seg_bounds = []
    start_i = valid_idx[0]
    for b in break_pts:
        end_i = valid_idx[b]
        if (end_i - start_i) >= target_fs * 60:
            seg_bounds.append((start_i, end_i))
        start_i = valid_idx[b + 1]
    if (valid_idx[-1] - start_i) >= target_fs * 60:
        seg_bounds.append((start_i, valid_idx[-1]))

    if not seg_bounds:
        return None

    # 4th-order zero-phase Butterworth filters
    nyq = 0.5 * target_fs
    sos_bp = scipy.signal.butter(4, [0.5 / nyq, 10.0 / nyq], btype="bandpass", output="sos")
    sos_lp = scipy.signal.butter(4, 0.5 / nyq, btype="lowpass", output="sos")

    win_len = int(10.0 * target_fs)
    epochs = []
    rng = np.random.default_rng(42 + caseid)

    t = np.arange(len(ppg)) / target_fs

    for seg_id, (st, en) in enumerate(seg_bounds):
        seg_ppg = ppg[st:en]
        seg_t = t[st:en]
        seg_bt = bt_interp[st:en]

        # Interpolate NaNs in PPG if any
        if np.isnan(seg_ppg).any():
            v_p_idx = np.where(np.isfinite(seg_ppg))[0]
            if len(v_p_idx) < target_fs * 60:
                continue
            seg_ppg = np.interp(np.arange(len(seg_ppg)), v_p_idx, seg_ppg[v_p_idx])

        # Filter
        try:
            p_bp = scipy.signal.sosfiltfilt(sos_bp, seg_ppg)
            p_lp = scipy.signal.sosfiltfilt(sos_lp, seg_ppg)
        except Exception:
            continue

        n_win = len(seg_ppg) // win_len
        if n_win < 3:
            continue

        seg_epochs = []
        for w_i in range(n_win):
            w_st = w_i * win_len
            w_en = w_st + win_len
            w_bp = p_bp[w_st:w_en]
            w_lp = p_lp[w_st:w_en]
            w_t = seg_t[w_st:w_en]
            t_center = float(w_t[len(w_t) // 2])
            core_t = float(seg_bt[w_st + len(w_t) // 2])

            dc = float(np.mean(w_lp))
            ac = float(np.percentile(w_bp, 95) - np.percentile(w_bp, 5))

            if dc <= 0.0 or ac <= 0.0:
                continue

            # SQI check
            sd = float(np.std(w_bp, ddof=1))
            sk = float(scipy.stats.skew(w_bp))
            kt = float(scipy.stats.kurtosis(w_bp, fisher=False))

            sqi_pass = (sk >= 0.10) and (2.0 <= kt <= 6.0) and (0.5 <= sd <= 15.0) and (ac <= 60.0)
            if not sqi_pass:
                continue

            pi_val = (ac / dc) * 100.0
            if pi_val <= 0.0 or pi_val > 150.0:
                continue

            seg_epochs.append({
                "caseid": caseid,
                "seg_id": seg_id,
                "window_idx": w_i,
                "t_center_sec": t_center,
                "t_center_min": round(t_center / 60.0, 2),
                "t_core": round(core_t, 2),
                "ac": round(ac, 3),
                "dc": round(dc, 3),
                "pi": round(pi_val, 2)
            })

        if not seg_epochs:
            continue

        # Stage 3 Simulation
        df_seg = pd.DataFrame(seg_epochs)
        t_arr = df_seg["t_center_sec"].values
        core_arr = df_seg["t_core"].values
        pi_arr = df_seg["pi"].values
        n_seg_w = len(df_seg)

        t_amb_arr = rng.normal(loc=21.0, scale=0.30, size=n_seg_w)
        pi_eff = np.maximum(pi_arr * PI_SCALE, 1e-4)
        g_cs = A_TRUE + B_TRUE * np.power(pi_eff, ALPHA_TRUE)
        t_eq_arr = (g_cs * core_arr + t_amb_arr) / (1.0 + g_cs)

        t_skin_arr = np.zeros(n_seg_w, dtype=np.float64)
        t_skin_arr[0] = t_eq_arr[0]
        for k in range(1, n_seg_w):
            dt_step = t_arr[k] - t_arr[k - 1]
            if dt_step <= 0.0 or dt_step > 60.0:
                t_skin_arr[k] = t_eq_arr[k]
            else:
                decay = np.exp(-dt_step / TAU_TRUE)
                t_skin_arr[k] = t_skin_arr[k - 1] * decay + t_eq_arr[k] * (1.0 - decay)

        sensor_noise = rng.normal(loc=0.0, scale=0.05, size=n_seg_w)
        t_meas_arr = np.round((t_skin_arr + sensor_noise) / 0.02) * 0.02

        df_seg["t_amb"] = np.round(t_amb_arr, 3)
        df_seg["t_eq"] = np.round(t_eq_arr, 3)
        df_seg["t_skin"] = np.round(t_skin_arr, 3)
        df_seg["t_skin_meas"] = np.round(t_meas_arr, 2)

        epochs.append(df_seg)

    if not epochs:
        return None

    df_case = pd.concat(epochs, ignore_index=True)
    if len(df_case) < 30:  # Require at least 5 minutes of valid data
        return None

    return df_case

def collect_50_cases_dataset():
    """Builds or loads cohort of exactly 50 valid surgical cases."""
    if os.path.exists(CACHE_50_PARQUET):
        print(f"Loading cached 50-case dataset from: {os.path.basename(CACHE_50_PARQUET)}")
        df_cohort = pd.read_parquet(CACHE_50_PARQUET)
        return df_cohort

    print("Building 50-case cohort. Scanning VitalDB cases...")
    collected_dfs = []

    # First load local cases 1 to 10
    print("Loading validated local Cases 1-10...")
    for cid in range(1, 11):
        res = process_case_stage3(cid)
        if res.get("stats", {}).get("status") == "OK":
            df = res["df_sim"]
            collected_dfs.append(df)
            print(f"  -> Local Case {cid:2d}: ACCEPTED ({len(df):>4} epochs, {len(df)*10/60:5.1f}m)")

    # Stream subsequent cases from VitalDB
    tested_caseid = 11
    max_scan_id = 90
    print(f"\nCollected {len(collected_dfs)}/50 cases. Streaming candidates from api.vitaldb.net (6 workers)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        while len(collected_dfs) < TARGET_CASES_TOTAL and tested_caseid < max_scan_id:
            batch_size = min(12, TARGET_CASES_TOTAL - len(collected_dfs) + 4)
            batch_ids = list(range(tested_caseid, tested_caseid + batch_size))
            tested_caseid += batch_size

            future_to_id = {executor.submit(process_streamed_case, cid): cid for cid in batch_ids}
            for future in concurrent.futures.as_completed(future_to_id):
                cid = future_to_id[future]
                try:
                    df = future.result()
                    if df is not None:
                        collected_dfs.append(df)
                        print(f"  -> Streamed Case {cid:3d}: ACCEPTED ({len(df):>4} epochs, {len(df)*10/60:5.1f}m) | Total: {len(collected_dfs)}/{TARGET_CASES_TOTAL}")
                        if len(collected_dfs) >= TARGET_CASES_TOTAL:
                            break
                    else:
                        print(f"  -> Streamed Case {cid:3d}: Skipped (missing tracks or insufficient valid epochs)")
                except Exception as e:
                    print(f"  -> Streamed Case {cid:3d}: Error ({e})")

    df_cohort = pd.concat(collected_dfs[:TARGET_CASES_TOTAL], ignore_index=True)
    df_cohort.to_parquet(CACHE_50_PARQUET, index=False)
    print(f"\nSuccessfully cached {len(df_cohort):,} total epochs across {len(df_cohort['caseid'].unique())} cases to: {os.path.basename(CACHE_50_PARQUET)}")
    return df_cohort

def run_50_case_evaluation():
    t_start = time.time()
    print("=" * 115)
    print("SCALING VALIDATION: 50 SURGICAL CASES FROM VITALDB")
    print("Patient-Level Split: 70% Train (35 Patients) vs 30% Test (15 Patients)")
    print("=" * 115)

    df_cohort = collect_50_cases_dataset()
    unique_cases = sorted(list(df_cohort["caseid"].unique()))
    n_cases = len(unique_cases)

    # Deterministic patient-level split (70% Train, 30% Test)
    rng_split = np.random.default_rng(2026)
    shuffled_cases = unique_cases.copy()
    rng_split.shuffle(shuffled_cases)

    n_train = int(n_cases * TRAIN_RATIO)
    train_cases = sorted(shuffled_cases[:n_train])
    test_cases = sorted(shuffled_cases[n_train:])

    df_train = df_cohort[df_cohort["caseid"].isin(train_cases)].copy().reset_index(drop=True)
    df_test = df_cohort[df_cohort["caseid"].isin(test_cases)].copy().reset_index(drop=True)

    print(f"\n[PATIENT SPLIT AUDIT]")
    print(f"  Total Cases: {n_cases} | Total Epochs: {len(df_cohort):,} ({len(df_cohort)*10/3600:.2f} hours)")
    print(f"  Train Cohort: {len(train_cases)} Patients ({len(df_train):,} epochs, {len(df_train)*10/3600:.2f} hours)")
    print(f"  Test Cohort : {len(test_cases)} Patients ({len(df_test):,} epochs, {len(df_test)*10/3600:.2f} hours)")
    print(f"  Zero Leakage Check: {len(set(train_cases).intersection(set(test_cases))) == 0}")

    def model_func(X, a, b, alpha):
        ts, ta, p = X
        p_clipped = np.maximum(p, 0.05)
        p_eff = p_clipped * PI_SCALE
        denom = np.maximum(a + b * np.power(p_eff, alpha), 0.10)
        t_raw = ts + (ts - ta) / denom
        return np.clip(t_raw, 30.0, 45.0)

    # 1. Optimize on 35 Train Patients
    print("\n[FITTING PARAMETERS ON 35 TRAIN CASES VIA TRF CURVE_FIT]")
    X_tr = (df_train["t_skin_meas"].values, df_train["t_amb"].values, df_train["pi"].values)
    y_tr = df_train["t_core"].values

    popt, pcov = curve_fit(
        model_func, X_tr, y_tr,
        p0=[0.574, 0.850, 0.800],
        bounds=([0.1, 0.05, 0.1], [5.0, 10.0, 3.0]),
        method="trf"
    )
    a_fit, b_fit, alpha_fit = popt
    std_errs = np.sqrt(np.diag(pcov))

    print(f"  Fitted a     : {a_fit:.5f} +- {std_errs[0]:.5f}")
    print(f"  Fitted b     : {b_fit:.5f} +- {std_errs[1]:.5f}")
    print(f"  Fitted alpha : {alpha_fit:.5f} +- {std_errs[2]:.5f}")

    # 2. Evaluate on 15 Unseen Test Patients
    print("\n[EVALUATION ON 15 UNSEEN TEST CASES]")
    X_te = (df_test["t_skin_meas"].values, df_test["t_amb"].values, df_test["pi"].values)
    y_te = df_test["t_core"].values
    y_pred_te = model_func(X_te, a_fit, b_fit, alpha_fit)

    def calc_metrics(y_true, y_pred, y_base):
        res = y_pred - y_true
        mae = float(np.mean(np.abs(res)))
        rmse = float(np.sqrt(np.mean(res**2)))
        bias = float(np.mean(res))
        sd = float(np.std(res, ddof=1))
        upper_loa = float(bias + 1.96 * sd)
        lower_loa = float(bias - 1.96 * sd)
        acc_050 = float(np.mean(np.abs(res) <= 0.50) * 100.0)
        acc_035 = float(np.mean(np.abs(res) <= 0.35) * 100.0)
        acc_020 = float(np.mean(np.abs(res) <= 0.20) * 100.0)
        base_mae = float(np.mean(np.abs(y_base - y_true)))
        err_red = float((base_mae - mae) / base_mae * 100.0) if base_mae > 0 else 0.0
        ss_res = np.sum(res**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        r_val = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "bias": round(bias, 4),
            "lower_loa": round(lower_loa, 4),
            "upper_loa": round(upper_loa, 4),
            "acc_050": round(acc_050, 2),
            "acc_035": round(acc_035, 2),
            "acc_020": round(acc_020, 2),
            "base_mae": round(base_mae, 4),
            "err_red": round(err_red, 2),
            "r2": round(r2, 4),
            "pearson_r": round(r_val, 4)
        }

    pooled_metrics = calc_metrics(y_te, y_pred_te, df_test["t_skin_meas"].values)

    per_case_records = []
    cases_passed_count = 0

    for cid in test_cases:
        sub = df_test[df_test["caseid"] == cid]
        X_sub = (sub["t_skin_meas"].values, sub["t_amb"].values, sub["pi"].values)
        y_sub = sub["t_core"].values
        y_p_sub = model_func(X_sub, a_fit, b_fit, alpha_fit)
        cm = calc_metrics(y_sub, y_p_sub, sub["t_skin_meas"].values)

        bias_ok = abs(cm["bias"]) <= 0.20
        loa_ok = (cm["lower_loa"] >= -0.90) and (cm["upper_loa"] <= 0.90)
        r2_ok = cm["r2"] > 0.0
        mae_ok = cm["mae"] <= 0.35

        case_passed = bias_ok and loa_ok and r2_ok and mae_ok
        if case_passed:
            cases_passed_count += 1

        per_case_records.append({
            "Case": cid,
            "Epochs": len(sub),
            "Dur (m)": round(len(sub) * 10 / 60, 1),
            "MAE (°C)": cm["mae"],
            "RMSE (°C)": cm["rmse"],
            "Bias (°C)": cm["bias"],
            "LoA Lower": cm["lower_loa"],
            "LoA Upper": cm["upper_loa"],
            "Acc <=0.50C": f"{cm['acc_050']}%",
            "Acc <=0.35C": f"{cm['acc_035']}%",
            "R2": cm["r2"],
            "Verdict": "PASS" if case_passed else "FAIL",
            "metrics": cm
        })

    df_test_cases = pd.DataFrame(per_case_records)

    macro_mae = float(np.mean([r["metrics"]["mae"] for r in per_case_records]))
    macro_rmse = float(np.mean([r["metrics"]["rmse"] for r in per_case_records]))
    macro_bias = float(np.mean([r["metrics"]["bias"] for r in per_case_records]))
    macro_acc050 = float(np.mean([r["metrics"]["acc_050"] for r in per_case_records]))
    macro_acc035 = float(np.mean([r["metrics"]["acc_035"] for r in per_case_records]))
    macro_err_red = float(np.mean([r["metrics"]["err_red"] for r in per_case_records]))

    macro_summary = {
        "macro_mae": round(macro_mae, 4),
        "macro_rmse": round(macro_rmse, 4),
        "macro_bias": round(macro_bias, 4),
        "macro_acc_050": round(macro_acc050, 2),
        "macro_acc_035": round(macro_acc035, 2),
        "macro_err_red": round(macro_err_red, 2)
    }

    print("\n" + "=" * 115)
    print("50-CASE EVALUATION RESULTS: 15 UNSEEN TEST PATIENTS")
    print("=" * 115)
    print(df_test_cases[["Case", "Epochs", "Dur (m)", "MAE (°C)", "RMSE (°C)", "Bias (°C)", "LoA Lower", "LoA Upper", "Acc <=0.50C", "R2", "Verdict"]].to_string(index=False))
    print("-" * 115)
    print(f"POOLED TEST COHORT ({len(df_test):,} Epochs | {len(df_test)*10/3600:.2f} Hours across 15 Patients):")
    print(f"  * MAE: {pooled_metrics['mae']:.4f} °C (Clinical Gold <= 0.35 C: {'PASS' if pooled_metrics['mae'] <= 0.35 else 'FAIL'})")
    print(f"  * RMSE: {pooled_metrics['rmse']:.4f} °C | Bias: {pooled_metrics['bias']:+.4f} °C (Target <= +-0.20 C)")
    print(f"  * 95% LoA: [{pooled_metrics['lower_loa']:+.4f}, {pooled_metrics['upper_loa']:+.4f}] °C (Target [-0.90, +0.90] C)")
    print(f"  * Accuracy <= 0.50 °C: {pooled_metrics['acc_050']}% | Acc <= 0.35 °C: {pooled_metrics['acc_035']}%")
    print(f"  * Error Reduction: {pooled_metrics['err_red']}% (Baseline Skin Error: {pooled_metrics['base_mae']:.4f} °C)")
    print(f"  * Pearson r: {pooled_metrics['pearson_r']:.4f} | R2: {pooled_metrics['r2']:.4f}")
    print("-" * 115)
    print(f"MACRO-AVERAGED METRICS (Equal Weight per Patient, N = 15):")
    print(f"  * Macro MAE: {macro_summary['macro_mae']:.4f} °C | Macro RMSE: {macro_summary['macro_rmse']:.4f} °C")
    print(f"  * Macro Bias: {macro_summary['macro_bias']:+.4f} °C | Macro Acc <= 0.50 °C: {macro_summary['macro_acc_050']}%")
    print(f"  * Macro Error Drop: {macro_summary['macro_err_red']}%")
    print("-" * 115)
    print(f"CASE-LEVEL PASS RATE: {cases_passed_count}/{len(test_cases)} ({cases_passed_count/len(test_cases)*100:.1f}%)")
    print("=" * 115)

    summary_payload = {
        "study": "50-Case Scaling Feasibility Study (VitalDB Semi-Synthetic)",
        "cohort_size": n_cases,
        "n_train_cases": len(train_cases),
        "n_test_cases": len(test_cases),
        "train_patients": train_cases,
        "test_patients": test_cases,
        "fitted_parameters": {
            "a": round(float(a_fit), 5),
            "b": round(float(b_fit), 5),
            "alpha": round(float(alpha_fit), 5),
            "a_se": round(float(std_errs[0]), 5),
            "b_se": round(float(std_errs[1]), 5),
            "alpha_se": round(float(std_errs[2]), 5)
        },
        "pooled_test_metrics": pooled_metrics,
        "macro_metrics": macro_summary,
        "case_level_pass_rate_pct": round(cases_passed_count / len(test_cases) * 100.0, 1),
        "per_case_test_metrics": per_case_records,
        "runtime_sec": round(time.time() - t_start, 2)
    }

    def json_cleaner(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float64)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    with open(SUMMARY_50_JSON, "w") as f:
        json.dump(summary_payload, f, indent=2, default=json_cleaner)
    print(f"Saved 50-case evaluation summary JSON to: {SUMMARY_50_JSON}")
    print(f"Completed in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    run_50_case_evaluation()
