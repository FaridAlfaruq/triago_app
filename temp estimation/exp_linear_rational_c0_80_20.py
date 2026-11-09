"""
exp_linear_rational_c0_80_20.py - Linear Rational Bioheat with Metabolic Offset (c0) on 100 VitalDB Cases (80/20 Split)
Model:
    T_est = T_skin + (T_skin - T_amb) / (a + b * PI_eff) + c0
where:
    PI_eff = 0.10 * PI
    a  = Minimum conductive resistance
    b  = Perfusion sensitivity
    c0 = Basal metabolic core-to-skin offset (°C)
Split:
    80 Train Patients (55,816 epochs, 155.0h)
    20 Unseen Test Patients (19,372 epochs, 53.8h)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_100_PARQUET = os.path.join(CURRENT_DIR, "cohort100_epochs.parquet")
OUTPUT_SUMMARY_JSON = os.path.join(CURRENT_DIR, "linear_rational_c0_80_20_summary.json")

def run_evaluation():
    t_start = time.time()
    print("=" * 115)
    print("LINEAR RATIONAL BIOHEAT WITH METABOLIC OFFSET (c0)")
    print("Dataset: 100 VitalDB Surgical Cases | Patient Split: 80% Train (80) / 20% Test (20)")
    print("=" * 115)

    df = pd.read_parquet(CACHE_100_PARQUET)
    all_cases = sorted(list(df["caseid"].unique()))
    assert len(all_cases) == 100, f"Expected 100 cases, found {len(all_cases)}"

    # Effective perfusion scaling (0.10 * PI)
    df["pi_eff"] = np.clip(0.10 * df["pi"], 0.005, 10.0)
    df["delta_T"] = df["t_skin_meas"] - df["t_amb"]

    # Deterministic 80/20 Patient Split (Random Seed 2026)
    rng = np.random.default_rng(2026)
    shuffled = all_cases.copy()
    rng.shuffle(shuffled)
    n_train = 80
    train_cases = sorted(shuffled[:n_train])
    test_cases = sorted(shuffled[n_train:])

    df_train = df[df["caseid"].isin(train_cases)].copy().reset_index(drop=True)
    df_test = df[df["caseid"].isin(test_cases)].copy().reset_index(drop=True)

    print(f"\n[PATIENT SPLIT AUDIT]")
    print(f"  Total Cohort: {len(all_cases)} Patients ({len(df):,} epochs, {len(df)*10/3600:.1f} hours)")
    print(f"  Train Cohort: {len(train_cases)} Patients ({len(df_train):,} epochs, {len(df_train)*10/3600:.1f} hours)")
    print(f"  Test Cohort : {len(test_cases)} Patients ({len(df_test):,} epochs, {len(df_test)*10/3600:.1f} hours)")
    print(f"  Zero Leakage: {len(set(train_cases).intersection(set(test_cases))) == 0}")

    # Model Definition
    def model_func(X, a, b, c0):
        ts, ta, peff = X
        g_cs = np.maximum(a + b * peff, 0.10)
        t_raw = ts + (ts - ta) / g_cs + c0
        return np.clip(t_raw, 30.0, 45.0)

    # 1. Fitting on 80 Train Patients
    print("\n[FITTING PARAMETERS ON 80 TRAIN PATIENTS VIA TRF CURVE_FIT]")
    X_tr = (df_train["t_skin_meas"].values, df_train["t_amb"].values, df_train["pi_eff"].values)
    y_tr = df_train["t_core"].values

    popt, pcov = curve_fit(
        model_func, X_tr, y_tr,
        p0=[0.85, 1.05, 1.20],
        bounds=([0.1, 0.05, -5.0], [5.0, 10.0, 5.0]),
        method="trf"
    )
    se = np.sqrt(np.diag(pcov))
    a_fit, b_fit, c0_fit = popt

    print(f"  Fitted a  (Conductive Resistance) : {a_fit:.5f} +- {se[0]:.5f}")
    print(f"  Fitted b  (Perfusion Sensitivity) : {b_fit:.5f} +- {se[1]:.5f}")
    print(f"  Fitted c0 (Metabolic Offset)      : {c0_fit:+.5f} +- {se[2]:.5f} °C")

    # Evaluation Metric Calculator
    def calc_metrics(y_true, y_pred, cids, df_sub):
        res = y_pred - y_true
        mae = float(np.mean(np.abs(res)))
        rmse = float(np.sqrt(np.mean(res**2)))
        bias = float(np.mean(res))
        sd = float(np.std(res, ddof=1))
        lower_loa = float(bias - 1.96 * sd)
        upper_loa = float(bias + 1.96 * sd)
        acc_35 = float(np.mean(np.abs(res) <= 0.35) * 100.0)
        acc_20 = float(np.mean(np.abs(res) <= 0.20) * 100.0)
        acc_50 = float(np.mean(np.abs(res) <= 0.50) * 100.0)

        per_case = []
        for cid in sorted(cids):
            mask = (df_sub["caseid"] == cid)
            sub_res = res[mask]
            sub_y = y_true[mask]
            c_mae = float(np.mean(np.abs(sub_res)))
            c_bias = float(np.mean(sub_res))
            c_sd = float(np.std(sub_res, ddof=1)) if len(sub_res) > 1 else 0.0
            c_acc35 = float(np.mean(np.abs(sub_res) <= 0.35) * 100.0)
            c_pass = bool(c_mae <= 0.35)
            per_case.append({
                "caseid": int(cid),
                "n_epochs": int(np.sum(mask)),
                "duration_min": round(int(np.sum(mask)) * 10.0 / 60.0, 1),
                "t_core_mean": round(float(np.mean(sub_y)), 2),
                "t_core_min": round(float(np.min(sub_y)), 2),
                "t_core_max": round(float(np.max(sub_y)), 2),
                "mae": round(c_mae, 4),
                "bias": round(c_bias, 4),
                "sd": round(c_sd, 4),
                "acc_35": round(c_acc35, 1),
                "pass_mae35": c_pass
            })

        pass_count = sum(1 for c in per_case if c["pass_mae35"])
        pass_rate = round(pass_count / len(per_case) * 100.0, 1)
        macro_mae = round(float(np.mean([c["mae"] for c in per_case])), 4)
        macro_bias = round(float(np.mean([c["bias"] for c in per_case])), 4)

        return {
            "pooled_mae": round(mae, 4),
            "macro_mae": macro_mae,
            "macro_bias": macro_bias,
            "rmse": round(rmse, 4),
            "bias": round(bias, 4),
            "sd": round(sd, 4),
            "lower_loa": round(lower_loa, 4),
            "upper_loa": round(upper_loa, 4),
            "acc_50": round(acc_50, 2),
            "acc_35": round(acc_35, 2),
            "acc_20": round(acc_20, 2),
            "cases_passed": pass_count,
            "cases_total": len(per_case),
            "pass_rate_pct": pass_rate,
            "per_case": per_case
        }

    # 2. Evaluate on 20 Unseen Test Patients
    X_te = (df_test["t_skin_meas"].values, df_test["t_amb"].values, df_test["pi_eff"].values)
    y_te = df_test["t_core"].values
    y_pred_te = model_func(X_te, a_fit, b_fit, c0_fit)
    met_test = calc_metrics(y_te, y_pred_te, test_cases, df_test)

    # 3. Evaluate on 80 Train Patients
    y_pred_tr = model_func(X_tr, a_fit, b_fit, c0_fit)
    met_train = calc_metrics(y_tr, y_pred_tr, train_cases, df_train)

    # 4. Evaluate on All 100 Patients
    X_all = (df["t_skin_meas"].values, df["t_amb"].values, df["pi_eff"].values)
    y_all = df["t_core"].values
    y_pred_all = model_func(X_all, a_fit, b_fit, c0_fit)
    met_all = calc_metrics(y_all, y_pred_all, all_cases, df)

    print("\n" + "=" * 115)
    print(f"{'OVERALL METRICS EVALUATION: LINEAR RATIONAL BIOHEAT (+ c0)':^115}")
    print("=" * 115)
    print(f"{'Cohort':<28} | {'Pooled MAE':<11} | {'Macro MAE':<10} | {'Bias':<8} | {'SD':<7} | {'95% LoA':<17} | {'Pass Rate':<12} | {'Epoch Acc <=0.35C'}")
    print("-" * 115)

    for cohort_name, m in [
        ("Train (80 Patients)", met_train),
        ("Test  (20 Patients Unseen)", met_test),
        ("Full  (100 Patients Cohort)", met_all)
    ]:
        loa_str = f"[{m['lower_loa']:+.2f}, {m['upper_loa']:+.2f}]"
        pass_str = f"{m['cases_passed']}/{m['cases_total']} ({m['pass_rate_pct']}%)"
        print(f"{cohort_name:<28} | {m['pooled_mae']:<7.4f} °C | {m['macro_mae']:<7.4f} °C | {m['bias']:<+7.4f} | {m['sd']:<6.4f} | {loa_str:<17} | {pass_str:<12} | {m['acc_35']:.1f}%")

    # 5. Detailed Per-Patient Breakdown on 20 Test Patients
    print("\n" + "=" * 115)
    print(f"{'DETAILED BREAKDOWN: 20 UNSEEN TEST PATIENTS (19,372 Epochs, 53.8h)':^115}")
    print("=" * 115)
    print(f"{'Case ID':<8} | {'Epochs':<7} | {'Duration':<9} | {'T_core Mean':<12} | {'MAE (°C)':<9} | {'Bias (°C)':<9} | {'Acc <=0.35C':<12} | {'Clinical Status'}")
    print("-" * 115)

    for c in met_test["per_case"]:
        cid = c["caseid"]
        n_ep = c["n_epochs"]
        dur = f"{c['duration_min']}m"
        tc_m = f"{c['t_core_mean']:.2f} °C"
        c_mae = f"{c['mae']:.4f} °C"
        c_bias = f"{c['bias']:+.4f} °C"
        c_acc = f"{c['acc_35']:.1f}%"
        status = "PASSED (MAE <= 0.35)" if c["pass_mae35"] else "FAILED (MAE > 0.35)"
        print(f"Case {cid:<3d}  | {n_ep:<7d} | {dur:<9} | {tc_m:<12} | {c_mae:<9} | {c_bias:<9} | {c_acc:<12} | {status}")

    # Save summary JSON
    summary_data = {
        "model_name": "Linear Rational Bioheat with Metabolic Offset (c0)",
        "formula": "T_est = T_skin + (T_skin - T_amb) / (a + b * PI_eff) + c0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cohort_summary": {
            "total_patients": 100,
            "train_patients_count": 80,
            "test_patients_count": 20,
            "total_epochs": len(df),
            "train_epochs": len(df_train),
            "test_epochs": len(df_test)
        },
        "fitted_parameters": {
            "a": round(float(a_fit), 5),
            "b": round(float(b_fit), 5),
            "c0": round(float(c0_fit), 5),
            "std_errors": [round(float(s), 5) for s in se]
        },
        "train_cases": [int(c) for c in train_cases],
        "test_cases": [int(c) for c in test_cases],
        "test_metrics_20_patients": met_test,
        "train_metrics_80_patients": met_train,
        "full_cohort_metrics_100_patients": met_all,
        "runtime_sec": round(time.time() - t_start, 2)
    }

    with open(OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[OUTPUT] Saved summary report to: {OUTPUT_SUMMARY_JSON}")
    print(f"[RUNTIME] Completed evaluation in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    run_evaluation()
