"""
exp_basic_vs_extended_80_20.py - Head-to-Head: Basic Bioheat vs Extended Bioheat (80/20 Split)
Evaluates:
1. 100 VitalDB surgical cases (cohort100_epochs.parquet).
2. Strict Patient-Level 80/20 Split: 80 Train Patients, 20 Unseen Test Patients (zero leakage).
3. Models:
   - Basic Bioheat: T_est = T_skin + (T_skin - T_amb) / (a + b * PI_eff^alpha)
   - Extended Bioheat (Raw dT/dt): T_est = Basic + c * (dT_skin/dt)
   - Extended Bioheat (Filtered 60s dT/dt): T_est = Basic + c * (dT_skin_60s/dt)
4. Comprehensive clinical metrics: Pooled MAE, Macro MAE, Bias, SD, 95% LoA, Epoch Acc <= 0.35C, Patient Pass Rate.
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
OUTPUT_SUMMARY_JSON = os.path.join(CURRENT_DIR, "basic_vs_extended_80_20_summary.json")

def run_experiment():
    t_start = time.time()
    print("=" * 115)
    print("HEAD-TO-HEAD COMPARISON: BASIC BIOHEAT vs. EXTENDED BIOHEAT")
    print("Dataset: 100 VitalDB Surgical Cases | Patient Split: 80% Train (80) / 20% Test (20)")
    print("=" * 115)

    df = pd.read_parquet(CACHE_100_PARQUET)
    all_cases = sorted(list(df["caseid"].unique()))
    assert len(all_cases) == 100, f"Expected 100 cases, found {len(all_cases)}"

    # Feature Engineering (Causal strictly per case & segment)
    df["delta_T"] = df["t_skin_meas"] - df["t_amb"]
    df["pi_eff"] = np.clip(0.10 * df["pi"], 0.005, 10.0)

    # Causal derivatives
    grp = df.groupby(["caseid", "seg_id"])
    df["d_tskin_raw"] = grp["t_skin_meas"].diff().fillna(0.0) / 10.0  # °C/s
    
    # 60s rolling smoothed derivative
    df["tskin_roll60"] = grp["t_skin_meas"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["d_tskin_filt"] = grp["tskin_roll60"].diff().fillna(0.0) / 10.0 # °C/s

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
    print(f"  Zero Overlap Check: {len(set(train_cases).intersection(set(test_cases))) == 0}")

    # Model 1: Basic Bioheat (3 params: a, b, alpha)
    def basic_formula(X, a, b, alpha):
        ts, ta, peff = X
        g = np.maximum(a + b * np.power(peff, alpha), 0.10)
        t_calc = ts + (ts - ta) / g
        return np.clip(t_calc, 30.0, 45.0)

    # Model 2: Extended Bioheat (4 params: a, b, alpha, c)
    def extended_formula(X, a, b, alpha, c):
        ts, ta, peff, dtdt = X
        g = np.maximum(a + b * np.power(peff, alpha), 0.10)
        t_calc = ts + (ts - ta) / g + c * dtdt
        return np.clip(t_calc, 30.0, 45.0)

    # 1. Fit Basic Bioheat on 80 Train Patients
    print("\n[1/3] Fitting Model 1: Basic Bioheat (3 parameters)...")
    X_tr_basic = (df_train["t_skin_meas"].values, df_train["t_amb"].values, df_train["pi_eff"].values)
    y_tr = df_train["t_core"].values

    popt_basic, pcov_basic = curve_fit(
        basic_formula, X_tr_basic, y_tr,
        p0=[0.64, 0.90, 0.75],
        bounds=([0.1, 0.05, 0.1], [5.0, 10.0, 3.0]),
        method="trf"
    )
    se_basic = np.sqrt(np.diag(pcov_basic))
    a_b, b_b, alpha_b = popt_basic
    print(f"  Fitted a     : {a_b:.5f} +- {se_basic[0]:.5f}")
    print(f"  Fitted b     : {b_b:.5f} +- {se_basic[1]:.5f}")
    print(f"  Fitted alpha : {alpha_b:.5f} +- {se_basic[2]:.5f}")

    # 2. Fit Extended Bioheat (Raw dT/dt) on 80 Train Patients
    print("\n[2/3] Fitting Model 2: Extended Bioheat (Raw dT/dt, 4 parameters)...")
    X_tr_ext_raw = (df_train["t_skin_meas"].values, df_train["t_amb"].values, df_train["pi_eff"].values, df_train["d_tskin_raw"].values)
    popt_ext_raw, pcov_ext_raw = curve_fit(
        extended_formula, X_tr_ext_raw, y_tr,
        p0=[a_b, b_b, alpha_b, 5.0],
        bounds=([0.1, 0.05, 0.1, 0.0], [5.0, 10.0, 3.0, 600.0]),
        method="trf"
    )
    se_ext_raw = np.sqrt(np.diag(pcov_ext_raw))
    a_er, b_er, alpha_er, c_er = popt_ext_raw
    print(f"  Fitted a     : {a_er:.5f} +- {se_ext_raw[0]:.5f}")
    print(f"  Fitted b     : {b_er:.5f} +- {se_ext_raw[1]:.5f}")
    print(f"  Fitted alpha : {alpha_er:.5f} +- {se_ext_raw[2]:.5f}")
    print(f"  Fitted c     : {c_er:.5f} +- {se_ext_raw[3]:.5f} s")

    # 3. Fit Extended Bioheat (Filtered 60s dT/dt) on 80 Train Patients
    print("\n[3/3] Fitting Model 3: Extended Bioheat (Filtered 60s dT/dt, 4 parameters)...")
    X_tr_ext_filt = (df_train["t_skin_meas"].values, df_train["t_amb"].values, df_train["pi_eff"].values, df_train["d_tskin_filt"].values)
    popt_ext_filt, pcov_ext_filt = curve_fit(
        extended_formula, X_tr_ext_filt, y_tr,
        p0=[a_b, b_b, alpha_b, 10.0],
        bounds=([0.1, 0.05, 0.1, 0.0], [5.0, 10.0, 3.0, 600.0]),
        method="trf"
    )
    se_ext_filt = np.sqrt(np.diag(pcov_ext_filt))
    a_ef, b_ef, alpha_ef, c_ef = popt_ext_filt
    print(f"  Fitted a     : {a_ef:.5f} +- {se_ext_filt[0]:.5f}")
    print(f"  Fitted b     : {b_ef:.5f} +- {se_ext_filt[1]:.5f}")
    print(f"  Fitted alpha : {alpha_ef:.5f} +- {se_ext_filt[2]:.5f}")
    print(f"  Fitted c     : {c_ef:.5f} +- {se_ext_filt[3]:.5f} s")

    # Evaluation function
    def evaluate(y_true, y_pred, cids, df_sub):
        res = y_pred - y_true
        mae = float(np.mean(np.abs(res)))
        rmse = float(np.sqrt(np.mean(res**2)))
        bias = float(np.mean(res))
        sd = float(np.std(res, ddof=1))
        lower_loa = float(bias - 1.96 * sd)
        upper_loa = float(bias + 1.96 * sd)
        acc_35 = float(np.mean(np.abs(res) <= 0.35) * 100.0)
        acc_20 = float(np.mean(np.abs(res) <= 0.20) * 100.0)

        per_case = []
        for cid in sorted(cids):
            m = (df_sub["caseid"] == cid)
            c_res = res[m]
            c_mae = float(np.mean(np.abs(c_res)))
            c_bias = float(np.mean(c_res))
            c_sd = float(np.std(c_res, ddof=1)) if len(c_res) > 1 else 0.0
            c_acc35 = float(np.mean(np.abs(c_res) <= 0.35) * 100.0)
            c_pass = bool(c_mae <= 0.35)
            per_case.append({
                "caseid": int(cid),
                "n_epochs": int(np.sum(m)),
                "mae": round(c_mae, 4),
                "bias": round(c_bias, 4),
                "sd": round(c_sd, 4),
                "acc_35": round(c_acc35, 1),
                "pass_mae35": c_pass
            })

        pass_cnt = sum(1 for c in per_case if c["pass_mae35"])
        macro_mae = round(float(np.mean([c["mae"] for c in per_case])), 4)

        return {
            "pooled_mae": round(mae, 4),
            "macro_mae": macro_mae,
            "rmse": round(rmse, 4),
            "bias": round(bias, 4),
            "sd": round(sd, 4),
            "lower_loa": round(lower_loa, 4),
            "upper_loa": round(upper_loa, 4),
            "acc_35": round(acc_35, 2),
            "acc_20": round(acc_20, 2),
            "cases_passed": pass_cnt,
            "cases_total": len(per_case),
            "pass_rate_pct": round(pass_cnt / len(per_case) * 100.0, 1),
            "per_case": per_case
        }

    # Evaluate on Test Cohort (20 Unseen Patients)
    X_te_basic = (df_test["t_skin_meas"].values, df_test["t_amb"].values, df_test["pi_eff"].values)
    X_te_ext_raw = (df_test["t_skin_meas"].values, df_test["t_amb"].values, df_test["pi_eff"].values, df_test["d_tskin_raw"].values)
    X_te_ext_filt = (df_test["t_skin_meas"].values, df_test["t_amb"].values, df_test["pi_eff"].values, df_test["d_tskin_filt"].values)
    y_te = df_test["t_core"].values

    pred_te_basic = basic_formula(X_te_basic, *popt_basic)
    pred_te_ext_raw = extended_formula(X_te_ext_raw, *popt_ext_raw)
    pred_te_ext_filt = extended_formula(X_te_ext_filt, *popt_ext_filt)

    met_te_basic = evaluate(y_te, pred_te_basic, test_cases, df_test)
    met_te_ext_raw = evaluate(y_te, pred_te_ext_raw, test_cases, df_test)
    met_te_ext_filt = evaluate(y_te, pred_te_ext_filt, test_cases, df_test)

    # Evaluate on Train Cohort (80 Patients)
    pred_tr_basic = basic_formula(X_tr_basic, *popt_basic)
    pred_tr_ext_raw = extended_formula(X_tr_ext_raw, *popt_ext_raw)
    pred_tr_ext_filt = extended_formula(X_tr_ext_filt, *popt_ext_filt)

    met_tr_basic = evaluate(y_tr, pred_tr_basic, train_cases, df_train)
    met_tr_ext_raw = evaluate(y_tr, pred_tr_ext_raw, train_cases, df_train)
    met_tr_ext_filt = evaluate(y_tr, pred_tr_ext_filt, train_cases, df_train)

    # Print Side-by-Side Comparison
    print("\n" + "=" * 115)
    print(f"{'BENCHMARK ON 20 UNSEEN TEST PATIENTS (' + str(len(df_test)) + ' Epochs, ' + str(round(len(df_test)*10/3600, 1)) + 'h)':^115}")
    print("=" * 115)
    print(f"{'Model':<38} | {'Pooled MAE':<11} | {'Macro MAE':<10} | {'Bias':<8} | {'SD':<7} | {'95% LoA':<17} | {'Pass Rate':<12} | {'Epoch Acc <=0.35C'}")
    print("-" * 115)
    
    table_rows = [
        ("1. Basic Bioheat (3 params, c=0)", met_te_basic),
        ("2. Extended Bioheat (Raw dT/dt)", met_te_ext_raw),
        ("3. Extended Bioheat (Filtered 60s dT/dt)", met_te_ext_filt)
    ]

    for name, m in table_rows:
        loa_str = f"[{m['lower_loa']:+.2f}, {m['upper_loa']:+.2f}]"
        pass_str = f"{m['cases_passed']}/{m['cases_total']} ({m['pass_rate_pct']}%)"
        print(f"{name:<38} | {m['pooled_mae']:<7.4f} °C | {m['macro_mae']:<7.4f} °C | {m['bias']:<+7.4f} | {m['sd']:<6.4f} | {loa_str:<17} | {pass_str:<12} | {m['acc_35']:.1f}%")

    # Per-Case Head-to-Head Comparison Table on 20 Test Patients
    print("\n" + "=" * 115)
    print(f"{'PER-PATIENT COMPARISON ON 20 UNSEEN TEST CASES':^115}")
    print("=" * 115)
    print(f"{'Case ID':<8} | {'Epochs':<7} | {'Basic Bioheat':<14} | {'Ext Bio (Raw)':<14} | {'Ext Bio (Filt)':<14} | {'Delta (Ext - Basic)':<20} | {'Status'}")
    print("-" * 115)

    cases_summary = []
    for case_b, case_er, case_ef in zip(met_te_basic["per_case"], met_te_ext_raw["per_case"], met_te_ext_filt["per_case"]):
        cid = case_b["caseid"]
        n_ep = case_b["n_epochs"]
        m_b = case_b["mae"]
        m_er = case_er["mae"]
        m_ef = case_ef["mae"]
        delta = m_ef - m_b
        status = "PASSED (<=0.35)" if m_ef <= 0.35 else "FAILED (>0.35)"
        print(f"Case {cid:<3d}  | {n_ep:<7d} | {m_b:<10.4f} °C | {m_er:<10.4f} °C | {m_ef:<10.4f} °C | {delta:<+10.4f} °C         | {status}")
        cases_summary.append({
            "caseid": cid,
            "epochs": n_ep,
            "mae_basic": m_b,
            "mae_ext_raw": m_er,
            "mae_ext_filt": m_ef,
            "gain_filtered": round(delta, 4)
        })

    # Save summary json
    summary_data = {
        "study": "Basic vs Extended Bioheat on 100 VitalDB Cases (80/20 Split)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cohort": {
            "total_patients": 100,
            "train_patients_count": 80,
            "test_patients_count": 20,
            "total_epochs": len(df),
            "train_epochs": len(df_train),
            "test_epochs": len(df_test)
        },
        "fitted_parameters": {
            "basic_bioheat": {
                "a": round(float(a_b), 5),
                "b": round(float(b_b), 5),
                "alpha": round(float(alpha_b), 5),
                "se": [round(float(s), 5) for s in se_basic]
            },
            "extended_bioheat_raw": {
                "a": round(float(a_er), 5),
                "b": round(float(b_er), 5),
                "alpha": round(float(alpha_er), 5),
                "c_seconds": round(float(c_er), 5),
                "se": [round(float(s), 5) for s in se_ext_raw]
            },
            "extended_bioheat_filtered_60s": {
                "a": round(float(a_ef), 5),
                "b": round(float(b_ef), 5),
                "alpha": round(float(alpha_ef), 5),
                "c_seconds": round(float(c_ef), 5),
                "se": [round(float(s), 5) for s in se_ext_filt]
            }
        },
        "test_metrics_20_patients": {
            "basic_bioheat": met_te_basic,
            "extended_bioheat_raw": met_te_ext_raw,
            "extended_bioheat_filtered": met_te_ext_filt
        },
        "train_metrics_80_patients": {
            "basic_bioheat": met_tr_basic,
            "extended_bioheat_raw": met_tr_ext_raw,
            "extended_bioheat_filtered": met_tr_ext_filt
        },
        "per_patient_comparison": cases_summary,
        "runtime_sec": round(time.time() - t_start, 2)
    }

    with open(OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[OUTPUT] Saved summary report to: {OUTPUT_SUMMARY_JSON}")
    print(f"[RUNTIME] Completed in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    run_experiment()
