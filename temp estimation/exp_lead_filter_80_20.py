"""
exp_lead_filter_80_20.py - Subject-Wise 80/20 Evaluation of Lead Filter Bioheat Formula
Model:
    T_est = T_skin + (T_skin - T_amb) / (a + b * PI_eff) + k * (T_skin - EMA_60s) + c0
where:
    EMA_60s: Single-state causal exponential moving average (span=6 epochs = 60s)
    PI_eff = 0.10 * PI
    a  = Minimum conductive resistance
    b  = Perfusion sensitivity
    k  = Phase-lead anti-lag dynamic gain
    c0 = Basal metabolic offset (°C)

Dataset:
    100 VitalDB surgical cases (cohort100_epochs.parquet, 75,188 epochs, 208.9h)
    Subject-wise split: 80 Train subjects (55,816 epochs) / 20 Unseen Test subjects (19,372 epochs)
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
OUTPUT_SUMMARY_JSON = os.path.join(CURRENT_DIR, "lead_filter_80_20_summary.json")

def run_evaluation():
    t_start = time.time()
    print("=" * 115)
    print("SUBJECT-WISE 80/20 VALIDATION: LEAD FILTER (ANTI-LAG) BIOHEAT FORMULA")
    print("Dataset: 100 VitalDB Surgical Cases | Split: 80 Train Patients / 20 Unseen Test Patients")
    print("=" * 115)

    df = pd.read_parquet(CACHE_100_PARQUET)
    all_cases = sorted(list(df["caseid"].unique()))
    assert len(all_cases) == 100, f"Expected 100 cases, found {len(all_cases)}"

    # Effective perfusion scaling (0.10 * PI)
    df["pi_eff"] = np.clip(0.10 * df["pi"], 0.005, 10.0)
    df["delta_T"] = df["t_skin_meas"] - df["t_amb"]

    # Causal Single-State EMA per subject and segment (span=6 epochs = 60s)
    # alpha_ema = 2 / (span + 1) = 2/7 ≈ 0.2857
    grp = df.groupby(["caseid", "seg_id"])
    df["tskin_ema60"] = grp["t_skin_meas"].transform(lambda s: s.ewm(span=6, adjust=False).mean())
    df["lead_term"] = df["t_skin_meas"] - df["tskin_ema60"]

    # Deterministic Subject-Wise 80/20 Split (Seed 2026)
    rng = np.random.default_rng(2026)
    shuffled = all_cases.copy()
    rng.shuffle(shuffled)
    n_train = 80
    train_cases = sorted(shuffled[:n_train])
    test_cases = sorted(shuffled[n_train:])

    df_train = df[df["caseid"].isin(train_cases)].copy().reset_index(drop=True)
    df_test = df[df["caseid"].isin(test_cases)].copy().reset_index(drop=True)

    print(f"\n[SUBJECT-WISE SPLIT AUDIT]")
    print(f"  Total Cohort: {len(all_cases)} Patients ({len(df):,} epochs, {len(df)*10/3600:.1f} hours)")
    print(f"  Train Cohort: {len(train_cases)} Patients ({len(df_train):,} epochs, {len(df_train)*10/3600:.1f} hours)")
    print(f"  Test Cohort : {len(test_cases)} Patients ({len(df_test):,} epochs, {len(df_test)*10/3600:.1f} hours)")
    print(f"  Zero Subject Overlap: {len(set(train_cases).intersection(set(test_cases))) == 0}")

    # Model Functions
    # 1. Lead Filter Model (4 params: a, b, k, c0)
    def lead_model(X, a, b, k, c0):
        ts, ta, peff, lead = X
        g_cs = np.maximum(a + b * peff, 0.10)
        t_raw = ts + (ts - ta) / g_cs + k * lead + c0
        return np.clip(t_raw, 30.0, 45.0)

    # 2. Reference Linear Rational Model (3 params: a, b, c0)
    def ref_model(X, a, b, c0):
        ts, ta, peff = X
        g_cs = np.maximum(a + b * peff, 0.10)
        t_raw = ts + (ts - ta) / g_cs + c0
        return np.clip(t_raw, 30.0, 45.0)

    # 3. Basic Bioheat Model (3 params: a, b, alpha)
    def basic_model(X, a, b, alpha):
        ts, ta, peff = X
        g_cs = np.maximum(a + b * np.power(peff, alpha), 0.10)
        t_raw = ts + (ts - ta) / g_cs
        return np.clip(t_raw, 30.0, 45.0)

    # Fit Lead Filter on 80 Train Patients
    print("\n[FITTING LEAD FILTER PARAMETERS ON 80 TRAIN PATIENTS VIA TRF]")
    X_tr_lead = (df_train["t_skin_meas"].values, df_train["t_amb"].values, df_train["pi_eff"].values, df_train["lead_term"].values)
    y_tr = df_train["t_core"].values

    popt_lead, pcov_lead = curve_fit(
        lead_model, X_tr_lead, y_tr,
        p0=[0.85, 1.05, 1.50, 1.15],
        bounds=([0.1, 0.05, -10.0, -5.0], [5.0, 10.0, 20.0, 5.0]),
        method="trf"
    )
    se_lead = np.sqrt(np.diag(pcov_lead))
    a_l, b_l, k_l, c0_l = popt_lead

    print(f"  Fitted a  (Conductive Resistance) : {a_l:.5f} +- {se_lead[0]:.5f}")
    print(f"  Fitted b  (Perfusion Sensitivity) : {b_l:.5f} +- {se_lead[1]:.5f}")
    print(f"  Fitted k  (Phase-Lead Gain)       : {k_l:.5f} +- {se_lead[2]:.5f}")
    print(f"  Fitted c0 (Metabolic Offset)      : {c0_l:+.5f} +- {se_lead[3]:.5f} °C")

    # Fit Reference Linear Rational on 80 Train Patients
    X_tr_ref = (df_train["t_skin_meas"].values, df_train["t_amb"].values, df_train["pi_eff"].values)
    popt_ref, _ = curve_fit(ref_model, X_tr_ref, y_tr, p0=[0.85, 1.05, 1.20], method="trf")

    # Fit Basic Bioheat on 80 Train Patients
    popt_basic, _ = curve_fit(basic_model, X_tr_ref, y_tr, p0=[0.64, 0.90, 0.75], method="trf")

    # Metric Calculator
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
            "acc_35": round(acc_35, 2),
            "acc_20": round(acc_20, 2),
            "cases_passed": pass_count,
            "cases_total": len(per_case),
            "pass_rate_pct": pass_rate,
            "per_case": per_case
        }

    # Evaluate on Test Cohort (20 Unseen Patients)
    X_te_lead = (df_test["t_skin_meas"].values, df_test["t_amb"].values, df_test["pi_eff"].values, df_test["lead_term"].values)
    X_te_ref = (df_test["t_skin_meas"].values, df_test["t_amb"].values, df_test["pi_eff"].values)
    y_te = df_test["t_core"].values

    pred_te_lead = lead_model(X_te_lead, *popt_lead)
    pred_te_ref = ref_model(X_te_ref, *popt_ref)
    pred_te_basic = basic_model(X_te_ref, *popt_basic)

    met_te_lead = calc_metrics(y_te, pred_te_lead, test_cases, df_test)
    met_te_ref = calc_metrics(y_te, pred_te_ref, test_cases, df_test)
    met_te_basic = calc_metrics(y_te, pred_te_basic, test_cases, df_test)

    # Evaluate on Train Cohort (80 Patients)
    pred_tr_lead = lead_model(X_tr_lead, *popt_lead)
    met_tr_lead = calc_metrics(y_tr, pred_tr_lead, train_cases, df_train)

    # Evaluate on Full 100 Patient Cohort
    X_all_lead = (df["t_skin_meas"].values, df["t_amb"].values, df["pi_eff"].values, df["lead_term"].values)
    y_all = df["t_core"].values
    pred_all_lead = lead_model(X_all_lead, *popt_lead)
    met_all_lead = calc_metrics(y_all, pred_all_lead, all_cases, df)

    print("\n" + "=" * 115)
    print(f"{'HEAD-TO-HEAD COMPARISON ON 20 UNSEEN TEST PATIENTS (19,372 Epochs, 53.8h)':^115}")
    print("=" * 115)
    print(f"{'Model':<42} | {'Pooled MAE':<11} | {'Macro MAE':<10} | {'Bias':<8} | {'SD':<7} | {'95% LoA':<17} | {'Pass Rate':<12} | {'Epoch Acc <=0.35C'}")
    print("-" * 115)

    comp_rows = [
        ("1. Basic Bioheat (Baseline)", met_te_basic),
        ("2. Linear Rational + c0 (Reference)", met_te_ref),
        ("3. Lead Filter Bioheat (+ anti-lag)", met_te_lead)
    ]

    for name, m in comp_rows:
        loa_str = f"[{m['lower_loa']:+.2f}, {m['upper_loa']:+.2f}]"
        pass_str = f"{m['cases_passed']}/{m['cases_total']} ({m['pass_rate_pct']}%)"
        print(f"{name:<42} | {m['pooled_mae']:<7.4f} °C | {m['macro_mae']:<7.4f} °C | {m['bias']:<+7.4f} | {m['sd']:<6.4f} | {loa_str:<17} | {pass_str:<12} | {m['acc_35']:.1f}%")

    print("\n" + "=" * 115)
    print(f"{'FULL 100-PATIENT COHORT GENERALIZATION (75,188 Epochs, 208.9h)':^115}")
    print("=" * 115)
    loa_str = f"[{met_all_lead['lower_loa']:+.2f}, {met_all_lead['upper_loa']:+.2f}]"
    pass_str = f"{met_all_lead['cases_passed']}/{met_all_lead['cases_total']} ({met_all_lead['pass_rate_pct']}%)"
    print(f"{'Lead Filter Bioheat (Full 100 Patients)':<42} | {met_all_lead['pooled_mae']:<7.4f} °C | {met_all_lead['macro_mae']:<7.4f} °C | {met_all_lead['bias']:<+7.4f} | {met_all_lead['sd']:<6.4f} | {loa_str:<17} | {pass_str:<12} | {met_all_lead['acc_35']:.1f}%")

    # Detailed Per-Subject Breakdown on 20 Test Patients
    print("\n" + "=" * 115)
    print(f"{'DETAILED BREAKDOWN: 20 UNSEEN TEST PATIENTS (LEAD FILTER BIOHEAT)':^115}")
    print("=" * 115)
    print(f"{'Case ID':<8} | {'Epochs':<7} | {'Duration':<9} | {'Ref MAE':<10} | {'Lead MAE':<10} | {'Gain (°C)':<11} | {'Acc <=0.35C':<12} | {'Clinical Status'}")
    print("-" * 115)

    breakdown_list = []
    for c_l, c_r in zip(met_te_lead["per_case"], met_te_ref["per_case"]):
        cid = c_l["caseid"]
        n_ep = c_l["n_epochs"]
        dur = f"{c_l['duration_min']}m"
        m_ref = c_r["mae"]
        m_lead = c_l["mae"]
        gain = m_lead - m_ref
        acc = f"{c_l['acc_35']:.1f}%"
        status = "PASSED (MAE <= 0.35)" if c_l["pass_mae35"] else "FAILED (MAE > 0.35)"
        print(f"Case {cid:<3d}  | {n_ep:<7d} | {dur:<9} | {m_ref:<8.4f} °C | {m_lead:<8.4f} °C | {gain:<+9.4f} °C | {acc:<12} | {status}")
        breakdown_list.append({
            "caseid": cid,
            "epochs": n_ep,
            "mae_ref": m_ref,
            "mae_lead": m_lead,
            "delta_lead_vs_ref": round(gain, 4),
            "pass_mae35": c_l["pass_mae35"]
        })

    # Save summary JSON
    summary_data = {
        "model_name": "Lead Filter (Anti-Lag) Bioheat Formula",
        "formula": "T_est = T_skin + (T_skin - T_amb) / (a + b * PI_eff) + k * (T_skin - EMA_60s) + c0",
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
            "a": round(float(a_l), 5),
            "b": round(float(b_l), 5),
            "k": round(float(k_l), 5),
            "c0": round(float(c0_l), 5),
            "std_errors": [round(float(s), 5) for s in se_lead]
        },
        "test_metrics_20_patients": {
            "lead_filter": met_te_lead,
            "linear_rational_c0_ref": met_te_ref,
            "basic_bioheat_baseline": met_te_basic
        },
        "full_cohort_lead_metrics": met_all_lead,
        "per_subject_comparison": breakdown_list,
        "runtime_sec": round(time.time() - t_start, 2)
    }

    with open(OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[OUTPUT] Saved full evaluation to: {OUTPUT_SUMMARY_JSON}")
    print(f"[RUNTIME] Completed in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    run_evaluation()
