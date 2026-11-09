"""
exp_pi_sensitivity.py - Empirical Sensitivity & Internal Consistency Audit of PI Scaling
Author: Alex (Lead Orchestrator)
Master Reference: scratch/temp estimation/PRD_VitalDB_Temp_Validation.md

Audits:
1. Scaling factors tested: s in [0.01, 0.03, 0.10, 0.30, 1.00]
2. Experiment 1: Matched Inversion (s_sim = s_inv = s) -> Test optimizer identifiability & parameter scaling
3. Experiment 2: Biophysical Conductance Distribution & Sensitivity -> Test physiological plausibility of G_cs
4. Case-by-case & Pooled evaluation metrics: MAE, RMSE, Bias, LoA, R2
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Dict, Any, List

from stage2 import process_case_dsp
from stage3 import (
    compute_equilibrium_temperature,
    simulate_skin_temperature_ode,
    apply_mlx90614_noise
)

SCALING_FACTORS = [0.01, 0.03, 0.10, 0.30, 1.00]
TRAIN_CASES = [1, 2, 3, 4, 5, 6, 7]
TEST_CASES = [8, 9, 10]
ALL_CASES = list(range(1, 11))

A_TRUE = 0.574
B_TRUE = 0.850
ALPHA_TRUE = 0.800
TAU_TRUE = 180.0

def load_all_case_epochs():
    """Loads all valid epochs from Stage 2 DSP for 10 cases."""
    case_epochs = {}
    for cid in ALL_CASES:
        dsp_res = process_case_dsp(cid)
        df_epochs = dsp_res["df_epochs"]
        valid_mask = (
            df_epochs["sqi_passed"] &
            df_epochs["t_core"].notna() &
            df_epochs["pi"].notna() &
            (df_epochs["t_core"] >= 33.0) &
            (df_epochs["t_core"] <= 45.0) &
            (df_epochs["pi"] > 0.0)
        )
        df_valid = df_epochs[valid_mask].copy().reset_index(drop=True)
        df_valid["caseid"] = cid
        case_epochs[cid] = df_valid
    return case_epochs

def simulate_cohort(case_epochs, pi_scale, seed=42):
    """Simulates dynamic skin temperature and MLX90614 noise for given pi_scale."""
    sim_dfs = []
    for cid, df_valid in case_epochs.items():
        rng = np.random.default_rng(seed + cid)
        t_sec = df_valid["t_center_sec"].values
        t_core = df_valid["t_core"].values
        pi_vals = df_valid["pi"].values
        n = len(df_valid)

        t_amb = rng.normal(loc=21.0, scale=0.30, size=n)
        t_eq = compute_equilibrium_temperature(
            t_core=t_core,
            pi=pi_vals,
            t_amb=t_amb,
            a=A_TRUE,
            b=B_TRUE,
            alpha=ALPHA_TRUE,
            pi_scale=pi_scale
        )
        t_skin = simulate_skin_temperature_ode(t_sec, t_eq, tau=TAU_TRUE, dt_max=60.0)
        t_skin_meas = apply_mlx90614_noise(t_skin, sigma_sensor=0.05, adc_step=0.02, rng=rng)

        df_sim = df_valid.copy()
        df_sim["t_amb"] = t_amb
        df_sim["t_eq"] = t_eq
        df_sim["t_skin"] = t_skin
        df_sim["t_skin_meas"] = t_skin_meas
        df_sim["pi_scale"] = pi_scale
        sim_dfs.append(df_sim)

    return pd.concat(sim_dfs, ignore_index=True)

def core_temp_model(t_skin, t_amb, pi, a, b, alpha, s):
    pi_eff = np.maximum(pi * s, 1e-4)
    denom = np.maximum(a + b * np.power(pi_eff, alpha), 0.10)
    return t_skin + (t_skin - t_amb) / denom

def compute_metrics(y_true, y_pred):
    res = y_pred - y_true
    mae = float(np.mean(np.abs(res)))
    rmse = float(np.sqrt(np.mean(res**2)))
    bias = float(np.mean(res))
    sd = float(np.std(res, ddof=1))
    upper_loa = float(bias + 1.96 * sd)
    lower_loa = float(bias - 1.96 * sd)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "lower_loa": round(lower_loa, 4),
        "upper_loa": round(upper_loa, 4),
        "r2": round(r2, 4)
    }

def run_experiment():
    print("Loading Stage 2 epochs for 10 cases...")
    case_epochs = load_all_case_epochs()
    total_epochs = sum(len(df) for df in case_epochs.values())
    print(f"Loaded {total_epochs} total valid epochs.\n")

    # -------------------------------------------------------------------------
    # PART 1: Physical Conductance Inspection
    # -------------------------------------------------------------------------
    print("=" * 115)
    print("PART 1: BIOPHYSICAL CONDUCTANCE G_cs = 0.574 + 0.850 * (s * PI)^0.80 DISTRIBUTION AUDIT")
    print("Literature Benchmark: Minimal G_cs (basal) ~ 0.57 | Maximal G_cs (vasodilation) ~ 3.5 - 4.5")
    print("=" * 115)

    all_pi = np.concatenate([df["pi"].values for df in case_epochs.values()])
    conductance_stats = []

    for s in SCALING_FACTORS:
        pi_eff = all_pi * s
        g_cs = A_TRUE + B_TRUE * np.power(pi_eff, ALPHA_TRUE)
        conductance_stats.append({
            "scale_s": s,
            "pi_eff_min": round(float(np.min(pi_eff)), 3),
            "pi_eff_med": round(float(np.median(pi_eff)), 3),
            "pi_eff_mean": round(float(np.mean(pi_eff)), 3),
            "pi_eff_max": round(float(np.max(pi_eff)), 3),
            "g_cs_min": round(float(np.min(g_cs)), 3),
            "g_cs_med": round(float(np.median(g_cs)), 3),
            "g_cs_mean": round(float(np.mean(g_cs)), 3),
            "g_cs_max": round(float(np.max(g_cs)), 3),
            "physio_plausible": "YES" if (np.min(g_cs) >= 0.57 and np.max(g_cs) <= 8.0) else "NO (Unphysiological)"
        })

    df_g = pd.DataFrame(conductance_stats)
    print(df_g[["scale_s", "pi_eff_min", "pi_eff_med", "pi_eff_max", "g_cs_min", "g_cs_med", "g_cs_mean", "g_cs_max", "physio_plausible"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 2: Experiment 1 - Matched Inversion Consistency (s_sim = s_inv = s)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PART 2: EXPERIMENT 1 - MATCHED INVERSION (s_sim = s_inv = s)")
    print("Tests mathematical identifiability: Does curve_fit recover model under matched scaling?")
    print("=" * 115)

    exp1_results = []

    for s in SCALING_FACTORS:
        df_sim = simulate_cohort(case_epochs, pi_scale=s)
        df_tr = df_sim[df_sim["caseid"].isin(TRAIN_CASES)]
        df_te = df_sim[df_sim["caseid"].isin(TEST_CASES)]

        # Objective for curve_fit
        def fit_obj(X, a, b, alpha):
            ts, ta, p = X
            return core_temp_model(ts, ta, p, a, b, alpha, s)

        X_tr = (df_tr["t_skin_meas"].values, df_tr["t_amb"].values, df_tr["pi"].values)
        y_tr = df_tr["t_core"].values

        popt, _ = curve_fit(
            fit_obj, X_tr, y_tr,
            p0=[0.574, 0.850, 0.800],
            bounds=([0.1, 0.01, 0.1], [5.0, 50.0, 3.0]),
            method="trf"
        )
        a_hat, b_hat, alpha_hat = popt

        # Evaluate on Test Cohort (pooled & per case)
        X_te = (df_te["t_skin_meas"].values, df_te["t_amb"].values, df_te["pi"].values)
        y_te = df_te["t_core"].values
        y_pred_te = core_temp_model(X_te[0], X_te[1], X_te[2], a_hat, b_hat, alpha_hat, s)
        m_pool = compute_metrics(y_te, y_pred_te)

        # Case 8, 9, 10
        case_mets = {}
        for cid in TEST_CASES:
            sub = df_te[df_te["caseid"] == cid]
            yp = core_temp_model(sub["t_skin_meas"].values, sub["t_amb"].values, sub["pi"].values, a_hat, b_hat, alpha_hat, s)
            case_mets[cid] = compute_metrics(sub["t_core"].values, yp)

        exp1_results.append({
            "scale_s": s,
            "fitted_a": round(float(a_hat), 4),
            "fitted_b": round(float(b_hat), 4),
            "fitted_alpha": round(float(alpha_hat), 4),
            "pooled_mae": m_pool["mae"],
            "pooled_rmse": m_pool["rmse"],
            "pooled_bias": m_pool["bias"],
            "pooled_lower_loa": m_pool["lower_loa"],
            "pooled_upper_loa": m_pool["upper_loa"],
            "pooled_r2": m_pool["r2"],
            "c8_mae": case_mets[8]["mae"],
            "c8_r2": case_mets[8]["r2"],
            "c9_mae": case_mets[9]["mae"],
            "c9_bias": case_mets[9]["bias"],
            "c9_lower_loa": case_mets[9]["lower_loa"],
            "c9_r2": case_mets[9]["r2"],
            "c10_mae": case_mets[10]["mae"],
            "c10_r2": case_mets[10]["r2"]
        })

    df_exp1 = pd.DataFrame(exp1_results)
    print("\nFitted Parameters & Test Cohort Metrics:")
    print(df_exp1[["scale_s", "fitted_a", "fitted_b", "fitted_alpha", "pooled_mae", "pooled_rmse", "pooled_bias", "pooled_r2"]].to_string(index=False))
    print("\nCase 9 Breakdown across Scaling Factors:")
    print(df_exp1[["scale_s", "c9_mae", "c9_bias", "c9_lower_loa", "c9_r2"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 3: Experiment 2 - Sensitivity to Inversion Scale Mismatch
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PART 3: EXPERIMENT 2 - SENSITIVITY TO SCALE MISMATCH (Fixed Forward s_sim = 0.10, Inversion s_inv in S)")
    print("=" * 115)

    df_sim_base = simulate_cohort(case_epochs, pi_scale=0.10)
    df_tr_base = df_sim_base[df_sim_base["caseid"].isin(TRAIN_CASES)]
    df_te_base = df_sim_base[df_sim_base["caseid"].isin(TEST_CASES)]

    # Model fitted under baseline s = 0.10
    a_base, b_base, alpha_base = 0.61448, 0.82574, 0.80720
    exp2_results = []

    for s_test in SCALING_FACTORS:
        # Evaluate model directly using s_test without retraining
        X_te = (df_te_base["t_skin_meas"].values, df_te_base["t_amb"].values, df_te_base["pi"].values)
        y_te = df_te_base["t_core"].values
        y_pred = core_temp_model(X_te[0], X_te[1], X_te[2], a_base, b_base, alpha_base, s_test)
        m = compute_metrics(y_te, y_pred)

        exp2_results.append({
            "s_eval": s_test,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "bias": m["bias"],
            "lower_loa": m["lower_loa"],
            "upper_loa": m["upper_loa"],
            "r2": m["r2"]
        })

    df_exp2 = pd.DataFrame(exp2_results)
    print(df_exp2.to_string(index=False))

    # Save summary
    out_json = os.path.join(os.path.dirname(__file__), "pi_sensitivity_summary.json")
    with open(out_json, "w") as f:
        json.dump({
            "conductance_inspection": conductance_stats,
            "exp1_matched_inversion": exp1_results,
            "exp2_scale_mismatch": exp2_results
        }, f, indent=2)
    print(f"\nSaved full sensitivity audit summary to: {out_json}")

if __name__ == "__main__":
    run_experiment()
