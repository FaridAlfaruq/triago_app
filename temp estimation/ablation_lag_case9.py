"""
ablation_lag_case9.py - Thermal-Lag Ablation Experiment for Case 9
Author: Alex (Lead Orchestrator)
Master Reference: scratch/temp estimation/PRD_VitalDB_Temp_Validation.md

Hypothesis Testing:
Does 1st-order thermal lag (tau = 180s) cause Case 9's negative bias (-0.21 C),
wide limits of agreement ([-1.25, +0.83] C), and negative R2 (-4.63)?

Conditions tested on Case 9 (N = 272 epochs, 45.3 min, T_core warming 33.8 -> 37.0 C):
1. tau = 180s (Standard model with Wissler dynamic thermal lag)
2. tau = 60s  (Reduced thermal lag)
3. tau = 0s   (Zero thermal lag / instantaneous Stolwijk-Hardy equilibrium)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from stage2 import process_case_dsp
from stage3 import (
    compute_equilibrium_temperature,
    simulate_skin_temperature_ode,
    apply_mlx90614_noise
)
from stage4 import core_temperature_model, TEST_CASES

def run_case9_ablation():
    print("=" * 110)
    print("CASE 9 THERMAL-LAG ABLATION STUDY: TESTING LAG CONTRIBUTION TO ERROR")
    print("=" * 110)

    # Fitted parameters from Stage 4 train cohort
    json_path = os.path.join(os.path.dirname(__file__), "stage4_calibration_summary.json")
    with open(json_path, "r") as f:
        s4_summary = json.load(f)
    popt = s4_summary["fitted_parameters"]
    a, b, alpha = popt["a"], popt["b"], popt["alpha"]

    print(f"Using Fitted Parameters: a = {a:.5f}, b = {b:.5f}, alpha = {alpha:.5f}")

    # Load Case 9 raw DSP epochs
    dsp_res = process_case_dsp(9)
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

    t_sec = df_valid["t_center_sec"].values
    t_core = df_valid["t_core"].values
    pi_vals = df_valid["pi"].values
    n_epochs = len(df_valid)

    # Fix random seed for exact reproducibility across conditions
    rng_amb = np.random.default_rng(42 + 9)
    t_amb = rng_amb.normal(loc=21.0, scale=0.30, size=n_epochs)

    # Compute steady-state equilibrium temperature
    t_eq = compute_equilibrium_temperature(
        t_core=t_core,
        pi=pi_vals,
        t_amb=t_amb,
        a=0.574,
        b=0.850,
        alpha=0.800,
        pi_scale=0.10
    )

    tau_conditions = [180.0, 60.0, 0.0]
    results = []

    for tau in tau_conditions:
        # Solve thermal ODE (or instantaneous if tau == 0)
        if tau > 0:
            t_skin = simulate_skin_temperature_ode(t_sec, t_eq, tau=tau, dt_max=60.0)
        else:
            t_skin = t_eq.copy()

        # Apply same MLX90614 sensor noise realization
        rng_sensor = np.random.default_rng(42 + 9)
        t_skin_meas = apply_mlx90614_noise(t_skin, sigma_sensor=0.05, adc_step=0.02, rng=rng_sensor)

        # Invert with Stage 4 model
        t_est = core_temperature_model(t_skin_meas, t_amb, pi_vals, a, b, alpha)

        residuals = t_est - t_core
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mean_bias = float(np.mean(residuals))
        sd_diff = float(np.std(residuals, ddof=1))
        upper_loa = float(mean_bias + 1.96 * sd_diff)
        lower_loa = float(mean_bias - 1.96 * sd_diff)

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((t_core - np.mean(t_core)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        r_val = float(np.corrcoef(t_core, t_est)[0, 1])

        results.append({
            "tau_sec": tau,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mean_bias": round(mean_bias, 4),
            "upper_loa": round(upper_loa, 4),
            "lower_loa": round(lower_loa, 4),
            "loa_width": round(upper_loa - lower_loa, 4),
            "r2_score": round(r2, 4),
            "pearson_r": round(r_val, 4),
            "bias_pass": abs(mean_bias) <= 0.20,
            "lower_loa_pass": lower_loa >= -0.90,
            "r2_pass": r2 > 0.0
        })

    df_res = pd.DataFrame(results)

    print("\nCASE 9 ABLATION RESULTS TABLE:")
    print("-" * 110)
    print(df_res[["tau_sec", "mae", "rmse", "mean_bias", "lower_loa", "upper_loa", "r2_score", "pearson_r", "bias_pass", "lower_loa_pass", "r2_pass"]].to_string(index=False))
    print("-" * 110)

    # Save summary
    out_json = os.path.join(os.path.dirname(__file__), "case9_ablation_summary.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Case 9 ablation summary to: {out_json}")

    return results

if __name__ == "__main__":
    run_case9_ablation()
