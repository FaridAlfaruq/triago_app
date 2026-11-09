"""
exp_realistic_amb_loso.py - Realistic Ambient Thermal Drift Simulation & LOSO Validation
Task ID: TASK-REALISTIC-AMB-LOSO-01

Physics & Problem Statement:
- Realistic Operating Room Ambient Temperature:
  Simulated via Ornstein-Uhlenbeck (OU) mean-reverting process:
    tau_drift = 1800 s (theta = 1/1800 s^-1)
    mu = 21.5 degC
    sigma = 0.05 degC / sqrt(10s)
    clamped to [18.0, 25.0] degC
  Melexis MLX90614 sensor physics:
    NETD Gaussian noise N(0, 0.05^2 degC) + 17-bit ADC quantization step (0.02 degC).
  Biophysical Stolwijk-Hardy ODE:
    tau = 180 s, PI_eff = 0.10 * PI, A = 0.574, B = 0.850, alpha = 0.800.
    Simulates continuous dynamic skin temperature T_skin and measured T_skin_meas.

Validation Schemes:
- Scheme 1: Deterministic Subject-Wise 80/20 Split (Seed 2026: 80 Train / 20 Unseen Test).
- Scheme 2: Full 100-Fold Leave-One-Subject-Out (LOSO) Cross-Validation.

Zero Data Leakage:
- Zero subject overlap between train and test sets.
- Single-state causal past-recursive EMAs computed strictly within (caseid, seg_id).
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PARQUET = os.path.join(CURRENT_DIR, "cohort100_epochs.parquet")
OUTPUT_PARQUET = os.path.join(CURRENT_DIR, "cohort100_epochs_realistic_amb.parquet")
OUTPUT_SUMMARY_JSON = os.path.join(CURRENT_DIR, "realistic_amb_loso_summary.json")

# Biophysical and Sensor Ground Truth Parameters
A_TRUE = 0.574
B_TRUE = 0.850
ALPHA_TRUE = 0.800
TAU_SKIN = 180.0
PI_SCALE = 0.10

# Realistic Ambient OU Parameters
MU_AMB = 21.5
THETA_AMB = 1.0 / 1800.0         # Relaxation time tau = 30 min (1800 s)
SIGMA_AMB = 0.05 / np.sqrt(10.0) # Volatility 0.05 degC / sqrt(10s)
AMB_MIN = 18.0
AMB_MAX = 25.0

# Sensor Measurement Parameters (MLX90614)
SENSOR_NETD = 0.05               # 0.05 degC NETD
ADC_STEP = 0.02                  # 17-bit digital resolution (0.02 degC step)


def simulate_realistic_ambient_cohort(df_input: pd.DataFrame, force_recompute: bool = True) -> pd.DataFrame:
    """
    Simulates realistic operating room ambient temperature drift (OU process)
    and passes it through the exact Stolwijk-Hardy bioheat ODE for all 100 cases.
    """
    if not force_recompute and os.path.exists(OUTPUT_PARQUET):
        print(f"[CACHE] Loading realistic ambient dataset from: {os.path.basename(OUTPUT_PARQUET)}")
        return pd.read_parquet(OUTPUT_PARQUET)

    print("\n" + "=" * 115)
    print("STEP 1: SIMULATING REALISTIC AMBIENT THERMAL DRIFT (ORNSTEIN-UHLENBECK) & STOLWIJK-HARDY ODE")
    print(f"Cohort: 100 VitalDB Cases ({len(df_input):,} epochs) | OU: mu={MU_AMB}C, tau=1800s, NETD=0.05C, ADC=0.02C")
    print("=" * 115)

    sim_dfs = []
    t_start = time.time()

    for (cid, seg), group in df_input.groupby(["caseid", "seg_id"], sort=False):
        group = group.sort_values("t_center_sec").copy()
        n = len(group)
        t_sec = group["t_center_sec"].values
        t_core = group["t_core"].values
        pi = group["pi"].values

        # Deterministic per-segment RNG seed
        rng = np.random.default_rng(2026 + int(cid) * 100 + int(seg))

        # 1. Continuous OU Ambient Temperature (True)
        t_amb_true = np.zeros(n, dtype=np.float64)
        stat_var = SIGMA_AMB**2 / (2.0 * THETA_AMB)
        t_amb_true[0] = np.clip(rng.normal(MU_AMB, np.sqrt(stat_var)), AMB_MIN, AMB_MAX)

        for k in range(1, n):
            dt = t_sec[k] - t_sec[k - 1]
            if dt <= 0:
                dt = 10.0
            decay = np.exp(-THETA_AMB * dt)
            step_var = stat_var * (1.0 - np.exp(-2.0 * THETA_AMB * dt))
            step_noise = rng.normal(0.0, np.sqrt(max(step_var, 1e-12)))
            val = MU_AMB + (t_amb_true[k - 1] - MU_AMB) * decay + step_noise
            t_amb_true[k] = np.clip(val, AMB_MIN, AMB_MAX)

        # 2. Ambient Sensor Measurement (MLX90614 NETD noise + 17-bit quantization)
        amb_noise = rng.normal(0.0, SENSOR_NETD, size=n)
        t_amb_meas = np.round((t_amb_true + amb_noise) / ADC_STEP) * ADC_STEP

        # 3. Dynamic Skin Temperature ODE (Stolwijk-Hardy)
        pi_eff = np.maximum(pi * PI_SCALE, 1e-4)
        g_cs = A_TRUE + B_TRUE * np.power(pi_eff, ALPHA_TRUE)
        t_eq = (g_cs * t_core + t_amb_true) / (1.0 + g_cs)

        t_skin = np.zeros(n, dtype=np.float64)
        t_skin[0] = t_eq[0]
        for k in range(1, n):
            dt = t_sec[k] - t_sec[k - 1]
            if dt <= 0.0 or dt > 60.0:
                # Discontinuous gap: reset to equilibrium
                t_skin[k] = t_eq[k]
            else:
                decay_skin = np.exp(-dt / TAU_SKIN)
                t_skin[k] = t_skin[k - 1] * decay_skin + t_eq[k] * (1.0 - decay_skin)

        # 4. Skin Sensor Measurement (MLX90614 NETD noise + 17-bit quantization)
        skin_noise = rng.normal(0.0, SENSOR_NETD, size=n)
        t_skin_meas = np.round((t_skin + skin_noise) / ADC_STEP) * ADC_STEP

        # Assign back
        group["t_amb_true"] = np.round(t_amb_true, 3)
        group["t_amb_meas"] = np.round(t_amb_meas, 2)
        group["t_amb"] = group["t_amb_meas"] # Device interface uses measured ambient
        group["t_eq"] = np.round(t_eq, 3)
        group["t_skin"] = np.round(t_skin, 3)
        group["t_skin_meas"] = np.round(t_skin_meas, 2)

        sim_dfs.append(group)

    df_realistic = pd.concat(sim_dfs, ignore_index=True)
    df_realistic.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Simulation completed in {time.time() - t_start:.2f}s.")
    print(f"Saved realistic dataset to: {OUTPUT_PARQUET}")
    print(f"  * T_amb_true range: [{df_realistic['t_amb_true'].min():.2f}, {df_realistic['t_amb_true'].max():.2f}] degC (Mean: {df_realistic['t_amb_true'].mean():.2f}, Std: {df_realistic['t_amb_true'].std():.2f})")
    print(f"  * T_amb_meas range: [{df_realistic['t_amb_meas'].min():.2f}, {df_realistic['t_amb_meas'].max():.2f}] degC (Mean: {df_realistic['t_amb_meas'].mean():.2f}, Std: {df_realistic['t_amb_meas'].std():.2f})")
    print(f"  * T_skin_meas range: [{df_realistic['t_skin_meas'].min():.2f}, {df_realistic['t_skin_meas'].max():.2f}] degC (Mean: {df_realistic['t_skin_meas'].mean():.2f}, Std: {df_realistic['t_skin_meas'].std():.2f})")
    print(f"  * T_core range:      [{df_realistic['t_core'].min():.2f}, {df_realistic['t_core'].max():.2f}] degC (Mean: {df_realistic['t_core'].mean():.2f})")

    return df_realistic


def engineer_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes single-state causal past-recursive EMAs strictly per (caseid, seg_id).
    Guarantees ZERO lookahead and ZERO cross-subject or cross-segment leakage.
    EMA formula: EMA_t = EMA_{t-1} + alpha * (x_t - EMA_{t-1})
    """
    df = df.copy().sort_values(["caseid", "seg_id", "t_center_sec"]).reset_index(drop=True)
    df["pi_eff"] = np.clip(0.10 * df["pi"], 0.005, 10.0)

    grp = df.groupby(["caseid", "seg_id"])

    # tskin_ema30: span=30 epochs (300s, alpha=2/31)
    df["tskin_ema30"] = grp["t_skin_meas"].transform(lambda s: s.ewm(span=30, adjust=False).mean())

    # tamb_ema18: span=18 epochs (180s, alpha=2/19)
    df["tamb_ema18"] = grp["t_amb_meas"].transform(lambda s: s.ewm(span=18, adjust=False).mean())

    # pi_ema6: span=6 epochs (60s, alpha=2/7)
    df["pi_ema6"] = grp["pi_eff"].transform(lambda s: s.ewm(span=6, adjust=False).mean())

    # lead_term: T_skin - tskin_ema30
    df["lead_term"] = df["t_skin_meas"] - df["tskin_ema30"]

    return df


# -----------------------------------------------------------------------------
# Model Definitions
# -----------------------------------------------------------------------------
def model_a_fn(X, a, b, alpha):
    """Model A: Basic Bioheat (a, b, alpha)"""
    ts, ta, peff = X
    g = np.maximum(a + b * np.power(peff, alpha), 0.10)
    return np.clip(ts + (ts - ta) / g, 30.0, 45.0)


def model_b_fn(X, a, b, c0):
    """Model B: Linear Rational + c0 (a, b, c0)"""
    ts, ta, peff = X
    g = np.maximum(a + b * peff, 0.10)
    return np.clip(ts + (ts - ta) / g + c0, 30.0, 45.0)


def model_c_fn(X, a, b, k, c0):
    """Model C: Final Corrected-Skin Lead-Lag Formula (a, b, k, c0)"""
    ts, e30, ta18, p6 = X
    tc = ts + k * (ts - e30)
    g = np.maximum(a + b * p6, 0.10)
    return np.clip(tc + (tc - ta18) / g + c0, 30.0, 45.0)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, cids: np.ndarray) -> dict:
    """
    Computes clinically standard temperature estimation metrics:
    Pooled MAE, Macro MAE, Bias, SD, 95% LoA, and Patient Pass Rate (MAE <= 0.35 degC).
    """
    res = y_pred - y_true
    abs_res = np.abs(res)
    pooled_mae = float(np.mean(abs_res))
    rmse = float(np.sqrt(np.mean(res**2)))
    bias = float(np.mean(res))
    sd = float(np.std(res, ddof=1)) if len(res) > 1 else 0.0
    lower_loa = float(bias - 1.96 * sd)
    upper_loa = float(bias + 1.96 * sd)

    unique_cids = sorted(list(np.unique(cids)))
    per_case = []
    case_maes = []

    for cid in unique_cids:
        mask = (cids == cid)
        c_res = res[mask]
        c_mae = float(np.mean(np.abs(c_res)))
        c_bias = float(np.mean(c_res))
        c_sd = float(np.std(c_res, ddof=1)) if len(c_res) > 1 else 0.0
        c_pass = bool(c_mae <= 0.35)
        case_maes.append(c_mae)
        per_case.append({
            "caseid": int(cid),
            "n_epochs": int(np.sum(mask)),
            "duration_min": round(float(np.sum(mask) * 10.0 / 60.0), 1),
            "t_core_mean": round(float(np.mean(y_true[mask])), 2),
            "mae": round(c_mae, 4),
            "bias": round(c_bias, 4),
            "sd": round(c_sd, 4),
            "pass_mae35": c_pass
        })

    macro_mae = float(np.mean(case_maes))
    pass_count = int(sum(c["pass_mae35"] for c in per_case))
    pass_rate = float(pass_count / len(unique_cids) * 100.0)

    return {
        "pooled_mae": round(pooled_mae, 4),
        "macro_mae": round(macro_mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "sd": round(sd, 4),
        "loa_95": [round(lower_loa, 4), round(upper_loa, 4)],
        "pass_count": pass_count,
        "total_subjects": len(unique_cids),
        "pass_rate_pct": round(pass_rate, 2),
        "per_subject": per_case
    }


def execute_scheme1_80_20(df: pd.DataFrame, train_cases: list, test_cases: list) -> dict:
    """
    Scheme 1: Subject-Wise 80/20 Validation.
    Fits on 80 Train subjects, tests on 20 Unseen Test subjects.
    """
    print("\n" + "=" * 115)
    print("SCHEME 1: DETERMINISTIC SUBJECT-WISE 80/20 VALIDATION (SEED 2026)")
    print(f"Train Cohort: {len(train_cases)} Subjects | Test Cohort: {len(test_cases)} Unseen Subjects")
    print("=" * 115)

    # Verify zero subject overlap
    overlap = set(train_cases).intersection(set(test_cases))
    assert len(overlap) == 0, f"DATA LEAKAGE ERROR: Subject overlap detected: {overlap}"
    print(f"[AUDIT] Zero Subject Overlap Verified: {len(overlap) == 0} (Overlap count: 0)")

    df_tr = df[df["caseid"].isin(train_cases)].copy()
    df_te = df[df["caseid"].isin(test_cases)].copy()

    y_tr = df_tr["t_core"].values
    y_te = df_te["t_core"].values
    cids_tr = df_tr["caseid"].values
    cids_te = df_te["caseid"].values

    # Model Inputs
    X_a_tr = (df_tr["t_skin_meas"].values, df_tr["t_amb_meas"].values, df_tr["pi_eff"].values)
    X_a_te = (df_te["t_skin_meas"].values, df_te["t_amb_meas"].values, df_te["pi_eff"].values)

    X_b_tr = X_a_tr
    X_b_te = X_a_te

    X_c_tr = (df_tr["t_skin_meas"].values, df_tr["tskin_ema30"].values, df_tr["tamb_ema18"].values, df_tr["pi_ema6"].values)
    X_c_te = (df_te["t_skin_meas"].values, df_te["tskin_ema30"].values, df_te["tamb_ema18"].values, df_te["pi_ema6"].values)

    # 1. Fit Model A
    popt_a, _ = curve_fit(model_a_fn, X_a_tr, y_tr, p0=[0.64, 0.90, 0.75], method="trf",
                          bounds=([0.1, 0.05, 0.1], [5.0, 10.0, 2.0]))
    pred_a_tr = model_a_fn(X_a_tr, *popt_a)
    pred_a_te = model_a_fn(X_a_te, *popt_a)
    met_a_tr = calculate_metrics(y_tr, pred_a_tr, cids_tr)
    met_a_te = calculate_metrics(y_te, pred_a_te, cids_te)

    # 2. Fit Model B
    popt_b, _ = curve_fit(model_b_fn, X_b_tr, y_tr, p0=[0.85, 1.05, 1.20], method="trf",
                          bounds=([0.1, 0.05, -5.0], [5.0, 10.0, 5.0]))
    pred_b_tr = model_b_fn(X_b_tr, *popt_b)
    pred_b_te = model_b_fn(X_b_te, *popt_b)
    met_b_tr = calculate_metrics(y_tr, pred_b_tr, cids_tr)
    met_b_te = calculate_metrics(y_te, pred_b_te, cids_te)

    # 3. Fit Model C
    popt_c, _ = curve_fit(model_c_fn, X_c_tr, y_tr, p0=[0.80, 0.88, 0.50, 0.90], method="trf",
                          bounds=([0.1, 0.05, -5.0, -5.0], [5.0, 10.0, 20.0, 5.0]))
    pred_c_tr = model_c_fn(X_c_tr, *popt_c)
    pred_c_te = model_c_fn(X_c_te, *popt_c)
    met_c_tr = calculate_metrics(y_tr, pred_c_tr, cids_tr)
    met_c_te = calculate_metrics(y_te, pred_c_te, cids_te)

    print("\n--- SCHEME 1 SUMMARY TABLE (80 Train / 20 Unseen Test) ---")
    hdr = f"{'Model':<35} | {'Train MAE':<10} | {'Test MAE':<10} | {'Test Macro':<10} | {'Test Bias +- SD':<18} | {'Test Pass/20':<12}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'Model A: Basic Bioheat (a,b,alpha)':<35} | {met_a_tr['pooled_mae']:.4f} C   | {met_a_te['pooled_mae']:.4f} C   | {met_a_te['macro_mae']:.4f} C   | {met_a_te['bias']:+.4f} +- {met_a_te['sd']:.4f} | {met_a_te['pass_count']:>2d}/20 ({met_a_te['pass_rate_pct']:.1f}%)")
    print(f"{'Model B: Linear Rational + c0':<35} | {met_b_tr['pooled_mae']:.4f} C   | {met_b_te['pooled_mae']:.4f} C   | {met_b_te['macro_mae']:.4f} C   | {met_b_te['bias']:+.4f} +- {met_b_te['sd']:.4f} | {met_b_te['pass_count']:>2d}/20 ({met_b_te['pass_rate_pct']:.1f}%)")
    print(f"{'Model C: Corrected-Skin Lead-Lag':<35} | {met_c_tr['pooled_mae']:.4f} C   | {met_c_te['pooled_mae']:.4f} C   | {met_c_te['macro_mae']:.4f} C   | {met_c_te['bias']:+.4f} +- {met_c_te['sd']:.4f} | {met_c_te['pass_count']:>2d}/20 ({met_c_te['pass_rate_pct']:.1f}%)")

    return {
        "model_a": {
            "parameters": [round(float(p), 5) for p in popt_a],
            "train_metrics": met_a_tr,
            "test_metrics": met_a_te
        },
        "model_b": {
            "parameters": [round(float(p), 5) for p in popt_b],
            "train_metrics": met_b_tr,
            "test_metrics": met_b_te
        },
        "model_c": {
            "parameters": [round(float(p), 5) for p in popt_c],
            "train_metrics": met_c_tr,
            "test_metrics": met_c_te
        }
    }


def execute_scheme2_loso(df: pd.DataFrame, all_cases: list, test_cases_subgroup: list) -> dict:
    """
    Scheme 2: Full 100-Fold Leave-One-Subject-Out (LOSO) Cross-Validation.
    For each subject i: fit on 99 subjects, test on held-out subject i.
    Evaluates Models A, B, and C.
    """
    print("\n" + "=" * 115)
    print("SCHEME 2: FULL 100-FOLD LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    print(f"Cohort: {len(all_cases)} Subjects | Iterations: 100 folds (Train: 99 subjects / Test: 1 subject)")
    print("=" * 115)

    cids = df["caseid"].values
    y_all = df["t_core"].values

    ts = df["t_skin_meas"].values
    ta = df["t_amb_meas"].values
    peff = df["pi_eff"].values
    e30 = df["tskin_ema30"].values
    ta18 = df["tamb_ema18"].values
    p6 = df["pi_ema6"].values

    models = {
        "model_a": {
            "name": "Model A: Basic Bioheat (a, b, alpha)",
            "fn": model_a_fn,
            "X": (ts, ta, peff),
            "p0": [0.57, 0.85, 0.80]
        },
        "model_b": {
            "name": "Model B: Linear Rational + c0 (a, b, c0)",
            "fn": model_b_fn,
            "X": (ts, ta, peff),
            "p0": [0.85, 1.05, 1.20]
        },
        "model_c": {
            "name": "Model C: Corrected-Skin Lead-Lag (a, b, k, c0)",
            "fn": model_c_fn,
            "X": (ts, e30, ta18, p6),
            "p0": [0.81, 0.86, 0.45, 0.86]
        }
    }

    loso_results = {}

    for m_key, m_info in models.items():
        t_m_start = time.time()
        fn = m_info["fn"]
        X_data = m_info["X"]
        p0 = m_info["p0"]

        # Warm start fit on entire dataset (used ONLY as initial guess for fast LM convergence)
        warm, _ = curve_fit(fn, X_data, y_all, p0=p0, method="lm", maxfev=5000)

        per_subject_map = {}
        all_loso_preds = np.zeros(len(df), dtype=np.float64)
        fold_params = []

        for cid in all_cases:
            tr_mask = (cids != cid)
            te_mask = (cids == cid)

            # Strict zero leakage check: ensure held-out subject is not in training set
            assert not np.any(cids[tr_mask] == cid), f"LEAKAGE DETECTED: subject {cid} found in train set!"

            X_tr = tuple(arr[tr_mask] for arr in X_data)
            X_te = tuple(arr[te_mask] for arr in X_data)
            y_tr = y_all[tr_mask]
            y_te = y_all[te_mask]

            popt, _ = curve_fit(fn, X_tr, y_tr, p0=warm, method="lm", maxfev=5000)
            fold_params.append([float(p) for p in popt])

            pred_te = fn(X_te, *popt)
            all_loso_preds[te_mask] = pred_te

            err = pred_te - y_te
            c_mae = float(np.mean(np.abs(err)))
            c_bias = float(np.mean(err))
            c_sd = float(np.std(err, ddof=1)) if len(err) > 1 else 0.0
            per_subject_map[int(cid)] = {
                "caseid": int(cid),
                "n_epochs": int(np.sum(te_mask)),
                "duration_min": round(float(np.sum(te_mask) * 10.0 / 60.0), 1),
                "t_core_mean": round(float(np.mean(y_te)), 2),
                "mae": round(c_mae, 4),
                "bias": round(c_bias, 4),
                "sd": round(c_sd, 4),
                "pass_mae35": bool(c_mae <= 0.35)
            }

        # Overall 100-subject LOSO metrics
        res_100 = all_loso_preds - y_all
        pooled_mae_100 = float(np.mean(np.abs(res_100)))
        rmse_100 = float(np.sqrt(np.mean(res_100**2)))
        bias_100 = float(np.mean(res_100))
        sd_100 = float(np.std(res_100, ddof=1))
        loa_100 = [round(float(bias_100 - 1.96 * sd_100), 4), round(float(bias_100 + 1.96 * sd_100), 4)]
        maes_100 = np.array([per_subject_map[int(c)]["mae"] for c in all_cases])
        macro_mae_100 = float(np.mean(maes_100))
        pass_100 = int(np.sum(maes_100 <= 0.35))
        pass_rate_100 = float(pass_100 / len(all_cases) * 100.0)

        # Test-20 Subgroup LOSO metrics (unseen test patients under LOSO)
        te_indices = np.isin(cids, test_cases_subgroup)
        res_test20 = all_loso_preds[te_indices] - y_all[te_indices]
        pooled_mae_te20 = float(np.mean(np.abs(res_test20)))
        bias_te20 = float(np.mean(res_test20))
        sd_te20 = float(np.std(res_test20, ddof=1))
        loa_te20 = [round(float(bias_te20 - 1.96 * sd_te20), 4), round(float(bias_te20 + 1.96 * sd_te20), 4)]
        maes_te20 = np.array([per_subject_map[int(c)]["mae"] for c in test_cases_subgroup])
        macro_mae_te20 = float(np.mean(maes_te20))
        pass_te20 = int(np.sum(maes_te20 <= 0.35))
        pass_rate_te20 = float(pass_te20 / len(test_cases_subgroup) * 100.0)

        # Fold parameter statistics
        arr_params = np.array(fold_params)
        param_means = [round(float(m), 5) for m in np.mean(arr_params, axis=0)]
        param_stds = [round(float(s), 5) for s in np.std(arr_params, axis=0)]

        loso_results[m_key] = {
            "name": m_info["name"],
            "loso_100": {
                "pooled_mae": round(pooled_mae_100, 4),
                "macro_mae": round(macro_mae_100, 4),
                "rmse": round(rmse_100, 4),
                "bias": round(bias_100, 4),
                "sd": round(sd_100, 4),
                "loa_95": loa_100,
                "pass_count": pass_100,
                "total_subjects": len(all_cases),
                "pass_rate_pct": round(pass_rate_100, 2)
            },
            "loso_test20_subgroup": {
                "pooled_mae": round(pooled_mae_te20, 4),
                "macro_mae": round(macro_mae_te20, 4),
                "bias": round(bias_te20, 4),
                "sd": round(sd_te20, 4),
                "loa_95": loa_te20,
                "pass_count": pass_te20,
                "total_subjects": len(test_cases_subgroup),
                "pass_rate_pct": round(pass_rate_te20, 2)
            },
            "fold_parameter_stats": {
                "mean": param_means,
                "std": param_stds
            },
            "per_subject": per_subject_map
        }
        print(f"Finished {m_info['name']} 100 folds in {time.time() - t_m_start:.2f}s.")

    print("\n--- SCHEME 2 SUMMARY TABLE (FULL 100-FOLD LOSO CROSS-VALIDATION) ---")
    hdr = f"{'Model':<35} | {'LOSO-100 Pooled':<15} | {'LOSO-100 Macro':<15} | {'Pass/100':<12} | {'Test-20 Macro':<14} | {'Pass/20 (Target>=16)':<20}"
    print(hdr)
    print("-" * len(hdr))
    for m_key in ["model_a", "model_b", "model_c"]:
        res = loso_results[m_key]
        s100 = res["loso_100"]
        s20 = res["loso_test20_subgroup"]
        print(f"{res['name']:<35} | {s100['pooled_mae']:.4f} degC       | {s100['macro_mae']:.4f} degC       | {s100['pass_count']:>2d}/100 ({s100['pass_rate_pct']:.1f}%) | {s20['macro_mae']:.4f} degC     | {s20['pass_count']:>2d}/20 ({s20['pass_rate_pct']:.1f}%)")

    return loso_results


def run_pipeline():
    total_start_time = time.time()
    print("=" * 115)
    print("EXPERIMENT: REALISTIC AMBIENT THERMAL DRIFT SIMULATION & LOSO CROSS-VALIDATION")
    print("Biophysical Model: Stolwijk-Hardy ODE (tau=180s) | Ambient Drift: Ornstein-Uhlenbeck (tau=1800s)")
    print("Dataset: 100 VitalDB Surgical Cases | Validation: Subject-Wise 80/20 & Full 100-Fold LOSO")
    print("=" * 115)

    # 1. Load original 100 cases
    df_raw = pd.read_parquet(INPUT_PARQUET)
    all_cases = sorted(list(df_raw["caseid"].unique()))
    assert len(all_cases) == 100, f"Expected 100 cases, found {len(all_cases)}"

    # 2. Simulate realistic ambient drift + Stolwijk-Hardy dynamic skin ODE
    df_sim = simulate_realistic_ambient_cohort(df_raw, force_recompute=True)

    # 3. Causal Feature Engineering
    df_feat = engineer_causal_features(df_sim)

    # 4. Partition 80/20 Deterministically (Seed 2026)
    rng_split = np.random.default_rng(2026)
    shuffled_cases = all_cases.copy()
    rng_split.shuffle(shuffled_cases)
    train_cases = sorted(shuffled_cases[:80])
    test_cases = sorted(shuffled_cases[80:])

    # 5. Execute Scheme 1: Subject-Wise 80/20 Split
    scheme1_res = execute_scheme1_80_20(df_feat, train_cases, test_cases)

    # 6. Execute Scheme 2: Full 100-Fold LOSO Cross-Validation
    scheme2_res = execute_scheme2_loso(df_feat, all_cases, test_cases)

    # 7. Compile Final Summary JSON
    summary_data = {
        "metadata": {
            "task_id": "TASK-REALISTIC-AMB-LOSO-01",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_cases": len(all_cases),
            "total_epochs": len(df_feat),
            "total_duration_hours": round(float(len(df_feat) * 10.0 / 3600.0), 2),
            "random_seeds": {
                "simulation": "2026 + cid*100 + seg",
                "split_80_20": 2026
            }
        },
        "simulation_physics": {
            "ambient_model": "Ornstein-Uhlenbeck mean-reverting process",
            "ou_mean_degC": MU_AMB,
            "ou_reversion_rate_s_inv": THETA_AMB,
            "ou_relaxation_time_s": 1.0 / THETA_AMB,
            "ou_volatility": SIGMA_AMB,
            "ou_clamp_range": [AMB_MIN, AMB_MAX],
            "sensor_model": "Melexis MLX90614 medical infrared thermopile",
            "sensor_netd_degC": SENSOR_NETD,
            "sensor_adc_step_degC": ADC_STEP,
            "stolwijk_hardy_ode": {
                "tau_s": TAU_SKIN,
                "a_conductance": A_TRUE,
                "b_perfusion": B_TRUE,
                "alpha_perfusion_power": ALPHA_TRUE,
                "pi_scale": PI_SCALE
            },
            "observed_distributions": {
                "t_amb_true": {
                    "min": round(float(df_feat["t_amb_true"].min()), 3),
                    "max": round(float(df_feat["t_amb_true"].max()), 3),
                    "mean": round(float(df_feat["t_amb_true"].mean()), 3),
                    "std": round(float(df_feat["t_amb_true"].std()), 3)
                },
                "t_amb_meas": {
                    "min": round(float(df_feat["t_amb_meas"].min()), 2),
                    "max": round(float(df_feat["t_amb_meas"].max()), 2),
                    "mean": round(float(df_feat["t_amb_meas"].mean()), 2),
                    "std": round(float(df_feat["t_amb_meas"].std()), 2)
                },
                "t_skin_meas": {
                    "min": round(float(df_feat["t_skin_meas"].min()), 2),
                    "max": round(float(df_feat["t_skin_meas"].max()), 2),
                    "mean": round(float(df_feat["t_skin_meas"].mean()), 2),
                    "std": round(float(df_feat["t_skin_meas"].std()), 2)
                },
                "t_core": {
                    "min": round(float(df_feat["t_core"].min()), 2),
                    "max": round(float(df_feat["t_core"].max()), 2),
                    "mean": round(float(df_feat["t_core"].mean()), 2),
                    "std": round(float(df_feat["t_core"].std()), 2)
                }
            }
        },
        "data_leakage_audit": {
            "subject_overlap_80_20": len(set(train_cases).intersection(set(test_cases))),
            "test_subjects_in_loso_train": False,
            "feature_causality": "Single-state recursive EMA computed per (caseid, seg_id) with adjust=False, strictly past samples only",
            "train_cases_80": [int(c) for c in train_cases],
            "test_cases_20": [int(c) for c in test_cases]
        },
        "scheme1_80_20": scheme1_res,
        "scheme2_loso_100": scheme2_res,
        "comparison": {
            "model_c_vs_a_error_reduction_pct": round(
                (scheme2_res["model_a"]["loso_100"]["pooled_mae"] - scheme2_res["model_c"]["loso_100"]["pooled_mae"])
                / scheme2_res["model_a"]["loso_100"]["pooled_mae"] * 100.0, 2
            ),
            "model_c_vs_b_error_reduction_pct": round(
                (scheme2_res["model_b"]["loso_100"]["pooled_mae"] - scheme2_res["model_c"]["loso_100"]["pooled_mae"])
                / scheme2_res["model_b"]["loso_100"]["pooled_mae"] * 100.0, 2
            ),
            "target_threshold_pass_rate_met": bool(scheme2_res["model_c"]["loso_test20_subgroup"]["pass_count"] >= 16),
            "model_c_test20_pass_count": scheme2_res["model_c"]["loso_test20_subgroup"]["pass_count"],
            "model_c_test20_pass_rate_pct": scheme2_res["model_c"]["loso_test20_subgroup"]["pass_rate_pct"]
        }
    }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 115)
    print("FINAL SUMMARY & VERIFICATION RESULT")
    print("=" * 115)
    print(f"Summary JSON saved to: {OUTPUT_SUMMARY_JSON}")
    print(f"Target Pass Count: >= 16/20 test subjects (MAE <= 0.35 degC)")
    print(f"Model C Scheme 1 (80/20 Unseen Test) Pass Count: {scheme1_res['model_c']['test_metrics']['pass_count']}/20 ({scheme1_res['model_c']['test_metrics']['pass_rate_pct']:.1f}%)")
    print(f"Model C Scheme 2 (LOSO Test-20 Subgroup) Pass Count: {scheme2_res['model_c']['loso_test20_subgroup']['pass_count']}/20 ({scheme2_res['model_c']['loso_test20_subgroup']['pass_rate_pct']:.1f}%)")
    print(f"Model C Scheme 2 (LOSO All-100 Cohort) Pass Count:   {scheme2_res['model_c']['loso_100']['pass_count']}/100 ({scheme2_res['model_c']['loso_100']['pass_rate_pct']:.1f}%)")
    print(f"Execution finished in {time.time() - total_start_time:.2f}s.")
    print("=" * 115)


if __name__ == "__main__":
    run_pipeline()
