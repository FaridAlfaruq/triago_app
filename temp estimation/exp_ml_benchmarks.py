"""
exp_ml_benchmarks.py
=============================================================================
TASK-ML-EXP-01: Machine Learning & Deep Learning Benchmarks for
Noninvasive Core Body Temperature (T_core) Estimation.

Cohort: 50 surgical patients from VitalDB (39,355 epochs).
Strict Split: 35 Train Patients, 15 Unseen Test Patients (Zero Leakage).
Features: Static Bioheat + Dynamic Thermal Lag & Perfusion Rate-of-Change.
Models:
  1. Analytical Bioheat Baseline (Stage 4 curve_fit)
  2. TriaGo Empirical Baseline (k_adapt * delta_T + C_offset)
  3. Regularized Ridge (Degree-2 Polynomial Interaction)
  4. Random Forest Regressor (n=100, max_depth=12)
  5. Extra Trees Regressor (n=100, max_depth=12)
  6. HistGradientBoosting Regressor (l2_reg=1.0, max_depth=6)
  7. Multi-Layer Perceptron Regressor (MLP: 64-32, ReLU, early stopping)

Author: Amin (ML & Deep Learning Implementation Specialist)
=============================================================================
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# -----------------------------------------------------------------------------
# 1. Configuration & Paths
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
PI_SCALE = 0.10  # Analytical bioheat scaling factor
CLIP_T_MIN = 30.0
CLIP_T_MAX = 45.0

BASE_DIR = Path(r"C:\Users\Adyty\Documents\Farid ITS\Tugas Akhir\scratch\temp estimation")
PARQUET_PATH = BASE_DIR / "cohort50_epochs.parquet"
EVAL_SUMMARY_PATH = BASE_DIR / "cohort50_evaluation_summary.json"
OUTPUT_JSON_PATH = BASE_DIR / "ml_benchmarks_summary.json"

FEATURE_COLS = [
    # Static Bioheat Features
    "t_skin_meas",
    "t_amb",
    "delta_T",
    "pi",
    "log_pi",
    "ac",
    "dc",
    "delta_T_x_pi",
    "delta_T_div_pi",
    # Dynamic Lag & Rate-of-Change Features
    "d_Tskin_dt",
    "d_deltaT_dt",
    "d_pi_dt",
    "rolling_mean_Tskin_60s",
    "rolling_mean_Tskin_180s",
    "lag_diff_180s",
]


# -----------------------------------------------------------------------------
# 2. Feature Engineering Pipeline (Zero Future Leakage)
# -----------------------------------------------------------------------------
def extract_bioheat_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Computes static and dynamic features strictly per patient and segment.

    Guarantees zero future leakage:
      - Rolling windows strictly causal (past window only).
      - Differencing bounded to segment boundaries.
    """
    df = df_raw.copy()

    # Static bioheat features
    df["delta_T"] = df["t_skin_meas"] - df["t_amb"]
    df["log_pi"] = np.log(np.maximum(df["pi"], 1e-4))
    df["delta_T_x_pi"] = df["delta_T"] * df["pi"]
    df["delta_T_div_pi"] = df["delta_T"] / (df["pi"] + 0.1)

    # Dynamic features computed strictly per (caseid, seg_id)
    dfs = []
    for (cid, sid), g in df.groupby(["caseid", "seg_id"], sort=False):
        g = g.sort_values("t_center_sec").copy()

        # Finite difference delta_t
        dt = g["t_center_sec"].diff().fillna(10.0).replace(0, 10.0)
        valid_dt = np.where((dt > 0) & (dt <= 60.0), dt, 10.0)

        # Rate of change features (°C/s, %/s)
        g["d_Tskin_dt"] = g["t_skin_meas"].diff().fillna(0.0) / valid_dt
        g["d_deltaT_dt"] = g["delta_T"].diff().fillna(0.0) / valid_dt
        g["d_pi_dt"] = g["pi"].diff().fillna(0.0) / valid_dt

        # Causal rolling windows by time
        s_time = pd.to_datetime(g["t_center_sec"], unit="s")
        tskin_s = pd.Series(g["t_skin_meas"].values, index=s_time)
        g["rolling_mean_Tskin_60s"] = tskin_s.rolling("60s", min_periods=1).mean().values
        g["rolling_mean_Tskin_180s"] = tskin_s.rolling("180s", min_periods=1).mean().values
        g["lag_diff_180s"] = g["t_skin_meas"].values - g["rolling_mean_Tskin_180s"].values

        dfs.append(g)

    df_out = pd.concat(dfs, ignore_index=True)
    return df_out


# -----------------------------------------------------------------------------
# 3. Comprehensive Evaluation Metrics
# -----------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> dict:
    """Computes clinical and statistical regression metrics."""
    res = y_pred - y_true
    n = len(res)
    mae = float(np.mean(np.abs(res)))
    rmse = float(np.sqrt(np.mean(res**2)))
    bias = float(np.mean(res))
    sd = float(np.std(res, ddof=1)) if n > 1 else 0.0
    lower_loa = float(bias - 1.96 * sd)
    upper_loa = float(bias + 1.96 * sd)

    acc_050 = float(np.mean(np.abs(res) <= 0.50) * 100.0)
    acc_035 = float(np.mean(np.abs(res) <= 0.35) * 100.0)
    acc_020 = float(np.mean(np.abs(res) <= 0.20) * 100.0)

    base_mae = float(np.mean(np.abs(y_base - y_true)))
    err_red = float((base_mae - mae) / base_mae * 100.0) if base_mae > 0 else 0.0

    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    r_val = float(np.corrcoef(y_true, y_pred)[0, 1]) if n > 1 and np.std(y_pred) > 0 else 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "sd": round(sd, 4),
        "lower_loa": round(lower_loa, 4),
        "upper_loa": round(upper_loa, 4),
        "acc_050": round(acc_050, 2),
        "acc_035": round(acc_035, 2),
        "acc_020": round(acc_020, 2),
        "base_mae": round(base_mae, 4),
        "err_red": round(err_red, 2),
        "r2": round(r2, 4),
        "pearson_r": round(r_val, 4),
    }


def evaluate_cohort(df_split: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Evaluates pooled, macro, and per-case metrics across a patient cohort."""
    y_pred_clipped = np.clip(y_pred, CLIP_T_MIN, CLIP_T_MAX)
    y_true = df_split["t_core"].values
    y_base = df_split["t_skin_meas"].values

    pooled = compute_metrics(y_true, y_pred_clipped, y_base)

    cases = sorted(df_split["caseid"].unique().tolist())
    per_case = []
    cases_passed_35 = 0
    cases_passed_clinical = 0

    for c in cases:
        mask = (df_split["caseid"] == c).values
        sub_true = y_true[mask]
        sub_pred = y_pred_clipped[mask]
        sub_base = y_base[mask]

        cm = compute_metrics(sub_true, sub_pred, sub_base)

        mae_ok = cm["mae"] <= 0.35
        bias_ok = abs(cm["bias"]) <= 0.20
        loa_ok = (cm["lower_loa"] >= -0.90) and (cm["upper_loa"] <= 0.90)
        r2_ok = cm["r2"] > 0.0
        clin_ok = mae_ok and bias_ok and loa_ok and r2_ok

        if mae_ok:
            cases_passed_35 += 1
        if clin_ok:
            cases_passed_clinical += 1

        per_case.append(
            {
                "caseid": int(c),
                "n_epochs": int(np.sum(mask)),
                "mae": cm["mae"],
                "rmse": cm["rmse"],
                "bias": cm["bias"],
                "sd": cm["sd"],
                "lower_loa": cm["lower_loa"],
                "upper_loa": cm["upper_loa"],
                "acc_050": cm["acc_050"],
                "acc_035": cm["acc_035"],
                "r2": cm["r2"],
                "pass_mae35": bool(mae_ok),
                "pass_clinical": bool(clin_ok),
            }
        )

    macro_mae = float(np.mean([p["mae"] for p in per_case]))
    macro_rmse = float(np.mean([p["rmse"] for p in per_case]))
    macro_bias = float(np.mean([p["bias"] for p in per_case]))
    macro_acc050 = float(np.mean([p["acc_050"] for p in per_case]))
    macro_acc035 = float(np.mean([p["acc_035"] for p in per_case]))

    n_cases = len(cases)
    pass_rate_35 = round((cases_passed_35 / n_cases) * 100.0, 1) if n_cases > 0 else 0.0
    pass_rate_clinical = (
        round((cases_passed_clinical / n_cases) * 100.0, 1) if n_cases > 0 else 0.0
    )

    return {
        "pooled": pooled,
        "macro": {
            "macro_mae": round(macro_mae, 4),
            "macro_rmse": round(macro_rmse, 4),
            "macro_bias": round(macro_bias, 4),
            "macro_acc_050": round(macro_acc050, 2),
            "macro_acc_035": round(macro_acc035, 2),
        },
        "case_pass_rate_mae35_pct": pass_rate_35,
        "cases_passed_mae35_count": cases_passed_35,
        "case_pass_rate_clinical_pct": pass_rate_clinical,
        "cases_passed_clinical_count": cases_passed_clinical,
        "per_case": per_case,
    }


# -----------------------------------------------------------------------------
# 4. Main Execution Function
# -----------------------------------------------------------------------------
def run_ml_benchmarks():
    t_start = time.time()
    print("=" * 115)
    print("TASK-ML-EXP-01: BENCHMARKING MACHINE LEARNING FOR CORE BODY TEMPERATURE ESTIMATION")
    print(f"Dataset: {PARQUET_PATH}")
    print("=" * 115)

    # 1. Load Data and Metadata
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet dataset not found at: {PARQUET_PATH}")
    if not EVAL_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Evaluation summary JSON not found at: {EVAL_SUMMARY_PATH}")

    df_raw = pd.read_parquet(PARQUET_PATH)
    with open(EVAL_SUMMARY_PATH, "r") as f:
        meta = json.load(f)

    train_patients = meta["train_patients"]
    test_patients = meta["test_patients"]

    # Enforce strict subject-level splitting
    overlap = set(train_patients).intersection(set(test_patients))
    assert len(overlap) == 0, f"FATAL: Patient leakage detected: {overlap}"
    print(
        f"Splits verified: {len(train_patients)} Train Patients | {len(test_patients)} Test Patients (Zero Overlap)"
    )

    # 2. Extract Features
    print("\n[1/4] Extracting Static Bioheat & Dynamic Causal Features...")
    df_feat = extract_bioheat_features(df_raw)
    print(f"  Feature set created: {len(FEATURE_COLS)} features")
    for col in FEATURE_COLS:
        print(f"    - {col}")

    # Partition by patient IDs
    df_train = df_feat[df_feat["caseid"].isin(train_patients)].copy().reset_index(drop=True)
    df_test = df_feat[df_feat["caseid"].isin(test_patients)].copy().reset_index(drop=True)
    print(
        f"\n  Cohort Partitioning: Train = {len(df_train):,} epochs | Test = {len(df_test):,} epochs"
    )

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["t_core"].values
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test["t_core"].values

    models_dict = {}
    train_preds = {}
    test_preds = {}
    fit_times = {}

    # -------------------------------------------------------------------------
    # Model 1: Analytical Bioheat Baseline (Stage 4 curve_fit)
    # -------------------------------------------------------------------------
    print("\n[2/4] Training Models on 35 Train Patients...")
    print("  Fitting Model 1: Baseline Bioheat (TRF curve_fit)...")
    t0 = time.time()

    def bioheat_func(X, a, b, alpha):
        ts, ta, p = X
        p_clipped = np.maximum(p, 0.05)
        p_eff = p_clipped * PI_SCALE
        denom = np.maximum(a + b * np.power(p_eff, alpha), 0.10)
        t_raw = ts + (ts - ta) / denom
        return np.clip(t_raw, CLIP_T_MIN, CLIP_T_MAX)

    X_tr_bio = (
        df_train["t_skin_meas"].values,
        df_train["t_amb"].values,
        df_train["pi"].values,
    )
    X_te_bio = (df_test["t_skin_meas"].values, df_test["t_amb"].values, df_test["pi"].values)

    popt_bio, pcov_bio = curve_fit(
        bioheat_func,
        X_tr_bio,
        y_train,
        p0=[0.574, 0.850, 0.800],
        bounds=([0.1, 0.05, 0.1], [5.0, 10.0, 3.0]),
        method="trf",
    )
    fit_times["1. Baseline Bioheat"] = time.time() - t0
    train_preds["1. Baseline Bioheat"] = bioheat_func(X_tr_bio, *popt_bio)
    test_preds["1. Baseline Bioheat"] = bioheat_func(X_te_bio, *popt_bio)
    models_dict["1. Baseline Bioheat"] = {
        "type": "Analytical Bioheat (curve_fit)",
        "params": {
            "a": float(popt_bio[0]),
            "b": float(popt_bio[1]),
            "alpha": float(popt_bio[2]),
        },
    }

    # -------------------------------------------------------------------------
    # Model 2: TriaGo Empirical Baseline
    # T_est = T_skin + k_adapt * delta_T + C_offset
    # -------------------------------------------------------------------------
    print("  Fitting Model 2: TriaGo Empirical Baseline (OLS)...")
    t0 = time.time()
    delta_T_tr = df_train["delta_T"].values.reshape(-1, 1)
    y_diff_tr = y_train - df_train["t_skin_meas"].values
    ols_triago = LinearRegression().fit(delta_T_tr, y_diff_tr)
    k_adapt = float(ols_triago.coef_[0])
    c_offset = float(ols_triago.intercept_)
    fit_times["2. TriaGo Empirical"] = time.time() - t0

    train_preds["2. TriaGo Empirical"] = (
        df_train["t_skin_meas"].values + k_adapt * df_train["delta_T"].values + c_offset
    )
    test_preds["2. TriaGo Empirical"] = (
        df_test["t_skin_meas"].values + k_adapt * df_test["delta_T"].values + c_offset
    )
    models_dict["2. TriaGo Empirical"] = {
        "type": "Linear Empirical (OLS)",
        "params": {"k_adapt": k_adapt, "c_offset": c_offset},
    }

    # -------------------------------------------------------------------------
    # Model 3: Regularized Ridge (Degree-2 Polynomial Interaction)
    # -------------------------------------------------------------------------
    print("  Fitting Model 3: Regularized Ridge (Poly Degree-2 Interaction)...")
    t0 = time.time()
    ridge_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("ridge", Ridge(alpha=50.0, random_state=RANDOM_SEED)),
        ]
    )
    ridge_pipe.fit(X_train, y_train)
    fit_times["3. Ridge (Poly Interaction)"] = time.time() - t0
    train_preds["3. Ridge (Poly Interaction)"] = ridge_pipe.predict(X_train)
    test_preds["3. Ridge (Poly Interaction)"] = ridge_pipe.predict(X_test)
    models_dict["3. Ridge (Poly Interaction)"] = {
        "type": "Ridge Regression with Degree-2 Interaction",
        "params": {"alpha": 50.0, "n_features_in": len(FEATURE_COLS)},
    }

    # -------------------------------------------------------------------------
    # Model 4: Random Forest Regressor
    # -------------------------------------------------------------------------
    print("  Fitting Model 4: Random Forest Regressor (n=100, max_depth=12)...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    fit_times["4. Random Forest"] = time.time() - t0
    train_preds["4. Random Forest"] = rf.predict(X_train)
    test_preds["4. Random Forest"] = rf.predict(X_test)
    models_dict["4. Random Forest"] = {
        "type": "RandomForestRegressor",
        "params": {"n_estimators": 100, "max_depth": 12, "random_state": RANDOM_SEED},
    }

    # -------------------------------------------------------------------------
    # Model 5: Extra Trees Regressor
    # -------------------------------------------------------------------------
    print("  Fitting Model 5: Extra Trees Regressor (n=100, max_depth=12)...")
    t0 = time.time()
    et = ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
    et.fit(X_train, y_train)
    fit_times["5. Extra Trees"] = time.time() - t0
    train_preds["5. Extra Trees"] = et.predict(X_train)
    test_preds["5. Extra Trees"] = et.predict(X_test)
    models_dict["5. Extra Trees"] = {
        "type": "ExtraTreesRegressor",
        "params": {"n_estimators": 100, "max_depth": 12, "random_state": RANDOM_SEED},
    }

    # -------------------------------------------------------------------------
    # Model 6: HistGradientBoosting Regressor
    # -------------------------------------------------------------------------
    print("  Fitting Model 6: HistGradientBoosting Regressor...")
    t0 = time.time()
    hgb = HistGradientBoostingRegressor(
        max_iter=150, max_depth=6, l2_regularization=1.0, random_state=RANDOM_SEED
    )
    hgb.fit(X_train, y_train)
    fit_times["6. HistGradientBoosting"] = time.time() - t0
    train_preds["6. HistGradientBoosting"] = hgb.predict(X_train)
    test_preds["6. HistGradientBoosting"] = hgb.predict(X_test)
    models_dict["6. HistGradientBoosting"] = {
        "type": "HistGradientBoostingRegressor",
        "params": {
            "max_iter": 150,
            "max_depth": 6,
            "l2_regularization": 1.0,
            "random_state": RANDOM_SEED,
        },
    }

    # -------------------------------------------------------------------------
    # Model 7: Multi-Layer Perceptron (MLPRegressor)
    # -------------------------------------------------------------------------
    print("  Fitting Model 7: MLPRegressor (64-32, ReLU, early stopping)...")
    t0 = time.time()
    mlp_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=250,
                    early_stopping=True,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    mlp_pipe.fit(X_train, y_train)
    fit_times["7. MLP Neural Net"] = time.time() - t0
    train_preds["7. MLP Neural Net"] = mlp_pipe.predict(X_train)
    test_preds["7. MLP Neural Net"] = mlp_pipe.predict(X_test)
    models_dict["7. MLP Neural Net"] = {
        "type": "MLPRegressor Pipeline",
        "params": {
            "hidden_layer_sizes": [64, 32],
            "activation": "relu",
            "early_stopping": True,
            "random_state": RANDOM_SEED,
        },
    }

    # -------------------------------------------------------------------------
    # 5. Evaluate All Models on Train and Test Sets
    # -------------------------------------------------------------------------
    print("\n[3/4] Evaluating Models on Train and Unseen Test Cohorts...")
    results_summary = {}

    for name in models_dict.keys():
        eval_train = evaluate_cohort(df_train, train_preds[name])
        eval_test = evaluate_cohort(df_test, test_preds[name])

        results_summary[name] = {
            "model_info": models_dict[name],
            "fit_time_sec": round(fit_times[name], 3),
            "train_metrics": {
                "pooled": eval_train["pooled"],
                "macro": eval_train["macro"],
                "case_pass_rate_mae35_pct": eval_train["case_pass_rate_mae35_pct"],
                "cases_passed_count": eval_train["cases_passed_mae35_count"],
            },
            "test_metrics": {
                "pooled": eval_test["pooled"],
                "macro": eval_test["macro"],
                "case_pass_rate_mae35_pct": eval_test["case_pass_rate_mae35_pct"],
                "cases_passed_mae35_count": eval_test["cases_passed_mae35_count"],
                "case_pass_rate_clinical_pct": eval_test["case_pass_rate_clinical_pct"],
                "cases_passed_clinical_count": eval_test["cases_passed_clinical_count"],
                "per_case": eval_test["per_case"],
            },
        }

    # -------------------------------------------------------------------------
    # 6. Print ASCII Comparison Tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("BENCHMARK RESULTS ON 15 UNSEEN TEST PATIENTS (9,761 Epochs | Zero Overlap)")
    print("=" * 115)

    header = f"{'Model':<30} | {'Pooled MAE':<10} | {'Macro MAE':<10} | {'Bias':<8} | {'SD':<7} | {'95% LoA':<17} | {'Pass Rate':<9} | {'Acc <=0.35C':<11} | {'Fit (s)':<7}"
    print(header)
    print("-" * 115)

    for name, r in results_summary.items():
        tm = r["test_metrics"]
        p = tm["pooled"]
        m = tm["macro"]
        loa_str = f"[{p['lower_loa']:+.2f}, {p['upper_loa']:+.2f}]"
        pass_str = f"{tm['cases_passed_mae35_count']}/15 ({tm['case_pass_rate_mae35_pct']:.0f}%)"
        print(
            f"{name:<30} | {p['mae']:<6.4f} °C | {m['macro_mae']:<6.4f} °C | {p['bias']:+6.4f} | {p['sd']:<7.4f} | {loa_str:<17} | {pass_str:<9} | {p['acc_035']:<5.1f}%     | {r['fit_time_sec']:<5.2f}s"
        )
    print("=" * 115)

    # Generalization (Overfitting) Table
    print("\n" + "-" * 85)
    print("GENERALIZATION CHECK: TRAIN VS TEST POOLED MAE (°C)")
    print("-" * 85)
    print(
        f"{'Model':<30} | {'Train MAE':<12} | {'Test MAE':<12} | {'Delta (Test-Train)':<18} | {'Verdict':<10}"
    )
    print("-" * 85)
    for name, r in results_summary.items():
        tr_mae = r["train_metrics"]["pooled"]["mae"]
        te_mae = r["test_metrics"]["pooled"]["mae"]
        delta = te_mae - tr_mae
        verdict = "EXCELLENT" if delta < 0.05 else ("GOOD" if delta < 0.15 else "OVERFIT")
        print(
            f"{name:<30} | {tr_mae:<10.4f} °C | {te_mae:<10.4f} °C | {delta:+12.4f} °C    | {verdict:<10}"
        )
    print("-" * 85)

    # Per-Case Breakdown Comparison
    print("\n" + "=" * 115)
    print("PER-CASE TEST PATIENT BREAKDOWN (MAE in °C across Models)")
    print("=" * 115)
    top_models = [
        "1. Baseline Bioheat",
        "2. TriaGo Empirical",
        "3. Ridge (Poly Interaction)",
        "6. HistGradientBoosting",
        "7. MLP Neural Net",
    ]
    sub_header = f"{'Case ID':<8} | {'Epochs':<7} | {'Bioheat':<9} | {'TriaGo':<9} | {'Ridge':<9} | {'HistGB':<9} | {'MLP':<9} | {'Best Model':<12}"
    print(sub_header)
    print("-" * 115)

    test_case_ids = [p["caseid"] for p in results_summary["1. Baseline Bioheat"]["test_metrics"]["per_case"]]
    for i, cid in enumerate(test_case_ids):
        epochs = results_summary["1. Baseline Bioheat"]["test_metrics"]["per_case"][i]["n_epochs"]
        maes = {}
        for m_name in top_models:
            maes[m_name] = results_summary[m_name]["test_metrics"]["per_case"][i]["mae"]

        best_m = min(maes.keys(), key=lambda k: maes[k]).split(". ")[1]
        print(
            f"Case {cid:<3d}  | {epochs:<7d} | {maes['1. Baseline Bioheat']:<7.4f} °C | {maes['2. TriaGo Empirical']:<7.4f} °C | {maes['3. Ridge (Poly Interaction)']:<7.4f} °C | {maes['6. HistGradientBoosting']:<7.4f} °C | {maes['7. MLP Neural Net']:<7.4f} °C | {best_m:<12}"
        )
    print("=" * 115)

    # -------------------------------------------------------------------------
    # 7. Save JSON Deliverable
    # -------------------------------------------------------------------------
    print("\n[4/4] Writing Deliverable Artifacts...")
    summary_deliverable = {
        "task_id": "TASK-ML-EXP-01",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": {
            "source": str(PARQUET_PATH),
            "n_train_cases": len(train_patients),
            "n_test_cases": len(test_patients),
            "train_epochs": len(df_train),
            "test_epochs": len(df_test),
            "train_patients": train_patients,
            "test_patients": test_patients,
        },
        "feature_set": {
            "count": len(FEATURE_COLS),
            "features": FEATURE_COLS,
            "thermal_lag_window_sec": 180,
            "rate_of_change_delta_t_guard_sec": 60.0,
        },
        "benchmarks": results_summary,
        "runtime_total_sec": round(time.time() - t_start, 2),
    }

    def json_cleaner(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(summary_deliverable, f, indent=2, default=json_cleaner)

    print(f"Summary deliverable saved to: {OUTPUT_JSON_PATH}")
    print(f"Total benchmark run completed in {time.time() - t_start:.2f} seconds.")


if __name__ == "__main__":
    run_ml_benchmarks()
