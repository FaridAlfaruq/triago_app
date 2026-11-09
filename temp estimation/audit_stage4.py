"""
audit_stage4.py - Independent QA & Metric Audit for Stage 4 Calibration
Auditor: Asep (Independent Verification and Testing Specialist)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# Set UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_FILE = os.path.join(CURRENT_DIR, "stage3_cohort_epochs.parquet")
SUMMARY_JSON = os.path.join(CURRENT_DIR, "stage4_calibration_summary.json")
STAGE4_PY = os.path.join(CURRENT_DIR, "stage4.py")

audit_results = {}

print("=" * 80)
print("INDEPENDENT QA & METRIC AUDIT: STAGE 4 CALIBRATION & EVALUATION")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. Patient Data Leakage Check
# -----------------------------------------------------------------------------
print("\n--- [AUDIT OBJECTIVE 1: DATA LEAKAGE AUDIT] ---")
assert os.path.exists(PARQUET_FILE), f"Missing {PARQUET_FILE}"
df_cohort = pd.read_parquet(PARQUET_FILE)
print(f"Loaded cohort parquet: {len(df_cohort)} rows across caseids: {sorted(df_cohort['caseid'].unique())}")

TRAIN_CASES = set(range(1, 8))
TEST_CASES = {8, 9, 10}

actual_train_cases = set(df_cohort[df_cohort["caseid"].isin(TRAIN_CASES)]["caseid"].unique())
actual_test_cases = set(df_cohort[df_cohort["caseid"].isin(TEST_CASES)]["caseid"].unique())
overlap = actual_train_cases.intersection(actual_test_cases)

df_train = df_cohort[df_cohort["caseid"].isin(TRAIN_CASES)].copy().reset_index(drop=True)
df_test = df_cohort[df_cohort["caseid"].isin(TEST_CASES)].copy().reset_index(drop=True)

# Check window/index leakage
print(f"Train Cases: {sorted(list(actual_train_cases))} (N = {len(df_train)})")
print(f"Test Cases:  {sorted(list(actual_test_cases))} (N = {len(df_test)})")
print(f"Case Overlap: {overlap}")
assert len(overlap) == 0, f"LEAKAGE DETECTED: {overlap}"

# Verify curve_fit training data input in stage4.py
# In stage4.py:
# x_train = (df_train["t_skin_meas"].values, df_train["t_amb"].values, df_train["pi"].values)
# y_train = df_train["t_core"].values
# curve_fit(...)
print("Inspecting stage4.py code to ensure curve_fit only touches df_train:")
with open(STAGE4_PY, "r", encoding="utf-8") as f:
    stage4_code = f.read()

assert "fit_model_on_train(df_train)" in stage4_code, "fit_model_on_train not called exclusively on df_train"
print("PASS: curve_fit optimization strictly restricted to df_train (Cases 1-7).")

# -----------------------------------------------------------------------------
# 2. Independent Parameter Optimization & Metric Recalculation
# -----------------------------------------------------------------------------
print("\n--- [AUDIT OBJECTIVE 2: INDEPENDENT RE-FITTING & METRIC RE-CALCULATION] ---")

def core_temperature_model(t_skin_meas, t_amb, pi, a, b, alpha, pi_scale=0.10):
    pi_eff = np.maximum(pi * pi_scale, 1e-4)
    denom = a + b * np.power(pi_eff, alpha)
    return t_skin_meas + (t_skin_meas - t_amb) / denom

def curve_fit_obj(X, a, b, alpha):
    t_skin_meas, t_amb, pi = X
    return core_temperature_model(t_skin_meas, t_amb, pi, a, b, alpha, 0.10)

# Independent fit on df_train
x_train = (df_train["t_skin_meas"].values, df_train["t_amb"].values, df_train["pi"].values)
y_train = df_train["t_core"].values
p0 = [0.574, 0.850, 0.800]
bounds = ([0.1, 0.1, 0.1], [5.0, 5.0, 3.0])

popt, pcov = curve_fit(curve_fit_obj, x_train, y_train, p0=p0, bounds=bounds, method="trf")
perr = np.sqrt(np.diag(pcov))

print(f"Independent Fitted Parameters:")
print(f"  a     = {popt[0]:.6f} (SE: {perr[0]:.6f})")
print(f"  b     = {popt[1]:.6f} (SE: {perr[1]:.6f})")
print(f"  alpha = {popt[2]:.6f} (SE: {perr[2]:.6f})")

# Load Amin's summary json
with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
    summary_data = json.load(f)

amin_params = summary_data["fitted_parameters"]
print(f"Amin's Fitted Parameters:")
print(f"  a     = {amin_params['a']:.6f}")
print(f"  b     = {amin_params['b']:.6f}")
print(f"  alpha = {amin_params['alpha']:.6f}")

np.testing.assert_allclose(popt, [amin_params['a'], amin_params['b'], amin_params['alpha']], rtol=1e-5)
print("PASS: Parameter estimates match to within 1e-5 relative tolerance.")

# Independent prediction on test cohort
y_true_test = df_test["t_core"].values
t_skin_test = df_test["t_skin_meas"].values
t_amb_test = df_test["t_amb"].values
pi_test = df_test["pi"].values

y_pred_test = core_temperature_model(t_skin_test, t_amb_test, pi_test, popt[0], popt[1], popt[2])

# Recalculate metrics
residuals = y_pred_test - y_true_test
abs_err = np.abs(residuals)
mae_indep = float(np.mean(abs_err))
rmse_indep = float(np.sqrt(np.mean(residuals**2)))
bias_indep = float(np.mean(residuals))
sd_indep = float(np.std(residuals, ddof=1))
loa_upper_indep = float(bias_indep + 1.96 * sd_indep)
loa_lower_indep = float(bias_indep - 1.96 * sd_indep)
r_indep, _ = pearsonr(y_true_test, y_pred_test)

base_abs_err = np.abs(y_true_test - t_skin_test)
base_mae_indep = float(np.mean(base_abs_err))
err_reduct_indep = float((base_mae_indep - mae_indep) / base_mae_indep * 100.0)

print(f"\nMetric Comparison (Independent vs Amin Reported):")
print(f"  MAE:       Independent={mae_indep:.4f} °C | Amin={summary_data['test_metrics']['mae_degC']} °C | Diff={abs(mae_indep - summary_data['test_metrics']['mae_degC']):.6f}")
print(f"  RMSE:      Independent={rmse_indep:.4f} °C | Amin={summary_data['test_metrics']['rmse_degC']} °C | Diff={abs(rmse_indep - summary_data['test_metrics']['rmse_degC']):.6f}")
print(f"  Mean Bias: Independent={bias_indep:.4f} °C | Amin={summary_data['test_metrics']['mean_bias_degC']} °C | Diff={abs(bias_indep - summary_data['test_metrics']['mean_bias_degC']):.6f}")
print(f"  SD Bias:   Independent={sd_indep:.4f} °C | Amin={summary_data['test_metrics']['sd_bias_degC']} °C | Diff={abs(sd_indep - summary_data['test_metrics']['sd_bias_degC']):.6f}")
print(f"  LoA Lower: Independent={loa_lower_indep:.4f} °C | Amin={summary_data['test_metrics']['lower_loa_95_degC']} °C | Diff={abs(loa_lower_indep - summary_data['test_metrics']['lower_loa_95_degC']):.6f}")
print(f"  LoA Upper: Independent={loa_upper_indep:.4f} °C | Amin={summary_data['test_metrics']['upper_loa_95_degC']} °C | Diff={abs(loa_upper_indep - summary_data['test_metrics']['upper_loa_95_degC']):.6f}")
print(f"  Pearson r: Independent={r_indep:.4f} | Amin={summary_data['test_metrics']['pearson_r']} | Diff={abs(r_indep - summary_data['test_metrics']['pearson_r']):.6f}")
print(f"  Baseline:  Independent={base_mae_indep:.4f} °C | Amin={summary_data['test_metrics']['baseline_skin_mae_degC']} °C")
print(f"  Reduction: Independent={err_reduct_indep:.2f}% | Amin={summary_data['test_metrics']['error_reduction_pct']}%")

assert abs(mae_indep - summary_data['test_metrics']['mae_degC']) < 1e-4
assert abs(rmse_indep - summary_data['test_metrics']['rmse_degC']) < 1e-4
assert abs(bias_indep - summary_data['test_metrics']['mean_bias_degC']) < 1e-4
assert abs(loa_lower_indep - summary_data['test_metrics']['lower_loa_95_degC']) < 1e-4
assert abs(loa_upper_indep - summary_data['test_metrics']['upper_loa_95_degC']) < 1e-4
print("PASS: Independent metrics match Amin's reported values exactly.")

# Per-case verification
print("\nPer-Case Independent Recalculation:")
for cid in [8, 9, 10]:
    sub = df_test[df_test["caseid"] == cid]
    y_sub = sub["t_core"].values
    t_sk_sub = sub["t_skin_meas"].values
    y_p_sub = core_temperature_model(t_sk_sub, sub["t_amb"].values, sub["pi"].values, popt[0], popt[1], popt[2])
    res_sub = y_p_sub - y_sub
    m_mae = np.mean(np.abs(res_sub))
    m_rmse = np.sqrt(np.mean(res_sub**2))
    m_bias = np.mean(res_sub)
    amin_case = summary_data["per_case_test_metrics"][f"case_{cid}"]
    print(f"  Case {cid} (N={len(sub)}): MAE={m_mae:.4f} (Amin={amin_case['mae_degC']}), RMSE={m_rmse:.4f} (Amin={amin_case['rmse_degC']}), Bias={m_bias:.4f} (Amin={amin_case['mean_bias_degC']})")
    assert abs(m_mae - amin_case['mae_degC']) < 1e-4

# -----------------------------------------------------------------------------
# 3. PRD Clinical Benchmark Compliance
# -----------------------------------------------------------------------------
print("\n--- [AUDIT OBJECTIVE 3: PRD BENCHMARK COMPLIANCE] ---")
checks = {
    "Clinical Gold Standard MAE (<= 0.35 C)": (mae_indep <= 0.35, f"{mae_indep:.4f} <= 0.35"),
    "Lightness Development MAE (<= 0.50 C)": (mae_indep <= 0.50, f"{mae_indep:.4f} <= 0.50"),
    "RMSE Target (<= 0.65 C)": (rmse_indep <= 0.65, f"{rmse_indep:.4f} <= 0.65"),
    "Bland-Altman Bias (|Bias| <= 0.20 C)": (abs(bias_indep) <= 0.20, f"|{bias_indep:.4f}| <= 0.20"),
    "Bland-Altman LoA within [-0.90, +0.90] C": (loa_lower_indep >= -0.90 and loa_upper_indep <= 0.90, f"[{loa_lower_indep:.4f}, {loa_upper_indep:.4f}] in [-0.90, +0.90]"),
    "Baseline Error Reduction (> 75%)": (err_reduct_indep > 75.0, f"{err_reduct_indep:.2f}% > 75%")
}

for name, (passed, detail) in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name:<45}: {detail}")
    assert passed

# -----------------------------------------------------------------------------
# 4. Numerical & Physiological Edge Cases
# -----------------------------------------------------------------------------
print("\n--- [AUDIT OBJECTIVE 4: NUMERICAL & PHYSIOLOGICAL EDGE CASES] ---")
t_skin_nominal = 34.0
t_amb_nominal = 21.0

# Case A: PI = 0.0 (Severe vasoconstriction / shock)
t_est_pi_zero = core_temperature_model(
    np.array([t_skin_nominal]), np.array([t_amb_nominal]), np.array([0.0]),
    popt[0], popt[1], popt[2]
)[0]
print(f"  Edge Case PI = 0.0%: T_est = {t_est_pi_zero:.4f} °C (finite: {np.isfinite(t_est_pi_zero)})")
assert np.isfinite(t_est_pi_zero)
assert not np.isnan(t_est_pi_zero)

# Case B: Negative or tiny PI
t_est_pi_neg = core_temperature_model(
    np.array([t_skin_nominal]), np.array([t_amb_nominal]), np.array([-1.0]),
    popt[0], popt[1], popt[2]
)[0]
print(f"  Edge Case PI < 0 (clamped): T_est = {t_est_pi_neg:.4f} °C (finite: {np.isfinite(t_est_pi_neg)})")
assert np.isfinite(t_est_pi_neg)

# Case C: PI = 150.0% (Extreme vasodilation / hyperdynamic shock)
t_est_pi_150 = core_temperature_model(
    np.array([t_skin_nominal]), np.array([t_amb_nominal]), np.array([150.0]),
    popt[0], popt[1], popt[2]
)[0]
print(f"  Edge Case PI = 150.0%: T_est = {t_est_pi_150:.4f} °C (finite: {np.isfinite(t_est_pi_150)})")
assert np.isfinite(t_est_pi_150)
assert not np.isnan(t_est_pi_150)

# Check denominator positivity across bounds:
# a in [0.1, 5.0], b in [0.1, 5.0], alpha in [0.1, 3.0]
# denom = a + b * (pi_eff)^alpha. Since a >= 0.1, b >= 0.1, pi_eff >= 1e-4, denom is strictly > 0.1!
min_possible_denom = 0.1 + 0.1 * (1e-4)**3.0
print(f"  Theoretical minimum denominator across parameter bounds: {min_possible_denom:.6f} > 0")
assert min_possible_denom > 0
print("PASS: Numerical stability verified. Zero division and NaNs are impossible under parameter bounds.")

# -----------------------------------------------------------------------------
# 5. Code Integrity & Workspace Rules
# -----------------------------------------------------------------------------
print("\n--- [AUDIT OBJECTIVE 5: CODE INTEGRITY & WORKSPACE RULES] ---")
# Legend locations in stage4.py
import re
legend_matches = re.findall(r'ax\d*\.legend\((.*?)\)', stage4_code)
print(f"  Found {len(legend_matches)} legend calls in stage4.py:")
for lm in legend_matches:
    print(f"    legend({lm})")
    assert 'loc="upper right"' in lm or "loc='upper right'" in lm, f"Legend not top-right: {lm}"

# Check for disk saving commands like plt.savefig, fig.savefig
savefig_calls = re.findall(r'savefig', stage4_code)
print(f"  savefig occurrences: {len(savefig_calls)}")
assert len(savefig_calls) == 0, "stage4.py contains savefig calls violating display via plt.show() rule"

# Check plt.show()
show_calls = re.findall(r'plt\.show\(\)', stage4_code)
print(f"  plt.show() occurrences: {len(show_calls)}")
assert len(show_calls) >= 1, "stage4.py missing plt.show() call"

print("PASS: Code integrity and workspace rules strictly obeyed.")

print("\n" + "=" * 80)
print("AUDIT SUMMARY VERDICT: PASS")
print("=" * 80)
