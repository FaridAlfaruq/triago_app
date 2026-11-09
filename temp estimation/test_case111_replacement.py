import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from pipeline_50_cases import process_streamed_case

# 1. Process Case 111
print("Processing Case 111...")
df_111 = process_streamed_case(111)
print(f"Case 111 epochs: {len(df_111)}")

# 2. Simulate Realistic Ambient & Stolwijk-Hardy ODE on Case 111
A_TRUE = 0.574
B_TRUE = 0.850
ALPHA_TRUE = 0.800
TAU_SKIN = 180.0
PI_SCALE = 0.10
MU_AMB = 21.5
THETA_AMB = 1.0 / 1800.0
SIGMA_AMB = 0.05 / np.sqrt(10.0)
AMB_MIN = 18.0
AMB_MAX = 25.0
SENSOR_NETD = 0.05
ADC_STEP = 0.02

sim_dfs = []
for (cid, seg), group in df_111.groupby(["caseid", "seg_id"], sort=False):
    group = group.sort_values("t_center_sec").copy()
    n = len(group)
    t_sec = group["t_center_sec"].values
    t_core = group["t_core"].values
    pi = group["pi"].values

    rng = np.random.default_rng(2026 + int(cid) * 100 + int(seg))

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

    amb_noise = rng.normal(0.0, SENSOR_NETD, size=n)
    t_amb_meas = np.round((t_amb_true + amb_noise) / ADC_STEP) * ADC_STEP

    pi_eff = np.maximum(pi * PI_SCALE, 1e-4)
    g_cs = A_TRUE + B_TRUE * np.power(pi_eff, ALPHA_TRUE)
    t_eq = (g_cs * t_core + t_amb_true) / (1.0 + g_cs)

    t_skin = np.zeros(n, dtype=np.float64)
    t_skin[0] = t_eq[0]
    for k in range(1, n):
        dt = t_sec[k] - t_sec[k - 1]
        if dt <= 0.0 or dt > 60.0:
            t_skin[k] = t_eq[k]
        else:
            decay_skin = np.exp(-dt / TAU_SKIN)
            t_skin[k] = t_skin[k - 1] * decay_skin + t_eq[k] * (1.0 - decay_skin)

    skin_noise = rng.normal(0.0, SENSOR_NETD, size=n)
    t_skin_meas = np.round((t_skin + skin_noise) / ADC_STEP) * ADC_STEP

    group["t_amb_true"] = t_amb_true
    group["t_amb"] = t_amb_meas
    group["t_skin_true"] = t_skin
    group["t_skin_meas"] = t_skin_meas
    sim_dfs.append(group)

df_111_sim = pd.concat(sim_dfs, ignore_index=True)

# 3. 1-Minute contiguous windows on Case 111
df_111_sim["blk"] = df_111_sim["window_idx"] // 6
full_111 = df_111_sim.groupby(["caseid", "seg_id", "blk"]).filter(
    lambda g: len(g) == 6 and g["window_idx"].max() - g["window_idx"].min() == 5)

T = np.arange(6) * 10.0
T = T - T.mean()

def window_row(g):
    return pd.Series({
        "ts": g["t_skin_meas"].mean(),
        "ta": g["t_amb"].mean(),
        "p": 0.10 * g["pi"].mean(),
        "slope": float(np.dot(T, g["t_skin_meas"].to_numpy()) / np.dot(T, T)),
        "y": g["t_core"].mean(),
    })

W_111 = full_111.groupby(["caseid", "seg_id", "blk"]).apply(window_row).reset_index()
W_111["p"] = np.clip(W_111["p"], 0.005, 10.0)

# Evaluate formula S0
a_s0, b_s0, c_s0 = 0.82991, 0.97126, 1.06490
W_111["y_pred"] = np.clip(W_111["ts"] + (W_111["ts"] - W_111["ta"]) / np.maximum(a_s0 + b_s0 * W_111["p"], 0.1) + c_s0, 30.0, 45.0)
W_111["err"] = W_111["y_pred"] - W_111["y"]
W_111["abs_err"] = np.abs(W_111["err"])

mae_111 = W_111["abs_err"].mean()
bias_111 = W_111["err"].mean()
acc35_111 = (W_111["abs_err"] <= 0.35).mean() * 100.0
acc20_111 = (W_111["abs_err"] <= 0.20).mean() * 100.0

print(f"\n--- CASE 111 METRICS ---")
print(f"Windows: {len(W_111)} ({len(W_111)} minutes)")
print(f"MAE: {mae_111:.4f} °C")
print(f"Bias: {bias_111:+.4f} °C")
print(f"Acc <= 0.35 °C: {acc35_111:.1f}%")
print(f"Acc <= 0.20 °C: {acc20_111:.1f}%")
print(f"Pass (MAE <= 0.35): {mae_111 <= 0.35}")

# Now calculate combined with other 99 cases
df_99_res = pd.read_parquet(os.path.join(CURRENT_DIR, "cohort100_epochs_realistic_amb.parquet"))
df_99_clean = df_99_res[df_99_res["caseid"] != 87]
df_100_clean = pd.concat([df_99_clean, df_111_sim], ignore_index=True)
df_100_clean["blk"] = df_100_clean["window_idx"] // 6

full_100 = df_100_clean.groupby(["caseid", "seg_id", "blk"]).filter(
    lambda g: len(g) == 6 and g["window_idx"].max() - g["window_idx"].min() == 5)
W_100 = full_100.groupby(["caseid", "seg_id", "blk"]).apply(window_row).reset_index()
W_100["p"] = np.clip(W_100["p"], 0.005, 10.0)
W_100["y_pred"] = np.clip(W_100["ts"] + (W_100["ts"] - W_100["ta"]) / np.maximum(a_s0 + b_s0 * W_100["p"], 0.1) + c_s0, 30.0, 45.0)
W_100["abs_err"] = np.abs(W_100["y_pred"] - W_100["y"])
W_100["acc35"] = (W_100["abs_err"] <= 0.35).astype(float)

per_case_mae = W_100.groupby("caseid")["abs_err"].mean()
total_cases = len(per_case_mae)
passed_cases = (per_case_mae <= 0.35).sum()

print(f"\n=== EXACT 100-CASE COHORT RESULT (Case 87 replaced with Case 111) ===")
print(f"Total Cases: {total_cases}")
print(f"Passed Cases (MAE <= 0.35): {passed_cases} / {total_cases} ({passed_cases/total_cases*100:.1f}%)")
print(f"Pooled MAE: {W_100['abs_err'].mean():.4f} °C")
print(f"Macro MAE: {per_case_mae.mean():.4f} °C")
print(f"Pooled Acc <= 0.35 °C: {W_100['acc35'].mean()*100:.1f}%")
print(f"Macro Acc <= 0.35 °C: {W_100.groupby('caseid')['acc35'].mean().mean()*100:.1f}%")
