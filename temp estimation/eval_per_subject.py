import os
import json
import numpy as np
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(DIR, "cohort100_epochs_realistic_amb.parquet"))
df = df.sort_values(["caseid", "seg_id", "window_idx"]).reset_index(drop=True)
df["blk"] = df["window_idx"] // 6

# Canonical split
cases = sorted(df["caseid"].unique())
rng = np.random.default_rng(2026)
sh = list(cases)
rng.shuffle(sh)
TRAIN, TEST = sorted(sh[:80]), sorted(sh[80:])

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

full = df.groupby(["caseid", "seg_id", "blk"]).filter(
    lambda g: len(g) == 6 and g["window_idx"].max() - g["window_idx"].min() == 5)
W = full.groupby(["caseid", "seg_id", "blk"]).apply(window_row).reset_index()
W["p"] = np.clip(W["p"], 0.005, 10.0)

# Formula S0 parameters from fixed 80/20 fit:
# a = 0.82991, b = 0.97126, c = 1.06490
a_s0, b_s0, c_s0 = 0.82991, 0.97126, 1.06490
W["y_pred_s0"] = np.clip(W["ts"] + (W["ts"] - W["ta"]) / np.maximum(a_s0 + b_s0 * W["p"], 0.1) + c_s0, 30.0, 45.0)
W["err_s0"] = W["y_pred_s0"] - W["y"]
W["abs_err_s0"] = np.abs(W["err_s0"])
W["acc_35_s0"] = (W["abs_err_s0"] <= 0.35).astype(float)
W["acc_20_s0"] = (W["abs_err_s0"] <= 0.20).astype(float)

# Compute per-case statistics
res = []
for cid, g in W.groupby("caseid"):
    is_test = cid in TEST
    mae = g["abs_err_s0"].mean()
    bias = g["err_s0"].mean()
    sd = g["err_s0"].std(ddof=1) if len(g) > 1 else 0.0
    acc35 = g["acc_35_s0"].mean() * 100.0
    acc20 = g["acc_20_s0"].mean() * 100.0
    n_win = len(g)
    dur_min = n_win * 1.0
    t_core_mean = g["y"].mean()
    t_skin_mean = g["ts"].mean()
    t_amb_mean = g["ta"].mean()
    pi_mean = g["p"].mean() / 0.10
    passed = mae <= 0.35
    res.append({
        "caseid": int(cid),
        "split": "TEST" if is_test else "TRAIN",
        "n_windows": int(n_win),
        "dur_min": float(dur_min),
        "t_core_mean": float(t_core_mean),
        "t_skin_mean": float(t_skin_mean),
        "t_amb_mean": float(t_amb_mean),
        "pi_raw_mean": float(pi_mean),
        "mae": float(mae),
        "bias": float(bias),
        "sd": float(sd),
        "acc_35": float(acc35),
        "acc_20": float(acc20),
        "pass": bool(passed)
    })

df_res = pd.DataFrame(res)

print("=== 20 UNSEEN TEST CASES (Fixed 80/20) ===")
test_df = df_res[df_res["split"] == "TEST"].sort_values("mae").reset_index(drop=True)
print(test_df[["caseid", "dur_min", "t_core_mean", "mae", "bias", "acc_35", "acc_20", "pass"]].to_string(index=False))

print("\nTEST SUMMARY:")
print(f"Total Test Cases: {len(test_df)}")
print(f"Pass Rate (MAE <= 0.35): {test_df['pass'].sum()} / {len(test_df)} ({test_df['pass'].mean()*100:.1f}%)")
print(f"Pooled MAE: {W[W['caseid'].isin(TEST)]['abs_err_s0'].mean():.4f}")
print(f"Macro MAE: {test_df['mae'].mean():.4f}")
print(f"Pooled Acc <= 0.35 C: {W[W['caseid'].isin(TEST)]['acc_35_s0'].mean()*100:.1f}%")
print(f"Macro Acc <= 0.35 C: {test_df['acc_35'].mean():.1f}%")
print(f"Pooled Acc <= 0.20 C: {W[W['caseid'].isin(TEST)]['acc_20_s0'].mean()*100:.1f}%")

print("\n=== ALL 99 CASES SUMMARY (Fixed Model) ===")
print(f"Total Cases: {len(df_res)}")
print(f"Pass Rate (MAE <= 0.35): {df_res['pass'].sum()} / {len(df_res)} ({df_res['pass'].mean()*100:.1f}%)")
print(f"Pooled MAE: {W['abs_err_s0'].mean():.4f}")
print(f"Macro MAE: {df_res['mae'].mean():.4f}")
print(f"Pooled Acc <= 0.35 C: {W['acc_35_s0'].mean()*100:.1f}%")
print(f"Macro Acc <= 0.35 C: {df_res['acc_35'].mean():.1f}%")

# Distribution tiers across all 99 cases
print("\nTIER DISTRIBUTION (All 99 cases):")
t1 = (df_res['mae'] <= 0.20).sum()
t2 = ((df_res['mae'] > 0.20) & (df_res['mae'] <= 0.35)).sum()
t3 = (df_res['mae'] > 0.35).sum()
print(f"MAE <= 0.20 C (Excellent): {t1} cases ({t1/len(df_res)*100:.1f}%)")
print(f"0.20 < MAE <= 0.35 C (Good/Pass): {t2} cases ({t2/len(df_res)*100:.1f}%)")
print(f"MAE > 0.35 C (Fail/Extreme): {t3} cases ({t3/len(df_res)*100:.1f}%)")

# Failed cases analysis
failed = df_res[~df_res["pass"]].sort_values("mae", ascending=False)
print("\nTop 5 Failed Cases:")
print(failed[["caseid", "split", "dur_min", "t_core_mean", "t_skin_mean", "t_amb_mean", "pi_raw_mean", "mae", "bias"]].head(10).to_string(index=False))

# Save full results json
with open(os.path.join(DIR, "per_subject_1min_s0.json"), "w") as f:
    json.dump(res, f, indent=2)
