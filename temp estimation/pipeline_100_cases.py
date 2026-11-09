"""
pipeline_100_cases.py - End-to-End Pipeline & Evaluation on 100 VitalDB Surgical Cases
Evaluates:
1. Scaling up from 50 cases to 100 surgical cases from VitalDB.
2. Combines cached 50 cases (cohort50_epochs.parquet) with 50 newly streamed & validated cases.
3. Feature engineering: causal past rolling averages (60s, 180s), rate-of-change, effective PI.
4. Evaluation of:
   - Basic Biophysical Bioheat Formula (a, b, alpha)
   - Linearized Bioheat Ridge Model (Fixed Weights: w0=14.307376, w1=1.189176, w2=0.644180, w3=-0.591838, w4=-0.904093)
   - TriaGo Empirical Baseline
5. Generalization metrics across 100 cases and strict 70/30 train/test split.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import streamer from local pipeline_50_cases
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from pipeline_50_cases import process_streamed_case

CACHE_50_PARQUET = os.path.join(CURRENT_DIR, "cohort50_epochs.parquet")
CACHE_100_PARQUET = os.path.join(CURRENT_DIR, "cohort100_epochs.parquet")
SUMMARY_100_JSON = os.path.join(CURRENT_DIR, "cohort100_evaluation_summary.json")

TARGET_CASES_TOTAL = 100
TARGET_NEW_CASES = 50

# Fixed parameters established previously
A_FIXED = 0.641820
B_FIXED = 0.907050
ALPHA_FIXED = 0.747250

W0_FIXED = 14.307376
W1_FIXED =  1.189176
W2_FIXED =  0.644180
W3_FIXED = -0.591838
W4_FIXED = -0.904093

def load_or_create_100_cases():
    if os.path.exists(CACHE_100_PARQUET):
        print(f"[CACHE] Loading cached 100-case dataset from: {os.path.basename(CACHE_100_PARQUET)}")
        df_100 = pd.read_parquet(CACHE_100_PARQUET)
        return df_100

    print(f"[COLLECTION] Loading existing 50 cases from: {os.path.basename(CACHE_50_PARQUET)}")
    df_50 = pd.read_parquet(CACHE_50_PARQUET)
    existing_cids = set(df_50["caseid"].unique())
    print(f"  -> Existing cases loaded: {len(existing_cids)} cases ({len(df_50):,} epochs)")

    new_dfs = []
    # Search from case 58 upwards
    candidate_cids = [c for c in range(58, 250) if c not in existing_cids]
    print(f"[STREAMING] Concurrently streaming {TARGET_NEW_CASES} new cases using ThreadPoolExecutor (max_workers=6)...")

    t_stream_start = time.time()
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_cid = {executor.submit(process_streamed_case, cid): cid for cid in candidate_cids}
        for future in as_completed(future_to_cid):
            cid = future_to_cid[future]
            try:
                res_df = future.result()
                if res_df is not None and len(res_df) >= 30:
                    new_dfs.append(res_df)
                    print(f"  [STREAM SUCCESS] Case {cid:3d}: {len(res_df):>4} epochs ({len(res_df)*10/60:5.1f}m) | Collected: {len(new_dfs)}/{TARGET_NEW_CASES}")
                    if len(new_dfs) >= TARGET_NEW_CASES:
                        # Cancel remaining
                        for f in future_to_cid:
                            f.cancel()
                        break
                else:
                    pass
            except Exception as e:
                pass

    print(f"[STREAMING] Finished streaming {len(new_dfs)} new cases in {time.time() - t_stream_start:.1f}s")
    df_new = pd.concat(new_dfs[:TARGET_NEW_CASES], ignore_index=True)
    df_100 = pd.concat([df_50, df_new], ignore_index=True)
    
    # Save cache
    df_100.to_parquet(CACHE_100_PARQUET, index=False)
    print(f"[CACHE] Saved {len(df_100):,} epochs across {len(df_100['caseid'].unique())} cases to: {os.path.basename(CACHE_100_PARQUET)}")
    return df_100

def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes causal past dynamic features per (caseid, seg_id)."""
    df = df.copy()
    df["delta_T"] = df["t_skin_meas"] - df["t_amb"]
    df["pi_eff"] = np.clip(0.10 * df["pi"], 0.005, 10.0)
    
    # Biophysical conductance term using fixed a, b, alpha
    g_cs = A_FIXED + B_FIXED * np.power(df["pi_eff"], ALPHA_FIXED)
    df["bio_conductance_term"] = df["delta_T"] / np.maximum(g_cs, 0.10)
    
    # Grouped causal rolling features
    grp = df.groupby(["caseid", "seg_id"])
    df["tskin_roll60"] = grp["t_skin_meas"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["tskin_roll180"] = grp["t_skin_meas"].transform(lambda s: s.rolling(18, min_periods=1).mean())
    df["d_tskin_dt"] = grp["t_skin_meas"].diff().fillna(0.0) / 10.0
    return df

def evaluate_models_on_100_cases():
    t_start = time.time()
    print("=" * 115)
    print("SCALING VALIDATION: 100 SURGICAL CASES FROM VITALDB")
    print("Evaluating Basic Biophysical Bioheat vs. Linearized Bioheat Ridge (Fixed Weights)")
    print("=" * 115)

    df_raw = load_or_create_100_cases()
    df = add_causal_features(df_raw)
    
    all_cases = sorted(list(df["caseid"].unique()))
    n_cases = len(all_cases)
    total_epochs = len(df)
    total_hours = total_epochs * 10.0 / 3600.0

    print(f"\n[DATASET SUMMARY]")
    print(f"  Total Valid Cases : {n_cases}")
    print(f"  Total 10s Epochs   : {total_epochs:,} epochs")
    print(f"  Total Surgical Time: {total_hours:.2f} hours")
    print(f"  Mean Case Duration : {total_hours / n_cases * 60.0:.1f} minutes")

    # 1. Deterministic 70/30 Patient Split
    rng = np.random.default_rng(2026)
    shuffled = all_cases.copy()
    rng.shuffle(shuffled)
    n_train = 70
    train_cases = sorted(shuffled[:n_train])
    test_cases = sorted(shuffled[n_train:])

    df_train = df[df["caseid"].isin(train_cases)].copy()
    df_test = df[df["caseid"].isin(test_cases)].copy()

    print(f"\n[PATIENT SPLIT (70 Train / 30 Test)]")
    print(f"  Train Cohort: {len(train_cases)} Patients ({len(df_train):,} epochs, {len(df_train)*10/3600:.1f}h)")
    print(f"  Test Cohort : {len(test_cases)} Patients ({len(df_test):,} epochs, {len(df_test)*10/3600:.1f}h)")
    print(f"  Zero Leakage Check: {len(set(train_cases).intersection(set(test_cases))) == 0}")

    # Predictions with Linearized Bioheat (Fixed Weights)
    def pred_linearized_bioheat(d):
        t_calc = (
            W0_FIXED
            + W1_FIXED * d["t_skin_meas"]
            + W2_FIXED * d["bio_conductance_term"]
            + W3_FIXED * d["tskin_roll180"]
            + W4_FIXED * d["d_tskin_dt"]
        )
        return np.clip(t_calc, 30.0, 45.0)

    # Predictions with Basic Bioheat Formula (Fixed a, b, alpha)
    def pred_basic_bioheat(d):
        g = A_FIXED + B_FIXED * np.power(d["pi_eff"], ALPHA_FIXED)
        t_calc = d["t_skin_meas"] + (d["t_skin_meas"] - d["t_amb"]) / np.maximum(g, 0.10)
        return np.clip(t_calc, 30.0, 45.0)

    # Predictions with TriaGo Empirical Baseline
    def pred_triago_empirical(d):
        k_env_base = -0.1858
        beta_pi = -0.1980
        pi_norm = np.clip(d["pi_eff"], 0.05, 3.0)
        k_adapt = k_env_base / (1.0 + beta_pi * (pi_norm - 0.2))
        c_offset = 5.6694
        t_calc = d["t_skin_meas"] + k_adapt * d["delta_T"] + c_offset
        return np.clip(t_calc, 30.0, 45.0)

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
        
        # Per-case metrics
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
                "mae": round(c_mae, 4),
                "bias": round(c_bias, 4),
                "sd": round(c_sd, 4),
                "acc_35": round(c_acc35, 1),
                "pass_mae35": c_pass
            })
            
        pass_count = sum(1 for c in per_case if c["pass_mae35"])
        pass_rate = round(pass_count / len(per_case) * 100.0, 1)
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
            "cases_passed": pass_count,
            "cases_total": len(per_case),
            "pass_rate_pct": pass_rate,
            "per_case": per_case
        }

    # Evaluate on:
    # 1. 30 Unseen Test Cases
    # 2. All 100 Cases (Full Generalization Cohort)
    results = {}
    
    models = {
        "1. Basic Bioheat (Fixed a,b,alpha)": pred_basic_bioheat,
        "2. Linearized Bioheat Ridge (Fixed Weights)": pred_linearized_bioheat,
        "3. TriaGo Empirical Formula": pred_triago_empirical
    }

    print("\n" + "=" * 115)
    print(f"{'EVALUATION ON 30 UNSEEN TEST PATIENTS (' + str(len(df_test)) + ' Epochs)':^115}")
    print("=" * 115)
    print(f"{'Model':<44} | {'Pooled MAE':<11} | {'Macro MAE':<10} | {'Bias':<8} | {'SD':<7} | {'95% LoA':<17} | {'Pass Rate':<12} | {'Epoch Acc <=0.35C'}")
    print("-" * 115)

    test_metrics = {}
    for m_name, fn in models.items():
        y_pred = fn(df_test)
        met = calc_metrics(df_test["t_core"].values, y_pred.values, test_cases, df_test)
        test_metrics[m_name] = met
        loa_str = f"[{met['lower_loa']:+.2f}, {met['upper_loa']:+.2f}]"
        pass_str = f"{met['cases_passed']}/{met['cases_total']} ({met['pass_rate_pct']}%)"
        print(f"{m_name:<44} | {met['pooled_mae']:<7.4f} °C | {met['macro_mae']:<7.4f} °C | {met['bias']:<+7.4f} | {met['sd']:<6.4f} | {loa_str:<17} | {pass_str:<12} | {met['acc_35']:.1f}%")

    print("\n" + "=" * 115)
    print(f"{'EVALUATION ON FULL 100 SURGICAL PATIENT COHORT (' + str(len(df)) + ' Epochs)':^115}")
    print("=" * 115)
    print(f"{'Model':<44} | {'Pooled MAE':<11} | {'Macro MAE':<10} | {'Bias':<8} | {'SD':<7} | {'95% LoA':<17} | {'Pass Rate':<12} | {'Epoch Acc <=0.35C'}")
    print("-" * 115)

    full_metrics = {}
    for m_name, fn in models.items():
        y_pred = fn(df)
        met = calc_metrics(df["t_core"].values, y_pred.values, all_cases, df)
        full_metrics[m_name] = met
        loa_str = f"[{met['lower_loa']:+.2f}, {met['upper_loa']:+.2f}]"
        pass_str = f"{met['cases_passed']}/{met['cases_total']} ({met['pass_rate_pct']}%)"
        print(f"{m_name:<44} | {met['pooled_mae']:<7.4f} °C | {met['macro_mae']:<7.4f} °C | {met['bias']:<+7.4f} | {met['sd']:<6.4f} | {loa_str:<17} | {pass_str:<12} | {met['acc_35']:.1f}%")

    # Save summary json
    summary_data = {
        "study": "100-Case VitalDB Temperature Validation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": n_cases,
        "total_epochs": total_epochs,
        "total_hours": round(total_hours, 2),
        "fixed_parameters": {
            "a": A_FIXED,
            "b": B_FIXED,
            "alpha": ALPHA_FIXED,
            "w0": W0_FIXED,
            "w1": W1_FIXED,
            "w2": W2_FIXED,
            "w3": W3_FIXED,
            "w4": W4_FIXED
        },
        "train_patients": [int(c) for c in train_cases],
        "test_patients": [int(c) for c in test_cases],
        "test_metrics_30_patients": test_metrics,
        "full_metrics_100_patients": full_metrics,
        "runtime_total_sec": round(time.time() - t_start, 2)
    }

    with open(SUMMARY_100_JSON, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[OUTPUT] Saved detailed 100-case summary to: {SUMMARY_100_JSON}")
    print(f"[RUNTIME] Total 100-case pipeline completed in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    evaluate_models_on_100_cases()
