"""
run_ablations.py
Executes the two key scientific ablations on trained CalBPNet checkpoints:
1. CalFree model + 1-Point Anchor Calibration on the 20 unseen test subjects.
   -> Isolates the pure effect of calibration on unseen patients.
2. CalBased model evaluated RAW (without calibration offset) on the test segments.
   -> Isolates the effect of seeing the patient during training (subject morphology memorization).
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

# Add remote job dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CalBPNet, split_calfree, split_calbased, compute_metrics, set_seed

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("PULSEDB CALBPNET ABLATION STUDY: 2x2 FACTORIAL ANALYSIS")
    print(f"Device: {device}")
    print("=" * 76)

    # 1. Load Dataset
    data_path = Path("scratch/gemastik/pulsedb_100subj_7ch.npz")
    if not data_path.exists():
        data_path = Path(__file__).resolve().parents[3] / "scratch/gemastik/pulsedb_100subj_7ch.npz"
    data = np.load(data_path)
    X = data["X"]
    y_sbp = data["y_sbp"]
    y_dbp = data["y_dbp"]
    groups = data["groups"]

    out_dir = Path(__file__).resolve().parent / "output_v3"
    cf_ckpt_path = out_dir / "best_calbpnet_calfree.pt"
    cb_ckpt_path = out_dir / "best_calbpnet_calbased.pt"

    # =========================================================================
    # ABLATION 1: CalFree Model + 1-Point Anchor Offset on Unseen Test Subjects
    # =========================================================================
    print("\n--- Running Ablation 1: CalFree Model + 1-Point Anchor Calibration ---")
    train_idx_cf, test_idx_cf, tr_subjs_cf, te_subjs_cf = split_calfree(groups, test_ratio=0.20, seed=42)
    s_mean_cf = float(y_sbp[train_idx_cf].mean())
    s_std_cf = float(y_sbp[train_idx_cf].std() + 1e-6)
    d_mean_cf = float(y_dbp[train_idx_cf].mean())
    d_std_cf = float(y_dbp[train_idx_cf].std() + 1e-6)

    ckpt_cf = torch.load(cf_ckpt_path, map_location=device)
    model_cf = CalBPNet(in_channels=7).to(device)
    model_cf.load_state_dict(ckpt_cf["model_state_dict"])
    model_cf.eval()

    # For each unseen test subject, segment 0 (first chronological segment) is anchor
    test_groups_cf = groups[test_idx_cf]
    unique_te_subjs = np.unique(test_groups_cf)

    anchor_idx_cf = {}
    eval_idx_cf = []
    for s in unique_te_subjs:
        s_pos = np.where(test_groups_cf == s)[0]
        # First segment of subject in test set is calibration anchor
        anchor_global_idx = test_idx_cf[s_pos[0]]
        anchor_idx_cf[s] = anchor_global_idx
        # Remaining segments (t >= 1) form the evaluation set
        eval_idx_cf.extend(test_idx_cf[s_pos[1:]])

    eval_idx_cf = np.array(eval_idx_cf)
    eval_groups_cf = groups[eval_idx_cf]

    with torch.no_grad():
        # Predict on eval segments
        x_eval_cf = torch.tensor(X[eval_idx_cf], dtype=torch.float32).to(device)
        ps_z, pd_z = model_cf(x_eval_cf)
        raw_ps_cf = ps_z.cpu().numpy() * s_std_cf + s_mean_cf
        raw_pd_cf = pd_z.cpu().numpy() * d_std_cf + d_mean_cf

        # Predict on anchor segments
        anc_subjs = sorted(list(anchor_idx_cf.keys()))
        anc_indices = [anchor_idx_cf[s] for s in anc_subjs]
        x_anc = torch.tensor(X[anc_indices], dtype=torch.float32).to(device)
        anc_s_z, anc_d_z = model_cf(x_anc)
        anc_ps = anc_s_z.cpu().numpy() * s_std_cf + s_mean_cf
        anc_pd = anc_d_z.cpu().numpy() * d_std_cf + d_mean_cf

    # Compute calibration deltas: Delta = true_anchor - pred_anchor
    delta_s_cf = {s: y_sbp[idx] - p for s, idx, p in zip(anc_subjs, anc_indices, anc_ps)}
    delta_d_cf = {s: y_dbp[idx] - p for s, idx, p in zip(anc_subjs, anc_indices, anc_pd)}

    cal_ps_cf = raw_ps_cf + np.array([delta_s_cf[s] for s in eval_groups_cf])
    cal_pd_cf = raw_pd_cf + np.array([delta_d_cf[s] for s in eval_groups_cf])

    t_s_cf = y_sbp[eval_idx_cf]
    t_d_cf = y_dbp[eval_idx_cf]

    m_ablation1_s = compute_metrics(t_s_cf, cal_ps_cf)
    m_ablation1_d = compute_metrics(t_d_cf, cal_pd_cf)
    mae_ablation1 = round((m_ablation1_s["mae"] + m_ablation1_d["mae"]) / 2.0, 2)
    rmse_ablation1 = round((m_ablation1_s["rmse"] + m_ablation1_d["rmse"]) / 2.0, 2)

    # Also compute Raw on the exact same eval subset (t >= 1) for fair subtraction
    raw_eval_s = compute_metrics(t_s_cf, raw_ps_cf)
    raw_eval_d = compute_metrics(t_d_cf, raw_pd_cf)
    mae_raw_eval = round((raw_eval_s["mae"] + raw_eval_d["mae"]) / 2.0, 2)

    # =========================================================================
    # ABLATION 2: CalBased Model Evaluated RAW (No Calibration Offset)
    # =========================================================================
    print("--- Running Ablation 2: CalBased Model Evaluated RAW (No Offset) ---")
    train_idx_cb, test_idx_cb, anchors_cb = split_calbased(groups, test_ratio=0.10, min_segments_per_subj=5, seed=42)
    s_mean_cb = float(y_sbp[train_idx_cb].mean())
    s_std_cb = float(y_sbp[train_idx_cb].std() + 1e-6)
    d_mean_cb = float(y_dbp[train_idx_cb].mean())
    d_std_cb = float(y_dbp[train_idx_cb].std() + 1e-6)

    ckpt_cb = torch.load(cb_ckpt_path, map_location=device)
    model_cb = CalBPNet(in_channels=7).to(device)
    model_cb.load_state_dict(ckpt_cb["model_state_dict"])
    model_cb.eval()

    with torch.no_grad():
        x_test_cb = torch.tensor(X[test_idx_cb], dtype=torch.float32).to(device)
        ps_z_cb, pd_z_cb = model_cb(x_test_cb)
        raw_ps_cb = ps_z_cb.cpu().numpy() * s_std_cb + s_mean_cb
        raw_pd_cb = pd_z_cb.cpu().numpy() * d_std_cb + d_mean_cb

    t_s_cb = y_sbp[test_idx_cb]
    t_d_cb = y_dbp[test_idx_cb]

    m_ablation2_s = compute_metrics(t_s_cb, raw_ps_cb)
    m_ablation2_d = compute_metrics(t_d_cb, raw_pd_cb)
    mae_ablation2 = round((m_ablation2_s["mae"] + m_ablation2_d["mae"]) / 2.0, 2)
    rmse_ablation2 = round((m_ablation2_s["rmse"] + m_ablation2_d["rmse"]) / 2.0, 2)

    # =========================================================================
    # Load Existing Baselines for Comparison
    # =========================================================================
    baseline_cf_s = ckpt_cf["metrics"]["sbp"]
    baseline_cf_d = ckpt_cf["metrics"]["dbp"]
    baseline_cb_s = ckpt_cb["metrics"]["sbp"]
    baseline_cb_d = ckpt_cb["metrics"]["dbp"]

    # =========================================================================
    # Print Comprehensive 2x2 Factorial Matrix
    # =========================================================================
    print("\n" + "=" * 80)
    print("2x2 FACTORIAL DECOMPOSITION: CALIBRATION OFFSET vs SUBJECT FAMILIARITY")
    print("=" * 80)
    print(f"{'Condition':<40s} | {'MAE (S / D / All)':<20s} | {'RMSE (S / D)':<16s} | {'BHS (S/D)':<9s} | {'AAMI'}")
    print("-" * 80)
    print(f"{'1. Unseen Subjs + RAW (CalFree Baseline)':<40s} | {baseline_cf_s['mae']:5.2f} / {baseline_cf_d['mae']:5.2f} / {ckpt_cf['val_mae']:5.2f} | {baseline_cf_s['rmse']:5.2f} / {baseline_cf_d['rmse']:5.2f} |   {baseline_cf_s['bhs_grade']} / {baseline_cf_d['bhs_grade']}   | {'PASS' if ckpt_cf['metrics']['aami_pass'] else 'FAIL'}")
    print(f"{'2. Unseen Subjs + 1-Pt Cal (Ablation 1)':<40s} | {m_ablation1_s['mae']:5.2f} / {m_ablation1_d['mae']:5.2f} / {mae_ablation1:5.2f} | {m_ablation1_s['rmse']:5.2f} / {m_ablation1_d['rmse']:5.2f} |   {m_ablation1_s['bhs_grade']} / {m_ablation1_d['bhs_grade']}   | {'PASS' if (m_ablation1_s['aami_pass'] and m_ablation1_d['aami_pass']) else 'FAIL'}")
    print(f"{'3. Seen Subjs + RAW (Ablation 2)':<40s} | {m_ablation2_s['mae']:5.2f} / {m_ablation2_d['mae']:5.2f} / {mae_ablation2:5.2f} | {m_ablation2_s['rmse']:5.2f} / {m_ablation2_d['rmse']:5.2f} |   {m_ablation2_s['bhs_grade']} / {m_ablation2_d['bhs_grade']}   | {'PASS' if (m_ablation2_s['aami_pass'] and m_ablation2_d['aami_pass']) else 'FAIL'}")
    print(f"{'4. Seen Subjs + 1-Pt Cal (CalBased Baseline)':<40s} | {baseline_cb_s['mae']:5.2f} / {baseline_cb_d['mae']:5.2f} / {ckpt_cb['val_mae']:5.2f} | {baseline_cb_s['rmse']:5.2f} / {baseline_cb_d['rmse']:5.2f} |   {baseline_cb_s['bhs_grade']} / {baseline_cb_d['bhs_grade']}   | {'PASS' if ckpt_cb['metrics']['aami_pass'] else 'FAIL'}")
    print("=" * 80)

    # Save summary JSON
    results = {
        "unseen_raw_calfree": {
            "overall_mae": ckpt_cf["val_mae"],
            "sbp": baseline_cf_s,
            "dbp": baseline_cf_d,
            "aami_pass": ckpt_cf["metrics"]["aami_pass"]
        },
        "unseen_with_anchor_ablation1": {
            "overall_mae": mae_ablation1,
            "overall_rmse": rmse_ablation1,
            "sbp": m_ablation1_s,
            "dbp": m_ablation1_d,
            "aami_pass": bool(m_ablation1_s["aami_pass"] and m_ablation1_d["aami_pass"]),
            "eval_segments": len(eval_idx_cf),
            "unseen_test_subjects": len(unique_te_subjs)
        },
        "seen_raw_no_anchor_ablation2": {
            "overall_mae": mae_ablation2,
            "overall_rmse": rmse_ablation2,
            "sbp": m_ablation2_s,
            "dbp": m_ablation2_d,
            "aami_pass": bool(m_ablation2_s["aami_pass"] and m_ablation2_d["aami_pass"]),
            "test_segments": len(test_idx_cb),
            "subjects": len(np.unique(groups[test_idx_cb]))
        },
        "seen_with_anchor_calbased": {
            "overall_mae": ckpt_cb["val_mae"],
            "sbp": baseline_cb_s,
            "dbp": baseline_cb_d,
            "aami_pass": ckpt_cb["metrics"]["aami_pass"]
        },
        "variance_decomposition": {
            "total_gap_mmhg": round(ckpt_cf["val_mae"] - ckpt_cb["val_mae"], 2),
            "pure_calibration_gain_on_unseen_mmhg": round(ckpt_cf["val_mae"] - mae_ablation1, 2),
            "subject_familiarity_gain_raw_mmhg": round(ckpt_cf["val_mae"] - mae_ablation2, 2),
            "anchor_gain_on_seen_mmhg": round(mae_ablation2 - ckpt_cb["val_mae"], 2),
        }
    }

    out_file = out_dir / "ablation_factorial_summary.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved factorial ablation summary to: {out_file}")

if __name__ == "__main__":
    main()
