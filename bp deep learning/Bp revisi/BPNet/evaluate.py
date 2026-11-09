"""
evaluate.py
Evaluation module for CalBPNet models on PulseDB under CalFree and CalBased protocols.

Metrics Evaluated:
1. Mean Absolute Error (MAE in mmHg).
2. Root Mean Squared Error (RMSE in mmHg).
3. Mean Error & Standard Deviation (ME +/- STD in mmHg).
4. AAMI Standard Verification (|ME| <= 5.0 mmHg, STD <= 8.0 mmHg).
5. British Hypertension Society (BHS) Standard Grading (Grades A, B, C, D).
6. Pearson Correlation Coefficient (r) & Coefficient of Determination (R^2).
7. Visualizations:
   - Row 0: SBP & DBP R^2 Regression Graphs (True vs Predicted with y=x unity line).
   - Row 1: SBP & DBP Bland-Altman Plots (Mean BP vs Difference with +/- 1.96 SD limits).
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from load import load_raw_subject, list_available_subjects
from signal_processing import preprocess_subject_signals
from data_split import split_dataset
from models import CalBPNet


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def compute_bhs_grades(errors: np.ndarray) -> Dict[str, Any]:
    """
    Compute cumulative error percentages and assign British Hypertension Society (BHS) grade.
    Criteria:
      Grade A: <=5 mmHg >= 60%, <=10 mmHg >= 85%, <=15 mmHg >= 95%
      Grade B: <=5 mmHg >= 50%, <=10 mmHg >= 75%, <=15 mmHg >= 90%
      Grade C: <=5 mmHg >= 40%, <=10 mmHg >= 65%, <=15 mmHg >= 85%
      Grade D: Below Grade C thresholds
    """
    abs_err = np.abs(errors)
    p5 = float(np.mean(abs_err <= 5.0) * 100.0)
    p10 = float(np.mean(abs_err <= 10.0) * 100.0)
    p15 = float(np.mean(abs_err <= 15.0) * 100.0)

    if p5 >= 60.0 and p10 >= 85.0 and p15 >= 95.0:
        grade = "A"
    elif p5 >= 50.0 and p10 >= 75.0 and p15 >= 90.0:
        grade = "B"
    elif p5 >= 40.0 and p10 >= 65.0 and p15 >= 85.0:
        grade = "C"
    else:
        grade = "D"

    return {
        "pct_le_5": round(p5, 1),
        "pct_le_10": round(p10, 1),
        "pct_le_15": round(p15, 1),
        "bhs_grade": grade,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute comprehensive clinical and statistical performance metrics.
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    me = float(np.mean(errors))
    std = float(np.std(errors))

    # AAMI Standard: |ME| <= 5 mmHg and STD <= 8 mmHg
    aami_pass = bool(abs(me) <= 5.0 and std <= 8.0)

    # BHS Grade
    bhs = compute_bhs_grades(errors)

    # Pearson Correlation (r) & R^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - float(ss_res / (ss_tot + 1e-8))

    if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "me": round(me, 2),
        "std": round(std, 2),
        "r2": round(r2, 3),
        "pearson_r": round(r, 3),
        "aami_pass": aami_pass,
        "bhs_grade": bhs["bhs_grade"],
        "pct_le_5": bhs["pct_le_5"],
        "pct_le_10": bhs["pct_le_10"],
        "pct_le_15": bhs["pct_le_15"],
    }


def plot_evaluation_graphs(
    y_s_true: np.ndarray,
    y_s_pred: np.ndarray,
    y_d_true: np.ndarray,
    y_d_pred: np.ndarray,
    protocol: str = "calfree",
) -> None:
    """
    Visualize R^2 Regression plots and Bland-Altman agreement plots in 2x2 grid:
    - Row 0: SBP & DBP R^2 Correlation Graphs.
    - Row 1: SBP & DBP Bland-Altman Error Plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # --------------------------------------------------------------------------
    # ROW 0, COL 0: SBP R^2 REGRESSION
    # --------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.scatter(y_s_true, y_s_pred, color="#1f77b4", alpha=0.7, s=40, label="Data")

    # Unity line y = x
    min_v = min(y_s_true.min(), y_s_pred.min()) - 5
    max_v = max(y_s_true.max(), y_s_pred.max()) + 5
    ax.plot([min_v, max_v], [min_v, max_v], color="gray", linestyle="--", linewidth=1.1, label="y = x")

    # Best-fit regression line
    if len(y_s_true) > 1 and np.std(y_s_true) > 1e-6:
        poly = np.polyfit(y_s_true, y_s_pred, deg=1)
        fit_x = np.linspace(min_v, max_v, 100)
        ax.plot(fit_x, np.polyval(poly, fit_x), color="red", linewidth=1.2, label="Fit")

    ax.set_title("SBP - R^2 Regression", fontsize=11, weight="bold")
    ax.set_xlabel("True SBP (mmHg)")
    ax.set_ylabel("Predicted SBP (mmHg)")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

    # --------------------------------------------------------------------------
    # ROW 0, COL 1: DBP R^2 REGRESSION
    # --------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.scatter(y_d_true, y_d_pred, color="#2ca02c", alpha=0.7, s=40, label="Data")

    min_v = min(y_d_true.min(), y_d_pred.min()) - 5
    max_v = max(y_d_true.max(), y_d_pred.max()) + 5
    ax.plot([min_v, max_v], [min_v, max_v], color="gray", linestyle="--", linewidth=1.1, label="y = x")

    if len(y_d_true) > 1 and np.std(y_d_true) > 1e-6:
        poly = np.polyfit(y_d_true, y_d_pred, deg=1)
        fit_x = np.linspace(min_v, max_v, 100)
        ax.plot(fit_x, np.polyval(poly, fit_x), color="red", linewidth=1.2, label="Fit")

    ax.set_title("DBP - R^2 Regression", fontsize=11, weight="bold")
    ax.set_xlabel("True DBP (mmHg)")
    ax.set_ylabel("Predicted DBP (mmHg)")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

    # --------------------------------------------------------------------------
    # ROW 1, COL 0: SBP BLAND-ALTMAN
    # --------------------------------------------------------------------------
    ax = axes[1, 0]
    mean_s = (y_s_true + y_s_pred) / 2.0
    diff_s = y_s_pred - y_s_true
    bias_s = float(np.mean(diff_s))
    sd_s = float(np.std(diff_s))

    ax.scatter(mean_s, diff_s, color="#1f77b4", alpha=0.7, s=40, label="Error")
    ax.axhline(bias_s, color="blue", linewidth=1.2, label=f"Mean: {bias_s:.1f}")
    ax.axhline(bias_s + 1.96 * sd_s, color="red", linestyle="--", linewidth=1.1, label=f"+1.96 SD: {bias_s+1.96*sd_s:.1f}")
    ax.axhline(bias_s - 1.96 * sd_s, color="red", linestyle="--", linewidth=1.1, label=f"-1.96 SD: {bias_s-1.96*sd_s:.1f}")

    ax.set_title("SBP - Bland-Altman", fontsize=11, weight="bold")
    ax.set_xlabel("Mean SBP (mmHg)")
    ax.set_ylabel("Error (mmHg)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7.5)

    # --------------------------------------------------------------------------
    # ROW 1, COL 1: DBP BLAND-ALTMAN
    # --------------------------------------------------------------------------
    ax = axes[1, 1]
    mean_d = (y_d_true + y_d_pred) / 2.0
    diff_d = y_d_pred - y_d_true
    bias_d = float(np.mean(diff_d))
    sd_d = float(np.std(diff_d))

    ax.scatter(mean_d, diff_d, color="#2ca02c", alpha=0.7, s=40, label="Error")
    ax.axhline(bias_d, color="blue", linewidth=1.2, label=f"Mean: {bias_d:.1f}")
    ax.axhline(bias_d + 1.96 * sd_d, color="red", linestyle="--", linewidth=1.1, label=f"+1.96 SD: {bias_d+1.96*sd_d:.1f}")
    ax.axhline(bias_d - 1.96 * sd_d, color="red", linestyle="--", linewidth=1.1, label=f"-1.96 SD: {bias_d-1.96*sd_d:.1f}")

    ax.set_title("DBP - Bland-Altman", fontsize=11, weight="bold")
    ax.set_xlabel("Mean DBP (mmHg)")
    ax.set_ylabel("Error (mmHg)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7.5)

    fig.suptitle(f"Model Evaluation: R^2 Regression & Bland-Altman ({protocol.upper()})", fontsize=12, weight="bold")
    print("Displaying R^2 Regression & Bland-Altman graphs via plt.show()...")
    plt.show()


def evaluate_checkpoint(
    protocol: str = "calfree",
    source: str = "PulseDB_Vital",
    show_plot: bool = True,
) -> Dict[str, Any]:
    """
    Load trained model checkpoint, run inference on held-out test data,
    compute all clinical metrics, and optionally display R^2 & Bland-Altman plots.
    """
    ckpt_path = CHECKPOINT_DIR / f"best_calbpnet_{protocol}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}. Please run train.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    test_subjs = ckpt["test_subjects"]

    print("=" * 70)
    print(f"EVALUATING CALBPNET CHECKPOINT | PROTOCOL: {protocol.upper()}")
    print("=" * 70)
    print(f"Checkpoint Path       : {ckpt_path.name}")
    print(f"Trained Protocol      : {ckpt.get('protocol', protocol)}")
    print(f"Best Training Val MAE : {ckpt.get('val_mae_all', 0.0):.2f} mmHg")
    print(f"Test Subjects Evaluated: {test_subjs}")

    # 1. Load Test Subjects
    all_X, all_s, all_d, all_g = [], [], [], []
    for s in test_subjs:
        e, p, a, sbp, dbp = load_raw_subject(s, max_segments=10, source=source)
        Xc, sc, dc, mask = preprocess_subject_signals(e, p, sbp, dbp, fs=125)
        if len(Xc) > 0:
            all_X.append(Xc)
            all_s.append(sc)
            all_d.append(dc)
            all_g.extend([s] * len(Xc))

    if len(all_X) == 0:
        raise ValueError("No test segments found for evaluation.")

    X_test = np.concatenate(all_X, axis=0)
    y_s_true = np.concatenate(all_s, axis=0)
    y_d_true = np.concatenate(all_d, axis=0)
    groups_test = np.array(all_g)

    # 2. Model Inference
    model = CalBPNet(in_channels=7)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        p_s_z, p_d_z = model(torch.tensor(X_test, dtype=torch.float32))
        p_sbp = (p_s_z * ckpt["s_std"] + ckpt["s_mean"]).numpy()
        p_dbp = (p_d_z * ckpt["d_std"] + ckpt["d_mean"]).numpy()

    # 3. Compute Metrics
    s_metrics = compute_metrics(y_s_true, p_sbp)
    d_metrics = compute_metrics(y_d_true, p_dbp)
    mean_mae = (s_metrics["mae"] + d_metrics["mae"]) / 2.0

    # --------------------------------------------------------------------------
    # PRINT RESULTS TABLE TO CONSOLE
    # --------------------------------------------------------------------------
    print("-" * 70)
    print(f"{'Target':<6s} | {'MAE':>6s} | {'RMSE':>6s} | {'ME +/- STD (mmHg)':>18s} | {'R^2':>6s} | {'r':>6s} | {'BHS':>4s} | {'AAMI':>5s}")
    print("-" * 70)
    print(
        f"{'SBP':<6s} | {s_metrics['mae']:>6.2f} | {s_metrics['rmse']:>6.2f} | "
        f"{s_metrics['me']:>5.2f} +/- {s_metrics['std']:<5.2f} mmHg | "
        f"{s_metrics['r2']:>6.3f} | {s_metrics['pearson_r']:>6.3f} | "
        f"{s_metrics['bhs_grade']:^4s} | {'PASS' if s_metrics['aami_pass'] else 'FAIL':^5s}"
    )
    print(
        f"{'DBP':<6s} | {d_metrics['mae']:>6.2f} | {d_metrics['rmse']:>6.2f} | "
        f"{d_metrics['me']:>5.2f} +/- {d_metrics['std']:<5.2f} mmHg | "
        f"{d_metrics['r2']:>6.3f} | {d_metrics['pearson_r']:>6.3f} | "
        f"{d_metrics['bhs_grade']:^4s} | {'PASS' if d_metrics['aami_pass'] else 'FAIL':^5s}"
    )
    print("-" * 70)
    print(f"Overall Test MAE      : {mean_mae:.2f} mmHg (SBP: {s_metrics['mae']:.2f}, DBP: {d_metrics['mae']:.2f})")
    print(f"BHS Grade Cumulative  : SBP <=5: {s_metrics['pct_le_5']}%, <=10: {s_metrics['pct_le_10']}%, <=15: {s_metrics['pct_le_15']}%")
    print(f"                        DBP <=5: {d_metrics['pct_le_5']}%, <=10: {d_metrics['pct_le_10']}%, <=15: {d_metrics['pct_le_15']}%")
    print("=" * 70 + "\n")

    # 4. Save JSON Results
    eval_json_path = CHECKPOINT_DIR / f"eval_metrics_{protocol}.json"
    with open(eval_json_path, "w") as f:
        json.dump({
            "protocol": protocol,
            "overall_mae": round(mean_mae, 2),
            "sbp": s_metrics,
            "dbp": d_metrics,
            "test_segments": len(X_test),
            "test_subjects": test_subjs,
        }, f, indent=2)

    # 5. Visual Plot
    if show_plot:
        plot_evaluation_graphs(y_s_true, p_sbp, y_d_true, p_dbp, protocol=protocol)

    return {
        "overall_mae": mean_mae,
        "sbp": s_metrics,
        "dbp": d_metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CalBPNet checkpoint with R^2 & Bland-Altman.")
    parser.add_argument("--protocol", type=str, default="calfree", choices=["calfree", "calbased"])
    parser.add_argument("--source", type=str, default="PulseDB_Vital", choices=["PulseDB_Vital", "PulseDB_MIMIC"])
    parser.add_argument("--no-plot", action="store_true", help="Print metrics without opening plot window")
    args = parser.parse_args()

    evaluate_checkpoint(
        protocol=args.protocol,
        source=args.source,
        show_plot=not args.no_plot,
    )
