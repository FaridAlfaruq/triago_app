import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add paths to TriaGo and vitaldb_rr_test
TRIAGO_DIR = Path(r"C:\Users\Adyty\Documents\Farid ITS\TriaGo")
STAGE_DIR = Path(__file__).resolve().parent
if str(TRIAGO_DIR) not in sys.path:
    sys.path.insert(0, str(TRIAGO_DIR))
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

import stage2
from processing_data.processing_data import ECGProcessor
from respiratory_rate.pipeline import MultimodalRespirationEstimator

FS = 125.0


def evaluate_cohort_rr(case_id: int = 1, max_segs: int | None = None):
    """Run TriaGo respiratory rate estimator on all SQI-passed 60s segments for a case."""
    df = stage2.load_vital(case_id)
    good_segs, _ = stage2.segment_and_process(df)

    if max_segs:
        good_segs = good_segs[:max_segs]

    ecg_proc = ECGProcessor(target_fs=int(FS))
    estimator = MultimodalRespirationEstimator(min_duration=10.0)

    results = []
    sample_details = None

    for idx, seg in enumerate(good_segs):
        r_peaks, _ = ecg_proc.detect_r_peaks(seg["ecg_raw"], fs=FS)
        res = estimator.estimate(
            ecg=seg["ecg_raw"],
            r_peaks=r_peaks,
            ppg=seg["ppg_raw"],
            fs=FS,
        )

        ref_rr = seg["stats"]["rr_mean"]
        est_rr = res["rr"]
        err = abs(est_rr - ref_rr)

        results.append({
            "seg_idx": seg["seg_idx"],
            "time_min": round((seg["seg_idx"] * 60.0) / 60.0, 1),
            "ref_rr": ref_rr,
            "est_rr": est_rr,
            "abs_err": err,
            "quality": res["quality"],
            "mode": res["mode"],
        })

        if idx == 0:
            sample_details = {
                "seg": seg,
                "res": res,
                "r_peaks": r_peaks,
            }

    df_res = pd.DataFrame(results)
    return df_res, sample_details


def plot_stage3_evaluation(df_res: pd.DataFrame, sample: dict):
    """Visualize RR evaluation into 2 columns: Column 1 = Segment Estimation vs Reference; Column 2 = Error Distribution & Scatter."""
    mae = df_res["abs_err"].mean()
    within_2 = (df_res["abs_err"] <= 2.0).mean() * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Top-Left: Time-series tracking across 60s segments
    axes[0, 0].plot(df_res["seg_idx"], df_res["ref_rr"], "g-o", label="Reference (VENT_RR)", lw=1.5, markersize=3)
    axes[0, 0].plot(df_res["seg_idx"], df_res["est_rr"], "b-s", label="Estimated (TriaGo Fusion)", lw=1.2, markersize=3, alpha=0.85)
    axes[0, 0].set_title(f"Respiratory Rate Tracking (MAE: {mae:.2f} rpm, CP2: {within_2:.1f}%)")
    axes[0, 0].set_ylabel("RR (rpm)")
    axes[0, 0].set_xlabel("Segment Index (60s)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-Right: Scatter Plot (Est vs Ref) with ±2 rpm tolerance
    r_min, r_max = min(df_res["ref_rr"].min(), df_res["est_rr"].min()) - 2, max(df_res["ref_rr"].max(), df_res["est_rr"].max()) + 2
    axes[0, 1].scatter(df_res["ref_rr"], df_res["est_rr"], color="royalblue", alpha=0.7, edgecolors="none")
    axes[0, 1].plot([r_min, r_max], [r_min, r_max], "k--", label="Ideal Identity (y=x)")
    axes[0, 1].plot([r_min, r_max], [r_min + 2, r_max + 2], "r:", label="+2 rpm Tol")
    axes[0, 1].plot([r_min, r_max], [r_min - 2, r_max - 2], "r:", label="-2 rpm Tol")
    axes[0, 1].set_xlim([r_min, r_max])
    axes[0, 1].set_ylim([r_min, r_max])
    axes[0, 1].set_title("Correlation & Agreement (Identity vs Tolerance)")
    axes[0, 1].set_xlabel("Reference VENT_RR (rpm)")
    axes[0, 1].set_ylabel("Estimated RR (rpm)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-Left: Reconstructed Respiratory Waveform (First 60s Segment)
    t_sec = np.arange(len(sample["res"]["resp_signal"])) / FS
    axes[1, 0].plot(t_sec, sample["res"]["resp_signal"], color="darkgreen", lw=1.2)
    pks = sample["res"]["resp_peaks"]
    if len(pks):
        axes[1, 0].plot(t_sec[pks], sample["res"]["resp_signal"][pks], "ro", markersize=5, label="Breath Peaks")
    axes[1, 0].set_title(f"Sample Segment #0 Respiration Waveform (Est: {sample['res']['rr']:.1f} rpm)")
    axes[1, 0].set_ylabel("Normalized Amplitude")
    axes[1, 0].set_xlabel("Time in Segment (s)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-Right: Error Histogram
    axes[1, 1].hist(df_res["abs_err"], bins=15, color="teal", edgecolor="black", alpha=0.75)
    axes[1, 1].axvline(mae, color="red", linestyle="--", lw=1.5, label=f"Mean Absolute Error: {mae:.2f} rpm")
    axes[1, 1].axvline(2.0, color="orange", linestyle=":", lw=1.5, label="AAMI/ISO Tolerance (2.0 rpm)")
    axes[1, 1].set_title("Absolute Error Distribution")
    axes[1, 1].set_xlabel("Absolute Error (rpm)")
    axes[1, 1].set_ylabel("Segment Count")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Stage 3: TriaGo Multimodal Respiratory Rate Evaluation on VitalDB Case 1", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("[STAGE 3] Running TriaGo Multimodal RR Estimation on VitalDB Case 1...")
    df_eval, sample_data = evaluate_cohort_rr(case_id=1)

    mae = df_eval["abs_err"].mean()
    rmse = np.sqrt((df_eval["abs_err"] ** 2).mean())
    cp2 = (df_eval["abs_err"] <= 2.0).mean() * 100.0
    corr = np.corrcoef(df_eval["ref_rr"], df_eval["est_rr"])[0, 1]

    print("=" * 65)
    print(f"Total Evaluated Segments : {len(df_eval)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} rpm")
    print(f"Root Mean Sq Error (RMSE): {rmse:.2f} rpm")
    print(f"Pearson Correlation (r)  : {corr:.3f}")
    print(f"Within +/- 2.0 rpm (CP2) : {cp2:.1f} %")
    print("=" * 65)

    plot_stage3_evaluation(df_eval, sample_data)
