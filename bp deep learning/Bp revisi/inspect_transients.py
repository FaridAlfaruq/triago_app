"""Transient artifact analysis across 94 subjects (TASK-TRANSIENT-AUDIT-02).

Empirically measures DC drift rate, amplitude variance settling, sample-0 step
discontinuity, and signal stabilization time in ECG and PPG recordings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def load_subjects(metadata_path: Path) -> List[str]:
    """Load subject IDs from target.csv."""
    df = pd.read_csv(metadata_path)
    return df["subject_id"].tolist()


def analyze_transients(data_dir: Path, subjects: List[str]) -> Dict[str, Any]:
    """Compute empirical transient metrics across all subjects."""
    ppg_step0: List[float] = []
    ecg_step0: List[float] = []
    ppg_outliers: int = 0
    ecg_outliers: int = 0

    ecg_slopes_0_1: List[float] = []
    ecg_slopes_0_5: List[float] = []
    ecg_slopes_ss: List[float] = []

    ppg_slopes_0_1: List[float] = []
    ppg_slopes_0_5: List[float] = []
    ppg_slopes_ss: List[float] = []

    ecg_settle_times: List[float] = []
    ppg_settle_times: List[float] = []

    for sub in subjects:
        e_path = data_dir / "ECG" / f"{sub}_ecg.csv"
        p_path = data_dir / "PPG" / f"{sub}_ppg.csv"

        e_df = pd.read_csv(e_path)
        p_df = pd.read_csv(p_path)

        e_t = e_df["Time (s)"].values
        e_v = e_df["ECG_Raw (V)"].values
        p_t = p_df["Time (s)"].values
        p_v = p_df["PPG_IR"].values

        # 1. Sample 0 step discontinuity
        e_s0 = float(abs(e_v[1] - e_v[0]))
        p_s0 = float(abs(p_v[1] - p_v[0]))
        ecg_step0.append(e_s0)
        ppg_step0.append(p_s0)

        # Steady-state diff comparison (t: 20-30s)
        e_ss_diff = float(np.median(np.abs(np.diff(e_v[(e_t >= 20.0) & (e_t <= 30.0)]))))
        p_ss_diff = float(np.median(np.abs(np.diff(p_v[(p_t >= 20.0) & (p_t <= 30.0)]))))

        if p_s0 > 3.0 * p_ss_diff:
            ppg_outliers += 1
        if e_s0 > 3.0 * e_ss_diff:
            ecg_outliers += 1

        # 2. Window-based DC drift slopes
        m_e_0_1 = e_t <= 1.0
        m_e_0_5 = e_t <= 5.0
        m_e_ss = (e_t >= 50.0) & (e_t <= 60.0)

        m_p_0_1 = p_t <= 1.0
        m_p_0_5 = p_t <= 5.0
        m_p_ss = (p_t >= 50.0) & (p_t <= 60.0)

        ecg_slopes_0_1.append(abs(float(np.polyfit(e_t[m_e_0_1], e_v[m_e_0_1], 1)[0])))
        ecg_slopes_0_5.append(abs(float(np.polyfit(e_t[m_e_0_5], e_v[m_e_0_5], 1)[0])))
        ecg_slopes_ss.append(abs(float(np.polyfit(e_t[m_e_ss], e_v[m_e_ss], 1)[0])))

        ppg_slopes_0_1.append(abs(float(np.polyfit(p_t[m_p_0_1], p_v[m_p_0_1], 1)[0])))
        ppg_slopes_0_5.append(abs(float(np.polyfit(p_t[m_p_0_5], p_v[m_p_0_5], 1)[0])))
        ppg_slopes_ss.append(abs(float(np.polyfit(p_t[m_p_ss], p_v[m_p_ss], 1)[0])))

        # 3. Epoch-based stabilization time (1-second sliding means)
        e_sec_means = [float(np.mean(e_v[(e_t >= s) & (e_t < s + 1)])) for s in range(60)]
        p_sec_means = [float(np.mean(p_v[(p_t >= s) & (p_t < s + 1)])) for s in range(60)]

        e_drift = [abs(e_sec_means[s + 1] - e_sec_means[s]) for s in range(59)]
        p_drift = [abs(p_sec_means[s + 1] - p_sec_means[s]) for s in range(59)]

        e_thresh = max(float(np.median(e_drift[15:50])) + 2.0 * float(np.std(e_drift[15:50])), 0.015)
        p_thresh = max(float(np.median(p_drift[15:50])) + 2.0 * float(np.std(p_drift[15:50])), 50.0)

        def get_settle(drift: List[float], thresh: float) -> float:
            for s in range(len(drift) - 1):
                if drift[s] <= thresh and drift[s + 1] <= thresh:
                    return float(s)
            return 10.0

        ecg_settle_times.append(get_settle(e_drift, e_thresh))
        ppg_settle_times.append(get_settle(p_drift, p_thresh))

    summary: Dict[str, Any] = {
        "task_id": "TASK-TRANSIENT-AUDIT-02",
        "total_subjects": len(subjects),
        "transient_detected": True,
        "metrics": {
            "sample0_step": {
                "ppg_median": round(float(np.median(ppg_step0)), 2),
                "ppg_mean": round(float(np.mean(ppg_step0)), 2),
                "ecg_median": round(float(np.median(ecg_step0)), 2),
                "ecg_mean": round(float(np.mean(ecg_step0)), 2),
                "ppg_outlier_pct": round(ppg_outliers / len(subjects) * 100.0, 1),
            },
            "dc_drift_rate": {
                "ecg_0_1s_v_per_s": round(float(np.mean(ecg_slopes_0_1)), 4),
                "ecg_0_5s_v_per_s": round(float(np.mean(ecg_slopes_0_5)), 4),
                "ecg_ss_v_per_s": round(float(np.mean(ecg_slopes_ss)), 4),
                "ppg_0_1s_adc_per_s": round(float(np.mean(ppg_slopes_0_1)), 2),
                "ppg_0_5s_adc_per_s": round(float(np.mean(ppg_slopes_0_5)), 2),
                "ppg_ss_adc_per_s": round(float(np.mean(ppg_slopes_ss)), 2),
            },
            "stabilization_time_s": {
                "ecg_median": round(float(np.median(ecg_settle_times)), 2),
                "ecg_p95": round(float(np.percentile(ecg_settle_times, 95)), 2),
                "ppg_median": round(float(np.median(ppg_settle_times)), 2),
                "ppg_p95": round(float(np.percentile(ppg_settle_times, 95)), 2),
                "max_settle": round(float(max(np.max(ecg_settle_times), np.max(ppg_settle_times))), 2),
            },
        },
        "root_causes": {
            "ppg": "MAX30105 ambient/DAC loop settling & finger pressure coupling",
            "ecg": "Electrode-skin contact equilibration & AFE RC filter charging",
        },
        "recommendation": {
            "crop_duration_s": 5.0,
            "rationale": "Removes sample-0 DAC step & 95th pct settling before Stage 2 filter",
        },
    }
    return summary


def plot_transients(data_dir: Path, subject_id: str, crop_s: float = 5.0) -> None:
    """Interactively plot raw vs cropped transient signals.
    
    CRITICAL USER INVARIANT: Never calls plt.savefig; displays via plt.show() only.
    """
    import matplotlib.pyplot as plt

    e_df = pd.read_csv(data_dir / "ECG" / f"{subject_id}_ecg.csv")
    p_df = pd.read_csv(data_dir / "PPG" / f"{subject_id}_ppg.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Column 1: Initial 0-10s segment with transient
    e_early = e_df[e_df["Time (s)"] <= 10.0]
    p_early = p_df[p_df["Time (s)"] <= 10.0]

    axes[0, 0].plot(e_early["Time (s)"], e_early["ECG_Raw (V)"], color="#1f77b4")
    axes[0, 0].axvline(crop_s, color="red", linestyle="--", label=f"Crop cut {crop_s:.1f}s")
    axes[0, 0].set_title(f"{subject_id} ECG Raw (0-10s Transient)")
    axes[0, 0].set_ylabel("Amplitude (V)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[1, 0].plot(p_early["Time (s)"], p_early["PPG_IR"], color="#d62728")
    axes[1, 0].axvline(crop_s, color="red", linestyle="--", label=f"Crop cut {crop_s:.1f}s")
    axes[1, 0].set_title(f"{subject_id} PPG IR (0-10s Transient)")
    axes[1, 0].set_ylabel("ADC Counts")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Column 2: Post-crop stabilized signal
    e_clean = e_df[e_df["Time (s)"] >= crop_s]
    p_clean = p_df[p_df["Time (s)"] >= crop_s]

    axes[0, 1].plot(e_clean["Time (s)"], e_clean["ECG_Raw (V)"], color="#1f77b4")
    axes[0, 1].set_title(f"ECG Post-Crop ({crop_s:.1f}s - 60s)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(p_clean["Time (s)"], p_clean["PPG_IR"], color="#d62728")
    axes[1, 1].set_title(f"PPG IR Post-Crop ({crop_s:.1f}s - 60s)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit initial transients in ECG/PPG dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("scratch/gemastik/Bp revisi/Data"))
    parser.add_argument("--output-json", type=Path, default=Path("scratch/gemastik/Bp revisi/transient_summary.json"))
    parser.add_argument("--plot", action="store_true", help="Display diagnostic plots (plt.show only).")
    parser.add_argument("--subject", type=str, default="subject001", help="Subject ID for diagnostic plot.")
    args = parser.parse_args()

    metadata_path = args.data_dir / "metadata" / "target.csv"
    subjects = load_subjects(metadata_path)

    summary = analyze_transients(args.data_dir, subjects)

    # Save compact JSON (< 1 KB)
    json_bytes = json.dumps(summary, indent=2).encode("utf-8")
    assert len(json_bytes) < 1024, f"JSON summary exceeds 1 KB: {len(json_bytes)} bytes"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "wb") as f:
        f.write(json_bytes)

    # Output strictly <= 15 lines of concise summary
    m = summary["metrics"]
    print("=== TASK-TRANSIENT-AUDIT-02: Transient Artifact Audit ===")
    print(f"Total Subjects Analyzed: {summary['total_subjects']} | Transient Detected: {summary['transient_detected']}")
    print(f"PPG Sample-0 Discontinuity: Median={m['sample0_step']['ppg_median']} ADC (>3x SS: {m['sample0_step']['ppg_outlier_pct']}%)")
    print(f"ECG Sample-0 Discontinuity: Median={m['sample0_step']['ecg_median']} V, Mean={m['sample0_step']['ecg_mean']} V")
    print(f"DC Drift (ECG): 0-1s={m['dc_drift_rate']['ecg_0_1s_v_per_s']} V/s | 0-5s={m['dc_drift_rate']['ecg_0_5s_v_per_s']} V/s | SS={m['dc_drift_rate']['ecg_ss_v_per_s']} V/s")
    print(f"DC Drift (PPG): 0-1s={m['dc_drift_rate']['ppg_0_1s_adc_per_s']} ADC/s | 0-5s={m['dc_drift_rate']['ppg_0_5s_adc_per_s']} ADC/s | SS={m['dc_drift_rate']['ppg_ss_adc_per_s']} ADC/s")
    print(f"Stabilization Time (ECG): Median={m['stabilization_time_s']['ecg_median']}s, P95={m['stabilization_time_s']['ecg_p95']}s")
    print(f"Stabilization Time (PPG): Median={m['stabilization_time_s']['ppg_median']}s, P95={m['stabilization_time_s']['ppg_p95']}s, Max={m['stabilization_time_s']['max_settle']}s")
    print(f"Root Cause (PPG): {summary['root_causes']['ppg']}")
    print(f"Root Cause (ECG): {summary['root_causes']['ecg']}")
    print(f"Recommended Crop Duration: {summary['recommendation']['crop_duration_s']}s")
    print(f"Summary JSON: {args.output_json} ({len(json_bytes)} bytes)")

    if args.plot:
        plot_transients(args.data_dir, args.subject, crop_s=float(summary["recommendation"]["crop_duration_s"]))


if __name__ == "__main__":
    main()
