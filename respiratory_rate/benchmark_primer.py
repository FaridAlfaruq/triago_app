"""Benchmark respiratory rate estimation pada dataset primer TriaGo (21 subjek).

Evaluasi terhadap Ground Truth klinis:
- data_primer/Data1.csv sampai Data21.csv
Membandingkan:
1. Legacy ECG
2. ECG Multi-EDR Fusion
3. PPG Multi-Modulation Fusion
4. Multimodal Smart Fusion (ECG + PPG)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from processing_data.processing_data import ECGProcessor, PPGProcessor
from respiratory_rate.multimodal_pipeline import MultimodalRespirationEstimator

DATA_PRIMER_DIR = PROJECT_ROOT / "data_primer"
OUTPUT_DIR = PROJECT_ROOT / "respiratory_rate" / "results"


def run_benchmark_primer():
  ecg_proc = ECGProcessor(target_fs=125)
  ppg_proc = PPGProcessor(target_fs=125)
  multi_est = MultimodalRespirationEstimator(
      window_seconds=60.0, step_seconds=15.0, min_duration=10.0
  )

  files = sorted(list(DATA_PRIMER_DIR.glob("Data*.csv")), key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0))
  print(f"Found {len(files)} primary dataset files in {DATA_PRIMER_DIR}")

  rows = []
  for f in files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()

    raw_time = df["Time (s)"].to_numpy(dtype=float)
    raw_red = df["PPG_Red"].to_numpy(dtype=float)
    raw_ir = df["PPG_IR"].to_numpy(dtype=float)
    raw_ecg = df["ECG"].to_numpy(dtype=float)

    gt_rr = float(df["Respiratory_Rate_Ground_Truth"].iloc[0])
    gt_hr = float(df["HR_Ground_Truth"].iloc[0])

    # 1. ECG Preprocessing
    ecg_125, time_125 = ecg_proc.downsample(raw_ecg, raw_time, 400)
    sig_notch = ecg_proc.notch(ecg_125, freq=50.0, fs=125)
    sig_detrend = ecg_proc.detrending(sig_notch, fs=125)
    sig_lpf = ecg_proc.lowpass(sig_detrend, lowcut=35.0, fs=125)
    ecg_smooth = ecg_proc.savgol(sig_lpf, window_size=11, poly_order=2)
    r_peaks, _ = ecg_proc.detect_r_peaks(ecg_125, fs=125)

    # 2. PPG Preprocessing
    ppg_res = ppg_proc.process_ppg(raw_time, raw_red, raw_ir, fs_orig=400)
    ir_clean = ppg_res["ir_clean"]
    ir_peaks = ppg_res["ir_peaks"]

    # 3. Method 1: Legacy ECG
    rr_leg, _, _ = ecg_proc.calculate_respiration_rate_legacy(ecg_smooth, r_peaks, fs=125)

    # 4. Method 2: Current ECG-only Fusion
    rr_ecg, _, _ = ecg_proc.calculate_respiration_rate(ecg_smooth, r_peaks, fs=125)

    # 5. Method 3: PPG-only Fusion
    res_ppg = multi_est.estimate(ecg=None, r_peaks=None, ppg=ir_clean, ppg_peaks=ir_peaks, fs=125)
    rr_ppg = res_ppg["rr"]

    # 6. Method 4: Multimodal Smart Fusion (ECG + PPG)
    res_multi = multi_est.estimate(
        ecg=ecg_smooth, r_peaks=r_peaks, ppg=ir_clean, ppg_peaks=ir_peaks, fs=125
    )
    rr_multi = res_multi["rr"]

    rows.append({
        "file": f.name,
        "gt_rr": gt_rr,
        "gt_hr": gt_hr,
        "rr_legacy": rr_leg,
        "rr_ecg_fusion": rr_ecg,
        "rr_ppg_fusion": rr_ppg,
        "rr_multimodal": rr_multi,
        "quality_multi": res_multi.get("quality", 0.0),
        "mode_multi": res_multi.get("mode", ""),
    })

  res_df = pd.DataFrame(rows)

  # Summaries
  summaries = []
  for method in ("legacy", "ecg_fusion", "ppg_fusion", "multimodal"):
    col = f"rr_{method}"
    err = res_df[col] - res_df["gt_rr"]
    abs_err = err.abs()
    summaries.append({
        "method": method,
        "n_evaluated": len(res_df),
        "mae_bpm": float(np.round(abs_err.mean(), 3)),
        "rmse_bpm": float(np.round(np.sqrt(np.mean(err**2)), 3)),
        "bias_bpm": float(np.round(err.mean(), 3)),
        "mape_percent": float(np.round(100.0 * np.mean(abs_err / res_df["gt_rr"]), 2)),
        "within_2_bpm_percent": float(np.round(100.0 * np.mean(abs_err <= 2.0), 2)),
    })
  summary_df = pd.DataFrame(summaries)

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  res_df.to_csv(OUTPUT_DIR / "primary_data_results.csv", index=False)
  summary_df.to_csv(OUTPUT_DIR / "primary_data_summary.csv", index=False)

  print("\n" + "=" * 80)
  print(f"PRIMARY DATASET BENCHMARK RESULTS (N={len(res_df)} subjects)")
  print("=" * 80)
  print(summary_df.to_string(index=False))
  print("=" * 80)
  print("\nDetail per subjek:")
  print(res_df[["file", "gt_rr", "rr_legacy", "rr_ecg_fusion", "rr_ppg_fusion", "rr_multimodal", "mode_multi"]].to_string(index=False))


if __name__ == "__main__":
  run_benchmark_primer()
