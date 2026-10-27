"""Evaluasi seluruh rekaman klinis bed pada data_pengukuran/*.csv.

Menguji stabilitas estimasi RR pada data pengukuran GUI, termasuk file berdurasi 10 detik (BedA1).
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

DATA_PENGUKURAN_DIR = PROJECT_ROOT / "data_pengukuran"
OUTPUT_DIR = PROJECT_ROOT / "respiratory_rate" / "results"


def run_test_pengukuran():
  ecg_proc = ECGProcessor(target_fs=125)
  ppg_proc = PPGProcessor(target_fs=125)
  multi_est = MultimodalRespirationEstimator(
      window_seconds=60.0, step_seconds=15.0, min_duration=10.0
  )

  files = sorted(list(DATA_PENGUKURAN_DIR.glob("*.csv")))
  rows = []

  for f in files:
    # Skip non-measurement files seperti pendaftaran
    if "pendaftaran" in f.name.lower():
      continue

    try:
      df = pd.read_csv(f)
    except Exception as e:
      print(f"Skipping {f.name}: {e}")
      continue

    if "ECG_Raw" not in df.columns or "PPG_IR" not in df.columns:
      continue

    raw_time = df["Time (s)"].to_numpy(dtype=float)
    raw_red = df["PPG_Red"].to_numpy(dtype=float)
    raw_ir = df["PPG_IR"].to_numpy(dtype=float)
    raw_ecg = df["ECG_Raw"].to_numpy(dtype=float)

    fs_orig = 400.0 if len(raw_time) > 1 and (raw_time[1] - raw_time[0]) < 0.005 else 125.0
    duration = len(raw_time) / fs_orig

    # Preprocessing
    ecg_125, time_125 = ecg_proc.downsample(raw_ecg, raw_time, fs=int(fs_orig), fs_target=125)
    sig_notch = ecg_proc.notch(ecg_125, freq=50.0, fs=125)
    sig_detrend = ecg_proc.detrending(sig_notch, fs=125)
    sig_lpf = ecg_proc.lowpass(sig_detrend, lowcut=35.0, fs=125)
    ecg_smooth = ecg_proc.savgol(sig_lpf, window_size=11, poly_order=2)
    r_peaks, _ = ecg_proc.detect_r_peaks(ecg_125, fs=125)

    ppg_res = ppg_proc.process_ppg(raw_time, raw_red, raw_ir, fs_orig=int(fs_orig))
    ir_clean = ppg_res["ir_clean"]
    ir_peaks = ppg_res["ir_peaks"]

    # 1. Legacy
    rr_leg, _, _ = ecg_proc.calculate_respiration_rate_legacy(ecg_smooth, r_peaks, fs=125)

    # 2. Existing ECG-only
    rr_ecg, _, _ = ecg_proc.calculate_respiration_rate(ecg_smooth, r_peaks, fs=125)

    # 3. New Multimodal Smart Fusion
    res_multi = multi_est.estimate(
        ecg=ecg_smooth, r_peaks=r_peaks, ppg=ir_clean, ppg_peaks=ir_peaks, fs=125
    )
    rr_multi = res_multi["rr"]
    mode_multi = res_multi["mode"]
    quality_multi = res_multi["quality"]

    rows.append({
        "file": f.name,
        "duration_s": round(duration, 1),
        "hr_ecg": ecg_proc.calculate_heart_rate(r_peaks, fs=125),
        "spo2": ppg_res["spo2"],
        "rr_legacy": rr_leg,
        "rr_ecg_old": rr_ecg,
        "rr_multimodal": rr_multi,
        "mode": mode_multi,
        "quality": quality_multi,
    })

  res_df = pd.DataFrame(rows)
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  res_df.to_csv(OUTPUT_DIR / "pengukuran_bed_results.csv", index=False)

  print("\n" + "=" * 80)
  print(f"EVALUASI REKAMAN DATA PENGUKURAN BED CLINICAL (N={len(res_df)} files)")
  print("=" * 80)
  print(res_df[["file", "duration_s", "rr_legacy", "rr_ecg_old", "rr_multimodal", "mode", "quality"]].to_string(index=False))

  # Highlight short file handling:
  short_files = res_df[res_df["duration_s"] < 25.0]
  if not short_files.empty:
    print("\n--- HANDLING FILE DURASI PENDEK (< 25s) ---")
    print(short_files[["file", "duration_s", "rr_ecg_old", "rr_multimodal", "mode"]].to_string(index=False))


if __name__ == "__main__":
  run_test_pengukuran()
