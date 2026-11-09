"""Benchmark multimodal respiratory rate estimation pada dataset BIDMC (53 subjek).

Membandingkan 4 metode:
1. Legacy (ECG peak counting)
2. ECG Multi-EDR Fusion (baseline single-modality)
3. PPG Multi-Modulation Fusion (RIAV, RIIV, RIFV single-modality)
4. Multimodal Smart Fusion (ECG + PPG, Birrenkott et al. 2018 / Charlton et al. 2018)
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from processing_data.processing_data import ECGProcessor
from respiratory_rate.multimodal_pipeline import MultimodalRespirationEstimator

FS = 125.0
DEFAULT_DATA = PROJECT_ROOT / "respiratory_rate" / "data" / "bidmc"
if not DEFAULT_DATA.exists() or not list(DEFAULT_DATA.glob("*")):
  DEFAULT_DATA = PROJECT_ROOT / "bidmc"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def load_subject(subject_id: int, data_dir: Path):
  name = f"bidmc_{subject_id:02d}"
  sig_path = data_dir / f"{name}_Signals.csv"
  br_path = data_dir / f"{name}_Breaths.csv"
  if not sig_path.exists() or not br_path.exists():
    return None, None
  return (
      pd.read_csv(sig_path, skipinitialspace=True),
      pd.read_csv(br_path, skipinitialspace=True),
  )


def get_annotations(breaths: pd.DataFrame, fs: float = 125.0) -> list:
  annotations = []
  for col in breaths.columns:
    s = pd.to_numeric(breaths[col], errors="coerce").dropna()
    if len(s):
      annotations.append(s.to_numpy(dtype=float) / fs)
  return annotations


def get_reference_rr(annotations: list, start: float, end: float) -> float:
  counts = []
  for arr in annotations:
    sel = arr[(arr >= start) & (arr < end)]
    if len(sel) >= 2:
      intervals = np.diff(sel)
      if len(intervals) > 0 and np.mean(intervals) > 0:
        counts.append(60.0 / np.mean(intervals))
  return float(np.mean(counts)) if counts else np.nan


def evaluate_subject(
    subject_id: int,
    data_dir: Path,
    window_seconds: float = 60.0,
    step_seconds: float = 30.0,
) -> list:
  signals, breaths = load_subject(subject_id, data_dir)
  if signals is None or breaths is None:
    return []

  processor = ECGProcessor(target_fs=int(FS))
  multi_estimator = MultimodalRespirationEstimator(
      window_seconds=window_seconds, step_seconds=step_seconds
  )
  annotations = get_annotations(breaths, FS)
  duration = float(signals["Time [s]"].iloc[-1] - signals["Time [s]"].iloc[0])

  rows = []
  starts = np.arange(0.0, duration - window_seconds + 1e-9, step_seconds)
  for start in starts:
    end = start + window_seconds
    ref_rr = get_reference_rr(annotations, start, end)
    if np.isnan(ref_rr):
      continue

    mask = (signals["Time [s]"] >= start) & (signals["Time [s]"] < end)
    win_df = signals.loc[mask]
    ecg = win_df["II"].to_numpy(dtype=float)
    ppg = win_df["PLETH"].to_numpy(dtype=float)

    r_peaks, _ = processor.detect_r_peaks(ecg, fs=int(FS))

    # 1. Legacy ECG
    rr_legacy, _, _ = processor.calculate_respiration_rate_legacy(ecg, r_peaks, fs=int(FS))
    # 2. ECG Fusion (existing)
    rr_ecg_fusion, _, _ = processor.calculate_respiration_rate(ecg, r_peaks, fs=int(FS))
    # 3. PPG Only Fusion
    res_ppg = multi_estimator.estimate(ecg=None, r_peaks=None, ppg=ppg, ppg_peaks=None, fs=int(FS))
    rr_ppg_fusion = res_ppg["rr"]
    # 4. Multimodal Smart Fusion (ECG + PPG)
    res_multi = multi_estimator.estimate(ecg=ecg, r_peaks=r_peaks, ppg=ppg, ppg_peaks=None, fs=int(FS))
    rr_multimodal = res_multi["rr"]

    rows.append({
        "subject_id": subject_id,
        "start_seconds": start,
        "end_seconds": end,
        "rr_reference": ref_rr,
        "rr_legacy": rr_legacy,
        "rr_ecg_fusion": rr_ecg_fusion,
        "rr_ppg_fusion": rr_ppg_fusion,
        "rr_multimodal": rr_multimodal,
        "quality_ecg": processor.last_respiration_details.get("quality", 0.0) if processor.last_respiration_details else 0.0,
        "quality_ppg": res_ppg.get("quality", 0.0),
        "quality_multi": res_multi.get("quality", 0.0),
        "mode_multi": res_multi.get("mode", ""),
    })
  return rows


def make_summary(results: pd.DataFrame) -> pd.DataFrame:
  summaries = []
  total_windows = len(results)
  for method in ("legacy", "ecg_fusion", "ppg_fusion", "multimodal"):
    col = f"rr_{method}"
    valid = results[results[col] > 0]
    error = valid[col] - valid["rr_reference"]
    abs_err = error.abs()
    summaries.append({
        "method": method,
        "n_evaluated": len(valid),
        "coverage_percent": 100.0 * len(valid) / total_windows if total_windows > 0 else 0.0,
        "mae_bpm": float(np.round(abs_err.mean(), 3)),
        "rmse_bpm": float(np.round(np.sqrt(np.mean(error**2)), 3)),
        "bias_bpm": float(np.round(error.mean(), 3)),
        "mape_percent": float(np.round(100.0 * np.mean(abs_err / valid["rr_reference"]), 2)),
        "within_2_bpm_percent": float(np.round(100.0 * np.mean(abs_err <= 2.0), 2)),
    })
  return pd.DataFrame(summaries)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--first-subject", type=int, default=1)
  parser.add_argument("--last-subject", type=int, default=53)
  args = parser.parse_args()

  print(f"Starting BIDMC benchmark: Subjects {args.first_subject} to {args.last_subject}...")
  t0 = time.time()
  rows = []
  for sub_id in range(args.first_subject, args.last_subject + 1):
    sub_rows = evaluate_subject(sub_id, args.data_dir)
    rows.extend(sub_rows)
    print(f"bidmc_{sub_id:02d}: {len(sub_rows)} windows")

  if not rows:
    raise SystemExit("Data BIDMC tidak ditemukan.")

  results = pd.DataFrame(rows)
  summary = make_summary(results)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  results.to_csv(args.output_dir / "bidmc_multimodal_results.csv", index=False)
  summary.to_csv(args.output_dir / "bidmc_multimodal_summary.csv", index=False)

  print("\n" + "=" * 80)
  print(f"BENCHMARK COMPLETED in {time.time() - t0:.1f}s (Total Windows: {len(results)})")
  print("=" * 80)
  print(summary.to_string(index=False))
  print("=" * 80)


if __name__ == "__main__":
  main()
