"""Bandingkan pipeline RR lama dan baru pada dataset BIDMC."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from processing_data.processing_data import ECGProcessor


FS = 125.0
DEFAULT_DATA = PROJECT_ROOT / "draft_filter" / "bidmc"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def load_subject(subject_id, data_dir):
  name = f"bidmc_{subject_id:02d}"
  signal_path = data_dir / f"{name}_Signals.csv"
  breath_path = data_dir / f"{name}_Breaths.csv"
  if not signal_path.exists() or not breath_path.exists():
    return None, None
  return (
      pd.read_csv(signal_path, skipinitialspace=True),
      pd.read_csv(breath_path, skipinitialspace=True),
  )


def get_annotations(breaths):
  """Ubah nomor sampel anotasi menjadi waktu napas dalam detik."""
  annotations = []
  for column in breaths.columns:
    samples = pd.to_numeric(breaths[column], errors="coerce").dropna()
    if len(samples):
      annotations.append(samples.to_numpy(dtype=float) / FS)
  return annotations


def get_reference_rr(annotations, start, end):
  """Hitung ground-truth RR dari median jarak napas dua anotator."""
  estimates = []
  for times in annotations:
    selected = times[(times >= start) & (times < end)]
    intervals = np.diff(selected)
    intervals = intervals[(intervals >= 0.7) & (intervals <= 10.0)]
    if len(intervals) >= 2:
      estimates.append(60.0 / np.median(intervals))
  return float(np.mean(estimates)) if estimates else np.nan


def evaluate_subject(subject_id, data_dir, window=60.0, step=30.0):
  signals, breaths = load_subject(subject_id, data_dir)
  if signals is None:
    return []

  ecg = pd.to_numeric(signals["II"], errors="coerce").interpolate().to_numpy()
  annotations = get_annotations(breaths)
  processor = ECGProcessor(target_fs=int(FS))
  duration = len(ecg) / FS
  starts = list(np.arange(0.0, duration - window + 1e-9, step))
  final_start = duration - window
  if final_start >= 0 and (not starts or final_start - starts[-1] > 0.5):
    starts.append(final_start)

  results = []
  for start in starts:
    end = start + window
    segment = ecg[int(start * FS) : int(end * FS)]
    r_peaks, _ = processor.detect_r_peaks(segment, fs=FS)
    old_rr, _, _ = processor.calculate_respiration_rate_legacy(
        segment, r_peaks, fs=FS
    )
    new_rr, _, _ = processor.calculate_respiration_rate(
        segment, r_peaks, fs=FS
    )
    details = processor.last_respiration_details or {}
    results.append({
        "subject": f"bidmc_{subject_id:02d}",
        "start_seconds": start,
        "end_seconds": end,
        "n_r_peaks": len(r_peaks),
        "rr_reference": get_reference_rr(annotations, start, end),
        "rr_legacy": old_rr if old_rr > 0 else np.nan,
        "rr_fusion": new_rr if new_rr > 0 else np.nan,
        "fusion_quality": details.get("quality", 0.0),
    })
  return results


def make_summary(results):
  summaries = []
  for method in ("legacy", "fusion"):
    estimate = f"rr_{method}"
    valid = results.dropna(subset=["rr_reference", estimate])
    error = valid[estimate] - valid["rr_reference"]
    absolute_error = error.abs()
    summaries.append({
        "method": method,
        "n_windows": len(valid),
        "coverage_percent": 100.0 * len(valid) / len(results),
        "mae_bpm": absolute_error.mean(),
        "rmse_bpm": np.sqrt(np.mean(error**2)),
        "bias_bpm": error.mean(),
        "mape_percent": 100.0 * np.mean(
            absolute_error / valid["rr_reference"]
        ),
        "within_2_bpm_percent": 100.0 * np.mean(absolute_error <= 2.0),
    })
  return pd.DataFrame(summaries)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--first-subject", type=int, default=1)
  parser.add_argument("--last-subject", type=int, default=53)
  args = parser.parse_args()

  rows = []
  for subject_id in range(args.first_subject, args.last_subject + 1):
    subject_rows = evaluate_subject(subject_id, args.data_dir)
    rows.extend(subject_rows)
    print(f"bidmc_{subject_id:02d}: {len(subject_rows)} windows")

  if not rows:
    raise SystemExit("Data BIDMC tidak ditemukan.")

  results = pd.DataFrame(rows)
  for method in ("legacy", "fusion"):
    error = results[f"rr_{method}"] - results["rr_reference"]
    results[f"{method}_error"] = error
    results[f"{method}_abs_error"] = error.abs()
    results[f"{method}_within_2"] = error.abs() <= 2.0

  summary = make_summary(results)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  results.to_csv(args.output_dir / "bidmc_window_results.csv", index=False)
  summary.to_csv(args.output_dir / "bidmc_summary.csv", index=False)

  print("\nRINGKASAN BENCHMARK ECG-ONLY")
  print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
  main()
