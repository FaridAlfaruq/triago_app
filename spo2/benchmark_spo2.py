"""Benchmark Algoritma Estimasi SpO2 berbasis Dual-Wavelength PPG (Red & IR).

Mendukung otomatisasi pembacaan dari:
- Dataset Primer 1 (*_ppg.csv & *_metadata.json)
- PhysioNet Pulse Transit Time PPG WFDB (.hea/.dat)

Metode SpO2 Ratio-of-Ratios:
    R = (AC_red / DC_red) / (AC_ir / DC_ir)
    SpO2 = A * R^2 + B * R + C
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing_data.processing_data import PPGProcessor
from spo2.dataset_loader import (
    PpgRecord,
    load_physionet,
    load_primer,
)

DEFAULT_DATA_DIR = PROJECT_ROOT / "spo2" / "data"
if not DEFAULT_DATA_DIR.exists() or not list(DEFAULT_DATA_DIR.glob("*")):
    DEFAULT_DATA_DIR = PROJECT_ROOT / "draft_filter"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def evaluate_record_windowed(
    record: PpgRecord,
    processor: PPGProcessor,
    window_seconds: float = 30.0,
    step_seconds: float = 10.0,
) -> List[dict]:
    """Evaluasi satu PpgRecord dengan teknik sliding window."""
    if record.spo2_ref is None or not np.isfinite(record.spo2_ref):
        return []

    fs = record.fs
    duration = record.duration
    window_size = min(window_seconds, duration)

    starts = list(np.arange(0.0, duration - window_size + 1e-9, step_seconds))
    final_start = duration - window_size
    if final_start >= 0 and (not starts or final_start - starts[-1] > 0.5):
        starts.append(final_start)

    results = []
    for start in starts:
        end = start + window_size
        idx_start = int(round(start * fs))
        idx_end = int(round(end * fs))

        seg_time = record.time[idx_start:idx_end]
        seg_red = record.red[idx_start:idx_end]
        seg_ir = record.ir[idx_start:idx_end]

        if len(seg_time) < int(0.8 * window_size * fs):
            continue

        try:
            ppg_res = processor.process_ppg(
                raw_time=seg_time,
                raw_red=seg_red,
                raw_ir=seg_ir,
                fs_orig=fs,
            )

            spo2_est = ppg_res["spo2"]
            pi_red = ppg_res["pi_red"]
            pi_ir = ppg_res["pi_ir"]

            results.append({
                "record": record.name,
                "start_seconds": start,
                "end_seconds": end,
                "spo2_ref": float(record.spo2_ref),
                "spo2_est": float(spo2_est) if spo2_est > 0 else np.nan,
                "pi_red": pi_red,
                "pi_ir": pi_ir,
                "ppg_hr": ppg_res["ppg_hr"],
            })
        except Exception as e:
            print(f"[WARN] Error pada {record.name} [{start:.1f}-{end:.1f}s]: {e}")

    return results


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung statistik performa evaluasi SpO2."""
    valid = df.dropna(subset=["spo2_ref", "spo2_est"]).copy()
    if len(valid) == 0:
        return pd.DataFrame()

    error = valid["spo2_est"] - valid["spo2_ref"]
    abs_error = error.abs()

    mae = float(abs_error.mean())
    rmse = float(np.sqrt(np.mean(error**2)))
    bias = float(error.mean())
    mape = float(100.0 * np.mean(abs_error / valid["spo2_ref"]))
    within_2 = float(100.0 * np.mean(abs_error <= 2.0))
    within_3 = float(100.0 * np.mean(abs_error <= 3.0))
    coverage = float(100.0 * len(valid) / len(df))

    return pd.DataFrame([{
        "n_windows": len(valid),
        "coverage_percent": coverage,
        "mae_spo2_percent": mae,
        "rmse_spo2_percent": rmse,
        "bias_spo2_percent": bias,
        "mape_percent": mape,
        "accuracy_percent": 100.0 - mape,
        "within_2_percent": within_2,
        "within_3_percent": within_3,
    }])


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Algoritma SpO2 pada Dataset Dual-Wavelength PPG."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder dataset (misal: draft_filter, dataset_primer_1, atau physionet.org)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="auto",
        choices=["auto", "primer", "physionet"],
        help="Tipe dataset (auto, primer, physionet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder simpan hasil benchmark",
    )
    parser.add_argument(
        "--window", type=float, default=30.0, help="Durasi window analisis (detik)"
    )
    parser.add_argument(
        "--step", type=float, default=10.0, help="Step sliding window (detik)"
    )
    args = parser.parse_args()

    data_dir_str = str(args.data_dir)
    print(f"Membaca dataset dari: {data_dir_str}")

    records: List[PpgRecord] = []
    if args.dataset_type == "primer" or (
        args.dataset_type == "auto"
        and glob.glob(os.path.join(data_dir_str, "*_ppg.csv"))
    ):
        print("Tipe dataset terdeteksi: Dataset Primer 1 (*_ppg.csv)")
        records = load_primer(primer_dir=data_dir_str)
    else:
        print("Tipe dataset terdeteksi: PhysioNet WFDB (*.hea / *.dat)")
        records = load_physionet(physionet_dir=data_dir_str)

    print(f"Ditemukan {len(records)} record PPG dual-wavelength.")
    if not records:
        print(f"[ERROR] Tidak ada record PPG yang valid ditemukan di {data_dir_str}.")
        sys.exit(1)

    processor = PPGProcessor(target_fs=125)
    all_rows = []

    for rec in records:
        rows = evaluate_record_windowed(
            rec, processor, window_seconds=args.window, step_seconds=args.step
        )
        all_rows.extend(rows)
        print(f" - {rec.name}: {len(rows)} windows, Ground-Truth SpO2: {rec.spo2_ref}%")

    if not all_rows:
        print("[ERROR] Gagal memproses windows dari record yang ditemukan.")
        sys.exit(1)

    df_results = pd.DataFrame(all_rows)
    df_summary = make_summary(df_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(args.output_dir / "spo2_window_results.csv", index=False)
    df_summary.to_csv(args.output_dir / "spo2_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("RINGKASAN HASIL BENCHMARK SPO2 DUAL-WAVELENGTH")
    print("=" * 60)
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nHasil lengkap disimpan di folder: {args.output_dir}")


if __name__ == "__main__":
    main()
