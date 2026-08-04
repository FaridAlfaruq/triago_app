"""Loaders untuk SpO2 Testing & Benchmarking.

Mendukung dua dataset PPG Dual-Wavelength (Red + IR):
1. dataset_primer_1: CSV (*_ppg.csv) & Metadata JSON (*_metadata.json) untuk ground-truth SpO2.
2. PhysioNet Pulse Transit Time PPG (WFDB): pleth_1 (Red 660 nm) + pleth_2 (IR 880 nm) & header comments.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PRIMER_DIR = os.path.join(_ROOT, "dataset_primer_1")
PHYSIONET_DIR = os.path.join(
    _ROOT, "physionet.org", "files", "pulse-transit-time-ppg", "1.1.0"
)


@dataclass
class PpgRecord:
    name: str
    red: np.ndarray
    ir: np.ndarray
    fs: float
    time: np.ndarray
    spo2_ref: Optional[float]      # ground-truth SpO2 (%) from metadata

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])


def _fs_from_time(time: np.ndarray) -> float:
    time = np.asarray(time, dtype=float)
    if time.size < 2:
        return 125.0
    return float((time.size - 1) / (time[-1] - time[0]))


def _to_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Dataset Primer 1 Loader (*_ppg.csv & *_metadata.json)
# --------------------------------------------------------------------------- #
def load_primer_record(ppg_path: str) -> PpgRecord:
    """Load satu record dari dataset_primer_1."""
    name = os.path.basename(ppg_path).replace("_ppg.csv", "")
    df = pd.read_csv(ppg_path)
    df.columns = df.columns.str.strip()
    time = df["Time (s)"].to_numpy(dtype=float)
    red = df["PPG_Red"].to_numpy(dtype=float)
    ir = df["PPG_IR"].to_numpy(dtype=float)

    fs = _fs_from_time(time)
    spo2 = None
    meta_path = ppg_path.replace("_ppg.csv", "_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        spo2 = _to_float(meta.get("ground_truth", {}).get("oxygen_saturation_percent"))
        specs = meta.get("signal_specs", {}).get("ppg", {})
        if specs.get("sampling_rate_hz"):
            fs = float(specs["sampling_rate_hz"])

    return PpgRecord(name, red, ir, fs, time, spo2)


def load_primer(primer_dir: str = PRIMER_DIR, limit: Optional[int] = None) -> List[PpgRecord]:
    """Load seluruh record di folder dataset_primer_1."""
    paths = sorted(glob.glob(os.path.join(primer_dir, "*_ppg.csv")))
    if limit:
        paths = paths[:limit]
    return [load_primer_record(p) for p in paths]


# --------------------------------------------------------------------------- #
# PhysioNet "Pulse Transit Time PPG" dataset (WFDB)
#   pleth_1 = RED (660 nm), pleth_2 = IR (880 nm), distal finger, fs = 500 Hz.
#   Ground-truth SpO2 is the mean of <spo2_start> and <spo2_end> in the header.
# --------------------------------------------------------------------------- #
def load_physionet_record(record_base: str) -> PpgRecord:
    """Load satu WFDB record berdasarkan base path-nya (tanpa ekstensi)."""
    import wfdb  # imported lazily agar tidak bergantung pada wfdb jika hanya memuat primer

    record_base = str(record_base)
    if record_base.endswith(".hea") or record_base.endswith(".dat"):
        record_base = os.path.splitext(record_base)[0]

    name = os.path.basename(record_base)
    rec = wfdb.rdrecord(record_base, channel_names=["pleth_1", "pleth_2"])
    fs = float(rec.fs)
    red = rec.p_signal[:, rec.sig_name.index("pleth_1")].astype(float)
    ir = rec.p_signal[:, rec.sig_name.index("pleth_2")].astype(float)
    min_len = min(len(red), len(ir))
    red = red[:min_len]
    ir = ir[:min_len]
    time = np.arange(min_len) / fs

    comment = " ".join(rec.comments) if rec.comments else ""
    vals = [int(m) for m in re.findall(r"<spo2_(?:start|end)>:\s*(\d+)", comment)]
    spo2 = float(np.mean(vals)) if vals else None

    return PpgRecord(name, red, ir, fs, time, spo2)


def load_physionet(physionet_dir: str = PHYSIONET_DIR, limit: Optional[int] = None) -> List[PpgRecord]:
    """Load seluruh record PhysioNet dari folder target."""
    records_file = os.path.join(physionet_dir, "RECORDS")
    if os.path.exists(records_file):
        with open(records_file) as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        bases = [os.path.join(physionet_dir, n) for n in names]
    else:
        hea_paths = sorted(glob.glob(os.path.join(physionet_dir, "*.hea")))
        bases = [os.path.splitext(p)[0] for p in hea_paths]

    if limit:
        bases = bases[:limit]

    records = []
    for b in bases:
        try:
            records.append(load_physionet_record(b))
        except Exception as e:
            print(f"[WARN] Gagal memuat record WFDB {b}: {e}")
    return records
