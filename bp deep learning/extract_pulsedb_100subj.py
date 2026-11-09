"""
extract_pulsedb_100subj.py
Extract clean 7-channel waveform segments from PulseDB v2.0 (VitalDB + MIMIC).
Produces a 100-subject dataset (60 VitalDB + 40 MIMIC), up to 80 segments per subject.
Channels: [ECG, PPG, VPG, APG, ECG*PPG Corr, ECG Envelope, PPG Envelope] @ 125 Hz.
"""

import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import hilbert


def resolve_pulsedb_root() -> Path:
    candidates = [
        Path(r"G:\.shortcut-targets-by-id\10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV\PulseDB_v2_0\Segment_Files"),
        Path(r"G:\My Drive\TriaGO_Project\PulseDB_v2_0\Segment_Files"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find PulseDB Segment_Files directory at specified paths.")


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + eps)


def extract_7ch(ecg: np.ndarray, ppg: np.ndarray) -> np.ndarray:
    ecg_norm = zscore(ecg)
    ppg_norm = zscore(ppg)
    vpg_norm = zscore(np.gradient(ppg_norm))
    apg_norm = zscore(np.gradient(vpg_norm))
    corr_norm = zscore(ecg_norm * ppg_norm)
    ecg_env_norm = zscore(np.abs(hilbert(ecg_norm)))
    ppg_env_norm = zscore(np.abs(hilbert(ppg_norm)))
    return np.column_stack([
        ecg_norm, ppg_norm, vpg_norm, apg_norm, corr_norm, ecg_env_norm, ppg_env_norm
    ]).astype(np.float32)


def extract_subject_segments(hf: h5py.File, max_segs: int = 80):
    g = hf["Subj_Wins"]
    n = g["ECG_F"].shape[1]
    inc_refs = g["IncludeFlag"][0]
    sbp_refs = g["SegSBP"][0]
    dbp_refs = g["SegDBP"][0]

    # Sample candidate indices uniformly across the entire recording
    cand_count = min(n, 200)
    cand_indices = np.linspace(0, n - 1, cand_count, dtype=int) if n > cand_count else np.arange(n)

    valid_candidates = []
    for i in cand_indices:
        try:
            if hf[inc_refs[i]][0, 0] != 1:
                continue
            sbp = float(hf[sbp_refs[i]][0, 0])
            dbp = float(hf[dbp_refs[i]][0, 0])
            if 70 <= sbp <= 200 and 40 <= dbp <= 120 and (sbp - dbp) >= 15:
                valid_candidates.append((i, sbp, dbp))
        except Exception:
            continue

    if len(valid_candidates) > max_segs:
        pick_pos = np.linspace(0, len(valid_candidates) - 1, max_segs, dtype=int)
        chosen = [valid_candidates[p] for p in pick_pos]
    else:
        chosen = valid_candidates

    sub_X, sub_sbp, sub_dbp = [], [], []
    for i, sbp, dbp in chosen:
        try:
            ecg = np.array(hf[g["ECG_F"][0, i]]).flatten()
            ppg = np.array(hf[g["PPG_F"][0, i]]).flatten()
            if len(ecg) != 1250 or len(ppg) != 1250:
                continue
            if np.isnan(ecg).any() or np.isnan(ppg).any():
                continue
            feat7 = extract_7ch(ecg, ppg)
            if np.isnan(feat7).any():
                continue
            sub_X.append(feat7)
            sub_sbp.append(sbp)
            sub_dbp.append(dbp)
        except Exception:
            continue

    return sub_X, sub_sbp, sub_dbp


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    t_start = time.time()
    print("PulseDB 100-subject extraction pipeline started.", flush=True)

    base_dir = resolve_pulsedb_root()
    vital_dir = base_dir / "PulseDB_Vital"
    mimic_dir = base_dir / "PulseDB_MIMIC"
    print(f"PulseDB root: {base_dir}", flush=True)

    target_vital = 60
    target_mimic = 40
    max_file_size_bytes = 250 * 1024 * 1024  # Skip multi-gigabyte outlier recordings

    vital_files = [f for f in sorted(vital_dir.glob("*.mat")) if f.stat().st_size <= max_file_size_bytes]
    mimic_files = [f for f in sorted(mimic_dir.glob("*.mat")) if f.stat().st_size <= max_file_size_bytes]

    all_X, all_sbp, all_dbp, all_groups = [], [], [], []

    # 1. VitalDB extraction
    print(f"VitalDB: targeting {target_vital} subjects (max 80 segs/subj)...", flush=True)
    vital_count = 0
    for fpath in vital_files:
        if vital_count >= target_vital:
            break
        try:
            with h5py.File(fpath, "r") as hf:
                s_X, s_s, s_d = extract_subject_segments(hf, max_segs=80)
            if len(s_X) > 0:
                all_X.extend(s_X)
                all_sbp.extend(s_s)
                all_dbp.extend(s_d)
                all_groups.extend([fpath.stem] * len(s_X))
                vital_count += 1
                if vital_count in (20, 40):
                    print(f"[VitalDB {vital_count}/{target_vital}] {len(all_X)} segments accumulated.", flush=True)
        except Exception:
            continue
    print(f"[VitalDB {vital_count}/{target_vital}] Complete: {len(all_X)} segments accumulated.", flush=True)

    # 2. MIMIC extraction
    print(f"MIMIC: targeting {target_mimic} subjects (max 80 segs/subj)...", flush=True)
    mimic_count = 0
    vital_segs = len(all_X)
    for fpath in mimic_files:
        if mimic_count >= target_mimic:
            break
        try:
            with h5py.File(fpath, "r") as hf:
                s_X, s_s, s_d = extract_subject_segments(hf, max_segs=80)
            if len(s_X) > 0:
                all_X.extend(s_X)
                all_sbp.extend(s_s)
                all_dbp.extend(s_d)
                all_groups.extend([fpath.stem] * len(s_X))
                mimic_count += 1
                if mimic_count in (15, 30):
                    print(f"[MIMIC {mimic_count}/{target_mimic}] {len(all_X) - vital_segs} segments accumulated.", flush=True)
        except Exception:
            continue
    print(f"[MIMIC {mimic_count}/{target_mimic}] Complete: {len(all_X) - vital_segs} segments accumulated.", flush=True)

    # 3. Format and save NPZ
    X_arr = np.array(all_X, dtype=np.float32)
    sbp_arr = np.array(all_sbp, dtype=np.float32)
    dbp_arr = np.array(all_dbp, dtype=np.float32)
    groups_arr = np.array(all_groups)

    out_npz = Path("scratch/gemastik/pulsedb_100subj_7ch.npz")
    print(f"Saving compressed dataset to {out_npz}...", flush=True)
    np.savez_compressed(out_npz, X=X_arr, y_sbp=sbp_arr, y_dbp=dbp_arr, groups=groups_arr)

    npz_size_mb = out_npz.stat().st_size / (1024 * 1024)
    print(f"Dataset saved: shape={X_arr.shape}, size={npz_size_mb:.2f} MB.", flush=True)

    # 4. SHA-256 and JSON Summary
    print("Calculating SHA-256 hash and writing summary JSON...", flush=True)
    sha256_str = compute_sha256(out_npz)

    summary = {
        "n_subjects": int(len(np.unique(groups_arr))),
        "n_vital_subjects": int(vital_count),
        "n_mimic_subjects": int(mimic_count),
        "total_segments": int(len(X_arr)),
        "tensor_shape": list(X_arr.shape),
        "sbp_mean": round(float(np.mean(sbp_arr)), 2),
        "sbp_std": round(float(np.std(sbp_arr)), 2),
        "sbp_min": round(float(np.min(sbp_arr)), 2),
        "sbp_max": round(float(np.max(sbp_arr)), 2),
        "dbp_mean": round(float(np.mean(dbp_arr)), 2),
        "dbp_std": round(float(np.std(dbp_arr)), 2),
        "dbp_min": round(float(np.min(dbp_arr)), 2),
        "dbp_max": round(float(np.max(dbp_arr)), 2),
        "file_size_mb": round(npz_size_mb, 2),
        "sha256": sha256_str,
    }

    out_json = Path("scratch/gemastik/stage1_dataset_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    json_size_bytes = out_json.stat().st_size
    assert json_size_bytes < 1024, f"JSON size {json_size_bytes} exceeds 1 KB limit!"

    elapsed = time.time() - t_start
    print(f"Summary JSON: {out_json} ({json_size_bytes} B, sha256={sha256_str[:12]}...).", flush=True)
    print(f"Extraction pipeline complete in {elapsed:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
