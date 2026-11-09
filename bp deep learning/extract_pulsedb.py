"""
Extract clean 7-channel segments from PulseDB v2.0 (VitalDB + MIMIC).
Input path: G:\.shortcut-targets-by-id\10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV\PulseDB_v2_0
"""

import time
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import hilbert


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + eps)


def extract_7ch(ecg: np.ndarray, ppg: np.ndarray) -> np.ndarray:
    """Extract 7-channel feature tensor [1250, 7]."""
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


def main():
    base_pulse = Path(r"G:\.shortcut-targets-by-id\10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV\PulseDB_v2_0\Segment_Files")
    vital_files = sorted((base_pulse / "PulseDB_Vital").glob("*.mat"))
    mimic_files = sorted((base_pulse / "PulseDB_MIMIC").glob("*.mat"))

    print(f"Available files: {len(vital_files)} VitalDB, {len(mimic_files)} MIMIC")

    # Sample 25 VitalDB + 15 MIMIC subjects (total 40 subjects)
    selected_files = vital_files[:25] + mimic_files[:15]

    all_X, all_s, all_d, all_g = [], [], [], []
    max_segs_per_subj = 150
    t0 = time.time()

    for idx, fpath in enumerate(selected_files, 1):
        subj_name = fpath.stem
        try:
            with h5py.File(fpath, "r") as hf:
                g = hf["Subj_Wins"]
                n = g["ECG_F"].shape[1]
                inc_flags = [int(np.array(hf[g["IncludeFlag"][0, i]]).item()) for i in range(n)]

                valid_indices = [i for i in range(n) if inc_flags[i] == 1]
                if len(valid_indices) > max_segs_per_subj:
                    # Select uniformly spaced segments across recording
                    valid_indices = np.linspace(0, len(valid_indices) - 1, max_segs_per_subj, dtype=int).tolist()

                for i in valid_indices:
                    ref_ecg = g["ECG_F"][0, i]
                    ref_ppg = g["PPG_F"][0, i]
                    ref_sbp = g["SegSBP"][0, i]
                    ref_dbp = g["SegDBP"][0, i]

                    ecg = np.array(hf[ref_ecg]).flatten()
                    ppg = np.array(hf[ref_ppg]).flatten()
                    sbp = float(np.array(hf[ref_sbp]).item())
                    dbp = float(np.array(hf[ref_dbp]).item())

                    # Filter physiological bounds
                    if len(ecg) != 1250 or len(ppg) != 1250:
                        continue
                    if not (70 <= sbp <= 200 and 40 <= dbp <= 120 and sbp > dbp):
                        continue

                    feat7 = extract_7ch(ecg, ppg)
                    all_X.append(feat7)
                    all_s.append(sbp)
                    all_d.append(dbp)
                    all_g.append(subj_name)

            print(f"[{idx:02d}/40] {subj_name}: added {len(all_X) - (sum(len(x) for x in all_X[:-len(valid_indices)]) if all_X else 0)} segs | Total: {len(all_X)}")
        except Exception as e:
            print(f"[{idx:02d}/40] {subj_name} failed: {e}")

    X_arr = np.array(all_X, dtype=np.float32)
    s_arr = np.array(all_s, dtype=np.float32)
    d_arr = np.array(all_d, dtype=np.float32)
    g_arr = np.array(all_g)

    out_path = Path("scratch/gemastik/pulsedb_benchmark_7ch.npz")
    np.savez_compressed(out_path, X=X_arr, y_sbp=s_arr, y_dbp=d_arr, groups=g_arr)

    print(f"\nSaved {len(X_arr)} segments from {len(np.unique(g_arr))} patients to {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Elapsed: {time.time()-t0:.2f}s")
    print(f"SBP: mean={s_arr.mean():.1f}, std={s_arr.std():.1f}, range=[{s_arr.min():.1f}, {s_arr.max():.1f}]")
    print(f"DBP: mean={d_arr.mean():.1f}, std={d_arr.std():.1f}, range=[{d_arr.min():.1f}, {d_arr.max():.1f}]")


if __name__ == "__main__":
    main()
