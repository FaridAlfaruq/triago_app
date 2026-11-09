"""
load.py
Module for loading true raw PulseDB signals directly from PulseDB v2.0
at 'G:\\My Drive\\TriaGO_Project\\PulseDB_v2_0.lnk' (.mat HDF5 format)
and visualizing raw signals in a 2x2 grid (Figure 1: Whole vs 10s Zoom).
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Default PulseDB raw shortcut / path
DEFAULT_PULSEDB_PATH = Path(r"G:\My Drive\TriaGO_Project\PulseDB_v2_0.lnk")


def resolve_pulsedb_root(path: str | Path = DEFAULT_PULSEDB_PATH) -> Path:
    """
    Resolve PulseDB root directory from Windows shortcut (.lnk) or direct path.
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".lnk":
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            target = Path(shell.CreateShortCut(str(p.resolve())).TargetPath)
            if target.exists() and (target / "Segment_Files").exists():
                return target
        except Exception:
            pass

    fallbacks = [
        p,
        Path(r"G:\.shortcut-targets-by-id\10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV\PulseDB_v2_0"),
        Path(r"G:\My Drive\TriaGO_Project\PulseDB_v2_0"),
    ]
    for fb in fallbacks:
        if fb.exists() and (fb / "Segment_Files").exists():
            return fb

    raise FileNotFoundError(f"PulseDB_v2_0 raw directory not found from: {path}")


def list_available_subjects(
    pulsedb_root: Optional[Path] = None,
    source: str = "PulseDB_Vital",
) -> List[str]:
    """
    List subject identifiers available in the PulseDB segment directory.
    """
    root = pulsedb_root or resolve_pulsedb_root()
    folder = root / "Segment_Files" / source
    if not folder.exists():
        raise FileNotFoundError(f"Segment folder not found: {folder}")
    return sorted([f.stem for f in folder.glob("*.mat")])


def load_raw_subject(
    subject_id: str = "p003705",
    max_segments: int = 50,
    source: str = "PulseDB_Vital",
    pulsedb_root: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load true raw signals (ECG_Raw, PPG_Raw, ABP_Raw, SegSBP, SegDBP) from PulseDB .mat file.

    Parameters:
        subject_id: Subject file stem (e.g. 'p003705').
        max_segments: Number of 10s windows to read (default: 50 = 500 seconds).
        source: 'PulseDB_Vital' or 'PulseDB_MIMIC'.
        pulsedb_root: Resolved PulseDB root path.

    Returns:
        ecg_raw: Array of shape (N_segments, 1250)
        ppg_raw: Array of shape (N_segments, 1250)
        abp_raw: Array of shape (N_segments, 1250)
        sbp: Array of shape (N_segments,)
        dbp: Array of shape (N_segments,)
    """
    root = pulsedb_root or resolve_pulsedb_root()
    mat_path = root / "Segment_Files" / source / f"{subject_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Subject file not found: {mat_path}")

    with h5py.File(mat_path, "r") as hf:
        g = hf["Subj_Wins"]
        total_segs = g["ECG_Raw"].shape[1]
        n = min(max_segments, total_segs) if max_segments else total_segs

        ecg_list, ppg_list, abp_list, sbp_list, dbp_list = [], [], [], [], []
        for i in range(n):
            ecg_list.append(np.array(hf[g["ECG_Raw"][0, i]]).flatten())
            ppg_list.append(np.array(hf[g["PPG_Raw"][0, i]]).flatten())
            abp_list.append(np.array(hf[g["ABP_Raw"][0, i]]).flatten())
            sbp_list.append(float(np.array(hf[g["SegSBP"][0, i]]).item()))
            dbp_list.append(float(np.array(hf[g["SegDBP"][0, i]]).item()))

    return (
        np.array(ecg_list, dtype=np.float32),
        np.array(ppg_list, dtype=np.float32),
        np.array(abp_list, dtype=np.float32),
        np.array(sbp_list, dtype=np.float32),
        np.array(dbp_list, dtype=np.float32),
    )


def get_dataset_summary(
    ecg: np.ndarray,
    ppg: np.ndarray,
    sbp: np.ndarray,
    dbp: np.ndarray,
    subject_id: str,
) -> Dict[str, Any]:
    """
    Summary statistics of the loaded raw PulseDB subject recording.
    """
    return {
        "dataset_name": "PulseDB v2.0 (True Raw Signals)",
        "subject_id": subject_id,
        "total_segments": int(len(ecg)),
        "samples_per_segment": int(ecg.shape[1]),
        "total_duration_seconds": float(len(ecg) * ecg.shape[1] / 125.0),
        "ecg_raw_range": [round(float(ecg.min()), 3), round(float(ecg.max()), 3)],
        "ppg_raw_range": [round(float(ppg.min()), 3), round(float(ppg.max()), 3)],
        "sbp_mean_std": [round(float(sbp.mean()), 1), round(float(sbp.std()), 1)],
        "dbp_mean_std": [round(float(dbp.mean()), 1), round(float(dbp.std()), 1)],
    }


def plot_signal(
    subject_id: str = "p003705",
    max_segments: int = 50,
    source: str = "PulseDB_Vital",
    fs: int = 125,
) -> None:
    """
    Figure 1: Visualize raw PulseDB signals in a 2x2 grid (2 columns x 2 rows):
    - Row 0: Whole Raw Signal (Col 0: ECG, Col 1: PPG)
    - Row 1: Zoomed-in 10-Second Window (Col 0: ECG, Col 1: PPG)
    """
    root = resolve_pulsedb_root()
    print(f"Loading true raw PulseDB subject '{subject_id}' from: {root / 'Segment_Files' / source}")

    ecg_raw, ppg_raw, abp_raw, sbp, dbp = load_raw_subject(
        subject_id=subject_id,
        max_segments=max_segments,
        source=source,
        pulsedb_root=root,
    )

    sbp_val = sbp[0]
    dbp_val = dbp[0]

    # Flatten signals across all consecutive 10s segments
    ecg_whole = ecg_raw.flatten()
    ppg_whole = ppg_raw.flatten()

    total_samples = len(ecg_whole)
    total_duration_s = total_samples / fs
    time_whole = np.linspace(0, total_duration_s, total_samples)

    # First 10-second segment (1250 samples)
    ecg_10s = ecg_raw[0]
    ppg_10s = ppg_raw[0]
    time_10s = np.linspace(0, 10.0, len(ecg_10s))

    # 2 Rows x 2 Columns layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # [Row 0, Col 0] ECG Whole Raw Signal
    ax = axes[0, 0]
    ax.plot(time_whole, ecg_whole, color="#1f77b4", linewidth=0.7)
    ax.axvspan(0, 10, color="red", alpha=0.25, label="10s Zoom Window")
    ax.set_title(f"ECG - Whole Raw Signal ({total_duration_s:.0f}s, {len(ecg_raw)} segments)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Raw Amplitude")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

    # [Row 0, Col 1] PPG Whole Raw Signal
    ax = axes[0, 1]
    ax.plot(time_whole, ppg_whole, color="#2ca02c", linewidth=0.7)
    ax.axvspan(0, 10, color="red", alpha=0.25, label="10s Zoom Window")
    ax.set_title(f"PPG - Whole Raw Signal ({total_duration_s:.0f}s | ABP Ref: {sbp_val:.0f}/{dbp_val:.0f} mmHg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Raw Amplitude")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

    # [Row 1, Col 0] ECG 10s Zoom Raw
    ax = axes[1, 0]
    ax.plot(time_10s, ecg_10s, color="#1f77b4", linewidth=1.2)
    ax.set_title("ECG - Zoomed 10-Second Raw Window (1250 samples @ 125 Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Raw Amplitude")
    ax.grid(True, linestyle=":", alpha=0.6)

    # [Row 1, Col 1] PPG 10s Zoom Raw
    ax = axes[1, 1]
    ax.plot(time_10s, ppg_10s, color="#2ca02c", linewidth=1.2)
    ax.set_title("PPG - Zoomed 10-Second Raw Window (1250 samples @ 125 Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Raw Amplitude")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(
        f"Figure 1: PulseDB Raw Multi-Channel Recording (Subject: {subject_id} | Source: {source})",
        fontsize=13,
        weight="bold",
    )
    plt.tight_layout()
    print("Displaying Figure 1 (ECG & PPG 2x2 grid) via plt.show()...")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load raw PulseDB dataset directly from PulseDB_v2_0.")
    parser.add_argument("--subject", type=str, default="p003705", help="Subject ID (default: p003705)")
    parser.add_argument("--segments", type=int, default=50, help="Number of 10s segments to load (default: 50)")
    parser.add_argument("--source", type=str, default="PulseDB_Vital", choices=["PulseDB_Vital", "PulseDB_MIMIC"])
    parser.add_argument("--no-plot", action="store_true", help="Print summary without showing plot window")
    args = parser.parse_args()

    if args.no_plot:
        ecg, ppg, abp, sbp, dbp = load_raw_subject(
            subject_id=args.subject,
            max_segments=args.segments,
            source=args.source,
        )
        summary = get_dataset_summary(ecg, ppg, sbp, dbp, args.subject)
        print("\n--- Raw PulseDB Subject Summary ---")
        for k, v in summary.items():
            print(f"{k:25s}: {v}")
    else:
        plot_signal(subject_id=args.subject, max_segments=args.segments, source=args.source)
