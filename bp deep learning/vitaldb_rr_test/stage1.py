import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import vitaldb

VITAL_DIR = Path(__file__).resolve().parents[2] / "temp estimation" / "vital_files"
TRACKS = ["SNUADC/ECG_II", "SNUADC/PLETH", "Solar8000/VENT_RR"]
FS_RAW = 500.0
FS_DS = 125.0


def load_vital(case_id: int, fs: float):
    """Load case from local vital file, trimmed strictly to valid VENT_RR (> 0)."""
    path = VITAL_DIR / f"{case_id}.vital"
    if not path.exists():
        raise FileNotFoundError(f"Local file missing: {path}")

    vf = vitaldb.VitalFile(str(path))
    df = vf.to_pandas(TRACKS, interval=1.0 / fs)

    # Trim strictly to segment where VENT_RR has valid positive readings
    valid_mask = df["Solar8000/VENT_RR"].notna() & (df["Solar8000/VENT_RR"] > 0)
    valid_indices = df.index[valid_mask]
    if len(valid_indices) == 0:
        raise ValueError(f"No valid VENT_RR in Case {case_id}")

    df_valid = df.loc[valid_indices[0] : valid_indices[-1]].copy()
    df_valid["Solar8000/VENT_RR"] = df_valid["Solar8000/VENT_RR"].ffill()
    return df_valid.reset_index(drop=True)


def plot_comparison(case_id: int = 1, seconds: int = 10):
    """Plot 2 columns (500 Hz Raw vs 125 Hz Downsampled) for first N seconds of valid VENT_RR."""
    df_raw = load_vital(case_id, FS_RAW).iloc[: int(seconds * FS_RAW)]
    df_ds = load_vital(case_id, FS_DS).iloc[: int(seconds * FS_DS)]

    t_raw = np.arange(len(df_raw)) / FS_RAW
    t_ds = np.arange(len(df_ds)) / FS_DS

    fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex=True)

    # Column 1: Raw Signal (500 Hz)
    axes[0, 0].plot(t_raw, df_raw["SNUADC/ECG_II"], color="navy")
    axes[0, 0].set_title(f"Raw ECG II ({int(FS_RAW)} Hz)")
    axes[0, 0].set_ylabel("ECG II (mV)")

    axes[1, 0].plot(t_raw, df_raw["SNUADC/PLETH"], color="crimson")
    axes[1, 0].set_title(f"Raw PPG ({int(FS_RAW)} Hz)")
    axes[1, 0].set_ylabel("PPG (PLETH)")

    axes[2, 0].plot(t_raw, df_raw["Solar8000/VENT_RR"], color="green", lw=1.5)
    axes[2, 0].set_title("Raw VENT_RR (Valid)")
    axes[2, 0].set_ylabel("VENT_RR (rpm)")
    axes[2, 0].set_xlabel("Time (s)")

    # Column 2: Downsampled Signal (125 Hz)
    axes[0, 1].plot(t_ds, df_ds["SNUADC/ECG_II"], color="royalblue")
    axes[0, 1].set_title(f"Downsampled ECG II ({int(FS_DS)} Hz)")

    axes[1, 1].plot(t_ds, df_ds["SNUADC/PLETH"], color="indianred")
    axes[1, 1].set_title(f"Downsampled PPG ({int(FS_DS)} Hz)")

    axes[2, 1].plot(t_ds, df_ds["Solar8000/VENT_RR"], color="limegreen", lw=1.5)
    axes[2, 1].set_title("Downsampled VENT_RR (Valid)")
    axes[2, 1].set_xlabel("Time (s)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_comparison(case_id=1, seconds=10)
