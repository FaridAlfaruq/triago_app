import numpy as np
from scipy.signal import butter, sosfiltfilt


def temperature_samples(values, low, high):
    samples = np.asarray(values, dtype=float)
    return np.where(np.isfinite(samples) & (samples >= low) & (samples <= high), samples, np.nan)


def perfusion_percent(ac, dc):
    if not np.isfinite(ac) or not np.isfinite(dc) or ac <= 0 or dc <= 0:
        return None
    pi = 100.0 * (ac / dc)
    return float(pi) if np.isfinite(pi) else None


def ir_perfusion_percent(raw_ir, fs):
    if raw_ir is None:
        return None
    samples = np.asarray(raw_ir, dtype=float)
    if samples.size < 2 * fs or not np.isfinite(samples).all() or np.any(samples <= 0):
        return None
    ac = sosfiltfilt(butter(4, [0.5, 10], fs=fs, btype="bandpass", output="sos"), samples)
    dc = sosfiltfilt(butter(4, 0.5, fs=fs, output="sos"), samples)
    low, high = np.percentile(ac, [5, 95])
    return perfusion_percent(high - low, np.mean(dc)) if np.ptp(samples) > 0 else None


def estimate_core_temperature(skin_c, ambient_c, pi_percent):
    if any(value is None or not np.isfinite(value) for value in (skin_c, ambient_c, pi_percent)):
        return None
    if pi_percent <= 0:
        return None
    return float(skin_c + (skin_c - ambient_c) / (0.82991 + 0.97126 * pi_percent) + 1.06490)
