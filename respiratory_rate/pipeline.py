"""Pipeline Respiratory Rate dari ECG.

Alur singkat:
ECG -> rapikan R-peak -> buat 5 sinyal EDR -> cek kualitas
    -> gabungkan spektrum terbaik -> Respiratory Rate.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch


class ECGRespirationEstimator:
  """Estimator RR ECG-only dengan multi-EDR spectral fusion."""

  def __init__(
      self,
      edr_fs=4.0,
      window_seconds=60.0,
      step_seconds=15.0,
      min_duration=20.0,
      min_rr=6.0,
      max_rr=42.0,
  ):
    self.edr_fs = edr_fs
    self.window_seconds = window_seconds
    self.step_seconds = step_seconds
    self.min_duration = min_duration
    self.low_hz = min_rr / 60.0
    self.high_hz = max_rr / 60.0

  # -----------------------------------------------------------------------
  # 1. FUNGSI DASAR
  # -----------------------------------------------------------------------

  @staticmethod
  def _bandpass(signal, low_hz, high_hz, fs, order=3):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 20:
      return signal.copy()
    sos = butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, signal)

  @staticmethod
  def _normalize(values):
    """Normalisasi robust agar outlier tidak terlalu memengaruhi hasil."""
    values = np.asarray(values, dtype=float)
    center = np.median(values)
    mad = np.median(np.abs(values - center))
    scale = 1.4826 * mad
    if scale < 1e-9:
      scale = np.std(values)
    if scale < 1e-9:
      return np.zeros_like(values)
    return np.clip((values - center) / scale, -6.0, 6.0)

  @staticmethod
  def _weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
      return np.nan

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    middle = 0.5 * np.sum(weights)
    return float(values[np.searchsorted(np.cumsum(weights), middle)])

  # -----------------------------------------------------------------------
  # 2. R-PEAK DAN EKSTRAKSI LIMA SINYAL EDR
  # -----------------------------------------------------------------------

  def _refine_r_peaks(self, ecg, r_peaks, fs):
    """Geser kandidat R-peak ke ekstremum QRS terdekat."""
    qrs_signal = self._bandpass(ecg, 5.0, min(25.0, 0.45 * fs), fs)
    radius = int(round(0.12 * fs))
    refined = []

    for candidate in np.asarray(r_peaks, dtype=int):
      left = max(0, candidate - radius)
      right = min(len(ecg), candidate + radius + 1)
      segment = qrs_signal[left:right]
      if len(segment) < 3:
        continue
      peak = left + np.argmax(np.abs(segment - np.median(segment)))

      # Hindari dua kandidat jatuh pada QRS yang sama.
      if not refined or peak - refined[-1] >= int(0.25 * fs):
        refined.append(int(peak))
      elif abs(qrs_signal[peak]) > abs(qrs_signal[refined[-1]]):
        refined[-1] = int(peak)

    return np.asarray(refined, dtype=int)

  def _extract_edr_features(self, ecg, r_peaks, fs):
    """Ambil amplitudo, area, slope QRS, dan interval RR per denyut."""
    peaks = self._refine_r_peaks(ecg, r_peaks, fs)
    if len(peaks) < 12:
      return peaks, np.array([]), {}

    clean_ecg = self._bandpass(ecg, 0.5, min(40.0, 0.45 * fs), fs)
    radius = int(round(0.10 * fs))
    good_peaks = []
    amplitude, area, slope = [], [], []

    for peak in peaks:
      segment = clean_ecg[
          max(0, peak - radius) : min(len(ecg), peak + radius + 1)
      ]
      if len(segment) < 5:
        continue
      segment = segment - np.median(segment)
      good_peaks.append(peak)
      amplitude.append(np.ptp(segment))
      area.append(np.sum(np.abs(segment)) / fs)
      slope.append(np.max(np.abs(np.diff(segment))) * fs)

    peaks = np.asarray(good_peaks, dtype=int)
    beat_times = peaks / float(fs)
    if len(peaks) < 12:
      return peaks, beat_times, {}

    rr_intervals = np.r_[1.0, np.diff(beat_times)]
    valid_rr = rr_intervals[(rr_intervals >= 0.30) & (rr_intervals <= 2.0)]
    rr_fill = np.median(valid_rr) if len(valid_rr) else 1.0
    rr_intervals[(rr_intervals < 0.30) | (rr_intervals > 2.0)] = rr_fill
    rr_intervals[0] = rr_fill

    features = {
        "r_amplitude": self._normalize(amplitude),
        "qrs_area": self._normalize(area),
        "qrs_slope": self._normalize(slope),
        "rr_interval": self._normalize(rr_intervals),
    }
    return peaks, beat_times, features

  def _make_edr_signals(self, ecg, beat_times, features, fs):
    """Ubah fitur per denyut menjadi sinyal reguler 4 Hz."""
    duration = len(ecg) / float(fs)
    grid = np.arange(0.0, duration, 1.0 / self.edr_fs)
    signals = {}

    for name, values in features.items():
      interpolator = PchipInterpolator(beat_times, values, extrapolate=False)
      signal = interpolator(grid)
      signal[grid < beat_times[0]] = values[0]
      signal[grid > beat_times[-1]] = values[-1]
      signal = detrend(np.nan_to_num(signal))
      signal = self._bandpass(
          signal, self.low_hz, self.high_hz, self.edr_fs
      )
      if np.std(signal) > 1e-9:
        signals[name] = signal / np.std(signal)

    # Baseline ECG juga bergerak mengikuti napas.
    baseline = self._bandpass(ecg, self.low_hz, self.high_hz, fs)
    ecg_time = np.arange(len(ecg)) / float(fs)
    baseline = np.interp(grid, ecg_time, baseline)
    baseline = detrend(baseline)
    if np.std(baseline) > 1e-9:
      signals["baseline_wander"] = baseline / np.std(baseline)

    return grid, signals

  # -----------------------------------------------------------------------
  # 3. SPEKTRUM, QUALITY SCORE, DAN FUSION
  # -----------------------------------------------------------------------

  def _analyze_signal(self, signal):
    """Cari frekuensi napas dan quality score dari satu sinyal EDR."""
    nperseg = len(signal)
    frequencies, power = welch(
        signal,
        fs=self.edr_fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=max(1024, 2 ** int(np.ceil(np.log2(nperseg)))),
        detrend="linear",
    )
    respiratory_band = (
        (frequencies >= self.low_hz) & (frequencies <= self.high_hz)
    )
    frequencies = frequencies[respiratory_band]
    power = power[respiratory_band]
    if len(power) < 3 or np.sum(power) <= 1e-12:
      return None

    power = power / np.sum(power)
    peak_index = int(np.argmax(power))
    peak_hz = float(frequencies[peak_index])

    # Cegah harmonik kedua terbaca sebagai dua kali RR sebenarnya.
    half = np.abs(frequencies - peak_hz / 2.0) <= 0.025
    if peak_hz >= 2 * self.low_hz and np.any(half):
      half_indices = np.flatnonzero(half)
      half_index = int(half_indices[np.argmax(power[half_indices])])
      if power[half_index] >= 0.50 * power[peak_index]:
        peak_index = half_index
        peak_hz = float(frequencies[peak_index])

    # RQI = gabungan ketajaman peak dan periodisitas sinyal.
    near_peak = np.abs(frequencies - peak_hz) <= 0.03
    concentration = float(np.sum(power[near_peak]))
    lag = int(round(self.edr_fs / peak_hz))
    centered = signal - np.mean(signal)
    denominator = np.dot(centered, centered)
    periodicity = 0.0
    if 0 < lag < len(signal) and denominator > 1e-12:
      periodicity = float(np.clip(
          np.dot(centered[:-lag], centered[lag:]) / denominator,
          0.0,
          1.0,
      ))
    quality = float(np.sqrt(concentration * periodicity))

    return {
        "frequencies": frequencies,
        "power": power,
        "peak_hz": peak_hz,
        "quality": quality,
    }

  def _analyze_window(self, signals, mask):
    """Gabungkan spektrum fitur yang kualitasnya cukup."""
    analyzed = {}
    accepted = []

    for name, signal in signals.items():
      result = self._analyze_signal(signal[mask])
      if result is None:
        continue
      analyzed[name] = {
          "rr": result["peak_hz"] * 60.0,
          "quality": result["quality"],
      }
      if result["quality"] >= 0.04:
        accepted.append((name, result))

    if not accepted:
      return None

    weights = np.asarray([result["quality"] for _, result in accepted])
    spectra = np.vstack([result["power"] for _, result in accepted])
    fused_power = np.average(spectra, axis=0, weights=weights)
    frequencies = accepted[0][1]["frequencies"]
    fused_hz = float(frequencies[np.argmax(fused_power)])

    individual_hz = np.asarray([result["peak_hz"] for _, result in accepted])
    consensus_hz = self._weighted_median(individual_hz, weights)
    disagreement = self._weighted_median(
        np.abs(individual_hz - consensus_hz), weights
    )
    agreement = np.exp(-max(0.0, disagreement) / 0.08)
    near_peak = np.abs(frequencies - fused_hz) <= 0.03
    concentration = np.sum(fused_power[near_peak]) / np.sum(fused_power)
    quality = float(np.sqrt(concentration * agreement))

    if abs(fused_hz - 2.0 * consensus_hz) <= 0.04:
      fused_hz = consensus_hz

    return {
        "rr": fused_hz * 60.0,
        "quality": quality,
        "features": analyzed,
        "weights": {name: result["quality"] for name, result in accepted},
    }

  # -----------------------------------------------------------------------
  # 4. PIPELINE UTAMA
  # -----------------------------------------------------------------------

  def estimate(self, ecg, r_peaks, fs=125):
    """Hitung RR dan kembalikan waveform serta informasi kualitas."""
    ecg = np.asarray(ecg, dtype=float)
    empty = {
        "rr": 0.0,
        "resp_signal": np.zeros(len(ecg)),
        "resp_peaks": np.array([], dtype=int),
        "refined_r_peaks": np.array([], dtype=int),
        "quality": 0.0,
        "windows": [],
        "feature_weights": {},
    }
    duration = len(ecg) / float(fs)
    if duration < self.min_duration:
      return empty

    refined, beat_times, features = self._extract_edr_features(
        ecg, r_peaks, fs
    )
    empty["refined_r_peaks"] = refined
    if not features:
      return empty

    grid, signals = self._make_edr_signals(ecg, beat_times, features, fs)
    if not signals:
      return empty

    window_size = min(self.window_seconds, duration)
    starts = list(np.arange(
        0.0,
        duration - window_size + 1e-9,
        self.step_seconds,
    ))
    final_start = duration - window_size
    if not starts or final_start - starts[-1] > 0.5:
      starts.append(final_start)

    windows = []
    feature_quality = {name: [] for name in signals}
    for start in starts:
      end = start + window_size
      mask = (grid >= start) & (grid < end)
      if np.count_nonzero(mask) < self.min_duration * self.edr_fs:
        continue
      result = self._analyze_window(signals, mask)
      if result is None:
        continue
      result["start_seconds"] = float(start)
      result["end_seconds"] = float(end)
      windows.append(result)
      for name, weight in result["weights"].items():
        feature_quality[name].append(weight)

    if not windows:
      return empty

    rates = [window["rr"] for window in windows]
    qualities = [window["quality"] for window in windows]
    final_rr = self._weighted_median(rates, qualities)
    average_weights = {
        name: float(np.mean(values)) if values else 0.0
        for name, values in feature_quality.items()
    }

    active = {name: value for name, value in average_weights.items() if value > 0}
    fused_signal = np.average(
        np.vstack([signals[name] for name in active]),
        axis=0,
        weights=list(active.values()),
    )
    fused_signal = self._bandpass(
        fused_signal, self.low_hz, self.high_hz, self.edr_fs
    )
    peak_indices, _ = find_peaks(
        fused_signal,
        distance=max(1, int(self.edr_fs / self.high_hz)),
        prominence=max(0.10, 0.20 * np.std(fused_signal)),
    )
    resp_peaks = np.asarray(
        np.clip(np.round(grid[peak_indices] * fs), 0, len(ecg) - 1),
        dtype=int,
    )
    original_time = np.arange(len(ecg)) / float(fs)
    resp_signal = np.interp(original_time, grid, fused_signal)

    return {
        "rr": float(np.round(final_rr, 2)),
        "resp_signal": resp_signal,
        "resp_peaks": resp_peaks,
        "refined_r_peaks": refined,
        "quality": float(np.average(qualities, weights=qualities)),
        "windows": windows,
        "feature_weights": average_weights,
    }
