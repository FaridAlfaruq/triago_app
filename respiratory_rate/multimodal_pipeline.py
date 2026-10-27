"""Pipeline Multimodal Respiratory Rate Estimation (ECG + PPG).

Berdasarkan literatur ilmiah terkini:
1. Peter H. Charlton et al., "An assessment of algorithms to estimate respiratory rate
   from the electrocardiogram and photoplethysmogram", IEEE Reviews in Biomedical
   Engineering, 2018 / Physiol. Meas. 2016.
2. Drew A. Birrenkott, Marco A. F. Pimentel, Peter J. Watkinson, David A. Clifton,
   "A Robust Fusion Model for Estimating Respiratory Rate From Photoplethysmography
   and Electrocardiography", IEEE Transactions on Biomedical Engineering (TBME), 2018.
3. Walter Karlen et al., "Multiparameter Respiratory Rate Estimation from the
   Photoplethysmogram", IEEE TBME, 2013.

Arsitektur:
- Ekstraksi 10 Modulasi Respirasi:
  ECG: R-amplitude (AM), QRS area (AM), QRS slope (AM), RS amplitude (AM),
       RR interval / RSA (FM), ECG baseline wander (BW).
  PPG: RIAV (Amplitude Modulation), RIIV (Intensity/Trough Variation),
       RIFV (Pulse Rate Variability / FM), PPG baseline wander (BW).
- Deteksi Titik Fiducial & Outlier / Ectopic Beat Suppression.
- Resampling PCHIP ke grid reguler 4.0 Hz + Bandpass 0.10 - 0.70 Hz (6 - 42 bpm).
- Sub-bin Parabolic Peak Refinement pada spektrum daya (Welch PSD).
- Respiratory Quality Indices (RQI_spec * RQI_ac) + Consensus Agreement Penalization.
- Graceful Degradation State Machine:
  Bimodal (ECG+PPG) -> ECG-only -> PPG-only -> Blind Hold.
- Dukungan durasi fleksibel (10s hingga 60s+).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch


class MultimodalRespirationEstimator:
  """Estimator Respiratory Rate multimodal (ECG + PPG) dengan Smart Quality-Weighted Spectral Fusion."""

  def __init__(
      self,
      edr_fs: float = 4.0,
      window_seconds: float = 60.0,
      step_seconds: float = 30.0,
      min_duration: float = 10.0,
      min_rr: float = 10.0,
      max_rr: float = 42.0,
  ):
    self.edr_fs = float(edr_fs)
    self.window_seconds = float(window_seconds)
    self.step_seconds = float(step_seconds)
    self.min_duration = float(min_duration)
    self.min_rr = float(min_rr)
    self.max_rr = float(max_rr)
    self.low_hz = self.min_rr / 60.0
    self.high_hz = self.max_rr / 60.0

  # -------------------------------------------------------------------------
  # 1. FILTERING & PREKISI SUB-BIN
  # -------------------------------------------------------------------------

  @staticmethod
  def _bandpass(signal: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 3) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 20:
      return signal.copy()
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)

  @staticmethod
  def _parabolic_peak_hz(frequencies: np.ndarray, power: np.ndarray, peak_idx: int) -> float:
    """Estimasi frekuensi sub-bin menggunakan interpolasi parabolik 3-titik."""
    if peak_idx <= 0 or peak_idx >= len(power) - 1:
      return float(frequencies[peak_idx])

    alpha = float(power[peak_idx - 1])
    beta = float(power[peak_idx])
    gamma = float(power[peak_idx + 1])
    denom = alpha - 2.0 * beta + gamma
    if abs(denom) < 1e-12:
      return float(frequencies[peak_idx])

    delta = 0.5 * (alpha - gamma) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    df = float(frequencies[1] - frequencies[0])
    return float(frequencies[peak_idx] + delta * df)

  @staticmethod
  def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0:
      return 0.0
    if np.sum(weights) <= 1e-12:
      return float(np.median(values))

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    idx = int(np.searchsorted(cdf, 0.5))
    return float(values[min(idx, len(values) - 1)])

  @staticmethod
  def _normalize(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    if len(series) == 0:
      return series
    med = np.median(series)
    mad = 1.4826 * np.median(np.abs(series - med))
    std = mad if mad > 1e-6 else np.std(series)
    if std <= 1e-6:
      return np.zeros_like(series)
    return np.clip((series - med) / std, -4.0, 4.0)

  # -------------------------------------------------------------------------
  # 2. EKSTRAKSI FITUR RESPIRASI DARI ECG (EDR)
  # -------------------------------------------------------------------------

  def _refine_r_peaks(self, ecg: np.ndarray, r_peaks: np.ndarray, fs: float) -> np.ndarray:
    if r_peaks is None or len(r_peaks) == 0:
      return np.array([], dtype=int)
    radius = max(3, int(round(0.08 * fs)))
    refined = []
    for peak in r_peaks:
      start = max(0, peak - radius)
      end = min(len(ecg), peak + radius + 1)
      refined.append(start + int(np.argmax(ecg[start:end])))
    return np.unique(refined)

  def _extract_ecg_features(
      self, ecg: np.ndarray, r_peaks: Optional[np.ndarray], fs: float
  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    if ecg is None or len(ecg) < int(self.min_duration * fs):
      return np.array([]), np.array([]), {}

    ecg = np.asarray(ecg, dtype=float)
    if r_peaks is None or len(r_peaks) < 4:
      # Auto-detect R peaks jika tidak disediakan
      diff_ecg = np.diff(ecg)
      prom = max(1e-4, 0.3 * np.std(ecg))
      dist = max(5, int(0.35 * fs))
      r_peaks, _ = find_peaks(ecg, distance=dist, prominence=prom)

    peaks = self._refine_r_peaks(ecg, r_peaks, fs)
    if len(peaks) < 4:
      return np.array([]), np.array([]), {}

    beat_times = peaks / float(fs)
    qrs_radius = max(3, int(round(0.08 * fs)))
    s_radius = max(3, int(round(0.12 * fs)))

    amps, areas, slopes, rs_amps, s_amps, rr_intervals = [], [], [], [], [], []

    for i, peak in enumerate(peaks):
      start = max(0, peak - qrs_radius)
      end = min(len(ecg), peak + qrs_radius + 1)
      qrs = ecg[start:end]
      amps.append(float(np.ptp(qrs)))
      areas.append(float(np.sum(np.abs(qrs))))

      diff = np.diff(qrs)
      slopes.append(float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0)

      s_end = min(len(ecg), peak + s_radius + 1)
      s_val = float(np.min(ecg[peak:s_end])) if s_end > peak else float(ecg[peak])
      rs_amps.append(float(ecg[peak] - s_val))
      s_amps.append(s_val)

      if i == 0:
        rr_intervals.append(0.8)
      else:
        rr_intervals.append(beat_times[i] - beat_times[i - 1])

    rr_arr = np.asarray(rr_intervals, dtype=float)
    rr_med = np.median(rr_arr)
    bad_rr = (rr_arr < 0.35) | (rr_arr > 1.8) | (np.abs(rr_arr - rr_med) > 0.35 * rr_med)
    rr_arr[bad_rr] = rr_med
    rr_arr[0] = rr_med

    features = {
        "ecg_amplitude": self._normalize(amps),
        "ecg_qrs_area": self._normalize(areas),
        "ecg_qrs_slope": self._normalize(slopes),
        "ecg_rs_amplitude": self._normalize(rs_amps),
        "ecg_s_amplitude": self._normalize(s_amps),
        "ecg_rr_interval": self._normalize(rr_arr),
    }
    return peaks, beat_times, features

  # -------------------------------------------------------------------------
  # 3. EKSTRAKSI FITUR RESPIRASI DARI PPG (RIAV, RIIV, RIFV)
  # -------------------------------------------------------------------------

  def _extract_ppg_features(
      self, ppg: np.ndarray, ppg_peaks: Optional[np.ndarray], fs: float
  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    if ppg is None or len(ppg) < int(self.min_duration * fs):
      return np.array([]), np.array([]), {}

    ppg = np.asarray(ppg, dtype=float)
    if ppg_peaks is None or len(ppg_peaks) < 4:
      # Filter dan auto-detect systolic pulse peaks
      ppg_filt = self._bandpass(ppg, 0.5, 8.0, fs)
      min_dist = max(5, int(0.35 * fs))
      prom = max(1e-4, 0.15 * np.std(ppg_filt))
      ppg_peaks, _ = find_peaks(ppg_filt, distance=min_dist, prominence=prom)

    if len(ppg_peaks) < 4:
      return np.array([]), np.array([]), {}

    p_times = ppg_peaks / float(fs)
    riav, riiv, rifv = [], [], []
    trough_search = max(4, int(0.55 * fs))

    for i, pk in enumerate(ppg_peaks):
      t0 = max(0, pk - trough_search)
      t1 = max(t0 + 1, pk - int(0.04 * fs))
      seg = ppg[t0:t1]
      tr_val = float(np.min(seg)) if len(seg) > 0 else float(ppg[max(0, pk - 1)])

      pk_val = float(ppg[pk])
      riav.append(pk_val - tr_val)
      riiv.append(tr_val)

      if i == 0:
        rifv.append(0.8)
      else:
        rifv.append(p_times[i] - p_times[i - 1])

    rifv_arr = np.asarray(rifv, dtype=float)
    pr_med = np.median(rifv_arr)
    bad_pr = (rifv_arr < 0.35) | (rifv_arr > 1.8) | (np.abs(rifv_arr - pr_med) > 0.35 * pr_med)
    rifv_arr[bad_pr] = pr_med
    rifv_arr[0] = pr_med

    features = {
        "ppg_riav": self._normalize(riav),
        "ppg_riiv": self._normalize(riiv),
        "ppg_rifv": self._normalize(rifv_arr),
    }
    return ppg_peaks, p_times, features

  # -------------------------------------------------------------------------
  # 4. RESAMPLING 4 Hz & PEMBUATAN SINYAL RESPIRASI KONTINU
  # -------------------------------------------------------------------------

  def _make_respiration_signals(
      self,
      ecg: Optional[np.ndarray],
      ppg: Optional[np.ndarray],
      ecg_bt: np.ndarray,
      ecg_feat: Dict[str, np.ndarray],
      ppg_bt: np.ndarray,
      ppg_feat: Dict[str, np.ndarray],
      fs: float,
  ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    length = len(ecg) if ecg is not None else len(ppg)
    duration = length / float(fs)
    grid = np.arange(0.0, duration, 1.0 / self.edr_fs)
    signals = {}

    # Resample fitur ECG
    if ecg_feat and len(ecg_bt) >= 4:
      for name, values in ecg_feat.items():
        interp = PchipInterpolator(ecg_bt, values, extrapolate=False)
        sig = interp(grid)
        sig[grid < ecg_bt[0]] = values[0]
        sig[grid > ecg_bt[-1]] = values[-1]
        sig = detrend(np.nan_to_num(sig))
        sig = self._bandpass(sig, self.low_hz, self.high_hz, self.edr_fs)
        if np.std(sig) > 1e-9:
          signals[name] = sig / np.std(sig)

      # ECG Baseline Wander
      if ecg is not None:
        ecg_bw = self._bandpass(ecg, self.low_hz, self.high_hz, fs)
        t_ecg = np.arange(len(ecg)) / float(fs)
        ecg_bw_interp = detrend(np.interp(grid, t_ecg, ecg_bw))
        if np.std(ecg_bw_interp) > 1e-9:
          signals["ecg_bw"] = ecg_bw_interp / np.std(ecg_bw_interp)

    # Resample fitur PPG
    if ppg_feat and len(ppg_bt) >= 4:
      for name, values in ppg_feat.items():
        interp = PchipInterpolator(ppg_bt, values, extrapolate=False)
        sig = interp(grid)
        sig[grid < ppg_bt[0]] = values[0]
        sig[grid > ppg_bt[-1]] = values[-1]
        sig = detrend(np.nan_to_num(sig))
        sig = self._bandpass(sig, self.low_hz, self.high_hz, self.edr_fs)
        if np.std(sig) > 1e-9:
          signals[name] = sig / np.std(sig)

      # PPG Baseline Wander
      if ppg is not None:
        ppg_bw = self._bandpass(ppg, self.low_hz, self.high_hz, fs)
        t_ppg = np.arange(len(ppg)) / float(fs)
        ppg_bw_interp = detrend(np.interp(grid, t_ppg, ppg_bw))
        if np.std(ppg_bw_interp) > 1e-9:
          signals["ppg_bw"] = ppg_bw_interp / np.std(ppg_bw_interp)

    return grid, signals

  # -------------------------------------------------------------------------
  # 5. SPEKTRUM, QUALITY INDICES (RQI), DAN SMART FUSION
  # -------------------------------------------------------------------------

  def _analyze_signal(self, signal: np.ndarray) -> Optional[Dict]:
    nperseg = len(signal)
    nfft = max(2048, 2 ** int(np.ceil(np.log2(nperseg))))
    frequencies, power = welch(
        signal,
        fs=self.edr_fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nfft,
        detrend="linear",
    )
    band = (frequencies >= self.low_hz) & (frequencies <= self.high_hz)
    frequencies = frequencies[band]
    power = power[band]
    if len(power) < 3 or np.sum(power) <= 1e-12:
      return None

    power = power / np.sum(power)
    peak_idx = int(np.argmax(power))
    peak_hz = self._parabolic_peak_hz(frequencies, power, peak_idx)

    # Harmonic guard: hindari terbaca 2x frekuensi respirasi
    half = np.abs(frequencies - peak_hz / 2.0) <= 0.025
    if peak_hz >= 2 * self.low_hz and np.any(half):
      half_indices = np.flatnonzero(half)
      h_idx = int(half_indices[np.argmax(power[half_indices])])
      if power[h_idx] >= 0.45 * power[peak_idx]:
        peak_idx = h_idx
        peak_hz = self._parabolic_peak_hz(frequencies, power, peak_idx)

    # 1. Spectral Concentration Index (RQI_spec)
    bw = np.abs(frequencies - peak_hz) <= 0.03
    rqi_spec = float(np.sum(power[bw]))

    # 2. Autocorrelation Periodicity Index (RQI_ac)
    det = detrend(signal)
    var = np.var(det)
    if var > 1e-9 and peak_hz > 0:
      ac = np.correlate(det, det, mode="full")
      ac = ac[len(det) - 1 :] / (var * len(det))
      expected_lag = int(round(self.edr_fs / peak_hz))
      lag_win = max(2, int(0.15 * expected_lag))
      l0 = max(0, expected_lag - lag_win)
      l1 = min(len(ac), expected_lag + lag_win + 1)
      rqi_ac = float(np.max(ac[l0:l1])) if l1 > l0 else 0.0
      rqi_ac = float(np.clip(rqi_ac, 0.0, 1.0))
    else:
      rqi_ac = 0.0

    # Composite feature quality
    q_score = float(np.sqrt(np.clip(rqi_spec * rqi_ac, 0.0, 1.0)))
    return {
        "frequencies": frequencies,
        "power": power,
        "peak_hz": peak_hz,
        "rqi_spec": rqi_spec,
        "rqi_ac": rqi_ac,
        "quality": q_score,
    }

  def _analyze_window(self, signals: Dict[str, np.ndarray], mask: np.ndarray) -> Optional[Dict]:
    analyzed = {}
    for name, sig in signals.items():
      seg = sig[mask]
      if len(seg) < 20 or np.std(seg) <= 1e-9:
        continue
      res = self._analyze_signal(seg)
      if res is not None and res["quality"] >= 0.05:
        analyzed[name] = res

    if not analyzed:
      return None

    # Consensus frequency (weighted median of individual modulation peaks)
    peaks = [r["peak_hz"] for r in analyzed.values()]
    qualities = [r["quality"] for r in analyzed.values()]
    consensus_hz = self._weighted_median(peaks, qualities)

    # Consensus agreement weighting: penalti terhadap harmonik / artefak yang menyimpang
    weights = {}
    for name, r in analyzed.items():
      dev = abs(r["peak_hz"] - consensus_hz)
      agreement = np.exp(-dev / 0.08)
      weights[name] = r["quality"] * agreement

    total_w = sum(weights.values())
    if total_w <= 1e-12:
      return None

    f_axis = next(iter(analyzed.values()))["frequencies"]
    fused_power = np.zeros_like(f_axis)
    for name, r in analyzed.items():
      fused_power += weights[name] * r["power"]
    fused_power /= total_w

    fused_idx = int(np.argmax(fused_power))
    fused_hz = self._parabolic_peak_hz(f_axis, fused_power, fused_idx)

    # Second harmonic guard pada fused spectrum
    half = np.abs(f_axis - fused_hz / 2.0) <= 0.025
    if fused_hz >= 2 * self.low_hz and np.any(half):
      h_indices = np.flatnonzero(half)
      h_idx = int(h_indices[np.argmax(fused_power[h_indices])])
      if (
          fused_power[h_idx] >= 0.45 * fused_power[fused_idx]
          and abs(f_axis[h_idx] - consensus_hz) < abs(fused_hz - consensus_hz)
      ):
        fused_idx = h_idx
        fused_hz = self._parabolic_peak_hz(f_axis, fused_power, fused_idx)

    fused_quality = float(np.sum(fused_power[np.abs(f_axis - fused_hz) <= 0.03]))

    return {
        "rr": float(np.clip(fused_hz * 60.0, self.min_rr, self.max_rr)),
        "quality": fused_quality,
        "weights": weights,
    }

  # -------------------------------------------------------------------------
  # 6. PIPELINE UTAMA ESTIMATE
  # -------------------------------------------------------------------------

  def estimate(
      self,
      ecg: Optional[np.ndarray] = None,
      r_peaks: Optional[np.ndarray] = None,
      ppg: Optional[np.ndarray] = None,
      ppg_peaks: Optional[np.ndarray] = None,
      fs: float = 125.0,
  ) -> Dict:
    """Hitung RR multimodal (atau single-modal jika salah satu sensor tidak tersedia).

    Parameters
    ----------
    ecg : array-like, optional
        Sinyal ECG mentah atau terfilter.
    r_peaks : array-like, optional
        Indeks sampel R-peaks.
    ppg : array-like, optional
        Sinyal PPG (IR / Red / Pleth).
    ppg_peaks : array-like, optional
        Indeks sampel systolic pulse peaks.
    fs : float
        Frekuensi sampling (default: 125 Hz).

    Returns
    -------
    dict:
        - "rr": Respiration rate dalam bpm
        - "resp_signal": Sinyal pernapasan interpolasi pada sampling fs
        - "resp_peaks": Indeks puncak respirasi pada sampling fs
        - "quality": Respiratory Quality Score (0.0 - 1.0)
        - "mode": "BIMODAL", "ECG_ONLY", "PPG_ONLY", atau "BLIND_HOLD"
        - "windows": Daftar hasil per window
        - "feature_weights": Rata-rata bobot modulasi yang aktif
    """
    ecg_arr = np.asarray(ecg, dtype=float) if ecg is not None and len(ecg) > 0 else None
    ppg_arr = np.asarray(ppg, dtype=float) if ppg is not None and len(ppg) > 0 else None

    ref_signal = ecg_arr if ecg_arr is not None else ppg_arr
    if ref_signal is None:
      return {
          "rr": 0.0,
          "resp_signal": np.array([]),
          "resp_peaks": np.array([], dtype=int),
          "refined_r_peaks": np.array([], dtype=int),
          "quality": 0.0,
          "mode": "BLIND_HOLD",
          "windows": [],
          "feature_weights": {},
      }

    length = len(ref_signal)
    duration = length / float(fs)

    empty = {
        "rr": 0.0,
        "resp_signal": np.zeros(length),
        "resp_peaks": np.array([], dtype=int),
        "refined_r_peaks": np.array([], dtype=int),
        "quality": 0.0,
        "mode": "BLIND_HOLD",
        "windows": [],
        "feature_weights": {},
    }

    if duration < self.min_duration:
      empty["mode"] = "TOO_SHORT"
      return empty

    # Ekstraksi fitur per modulasi
    ecg_refined, ecg_bt, ecg_feat = self._extract_ecg_features(ecg_arr, r_peaks, fs)
    ppg_pks, ppg_bt, ppg_feat = self._extract_ppg_features(ppg_arr, ppg_peaks, fs)

    has_ecg = len(ecg_feat) > 0
    has_ppg = len(ppg_feat) > 0

    if not has_ecg and not has_ppg:
      empty["mode"] = "NO_FEATURES"
      return empty

    mode = "BIMODAL" if (has_ecg and has_ppg) else ("ECG_ONLY" if has_ecg else "PPG_ONLY")

    grid, signals = self._make_respiration_signals(
        ecg_arr if has_ecg else None,
        ppg_arr if has_ppg else None,
        ecg_bt,
        ecg_feat,
        ppg_bt,
        ppg_feat,
        fs,
    )

    if not signals:
      empty["mode"] = "NO_SIGNALS"
      return empty

    # Setup sliding window atau single adaptive window untuk durasi pendek
    window_size = min(self.window_seconds, duration)
    starts = list(np.arange(0.0, duration - window_size + 1e-9, self.step_seconds))
    final_start = duration - window_size
    if not starts or final_start - starts[-1] > 0.5:
      starts.append(final_start)

    windows = []
    feature_quality = {name: [] for name in signals}

    for start in starts:
      end = start + window_size
      mask = (grid >= start) & (grid < end)
      min_samples = max(20, int(self.min_duration * self.edr_fs * 0.8))
      if np.count_nonzero(mask) < min_samples:
        continue
      res = self._analyze_window(signals, mask)
      if res is None:
        continue
      res["start_seconds"] = float(start)
      res["end_seconds"] = float(end)
      windows.append(res)
      for name, weight in res["weights"].items():
        feature_quality[name].append(weight)

    if not windows:
      empty["mode"] = f"{mode}_NO_WINDOWS"
      return empty

    # Temporal consistency smoothing (rolling median 3-window)
    raw_rates = np.array([w["rr"] for w in windows], dtype=float)
    if len(raw_rates) >= 3:
      smoothed_rates = np.array([
          np.median(raw_rates[max(0, idx - 1) : min(len(raw_rates), idx + 2)])
          for idx in range(len(raw_rates))
      ])
      for idx, w in enumerate(windows):
        w["rr"] = float(smoothed_rates[idx])

    rates = [w["rr"] for w in windows]
    qualities = [w["quality"] for w in windows]
    final_rr = self._weighted_median(rates, qualities)

    average_weights = {
        name: float(np.mean(vals)) if vals else 0.0
        for name, vals in feature_quality.items()
    }

    # Sinyal respirasi komposit untuk visualisasi GUI
    active = {name: val for name, val in average_weights.items() if val > 0}
    if active:
      fused_signal = np.average(
          np.vstack([signals[name] for name in active]),
          axis=0,
          weights=list(active.values()),
      )
      fused_signal = self._bandpass(fused_signal, self.low_hz, self.high_hz, self.edr_fs)
    else:
      fused_signal = next(iter(signals.values()))

    # Temukan puncak-puncak respirasi
    peak_indices, _ = find_peaks(
        fused_signal,
        distance=max(1, int(self.edr_fs / self.high_hz)),
        prominence=max(0.10, 0.20 * np.std(fused_signal)),
    )
    resp_peaks = np.asarray(
        np.clip(np.round(grid[peak_indices] * fs), 0, length - 1),
        dtype=int,
    )

    orig_time = np.arange(length) / float(fs)
    resp_signal = np.interp(orig_time, grid, fused_signal)

    avg_qual = (
        float(np.average(qualities, weights=qualities))
        if sum(qualities) > 0
        else float(np.mean(qualities))
    )

    return {
        "rr": float(np.round(final_rr, 2)),
        "resp_signal": resp_signal,
        "resp_peaks": resp_peaks,
        "refined_r_peaks": ecg_refined,
        "quality": float(np.round(avg_qual, 4)),
        "mode": mode,
        "windows": windows,
        "feature_weights": average_weights,
    }


class ECGRespirationEstimator(MultimodalRespirationEstimator):
  """Subclass backwards-compatible untuk pemanggilan ECGRespirationEstimator yang sudah ada."""

  def estimate(
      self,
      ecg: np.ndarray,
      r_peaks: Optional[np.ndarray] = None,
      fs: float = 125.0,
      ppg: Optional[np.ndarray] = None,
      ppg_peaks: Optional[np.ndarray] = None,
  ) -> Dict:
    return super().estimate(ecg=ecg, r_peaks=r_peaks, ppg=ppg, ppg_peaks=ppg_peaks, fs=fs)
