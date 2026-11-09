"""Unit test untuk MultimodalRespirationEstimator."""

import unittest
import numpy as np
from respiratory_rate.multimodal_pipeline import MultimodalRespirationEstimator


class MultimodalRespirationEstimatorTests(unittest.TestCase):

  def setUp(self):
    self.fs = 125
    self.duration = 60
    self.time = np.arange(self.duration * self.fs) / self.fs

  def _generate_synthetic_ecg(self, rr_bpm=15.0):
    resp_hz = rr_bpm / 60.0
    ecg = 0.18 * np.sin(2 * np.pi * resp_hz * self.time)
    ecg += 0.01 * np.random.default_rng(42).normal(size=len(self.time))
    r_peaks = np.arange(self.fs, (self.duration - 1) * self.fs, self.fs, dtype=int)

    qrs_axis = np.arange(-12, 13)
    qrs = np.exp(-((qrs_axis / 3.0) ** 2))
    for peak in r_peaks:
      mod = 1.0 + 0.15 * np.sin(2 * np.pi * resp_hz * peak / self.fs)
      ecg[peak - 12 : peak + 13] += mod * qrs
    return ecg, r_peaks

  def _generate_synthetic_ppg(self, rr_bpm=15.0):
    resp_hz = rr_bpm / 60.0
    # PPG baseline wander + pulse waveform with amplitude modulation
    ppg = 0.25 * np.sin(2 * np.pi * resp_hz * self.time)
    ppg += 0.01 * np.random.default_rng(43).normal(size=len(self.time))
    p_peaks = np.arange(self.fs, (self.duration - 1) * self.fs, self.fs, dtype=int)

    pulse_axis = np.arange(-15, 25)
    pulse = np.exp(-((pulse_axis / 6.0) ** 2))
    for peak in p_peaks:
      mod = 1.0 + 0.20 * np.sin(2 * np.pi * resp_hz * peak / self.fs)
      ppg[peak - 15 : peak + 25] += mod * pulse
    return ppg, p_peaks

  def test_bimodal_estimation(self):
    """Bimodal mode (ECG + PPG) harus akurat mengestimasi 15 bpm."""
    ecg, r_peaks = self._generate_synthetic_ecg(rr_bpm=15.0)
    ppg, p_peaks = self._generate_synthetic_ppg(rr_bpm=15.0)

    estimator = MultimodalRespirationEstimator()
    res = estimator.estimate(ecg=ecg, r_peaks=r_peaks, ppg=ppg, ppg_peaks=p_peaks, fs=self.fs)

    self.assertEqual(res["mode"], "BIMODAL")
    self.assertAlmostEqual(res["rr"], 15.0, delta=0.5)
    self.assertGreater(res["quality"], 0.4)
    self.assertEqual(len(res["resp_signal"]), len(ecg))

  def test_ecg_only_fallback(self):
    """Jika PPG tidak ada, harus gracefully fallback ke ECG_ONLY."""
    ecg, r_peaks = self._generate_synthetic_ecg(rr_bpm=18.0)

    estimator = MultimodalRespirationEstimator()
    res = estimator.estimate(ecg=ecg, r_peaks=r_peaks, ppg=None, ppg_peaks=None, fs=self.fs)

    self.assertEqual(res["mode"], "ECG_ONLY")
    self.assertAlmostEqual(res["rr"], 18.0, delta=0.5)
    self.assertGreater(res["quality"], 0.4)

  def test_ppg_only_fallback(self):
    """Jika ECG tidak ada, harus gracefully fallback ke PPG_ONLY."""
    ppg, p_peaks = self._generate_synthetic_ppg(rr_bpm=20.0)

    estimator = MultimodalRespirationEstimator()
    res = estimator.estimate(ecg=None, r_peaks=None, ppg=ppg, ppg_peaks=p_peaks, fs=self.fs)

    self.assertEqual(res["mode"], "PPG_ONLY")
    self.assertAlmostEqual(res["rr"], 20.0, delta=0.5)
    self.assertGreater(res["quality"], 0.4)

  def test_signal_too_short(self):
    """Durasi di bawah min_duration harus ditolak dengan aman."""
    estimator = MultimodalRespirationEstimator(min_duration=10.0)
    res = estimator.estimate(ecg=np.zeros(500), fs=125)  # 4 detik
    self.assertEqual(res["rr"], 0.0)
    self.assertEqual(res["mode"], "TOO_SHORT")


if __name__ == "__main__":
  unittest.main()
