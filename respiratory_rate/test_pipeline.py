import unittest

import numpy as np

from respiratory_rate.pipeline import ECGRespirationEstimator


class ECGRespirationEstimatorTests(unittest.TestCase):

  def test_rr_15_bpm(self):
    """Pipeline harus menemukan sinyal napas sintetis 15 bpm."""
    fs = 125
    duration = 60
    expected_rr = 15.0
    resp_hz = expected_rr / 60.0
    time = np.arange(duration * fs) / fs
    ecg = 0.18 * np.sin(2 * np.pi * resp_hz * time)
    ecg += 0.01 * np.random.default_rng(7).normal(size=len(time))
    r_peaks = np.arange(fs, (duration - 1) * fs, fs, dtype=int)

    qrs_axis = np.arange(-12, 13)
    qrs = np.exp(-((qrs_axis / 3.0) ** 2))
    for peak in r_peaks:
      modulation = 1.0 + 0.12 * np.sin(2 * np.pi * resp_hz * peak / fs)
      ecg[peak - 12 : peak + 13] += modulation * qrs

    result = ECGRespirationEstimator().estimate(ecg, r_peaks, fs)

    self.assertAlmostEqual(result["rr"], expected_rr, delta=0.5)
    self.assertGreater(result["quality"], 0.5)
    self.assertEqual(len(result["resp_signal"]), len(ecg))

  def test_signal_too_short(self):
    estimator = ECGRespirationEstimator()
    result = estimator.estimate(np.zeros(1250), np.arange(125, 1125, 125), 125)
    self.assertEqual(result["rr"], 0.0)
    self.assertEqual(result["quality"], 0.0)


if __name__ == "__main__":
  unittest.main()
