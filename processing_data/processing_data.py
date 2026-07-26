import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, resample, savgol_filter


class ECGProcessor:
  """Kelas OOP untuk preprocessing sinyal ECG, visualisasi grafik halus,

  serta ekstraksi tanda-tanda vital (Heart Rate & Respiration Rate).
  """

  def __init__(self, target_fs=125):
    """Parameters:

    -----------
    target_fs : int
        Sampling rate target setelah downsampling (default: 125 Hz).
    """
    self.target_fs = target_fs

  # =========================================================================
  # 1. FUNGSI FILTER INDIVIDUAL & UTILS
  # =========================================================================

  def notch(self, ecg, freq=50.0, fs=125, Q=30.0):
    """Menghilangkan powerline noise (50 Hz)."""
    nyq = 0.5 * fs
    if freq >= nyq:
      return ecg
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, ecg)

  def downsample(self, x, time, fs, fs_target=125):
    """Menurunkan frekuensi sampling sinyal dan time array."""
    num_samples = int(np.round((len(x) * fs_target / fs)))
    x_resample = resample(x, num_samples)
    time_resample = np.linspace(time[0], time[-1], num_samples)
    return x_resample, time_resample

  def detrending(self, signal, fs=125):
    """Menghilangkan baseline wander menggunakan dua kali median filter."""
    w1 = int(np.round(0.2 * fs))
    if w1 % 2 == 0:
      w1 += 1
    w2 = int(np.round(0.6 * fs))
    if w2 % 2 == 0:
      w2 += 1

    baseline_step1 = median_filter(signal, size=w1)
    baseline_step2 = median_filter(baseline_step1, size=w2)
    return signal - baseline_step2

  def lowpass(self, signal, lowcut=35.0, fs=125):
    """Butterworth Lowpass Filter (orde 3)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(3, low, btype='lowpass', analog=False)
    return filtfilt(b, a, signal)

  def savgol(self, signal, window_size=11, poly_order=2):
    """Savitzky-Golay smoothing filter."""
    return savgol_filter(signal, window_size, poly_order)

  def bandpass(self, x, low_cut=5.0, high_cut=18.0, fs=125, order=4):
    """Butterworth Bandpass Filter."""
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    return filtfilt(b, a, x)

  # =========================================================================
  # 2. HELPER METHODS UNTUK R-PEAK DETECTION
  # =========================================================================

  def _calculate_mean_slope(self, signal, peak_idx, window_samples):
    """Menghitung rata-rata kemiringan (mean slope) 70 ms sebelum posisi peak_idx."""
    start_idx = max(0, peak_idx - window_samples)
    segment = signal[start_idx : peak_idx + 1]
    if len(segment) < 2:
      return 0.0
    return float(np.mean(np.abs(np.diff(segment))))

  def _calculate_meansb(self, r_peaks, all_peaks, current_peak_idx, signal):
    """Menghitung MEANSB: Rata-rata amplitudo 3 QRS sebelumnya dan 3 puncak setelahnya."""
    prev_qrs_vals = [signal[idx] for idx in r_peaks[-3:]] if r_peaks else []
    future_peaks = [idx for idx in all_peaks if idx > current_peak_idx]
    next_peak_vals = [signal[idx] for idx in future_peaks[:3]]

    combined_vals = (
        prev_qrs_vals + [signal[current_peak_idx]] + next_peak_vals
    )
    return (
        float(np.mean(combined_vals))
        if combined_vals
        else float(signal[current_peak_idx])
    )

  # =========================================================================
  # 3. DETEKSI R-PEAK, HEART RATE, DAN RESPIRATION RATE
  # =========================================================================

  def detect_r_peaks(self, signal, fs=125):
    """Deteksi R-peak menggunakan Pan-Tompkins Modifikasi dengan Dual Threshold."""
    # 1. Bandpass Filter (5 - 18 Hz)
    ecg = self.bandpass(signal, 5.0, 18.0, fs=fs)

    # 2. Diferensiasi Sinyal (5-Point Derivative)
    T = 1.0 / fs
    kernel = np.array([1, 2, 0, -2, -1]) * (1.0 / (8 * T))
    ecg_diff = np.convolve(ecg, kernel, mode='same')

    # 3. Pengkuadratan
    ecg_squared = ecg_diff**2

    # 4. Penghalusan (Flattop Window 60 ms)
    win_duration = 0.060
    n_smooth = int(np.round(win_duration * fs))
    if n_smooth % 2 == 0:
      n_smooth += 1
    n = np.arange(n_smooth)
    psi = (2 * np.pi * n) / n_smooth
    a0, a1, a2, a3, a4 = (
        0.2155789,
        0.4166316,
        0.27726316,
        0.08357895,
        0.00694737,
    )

    flatop_win = (
        a0
        - a1 * np.cos(psi)
        + a2 * np.cos(2 * psi)
        - a3 * np.cos(3 * psi)
        + a4 * np.cos(4 * psi)
    )
    flatop_win = flatop_win / np.sum(flatop_win)
    ecg_smoothed = np.convolve(ecg_squared, flatop_win, mode='same')

    # 5. Moving Window Integration (MWI 150 ms)
    mwi_win = 0.150
    n_mwi = int(np.round(mwi_win * fs))
    kernel_mwi = np.ones(n_mwi) / n_mwi
    ecg_mwi = np.convolve(ecg_smoothed, kernel_mwi, mode='same')

    # 6. Refractory Period (231 ms)
    refractory_period = 0.231
    min_distance = int(np.round(refractory_period * fs))
    r_peaks_index, _ = find_peaks(ecg_mwi, distance=min_distance)

    # 7. Inisialisasi Threshold (2 Detik Pertama)
    init_duration = 2.0
    init_sample = int(np.round(init_duration * fs))
    mwi_init = ecg_mwi[:init_sample]

    MAXF = np.max(mwi_init) if len(mwi_init) > 0 else np.max(ecg_mwi)
    MEANF = np.mean(mwi_init) if len(mwi_init) > 0 else np.mean(ecg_mwi)

    spk = MAXF / 3.0
    npk = 0.5 * MEANF
    th1 = spk
    th2 = npk

    N_70 = int(np.round(0.070 * fs))
    r_peaks = []
    noise_peaks = []

    # 8. Loop Decision Phase
    for peak_idx in r_peaks_index:
      peak_val = ecg_mwi[peak_idx]

      if len(r_peaks) > 0:
        last_r_idx = r_peaks[-1]
        current_rr_sec = (peak_idx - last_r_idx) / fs

        if len(r_peaks) >= 2:
          recent_r = r_peaks[-8:]
          mean_rr_sec = np.mean(np.diff(recent_r) / fs)
        else:
          mean_rr_sec = current_rr_sec
      else:
        current_rr_sec = 0.0
        mean_rr_sec = 0.0

      is_classified = False

      # Cabang 1: Evaluasi Puncak Normal
      if peak_val > th1:
        is_r_peak = True
        if len(r_peaks) > 0 and (
            current_rr_sec < 0.360 or current_rr_sec < (0.5 * mean_rr_sec)
        ):
          slope_curr = self._calculate_mean_slope(ecg_mwi, peak_idx, N_70)
          slope_prev = self._calculate_mean_slope(ecg_mwi, last_r_idx, N_70)

          if slope_curr < 0.60 * slope_prev:
            is_r_peak = False  # Terdeteksi Gelombang T

        if is_r_peak:
          r_peaks.append(peak_idx)
          spk = 0.125 * peak_val + 0.875 * spk
          is_classified = True
        else:
          noise_peaks.append(peak_idx)
          npk = 0.125 * peak_val + 0.875 * npk
          is_classified = True

      # Cabang 2: Search-Back Mechanism
      elif len(r_peaks) > 0 and (
          current_rr_sec > 1.0 or current_rr_sec > (1.66 * mean_rr_sec)
      ):
        meansb = self._calculate_meansb(
            r_peaks, r_peaks_index, peak_idx, ecg_mwi
        )
        th3 = 0.5 * th2 + 0.5 * meansb

        if peak_val > th3:
          r_peaks.append(peak_idx)
          spk = 0.75 * peak_val + 0.25 * spk
          is_classified = True

      # Cabang 3: Recovery dari Extreme High R-Peaks
      if not is_classified and len(r_peaks) > 0 and current_rr_sec > 1.4:
        if peak_val > (0.2 * th2):
          r_peaks.append(peak_idx)
          spk = 0.75 * peak_val + 0.25 * spk
          is_classified = True

      # Cabang 4: Puncak Kebisingan (Noise)
      if not is_classified:
        noise_peaks.append(peak_idx)
        npk = 0.125 * peak_val + 0.875 * npk

      # Pembaruan Threshold Dinamis
      th1 = npk + 0.25 * (spk - npk)
      th2 = 0.4 * th1

    return r_peaks, noise_peaks

  def calculate_heart_rate(self, r_peaks, fs=125):
    """Menghitung Heart Rate (BPM) berdasarkan R-peaks."""
    if len(r_peaks) < 2:
      return 0.0
    hr = 60.0 / np.mean(np.diff(r_peaks) / fs)
    return float(np.round(hr, 2))

  def calculate_respiration_rate(self, ecg, r_peaks, fs=125):
    """Menghitung Respiration Rate (RPM) menggunakan EDR (ECG-Derived Respiration)."""
    n_samples = len(ecg)
    total_seconds = n_samples / fs
    r_peaks = np.array(r_peaks)

    if len(r_peaks) < 4:
      return 0.0, np.zeros(n_samples), np.array([])

    T = 1.0 / fs
    kernel_diff = np.array([1, 2, 0, -2, -1]) * (1.0 / (8 * T))
    ecg_diff = np.convolve(ecg, kernel_diff, mode='same')
    win_samples = int(np.round(0.040 * fs))
    edr_samples = []

    for idx in r_peaks:
      t0 = max(0, idx - win_samples)
      t1 = min(n_samples, idx + win_samples + 1)
      max_slope = np.max(np.abs(ecg_diff[t0:t1]))
      edr_samples.append(max_slope)

    edr_samples = np.array(edr_samples)
    xm, xd, xc = 0.0, 1.0, 0
    r_signals = []

    for x in edr_samples:
      d = x - xm
      if xc < 500:
        xc += 1
        dn = d / xc
      else:
        dn = d / xc
        xdmax = 3.0 * xd / xc
        dn = np.clip(dn, -xdmax, xdmax)

      xm += dn
      xd += abs(dn) - xd / xc
      if xd < 1e-3:
        xd = 1e-3

      r = d / xd
      r_signals.append(r)

    r_signals = np.array(r_signals)
    cs = CubicSpline(r_peaks, r_signals, extrapolate=False)

    interp_indices = np.arange(r_peaks[0], r_peaks[-1] + 1)
    resp_signal = cs(interp_indices)

    resp_signal_full = np.zeros(n_samples)
    resp_signal_full[r_peaks[0] : r_peaks[-1] + 1] = resp_signal
    resp_signal_full[: r_peaks[0]] = resp_signal[0]
    resp_signal_full[r_peaks[-1] + 1 :] = resp_signal[-1]

    min_distance_resp = int(np.round(1.6 * fs))
    resp_peaks, _ = find_peaks(
        resp_signal_full,
        distance=min_distance_resp,
        height=np.mean(resp_signal_full),
    )

    resp_rate = (
        (len(resp_peaks) / (total_seconds / 60.0)) if total_seconds > 0 else 0.0
    )
    return float(np.round(resp_rate, 2)), resp_signal_full, resp_peaks

  # =========================================================================
  # 4. MAIN PIPELINE UTAMA
  # =========================================================================

  def process_all(
      self,
      raw_signal,
      raw_time,
      fs_orig=400,
      notch_freq=50.0,
      lpf_cut=35.0,
      savgol_win=11,
      savgol_poly=2,
  ):
    """Fungsi utama satu pintu untuk mengeksekusi seluruh pipeline:

    1. Downsample awal ke 125 Hz
    2. Pembuatan ecg_smooth (Notch -> Detrend -> LPF -> Savgol)
    3. Deteksi R-peaks
    4. Estimasi Heart Rate (HR)
    5. Estimasi Respiration Rate (RR) & Pernapasan Full Signal

    Returns:
    --------
    results : dict
        Dictionary berisi semua output siap pakai untuk GUI.
    """
    # Step 1: Downsample data mentah (misal: 400 Hz ke 125 Hz)
    ecg_125, time_125 = self.downsample(
        raw_signal, raw_time, fs=fs_orig, fs_target=self.target_fs
    )

    # Step 2: Sinyal Halus untuk Visualisasi GUI (ecg_smooth)
    sig_notch = self.notch(ecg_125, freq=notch_freq, fs=self.target_fs)
    sig_detrend = self.detrending(sig_notch, fs=self.target_fs)
    sig_lpf = self.lowpass(sig_detrend, lowcut=lpf_cut, fs=self.target_fs)
    ecg_smooth = self.savgol(
        sig_lpf, window_size=savgol_win, poly_order=savgol_poly
    )

    # Step 3: Deteksi R-Peaks (menggunakan data 125 Hz)
    r_peaks, noise_peaks = self.detect_r_peaks(ecg_125, fs=self.target_fs)

    # Step 4: Hitung Heart Rate
    hr = self.calculate_heart_rate(r_peaks, fs=self.target_fs)

    # Step 5: Hitung Respiration Rate & Sinyal EDR
    resp_rate, resp_signal_full, resp_peaks = self.calculate_respiration_rate(
        ecg_125, r_peaks, fs=self.target_fs
    )

    return {
        'ecg_125': ecg_125,
        'ecg_smooth': ecg_smooth,
        'time_125': time_125,
        'r_peaks': r_peaks,
        'noise_peaks': noise_peaks,
        'hr': hr,
        'rr': resp_rate,
        'resp_signal': resp_signal_full,
        'resp_peaks': resp_peaks,
    }

class PPGProcessor:
    """
    Kelas OOP untuk preprocessing sinyal PPG (Red & IR) serta ekstraksi tanda-tanda vital:
    SpO2, Perfusion Index (PI Red & PI IR), Heart Rate (HR), dan Respiration Rate (RR).
    """

    def __init__(self, target_fs=125):
        self.target_fs = target_fs

    # -------------------------------------------------------------------------
    # 1. HELPER & FILTER METHODS UNTUK PPG
    # -------------------------------------------------------------------------

    def downsample(self, x, time, fs_orig):
        """Menurunkan frekuensi sampling ke target_fs (125 Hz)."""
        if fs_orig == self.target_fs:
            return np.copy(x), np.copy(time)
        num_samples = int(np.round(len(x) * self.target_fs / fs_orig))
        x_resample = resample(x, num_samples)
        time_resample = np.linspace(time[0], time[-1], num_samples)
        return x_resample, time_resample

    def baseline_correction(self, time, x):
        """Step 3: Estimasi komponen DC menggunakan CubicSpline dari titik lembah (troughs)."""
        min_distance = int(0.5 * self.target_fs)
        prominence_thresh = 0.2 * np.std(x)
        peaks, _ = find_peaks(-x, distance=min_distance, prominence=prominence_thresh)

        if len(peaks) < 2:
            baseline_dc = np.full_like(x, np.mean(x))
            return x - baseline_dc, np.abs(baseline_dc)

        t_val = time[peaks]
        s_val = x[peaks]

        # Mencegah edge effect pada batas awal dan akhir
        if t_val[0] > time[0]:
            t_val = np.insert(t_val, 0, time[0])
            s_val = np.insert(s_val, 0, s_val[0])
        if t_val[-1] < time[-1]:
            t_val = np.append(t_val, time[-1])
            s_val = np.append(s_val, s_val[-1])

        cs = CubicSpline(t_val, s_val, bc_type='natural')
        baseline_dc = cs(time)
        baseline_dc = np.maximum(np.abs(baseline_dc), 1e-6)
        x_corrected = x - baseline_dc

        return x_corrected, baseline_dc

    def bandpass_ppg(self, x, low_cut=0.5, high_cut=15.0):
        """Step 4: Butterworth Bandpass Filter (0.5 - 15 Hz)."""
        nyq = 0.5 * self.target_fs
        b, a = butter(2, [low_cut / nyq, high_cut / nyq], btype='bandpass')
        return filtfilt(b, a, x)

    def savgol(self, x, window_size=11, poly_order=2):
        """Step 5: Savitzky-Golay Filter untuk smoothing."""
        if len(x) < window_size:
            window_size = len(x) // 2 * 2 - 1
        return savgol_filter(x, window_size, poly_order)

    def ac_p2p(self, ac_signal):
        """Step 6: Menghitung rata-rata amplitudo Peak-to-Peak (AC)."""
        min_distance = int(0.5 * self.target_fs)
        prominence_thresh = 0.2 * np.std(ac_signal)

        peaks, _ = find_peaks(ac_signal, distance=min_distance, prominence=prominence_thresh)
        troughs, _ = find_peaks(-ac_signal, distance=min_distance, prominence=prominence_thresh)

        if len(peaks) == 0 or len(troughs) == 0:
            return 0.0, peaks

        p2p_values = []
        for p in peaks:
            valid_troughs = troughs[troughs < p]
            if len(valid_troughs) > 0:
                nearest_trough = valid_troughs[-1]
                amp = ac_signal[p] - ac_signal[nearest_trough]
                p2p_values.append(amp)

        if len(p2p_values) == 0:
            return 0.0, peaks

        return float(np.mean(p2p_values)), peaks

    def calculate_spo2_and_pi(self, ac_ir, dc_ir, ac_red, dc_red):
        """Step 7: Menghitung SpO2, PI Red, dan PI IR."""
        mean_dc_ir = np.mean(np.abs(dc_ir))
        mean_dc_red = np.mean(np.abs(dc_red))

        pi_red = (ac_red / mean_dc_red) * 100 if mean_dc_red != 0 else 0.0
        pi_ir = (ac_ir / mean_dc_ir) * 100 if mean_dc_ir != 0 else 0.0

        if pi_ir == 0:
            return 0.0, round(pi_red, 2), round(pi_ir, 2)

        R = (ac_red / mean_dc_red) / (ac_ir / mean_dc_ir)
        spo2 = -17.8327 * (R**2) + 15.6006 * R + 94.6457
        spo2 = np.clip(spo2, 0.0, 100.0)

        return float(np.round(spo2, 2)), float(np.round(pi_red, 2)), float(np.round(pi_ir, 2))

    # -------------------------------------------------------------------------
    # 2. PIPELINE UTAMA PEMROSESAN PPG
    # -------------------------------------------------------------------------

    def process_ppg(self, raw_time, raw_red, raw_ir, fs_orig=125):
        """
        Menjalankan 7 tahapan pipeline PPG:
        1. Downsample ke 125 Hz
        2. Invert Sinyal (-x)
        3. Baseline Correction
        4. BPF (0.5 - 15 Hz)
        5. Savgol Filter
        6. Peak-to-Peak (p2p)
        7. Kalkulasi SpO2, PI Red, & PI IR
        """
        # Step 1: Downsample
        red_ds, time_125 = self.downsample(raw_red, raw_time, fs_orig)
        ir_ds, _ = self.downsample(raw_ir, raw_time, fs_orig)

        # Step 2: Invert Sinyal
        red_inv = -red_ds
        ir_inv = -ir_ds

        # Step 3: Baseline Correction (Mendapatkan AC & DC)
        red_ac_base, red_dc = self.baseline_correction(time_125, red_inv)
        ir_ac_base, ir_dc = self.baseline_correction(time_125, ir_inv)

        # Step 4: BPF (0.5 - 15 Hz)
        red_bpf = self.bandpass_ppg(red_ac_base)
        ir_bpf = self.bandpass_ppg(ir_ac_base)

        # Step 5: Savitzky-Golay Smoothing
        red_clean = self.savgol(red_bpf)
        ir_clean = self.savgol(ir_bpf)

        # Step 6: Peak-to-Peak Extraction
        ac_red_val, red_peaks = self.ac_p2p(red_clean)
        ac_ir_val, ir_peaks = self.ac_p2p(ir_clean)

        # Step 7: Hitung SpO2, PI Red, PI IR
        spo2_val, pi_red, pi_ir = self.calculate_spo2_and_pi(
            ac_ir=ac_ir_val, dc_ir=ir_dc, ac_red=ac_red_val, dc_red=red_dc
        )

        # Tambahan: Hitung Heart Rate dari Puncak PPG IR
        if len(ir_peaks) > 1:
            intervals = np.diff(time_125[ir_peaks])
            ppg_hr = float(np.round(60.0 / np.mean(intervals), 2))
        else:
            ppg_hr = 0.0

        return {
            'time_125': time_125,
            'red_clean': red_clean,
            'ir_clean': ir_clean,
            'red_peaks': red_peaks,
            'ir_peaks': ir_peaks,
            'spo2': spo2_val,
            'pi_red': pi_red,
            'pi_ir': pi_ir,
            'ppg_hr': ppg_hr,
        }
