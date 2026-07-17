import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

class LiveSignalFilter:
    def __init__(self):
        self.fs = 400.0  # Frekuensi sampling murni 400 Hz
        
        # 1. RANCANG FILTER PPG (SciPy Butterworth Bandpass 0.5 - 6 Hz)
        # Menurunkan cutoff atas ke 6 Hz untuk membantu meredam noise undakan
        self.b_ppg, self.a_ppg = butter(N=2, Wn=[0.5, 6.0], btype='bandpass', fs=self.fs)
        self.zi_ppg = lfilter_zi(self.b_ppg, self.a_ppg) * 0.0

        # 2. RANCANG FILTER ECG (SciPy Butterworth Bandpass 0.5 - 100 Hz)
        self.b_ecg, self.a_ecg = butter(N=2, Wn=[0.5, 100.0], btype='bandpass', fs=self.fs)
        self.zi_ecg = lfilter_zi(self.b_ecg, self.a_ecg) * 0.0

        # 3. KUNCI PERBAIKAN: Jendela Moving Average yang lebih lebar khusus PPG
        # Pada 400 Hz, jendela 16 sampel (setara 40 ms) sangat efektif menghapus 
        # efek tangga tanpa membuat puncak gelombang menjadi tumpul.
        self.ppg_window_size = 16
        self.ppg_sma_buffer = []

    def filter_ppg(self, raw_sample):
        """
        Memproses 1 sampel PPG secara real-time dengan kombinasi SciPy lfilter
        dan Moving Average pasca-filter untuk menghapus efek undakan tangga.
        """
        # Langkah A: Jalankan filter SciPy (Menghapus baseline wander & noise makro)
        filtered_sample, self.zi_ppg = lfilter(
            self.b_ppg, 
            self.a_ppg, 
            [raw_sample], 
            zi=self.zi_ppg
        )
        base_val = float(filtered_sample[0])
        
        # Langkah B: Jalankan Moving Average (Menghapus jitter mikro / undakan tangga)
        self.ppg_sma_buffer.append(base_val)
        if len(self.ppg_sma_buffer) > self.ppg_window_size:
            self.ppg_sma_buffer.pop(0)
            
        smooth_ppg = sum(self.ppg_sma_buffer) / len(self.ppg_sma_buffer)
        return -smooth_ppg

    def filter_ecg(self, raw_sample):
        """
        Memproses 1 sampel ECG secara real-time menggunakan SciPy lfilter.
        """
        filtered_sample, self.zi_ecg = lfilter(
            self.b_ecg, 
            self.a_ecg, 
            [raw_sample], 
            zi=self.zi_ecg
        )
        return float(filtered_sample[0])