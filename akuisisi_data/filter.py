from scipy.signal import butter, filtfilt

class RealTimeBiomedicalFilter:
    def __init__(self, alpha_hp=0.995, beta_lp=0.15):
        """
        alpha_hp: Mengontrol keagresifan Baseline Correction (High-Pass). 
                  Semakin dekat ke 1 (misal 0.995), sinyal semakin natural.
        beta_lp:  Mengontrol kehalusan grafik (Low-Pass).
                  Semakin kecil nilainya (misal 0.10), grafik semakin mulus bebas noise.
        """
        self.alpha_hp = alpha_hp
        self.beta_lp = beta_lp
        
        # State awal untuk High-Pass (Baseline Correction)
        self.prev_raw = None
        self.prev_hp = 0.0
        
        # State awal untuk Low-Pass (Smoothing)
        self.prev_lp = 0.0

    def process(self, raw_sample):
        """
        Memproses 1 sampel data mentah secara real-time.
        Mengembalikan data bersih yang sudah bebas baseline wander dan noise.
        """
        if self.prev_raw is None:
            self.prev_raw = raw_sample
            self.prev_lp = raw_sample
            return 0.0

        # Langkah 1: BASELINE CORRECTION (Digital High-Pass Filter)
        # Mengurangi fluktuasi DC baseline wander sehingga sinyal berpusat di angka 0
        current_hp = self.alpha_hp * (self.prev_hp + raw_sample - self.prev_raw)
        self.prev_raw = raw_sample
        self.prev_hp = current_hp

        # Langkah 2: SMOOTHING / FILTER NOISE (Digital Low-Pass Filter)
        # Menghilangkan noise frekuensi tinggi (jitter/tremor elektrik)
        current_lp = (self.beta_lp * current_hp) + ((1 - self.beta_lp) * self.prev_lp)
        self.prev_lp = current_lp

        return current_lp