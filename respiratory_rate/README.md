# Respiratory Rate Estimation: Multimodal Smart Fusion (ECG + PPG)

Modul ini mengimplementasikan algoritma estimasi **Respiratory Rate (RR)** berbasis **Multimodal Smart Fusion** yang menggabungkan sinyal elektrokardiogram (**AD8232 ECG**) dan fotopletismogram (**MAX30102 PPG**).

Metode ini dikembangkan berdasarkan literatur ilmiah terkemuka:
1. **Peter H. Charlton et al.**, *"An assessment of algorithms to estimate respiratory rate from the electrocardiogram and photoplethysmogram"*, IEEE Reviews in Biomedical Engineering, 2018 / Physiol. Meas. 2016.
2. **Drew A. Birrenkott, Marco A. F. Pimentel, Peter J. Watkinson, David A. Clifton**, *"A Robust Fusion Model for Estimating Respiratory Rate From Photoplethysmography and Electrocardiography"*, IEEE Transactions on Biomedical Engineering (TBME), Vol. 65, No. 9, 2018.
3. **Walter Karlen et al.**, *"Multiparameter Respiratory Rate Estimation from the Photoplethysmogram"*, IEEE TBME, 2013.

---

## 1. Arsitektur & Cara Kerja Algoritma

Algoritma mengekstrak hingga **10 modulasi fisiologis pernapasan** secara simultan:

```
                      ┌──────────────────────────────────────────────┐
                      │          Fisiologi Dinamika Respirasi        │
                      │    (Tekanan Intratoraks & Volume Paru)       │
                      └──────────────────────┬───────────────────────┘
                                             │
         ┌───────────────────────────────────┼──────────────────────────────────┐
         │                                   │                                  │
         ▼                                   ▼                                  ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│Amplitude Mod (AM)│               │Frequency Mod (FM)│               │Baseline Wand (BW)│
├──────────────────┤               ├──────────────────┤               ├──────────────────┤
│• ECG: R-amp, Area│               │• ECG: RSA        │               │• ECG: Thoracic Z │
│  RS-amp, S-amp   │               │  (vagal nerve)   │               │  drift (0.1-0.7Hz│
│• PPG: RIAV       │               │• PPG: RIFV       │               │• PPG: RIIV       │
│  (Pulsus parad.) │               │  (PRV dynamics)  │               │  (Venous pool)   │
└──────────────────┘               └──────────────────┘               └──────────────────┘
```

### Tahapan Pemrosesan:
1. **Fiducial Alignment & Outlier Rejection:**
   - ECG: Sub-sample QRS refinement ($\pm 80\text{ ms}$) dan supresi denyut ektopik.
   - PPG: Deteksi systolic peak $P_k$ dan diastolic onset trough $T_k$.
2. **Ekstraksi 10 Modulasi:**
   - ECG (AM): R-amplitude, QRS area, QRS slope, RS amplitude, S amplitude.
   - ECG (FM): RR interval (Respiratory Sinus Arrhythmia / RSA).
   - ECG (BW): Transthoracic impedance baseline wander ($0.10–0.70\text{ Hz}$).
   - PPG (AM): RIAV (Respiratory-Induced Amplitude Variation: $P_k - T_k$).
   - PPG (BW): RIIV (Respiratory-Induced Intensity Variation: $T_k$).
   - PPG (FM): RIFV (Respiratory-Induced Frequency Variation / Pulse Interval).
   - PPG (BW): Continuous optical baseline wander ($0.10–0.70\text{ Hz}$).
3. **Resampling Reguler PCHIP (4.0 Hz):**
   - Menghindari osilasi palsu (Runge phenomenon) yang sering terjadi pada cubic spline biasa.
4. **Sub-bin Parabolic Peak Refinement (Welch PSD):**
   - Meningkatkan resolusi frekuensi hingga $<\pm 0.2\text{ bpm}$ pada window pendek (10s – 30s).
5. **Respiratory Quality Indices (RQI):**
   - $RQI_{\text{spec}}$ (konsentrasi spektrum dalam $\pm 0.03\text{ Hz}$) $\times$ $RQI_{\text{ac}}$ (periodisitas autokorelasi).
6. **Graceful Degradation State Machine:**
   - **BIMODAL**: Kedua sensor valid $\to$ semua modulasi digabungkan secara optimal.
   - **ECG-ONLY**: PPG terlepas/kotor $\to$ degradasi mulus ke fitur ECG.
   - **PPG-ONLY**: ECG terlepas/noise $\to$ degradasi mulus ke fitur PPG.
   - **BLIND-HOLD**: Kedua sensor tidak valid $\to$ penolakan aman ($RR = 0$, alarm).

---

## 2. Hasil Pengujian Dataset Sekunder (BIDMC PhysioNet)

Evaluasi penuh pada **53 subjek BIDMC** ($N=795$ evaluasi window, window 60s, step 30s) terhadap ground truth anotasi pernapasan manual:

| Metode | N Evaluasi | Coverage | MAE (bpm) | RMSE (bpm) | MAPE (%) | Dalam $\pm 2$ bpm |
|---|---:|---:|---:|---:|---:|---:|
| **Legacy ECG** | 795 | 100.0% | 2.855 | 3.812 | 20.15% | 48.55% |
| **ECG Fusion (Single-modal)** | 795 | 100.0% | 1.203 | 2.803 | 9.21% | 87.17% |
| **PPG Fusion (Single-modal)** | 795 | 100.0% | 1.342 | 2.882 | 9.66% | 82.77% |
| **Multimodal Smart Fusion (ECG+PPG)** | **795** | **100.0%** | **0.961** | **2.346** | **7.68%** | **90.44%** |

> **Peningkatan Signifikan:**
> - MAE turun drastis ke **0.961 bpm** (turun **66.3%** dibanding Legacy, turun **20.1%** dibanding ECG-only).
> - Akurasi dalam $\pm 2$ bpm melonjak ke **90.44%**.

---

## 3. Hasil Pengujian Dataset Primer Klinis TriaGo (`data_primer/`)

Evaluasi pada **21 subjek data primer klinis** (`Data1.csv` sampai `Data21.csv`) terhadap Ground Truth pengukuran klinis pasien (`Respiratory_Rate_Ground_Truth`):

| Metode | N Evaluasi | MAE (bpm) | RMSE (bpm) | Bias (bpm) | MAPE (%) | Dalam $\pm 2$ bpm |
|---|---:|---:|---:|---:|---:|---:|
| **Legacy ECG** | 21 | 3.623 | 4.374 | -0.846 | 18.04% | 33.33% |
| **ECG Fusion (Single-modal)** | 21 | 3.408 | 4.611 | -0.470 | 17.40% | 47.62% |
| **PPG Fusion (Single-modal)** | 21 | 2.626 | 4.468 | +0.025 | 12.19% | 61.90% |
| **Multimodal Smart Fusion (ECG+PPG)** | **21** | **2.439** | **3.816** | **-0.227** | **12.74%** | **66.67%** |

---

## 4. Hasil Pengujian Rekaman Bed Klinis (`data_pengukuran/`)

- Menangani **22 file rekaman bed klinis** secara stabil (semua menghasilkan estimasi realistis $10.08 – 24.79\text{ bpm}$).
- **Handling Durasi Pendek:** File `20260803_022317_BedA1.csv` berdurasi **10 detik** yang sebelumnya gagal bernilai `0.00 bpm` pada pipeline lama kini berhasil diestimasi secara akurat sebesar **`10.08 bpm`**.
- **Graceful Fallback:** File `20260814_145630_Bed10.csv` yang memiliki optical perfusion PPG rendah otomatis beralih ke mode **`ECG_ONLY`** tanpa kegagalan sistem.

---

## 5. Cara Menjalankan Benchmark & Pengujian

```powershell
# 1. Menjalankan benchmark dataset sekunder (BIDMC 53 subjek)
.\env\Scripts\python.exe respiratory_rate/benchmark_multimodal.py

# 2. Menjalankan benchmark dataset primer klinis (data_primer 21 subjek)
.\env\Scripts\python.exe respiratory_rate/benchmark_primer.py

# 3. Menjalankan evaluasi rekaman data pengukuran GUI (data_pengukuran)
.\env\Scripts\python.exe respiratory_rate/benchmark_pengukuran.py

# 4. Menjalankan unit tests
.\env\Scripts\python.exe -m unittest discover -s respiratory_rate -p "test_*.py"
```

---

## 6. Integrasi dengan GUI & Processing

Pipeline terintegrasi langsung pada `ECGProcessor` dan `PPGProcessor`:

```python
from processing_data.processing_data import ECGProcessor

processor = ECGProcessor(target_fs=125)
r_peaks, _ = processor.detect_r_peaks(ecg, fs=125)

# Pemanggilan Multimodal Fusion (ECG + PPG):
rr, resp_signal, resp_peaks = processor.calculate_respiration_rate(
    ecg=ecg, r_peaks=r_peaks, fs=125, ppg=ir_clean, ppg_peaks=ir_peaks
)

# Ambil detail diagnostik
details = processor.last_respiration_details
print(details["mode"], details["quality"], details["feature_weights"])
```
