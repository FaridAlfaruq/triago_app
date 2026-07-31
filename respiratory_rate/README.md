# Respiratory Rate ECG-Only

Folder ini berisi seluruh pipeline RR baru:

- `pipeline.py`: kode utama yang dipakai aplikasi;
- `benchmark_bidmc.py`: pembanding metode lama dan baru;
- `test_pipeline.py`: sanity test;
- `results/`: hasil benchmark BIDMC.

Dataset tetap berada di `draft_filter/bidmc` agar notebook lama tidak rusak.

## Cara kerja kode

Pipeline dimulai dari:

```python
estimator = ECGRespirationEstimator()
result = estimator.estimate(ecg, r_peaks, fs=125)
```

Di dalam `estimate()`, urutannya:

### 1. Rapikan posisi R-peak

```python
refined = self._refine_r_peaks(ecg, r_peaks, fs)
```

R-peak dari detektor awal bisa bergeser beberapa sampel. Fungsi ini mencari
ekstremum QRS terdekat agar fitur yang diambil konsisten.

### 2. Buat empat fitur per denyut

```python
refined, beat_times, features = self._extract_edr_features(
    ecg, r_peaks, fs
)
```

Fitur yang diambil:

- `r_amplitude`: perubahan tinggi QRS;
- `qrs_area`: perubahan luas QRS;
- `qrs_slope`: perubahan kemiringan QRS;
- `rr_interval`: perubahan jarak antardetak.

Perubahan tersebut dapat mengikuti proses inspirasi dan ekspirasi.

### 3. Tambahkan baseline wander

```python
grid, signals = self._make_edr_signals(
    ecg, beat_times, features, fs
)
```

Empat fitur per denyut diinterpolasi menjadi sinyal 4 Hz. Fungsi yang sama
menambahkan `baseline_wander`, yaitu gerakan baseline ECG pada frekuensi
pernapasan.

Semua sinyal difilter pada 0,1–0,7 Hz atau 6–42 napas/menit.

### 4. Cari frekuensi dan kualitas

```python
result = self._analyze_signal(signal)
```

Welch PSD digunakan untuk mencari frekuensi dominan. Quality score dihitung
dari ketajaman peak spektrum dan periodisitas sinyal.

### 5. Gabungkan fitur terbaik

```python
result = self._analyze_window(signals, mask)
```

Spektrum dengan kualitas lebih baik mendapat bobot lebih besar. Jadi pipeline
tidak bergantung pada satu fitur ECG yang belum tentu bagus pada semua pasien.

### 6. Keluarkan hasil

```python
rr = result["rr"]
quality = result["quality"]
resp_signal = result["resp_signal"]
```

Jika durasi kurang dari 20 detik atau fitur tidak cukup, `rr` akan bernilai
`0.0` dan `quality` juga `0.0`.

## Pemakaian dari ECGProcessor

```python
from processing_data.processing_data import ECGProcessor

processor = ECGProcessor(target_fs=125)
r_peaks, _ = processor.detect_r_peaks(ecg, fs=125)
rr, resp_signal, resp_peaks = processor.calculate_respiration_rate(
    ecg, r_peaks, fs=125
)

quality = processor.last_respiration_details["quality"]
```

Metode lama masih tersedia:

```python
processor.calculate_respiration_rate_legacy(ecg, r_peaks, fs=125)
```

## Menjalankan benchmark

```powershell
python respiratory_rate/benchmark_bidmc.py
```

## Hasil BIDMC

Evaluasi menggunakan 53 subjek, anotasi napas manual, window 60 detik, dan
step 30 detik.

| Metrik | Lama | Baru |
|---|---:|---:|
| MAE | 2,758 bpm | 1,645 bpm |
| RMSE | 3,672 bpm | 3,631 bpm |
| Bias | +1,733 bpm | -0,250 bpm |
| MAPE | 18,47% | 10,17% |
| Dalam ±2 bpm | 48,48% | 82,53% |
| Coverage | 99,62% | 99,37% |

Jika akurasi ditulis sebagai `100 - MAPE`, nilainya berubah dari 81,53%
menjadi **89,83%**.

## Referensi metode

- https://pmc.ncbi.nlm.nih.gov/articles/PMC7612521/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC2929127/
- https://pubmed.ncbi.nlm.nih.gov/25118665/
- https://physionet.org/content/bidmc/1.0.0/
