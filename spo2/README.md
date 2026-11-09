# SpO2 Dual-Wavelength Benchmarking Module

Folder ini berisi seluruh pipeline benchmarking dan pengujian algoritma SpO2 berbasis PPG dual-wavelength (Red & IR):

- `dataset_loader.py`: loader dataset PhysioNet Pulse Transit Time PPG (WFDB `.hea` / `.dat`) dan dataset dual-wavelength PPG lainnya. Memuat dua saluran sinyal:
  - `pleth_1` = Sinyal Red (660 nm)
  - `pleth_2` = Sinyal IR (880 nm)
  - Ground-truth SpO2 (%) yang diekstraksi dari meta-komentar header `<spo2_start>` dan `<spo2_end>`.
- `benchmark_spo2.py`: skrip pengujian berbasis sliding window yang mengevaluasi estimasi SpO2 terhadap ground-truth.
- `results/`: folder output untuk file CSV hasil benchmark per window (`spo2_window_results.csv`) dan ringkasan metrik evaluasi (`spo2_summary.csv`).

---

## Formulasi Algoritma SpO2 Saat Ini

Algoritma di [`processing_data/processing_data.py`](../processing_data/processing_data.py) memakai *Ratio-of-Ratios* ($R$) per denyut:

$$R_{beat} = \frac{AC_{\text{red}} / DC_{\text{red}}}{AC_{\text{ir}} / DC_{\text{ir}}}$$

- Dihitung pada sinyal mentah (sebelum downsample); DC = lowpass 0,3 Hz, AC = bandpass 0,5–5 Hz.
- $R$ akhir = median $R_{beat}$ setelah membuang outlier > 3 MAD (artefak gerak).

Kalibrasi linear:

$$\text{SpO}_2 = 107{,}83 - 25 \cdot R$$

- Slope 25 mengikuti bentuk kurva empiris standar ($110 - 25R$) supaya desaturasi terdeteksi.
- Intercept dikalibrasi ulang pada dataset PhysioNet (22 subjek).

### Kenapa kurva lama diganti

Kurva lama ($3{,}069R^2 - 5{,}149R + 99{,}79$) hampir datar: di $R=1{,}0$ (SpO2 sebenarnya ~85%) hasilnya 97,7%, dan di $R=2{,}0$ hasilnya 100%. Akibatnya pasien hipoksia terbaca normal.

| R | Kurva lama | Kurva baru |
|---:|---:|---:|
| 0,4 | 98,2 | 97,8 |
| 0,6 | 97,8 | 92,8 |
| 0,8 | 97,6 | 87,8 |
| 1,0 | 97,7 | 82,8 |

### Batasan

Ground truth dataset hanya 96–98,5%, jadi bagian kurva di bawah 95% **belum tervalidasi**. Benchmark di dataset ini juga belum bisa membuktikan akurasi: tebakan konstan 97,1% saja sudah dapat MAE 0,61. Untuk validasi yang benar, perlu data dari alat TriaGO sendiri yang direkam bersamaan dengan oksimeter referensi, idealnya mencakup SpO2 < 95%.

## Cara Menjalankan Benchmark

Jalankan perintah berikut di terminal:

```powershell
python spo2/benchmark_spo2.py --data-dir "draft_filter"
```

Opsi tambahan:
- `--data-dir`: Path folder dataset WFDB (default: `draft_filter`).
- `--output-dir`: Path folder simpan hasil (default: `spo2/results`).
- `--window`: Durasi jendela analisis dalam detik (default: `30.0`).
- `--step`: Step sliding window dalam detik (default: `10.0`).

---

## Metrik Evaluasi

- **MAE (Mean Absolute Error)** dalam % SpO2.
- **RMSE (Root Mean Square Error)**.
- **Bias** (Rata-rata error terarah).
- **MAPE (%)** & **Akurasi (%)** ($100 - \text{MAPE}$).
- **Persentase Presisi ($\le \pm 2\%$ SpO2)** dan **($\le \pm 3\%$ SpO2)**.
