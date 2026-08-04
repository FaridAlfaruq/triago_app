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

Algoritma di [`processing_data/processing_data.py`](../processing_data/processing_data.py) menggunakan metode kuadratik *Ratio-of-Ratios* ($R$):

$$R = \frac{AC_{\text{red}} / DC_{\text{red}}}{AC_{\text{ir}} / DC_{\text{ir}}}$$

$$\text{SpO}_2 = A \cdot R^2 + B \cdot R + C$$

Dengan koefisien kalibrasi saat ini:
- $A = 3.069398$
- $B = -5.149127$
- $C = 99.79428$

---

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
