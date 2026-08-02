# TriaGO BPNet v5.3 TFLite

BPNet memperkirakan systolic blood pressure (SBP) dan diastolic blood pressure
(DBP) dari segmen ECG dan PPG. Artefak ini berbeda dari classifier XGBoost dan
karena itu ditempatkan dalam subfolder tersendiri.

## Artefak

- `notebooks/BPNet_v53_training.ipynb`: preprocessing, arsitektur, training,
  evaluasi, dan konversi TFLite.
- `models/triago_bpnet_v53_quant.tflite`: model TFLite dynamic-range quantized.
- `model_contract.json`: kontrak tensor, kanal, dan normalisasi target.

Model TFLite telah lolos pemeriksaan struktur. Model menerima satu segmen
`float32` berbentuk `[1, 1250, 7]`, yaitu 10 detik data pada 125 Hz. Tujuh
kanalnya adalah:

1. ECG ternormalisasi;
2. PPG ternormalisasi;
3. turunan pertama PPG (VPG);
4. turunan kedua PPG (APG);
5. hasil perkalian ECG dan PPG;
6. envelope energi Hilbert ECG;
7. envelope energi Hilbert PPG.

Gunakan nama output pada signature TFLite, bukan posisi output: `sbp_output`
dan `dbp_output`.

## Batasan penting

Keluaran model masih berupa Z-score. `target_scaler_params.json` yang berisi
mean dan standard deviation SBP/DBP data latih belum tersedia. Tanpa parameter
tersebut, keluaran tidak dapat dikonversi secara benar menjadi mmHg.

Nama file menggunakan istilah quantized, tetapi input, output, dan operasi
aktivasi tetap `float32`; optimasi yang dipakai adalah dynamic-range
quantization pada bobot, bukan model full INT8.

Notebook yang diterima tidak menyimpan hasil eksekusi evaluasi atau benchmark.
Karena itu klaim MAE, AAMI/BHS, accuracy drift, dan latensi belum dapat
diverifikasi dari artefak ini. Model harus tetap berstatus research-only sampai
scaler, preprocessing produksi, dan golden test tersedia.
