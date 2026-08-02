# Triage XGBoost ONNX

Model ini mengklasifikasikan pasien ke tiga kelas CTM dari tujuh tanda vital.
Notebook melatih XGBoost, menambahkan sepuluh fitur turunan, lalu mengekspor
model final ke ONNX.

## Artefak

- `notebooks/Triage_XGBoost.ipynb`: proses training dan ekspor.
- `models/triage_xgboost_model.onnx`: model ONNX hasil ekspor.
- `model_contract.json`: kontrak input, output, preprocessing, dan label.

Model ONNX telah lolos pemeriksaan struktur. Input-nya `float_input` bertipe
`float32` dengan bentuk `[batch, 17]`. Output-nya adalah `label` dan
`probabilities` untuk tiga kelas:

- `0`: Resuscitation (ESI 1)
- `1`: Urgent (ESI 2 dan 3)
- `2`: Non-Urgent (ESI 4 dan 5)

## Preprocessing wajib

ONNX hanya memuat classifier. Aplikasi harus melakukan imputasi median,
clipping fisiologis, feature engineering, dan penyusunan 17 fitur dalam urutan
yang tepat sebelum inferensi. Nilai median data latih belum tersedia dalam
artefak yang diterima. Karena itu, input yang memiliki nilai kosong belum dapat
diproses secara konsisten dengan training.

Nama fitur sengaja dilepas saat konversi ONNX. Jangan menyimpulkan urutan fitur
dari model biner; gunakan `model_contract.json`.

## Hasil notebook

| Model | Accuracy | F1-macro |
|---|---:|---:|
| Baseline | 0.8766 | 0.8851 |
| Final/tuned yang diekspor | 0.8743 | 0.8745 |

Model final yang diekspor memiliki F1-macro 1,20% lebih rendah daripada
baseline pada hold-out set. Recall kelas Resuscitation meningkat menjadi 0,96,
tetapi precision-nya turun menjadi 0,80. Pemilihan model harus ditentukan dari
biaya kesalahan klinis dan pengujian lanjutan, bukan accuracy saja.

Notebook mencatat satu sampel ONNX menghasilkan probabilitas yang sama dengan
XGBoost asli. Tambahkan golden test multi-sampel sebelum model diintegrasikan.
