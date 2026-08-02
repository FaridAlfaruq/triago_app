# Pipeline integrasi ONNX ke GUI

## 1. Startup aplikasi

`LoadingPage` membuat satu `TriageOnnxPredictor` ketika aplikasi dibuka. Adapter
memuat `triage_xgboost_model.onnx`, memvalidasi kontrak `[batch, 17]`, dan
menjalankan satu warm-up. Sesi yang sama dipakai ulang untuk semua pasien.

Langkah ini menggantikan perilaku lama yang membuka model joblib dua kali pada
setiap pemeriksaan.

## 2. Akuisisi sinyal

`ProcessingWorker` menerima ECG, PPG merah, PPG inframerah, suhu, dan GCS. Alur
pemrosesan sinyal yang sudah ada tetap menghitung:

1. heart rate dari R-peak ECG;
2. respiratory rate dari ECG-derived respiration;
3. SpO2 dan perfusion index dari PPG;
4. estimasi suhu inti dari suhu kulit dan lingkungan.

SBP dan DBP masih berasal dari `patient_info`. Selama BPNet belum mempunyai
`target_scaler_params.json`, GUI mempertahankan fallback lama `120/80` dan
menandainya dalam `triage_input_warnings`.

## 3. Penyusunan tujuh tanda vital

Worker menyerahkan nilai scalar berikut ke adapter:

1. `temperature_c`;
2. `spo2`;
3. `respiratory_rate`;
4. `heart_rate`;
5. `systolic_bp`;
6. `diastolic_bp`;
7. `gcs_total`.

Input fallback tidak disembunyikan. Hasil memuat `triage_input_quality` dengan
nilai `measured` atau `fallback`, serta daftar `triage_input_warnings`.

## 4. Preprocessing model

`preprocessing.py` menjalankan:

1. validasi tipe dan nilai finite;
2. penolakan nilai kosong jika median training tidak tersedia;
3. clipping tujuh tanda vital sesuai rentang training;
4. pembuatan sepuluh fitur turunan;
5. penyusunan tensor `float32` dalam urutan kontrak `[1, 17]`.

Perhitungan `news_vital_score` sengaja mereplikasi kondisi notebook secara
persis. Beberapa celah batas desimal pada notebook dipertahankan agar inferensi
sesuai dengan data training. Aturan ini harus diperbaiki melalui retraining,
bukan hanya dengan mengubah preprocessing deployment.

## 5. Inferensi ONNX

Adapter meminta dua output bernama:

- `label`: kelas `0`, `1`, atau `2`;
- `probabilities`: probabilitas ketiga kelas.

Output dipetakan menjadi `RESUSITASI`, `DARURAT`, atau `NON-DARURAT`. Confidence,
seluruh probabilitas, 17 nilai fitur, serta waktu inferensi disimpan dalam hasil
worker. CSV konsolidasi menyimpan backend model, confidence, waktu inferensi,
kualitas input, dan peringatan fallback untuk kebutuhan audit.

## 6. Output dan kegagalan

Jika inferensi berhasil, GUI menampilkan kelas dan mengirim hasil ke backend.
Jika model gagal dimuat, preprocessing gagal, atau output model tidak valid:

- status menjadi `TIDAK TERSEDIA`;
- badge GUI berubah abu-abu;
- hasil tidak dikirim ke backend;
- penyebab tersedia pada `triage_error`.

SHAP model joblib lama tidak digunakan karena tidak menjelaskan model ONNX yang
baru. Grafik SHAP dikosongkan sampai model XGBoost sumber terbaru atau metode
explanation yang tervalidasi tersedia.

## 7. Verifikasi

Jalankan unit dan smoke test:

```powershell
python -m unittest discover -s ml_xgboost/triage_xgboost/tests -v
```

Sebelum penggunaan klinis, tambahkan golden dataset untuk membandingkan
preprocessing notebook, XGBoost sumber, dan ONNX pada banyak sampel.
