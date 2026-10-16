# Model triase dengan pain score

Runtime dan ONNX pain score berasal dari `origin/test` pada commit
`c774dac1d8565054901cd03853019cb3da54ca00`. Artifact ONNX terakhir diperbarui
oleh commit `26b0b6fe0d0c042adea2e1b06f6bfb46079c2fcd` (edit model triage).
Commit `c774dac` memperbarui notebook, bukan mengekspor ulang artifact ONNX.
Integrasi ini menggunakan artifact yang benar-benar disimpan di Git tersebut;
tidak melatih ulang atau mengklaim hasil evaluasi notebook terbaru.

`triage_xgboost_model.onnx` menerima float32 dengan bentuk `[N, 11]`.
Urutan fitur ada pada `TRIAGE_FEATURES` di `deployment_inference.py`:

1. temperature_c
2. spo2
3. respiratory_rate
4. heart_rate
5. systolic_bp
6. diastolic_bp
7. gcs_total
8. pain_score
9. shock_index
10. pulse_pressure
11. MAP

Delapan input pertama berasal dari hasil pengukuran/registrasi; tiga lainnya
dihitung oleh wrapper dengan rumus yang sama seperti script pelatihan branch
`test`. Nama fitur `pulse_pressure` dalam artifact ini dipakai untuk rasio
SBP/DBP, dan `MAP` dihitung dari DBP + rasio/3. Nama tersebut dipertahankan
untuk kompatibilitas artifact; integrasi MQTT tidak mengubah rumus/model atau
ambang aturan pasca-inferensinya secara diam-diam.

Label 0/1/2 mengikuti wrapper sumber: RESUSITASI, DARURAT, NON-DARURAT.
Skor dan label akhir termasuk aturan pasca-inferensi di wrapper. Skor akhir
tersebut bisa berbeda dari probabilitas mentah ONNX.

Wrapper memeriksa jumlah fitur model saat dimuat, agar file ONNX lama 21-fitur
tidak dipakai dengan pembentuk input 11-fitur. Jangan memperbarui artifact
tanpa memperbarui wrapper dan memeriksa kesesuaian inputnya.

`xgboost_triase.py` diselaraskan dengan script pelatihan di branch `test`;
kredensial tertulis dalam sumber tidak dipindahkan. Bila pelatihan memerlukan
akses dataset, gunakan konfigurasi environment sendiri. Script pelatihan tidak
diimpor atau dijalankan saat GUI melakukan inferensi.

Tes memverifikasi format model, alur data, dan perilaku kode yang diintegrasikan.
Tes tersebut bukan evaluasi ulang akurasi atau validasi klinis model.
