# Machine-learning artifacts

Folder ini memisahkan model berdasarkan fungsi dan runtime-nya. Artefak baru
disimpan sebagai **research artifacts** sampai seluruh preprocessing, parameter
normalisasi, dan pengujian deployment tersedia.

## Struktur

- `model.py` dan `triage_model.joblib`: implementasi lama yang masih dipakai
  `GUI/loading_page.py`. Lokasinya dipertahankan untuk menjaga kompatibilitas.
- `triage_xgboost/`: notebook penelitian, model ONNX, dan kontrak model triase.
- `bpnet/`: notebook penelitian, model TFLite, dan kontrak model estimasi tekanan
  darah.

Setiap model memiliki `README.md` dan `model_contract.json`. Kontrak tersebut
menjelaskan bentuk tensor, urutan fitur atau kanal, mapping keluaran, serta
artefak pendamping yang masih belum tersedia.

## Status deployment

Model belum boleh dipakai langsung untuk keputusan klinis. Sebelum integrasi ke
aplikasi, lengkapi dan uji hal berikut:

1. preprocessing produksi yang identik dengan notebook;
2. parameter imputasi atau normalisasi yang berasal dari data latih;
3. golden test yang membandingkan runtime sumber dan runtime deployment;
4. evaluasi pada data eksternal yang representatif;
5. penanganan input hilang, input di luar rentang, dan kegagalan runtime.
