# Integrasi pain score branch test ke iot-hivemq

Pain score sekarang mengikuti alur:
registrasi → worker pemrosesan → XGBoost ONNX → GUI hasil → MQTT/HTTP
→ backend → database → dashboard.

Perubahan registrasi, runtime inferensi, model ONNX, dan script pelatihan
diambil dari `origin/test` (`c774dac`, dengan artifact dari `26b0b6f`).
Perubahan algoritma RR dari branch test tidak ikut dipindahkan pada integrasi ini.
Model/wrapper menggunakan 11 fitur, termasuk pain score; lihat
[asal dan kontrak model](../model/README.md).

## Raspi

Setelah commit integrasi tersedia pada branch `iot-hivemq`, hentikan GUI dan
perbarui kode dari root repository Raspi:

```bash
git status --short --branch
git pull --ff-only
```

Jangan melanjutkan bila ada konflik/perubahan lokal yang belum disimpan.
Aktifkan environment GUI yang sama dan jalankan `python GUI/main_gui.py`.
Registrasi mewajibkan pilihan bed, GCS, dan pain score **0–10**. Nol adalah
pilihan eksplisit; tanpa memilih pain score tombol Mulai tetap nonaktif.
Saat kembali ke registrasi, pilihan pain juga direset.

Model dan wrapper ONNX harus diperbarui bersama. `.env` dan konfigurasi broker
yang sudah berhasil tidak perlu diganti. Gunakan pengukuran percobaan dahulu
untuk memeriksa alur sistem. Grafik waveform tetap di Raspi; payload MQTT
mengirim hasil akhir, pain score dan skor XGBoost.

Metadata JSON pengukuran menyertakan `Pain Score`. Registrasi baru ditulis ke
`data_pengukuran/data_pendaftaran_pasien_pain.csv`, agar header CSV lama yang
belum memiliki kolom pain score tidak rusak. File CSV lama tidak dimodifikasi.

## Backend laptop dan database

Perbarui branch `iot-hivemq` di laptop, kemudian restart backend agar memuat
kode penerima/penyimpanan yang baru. Restart akan mereset status bed in-memory
seperti versi sebelumnya; riwayat pengukuran di database tetap tersimpan.

SQLite menambahkan kolom nullable `pain_score` secara otomatis saat startup.
Baris lama tetap NULL. MySQL/MariaDB memerlukan migrasi satu kali sebelum
mengaktifkan backend baru:

```sql
ALTER TABLE patient_data ADD COLUMN pain_score TINYINT UNSIGNED NULL;
```

Pilih database yang benar terlebih dahulu dan periksa apakah kolom sudah ada.
SQL yang sama tersedia di `backend/migrations/001_add_pain_score.sql`.
Tidak ada koneksi atau perubahan MySQL server yang dilakukan oleh integrasi lokal ini.

Payload MQTT tetap memakai `schema_version: 1`, dengan tambahan field opsional
di dalam `data`:

```json
{"pain_score": 8}
```

Nilai yang diisi harus integer 0–10. Pesan lama tanpa field ini atau dengan
`null` tetap diterima, dan dashboard menampilkan `—`, bukan menganggapnya nol.
Nilai nol yang benar-benar diisi ditampilkan sebagai `0 / 10`.

Untuk inferensi lokal tanpa pain score, wrapper/worker sumber masih memakai
fallback 0 dengan peringatan input; GUI baru mewajibkan pemilihan agar jalur
pengukuran baru tidak mengandalkan fallback tersebut.
