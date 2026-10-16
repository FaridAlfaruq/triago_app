# TriaGO: Raspberry Pi dan backend dengan HiveMQ Cloud

Pain score dan model XGBoost dari branch `test` sudah terintegrasi; lihat
[kontrak data dan langkah pembaruan pain score](PAIN_SCORE.md).

Untuk **laptop sebagai server dengan link HTTPS**, mulai dari
[panduan laptop](LAPTOP_SERVER.md). Profil itu memakai SQLite untuk uji;
panduan di bawah tetap menjelaskan konfigurasi backend MySQL.

Alur data: GUI Raspi → outbox SQLite → HiveMQ Cloud → inbox SQLite backend
→ MySQL + BedManager → WebSocket dashboard. Backend tidak memakai Azure.
Pengolahan sinyal dan model tetap berjalan di Raspi. Yang dikirim adalah
hasil triase, GCS, vital signs, dan identitas pasien jika tersedia; bukan waveform.
Gunakan Python 3.10 atau lebih baru.

## 1. Konfigurasi HiveMQ Cloud

Gunakan cluster pribadi, hostname cluster, dan port TLS `8883`.
Kode selalu memverifikasi sertifikat dan hostname broker.
Buat kredensial terpisah dengan izin topic:

| Perangkat | Izin | Topic |
| --- | --- | --- |
| Raspi 01 | Publish | `triago/biomed/devices/raspi-01/triage` |
| Backend | Subscribe | `triago/biomed/devices/+/triage` |

Setiap Raspi harus punya `TRIAGO_DEVICE_ID` dan `TRIAGO_MQTT_CLIENT_ID`
yang berbeda. Topic prefix harus sama pada Raspi dan backend.
Jangan ganti identitas/topic perangkat sebelum outbox kosong; pesan lama
tetap menggunakan device ID asal dan membutuhkan izin publish untuk ID itu.
Backend menggunakan session MQTT 3.1.1 persisten (`clean_session=False`),
sehingga client ID backend harus tetap sama setelah restart.
Pesan yang dikirim ketika backend offline hanya bisa diantrikan broker jika
backend sudah pernah terhubung dan subscribe; retensi/limit session mengikuti cluster.
Jalankan backend dan pastikan subscription berhasil sebelum pengukuran pertama.

## 2. Sisi Raspberry Pi

Dari root repository, di environment Python yang sudah dipakai GUI:

```bash
python -m pip install 'paho-mqtt>=2.1,<3' 'python-dotenv>=1,<2'
cp config/raspi.env.example .env
```

Edit `.env`: isi hostname, username, password cluster, dan ID perangkat.
Lalu jalankan GUI seperti biasa:

```bash
python GUI/main_gui.py
```

`TRIAGO_TRANSPORT=mqtt` memilih MQTT. Backend Flask lokal di Raspi tidak
perlu berjalan untuk mengirim hasil. GUI menggunakan API client yang sama,
tetapi pengiriman jaringan dilakukan worker background.

Log `[MQTT QUEUED]` berarti hasil sudah tersimpan di outbox Raspi.
Ini **belum berarti tersimpan di MySQL**. Outbox bertahan setelah aplikasi
ditutup dan dikirim ulang otomatis saat GUI berikutnya berjalan dan broker siap.
Konfigurasi yang tidak lengkap atau penyimpanan lokal gagal menghasilkan status gagal.
Kegagalan jaringan tidak membuat GUI menunggu koneksi broker.

Untuk kembali memakai HTTP: set `TRIAGO_TRANSPORT=http` dan `TRIAGO_API_URL`.
Alamat IP lama sudah tidak ditulis tetap di GUI. Tidak ada fallback otomatis
MQTT → HTTP, agar satu pengukuran tidak dikirim lewat dua jalur.

## 3. Sisi backend, untuk pengujian lokal dahulu

Di komputer backend, dari root repository:

```bash
python -m pip install -r backend/requirements.txt
cp config/backend.env.example .env
```

Isi kredensial subscriber HiveMQ dan konfigurasi MySQL yang sudah dipakai TriaGO.
Tabel `patient_data` memakai struktur backend lama dengan tambahan nullable
`pain_score`. Untuk MySQL lama, jalankan migrasi pain score satu kali seperti
[panduan pain score](PAIN_SCORE.md). SQLite mode laptop bermigrasi otomatis.
Password database sekarang harus diisi lewat environment, tanpa password bawaan dalam kode.

```bash
python -m backend.app
```

Dashboard tetap dilayani pada `http://localhost:5000`.
`GET /api/health` menampilkan status MQTT:

```json
{
  "status": "ok",
  "service": "triago-backend",
  "port": 5000,
  "mqtt": {"enabled": true, "connected": true, "subscribed": true, "pending": 0}
}
```

`connected` berarti terhubung ke broker; `subscribed` berarti izin subscription
diterima; `pending` adalah jumlah pesan inbox yang belum selesai diproses.
`TRIAGO_MQTT_ENABLED=0` menonaktifkan subscriber dan mempertahankan API HTTP.
Jalankan **satu proses backend**: BedManager dan receipt sementara berada dalam
memori, dan satu inbox hanya boleh diproses oleh satu worker.
Saat menguji Raspi + backend di komputer yang sama, gunakan environment berbeda
atau set variabel environment pada terminal masing-masing; variabel terminal
mengalahkan `.env` dan client ID keduanya harus berbeda.

## 4. Format pesan dan penanganan kegagalan

Topic pengukuran: `triago/biomed/devices/raspi-01/triage`.
QoS `1`, `retain=False`.

```json
{
  "schema_version": 1,
  "event_id": "9c34353a-c2f1-4a67-b55a-3f0074e633d8",
  "device_id": "raspi-01",
  "data": {
    "bed_id": "A1",
    "gcs_score": 15,
    "pain_score": 3,
    "triage_category": "green",
    "xgboost_score": 0.2,
    "vitals": {"hr": 75, "spo2": 98, "rr": 16, "temp_core": 36.5, "sys": 120, "dia": 80}
  }
}
```

Backend memvalidasi tipe JSON, angka finite, kategori, UUID event, dan kecocokan
device ID dengan topic. Pesan retained, rusak, dan di atas 64 KiB ditolak.
QoS 1 dapat mengirim duplikat: SQLite menyimpan event ID + hash untuk
mengabaikan pengiriman ulang dengan isi yang sama. Payload berbeda dengan
event ID yang sama ditolak.

Backend mengirim PUBACK setelah pesan tersimpan ke inbox, kemudian memprosesnya
di worker terpisah. Bila MySQL gagal, pesan tetap pending dan dicoba ulang
dengan jeda 2–60 detik; bed belum ditempati dan dashboard belum diperbarui.
Jika semua bed penuh, hasil tetap disimpan sebagai `WAITING`.
Sesudah pesan selesai, payload pasien di antrean dihapus, ID/hash disimpan untuk deduplikasi.

Antrean default berada di `runtime/mqtt-outbox.sqlite3` (Raspi) dan
`runtime/mqtt-inbox.sqlite3` (backend). Path bisa diubah lewat
`TRIAGO_MQTT_OUTBOX` / `TRIAGO_MQTT_INBOX`; gunakan path absolut jika diubah.
Folder ini dan `.env` tidak dilacak Git. Jangan hapus/memindahkan antrean
saat worker berjalan. Pending payload tersimpan sebagai plaintext lokal;
batasi akses folder ke akun aplikasi. ID/hash selesai tidak dipangkas otomatis.

Batas saat ini: bila proses/backend mati tepat setelah commit MySQL tetapi
sebelum inbox ditandai selesai, pengiriman ulang setelah restart dapat membuat
baris MySQL duplikat. Ini belum menjamin exactly-once lintas restart; untuk itu
perlu unique event ID dan transaksi inbox bersama MySQL. Status bed juga masih
in-memory seperti backend sebelumnya dan reset saat backend restart.
Konfirmasi MySQL dilihat di backend/riwayat, belum dikirim balik ke layar Raspi.

## 5. Pengujian

```bash
python -m unittest discover -s tests -v
```

Tes memakai broker/database simulasi dan SQLite sungguhan, tanpa perangkat
sensor atau kredensial HiveMQ. Untuk uji jaringan sesungguhnya, jalankan backend
dengan cluster pribadi, kirim satu hasil dari GUI, lalu cek `pending: 0`, riwayat
MySQL dan event dashboard. Matikan jaringan atau MySQL sementara untuk mengecek
outbox/inbox dan retry. Tidak ada koneksi ke broker publik.

Referensi protokol/client:
[HiveMQ Cloud Python example](https://github.com/hivemq-cloud/paho-mqtt-client-example),
[Eclipse Paho Python](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html).
