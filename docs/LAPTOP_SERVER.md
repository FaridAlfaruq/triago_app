# Laptop sebagai server TriaGO sementara

Raspi → HiveMQ Cloud → backend di laptop → database + dashboard.
Pengunjung → link HTTPS tunnel → dashboard di laptop.
Kode aplikasi tidak dipindahkan ke hosting saat uji ini. Laptop dan kedua
terminal backend/tunnel harus menyala. Gunakan data dummy untuk tahap awal.

## 1. Persiapan di laptop Windows

Dari folder root repository, gunakan Python 3.10+ dan virtual environment.
Contoh berikut memakai `.venv` yang sudah ada:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m service.laptop init
```

`init` membuat `.env` dari `config/laptop.env.example`, dengan password dashboard
acak. Jika `.env` sudah ada, file tidak ditimpa; cocokkan isinya dengan contoh laptop.
File `.env` dan `runtime/` tidak dilacak Git.

Isi tiga nilai HiveMQ secara langsung di `.env`:

```dotenv
TRIAGO_MQTT_HOST=hostname-cluster-kamu.s1.eu.hivemq.cloud
TRIAGO_MQTT_USERNAME=username-mqtt-kamu
TRIAGO_MQTT_PASSWORD=password-mqtt-kamu
```

HOST adalah hostname tanpa `https://`, path, atau port. Port TLS memakai `8883`.
Untuk uji seluruh aplikasi pada satu laptop, credential tersebut perlu izin
Publish dan Subscribe pada topic `triago/biomed/devices/raspi-01/triage` dan
`triago/biomed/devices/raspi-01/assignment` (balasan nomor bed ke GUI).
Publisher dan subscriber menggunakan client ID berbeda walau credential sama.
Password dashboard adalah `TRIAGO_DASHBOARD_PASSWORD`, berbeda dari password MQTT.
Jangan kirim password ke chat, memasukkannya ke URL, atau commit `.env`.

```powershell
.\.venv\Scripts\python.exe -m service.laptop check
```

Pengecekan ini memvalidasi konfigurasi; belum membuktikan koneksi broker.

## 2. Terminal pertama: jalankan backend + dashboard

```powershell
.\.venv\Scripts\python.exe -m service.laptop run
```

Buka `http://127.0.0.1:5000` di browser. Browser meminta username/password:
gunakan `TRIAGO_DASHBOARD_USER` dan `TRIAGO_DASHBOARD_PASSWORD` dari `.env`.
Semua pemegang login bisa melihat data dan melakukan aksi dashboard.
Ini login bersama untuk prototipe, belum akun per pengguna/role.

SQLite dibuat otomatis di `runtime/triago-laptop.sqlite3`; tidak perlu MySQL
atau phpMyAdmin untuk pengujian laptop. Database pengukuran ini berbeda dari
file antrean MQTT. MySQL tetap menjadi mode default untuk konfigurasi backend
lainnya: pilih `TRIAGO_DB_BACKEND=mysql` dan isi koneksi MySQL saat dibutuhkan.
Tidak ada fallback otomatis saat MySQL gagal.

Di browser, buka `/api/health` pada alamat yang sama. Harapkan:

Status juga bisa dibaca tanpa menyalin password ke command line:

```powershell
.\.venv\Scripts\python.exe -m service.laptop status
```

```json
{
  "mqtt": {"enabled": true, "connected": true, "subscribed": true, "pending": 0},
  "database": {"engine": "sqlite"},
  "access": {"login_required": true}
}
```

Ini potongan respons health. `connected` dan `subscribed` perlu bernilai true
sebelum mengirim pengukuran pertama. Jika belum, cek hostname, kredensial,
izin topic, internet dan port 8883. Credential berhasil di proyek lain belum
menjamin izin topic TriaGO sudah sesuai.

## 3. Terminal kedua: kirim satu hasil simulasi melalui HiveMQ

```powershell
.\.venv\Scripts\python.exe -m service.laptop demo
```

Demo mengirim vital sign sintetis berlabel `SIMULASI MQTT`. Pesan benar-benar
dikirim ke HiveMQ menggunakan client Paho; tidak dipintas langsung ke API lokal.
Outbox kosong berarti broker menerima pengiriman, belum konfirmasi database.
Cek pasien di dashboard, `/api/history`, serta `pending: 0` pada backend.
Jika backend belum pernah subscribe, broker bisa tidak menyimpan pesan yang
dikirim saat backend belum aktif. Jalankan backend dahulu.

## 4. Terminal kedua/ketiga: buat link HTTPS dari luar jaringan

Pasang CLI resmi Cloudflare ke folder lokal proyek:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_cloudflared.ps1
```

Installer mengambil rilis Windows x64 resmi dan memeriksa checksum SHA256
asset sebelum digunakan. Tidak memasang service Windows atau mengubah PATH.
Jika `cloudflared` sudah tersedia di PATH, langkah ini bisa dilewati.

```powershell
.\.venv\Scripts\python.exe -m service.laptop share
```

Perintah `share` memastikan backend benar-benar menolak akses tanpa login
sebelum menjalankan tunnel. Perintah ini menampilkan URL seperti
`https://nama-acak.trycloudflare.com`. Bagikan link dan login dashboard
melalui jalur pribadi. Pengunjung tidak membutuhkan kredensial HiveMQ/database.
Jangan gunakan URL yang menyertakan username/password.

Link menggunakan HTTPS di sisi pengunjung. Tunnel meneruskan koneksi ke
`127.0.0.1` di laptop; tidak perlu membuka port router. Mode laptop tidak
menyalakan Flask debug. Middleware login juga melindungi API dan `/socket.io`,
dan browser dari origin lain ditolak. Jangan menambahkan CORS `*` untuk uji ini.

Berhenti: Ctrl+C pada terminal tunnel dan backend. Link mati ketika tunnel
berhenti; menjalankannya ulang menghasilkan alamat baru. Laptop tidur atau
internet putus juga membuat aplikasi tidak dapat diakses. Quick Tunnel untuk
pengujian, tanpa jaminan uptime dan maksimal 200 request bersamaan; memakai
Socket.IO/WebSocket, bukan SSE.

Sumber: [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

## 5. Hubungkan Raspi yang sebenarnya

Jika kode belum tersedia di Raspi, ikuti
[langkah uji Raspi dan pemindahan publisher](RASPI_MQTT.md).

Di Raspi gunakan `config/raspi.env.example` sebagai `.env`, isi cluster/topic
yang sama dan credential dengan izin Publish. Jalankan GUI seperti sebelumnya.
Frontend dan backend tetap di laptop; hanya pengukuran yang berasal dari Raspi.
Jangan menjalankan publisher demo dan GUI Raspi dengan client ID sama.
Credential Subscriber di laptop boleh dipisahkan dari Publisher setelah uji awal.

## 6. Pindah ke ITS nanti

Pindahkan kode backend, `service/`, frontend `website/`, dan dependensi ke ITS.
Isi environment di server; rahasia tidak ikut Git. Jika ingin menggunakan
MySQL, siapkan tabel `patient_data` dan koneksinya. Data SQLite tidak otomatis
berpindah ke MySQL. Ganti tunnel dengan domain HTTPS dan reverse proxy ITS
yang meneruskan koneksi WebSocket. Jalankan satu proses backend untuk versi ini.
Raspi tetap mengirim ke broker yang sama, sehingga domain dashboard tidak
perlu dimasukkan ke Raspi untuk jalur MQTT.

Status bed masih in-memory dan reset saat backend restart. Retry MQTT belum
menjamin exactly-once jika proses mati tepat setelah commit database; lihat
[batas dan format MQTT](MQTT.md). Login bersama ini juga perlu diganti/diperluas
sesuai pengelolaan akses pengguna ketika aplikasi dipakai secara operasional.

HiveMQ mengumumkan Serverless berakhir pada **31 Desember 2026**. Cluster yang
masih aktif bisa dipakai untuk percobaan, tetapi broker perlu dipindah sebelum
tanggal tersebut. Perubahan broker memerlukan konfigurasi TLS, hostname,
credential dan izin pada kedua sisi, bukan menulis ulang logika triase.
[Pengumuman HiveMQ](https://docs.hivemq.com/hivemq-cloud/serverless-end-of-life.html).
