# Uji Raspi → HiveMQ → backend laptop

Backend laptop sudah mendukung MQTT. Raspi perlu kode branch `iot-hivemq` dan
konfigurasi cluster yang sama. Jangan menimpa perubahan lokal Raspi: periksa
status Git dahulu, dan hentikan proses bila Git melaporkan perubahan/konflik
yang belum dipahami.

## Ambil versi GitHub langsung pada repository Raspi

Dari root repository TriaGO yang sudah ada di Raspi, jalankan:

```bash
git status --short --branch
git fetch origin
git switch iot-hivemq
git pull --ff-only
```

`git switch` akan membuat branch lokal dari `origin/iot-hivemq` jika branch
lokal belum ada dan remote tersebut satu-satunya yang cocok. Jika Git menolak
karena perubahan lokal atau file untracked, jangan gunakan reset/clean/force;
simpan keluaran error untuk ditinjau. Data pengukuran tidak perlu dihapus.

Aktifkan environment Python yang selama ini dipakai aplikasi GUI, kemudian:

```bash
python -m pip install -r requirements-mqtt.txt
cp -n config/raspi.env.example .env
nano .env
```

Jika `.env` sudah ada, `cp -n` tidak mengubahnya; cocokkan dengan contoh Raspi
secara manual. Isi koneksi HiveMQ seperti bagian 2. Dengan backend laptop
aktif, kirim demo dahulu dari root repo memakai `python -m service.send_demo`.
Setelah berhasil, jalankan `python GUI/main_gui.py` dari desktop Raspi.
Panduan paket terpisah di bawah tetap bisa dipakai untuk uji tanpa mengubah repo.

## 1. Pastikan penerima di laptop aktif

Di laptop Windows, dari root proyek:

```powershell
.\.venv\Scripts\python.exe -m service.laptop status
```

Jika backend belum berjalan, buka terminal lain:

```powershell
.\.venv\Scripts\python.exe -m service.laptop run
```

Status MQTT perlu `connected: true`, `subscribed: true`. Buka dashboard
`http://127.0.0.1:5000` di laptop dan login dengan credential dashboard `.env`
laptop. Dashboard tidak perlu dibuka ke internet untuk menerima hasil Raspi.
Raspi dan laptop sama-sama perlu internet, tetapi tidak harus satu Wi-Fi.

## 2. Uji terpisah di Raspi dahulu

Paket `triago-raspi-mqtt.tar.gz` berisi publisher, contoh konfigurasi, dan
dependensi MQTT saja. Tidak berisi password laptop, data pasien, GUI, atau model.
Salin paket itu ke folder Downloads Raspi, kemudian buka terminal Raspi:

```bash
python3 --version
mkdir -p ~/triago-mqtt-test
tar -xzf ~/Downloads/triago-raspi-mqtt.tar.gz -C ~/triago-mqtt-test
cd ~/triago-mqtt-test
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-mqtt.txt
```

Gunakan Python 3.10+. Jika paket disalin ke folder lain, sesuaikan lokasi arsip
pada perintah `tar`. Folder uji ini terpisah dari aplikasi TriaGO lama.

Buat konfigurasi jika belum ada, tanpa menimpa konfigurasi uji sebelumnya:

```bash
cp -n config/raspi.env.example .env
nano .env
```

Isi hostname cluster, username dan password MQTT secara langsung di Raspi.
Hostname harus sama dengan cluster di laptop, tanpa `https://`, path, atau port.
Username/password boleh sama dengan yang diuji di laptop jika credential
memiliki izin Publish. Tidak perlu password dashboard atau MySQL di Raspi.

```dotenv
TRIAGO_TRANSPORT=mqtt
TRIAGO_DEVICE_ID=raspi-01
TRIAGO_MQTT_HOST=hostname-cluster-kamu
TRIAGO_MQTT_PORT=8883
TRIAGO_MQTT_USERNAME="username-mqtt-kamu"
TRIAGO_MQTT_PASSWORD="password-mqtt-kamu"
TRIAGO_MQTT_TOPIC_PREFIX=triago/biomed
TRIAGO_MQTT_CLIENT_ID=triago-raspi-01-publisher
```

Client ID Raspi harus berbeda dari client ID backend dan demo laptop. Jangan
salin `.env` laptop mentah-mentah: file Raspi tidak perlu mengaktifkan subscriber
atau berisi password dashboard. Di nano: Ctrl+O, Enter untuk menyimpan, Ctrl+X
untuk keluar.

Kirim satu pengukuran sintetis:

```bash
.venv/bin/python -m service.send_demo
```

Catat event ID yang dicetak. `Outbox kosong` membuktikan broker mengakui
pengiriman; pada terminal backend laptop, event ID yang sama harus muncul
dengan log `selesai disimpan dan diproses`. Dashboard laptop menampilkan hasil
bernama **SIMULASI MQTT** pada bed yang dialokasikan. Hasil ini tidak berasal
dari sensor. Jika semua bed penuh, hasil disimpan sebagai WAITING dan dapat
diperiksa lewat `/api/history` setelah login.

Jika belum berhasil, kirim keluaran terminal atau pesan error tanpa `.env`
atau password. Jangan mengirim ulang berkali-kali sebelum memeriksa status;
pesan tertunda tetap tersimpan dalam outbox SQLite.

## 3. Setelah demo Raspi berhasil: integrasikan GUI pengukuran

Periksa branch dan perubahan lokal di repository Raspi terlebih dahulu:

```bash
git status --short --branch
git log -1 --oneline
```

Jangan menimpa seluruh folder aplikasi atau file GUI yang sedang dikembangkan.
Gunakan branch Git `iot-hivemq` seperti petunjuk di atas, atau salin file
publisher dengan backup bila memang diperlukan setelah meninjau versi Raspi.
Publisher berintegrasi lewat `TriageApiClient.send_triage_result`, sehingga
nama metode lama tetap didukung. GUI versi baru juga menghapus IP backend
yang ditulis tetap dan membaca transport dari `.env`.

Pada repository Raspi yang sudah menerima update, pasang dependensi MQTT ke
environment GUI yang memang dipakai, isi `.env` Raspi, lalu jalankan GUI:

```bash
python -m pip install -r requirements-mqtt.txt
python GUI/main_gui.py
```

Jalankan dari sesi desktop Raspi agar jendela Qt terlihat. Hentikan demo publisher
sebelum GUI memakai client ID yang sama. Mode MQTT tidak memerlukan backend
Flask di Raspi; backend dan database tetap di laptop. Hasil model yang tidak
valid tidak dikirim oleh GUI.
