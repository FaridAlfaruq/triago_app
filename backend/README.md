# Menjalankan Backend TriaGO di Raspberry Pi

Backend Flask melayani dashboard dan API pada port `5000`. Secara default
server bind ke `0.0.0.0`, sehingga perangkat lain di jaringan lokal dapat
mengaksesnya menggunakan alamat IP Raspberry Pi.

## 1. Jalankan di Raspberry Pi

```bash
cd ~/triagos/triago_app
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m backend.app
```

Jangan menjalankan server dengan `sudo`.

## 2. Cari IP Raspberry Pi

Di terminal Raspberry Pi:

```bash
hostname -I
```

Contoh hasil:

```text
192.168.1.42
```

## 3. Buka dari laptop

Pastikan laptop dan Raspberry Pi berada pada Wi-Fi/LAN yang sama, lalu buka:

```text
http://192.168.1.42:5000
```

Ganti `192.168.1.42` dengan hasil `hostname -I`.

Tes API dari browser:

```text
http://192.168.1.42:5000/api/health
```

Respons yang benar:

```json
{"port": 5000, "service": "triago-backend", "status": "ok"}
```

## 4. Jika firewall aktif

Periksa:

```bash
sudo ufw status
```

Jika statusnya aktif, buka port backend:

```bash
sudo ufw allow 5000/tcp
```

## Konfigurasi opsional

```bash
export TRIAGO_HOST=0.0.0.0
export TRIAGO_PORT=5000
export TRIAGO_DEBUG=0
python -m backend.app
```

GUI yang berjalan di Raspberry Pi menggunakan
`http://127.0.0.1:5000` secara default. Untuk mengarahkan API client ke host
lain:

```bash
export TRIAGO_API_URL=http://192.168.1.42:5000
python GUI/main_gui.py
```

## Pemeriksaan masalah jaringan

Di Raspberry Pi:

```bash
ss -ltnp | grep 5000
```

Di laptop:

```bash
curl http://192.168.1.42:5000/api/health
```

Jika health-check di Raspberry Pi berhasil tetapi laptop gagal terhubung,
periksa apakah jaringan mengaktifkan client/AP isolation.
