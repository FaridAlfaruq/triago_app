# Bank pertanyaan kritis TriaGo

30 pertanyaan latihan, enam untuk setiap kriteria. Mulai dari 10 pertanyaan S. Gunakan pertanyaan lanjutan untuk mendalami jawaban dalam format final 15 menit.

- **1. Kecerdasan piranti (20%)** — S01, S02, S03, A01, A02, A03
- **2. Kompleksitas dan fungsionalitas (20%)** — S04, S05, A04, A05, A06, B01
- **3. Desain dan implementasi (20%)** — S06, S07, A07, A08, B02, B03
- **4. Efektivitas dan adaptabilitas (20%)** — S08, S09, A09, B04, B05, B06
- **5. Presentasi dan demonstrasi (20%)** — S10, A10, B07, B08, B09, B10

## Prioritas S

### S01 — Mengapa akurasi di Tabel 4.2 sebesar 90,67% tetapi confusion matrix memberi 88,20%?

**Kriteria** Kecerdasan piranti

**Pertanyaan lanjutan** Apakah gambar berasal dari versi lama? Mana hasil untuk ONNX demo?

### S02 — Apakah data XGBoost berasal dari pasien IGD nyata dan siapa yang menetapkan labelnya?

**Kriteria** Kecerdasan piranti

**Pertanyaan lanjutan** Apa dasar ESI 2 digabung dengan 1? Bagaimana menguji mapping itu di Indonesia?

### S03 — Bagaimana memastikan MAE BP tidak terlalu optimistis akibat segmen pasien yang sama masuk train dan test?
         
**Kriteria** Kecerdasan piranti

**Pertanyaan lanjutan** Berapa pasien unik di test? Apakah jendela tumpang tindih lintas split?

### S04 — Jika sensor gagal, apakah nilai normal buatan masih bisa memengaruhi triase?

**Kriteria** Kompleksitas dan fungsionalitas

**Pertanyaan lanjutan** Apakah fallback tampil jelas bagi perawat? Bisakah alokasi otomatis tetap berjalan?

### S05 — Apakah grade B/A berarti BP TriaGo sudah tervalidasi klinis?

**Kriteria** Kompleksitas dan fungsionalitas

**Pertanyaan lanjutan** Apakah orang hipotensi dan hipertensi terwakili? Berapa galat bias per subjek?

### S06 — Saat Wi-Fi putus, apa yang tetap berjalan dan bagaimana mencegah pesan ganda?

**Kriteria** Desain dan implementasi

**Pertanyaan lanjutan** Apa beda PUBACK dan konfirmasi bed? Bagaimana jika disk atau broker gagal?

### S07 — Dua perangkat meminta satu bed terakhir bersamaan. Siapa yang menang?

**Kriteria** Desain dan implementasi

**Pertanyaan lanjutan** Bagaimana setelah restart? Kalau simpan database gagal, apakah bed masih terisi?

### S08 — Apa bukti alat aman dipakai di IGD dan tidak meloloskan pasien gawat?

**Kriteria** Efektivitas dan adaptabilitas

**Pertanyaan lanjutan** Siapa mengambil keputusan akhir? Apa endpoint keselamatan pertama?

### S09 — Apa yang dihitung dalam klaim ≤2 menit, dan apakah bed sudah terkonfirmasi?

**Kriteria** Efektivitas dan adaptabilitas

**Pertanyaan lanjutan** Berapa median dan waktu terlama? Apa yang terjadi saat internet lambat?

### S10 — Bisakah Anda tunjukkan bahwa model di demo sama dengan model yang menghasilkan angka laporan?

**Kriteria** Presentasi dan demonstrasi

**Pertanyaan lanjutan** Berapa hash model yang dijalankan? Bisa tunjukkan satu jejak input–output?

## Prioritas A

### A01 — Berapa kasus Resusitasi yang luput pada matriks, dan apakah model keliru hanya pada kelas itu?

**Kriteria** Kecerdasan piranti

**Pertanyaan lanjutan** Mana jenis kesalahan terburuk? Apa dampak perubahan ambang?

### A02 — Mengapa memilih XGBoost, dan apakah guardrail benar-benar memperbaiki model?

**Kriteria** Kecerdasan piranti

**Pertanyaan lanjutan** Apakah rule-only cukup? Apakah aturan menaikkan false positive?

### A03 — Kode menamai SBP/DBP sebagai pulse pressure. Apakah fitur itu benar secara fisiologis?

**Kriteria** Kecerdasan piranti

**Pertanyaan lanjutan** Kalau dikoreksi, apakah perlu melatih ulang? Apakah angka 90,67% masih berlaku?

### A04 — Mengapa CNN-BiGRU, dan dari mana BP bisa diperkirakan tanpa manset?

**Kriteria** Kompleksitas dan fungsionalitas

**Pertanyaan lanjutan** Apa manfaat ECG dibanding PPG saja? Apakah gerak mengganggu alignment?

### A05 — Bagaimana deteksi denyut jantung dan RR, serta apa buktinya saat sinyal berisik?

**Kriteria** Kompleksitas dan fungsionalitas

**Pertanyaan lanjutan** Mengapa hasil primer berbeda? Apa yang terjadi di luar 10–42 napas per menit?

### A06 — Mengapa akurasi SpO₂ sekitar 99% belum membuktikan alat andal saat saturasi rendah?

**Kriteria** Kompleksitas dan fungsionalitas

**Pertanyaan lanjutan** Berapa data di bawah 90%? Apakah set kalibrasi terpisah?

### A07 — Kenapa ada STM32 dan Raspberry Pi sekaligus? Mengapa inferensi di edge?

**Kriteria** Desain dan implementasi

**Pertanyaan lanjutan** Apa bottleneck serial? Bisakah satu papan saja?

### A08 — Jika zona penuh, apakah pasien kritis dipindah ke bed yang tidak punya fasilitas resusitasi?

**Kriteria** Desain dan implementasi

**Pertanyaan lanjutan** Apakah kuning dapat memakai bed merah? Siapa boleh override?

### A09 — Apakah MAE suhu 0,25 °C benar-benar hasil MLX90614 terhadap suhu inti pasien?

**Kriteria** Efektivitas dan adaptabilitas

**Pertanyaan lanjutan** Suhu kulit dipengaruhi apa? Mana hasil 20 kasus baru?

### A10 — Apa sebenarnya yang dibuktikan video, dan apa yang harus ditunjukkan lewat kode atau log?

**Kriteria** Presentasi dan demonstrasi

**Pertanyaan lanjutan** Bisa tunjukkan event_id yang sama? Bagian mana yang berasal dari rekaman uji?

## Prioritas B

### B01 — Apakah penjepit tangan cocok untuk semua pasien dan aman secara listrik?

**Kriteria** Kompleksitas dan fungsionalitas

**Pertanyaan lanjutan** Bagaimana untuk pasien tremor, edema atau kedua tangan tak bisa dipakai?

### B02 — Apakah TLS dan login cukup untuk melindungi data pasien?

**Kriteria** Desain dan implementasi

**Pertanyaan lanjutan** Siapa bisa membuka riwayat pasien? Apa yang terjadi jika akun bocor?

### B03 — Jika dasbor macet atau basis data tidak tersedia, bagaimana membedakan status tampilan dan status bed?

**Kriteria** Desain dan implementasi

**Pertanyaan lanjutan** Jika proses restart, dari mana status bed dipulihkan?

### B04 — Rp15 juta itu biaya produksi atau harga jual? Apakah pembandingnya setara?

**Kriteria** Efektivitas dan adaptabilitas

**Pertanyaan lanjutan** Berapa BOM aktual? Berapa biaya per pemeriksaan?

### B05 — Powerbank 20.000 mAh dan 5,5–6 jam cukup untuk satu shift IGD?

**Kriteria** Efektivitas dan adaptabilitas

**Pertanyaan lanjutan** Bagaimana ketika daya habis saat akuisisi? Apa mode cadangan?

### B06 — Bagaimana rumah sakit dapat mengadopsi TriaGo dan bagaimana dampaknya akan diukur?

**Kriteria** Efektivitas dan adaptabilitas

**Pertanyaan lanjutan** Siapa pengguna, pembeli, dan pemilik data? Apa indikator SDG yang terukur?

### B07 — Di mana kebaruan TriaGo dan apa yang sebenarnya tercakup HKI?

**Kriteria** Presentasi dan demonstrasi

**Pertanyaan lanjutan** Apa yang tim bangun sendiri? Apakah klaim mencakup perangkat keras?

### B08 — Kontribusi pribadi Anda apa dan bagian mana yang bisa Anda jelaskan jika juri minta kode?

**Kriteria** Presentasi dan demonstrasi

**Pertanyaan lanjutan** Siapa yang menulis firmware? Bisa jelaskan satu trade-off dari kontribusi Anda?

### B09 — Kalau video, broker atau dasbor gagal di depan juri, apa yang bisa tetap Anda buktikan?

**Kriteria** Presentasi dan demonstrasi

**Pertanyaan lanjutan** Apa bukti cadangan untuk sinkronisasi bed? Bisa tunjukkan titik kode?

### B10 — Mengapa abstrak pengantar memakai angka lebih rendah dari laporan akhir?

**Kriteria** Presentasi dan demonstrasi

**Pertanyaan lanjutan** Kalau begitu, mengapa Gambar 4.10 juga menghasilkan 88,20%?
