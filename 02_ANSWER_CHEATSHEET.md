## Prioritas S — jawaban lengkap

### S01 — Mengapa akurasi di Tabel 4.2 sebesar 90,67% tetapi confusion matrix memberi 88,20%?

**VERIFIED** · Kecerdasan piranti

- **Jawaban utama** Di Tabel 4.2 kami melaporkan akurasi 90,67% dan F1 91,42%. Kalau dihitung langsung dari Gambar 4.10, diagonalnya berisi 14.112 prediksi benar dari 16.000, jadi akurasinya 88,20%. Ada ketidaksesuaian dalam laporan. Asal evaluasi masing-masing masih perlu dicocokkan.
- **Pendalaman** Gambar 4.10 juga menghasilkan macro F1 89,83% jika dihitung dari matriks. Notebook sel 17–18 menyimpan akurasi 90,69% dan macro F1 91,42% pada keluaran tersimpan. Ketiga rujukan ini perlu dipasangkan dengan target, prediksi, split, dan checkpoint masing-masing sebelum dijelaskan sebagai satu evaluasi.
- **Bukti** L hlm. 22 Gbr. 4.10 dan Tabel 4.2; N sel 17–18; V 5:30.
- **Tanya jawab lanjutan**
  - **Apakah gambar berasal dari versi lama?** Hitungan gambar menghasilkan 88,20%, sama dengan angka di abstrak lama. Itu petunjuk untuk menelusuri versi gambar. Asalnya masih perlu dicocokkan dengan keluaran evaluasi.
  - **Mana hasil untuk ONNX demo?** Angka tabel dan gambar bisa saya tunjukkan sumbernya. Untuk ONNX demo, hasil yang tepat adalah evaluasi dari file yang benar-benar dimuat saat rekaman. Pasangan file dan hasil itu belum terkonfirmasi.
- **Batas klaim** Sebut masing-masing angka dengan sumbernya. Jangan menggabungkan metrik tabel dengan matriks seolah satu evaluasi.
- **Kata kunci** tabel, matriks, run, rekonsiliasi.

### S02 — Apakah data XGBoost berasal dari pasien IGD nyata dan siapa yang menetapkan labelnya?

**PARTIALLY VERIFIED** · Kecerdasan piranti

- **Jawaban utama** Triagegeist adalah dataset simulasi. Label acuity lima tingkat berasal dari dataset, lalu kami memetakannya menjadi tiga kelas untuk purwarupa. Jadi angka di laporan menggambarkan hasil benchmark dengan pemetaan tersebut. Penerapannya pada pasien IGD nyata memerlukan validasi tersendiri.
- **Pendalaman** Notebook memuat Triagegeist dan split terstratifikasi per baris. Kode memetakan ESI 1–2 ke Resusitasi, 3 ke Darurat, 4–5 ke Non-Darurat. Kode pelatihan lain memiliki pembangkit data sintetis lokal saat data tak tersedia. Provenans ONNX demo masih perlu manifest.
- **Bukti** L hlm. 21 §4.3.8; N sel 2–3, 7, 11; model/xgboost_triase.py:39, 101; [Triagegeist](https://www.kaggle.com/competitions/triagegeist/data).
- **Tanya jawab lanjutan**
  - **Apa dasar ESI 2 digabung dengan 1?** Kami menggabungkan ESI 1 dan 2 sebagai kelas prioritas tertinggi pada pemetaan lima ke tiga kelas. Itu keputusan pengelompokan untuk benchmark. Kesesuaiannya dengan kategori Resusitasi dalam praktik lokal belum ditetapkan.
  - **Bagaimana menguji mapping itu di Indonesia?** Uji pada data IGD lokal yang penentuan prioritasnya dilakukan tenaga medis tanpa melihat prediksi model. Bandingkan hasil per kelas, terutama kasus yang diberi prioritas terlalu rendah.
- **Batas klaim** Pemetaan adalah desain tim. Jangan menyebut benchmark sebagai uji pada pasien IGD atau menganggap mapping setara pedoman.
- **Kata kunci** sintetis, label, ESI, validasi.

### S03 — Bagaimana memastikan MAE BP tidak terlalu optimistis akibat segmen pasien yang sama masuk train dan test?

**PARTIALLY VERIFIED** · Kecerdasan piranti

- **Jawaban utama** Unit pembagiannya adalah pasien. Notebook BP menggunakan GroupShuffleSplit supaya seluruh jendela milik satu ID pasien masuk ke satu bagian data. Untuk angka MAE di laporan, daftar ID dari run tersebut tetap diperlukan untuk memastikan tidak ada pasien atau rekaman yang muncul di kedua bagian.
- **Pendalaman** Notebook sel 4 mencoba menggabungkan subject ID ke patient ID, dengan fallback bila mapping tidak tersedia. Sel 19–20 memakai GroupShuffleSplit, menyaring target dan mencari split seimbang. Periksa duplikat lintas ID, statistik normalisasi hanya dari train, dan keputusan pemilihan split.
- **Bukti** L hlm. 21 §4.3.7; model/bp.ipynb sel 4, 18–20.
- **Tanya jawab lanjutan**
  - **Berapa pasien unik di test?** Dataset primer dilaporkan berisi 94 subjek. Jumlah pasien unik di bagian uji perlu dibaca dari daftar ID split, dan angka itu belum terkonfirmasi.
  - **Apakah jendela tumpang tindih lintas split?** Kalau ID pasien dipetakan dengan benar, semua jendela pasien yang sama masuk ke satu bagian data. Bukti untuk run laporan adalah daftar ID tiap split dan pemeriksaan duplikat rekaman.
- **Batas klaim** Ada kode pembagian kelompok. Jangan menjamin tidak ada leakage pada run laporan tanpa manifest dan hasil cek irisan.
- **Kata kunci** pasien unik, split, jendela, manifest.

### S04 — Jika sensor gagal, apakah nilai normal buatan masih bisa memengaruhi triase?

**VERIFIED** · Kompleksitas dan fungsionalitas

- **Jawaban utama** Ya. Kalau estimasi BP gagal, kode saat ini bisa memasukkan 120/80 mmHg sebagai nilai pengganti dan menandainya sebagai fallback. Nilai itu tetap masuk ke model dan bisa memengaruhi kategori. Dalam alur klinis yang kami usulkan, petugas mengukur ulang dan menilai pasien secara manual sebelum memakai rekomendasi itu.
- **Pendalaman** GUI juga mencatat fallback SpO₂ 98, RR 16, HR 75, GCS 15, dan nyeri 0 ketika input terkait kosong. Input tersebut perlu ditelusuri sampai tampilan/payload. Perilaku versi demo belum dipastikan.
- **Bukti** GUI/loading_page.py:150, 219, 244, 263, 297; L hlm. 14 Gbr. 3.4.
- **Tanya jawab lanjutan**
  - **Apakah fallback tampil jelas bagi perawat?** Layar BP pada kode saat ini menampilkan label fallback 120/80. Status triase juga punya peringatan saat kursor diarahkan ke label. Seberapa mudah petugas menyadari peringatan itu perlu diuji dengan pengguna.
  - **Bisakah alokasi otomatis tetap berjalan?** Bisa pada kode saat ini. Pengiriman ke backend memeriksa apakah hasil inferensi valid, dan fallback sendiri tidak memblokir pengiriman. Jika backend menerima dan memproses hasilnya, bed masih bisa dialokasikan.
- **Batas klaim** Klaim kode memberi penanda. Jangan menyebut semua nilai terukur atau mengklaim sistem memblokir semua fallback.
- **Kata kunci** fallback, BP 120/80, peringatan, manual.

### S05 — Apakah grade B/A berarti BP TriaGo sudah tervalidasi klinis?

**PARTIALLY VERIFIED** · Kompleksitas dan fungsionalitas

- **Jawaban utama** Grade B/A merangkum hasil evaluasi tekanan darah yang kami laporkan. Dibandingkan manset komersial, MAE-nya 4,90 mmHg untuk sistolik dan 2,30 mmHg untuk diastolik. Hasil itu mendukung estimasi pada kelompok yang diuji. Untuk penggunaan di IGD, kami masih menempatkannya sebagai prototipe dan memakai manset untuk konfirmasi.
- **Pendalaman** Di Tabel 4.1, persentase hasil dalam ±5, ±10, dan ±15 mmHg berturut-turut 59,76%, 87,23%, dan 97,35% untuk sistolik, serta 89,40%, 98,07%, dan 100% untuk diastolik. Angka ini menggambarkan sebaran galat pada data evaluasi. Verifikasi seluruh protokol BHS membutuhkan rincian peserta, pembanding, dan prosedur pengukuran.
- **Bukti** L hlm. 21 Gbr. 4.9, Tabel 4.1; M hlm. 4 Gbr. 7; V 5:00.
- **Tanya jawab lanjutan**
  - **Apakah orang hipotensi dan hipertensi terwakili?** Laporan menyebut mayoritas subjek berusia 18–22 tahun. Keterwakilan hipotensi dan hipertensi perlu dilihat dari distribusi pembacaan manset pada bagian uji, yang belum terkonfirmasi.
  - **Berapa galat bias per subjek?** Angka yang tersedia di laporan adalah MAE gabungan. Untuk bias tiap subjek, hitung rata-rata prediksi dikurangi nilai manset sehingga terlihat apakah estimasi cenderung terlalu tinggi atau rendah. Nilai per subjek belum tersedia di laporan.
- **Batas klaim** Sebut B/A sebagai kategori yang dilaporkan. Jangan mengklaim sertifikasi atau pengganti manset.
- **Kata kunci** MAE, manset, populasi, BHS.

### S06 — Saat Wi-Fi putus, apa yang tetap berjalan dan bagaimana mencegah pesan ganda?

**VERIFIED** · Desain dan implementasi

- **Jawaban utama** Akuisisi sensor dan inferensi berlangsung lokal, jadi keduanya tetap bisa berjalan saat Wi-Fi putus. Hasil masuk antrean lokal untuk dikirim ulang saat koneksi kembali. Backend memakai ID unik tiap kejadian untuk mengenali kiriman ulang. Alokasi bed baru dianggap terkonfirmasi setelah perangkat menerima balasan server.
- **Pendalaman** QoS 1 bisa menduplikasi pesan. Publisher menunggu PUBACK broker, sementara backend menyimpan inbox, memproses, dan membalas alokasi secara terpisah. Ada tes untuk kehilangan PUBACK dan pengiriman ulang; perilaku lintas crash dan deployment belum diukur di sini.
- **Bukti** L hlm. 24 §4.5; service/mqtt_client.py:97, 142; backend/mqtt_receiver.py:43; tests/test_mqtt.py:96, 225.
- **Tanya jawab lanjutan**
  - **Apa beda PUBACK dan konfirmasi bed?** PUBACK adalah tanda bahwa broker MQTT menerima publikasi QoS 1. Konfirmasi bed adalah balasan aplikasi setelah backend memproses permintaan alokasi.
  - **Bagaimana jika disk atau broker gagal?** Kalau broker tidak tersedia, pesan yang sudah tersimpan menunggu pengiriman ulang. Kalau penyimpanan lokal gagal, pesan mungkin tidak masuk antrean dan perangkat harus menganggap pengiriman belum berhasil.
- **Batas klaim** Jelaskan mekanisme pada kode sekarang. Jangan menjanjikan exactly-once, tanpa kehilangan data, atau alokasi offline.
- **Kata kunci** edge, outbox, event_id, konfirmasi.

### S07 — Dua perangkat meminta satu bed terakhir bersamaan. Siapa yang menang?

**PARTIALLY VERIFIED** · Desain dan implementasi

- **Jawaban utama** Dalam satu proses backend, permintaan yang lebih dulu memperoleh lock akan memeriksa dan mengalokasikan bed terlebih dahulu. Permintaan berikutnya melihat status bed setelah alokasi pertama selesai. Jaminan itu berlaku dalam satu proses. Beberapa worker memerlukan koordinasi melalui penyimpanan bersama.
- **Pendalaman** TriageService memiliki lock dan cache receipt. BedManager juga mengunci pemilihan dan penyimpanan bed. Kunci lokal tidak dibagi antarproses. Konfigurasi worker dan jaminan unik di deployment belum dibuktikan.
- **Bukti** L hlm. 24; backend/triage_service.py:17; backend/bed_manager.py:76, 85; tests/test_bed_layout.py:45.
- **Tanya jawab lanjutan**
  - **Bagaimana setelah restart?** Saat startup, backend membuat BedManager baru. Database dapat menyimpan riwayat dan antrean MQTT dapat bertahan, sementara status bed aktif dikelola di memori. Pemulihan otomatis status bed saat startup belum terkonfirmasi.
  - **Kalau simpan database gagal, apakah bed masih terisi?** Pada jalur MQTT, simpan ke database dilakukan sebelum bed ditandai terisi. Jika simpan gagal, permintaan itu belum mengisi bed dan pemrosesan dijadwalkan ulang.
- **Batas klaim** Sebut cakupan satu proses. Jangan menjamin bebas perebutan lintas worker atau restart.
- **Kata kunci** lock, satu proses, transaksi, restart.

### S08 — Apa bukti alat aman dipakai di IGD dan tidak meloloskan pasien gawat?

**PARTIALLY VERIFIED** · Efektivitas dan adaptabilitas

- **Jawaban utama** Kami menguji kinerja algoritma, mendemonstrasikan purwarupa pada subjek sehat, dan berkonsultasi dengan dokter. Bukti itu mendukung pengembangan TriaGo sebagai alat bantu triase. Untuk mengukur risiko pasien gawat diberi prioritas terlalu rendah, kami perlu membandingkan rekomendasi alat dengan penilaian tenaga medis pada pasien IGD.
- **Pendalaman** Pisahkan kesalahan model pada data simulasi dari under-triage pasien nyata. Uji prospektif perlu kasus kritis, protokol override dan kegagalan sensor, serta luaran yang disepakati dengan IGD. Saran laporan menyebut validasi langsung sebagai rencana pengembangan.
- **Bukti** L hlm. 22–23 §4.4, hlm. 25–27 §4.8 dan §5.2; V 7:00–7:30 cuplikan konsultasi.
- **Tanya jawab lanjutan**
  - **Siapa mengambil keputusan akhir?** Dalam penggunaan yang kami usulkan, tenaga medis tetap mengambil keputusan akhir dan dapat mengabaikan rekomendasi prototipe.
  - **Apa endpoint keselamatan pertama?** Untuk uji prospektif, saya akan mengutamakan proporsi kasus yang dinilai perlu resusitasi oleh tenaga medis tetapi diberi prioritas lebih rendah oleh sistem.
- **Batas klaim** Sebut hasil uji purwarupa dan masukan dokter. Jangan mengklaim aman klinis, penurunan mortalitas, atau siap menggantikan petugas.
- **Kata kunci** purwarupa, subjek sehat, klinisi, prospektif.

### S09 — Apa yang dihitung dalam klaim ≤2 menit, dan apakah bed sudah terkonfirmasi?

**PARTIALLY VERIFIED** · Efektivitas dan adaptabilitas

- **Jawaban utama** Yang bisa kami tunjukkan dari kode adalah perekaman 60 detik setelah sekitar dua detik pemanasan. Angka ≤2 menit berasal dari laporan. Kami belum bisa memastikan apakah durasi tersebut sudah menghitung registrasi, pemasangan alat, inferensi, dan balasan alokasi bed.
- **Pendalaman** Pisahkan pasang alat, registrasi GCS/nyeri, akuisisi, inferensi, dan balasan bed. Latensi XGBoost per sampel hanya satu bagian. Video rekaman yang diedit tidak menggantikan pengukuran waktu utuh.
- **Bukti** L hlm. 2, 7, 22 Tabel 4.2; GUI/plot_page.py:134–145; V 4:05.
- **Tanya jawab lanjutan**
  - **Berapa median dan waktu terlama?** Kode menunjukkan konfigurasi rekam 60 detik. Median dan durasi terlama untuk seluruh proses harus dihitung dari log per pengulangan, dan angka itu belum tersedia.
  - **Apa yang terjadi saat internet lambat?** Akuisisi dan inferensi berlangsung di perangkat. Koneksi lambat dapat menunda pengiriman dan balasan server, jadi bed belum boleh dianggap terkonfirmasi.
- **Batas klaim** Sebut ≤2 menit sebagai angka laporan. Jangan mengklaim jaminan waktu sampai alokasi atau lebih cepat dari triase manual tanpa uji sebanding.
- **Kata kunci** 60 detik, pemanasan, waktu ujung ke ujung, log.

### S10 — Bisakah Anda tunjukkan bahwa model di demo sama dengan model yang menghasilkan angka laporan?

**PARTIALLY VERIFIED** · Presentasi dan demonstrasi

- **Jawaban utama** Saya bisa menunjukkan arsitektur di laporan, tampilan hasil di video, dan jalur pemuatan model pada kode saat ini. Laporan menyebut BP melalui TFLite dan triase melalui ONNX. Untuk menyatakan file yang dijalankan saat demo sama dengan file yang diuji, saya perlu mencocokkan identitas kedua artefak.
- **Pendalaman** Video memperlihatkan hasil di layar, tetapi tidak menyatakan hash. Tautkan file model dan manifest ke snapshot kode, input evaluasi, dan satu prediksi yang dapat dilacak. Periksa apakah model BP revisi sudah ada saat rekaman.
- **Bukti** L hlm. 12–14; GUI/loading_page.py:20, 459–470; model/revised_bp_manifest.json; V 4:05.
- **Tanya jawab lanjutan**
  - **Berapa hash model yang dijalankan?** Hash mengidentifikasi file model secara spesifik. Hash file di workspace menunjukkan salinan saat ini. Hash artefak yang dijalankan ketika demo direkam belum terkonfirmasi.
  - **Bisa tunjukkan satu jejak input–output?** Saya bisa menunjukkan fitur yang masuk ke model dan bagaimana GUI membaca hasilnya. Itu menjelaskan alurnya. Contoh input dan output yang tercatat untuk file model demo belum tersedia.
- **Batas klaim** Sebut arsitektur laporan dan jalur kode saat ini secara terpisah. Jangan menyatakan keduanya sama tanpa bukti.
- **Kata kunci** snapshot, checkpoint, TFLite, ONNX.

## Prioritas A — jawaban lengkap

### A01 — Berapa kasus Resusitasi yang luput pada matriks, dan apakah model keliru hanya pada kelas itu?

**VERIFIED** · Kecerdasan piranti

- **Jawaban utama** Pada Gambar 4.10, 48 dari 644 kasus berlabel Resusitasi diprediksi Darurat. Ada juga 1.005 kasus Darurat yang diprediksi Non-Darurat. Dua angka ini menunjukkan arah kesalahan pada benchmark, yang penting untuk menilai risiko under-triage.
- **Pendalaman** Recall Resusitasi gambar 596/644 ≈92,55%. Untuk Darurat, 1.005/8.472 ≈11,86% turun ke Non-Darurat. Pisahkan tingkat kesalahan per kelas dari penilaian klinis. Verifikasi ulang setelah model dan data run dikunci.
- **Bukti** L hlm. 22 Gbr. 4.10 dan Tabel 4.3.
- **Tanya jawab lanjutan**
  - **Mana jenis kesalahan terburuk?** Kasus paling berisiko secara klinis adalah Resusitasi yang diprediksi Darurat. Ada 48 kasus seperti itu di gambar. Darurat ke Non-Darurat lebih sering, yaitu 1.005 kasus.
  - **Apa dampak perubahan ambang?** Ambang eskalasi yang lebih longgar bisa menangkap lebih banyak kasus gawat sekaligus menaikkan eskalasi keliru. Besarnya perubahan harus dihitung pada data uji yang sama.
- **Batas klaim** Sebut jumlah pada gambar. Jangan menyebut 48 kejadian klinis atau mengasumsikan seluruh metrik tabel berasal dari gambar.
- **Kata kunci** 48, 644, under-triage, matriks.

### A02 — Mengapa memilih XGBoost, dan apakah guardrail benar-benar memperbaiki model?

**PARTIALLY VERIFIED** · Kecerdasan piranti

- **Jawaban utama** Kami memilih XGBoost karena akurasi dan F1-nya tertinggi di antara model yang dibandingkan pada tabel laporan. Aturan eskalasi ditambahkan untuk menangani beberapa nilai input kritis. Kontribusi aturan itu terhadap kinerja perlu dipisahkan dari kinerja XGBoost sendiri.
- **Pendalaman** Bandingkan LR, SVM, RF dan DT pada split serta fitur identik. Latensi tabel XGBoost 0,0037 ms/sampel lebih lambat dari LR 0,0005. Kode memeriksa antara lain GCS, SpO₂, SBP dan HR. Ablasi ONNX sebelum/sesudah aturan diperlukan.
- **Bukti** L hlm. 14 Gbr. 3.4, hlm. 22 Tabel 4.2; model/deployment_inference.py:88.
- **Tanya jawab lanjutan**
  - **Apakah rule-only cukup?** Aturan saja bisa dijadikan pembanding karena langsung menangani kondisi yang memenuhi ambang. Untuk tahu apakah cukup, bandingkan aturan saja, XGBoost saja, dan gabungannya pada data uji yang sama.
  - **Apakah aturan menaikkan false positive?** Bisa. Aturan eskalasi dapat menangkap lebih banyak kasus gawat sekaligus menambah kasus yang keliru dinaikkan prioritasnya. Untuk mengetahui besar efeknya, bandingkan confusion matrix pada data yang sama.
- **Batas klaim** Klaim performa terbaik pada tabel laporan. Jangan menyebut tercepat, skor terkalibrasi, atau guardrail tervalidasi.
- **Kata kunci** XGBoost, baseline, guardrail, ablasi.

### A03 — Kode menamai SBP/DBP sebagai pulse pressure. Apakah fitur itu benar secara fisiologis?

**VERIFIED** · Kecerdasan piranti

- **Jawaban utama** Di kode, nama fitur pulse pressure mengacu pada rasio SBP dibagi DBP. Tekanan nadi secara fisiologis dihitung dari SBP dikurangi DBP. Jadi ketika menjelaskan model sekarang, saya akan menyebut fitur itu sebagai rasio dan menjelaskan bahwa fitur bernama MAP juga ikut memakai rasio tersebut.
- **Pendalaman** Rumus MAP di jalur itu memakai DBP ditambah sepertiga fitur rasio. Notebook, skrip dan inferensi perlu dibandingkan agar perubahan rumus tidak menciptakan perbedaan training dan deployment. Performa setelah koreksi belum diukur.
- **Bukti** N sel 12; model/xgboost_triase.py:111; model/deployment_inference.py:32.
- **Tanya jawab lanjutan**
  - **Kalau dikoreksi, apakah perlu melatih ulang?** Ya, jika kami ingin model memakai fitur dengan rumus yang dikoreksi, rumus harus disamakan pada pelatihan dan inferensi. Model perlu dilatih serta diuji ulang.
  - **Apakah angka 90,67% masih berlaku?** Angka 90,67% tetap angka yang dilaporkan untuk evaluasi lama. Setelah definisi fitur diubah, performanya harus diukur ulang.
- **Batas klaim** Jelaskan perhitungan aktual. Jangan menyebut variabel turunan sebagai pulse pressure atau MAP klinis yang valid.
- **Kata kunci** rasio, selisih, MAP, retraining.

### A04 — Mengapa CNN-BiGRU, dan dari mana BP bisa diperkirakan tanpa manset?

**PARTIALLY VERIFIED** · Kompleksitas dan fungsionalitas

- **Jawaban utama** CNN mengambil pola bentuk gelombang ECG dan PPG. BiGRU mempelajari urutan pola dalam jendela rekaman. Dari informasi itu model menghasilkan estimasi sistolik dan diastolik. Pada evaluasi primer yang dilaporkan, pembacaan manset dipakai sebagai nilai pembanding.
- **Pendalaman** Laporan juga menyebut turunan PPG dan attention. Waktu dari aktivitas ECG ke denyut perifer dapat memuat waktu pra-ejeksi, sehingga selang itu bukan pengukuran langsung transit arteri murni. Besar kontribusi tiap komponen perlu ablation pada data sama.
- **Bukti** L hlm. 9, 14 Gbr. 3.3, hlm. 21; M hlm. 2 Gbr. 2; model/bpnet_inference.py:136, 334.
- **Tanya jawab lanjutan**
  - **Apa manfaat ECG dibanding PPG saja?** ECG memberi penanda waktu aktivitas listrik jantung yang dapat dibandingkan dengan gelombang denyut perifer pada PPG. Manfaat tambahannya untuk akurasi perlu diuji dengan pembanding PPG saja.
  - **Apakah gerak mengganggu alignment?** Ya. Gerakan dapat menimbulkan puncak semu atau menggeser puncak yang terdeteksi, sehingga penyelarasan ECG dan PPG bisa meleset.
- **Batas klaim** Sebut estimasi dari sinyal dan pembanding. Jangan mengklaim mengukur tekanan langsung atau membuktikan CNN-BiGRU lebih unggul tanpa ablation.
- **Kata kunci** ECG, PPG, morfologi, temporal.

### A05 — Bagaimana deteksi denyut jantung dan RR, serta apa buktinya saat sinyal berisik?

**PARTIALLY VERIFIED** · Kompleksitas dan fungsionalitas

- **Jawaban utama** Denyut jantung dihitung dari jarak antarpuncak R pada ECG. RR, atau laju napas, diestimasi dari modulasi yang muncul pada ECG dan PPG, dengan bobot berdasarkan kualitas sinyal. Sebagai pembanding, hasil tersimpan menunjukkan MAE RR sekitar 0,96 napas per menit pada BIDMC dan 2,44 pada data primer. Keduanya perlu disebut dengan konteks masing-masing.
- **Pendalaman** Laporan mencatat MAE denyut jantung 0,695 bpm dan laju napas 0,96 napas/menit pada BIDMC. Untuk laju napas, hasil tersimpan mencakup 795 jendela BIDMC, sedangkan ringkasan data primer mencakup 21 evaluasi. Perbedaan jumlah dan sumber data penting saat membandingkan galat.
- **Bukti** L hlm. 8, 19–20; respiratory_rate/results/bidmc_multimodal_summary.csv; respiratory_rate/results/primary_data_summary.csv.
- **Tanya jawab lanjutan**
  - **Mengapa hasil primer berbeda?** MAE BIDMC dan data primer berasal dari rekaman serta konteks pengukuran yang berbeda. Ringkasannya tidak cukup untuk memisahkan pengaruh sensor, kualitas sinyal, dan pembanding.
  - **Apa yang terjadi di luar 10–42 napas per menit?** Estimator multimodal mencari frekuensi pada rentang yang dikonfigurasi. Napas di luar rentang 10–42 per menit dapat terlewat atau menghasilkan estimasi yang keliru.
- **Batas klaim** Pisahkan performa algoritma pada BIDMC dan evaluasi primer. Jangan mengklaim ketahanan untuk semua artefak atau RR ekstrem.
- **Kata kunci** puncak R, fusi, BIDMC, MAE.

### A06 — Mengapa akurasi SpO₂ sekitar 99% belum membuktikan alat andal saat saturasi rendah?

**PARTIALLY VERIFIED** · Kompleksitas dan fungsionalitas

- **Jawaban utama** Untuk SpO₂, MAX30105 membaca PPG merah dan inframerah. Kami mengubah rasio komponen berdenyut terhadap garis dasar menjadi estimasi saturasi. MAE 1,08 poin persentase pada 22 subjek adalah ringkasan seluruh dataset. Kinerja di bawah 90% perlu dibaca dari jumlah dan galat pada kelompok itu.
- **Pendalaman** Laporan memakai 107,83 − 25R. Rentang label, kondisi gerak, perfusi rendah, dan independensi data kalibrasi/uji menentukan validitas. CSV lokal menyimpan 1.046 jendela dengan MAE 1,080, tetapi RMSE 1,231 dibanding 1,47 di laporan. Versinya belum ditautkan.
- **Bukti** L hlm. 8, 20 §4.3.5; M hlm. 3–4; spo2/results/spo2_summary.csv.
- **Tanya jawab lanjutan**
  - **Berapa data di bawah 90%?** Yang tercatat adalah MAE agregat 1,08 poin persentase. Untuk menjawab jumlah data di bawah 90%, kami perlu menghitung langsung dari nilai referensi per subjek atau per jendela. Jumlahnya belum terkonfirmasi.
  - **Apakah set kalibrasi terpisah?** Pemisahan yang relevan adalah pada tingkat subjek. Daftar rekaman kalibrasi dan uji yang dapat membuktikan pemisahan itu belum terkonfirmasi.
- **Batas klaim** Sebut angka laporan dan satuan poin persentase. Jangan mengklaim sertifikasi atau akurasi yang sama pada hipoksemia.
- **Kata kunci** merah, IR, rasio, desaturasi.

### A07 — Kenapa ada STM32 dan Raspberry Pi sekaligus? Mengapa inferensi di edge?

**VERIFIED** · Desain dan implementasi

- **Jawaban utama** STM32 kami pakai untuk akuisisi sensor. Data dikirim lewat serial USB ke Raspberry Pi, yang menjalankan pemrosesan sinyal, inferensi, dan antarmuka. Dengan inferensi di perangkat, hasil pengukuran bisa dihitung tanpa menunggu server. Jaringan dipakai untuk pengiriman hasil dan balasan alokasi bed.
- **Pendalaman** I²C untuk PPG dan suhu, input analog untuk ECG. Laporan membedakan sampling ECG/PPG 400 Hz dan suhu 2 Hz. Publikasi hasil dan alokasi bed membutuhkan jalur jaringan. Pengukuran jitter dan latency serial perlu log firmware.
- **Bukti** L hlm. 7 Gbr. 2.1, hlm. 13 Tabel 3.2, hlm. 17; akusisi_data/get_stm32.py:39; V 2:20, 3:00.
- **Tanya jawab lanjutan**
  - **Apa bottleneck serial?** Kapasitas serial bergantung pada jumlah sampel per detik dan ukuran setiap paket. Jika beban mendekati kapasitas, antrean bisa bertambah atau paket bisa hilang. Besarnya belum diukur pada data yang tersedia.
  - **Bisakah satu papan saja?** Bisa secara desain bila satu papan menyediakan akuisisi yang cukup stabil, antarmuka analog dan digital yang diperlukan, serta kapasitas inferensi. Desain saat ini membagi tugas itu antara STM32 dan Pi.
- **Batas klaim** Jelaskan pembagian kerja dan alur. Jangan menyatakan jitter terukur atau kerja penuh tanpa jaringan.
- **Kata kunci** STM32, serial USB, Pi, edge.

### A08 — Jika zona penuh, apakah pasien kritis dipindah ke bed yang tidak punya fasilitas resusitasi?

**PARTIALLY VERIFIED** · Desain dan implementasi

- **Jawaban utama** Bisa terjadi secara logika prototipe. Saat zona utama penuh, BedManager mencoba zona cadangan sesuai urutan dalam kode. Sistem memeriksa ketersediaan bed berdasarkan zona. Petugas tetap perlu memastikan fasilitas di bed tujuan sesuai sebelum memindahkan pasien.
- **Pendalaman** Kode mengatur prioritas zona di BedManager dan hijau dapat berstatus menunggu. Peta zona model bukan daftar kapasitas klinis. Perlu definisi bed layak, eskalasi manual dan cara menaikkan prioritas pasien menunggu.
- **Bukti** L hlm. 24 §4.5; M hlm. 3; backend/bed_manager.py:76; tests/test_bed_layout.py:52, 61, 84.
- **Tanya jawab lanjutan**
  - **Apakah kuning dapat memakai bed merah?** Bisa menurut aturan prototipe. Pasien kuning mencoba bed kuning, lalu hijau, kemudian merah jika dua zona sebelumnya penuh.
  - **Siapa boleh override?** Dalam alur klinis yang diusulkan, petugas berwenang di IGD harus bisa menolak rekomendasi bed bila fasilitasnya tidak sesuai. Mekanisme override dan kewenangan akun spesifik belum saya verifikasi.
- **Batas klaim** Klaim aturan zona prototipe. Jangan menjamin setiap fallback bed cocok untuk resusitasi.
- **Kata kunci** zona, overflow, kelayakan, override.

### A09 — Apakah MAE suhu 0,25 °C benar-benar hasil MLX90614 terhadap suhu inti pasien?

**PARTIALLY VERIFIED** · Efektivitas dan adaptabilitas

- **Jawaban utama** Angka 0,250 °C adalah ringkasan 100 kasus dataset sekunder. Untuk 20 kasus baru, MAE yang dilaporkan 0,237 °C. Pada perangkat, MLX90614 membaca suhu kulit dan lingkungan, lalu model mengestimasi suhu tubuh dengan bantuan indeks perfusi PPG. Jadi dua angka tadi berlaku untuk evaluasi model yang disebut di laporan.
- **Pendalaman** Laporan menyebut kalibrasi 80 dan uji 20, serta uji fungsional dengan termogun pada alat. Kode memakai PI dalam persen. Verifikasi asal variabel kulit/ambient/PI pada dataset, pembanding dan kondisi akuisisi sebelum menyatakan ekuivalensi dengan suhu inti.
- **Bukti** L hlm. 9 Pers. 2.5–2.6, hlm. 17–18, hlm. 20 §4.3.6; model/temperature_estimation.py:17, 29.
- **Tanya jawab lanjutan**
  - **Suhu kulit dipengaruhi apa?** Suhu kulit dipengaruhi suhu sekitar, perfusi, lokasi dan posisi pengukuran. Perubahan tersebut dapat mengubah pembacaan tanpa perubahan yang sama pada suhu tubuh.
  - **Mana hasil 20 kasus baru?** Hasil 20 kasus baru tercantum di laporan halaman PDF 20, §4.3.6, dengan MAE 0,237 °C. Angka 0,250 °C adalah ringkasan seluruh 100 kasus.
- **Batas klaim** Sebut hasil menurut kondisi uji. Jangan menyebut 0,25 °C sebagai galat terverifikasi perangkat pada IGD.
- **Kata kunci** kulit, lingkungan, perfusi, VitalDB.

### A10 — Apa sebenarnya yang dibuktikan video, dan apa yang harus ditunjukkan lewat kode atau log?

**PARTIALLY VERIFIED** · Presentasi dan demonstrasi

- **Jawaban utama** Di video saya bisa menunjukkan alur IoT sekitar menit 3:00 dan hasil yang tampil di perangkat sekitar 4:05. Untuk menjelaskan pengiriman sampai alokasi, saya bisa menunjukkan cara kode mencocokkan pesan dan balasan dengan event_id. Bukti satu transaksi demo yang lengkap memerlukan log dengan ID yang sama di kedua ujung.
- **Pendalaman** Pada 3:00 terlihat diagram, sekitar 4:05 layar hasil, 5:30 tabel model, dan 6:00 tabel per kelas. Untuk bukti ujung ke ujung perlu timestamp serta event ID, status DB, dan balasan bed dari run yang sama.
- **Bukti** V 3:00, 4:05, 5:30, 6:00; L hlm. 15–16 dan 24.
- **Tanya jawab lanjutan**
  - **Bisa tunjukkan event_id yang sama?** Kode bisa menunjukkan di mana event_id dibuat dan dipakai untuk mencocokkan balasan. Untuk menunjukkannya pada video, saya perlu log transaksi yang merekam ID yang sama pada pengiriman dan balasan. Jejak itu belum tersedia.
  - **Bagian mana yang berasal dari rekaman uji?** Sekitar 4:05 terlihat layar hasil purwarupa. Pada 5:30 dan 6:00 tampil tabel evaluasi. Potongan tersebut memperlihatkan hasil dan angka yang disajikan tanpa log lengkap dari satu transaksi.
- **Batas klaim** Klaim yang terlihat pada video. Jangan menganggap potongan video sebagai bukti latency atau ketahanan operasional.
- **Kata kunci** video, layar, event_id, bukti.

## Prioritas B — jawaban lengkap

### B01 — Apakah penjepit tangan cocok untuk semua pasien dan aman secara listrik?

**PARTIALLY VERIFIED** · Kompleksitas dan fungsionalitas

- **Jawaban utama** Penjepit membuat sensor dan elektroda bisa dipasang bersamaan. Kami mendemonstrasikannya pada subjek sehat dalam beberapa posisi. Untuk pasien yang kedua tangannya tidak dapat dipasangi penjepit, alur yang kami usulkan memakai prosedur pengukuran standar. Keselamatan listrik perangkat lengkap belum tervalidasi.
- **Pendalaman** Laporan menunjukkan desain mekanik dan rangkaian. Prioritas uji berikutnya mencakup kontak elektroda, tekanan penjepit, kebersihan antarpasien, isolasi/leakage, dan pasien dengan akses tangan terbatas.
- **Bukti** L hlm. 12–13 Gbr. 3.1–3.2, hlm. 18, 23 Gbr. 4.12; V 7:55.
- **Tanya jawab lanjutan**
  - **Bagaimana untuk pasien tremor, edema atau kedua tangan tak bisa dipakai?** Jika tremor atau edema membuat sinyal tidak layak, atau penjepit tidak bisa dipasang, gunakan pengukuran standar yang sesuai di IGD. Uji yang kami laporkan belum menetapkan performa untuk kondisi tersebut.
- **Batas klaim** Klaim ada desain dan uji posisi. Jangan klaim cocok untuk semua anatomi, aman tersertifikasi, atau merekam ECG diagnostik.
- **Kata kunci** penjepit, kontak, isolasi, posisi.

### B02 — Apakah TLS dan login cukup untuk melindungi data pasien?

**PARTIALLY VERIFIED** · Desain dan implementasi

- **Jawaban utama** TLS melindungi transmisi MQTT dan kode memverifikasi sertifikat. Dasbor juga menyediakan login opsional. Perlindungan data pada penggunaan nyata masih bergantung pada siapa yang bisa membuka riwayat, hak akses akun, dan cara data disimpan.
- **Pendalaman** Pisahkan jalur tanda vital MQTT, endpoint riwayat, dan input identitas dari meja perawat. Jangan menyimpulkan ACL broker dari kode client. Tunjukkan konfigurasi yang disanitasi dan peran pengguna jika tersedia.
- **Bukti** L hlm. 10–11, 24, 27; service/mqtt_client.py:83; backend/access_control.py:10; backend/app.py:179.
- **Tanya jawab lanjutan**
  - **Siapa bisa membuka riwayat pasien?** Kode menyediakan login dasbor opsional yang membungkus akses aplikasi. Siapa yang benar-benar berwenang membuka riwayat bergantung pada konfigurasi deployment dan kebijakan rumah sakit yang belum saya verifikasi.
  - **Apa yang terjadi jika akun bocor?** Akun yang bocor dapat dipakai untuk mengakses apa pun yang diizinkan akun tersebut. Respons yang perlu disiapkan adalah mengganti atau mencabut kredensial dan memeriksa log akses.
- **Batas klaim** Sebut kontrol yang terlihat. Jangan klaim kepatuhan, audit keamanan, atau enkripsi saat tersimpan tanpa bukti.
- **Kata kunci** TLS, login, akses, retensi.

### B03 — Jika dasbor macet atau basis data tidak tersedia, bagaimana membedakan status tampilan dan status bed?

**PARTIALLY VERIFIED** · Desain dan implementasi

- **Jawaban utama** Pada jalur MQTT, backend menyimpan hasil sebelum menandai bed terisi. Pembaruan dasbor berjalan terpisah, sehingga layar yang tertinggal perlu dicek terhadap status yang dikonfirmasi server. Jika basis data gagal menyimpan pada jalur itu, permintaan dijadwalkan ulang dan bed belum terisi oleh permintaan tersebut.
- **Pendalaman** TriageService dapat meminta penyimpanan database sebelum meneruskan alokasi MQTT. Kegagalan broadcast dipisah agar simpan ulang tidak terjadi hanya karena UI gagal. Repository mendukung MySQL dan opsi SQLite untuk lingkungan lokal.
- **Bukti** L hlm. 11, 24; backend/triage_service.py:25; backend/database.py:16; tests/test_mqtt.py:82, 201.
- **Tanya jawab lanjutan**
  - **Jika proses restart, dari mana status bed dipulihkan?** Database menyimpan riwayat, sementara BedManager mengelola status bed aktif di memori. Setelah restart, petugas perlu mencocokkan status dengan kondisi aktual sebelum mengandalkan alokasi otomatis. Pemulihan dari riwayat saat startup belum terkonfirmasi.
- **Batas klaim** Klaim penanganan yang tampak dalam kode. Jangan menjanjikan pemulihan penuh atau sinkronisasi lintas server.
- **Kata kunci** database, konfirmasi, Socket.IO, refresh.

### B04 — Rp15 juta itu biaya produksi atau harga jual? Apakah pembandingnya setara?

**PARTIALLY VERIFIED** · Efektivitas dan adaptabilitas

- **Jawaban utama** Rp15 juta adalah estimasi harga dalam laporan. Kami menggunakannya untuk menunjukkan posisi harga yang diusulkan di antara perangkat pemantau tanda vital. Pembandingnya memiliki fungsi dan tingkat validasi yang berbeda. Biaya produksi perlu dihitung dari komponen dan perakitan.
- **Pendalaman** Tabel 4.5 menampilkan TriaGo dan tiga monitor. Pisahkan biaya perangkat, habis pakai, cloud, sertifikasi, pemeliharaan, pelatihan, dan integrasi. Tidak ada harga pasar terbaru yang diverifikasi dalam persiapan ini.
- **Bukti** L hlm. 25 Tabel 4.5; M hlm. 4 Tabel IV; V 8:10.
- **Tanya jawab lanjutan**
  - **Berapa BOM aktual?** Rp15 juta di laporan adalah estimasi harga. BOM dihitung dari daftar komponen dan biaya per item. Total BOM aktual belum tersedia.
  - **Berapa biaya per pemeriksaan?** Cara menghitungnya adalah membagi total biaya kepemilikan selama masa pakai dengan jumlah pemeriksaan. Masukkan juga pemeliharaan, operasional, dan bahan habis pakai. Nilai per pemeriksaan belum bisa diberikan dari estimasi harga alat saja.
- **Batas klaim** Sebut Rp15 juta sebagai estimasi proposal harga. Jangan klaim harga produksi, penghematan, atau substitusi produk komersial.
- **Kata kunci** estimasi, BOM, pemeliharaan, pembanding.

### B05 — Powerbank 20.000 mAh dan 5,5–6 jam cukup untuk satu shift IGD?

**PARTIALLY VERIFIED** · Efektivitas dan adaptabilitas

- **Jawaban utama** Pada kondisi uji di laporan, powerbank 20.000 mAh bertahan 5,5–6 jam. Kalau shift lebih lama dari itu, kami perlu menyediakan catu daya atau baterai cadangan. Angka yang sama juga bisa berubah jika beban perangkat berbeda dari kondisi uji.
- **Pendalaman** Durasi dipengaruhi tegangan nominal, konversi, kecerahan layar, koneksi, dan siklus inferensi. Laporan mencatat suhu Pi idle 33 °C dan aktif sampai 37 °C, yang bukan pengukuran konsumsi daya. Butuh log Wh dan profil pemakaian.
- **Bukti** L hlm. 12 Tabel 3.1 dan hlm. 25 §4.7.
- **Tanya jawab lanjutan**
  - **Bagaimana ketika daya habis saat akuisisi?** Pengukuran yang terputus tidak bisa dianggap selesai. Setelah perangkat kembali siap, ulangi perekaman atau gunakan prosedur pengukuran cadangan.
  - **Apa mode cadangan?** Cadangan yang kami usulkan adalah catu daya siap pakai dan alat ukur standar dengan alur manual. Pergantian otomatis saat akuisisi belum saya verifikasi.
- **Batas klaim** Klaim durasi terlapor pada kondisi uji. Jangan klaim efisiensi terukur per pasien atau operasi kontinu satu shift.
- **Kata kunci** powerbank, Wh, durasi, cadangan.

### B06 — Bagaimana rumah sakit dapat mengadopsi TriaGo dan bagaimana dampaknya akan diukur?

**UNVERIFIED** · Efektivitas dan adaptabilitas

- **Jawaban utama** Kami mengusulkan pilot di satu IGD dengan tenaga medis tetap memutuskan prioritas. Ukur waktu penilaian, kecocokan kategori, dan kegagalan pengukuran sebelum dan selama pilot. Dari sana rumah sakit bisa menilai apakah TriaGo membantu alur kerja.
- **Pendalaman** Laporan mencantumkan rencana integrasi sistem informasi, uji kegunaan, dan validasi langsung. Ukur baseline dan kondisi pilot dengan definisi sama. Tentukan penanggung jawab pelatihan, sanitasi, pembaruan, dan dukungan teknis sebelum perluasan.
- **Bukti** L hlm. 25–27 §4.6, §5.2; V 6:30 dan 7:00.
- **Tanya jawab lanjutan**
  - **Siapa pengguna, pembeli, dan pemilik data?** Dalam skenario adopsi, penggunanya tenaga medis IGD dan pengadaan dilakukan rumah sakit. Tanggung jawab atas data harus ditetapkan dalam kebijakan rumah sakit dan perjanjian pilot.
  - **Apa indikator SDG yang terukur?** Salah satu indikator yang bisa diukur adalah waktu dari awal penilaian sampai prioritas ditetapkan, dibandingkan sebelum dan selama pilot dengan definisi yang sama.
- **Batas klaim** Klaim rencana dan potensi. Jangan klaim mitra kontraktual, penghematan, atau dampak klinis yang sudah tercapai.
- **Kata kunci** pilot, baseline, adopsi, dukungan.

### B07 — Di mana kebaruan TriaGo dan apa yang sebenarnya tercakup HKI?

**PARTIALLY VERIFIED** · Presentasi dan demonstrasi

- **Jawaban utama** Kontribusi TriaGo ada pada integrasinya. Data dari penjepit multisensor diproses di perangkat untuk estimasi tanda vital dan klasifikasi, lalu hasilnya dipakai dalam alur alokasi bed. Surat pencatatan HKI yang kami periksa mencantumkan jenis ciptaan program komputer.
- **Pendalaman** Perbandingan Tabel 4.5 mendukung beda fungsi yang dipilih, tetapi belum memetakan seluruh produk sejenis. Sertifikat menyebut Sistem Estimasi Tekanan Darah dan Klasifikasi Triase Menggunakan Multisensor. Bukti pencatatan terpisah dari pembuktian kinerja atau keunikan global.
- **Bukti** L hlm. 7, 24–25; Sertifikat HKI hlm. 1, pratinjau Tahap 1.
- **Tanya jawab lanjutan**
  - **Apa yang tim bangun sendiri?** Kontribusi tim yang bisa saya jelaskan adalah merancang cara penjepit, pemrosesan sinyal, klasifikasi, dan alokasi bed bekerja sebagai satu sistem. Untuk rincian tiap komponen, saya bisa menunjukkan desain atau kode dan menjelaskan bagian yang kami integrasikan.
  - **Apakah klaim mencakup perangkat keras?** Surat yang diperiksa mencatat jenis ciptaan program komputer. Saya tidak memperluasnya menjadi klaim perlindungan perangkat keras tanpa dokumen terpisah.
- **Batas klaim** Sebut integrasi dan cakupan program komputer. Jangan mengklaim paten, kebaruan global, atau perlindungan seluruh perangkat.
- **Kata kunci** integrasi, program komputer, pembanding, cakupan.

### B08 — Kontribusi pribadi Anda apa dan bagian mana yang bisa Anda jelaskan jika juri minta kode?

**UNVERIFIED** · Presentasi dan demonstrasi

- **Jawaban utama** Saya mengerjakan [modul atau eksperimen]. Keputusan yang saya ambil adalah [keputusan] karena [alasan]. Dampaknya bisa saya tunjukkan pada [file atau hasil uji]. Bagian itu menerima [input] dan mengirim [output] ke modul yang dikerjakan [nama rekan].
- **Pendalaman** Isi bagian dalam kurung siku dengan kontribusi yang dikonfirmasi sebelum latihan. Pilih satu perubahan yang bisa ditunjukkan langsung di kode, rancangan, atau hasil eksperimen.
- **Bukti** L hlm. 1; P hlm. PDF 80; pembagian kontribusi belum terkonfirmasi.
- **Tanya jawab lanjutan**
  - **Siapa yang menulis firmware?** Firmware dikerjakan oleh [nama yang dikonfirmasi]. Bagian yang bisa kami tunjukkan adalah [file atau modul] yang menangani [fungsi].
  - **Bisa jelaskan satu trade-off dari kontribusi Anda?** Di [modul], saya mempertimbangkan [opsi pertama] dan [opsi kedua]. Saya memilih [opsi] karena [alasan]. Konsekuensinya adalah [trade-off], yang terlihat pada [kode atau hasil uji].
- **Batas klaim** Klaim hanya kontribusi yang dapat dikonfirmasi. Jangan mengatribusikan kepemilikan modul dari penampilan di video.
- **Kata kunci** kontribusi, keputusan, kode, bukti.

### B09 — Kalau video, broker atau dasbor gagal di depan juri, apa yang bisa tetap Anda buktikan?

**PARTIALLY VERIFIED** · Presentasi dan demonstrasi

- **Jawaban utama** Jika video daring gagal, saya bisa menunjukkan rekaman lokal dan bagian yang menampilkan hasil di perangkat. Jika broker atau dasbor gagal, saya bisa menjelaskan alur pengiriman lewat kode. Untuk menunjukkan bed benar-benar dialokasikan, saya akan memakai rekaman atau log transaksi jika tersedia.
- **Pendalaman** Siapkan MP4 lokal, gambar alur, dan contoh log ber-ID sama atau tangkapan layar berlabel rekaman. Siapkan kode sumber untuk permintaan juri. Jangan mengubah data operasional untuk demo.
- **Bukti** P hlm. PDF 80–81; V 3:00, 4:05; L hlm. 24.
- **Tanya jawab lanjutan**
  - **Apa bukti cadangan untuk sinkronisasi bed?** Bukti cadangan yang paling kuat adalah rekaman atau log dari satu transaksi dengan ID yang sama pada pengiriman perangkat dan balasan bed, disertai status di dasbor. Jika itu belum ada, saya akan menunjukkan rekaman yang tersedia dan menyebut batasnya.
  - **Bisa tunjukkan titik kode?** Ya. service/api_client.py menyimpan ID saat enqueue dan mencari balasan dengan ID itu. backend/mqtt_receiver.py memproses data lalu menerbitkan balasan alokasi dengan ID yang sama.
- **Batas klaim** Pisahkan rekaman dan kondisi saat itu. Jangan menyebut demo live berhasil ketika hanya menunjukkan rekaman.
- **Kata kunci** MP4 lokal, log, snapshot, fallback.

### B10 — Mengapa abstrak pengantar memakai angka lebih rendah dari laporan akhir?

**PARTIALLY VERIFIED** · Presentasi dan demonstrasi

- **Jawaban utama** “Mohon maaf, kami keliru mengunggah abstrak versi lama. Untuk angka yang kami laporkan, acuan kami adalah laporan akhir.”
- **Pendalaman** Gunakan satu kalimat pengakuan tersebut. Jika juri menunjuk Gambar 4.10 di dalam laporan akhir, tanggapi sebagai pertanyaan substantif S01.
- **Bukti** Dokumen Judul, Abstract, link youtube hlm. 1; L hlm. 2 dan 22; keterangan peserta tentang salah unggah.
- **Tanya jawab lanjutan**
  - **Kalau begitu, mengapa Gambar 4.10 juga menghasilkan 88,20%?** Dari diagonal Gambar 4.10, ada 14.112 prediksi benar dari 16.000, jadi hasil hitung ulangnya 88,20%. Paragraf dan tabel menulis 90,67%. Hubungan keduanya dengan satu run belum terkonfirmasi.
- **Batas klaim** Akui kelalaian unggah abstrak. Jangan memakai penjelasan itu untuk menutup selisih internal laporan.
- **Kata kunci** abstrak lama, pengakuan, laporan akhir.

## Singkatan rujukan

- **L** — Laporan Akhir TriaGo. Nomor halaman mengikuti halaman PDF.
- **M** — Makalah GEMASTIK.
- **N** — Notebook klasifikasi triase. Nomor sel dihitung dari awal, termasuk sel Markdown.
- **P** — Panduan GEMASTIK 2026.
- **V** — Video final. Waktu menunjuk cuplikan visual atau teks di layar.

Rujukan kode menunjuk versi yang ada di workspace. Kesamaannya dengan versi saat demo belum terkonfirmasi.
