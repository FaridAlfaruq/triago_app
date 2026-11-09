# Celah pengetahuan berisiko tinggi TriaGo

**VERIFIED** berarti fakta spesifik terlihat langsung pada sumber atau dapat dihitung dari data yang ditampilkan. **PARTIALLY VERIFIED** berarti sebagian jawaban didukung, tetapi kesimpulan lengkap masih memerlukan bukti. **UNVERIFIED** berarti bukti belum cukup. **NOT TESTED** hanya dipakai bila ada konfirmasi uji memang belum dilakukan. Bukti kode mendukung klaim tentang isi kode. Hasil benchmark dan angka laporan berlaku pada konteks evaluasinya.

| Prioritas | Pertanyaan yang belum bisa dijawab tuntas | Bukti yang diperlukan | Risiko di depan juri | Respons sementara |
|---|---|---|---|---|
| 1 · S01 · PARTIALLY VERIFIED | Run mana yang menghasilkan angka Tabel 4.2, Gambar 4.10, dan Tabel 4.3? | Target, prediksi, dataset, split dan checkpoint tiap hasil. | Grafik menghasilkan 88,20% dan macro F1 89,83%. Notebook menampilkan 90,69% dan macro F1 91,42% pada hasil tersimpan. | “Tabel dan gambar memuat hasil berbeda. Saya dapat menjelaskan masing-masing. Run sumbernya masih perlu kami telusuri.” |
| 2 · S02/S08 · PARTIALLY VERIFIED | Seberapa baik model pada pasien IGD nyata? | Kohort lokal yang terpisah, penilaian tenaga medis, definisi under-triage dan hasil per kelas. | Angka sintetis dapat dibaca keliru sebagai kemampuan klinis. | “Angka berasal dari benchmark Triagegeist. Pemetaan kelas dan kinerja pada pasien nyata memerlukan validasi.” |
| 3 · S03/S05 · PARTIALLY VERIFIED | Berapa subjek unik di test BP, dan apakah evaluasi bebas duplikat lintas pasien? | ID kelompok, cek irisan dan duplikat, prediksi vs manset, rentang BP, dan protokol alat referensi. | Galat dapat terlalu optimistis. Grade terlapor tidak membuktikan protokol lengkap. | “Notebook memakai split kelompok pasien. Manifest run penghasil MAE laporan masih perlu kami tunjukkan.” |
| 4 · S04 · VERIFIED pada kode saat ini | Bagaimana fallback ditampilkan dan digunakan di build demo? | Snapshot demo dan jejak satu kasus gagal sampai hasil dan payload. | Nilai seperti 120/80 dapat dianggap hasil pengukuran atau menurunkan urgensi. | “Kode saat ini menandai fallback dan masih memasukkannya ke fitur. Untuk keputusan klinis kami perlu mengukur ulang dan meminta penilaian petugas.” |
| 5 · S10/A03 · UNVERIFIED untuk versi demo | Artefak BP dan ONNX mana yang dipakai di demo? | Hash, manifest, snapshot, praproses, dataset sumber, dan prediksi. | Angka benchmark mungkin melekat pada artefak berbeda. Fitur bernama pulse pressure dihitung sebagai rasio pada kode. | “Saya bisa menunjukkan jalur kode saat ini. Hubungannya dengan artefak evaluasi dan demo masih perlu dibuktikan.” |
| 6 · S06/S07/B03 · PARTIALLY VERIFIED | Apa yang terjadi ketika server restart, broker gagal lama atau dua worker mengalokasikan satu bed? | Konfigurasi worker, jejak reconnect, transaksi, uji perebutan dan pemulihan. | PUBACK dan lock lokal dapat disalahpahami sebagai jaminan sistem menyeluruh. | “Antrean dan lock lokal terlihat pada kode. Ketahanan lintas proses dan restart perlu bukti terpisah.” |
| 7 · S09 · UNVERIFIED | Apa definisi dan sebaran waktu penggunaan ≤2 menit? | Log registrasi, pemasangan, akuisisi, inferensi dan konfirmasi bed per pengulangan. | Video edit dan latensi model tidak membuktikan waktu seluruh alur. | “Kode mengatur 60 detik rekam setelah pemanasan. Definisi waktu total dan konfirmasi bed masih perlu log.” |
| 8 · A05/A06/A09 · PARTIALLY VERIFIED | Apakah kinerja sensor bertahan saat gerakan, perfusi rendah, desaturasi atau demam? | Data berpasangan, alat pembanding, asal input suhu, rentang fisiologis, dan kondisi akuisisi. | Hasil dataset sekunder tidak menunjukkan kinerja perangkat pada seluruh kondisi IGD. | “Saya dapat menyebut MAE pada dataset yang diuji. Kondisi ekstrem membutuhkan evaluasi terpisah.” |
| 9 · B02/B04–B06 · UNVERIFIED | Bagaimana keamanan deployment, biaya riil dan pemeliharaannya? | Matriks peran dan ACL, BOM, profil energi, total biaya, dan rencana pilot. | TLS, login, dan harga estimasi hanya menjawab sebagian. | “Ada kontrol dasar dan estimasi biaya. Rincian deployment dan biaya kepemilikan masih perlu dibuktikan.” |
| 10 · B08 · UNVERIFIED | Apa kontribusi setiap anggota? | Konfirmasi anggota, modul/eksperimen, dan artefak kontribusi. | Jawaban berbeda antaranggota melemahkan kepercayaan juri. | “Saya akan menjelaskan kontribusi yang benar-benar saya kerjakan dan menunjukkan buktinya.” |

## Bukti yang perlu diingat

1. **L hlm. 22** — Gambar 4.10, Tabel 4.2 dan 4.3. Dari gambar, 596 Resusitasi benar dan 48 diprediksi Darurat. Tabel menulis akurasi 90,67% dan F1 91,42%.
2. **L hlm. 21** — PulseDB → transfer learning → data primer 94 subjek yang dilaporkan → pembanding manset → MAE 4,90/2,30 mmHg. Jumlah test terpisah belum terkonfirmasi.
3. **L hlm. 7 dan 24** — sensor → STM32 → Raspberry Pi → HiveMQ Cloud → Flask/MySQL → dasbor → balasan bed. Pahami perbedaan hasil lokal, ACK broker, dan konfirmasi alokasi.
4. **GUI/loading_page.py dan model/deployment_inference.py** — bedakan masukan terukur, fallback, dan aturan eskalasi.
5. **L hlm. 23, 25–27 dan V sekitar 7:00** — uji subjek sehat, masukan dokter, estimasi harga, dan rencana validasi langsung.

## Cakupan pemeriksaan

- Laporan akhir diperiksa terarah pada abstrak dan bab teknis hlm. 7–27. Visual yang berhasil ditinjau mencakup arsitektur, persamaan, tabel komponen, desain rangkaian, diagram model, antarmuka, dan Gambar 4.10 yang diperbesar untuk menghitung ulang matriks. Sebagian montase visual gagal ditampilkan. Bagian latar, referensi dan halaman penutup tidak ditelaah untuk bank jawaban.
- Makalah hlm. 1–5 diperiksa sebagai teks dan cuplikan visual. Pedoman menggunakan ketentuan umum dan Divisi IX yang telah dibaca pada Tahap 1, termasuk visual PDF hlm. 80–81.
- MP4 ditinjau melalui cuplikan visual dan teks di layar sekitar 0:30–8:35. Audio, transkrip penuh, dan kesamaan dengan unggahan YouTube belum diverifikasi.
- Pemeriksaan kode terarah. Eksperimen dan tes proyek belum dijalankan. Angka tambahan berasal dari keluaran tersimpan dan hitung ulang confusion matrix. Bukti Paperpal belum diperiksa.
- Penjelasan sumber Triagegeist berasal dari deskripsi resmi penyelenggara yang terbaca di hasil penelusuran. Pembukaan langsung URL tidak menampilkan teks. Notebook menunjukkan pemanggilan dataset tersebut. Provenans checkpoint demo masih perlu diverifikasi. [Halaman data resmi Triagegeist](https://www.kaggle.com/competitions/triagegeist/data)

## Sumber

- **L** — [Laporan Akhir TriaGo, salinan lokal](<C:/Users/Adyty/Downloads/Gemastik para pencari berkah.pdf>). Judul, 30 halaman, dan angka abstrak cocok dengan pratinjau Drive. Identitas byte belum dibandingkan.
- **M** — [Makalah Gemastik, salinan lokal](<C:/Users/Adyty/Downloads/Makalah Gemastik.pdf>).
- **P** — [Panduan GEMASTIK 2026](<C:/Users/Adyty/Documents/Farid ITS/Tugas Akhir/scratch/gemastik/Panduan GEMASTIK 2026_Final.pdf>), PDF hlm. 80–81. Final menyediakan 10 menit presentasi/demo video dan 15 menit tanya jawab, dengan lima kriteria berbobot 20%. Kode perlu siap ditampilkan.
- **V** — [MP4 final, salinan lokal](<C:/Users/Adyty/Downloads/Video FInal diperbaiki.mp4>). Timestamp menunjuk cuplikan visual dan teks layar.
- **N** — [Notebook triase](<C:/Users/Adyty/Documents/Farid ITS/TriaGo/model/Triage_XGBoost (1).ipynb>). Nomor sel mulai 1, termasuk sel Markdown.

Nomor halaman mengikuti indeks PDF. Jalur kode merujuk workspace TriaGo saat ini. Kesamaan snapshot tersebut dengan versi demo belum terkonfirmasi.
