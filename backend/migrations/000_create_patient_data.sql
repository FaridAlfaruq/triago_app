-- TriaGO: struktur lengkap tabel patient_data untuk instalasi MySQL BARU.
--
-- Jalankan satu kali sebagai user `triago` (bukan root), setelah admin membuat
-- database dan user-nya:
--   mysql -u triago -p triago < backend/migrations/000_create_patient_data.sql
--
-- File ini sudah mencakup kolom dari 001 (pain_score) dan 002 (patient_details).
-- Untuk instalasi baru JANGAN jalankan 001/002 lagi; keduanya hanya untuk
-- tabel lama yang dibuat sebelum kolom tersebut ada.
-- Aman dijalankan ulang: tabel yang sudah ada tidak diubah.

CREATE TABLE IF NOT EXISTS patient_data (
    id                INT UNSIGNED NOT NULL AUTO_INCREMENT,
    bed_id            VARCHAR(16)  NULL COMMENT 'Nomor bed hasil alokasi (01, 02, ...), atau WAITING bila zona penuh',
    zone              VARCHAR(8)   NULL COMMENT 'Zona bed: red=Resusitasi, yellow=Darurat, green=Non-Darurat; - untuk WAITING',
    status            VARCHAR(10)  NULL COMMENT 'Kategori triase: red, yellow, green',
    patient_name      VARCHAR(100) NULL,
    relative_name     VARCHAR(100) NULL COMMENT 'Nama penanggung jawab',
    gcs_score         TINYINT UNSIGNED NULL COMMENT '3-15',
    pain_score        TINYINT UNSIGNED NULL COMMENT '0-10; NULL berarti tidak diukur',
    xgboost_score     DOUBLE       NULL,
    heart_rate        FLOAT        NULL COMMENT 'BPM',
    spo2              FLOAT        NULL COMMENT '%',
    temperature       FLOAT        NULL COMMENT 'Suhu inti, derajat Celsius',
    respiration_rate  FLOAT        NULL COMMENT 'Napas per menit',
    systolic_bp       SMALLINT     NULL COMMENT 'mmHg',
    diastolic_bp      SMALLINT     NULL COMMENT 'mmHg',
    arrival_timestamp BIGINT       NULL COMMENT 'Waktu masuk, milidetik sejak 1970 (UTC)',
    patient_details   TEXT         NULL COMMENT 'JSON dari form registrasi dashboard (No. RM, NIK, penjamin, dll.)',
    PRIMARY KEY (id),
    -- Dipakai saat dashboard memperbarui identitas pasien terbaru di sebuah bed.
    KEY idx_bed_arrival (bed_id, arrival_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
