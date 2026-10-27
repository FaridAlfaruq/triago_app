-- For existing MySQL/MariaDB patient_data tables, run once before using the
-- extended patient registration form (No. RM, NIK, penjamin, keluhan, dll.).
-- Select TRIAGO_DB_NAME in phpMyAdmin/the SQL client first.
-- Stored as JSON text; existing rows remain NULL.
ALTER TABLE patient_data ADD COLUMN patient_details TEXT NULL;
