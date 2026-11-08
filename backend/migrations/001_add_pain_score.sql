-- For existing MySQL/MariaDB patient_data tables, run once before using pain score.
-- Select TRIAGO_DB_NAME in phpMyAdmin/the SQL client first.
-- Existing rows remain NULL: their pain score was not recorded.
ALTER TABLE patient_data ADD COLUMN pain_score TINYINT UNSIGNED NULL;
