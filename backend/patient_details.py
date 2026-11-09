"""Data registrasi pasien IGD yang dilengkapi petugas dari dashboard."""
from datetime import date
import re

# Field bebas: nama field -> panjang maksimum.
TEXT_FIELDS = {
    "medical_record_no": 30,
    "address": 300,
    "insurance_no": 30,
    "chief_complaint": 500,
    "allergies": 200,
    "relative_relation": 50,
}
CHOICE_FIELDS = {
    "sex": {"L", "P"},
    "arrival_mode": {"sendiri", "diantar", "ambulans", "rujukan"},
    "payer": {"umum", "bpjs", "asuransi"},
}
PHONE_FIELDS = {"phone", "relative_phone"}
LABELS = {
    "medical_record_no": "No. rekam medis", "address": "Alamat", "insurance_no": "No. kartu",
    "chief_complaint": "Keluhan utama", "allergies": "Riwayat alergi",
    "relative_relation": "Hubungan penanggung jawab", "sex": "Jenis kelamin",
    "arrival_mode": "Cara datang", "payer": "Penjamin", "phone": "No. HP pasien",
    "relative_phone": "No. HP penanggung jawab", "nik": "NIK", "birth_date": "Tanggal lahir",
}
ALL_FIELDS = set(TEXT_FIELDS) | set(CHOICE_FIELDS) | PHONE_FIELDS | {"nik", "birth_date"}

_PHONE = re.compile(r"^\+?[0-9 -]{6,20}$")


def validate_patient_details(details):
    """Kembalikan dict bersih tanpa field kosong; ValueError bila isinya tidak valid."""
    if details is None:
        return {}
    if not isinstance(details, dict):
        raise ValueError("details harus berupa object JSON")
    unknown = set(details) - ALL_FIELDS
    if unknown:
        raise ValueError(f"Field tidak dikenal: {', '.join(sorted(unknown))}")

    clean = {}
    for key, value in details.items():
        if value is None:
            continue
        label = LABELS[key]
        if not isinstance(value, str):
            raise ValueError(f"{label} harus berupa teks")
        value = value.strip()
        if not value:
            continue
        if key in TEXT_FIELDS and len(value) > TEXT_FIELDS[key]:
            raise ValueError(f"{label} maksimal {TEXT_FIELDS[key]} karakter")
        if key in CHOICE_FIELDS and value not in CHOICE_FIELDS[key]:
            raise ValueError(f"Pilihan {label} tidak valid")
        if key in PHONE_FIELDS and not _PHONE.match(value):
            raise ValueError(f"{label} tidak valid")
        if key == "nik" and not re.fullmatch(r"[0-9]{16}", value):
            raise ValueError("NIK harus 16 digit angka")
        if key == "birth_date":
            try:
                born = date.fromisoformat(value)
            except ValueError:
                raise ValueError("Tanggal lahir harus berformat YYYY-MM-DD") from None
            if born > date.today() or born.year < 1900:
                raise ValueError("Tanggal lahir tidak masuk akal")
        clean[key] = value
    return clean
