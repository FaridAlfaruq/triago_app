"""Tampilkan halaman hasil GUI tanpa sensor, memakai rekaman pengukuran tersimpan.

Contoh (dari folder repo):
    python scripts/demo_hasil.py                       # rekaman terakhir, layar penuh
    python scripts/demo_hasil.py --bed 05              # nomor bed yang ditampilkan
    python scripts/demo_hasil.py --file data_pengukuran/20260803_165308_Bed03.json
    python scripts/demo_hasil.py --kirim               # kirim sungguhan ke backend (.env)

Tanpa --kirim tidak ada data yang dikirim ke backend/dashboard; alokasi bed
disimulasikan. Esc atau Q untuk keluar.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bed pertama tiap zona pada denah default 3/6/3.
DEFAULT_BED = {"red": "01", "yellow": "04", "green": "10"}


def latest_recording():
    folder = PROJECT_ROOT / "data_pengukuran"
    files = sorted((f for f in folder.glob("*.json") if f.with_suffix(".csv").exists()),
                   key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def load_results(json_path):
    """Bentuk dictionary hasil seperti keluaran ProcessingWorker dari file .json + .csv."""
    meta = json.loads(Path(json_path).read_text(encoding="utf-8")) if json_path else {}
    results = {
        "gcs": meta.get("GCS Score", 15),
        "pain_score": meta.get("Pain Score"),
        "hr": float(meta.get("HR", 80.0)),
        "rr": float(meta.get("RR", 18.0)),
        "spo2": float(meta.get("SpO2", 97.0)),
        "systolic": float(meta.get("SBP", 120)),
        "diastolic": float(meta.get("DBP", 80)),
        "temperature": float(meta.get("Suhu Core", 36.8)),
        "temp_skin": float(meta.get("Suhu Skin", 34.0)),
        "triage_status": str(meta.get("Triage Status", "DARURAT")),
        "xgboost_score": float(meta.get("Triage Confidence", 0.9)),
        "triage_valid": True,
        "rr_measured": True,
    }
    csv_path = Path(json_path).with_suffix(".csv") if json_path else None
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path).dropna(subset=["Resample Time (s)", "ECG_Clean", "PPG_IR_Clean"])
        results["time_125"] = df["Resample Time (s)"].to_numpy()
        results["ecg_smooth"] = df["ECG_Clean"].to_numpy()
        results["ir_clean"] = df["PPG_IR_Clean"].to_numpy()
    else:
        # Sinyal sintetis bila rekaman tidak ada, agar grafik tetap terisi.
        t = np.arange(0, 10, 1 / 125)
        results["time_125"] = t
        results["ecg_smooth"] = np.sin(2 * np.pi * 1.3 * t) ** 15
        results["ir_clean"] = np.sin(2 * np.pi * 1.3 * t)
    return results


class SimulatedClient:
    """Pengganti TriageApiClient: tidak mengirim apa pun, langsung 'membalas' bed."""

    def __init__(self, bed):
        self.bed = bed
        self.assignment = None

    def send_triage_result(self, classification, **kwargs):
        category = classification.lower()
        bed = self.bed or DEFAULT_BED.get(category)
        self.assignment = {"status": "success", "assigned_bed": bed, "zone": category}
        return True

    def get_assignment(self):
        return self.assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", help="file .json rekaman (default: terbaru di data_pengukuran)")
    parser.add_argument("--bed", help="nomor bed yang ditampilkan saat simulasi, mis. 05")
    parser.add_argument("--kirim", action="store_true", help="kirim sungguhan ke backend sesuai .env")
    parser.add_argument("--jendela", action="store_true", help="jendela 1280x800, bukan layar penuh")
    args = parser.parse_args()

    json_path = Path(args.file) if args.file else latest_recording()
    print(f"[DEMO] Rekaman: {json_path or 'tidak ada, memakai data sintetis'}")

    app = QApplication(sys.argv)
    import GUI.output_page as output_page
    if not args.kirim:
        output_page.TriageApiClient = lambda: SimulatedClient(args.bed)
    page = output_page.OutputPage()
    page.setWindowTitle("TriaGO - Demo Halaman Hasil")
    page.update_results(load_results(json_path))
    for key in ("Esc", "Q"):
        QShortcut(QKeySequence(key), page, activated=app.quit)
    page.home_requested.connect(app.quit)
    if args.jendela:
        page.resize(1280, 800)
        page.show()
    else:
        page.setWindowState(Qt.WindowState.WindowFullScreen)
        page.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
