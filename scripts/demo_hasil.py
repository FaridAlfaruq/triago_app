"""Tampilkan halaman hasil GUI tanpa sensor, memakai rekaman pengukuran tersimpan.

Contoh (dari folder repo):
    python scripts/demo_hasil.py                       # rekaman terakhir, layar penuh
    python scripts/demo_hasil.py --bed 05              # nomor bed yang ditampilkan
    python scripts/demo_hasil.py --file data_pengukuran/20260803_165308_Bed03.json
    python scripts/demo_hasil.py --kirim               # kirim sungguhan ke backend (.env)
    python scripts/demo_hasil.py --sinyal rekaman      # grafik dari rekaman, bukan contoh

Grafik default memakai sinyal CONTOH (sintetis, bersih) dengan HR/RR rekaman;
jangan disajikan sebagai hasil sensor. Tanpa --kirim tidak ada data yang dikirim
ke backend/dashboard; alokasi bed disimulasikan. Esc atau Q untuk keluar.
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


def _ppg_pulse(x):
    """Satu denyut PPG terhadap waktu sejak onset (detik): naik curam, puncak
    membulat, lalu bahu dicrotic yang menempel di lereng turun."""
    peak = 0.15
    # Gaussian asimetris: naik cepat sebelum puncak, turun sedikit lebih lambat.
    width = np.where(x < peak, 0.055, 0.075)
    systolic = np.exp(-((x - peak) ** 2) / (2 * width ** 2))
    shoulder = 0.33 * np.exp(-((x - (peak + 0.16)) ** 2) / (2 * 0.045 ** 2))  # bahu dicrotic
    return systolic + shoulder


def example_signals(hr, rr, seconds=10.0, fs=125, seed=7):
    """ECG (P-QRS-T) dan PPG sintetis yang bersih, skala mirip satuan ADC perangkat."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fs)
    hr = float(np.clip(hr, 40, 160))
    rr = float(np.clip(rr, 8, 40))
    ecg = np.zeros_like(t)
    ppg = np.zeros_like(t)
    beat = 0.3
    while beat < seconds + 1:
        # Variasi detak kecil (HRV) agar tidak terlihat seperti sinus sempurna.
        period = 60.0 / hr * (1 + 0.03 * np.sin(2 * np.pi * rr / 60 * beat) + rng.normal(0, 0.01))
        for amp, offset, width in ((0.14, -0.19, 0.025), (-0.12, -0.035, 0.010), (1.30, 0.0, 0.013),
                                   (-0.75, 0.035, 0.014), (0.70, 0.24, 0.045)):
            ecg += amp * np.exp(-((t - beat - offset) ** 2) / (2 * width ** 2))
        # Onset PPG ~0.18 s setelah puncak R (waktu tempuh nadi ke jari).
        ppg += (1 + rng.normal(0, 0.03)) * _ppg_pulse(t - beat - 0.18)
        beat += period
    breathing = np.sin(2 * np.pi * rr / 60 * t)
    ecg = 650 * (ecg + 0.04 * breathing) + rng.normal(0, 3, t.size)
    ppg = 55 * ppg * (1 + 0.05 * breathing) + rng.normal(0, 0.08, t.size)
    return t, ecg - np.median(ecg), ppg - ppg.mean()


def load_results(json_path, signal="contoh", pain=None):
    """Bentuk dictionary hasil seperti keluaran ProcessingWorker dari file .json + .csv."""
    meta = json.loads(Path(json_path).read_text(encoding="utf-8")) if json_path else {}
    results = {
        "gcs": meta.get("GCS Score", 15),
        "pain_score": pain if pain is not None else meta.get("Pain Score"),
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
    if signal == "rekaman" and csv_path and csv_path.exists():
        df = pd.read_csv(csv_path).dropna(subset=["Resample Time (s)", "ECG_Clean", "PPG_IR_Clean"])
        results["time_125"] = df["Resample Time (s)"].to_numpy()
        results["ecg_smooth"] = df["ECG_Clean"].to_numpy()
        results["ir_clean"] = df["PPG_IR_Clean"].to_numpy()
    else:
        t, ecg, ppg = example_signals(results["hr"], results["rr"])
        results["time_125"], results["ecg_smooth"], results["ir_clean"] = t, ecg, ppg
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
    parser.add_argument("--sinyal", choices=("contoh", "rekaman"), default="contoh",
                        help="grafik: contoh sintetis yang bersih (default) atau sinyal rekaman")
    parser.add_argument("--nyeri", type=int, choices=range(0, 11), metavar="0-10",
                        help="pain score yang ditampilkan (rekaman lama belum punya)")
    args = parser.parse_args()

    json_path = Path(args.file) if args.file else latest_recording()
    print(f"[DEMO] Rekaman: {json_path or 'tidak ada, memakai nilai default'}")
    if args.sinyal == "contoh":
        print("[DEMO] Grafik memakai sinyal CONTOH (sintetis), bukan hasil sensor.")

    app = QApplication(sys.argv)
    import GUI.output_page as output_page
    if not args.kirim:
        output_page.TriageApiClient = lambda: SimulatedClient(args.bed)
    page = output_page.OutputPage()
    page.setWindowTitle("TriaGO - Demo Halaman Hasil")
    page.update_results(load_results(json_path, args.sinyal, args.nyeri))
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
