"""Putar ulang rekaman sensor tersimpan lewat GUI TriaGO lengkap, untuk screen record demo.

Seluruh alur GUI berjalan seperti biasa (beranda -> registrasi -> perekaman 60 s ->
pemrosesan -> hasil). Bedanya hanya sumber data: STM32 diganti pemutar ulang yang
mengirim sampel mentah (ECG, PPG Red/IR, suhu) dari CSV di data_pengukuran pada
400 Hz waktu nyata. Filter live, pemrosesan, BPNet, dan model triase memakai kode
asli, jadi angka di layar = hasil olahan rekaman itu.

Contoh (dari folder repo):
    python scripts/demo_rekaman.py --peringkat          # nilai semua rekaman, tampilkan 10 terbaik
    python scripts/demo_rekaman.py                      # putar rekaman terbaik (dinilai dulu)
    python scripts/demo_rekaman.py --file data_pengukuran/20261017_112458_Bed01.csv
    python scripts/demo_rekaman.py --file ... --jendela # jendela 1280x800, bukan layar penuh
    python scripts/demo_rekaman.py --file ... --kirim   # kirim hasil sungguhan ke backend (.env)

Tanpa --kirim tidak ada yang dikirim ke backend/dashboard (alokasi bed disimulasikan)
dan tidak ada file baru yang ditulis ke data_pengukuran. Ctrl+Q untuk keluar.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DATA_DIR = PROJECT_ROOT / "data_pengukuran"
FS = 400
RAW_COLUMNS = ["PPG_Red", "PPG_IR", "ECG_Raw", "Suhu Obj", "Suhu Amb"]


# =====================================================================
# Membaca rekaman
# =====================================================================
def load_raw(csv_path):
    """Kolom mentah 400 Hz dari CSV pengukuran (tanpa baris kosong di ujung)."""
    df = pd.read_csv(csv_path, usecols=RAW_COLUMNS).dropna(subset=["PPG_Red", "PPG_IR", "ECG_Raw"])
    return df.reset_index(drop=True)


def recording_files():
    return sorted(f for f in DATA_DIR.glob("2*.csv"))


def recorded_meta(csv_path):
    meta_path = Path(csv_path).with_suffix(".json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {}


# =====================================================================
# Penilaian rekaman (headless, memakai pipeline pemrosesan asli)
# =====================================================================
_MODELS = {}


def _models():
    if not _MODELS:
        from model.deployment_inference import TriageOnnxModel
        from model.bpnet_inference import RevisedBPPredictor
        try:
            _MODELS["triage"] = TriageOnnxModel()
        except Exception as exc:  # noqa: BLE001 - laporkan, tetap nilai sinyal
            print(f"[PERINGKAT] Model triase gagal dimuat: {exc}")
            _MODELS["triage"] = None
        try:
            _MODELS["bp"] = RevisedBPPredictor()
        except Exception as exc:  # noqa: BLE001
            print(f"[PERINGKAT] Model BP gagal dimuat: {exc}")
            _MODELS["bp"] = None
    return _MODELS


def process_offline(df, gcs, pain):
    """Jalankan ProcessingWorker.run() secara sinkron, kembalikan dictionary hasil."""
    from GUI.loading_page import ProcessingWorker
    from model.temperature_estimation import temperature_samples

    models = _models()
    patient = {
        "gcs": gcs,
        "pain_score": pain,
        "raw_temp_obj": temperature_samples(df["Suhu Obj"], -70, 380),
        "raw_temp_amb": temperature_samples(df["Suhu Amb"], -40, 125),
    }
    worker = ProcessingWorker(
        raw_ecg=df["ECG_Raw"].to_numpy(dtype=float),
        raw_time=np.arange(len(df)) / FS,
        raw_red=df["PPG_Red"].to_numpy(dtype=float),
        raw_ir=df["PPG_IR"].to_numpy(dtype=float),
        patient_info=patient,
        fs_orig=FS,
        triage_predictor=models["triage"],
        bpnet_predictor=models["bp"],
    )
    worker.msleep = lambda _ms: None  # jeda animasi tidak perlu saat menilai
    out = {}
    worker.processing_finished.connect(out.update)
    worker.run()
    return out


def score_recording(df, res):
    """Skor 'enak didemokan': sinyal bersih, tanpa fallback, angka vital wajar."""
    notes = []
    score = 0.0

    warnings = res.get("triage_input_warnings") or []
    score -= 3.0 * len(warnings)
    if warnings:
        notes.append("fallback: " + ", ".join(w.split()[0] for w in warnings))

    total = res.get("sqa_total_segments") or 0
    sqa_ratio = (res.get("sqa_passed_segments") or 0) / total if total else 0.0
    score += 4.0 * sqa_ratio
    # Layar hasil menulis "Fallback 120/80 mmHg (BP tidak lolos SQA)" jika BP gagal.
    bp_measured = not res.get("bp_fallback_used", True)
    if bp_measured:
        score += 3.0

    score += 2.0 * float(res.get("rr_quality") or 0.0)

    hr, ppg_hr = float(res.get("hr") or 0), float(res.get("ppg_hr") or 0)
    if hr > 0 and ppg_hr > 0:
        diff = abs(hr - ppg_hr)
        score += 3.0 * max(0.0, 1.0 - diff / 10.0)  # ECG & PPG sepakat = dua sinyal bersih
        if diff > 10:
            notes.append(f"HR ECG/PPG beda {diff:.0f}")
    else:
        notes.append("HR PPG kosong")

    def plausible(name, value, low, high):
        nonlocal score
        if value is not None and low <= value <= high:
            score += 2.5  # angka yang tampil di layar hasil paling menentukan kesan demo
        else:
            score -= 2.5
            notes.append(f"{name} {value if value is None else round(value, 1)} di luar {low}-{high}")

    plausible("HR", hr, 50, 120)
    plausible("RR", res.get("rr"), 10, 30)
    plausible("SpO2", res.get("spo2"), 94, 100)
    plausible("Suhu", res.get("temperature"), 35.5, 38.0)
    if bp_measured:
        plausible("SBP", res.get("systolic"), 90, 160)

    # Kliping ADC / jari lepas membuat grafik live jelek walau angka akhirnya bagus.
    ecg = df["ECG_Raw"].to_numpy()
    ir = df["PPG_IR"].to_numpy()
    clip = float(np.mean((ecg <= 0) | (ecg >= 4095)))
    off = float(np.mean(ir < 0.5 * np.median(ir))) + float(np.mean(ir >= 262143))
    score -= 10.0 * clip + 10.0 * off
    if clip > 0.02:
        notes.append(f"ECG kliping {clip:.0%}")
    if off > 0.02:
        notes.append(f"PPG lepas/jenuh {off:.0%}")
    return score, sqa_ratio, notes


def rank_recordings(gcs, pain, top=10):
    rows = []
    files = [f for f in recording_files() if f.stat().st_size > 100_000]
    print(f"[PERINGKAT] Menilai {len(files)} rekaman (GCS {gcs}, nyeri {pain})...")
    for i, path in enumerate(files, 1):
        try:
            df = load_raw(path)
            if len(df) < 20 * FS:
                continue
            res = process_offline(df, gcs, pain)
            score, sqa_ratio, notes = score_recording(df, res)
        except Exception as exc:  # noqa: BLE001 - satu file rusak tidak menghentikan penilaian
            print(f"  [{i}/{len(files)}] {path.name}: gagal ({exc})")
            continue
        rows.append({
            "file": path, "score": score, "sqa": sqa_ratio, "notes": notes,
            "hr": res.get("hr"), "rr": res.get("rr"), "spo2": res.get("spo2"),
            "sbp": res.get("systolic"), "dbp": res.get("diastolic"),
            "temp": res.get("temperature"), "triage": res.get("triage_status"),
        })
        print(f"  [{i}/{len(files)}] {path.name}: skor {score:5.1f}")

    rows.sort(key=lambda r: r["score"], reverse=True)
    print("\nPeringkat rekaman (terbaik di atas):")
    print(f"{'#':>2}  {'file':<28}{'skor':>6}{'SQA':>6}{'HR':>6}{'RR':>6}{'SpO2':>6}"
          f"{'BP':>9}{'Suhu':>6}  triase      catatan")
    for n, r in enumerate(rows[:top], 1):
        bp = f"{r['sbp']:.0f}/{r['dbp']:.0f}" if r["sbp"] is not None else "-"
        temp = f"{r['temp']:.1f}" if r["temp"] is not None else "-"
        print(f"{n:>2}  {r['file'].name:<28}{r['score']:6.1f}{r['sqa']:6.0%}{r['hr']:6.0f}{r['rr']:6.0f}"
              f"{r['spo2']:6.0f}{bp:>9}{temp:>6}  {str(r['triage']):<11} {'; '.join(r['notes'])}")
    return rows


# =====================================================================
# Pengganti STM32Worker: memutar ulang CSV pada 400 Hz
# =====================================================================
class ReplayWorker(QThread):
    """Mengirim paket berformat sama dengan stream_stm32_data, dari rekaman CSV.

    PlotPage membuang 2 s pertama sebagai warmup, jadi 2 s awal rekaman dikirim dua
    kali: sekali untuk warmup, lalu seluruh rekaman utuh. Data yang diolah GUI pun
    sama persis dengan rekaman aslinya.
    """

    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    frame = None  # diisi main() sebelum GUI berjalan
    speed = 1.0
    CHUNK = 8  # paket per tidur, cukup halus untuk grafik dan ringan untuk Raspi

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        df = ReplayWorker.frame
        warmup = int(2.0 * FS)
        order = np.concatenate([np.arange(warmup), np.arange(len(df))])
        red, ir, ecg = (df[c].to_numpy() for c in ("PPG_Red", "PPG_IR", "ECG_Raw"))
        obj, amb = df["Suhu Obj"].to_numpy(), df["Suhu Amb"].to_numpy()
        period = 1.0 / (FS * ReplayWorker.speed)
        next_t = time.perf_counter()
        for n, i in enumerate(order):
            if not self.running or self.isInterruptionRequested():
                return
            self.data_received.emit({
                "status": "OK", "timestamp": time.time(),
                "ppg": {"red": int(red[i]), "ir": int(ir[i]), "green": 0},
                "ecg": int(ecg[i]),
                "temperature": {"ambient": float(amb[i]), "object": float(obj[i])},
            })
            next_t += period
            if n % self.CHUNK == 0:
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

    def stop(self):
        self.running = False
        self.requestInterruption()
        self.wait(2000)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", help="CSV rekaman yang diputar (default: rekaman terbaik hasil penilaian)")
    parser.add_argument("--peringkat", action="store_true", help="hanya nilai semua rekaman lalu keluar")
    parser.add_argument("--top", type=int, default=10, help="jumlah baris peringkat yang dicetak")
    parser.add_argument("--gcs", type=int, default=15, choices=range(3, 16), metavar="3-15",
                        help="GCS yang dipakai saat menilai rekaman (default 15)")
    parser.add_argument("--nyeri", type=int, default=0, choices=range(0, 11), metavar="0-10",
                        help="skor nyeri yang dipakai saat menilai rekaman (default 0)")
    parser.add_argument("--kirim", action="store_true", help="kirim hasil sungguhan ke backend sesuai .env")
    parser.add_argument("--simpan", action="store_true",
                        help="simpan registrasi & hasil ke data_pengukuran seperti aplikasi asli")
    parser.add_argument("--jendela", action="store_true", help="jendela 1280x800, bukan layar penuh")
    parser.add_argument("--cepat", type=float, default=1.0,
                        help="kelipatan kecepatan pemutaran, mis. 4 untuk uji coba (default 1 = waktu nyata)")
    parser.add_argument("--otomatis", action="store_true",
                        help="tekan Mulai, GCS, nyeri, dan Mulai otomatis (tanpa sentuh layar)")
    parser.add_argument("--keluar-setelah", type=float, metavar="DETIK",
                        help="dengan --otomatis: tutup aplikasi sekian detik setelah halaman hasil muncul")
    args = parser.parse_args()

    if args.peringkat:
        rank_recordings(args.gcs, args.nyeri, top=args.top)
        return

    if args.file:
        csv_path = Path(args.file)
        if not csv_path.is_absolute() and not csv_path.exists():
            csv_path = PROJECT_ROOT / csv_path
    else:
        rows = rank_recordings(args.gcs, args.nyeri, top=5)
        if not rows:
            sys.exit("[DEMO] Tidak ada rekaman yang bisa diputar di data_pengukuran.")
        csv_path = rows[0]["file"]
        print(f"\n[DEMO] Lain kali langsung: python scripts/demo_rekaman.py "
              f"--file data_pengukuran/{csv_path.name}")

    df = load_raw(csv_path)
    meta = recorded_meta(csv_path)
    print(f"[DEMO] Memutar ulang {csv_path.name}: {len(df) / FS:.0f} s data sensor tersimpan.")
    if meta.get("GCS Score") is not None:
        print(f"[DEMO] Saat direkam dulu GCS = {meta['GCS Score']}"
              + (f", nyeri = {meta['Pain Score']}" if meta.get("Pain Score") is not None else ""))
    print(f"[DEMO] Pilih GCS {args.gcs} dan nyeri {args.nyeri} di layar registrasi "
          "agar hasil sama dengan penilaian.")

    ReplayWorker.frame = df
    ReplayWorker.speed = max(0.1, args.cepat)

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeySequence, QShortcut
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    import GUI.plot_page as plot_page
    import GUI.output_page as output_page
    import GUI.main_gui as main_gui
    plot_page.STM32Worker = ReplayWorker
    if not args.kirim:
        from demo_hasil import SimulatedClient
        output_page.TriageApiClient = lambda: SimulatedClient(None)
        print("[DEMO] Mode simulasi: hasil TIDAK dikirim ke backend, bed disimulasikan.")

    window = main_gui.TriaGoApplication()
    if not args.simpan:
        window.save_consolidated_csv = lambda _results: None
        registration = window.page_registration

        def emit_without_saving():
            registration.measurement_started.emit({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gcs": registration.selected_gcs,
                "pain_score": registration.selected_pain if registration.selected_pain is not None else 0,
            })

        registration.btn_start.clicked.disconnect()
        registration.btn_start.clicked.connect(emit_without_saving)

    if args.otomatis:
        from PyQt6.QtCore import QTimer
        reg = window.page_registration
        # Jeda dibuat terlihat di rekaman layar, seperti operator yang menekan tombol.
        steps = [
            (2500, window.page_home.btn_start.click),
            (2000, reg.gcs_buttons[args.gcs].click),
            (1500, reg.pain_buttons[args.nyeri].click),
            (1500, reg.btn_start.click),
        ]
        delay = 0
        for wait_ms, action in steps:
            delay += wait_ms
            QTimer.singleShot(delay, action)
        if args.keluar_setelah is not None:
            def quit_later(index):
                if index == 4:
                    QTimer.singleShot(int(args.keluar_setelah * 1000), app.quit)

            window.stacked_widget.currentChanged.connect(quit_later)

    QShortcut(QKeySequence("Ctrl+Q"), window, activated=app.quit)
    app.aboutToQuit.connect(window.page_live_data.close_threads)

    if args.jendela:
        window.resize(1280, 800)
        window.show()
    else:
        window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
