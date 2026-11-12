"""Demo halaman perekaman: grafik ECG/PPG selalu bersih selama jari menempel di sensor.

Sensor STM32 tetap dibaca sungguhan, tetapi hanya untuk mendeteksi KONTAK jari
(level DC PPG IR). Selama jari terdeteksi, grafik menampilkan sinyal CONTOH
sintetis yang bersih; begitu jari dilepas, grafik langsung datar. Jangan sajikan
grafik ini sebagai hasil sensor.

Contoh (dari folder repo):
    python scripts/demo_live.py                    # pakai sensor, layar penuh
    python scripts/demo_live.py --jendela          # jendela 1280x800
    python scripts/demo_live.py --ambang 40000     # ambang IR kontak jari
    python scripts/demo_live.py --hr 88 --rr 20    # irama sinyal contoh
    python scripts/demo_live.py --simulasi         # tanpa sensor: Spasi = tempel/lepas jari

Konsol mencetak level IR tiap detik untuk kalibrasi --ambang. Esc atau Q untuk keluar.
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtWidgets import QApplication, QLabel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FS = 400  # Hz, sama dengan PlotPage.SAMPLE_RATE_HZ


def _ppg_pulse(x):
    """Satu denyut PPG (sama dengan demo_hasil.py): naik curam, bahu dicrotic."""
    peak = 0.15
    width = np.where(x < peak, 0.055, 0.075)
    systolic = np.exp(-((x - peak) ** 2) / (2 * width ** 2))
    shoulder = 0.33 * np.exp(-((x - (peak + 0.16)) ** 2) / (2 * 0.045 ** 2))
    return systolic + shoulder


ECG_WAVES = ((0.14, -0.19, 0.025), (-0.12, -0.035, 0.010), (1.30, 0.0, 0.013),
             (-0.75, 0.035, 0.014), (0.70, 0.24, 0.045))  # P, Q, R, S, T


class SyntheticSource:
    """Pembangkit ECG + PPG bersih sampel demi sampel, tanpa sambungan/loop."""

    def __init__(self, hr, rr, seed=7):
        self.rng = np.random.default_rng(seed)
        self.hr = float(np.clip(hr, 40, 160))
        self.rr = float(np.clip(rr, 8, 40))
        self.t = 0.0
        self.beats = deque()  # (waktu puncak R, skala amplitudo PPG)
        self.next_beat = 0.3

    def _schedule(self):
        while self.next_beat < self.t + 1.0:
            self.beats.append((self.next_beat, 1 + self.rng.normal(0, 0.03)))
            period = 60.0 / self.hr * (1 + 0.03 * np.sin(2 * np.pi * self.rr / 60 * self.next_beat)
                                       + self.rng.normal(0, 0.01))
            self.next_beat += period
        while self.beats and self.beats[0][0] < self.t - 1.5:
            self.beats.popleft()

    def step(self):
        self._schedule()
        t = self.t
        ecg = 0.0
        ppg = 0.0
        for beat, scale in self.beats:
            for amp, offset, width in ECG_WAVES:
                ecg += amp * np.exp(-((t - beat - offset) ** 2) / (2 * width ** 2))
            ppg += scale * float(_ppg_pulse(np.array(t - beat - 0.18)))
        breathing = np.sin(2 * np.pi * self.rr / 60 * t)
        ecg = 650 * (ecg + 0.04 * breathing) + self.rng.normal(0, 3)
        ppg = 55 * ppg * (1 + 0.05 * breathing) + self.rng.normal(0, 0.08)
        self.t += 1 / FS
        return ecg, ppg  # baseline PPG di nol, sama dengan garis datar saat jari lepas


class ContactDetector:
    """Kontak jari dari median IR mentah, dengan histeresis agar tidak berkedip."""

    def __init__(self, threshold, window=40):
        self.on_level = threshold
        self.off_level = threshold * 0.85
        self.buffer = deque(maxlen=window)  # 40 sampel = 0,1 s
        self.contact = False

    def update(self, ir):
        self.buffer.append(ir)
        level = float(np.median(self.buffer))
        if self.contact and level < self.off_level:
            self.contact = False
        elif not self.contact and level > self.on_level:
            self.contact = True
        return self.contact, level


class DemoFilter:
    """Pengganti LiveSignalFilter: keluarannya sinyal contoh saat ada kontak, datar saat tidak.

    PlotPage memanggil filter_ecg(ecg) lalu filter_ppg(ir) untuk setiap paket; kontak
    dihitung dari IR di filter_ppg, ECG memakai status kontak sampel sebelumnya.
    """

    FADE_SEC = 0.4

    def __init__(self, source, detector, on_contact_change):
        self.source = source
        self.detector = detector
        self.on_contact_change = on_contact_change
        self.contact = False
        self.gain = 0.0
        self.pending_ppg = 0.0
        self.last_report = time.monotonic()
        self.level = 0.0

    def filter_ecg(self, _raw_ecg):
        ecg, self.pending_ppg = self.source.step()
        target = 1.0 if self.contact else 0.0
        step = 1.0 / (self.FADE_SEC * FS)
        # Masuk perlahan (seperti sensor yang baru stabil), keluar seketika.
        self.gain = min(target, self.gain + step) if target > self.gain else target
        return ecg * self.gain

    def filter_ppg(self, raw_ir):
        contact, self.level = self.detector.update(raw_ir)
        if contact != self.contact:
            self.contact = contact
            self.on_contact_change(contact)
        now = time.monotonic()
        if now - self.last_report >= 1.0:
            self.last_report = now
            print(f"[DEMO] IR median {self.level:9.0f} | kontak: {'YA' if contact else 'tidak'}")
        return self.pending_ppg * self.gain


class SimulatedWorker(QThread):
    """Pengganti STM32Worker: paket palsu 400 Hz, IR tinggi jika 'jari' ditempel."""

    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    finger_on = False

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        rng = np.random.default_rng(1)
        next_t = time.perf_counter()
        while self.running and not self.isInterruptionRequested():
            ir = 100000 if SimulatedWorker.finger_on else 3000
            self.data_received.emit({
                "status": "OK", "timestamp": time.time(),
                "ppg": {"red": ir, "ir": ir + int(rng.normal(0, 50)), "green": 0},
                "ecg": 2048, "temperature": {"ambient": 28.0, "object": 34.5},
            })
            next_t += 1 / FS
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def stop(self):
        self.running = False
        self.requestInterruption()
        self.wait(2000)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ambang", type=float, default=50000,
                        help="level IR mentah minimal untuk dianggap ada jari (default 50000)")
    parser.add_argument("--hr", type=float, default=82, help="detak/menit sinyal contoh")
    parser.add_argument("--rr", type=float, default=18, help="napas/menit sinyal contoh")
    parser.add_argument("--jendela", action="store_true", help="jendela 1280x800, bukan layar penuh")
    parser.add_argument("--simulasi", action="store_true",
                        help="tanpa sensor; tekan Spasi untuk menempel/melepas jari")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    import GUI.plot_page as plot_page
    if args.simulasi:
        plot_page.STM32Worker = SimulatedWorker

    page = plot_page.PlotPage()
    page.RECORD_DURATION_SEC = 10 ** 6  # demo tidak berhenti sendiri
    page.total_target_samples = int((page.WARMUP_DURATION_SEC + page.RECORD_DURATION_SEC) * FS)
    page.setWindowTitle("TriaGO - Demo Perekaman")

    status = QLabel()
    status.setAlignment(Qt.AlignmentFlag.AlignCenter)
    page.layout().insertWidget(1, status)

    def show_contact(contact):
        if contact:
            status.setText("Jari terdeteksi — sinyal stabil")
            status.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF; "
                                 "background: #2E8B57; border-radius: 10px; padding: 6px;")
        else:
            status.setText("Letakkan jari pada sensor")
            status.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF; "
                                 "background: #C0392B; border-radius: 10px; padding: 6px;")

    show_contact(False)
    # Skala tetap: autorange akan memperbesar garis datar sampai terlihat seperti sinyal.
    for plot, (low, high) in ((page.ecg_plot, (-600, 1000)), (page.ppg_plot, (-10, 70))):
        plot.getViewBox().disableAutoRange(axis="y")
        plot.setYRange(low, high, padding=0)

    original_start = page.start_session

    def start_session(info):
        original_start(info)
        page.live_filter = DemoFilter(SyntheticSource(args.hr, args.rr),
                                      ContactDetector(args.ambang), show_contact)

    page.start_session = start_session
    page.sensor_error.connect(lambda msg: (status.setText(f"Sensor error: {msg}"),
                                           print(f"[DEMO] Sensor error: {msg}")))

    for key in ("Esc", "Q"):
        QShortcut(QKeySequence(key), page, activated=app.quit)
    if args.simulasi:
        def toggle():
            SimulatedWorker.finger_on = not SimulatedWorker.finger_on
        QShortcut(QKeySequence("Space"), page, activated=toggle)
        print("[DEMO] Mode simulasi: tekan Spasi untuk menempel/melepas jari.")
    app.aboutToQuit.connect(page.close_threads)

    if args.jendela:
        page.resize(1280, 800)
        page.show()
    else:
        page.setWindowState(Qt.WindowState.WindowFullScreen)
        page.show()
    page.start_session({"name": "Demo"})
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
