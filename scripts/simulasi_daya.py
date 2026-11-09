# -*- coding: utf-8 -*-
"""Simulasi beban & daya TriaGO di Raspberry Pi.

Menjalankan SELURUH pipeline TriaGO tanpa sensor fisik:
  1. Akuisisi  : kode asli ``akusisi_data.get_stm32.stream_stm32_data`` dengan
                 port serial virtual yang me-replay rekaman CSV pada 400 Hz
                 (format paket STM32: RED,IR,GREEN,ECG,T_AMB,T_OBJ).
  2. Pemrosesan: filter ECG, R-peak/HR, PPG SpO2/PI, RR multimodal ECG+PPG.
  3. Deep learning: SQA 10 s + BPNet TFLite (int8) + XGBoost ONNX (triase).

Dua skenario dibandingkan agar pengurangan daya terlihat:
  * baseline  : akuisisi busy-polling (tanpa jeda 5 ms) + inferensi diulang
                tiap 2 detik selama akuisisi (desain real-time naif).
  * optimized : desain TriaGO saat ini -- baca serial per-bongkahan dengan
                jeda 5 ms, inferensi sekali setelah akuisisi, lalu idle.

Sumber daya listrik (otomatis):
  * Raspberry Pi 5 -> ``vcgencmd pmic_read_adc`` (pengukuran PMIC, sum V x I).
  * Lainnya        -> estimasi linear P = P_idle + (P_max - P_idle) x CPU%.
Untuk validasi, colok USB power meter: setiap pergantian fase dicetak dengan
jam dinding sehingga bisa dicocokkan dengan bacaan meter.

Contoh (di Raspberry Pi, dari root repo):
  python3 scripts/simulasi_daya.py                       # kedua skenario, 60 s
  python3 scripts/simulasi_daya.py --skenario optimized --siklus 3
  python3 scripts/simulasi_daya.py --durasi 30 --idle 5 --csv data_pengukuran/xxx.csv

Default BPNet dijalankan pada SEMUA segmen 10 s (beban DL penuh); --ikuti-sqa
meniru GUI yang hanya menginferensi segmen lolos SQA.

Hasil: runtime/simulasi_daya/<waktu>/ (sampel.csv, ringkasan.json, grafik.png)
"""

import argparse
import contextlib
import glob
import json
import os
import subprocess
import sys
import threading
import time
import types
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from akusisi_data import get_stm32  # noqa: E402
from processing_data.processing_data import ECGProcessor, PPGProcessor  # noqa: E402
from model import bpnet_inference  # noqa: E402
from model.deployment_inference import TriageOnnxModel  # noqa: E402

FS_RAW = 400
# Profil daya papan (W): (idle, beban penuh 4 core). Hanya dipakai untuk mode
# estimasi; angka mengikuti pengukuran umum Raspberry Pi tanpa periferal.
PROFIL_DAYA = {
    "Raspberry Pi 5": (2.7, 7.0),
    "Raspberry Pi 4": (2.7, 6.4),
    "Raspberry Pi 3": (1.9, 5.1),
    "Raspberry Pi Zero 2": (0.6, 3.0),
}


# -------------------------------------------------------------------------
# 1. MONITOR CPU & DAYA
# -------------------------------------------------------------------------
def baca_model_papan():
    try:
        return Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except OSError:
        return "Bukan Raspberry Pi"


def baca_pmic_watt():
    """Jumlahkan V x I seluruh rail PMIC Raspberry Pi 5. None jika tidak ada."""
    try:
        out = subprocess.run(
            ["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    arus, volt = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2 or "=" not in parts[1]:
            continue
        nama, nilai = parts[0], parts[1].split("=")[1]
        try:
            angka = float(nilai.rstrip("AV"))
        except ValueError:
            continue
        if nama.endswith("_A"):
            arus[nama[:-2]] = angka
        elif nama.endswith("_V"):
            volt[nama[:-2]] = angka
    rail = set(arus) & set(volt)
    if not rail:
        return None
    return sum(arus[r] * volt[r] for r in rail)


class PembacaCPU:
    """Utilisasi CPU rata-rata semua core (0-100) sejak panggilan sebelumnya."""

    def __init__(self):
        self._prev = self._stat()
        if self._prev is None:
            import psutil  # fallback non-Linux (mis. uji di laptop)
            self._psutil = psutil
            psutil.cpu_percent(interval=None)

    @staticmethod
    def _stat():
        try:
            with open("/proc/stat") as f:
                nilai = [int(v) for v in f.readline().split()[1:]]
        except OSError:
            return None
        idle = nilai[3] + nilai[4]
        return sum(nilai), idle

    def baca(self):
        if self._prev is None:
            return float(self._psutil.cpu_percent(interval=None))
        total, idle = self._stat()
        d_total, d_idle = total - self._prev[0], idle - self._prev[1]
        self._prev = (total, idle)
        return 100.0 * (1.0 - d_idle / d_total) if d_total > 0 else 0.0


def baca_sysfs(path, skala):
    try:
        return int(Path(path).read_text()) / skala
    except (OSError, ValueError):
        return None


class MonitorDaya(threading.Thread):
    def __init__(self, interval, p_idle, p_max, pakai_pmic):
        super().__init__(daemon=True)
        self.interval = interval
        self.p_idle, self.p_max = p_idle, p_max
        self.pakai_pmic = pakai_pmic
        self.cpu = PembacaCPU()
        self.skenario, self.fase = "-", "persiapan"
        self.sampel = []
        self._stop = threading.Event()
        self.t0 = time.monotonic()

    def set_fase(self, skenario, fase):
        self.skenario, self.fase = skenario, fase
        print(f"[{datetime.now():%H:%M:%S}] >>> {skenario} | fase: {fase}", flush=True)

    def run(self):
        terakhir_cetak = t_prev = 0.0
        while not self._stop.wait(self.interval):
            cpu = self.cpu.baca()
            p_est = self.p_idle + (self.p_max - self.p_idle) * cpu / 100.0
            p_pmic = baca_pmic_watt() if self.pakai_pmic else None
            t = time.monotonic() - self.t0
            # dt nyata dipakai untuk integrasi energi: thread monitor bisa telat
            # bangun saat skenario busy-poll menahan GIL.
            dt, t_prev = t - t_prev, t
            self.sampel.append({
                "t_s": round(t, 3),
                "dt_s": round(dt, 4),
                "jam": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "skenario": self.skenario,
                "fase": self.fase,
                "cpu_pct": round(cpu, 2),
                "daya_w": round(p_pmic if p_pmic is not None else p_est, 4),
                "daya_estimasi_w": round(p_est, 4),
                "daya_pmic_w": None if p_pmic is None else round(p_pmic, 4),
                "suhu_c": baca_sysfs("/sys/class/thermal/thermal_zone0/temp", 1000.0),
                "freq_mhz": baca_sysfs(
                    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000.0
                ),
            })
            if t - terakhir_cetak >= 5.0:
                s = self.sampel[-1]
                print(f"    t={t:6.1f}s  CPU={s['cpu_pct']:5.1f}%  P={s['daya_w']:.2f} W"
                      f"  suhu={s['suhu_c']}", flush=True)
                terakhir_cetak = t

    def berhenti(self):
        self._stop.set()
        self.join()


# -------------------------------------------------------------------------
# 2. SERIAL VIRTUAL STM32 (REPLAY REKAMAN)
# -------------------------------------------------------------------------
def muat_rekaman(csv_path):
    """Ubah CSV rekaman TriaGO menjadi baris paket STM32 (bytes)."""
    import pandas as pd

    if csv_path:
        df = pd.read_csv(csv_path, usecols=["Raw_ECG", "Raw_PPG_Red", "Raw_PPG_IR", "Suhu_C"])
        df = df.dropna(subset=["Raw_ECG", "Raw_PPG_Red", "Raw_PPG_IR"])
        baris = [
            f"{int(r)},{int(i)},{int(i)},{int(e)},28.00,34.50\n".encode()
            for e, r, i in zip(df["Raw_ECG"], df["Raw_PPG_Red"], df["Raw_PPG_IR"])
        ]
        return baris, Path(csv_path).name

    # Tanpa rekaman: sinyal sintetis 72 bpm / 15 napas per menit.
    t = np.arange(FS_RAW * 60) / FS_RAW
    fase = (t * 1.2) % 1.0
    ecg = 600 + 900 * np.exp(-((fase - 0.3) ** 2) / 0.0004) + 40 * np.sin(2 * np.pi * 0.25 * t)
    ppg = np.exp(-((fase - 0.45) ** 2) / 0.01) * (1 + 0.1 * np.sin(2 * np.pi * 0.25 * t))
    red, ir = 27000 - 300 * ppg, 32000 - 400 * ppg
    baris = [
        f"{int(r)},{int(i)},{int(i)},{int(e)},28.00,34.50\n".encode()
        for e, r, i in zip(ecg, red, ir)
    ]
    return baris, "sintetis"


class SerialReplay:
    """Pengganti ``serial.Serial``: data mengalir 400 sampel/detik setelah START."""

    def __init__(self, baris, fs):
        self.baris, self.fs = baris, fs
        self.is_open, self.dtr, self.rts = True, False, False
        self._buf = bytearray()
        self._t0, self._terkirim = None, 0

    def _pompa(self):
        if self._t0 is None:
            return
        jatuh_tempo = int((time.monotonic() - self._t0) * self.fs)
        while self._terkirim < jatuh_tempo:
            self._buf += self.baris[self._terkirim % len(self.baris)]
            self._terkirim += 1

    @property
    def in_waiting(self):
        self._pompa()
        return len(self._buf)

    def read(self, n=1):
        self._pompa()
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def write(self, data):
        if b"START" in data and self._t0 is None:
            self._t0 = time.monotonic()
        return len(data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        self._buf.clear()

    def reset_output_buffer(self):
        pass

    def close(self):
        self.is_open = False


def akuisisi(baris, durasi, busy_poll, saat_paket=None):
    """Jalankan stream_stm32_data asli selama ``durasi`` detik."""
    data = {"ecg": [], "red": [], "ir": [], "tamb": [], "tobj": []}
    batas = time.monotonic() + durasi
    patches = [mock.patch.object(get_stm32.serial, "Serial",
                                 lambda *a, **k: SerialReplay(baris, FS_RAW))]
    if busy_poll:
        # Hilangkan "napas komputasi" 5 ms -> loop polling memakan satu core penuh.
        tanpa_jeda = types.SimpleNamespace(
            sleep=lambda _s: None, monotonic=time.monotonic, time=time.time
        )
        patches.append(mock.patch.object(get_stm32, "time", tanpa_jeda))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        for pkt in get_stm32.stream_stm32_data(
            port="REPLAY", should_stop=lambda: time.monotonic() >= batas
        ):
            if pkt.get("status") != "OK":
                continue
            data["ecg"].append(pkt["ecg"])
            data["red"].append(pkt["ppg"]["red"])
            data["ir"].append(pkt["ppg"]["ir"])
            data["tamb"].append(pkt["temperature"]["ambient"])
            data["tobj"].append(pkt["temperature"]["object"])
            if saat_paket:
                saat_paket(data)
    return data


# -------------------------------------------------------------------------
# 3. PIPELINE PEMROSESAN + DEEP LEARNING (mengikuti GUI/loading_page.py)
# -------------------------------------------------------------------------
def bpnet_per_segmen(ecg_125, ppg_125, bpnet, paksa, fs=125.0):
    """SQA + fitur 7 kanal + BPNet per segmen 10 s (stride 2 s).

    ``paksa=True`` menjalankan BPNet pada SEMUA segmen walau SQA menolak, agar
    beban deep learning selalu ada di simulasi. Tanpa interpreter TFLite hanya
    SQA dan ekstraksi fitur yang dihitung.
    """
    lolos = total = 0
    sbp, dbp = [], []
    for start in range(0, min(len(ecg_125), len(ppg_125)) - 1250 + 1, 250):
        total += 1
        e, p = ecg_125[start:start + 1250], ppg_125[start:start + 1250]
        ok, _ = bpnet_inference.evaluate_segment_sqa_v2(e, p, fs=fs)
        lolos += int(ok)
        if ok or paksa:
            p_al, _ = bpnet_inference.align_ppg_phase_cross_correlation(e, p, fs=fs)
            fitur = bpnet_inference.extract_7channel_features(e, p_al, fs=fs)
            if bpnet is not None:
                s, d = bpnet.predict_segment(fitur)
                sbp.append(s)
                dbp.append(d)
    return {"passed_segments": lolos, "total_segments": total, "inferensi_bpnet": len(sbp),
            "sbp": float(np.clip(np.median(sbp), 70, 220)) if sbp else 120.0,
            "dbp": float(np.clip(np.median(dbp), 40, 140)) if dbp else 80.0}


def jalankan_pipeline(data, bpnet, triase, paksa_bpnet=True):
    n = len(data["ecg"])
    raw_time = np.arange(n) / FS_RAW
    ecg = np.asarray(data["ecg"], dtype=float)
    red = np.asarray(data["red"], dtype=float)
    ir = np.asarray(data["ir"], dtype=float)
    ecg_p, ppg_p = ECGProcessor(target_fs=125), PPGProcessor(target_fs=125)
    waktu = {}

    def tahap(nama, fungsi):
        t = time.perf_counter()
        hasil = fungsi()
        waktu[nama] = round((time.perf_counter() - t) * 1000, 1)
        return hasil

    def filter_ecg():
        ecg_125, _ = ecg_p.downsample(ecg, raw_time, fs=FS_RAW, fs_target=125)
        x = ecg_p.notch(ecg_125, freq=50.0, fs=125)
        x = ecg_p.detrending(x, fs=125)
        x = ecg_p.lowpass(x, lowcut=35.0, fs=125)
        return ecg_125, ecg_p.savgol(x, window_size=11, poly_order=2)

    ecg_125, ecg_smooth = tahap("filter_ecg_ms", filter_ecg)
    r_peaks, _ = tahap("rpeak_ms", lambda: ecg_p.detect_r_peaks(ecg_125, fs=125))
    hr = ecg_p.calculate_heart_rate(r_peaks, fs=125)
    ppg = tahap("ppg_spo2_ms", lambda: ppg_p.process_ppg(
        raw_time=raw_time, raw_red=red, raw_ir=ir, fs_orig=FS_RAW))
    rr, _, _ = tahap("rr_multimodal_ms", lambda: ecg_p.calculate_respiration_rate(
        ecg_smooth, r_peaks, fs=125, ppg=ppg["ir_clean"], ppg_peaks=ppg.get("ir_peaks")))

    def deep_learning_bp():
        return bpnet_per_segmen(ecg_smooth, ppg["ir_clean"], bpnet, paksa_bpnet)

    bp = tahap("sqa_bpnet_ms", deep_learning_bp)

    # Estimasi suhu inti (bioheat) sama seperti GUI.
    t_skin, t_amb = float(np.mean(data["tobj"])), float(np.mean(data["tamb"]))
    pi_norm = float(np.clip(ppg["pi_ir"] if ppg["pi_ir"] > 0 else 0.2, 0.05, 3.0))
    k_env = -0.1858 / (1.0 + -0.1980 * (pi_norm - 0.2))
    suhu = float(np.clip(t_skin + k_env * (t_skin - t_amb) + 5.6694, 30.0, 43.0))

    vitals = {
        "temperature_c": suhu,
        "spo2": float(ppg["spo2"] if ppg["spo2"] > 0 else 98.0),
        "respiratory_rate": float(rr if rr > 0 else 16.0),
        "heart_rate": float(hr if hr > 0 else 75.0),
        "systolic_bp": float(bp["sbp"]),
        "diastolic_bp": float(bp["dbp"]),
        "gcs_total": 15.0,
        "pain_score": 0.0,
    }
    # Simulasi beban: ONNX tetap dijalankan walau SQA gagal (GUI akan melewatinya).
    label, conf, _ = tahap("xgboost_onnx_ms", lambda: triase.predict(vitals))
    return {
        "waktu_ms": waktu,
        "hr": round(vitals["heart_rate"], 1),
        "spo2": round(vitals["spo2"], 1),
        "rr": round(vitals["respiratory_rate"], 1),
        "sbp": round(vitals["systolic_bp"], 1),
        "dbp": round(vitals["diastolic_bp"], 1),
        "sqa": f"{bp['passed_segments']}/{bp['total_segments']}",
        "inferensi_bpnet": bp["inferensi_bpnet"],
        "triase": label,
        "confidence": round(float(conf), 3),
    }


# -------------------------------------------------------------------------
# 4. SKENARIO
# -------------------------------------------------------------------------
def skenario_optimized(monitor, baris, args, bpnet, triase):
    hasil = []
    for i in range(args.siklus):
        monitor.set_fase("optimized", "akuisisi")
        data = akuisisi(baris, args.durasi, busy_poll=False)
        monitor.set_fase("optimized", "proses")
        hasil.append(jalankan_pipeline(data, bpnet, triase, not args.ikuti_sqa))
        print(f"    siklus {i + 1}: {hasil[-1]}", flush=True)
    return hasil


def skenario_baseline(monitor, baris, args, bpnet, triase):
    hasil, jumlah_inferensi = [], [0]
    for i in range(args.siklus):
        snapshot = {"data": None}
        kunci = threading.Lock()
        selesai = threading.Event()

        def saat_paket(data):
            with kunci:
                snapshot["data"] = data

        def inferensi_kontinu():
            # Desain naif: pipeline penuh diulang tiap 2 s pada jendela 30 s terakhir.
            while not selesai.wait(2.0):
                with kunci:
                    d = snapshot["data"]
                    jendela = None if d is None or len(d["ecg"]) < FS_RAW * 12 else {
                        k: list(v[-FS_RAW * 30:]) for k, v in d.items()}
                if jendela:
                    jalankan_pipeline(jendela, bpnet, triase, not args.ikuti_sqa)
                    jumlah_inferensi[0] += 1

        monitor.set_fase("baseline", "akuisisi")
        pekerja = threading.Thread(target=inferensi_kontinu, daemon=True)
        pekerja.start()
        data = akuisisi(baris, args.durasi, busy_poll=True, saat_paket=saat_paket)
        selesai.set()
        pekerja.join()
        monitor.set_fase("baseline", "proses")
        hasil.append(jalankan_pipeline(data, bpnet, triase, not args.ikuti_sqa))
        print(f"    siklus {i + 1}: {hasil[-1]} (inferensi kontinu: {jumlah_inferensi[0]}x)",
              flush=True)
    return hasil


def rangkum(sampel):
    ringkasan = {}
    for s in sampel:
        if s["skenario"] == "-":
            continue
        fase = ringkasan.setdefault(s["skenario"], {}).setdefault(
            s["fase"], {"durasi_s": 0.0, "cpu": 0.0, "energi_j": 0.0})
        fase["durasi_s"] += s["dt_s"]
        fase["cpu"] += s["cpu_pct"] * s["dt_s"]
        fase["energi_j"] += s["daya_w"] * s["dt_s"]
    for fases in ringkasan.values():
        for f in fases.values():
            f["cpu_rata_pct"] = round(f.pop("cpu") / f["durasi_s"], 1)
            f["daya_rata_w"] = round(f["energi_j"] / f["durasi_s"], 3)
            f["energi_j"] = round(f["energi_j"], 1)
            f["durasi_s"] = round(f["durasi_s"], 1)
    return ringkasan


def buat_grafik(sampel, ringkasan, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    warna = {"baseline": "#d1495b", "optimized": "#2e86ab"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5))
    skenario = [k for k in ringkasan if k in warna]
    for sk in skenario:
        titik = [s for s in sampel if s["skenario"] == sk]
        t0 = titik[0]["t_s"]
        ax1.plot([s["t_s"] - t0 for s in titik], [s["daya_w"] for s in titik],
                 color=warna.get(sk), label=sk, lw=1.4)
    ax1.set_xlabel("Waktu sejak awal skenario (s)")
    ax1.set_ylabel("Daya (W)")
    ax1.set_title("Profil daya Raspberry Pi per skenario")
    ax1.grid(alpha=0.3)
    ax1.legend()

    fase_urut = [f for f in ("idle_awal", "akuisisi", "proses", "idle_akhir")
                 if any(f in ringkasan[sk] for sk in skenario)]
    lebar = 0.38
    for j, sk in enumerate(skenario):
        nilai = [ringkasan[sk].get(f, {}).get("daya_rata_w", 0) for f in fase_urut]
        x = np.arange(len(fase_urut)) + (j - 0.5) * lebar
        batang = ax2.bar(x, nilai, lebar, color=warna.get(sk), label=sk)
        ax2.bar_label(batang, fmt="%.2f", fontsize=8)
    ax2.set_xticks(np.arange(len(fase_urut)), fase_urut)
    ax2.set_ylabel("Daya rata-rata (W)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skenario", choices=["keduanya", "baseline", "optimized"],
                    default="keduanya")
    ap.add_argument("--durasi", type=float, default=60.0, help="detik akuisisi per siklus")
    ap.add_argument("--siklus", type=int, default=1, help="jumlah pasien per skenario")
    ap.add_argument("--idle", type=float, default=10.0, help="detik idle sebelum & sesudah")
    ap.add_argument("--interval", type=float, default=0.5, help="periode sampling daya (s)")
    ap.add_argument("--csv", help="rekaman TriaGO_FullData_*.csv untuk di-replay")
    ap.add_argument("--ikuti-sqa", action="store_true",
                    help="BPNet hanya untuk segmen lolos SQA (default: semua segmen)")
    ap.add_argument("--p-idle", type=float, help="override daya idle (W) untuk estimasi")
    ap.add_argument("--p-max", type=float, help="override daya beban penuh (W) untuk estimasi")
    ap.add_argument("--out", help="folder hasil (default runtime/simulasi_daya/<waktu>)")
    args = ap.parse_args()

    papan = baca_model_papan()
    profil = next((v for k, v in PROFIL_DAYA.items() if papan.startswith(k)), (2.7, 6.4))
    p_idle = args.p_idle if args.p_idle is not None else profil[0]
    p_max = args.p_max if args.p_max is not None else profil[1]
    pakai_pmic = baca_pmic_watt() is not None
    sumber = "PMIC vcgencmd (terukur)" if pakai_pmic else (
        f"estimasi CPU% (idle {p_idle} W, penuh {p_max} W)")

    csv_path = args.csv
    if csv_path is None:
        kandidat = sorted(glob.glob(str(ROOT / "data_pengukuran" / "TriaGO_FullData_*.csv")))
        csv_path = kandidat[-1] if kandidat else None
    baris, nama_rekaman = muat_rekaman(csv_path)

    out = Path(args.out) if args.out else (
        ROOT / "runtime" / "simulasi_daya" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f" Papan         : {papan}")
    print(f" Sumber daya   : {sumber}")
    print(f" Rekaman       : {nama_rekaman} ({len(baris) / FS_RAW:.0f} s @ {FS_RAW} Hz)")
    print(f" Skenario      : {args.skenario}, {args.siklus} siklus x {args.durasi:g} s")
    print("=" * 66, flush=True)

    monitor = MonitorDaya(args.interval, p_idle, p_max, pakai_pmic)
    monitor.start()
    monitor.set_fase("-", "muat_model")
    triase = TriageOnnxModel()
    try:
        bpnet = bpnet_inference.BPNetTflitePredictor()
        print(f"    BPNet TFLite via {bpnet.interpreter_backend}")
    except (ImportError, FileNotFoundError) as exc:
        bpnet = None
        print(f"    [PERINGATAN] BPNet tidak aktif ({exc}); hanya SQA + fitur yang dijalankan.")

    # Beri jeda agar lonjakan CPU saat memuat model tidak terhitung sebagai idle.
    monitor.set_fase("-", "stabilisasi")
    time.sleep(max(3.0, 2 * args.interval))

    urutan = ["baseline", "optimized"] if args.skenario == "keduanya" else [args.skenario]
    jalankan = {"baseline": skenario_baseline, "optimized": skenario_optimized}
    hasil_pipeline = {}
    for sk in urutan:
        monitor.set_fase(sk, "idle_awal")
        time.sleep(args.idle)
        hasil_pipeline[sk] = jalankan[sk](monitor, baris, args, bpnet, triase)
        monitor.set_fase(sk, "idle_akhir")
        time.sleep(args.idle)
    monitor.berhenti()

    ringkasan = rangkum(monitor.sampel)
    print("\n" + "=" * 66)
    print(f" {'skenario':<10} {'fase':<11} {'durasi':>7} {'CPU%':>6} {'P rata':>8} {'energi':>9}")
    for sk, fases in ringkasan.items():
        for fase, f in fases.items():
            print(f" {sk:<10} {fase:<11} {f['durasi_s']:>6}s {f['cpu_rata_pct']:>6}"
                  f" {f['daya_rata_w']:>7}W {f['energi_j']:>8}J")
    if {"baseline", "optimized"} <= set(ringkasan):
        def energi_kerja(sk):
            return sum(ringkasan[sk].get(f, {}).get("energi_j", 0) for f in ("akuisisi", "proses"))
        e_base, e_opt = energi_kerja("baseline"), energi_kerja("optimized")
        p_base = ringkasan["baseline"]["akuisisi"]["daya_rata_w"]
        p_opt = ringkasan["optimized"]["akuisisi"]["daya_rata_w"]
        ringkasan["perbandingan"] = {
            "energi_per_siklus_baseline_j": round(e_base / args.siklus, 1),
            "energi_per_siklus_optimized_j": round(e_opt / args.siklus, 1),
            "penghematan_energi_pct": round(100 * (e_base - e_opt) / e_base, 1),
            "penurunan_daya_akuisisi_pct": round(100 * (p_base - p_opt) / p_base, 1),
        }
        print("-" * 66)
        for k, v in ringkasan["perbandingan"].items():
            print(f" {k:<34}: {v}")
    print("=" * 66)

    import csv
    with open(out / "sampel.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(monitor.sampel[0].keys()))
        writer.writeheader()
        writer.writerows(monitor.sampel)
    meta = {"papan": papan, "sumber_daya": sumber, "rekaman": nama_rekaman,
            "argumen": vars(args), "ringkasan": ringkasan, "hasil_pipeline": hasil_pipeline}
    (out / "ringkasan.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    ada_grafik = buat_grafik(monitor.sampel, ringkasan, out / "grafik.png")
    print(f" Hasil disimpan di: {out}" + (" (termasuk grafik.png)" if ada_grafik else ""))


if __name__ == "__main__":
    main()
