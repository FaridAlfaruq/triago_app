# -*- coding: utf-8 -*-
"""Modul bpnet_inference.py

Menyediakan pipeline inferensi Deep Learning BPNet (TFLite) untuk estimasi
Tekanan Darah (SBP & DBP) berdasarkan sinyal ECG & PPG 10s dengan SQA Multi-Faktor,
stride overlap 2s, phase alignment korelasi silang, dan ekstraksi 7-channel features.
"""

import os
from pathlib import Path
import numpy as np
import scipy.signal as signal
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks, correlate

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            tflite = None

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_TFLITE_PATH = MODEL_DIR / "triago_bpnet_v53_quant.tflite"
DEFAULT_SCALER_PATH = MODEL_DIR / "target_scaler_params.json"

# Parameter Scaler SBP/DBP Latih (Fallback if file unavailable)
DEFAULT_SBP_MEAN = 114.1925
DEFAULT_SBP_STD = 16.9032
DEFAULT_DBP_MEAN = 76.4535
DEFAULT_DBP_STD = 9.7663


# -------------------------------------------------------------------------
# 1. HELPER METRIK SQA (SKVNENESS, KURTOSIS, SPECTRAL SNR)
# -------------------------------------------------------------------------
def compute_ppg_skewness(ppg_seg):
    return float(skew(ppg_seg))


def compute_ppg_kurtosis(ppg_seg):
    return float(kurtosis(ppg_seg, fisher=False))


def compute_spectral_snr(sig_seg, fs=125.0, f_low=0.5, f_high=8.0):
    freqs, psd = welch(sig_seg, fs=fs, nperseg=int(min(len(sig_seg), 2 * fs)))
    total_power = np.sum(psd) + 1e-8
    band_power = np.sum(psd[(freqs >= f_low) & (freqs <= f_high)])
    return float(band_power / total_power)


def evaluate_segment_sqa_v2(ecg_seg, ppg_seg, fs=125.0, hr_diff_max=12.0):
    """Evaluasi Kualitas Sinyal Multimodal (ECG & PPG 10-Detik):

    Sensitifitas Disesuaikan untuk Penggunaan Klinis/Hardware Riil:
    1. Check Flatline / NaNs / Infs
    2. Peak Detection ECG & PPG (Min 3 puncak)
    3. Range HR Fisiologis (35 - 200 bpm)
    4. HR Mismatch ECG vs PPG (<= 12 bpm)
    5. Indeks Statistik PPG (Skewness > -0.4, Kurtosis >= 1.2)
    6. Relative Spectral Power Ratio (SNR >= 35%)
    """
    if np.isnan(ecg_seg).any() or np.isnan(ppg_seg).any():
        return False, "NaN/Inf Found"
    if np.std(ecg_seg) < 1e-4 or np.std(ppg_seg) < 1e-4:
        return False, "Flatline Signal"

    ecg_peaks, _ = find_peaks(ecg_seg, distance=int(0.3 * fs), prominence=0.3 * np.std(ecg_seg))
    if len(ecg_peaks) < 3:
        return False, f"Puncak ECG Terlalu Sedikit ({len(ecg_peaks)})"

    ppg_peaks, _ = find_peaks(ppg_seg, distance=int(0.3 * fs), prominence=0.20 * np.std(ppg_seg))
    if len(ppg_peaks) < 3:
        return False, f"Puncak PPG Terlalu Sedikit ({len(ppg_peaks)})"

    hr_ecg = 60.0 / np.mean(np.diff(ecg_peaks) / fs)
    hr_ppg = 60.0 / np.mean(np.diff(ppg_peaks) / fs)

    if not (35.0 <= hr_ecg <= 200.0) or not (35.0 <= hr_ppg <= 200.0):
        return False, f"HR Luar Batas (ECG: {hr_ecg:.1f}, PPG: {hr_ppg:.1f})"

    if abs(hr_ecg - hr_ppg) > hr_diff_max:
        return False, f"HR Mismatch (ECG: {hr_ecg:.1f} vs PPG: {hr_ppg:.1f})"

    s_sqa = compute_ppg_skewness(ppg_seg)
    k_sqa = compute_ppg_kurtosis(ppg_seg)

    if s_sqa <= -0.4:
        return False, f"PPG Skewness Invalid ({s_sqa:.2f})"
    if k_sqa < 1.2:
        return False, f"PPG Kurtosis Invalid ({k_sqa:.2f})"

    snr_ratio = compute_spectral_snr(ppg_seg, fs=fs, f_low=0.5, f_high=8.0)
    if snr_ratio < 0.35:
        return False, f"PPG Low Spectral SNR ({snr_ratio*100:.1f}%)"

    return True, "VALID_SQA"


# -------------------------------------------------------------------------
# 2. PHASE ALIGNMENT & EXTRACTION 7-CHANNEL FEATURES
# -------------------------------------------------------------------------
def align_ppg_phase_cross_correlation(ecg_seg, ppg_seg, fs=125.0, max_lag_sec=0.4):
    vpg = np.gradient(ppg_seg, 1.0 / fs)
    max_lag_samples = int(max_lag_sec * fs)
    corr = correlate(vpg, ecg_seg, mode='full')
    lags = np.arange(-len(ecg_seg) + 1, len(ecg_seg))

    valid_mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr_bounded = corr[valid_mask]
    lags_bounded = lags[valid_mask]
    optimal_lag = lags_bounded[np.argmax(corr_bounded)]

    if optimal_lag > 0:
        ppg_aligned = np.pad(ppg_seg[optimal_lag:], (0, optimal_lag), mode='edge')
    elif optimal_lag < 0:
        ppg_aligned = np.pad(ppg_seg[:optimal_lag], (-optimal_lag, 0), mode='edge')
    else:
        ppg_aligned = ppg_seg.copy()

    return ppg_aligned, float(optimal_lag / fs)


def extract_7channel_features(ecg_seg, ppg_seg, fs=125.0):
    ecg_norm = (ecg_seg - np.mean(ecg_seg)) / (np.std(ecg_seg) + 1e-8)
    ppg_norm = (ppg_seg - np.mean(ppg_seg)) / (np.std(ppg_seg) + 1e-8)

    vpg = np.gradient(ppg_norm, 1.0 / fs)
    vpg_norm = (vpg - np.mean(vpg)) / (np.std(vpg) + 1e-8)

    apg = np.gradient(vpg_norm, 1.0 / fs)
    apg_norm = (apg - np.mean(apg)) / (np.std(apg) + 1e-8)

    corr_sig = ecg_norm * ppg_norm
    corr_norm = (corr_sig - np.mean(corr_sig)) / (np.std(corr_sig) + 1e-8)

    analytic_ecg = signal.hilbert(ecg_norm)
    ecg_env = np.abs(analytic_ecg)
    ecg_env_norm = (ecg_env - np.mean(ecg_env)) / (np.std(ecg_env) + 1e-8)

    analytic_ppg = signal.hilbert(ppg_norm)
    ppg_env = np.abs(analytic_ppg)
    ppg_env_norm = (ppg_env - np.mean(ppg_env)) / (np.std(ppg_env) + 1e-8)

    multichannel_tensor = np.stack([
        ecg_norm, ppg_norm, vpg_norm, apg_norm,
        corr_norm, ecg_env_norm, ppg_env_norm
    ], axis=-1)

    return multichannel_tensor


# -------------------------------------------------------------------------
# 3. CLASS BPNET INFERENCE ENGINE (TFLITE)
# -------------------------------------------------------------------------
class BPNetTflitePredictor:
    def __init__(self, model_path=None, scaler_path=None):
        if tflite is None:
            raise ImportError("Interpreter TFLite (ai-edge-litert / tflite_runtime / tensorflow.lite) tidak tersedia di environment.")
        self.model_path = model_path or DEFAULT_TFLITE_PATH
        self.scaler_path = scaler_path or DEFAULT_SCALER_PATH

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"File TFLite BPNet tidak ditemukan: {self.model_path}")

        self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load scaler parameters
        import json
        if Path(self.scaler_path).exists():
            with open(self.scaler_path, "r", encoding="utf-8") as f:
                sc = json.load(f)
                target = sc.get("target_scaler", sc)
                self.sbp_mean = float(target.get("sbp_mean", DEFAULT_SBP_MEAN))
                self.sbp_std = float(target.get("sbp_std", DEFAULT_SBP_STD))
                self.dbp_mean = float(target.get("dbp_mean", DEFAULT_DBP_MEAN))
                self.dbp_std = float(target.get("dbp_std", DEFAULT_DBP_STD))
        else:
            self.sbp_mean, self.sbp_std = DEFAULT_SBP_MEAN, DEFAULT_SBP_STD
            self.dbp_mean, self.dbp_std = DEFAULT_DBP_MEAN, DEFAULT_DBP_STD

        # Temukan index output sbp & dbp
        self.sbp_idx, self.dbp_idx = 0, 1
        for idx, out in enumerate(self.output_details):
            name_lower = out['name'].lower()
            if 'sbp' in name_lower:
                self.sbp_idx = idx
            elif 'dbp' in name_lower:
                self.dbp_idx = idx

    def predict_segment(self, feat_7ch: np.ndarray):
        """Memprediksi SBP & DBP (mmHg) untuk 1 segmen [1250, 7]."""
        sample_input = np.expand_dims(feat_7ch, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], sample_input)
        self.interpreter.invoke()

        sbp_z = float(self.interpreter.get_tensor(self.output_details[self.sbp_idx]['index']).flatten()[0])
        dbp_z = float(self.interpreter.get_tensor(self.output_details[self.dbp_idx]['index']).flatten()[0])

        sbp_mmhg = sbp_z * self.sbp_std + self.sbp_mean
        dbp_mmhg = dbp_z * self.dbp_std + self.dbp_mean
        return sbp_mmhg, dbp_mmhg

    def predict_recording(self, ecg_125: np.ndarray, ppg_125: np.ndarray, fs: float = 125.0, window_sec: float = 10.0, stride_sec: float = 2.0):
        """Metode utama: Memecah rekaman sinyal ke segmen 10s dengan stride overlap 2s,

        mengevaluasi SQA, dan memprediksi BP (SBP & DBP) via TFLite.

        Returns:
        --------
        dict: {
            "sqa_passed": bool,
            "passed_segments": int,
            "total_segments": int,
            "sbp": float,
            "dbp": float,
            "rejections": dict
        }
        """
        window_len = int(window_sec * fs)  # 1250 sampel
        stride_len = int(stride_sec * fs)  # 250 sampel

        if len(ecg_125) < window_len or len(ppg_125) < window_len:
            return {
                "sqa_passed": False,
                "passed_segments": 0,
                "total_segments": 0,
                "sbp": 120.0,
                "dbp": 80.0,
                "rejections": {"Signal Too Short": 1}
            }

        total_segments = 0
        passed_segments = 0
        rejections = {}
        sbp_preds = []
        dbp_preds = []

        start = 0
        n_samples = min(len(ecg_125), len(ppg_125))

        print("\n==========================================================")
        print("        EVALUASI PER-SEGMEN SIGNAL QUALITY ASSESSMENT (SQA)")
        print("==========================================================")

        while start + window_len <= n_samples:
            total_segments += 1
            t_start_sec = start / fs
            t_end_sec = (start + window_len) / fs
            ecg_seg = ecg_125[start:start + window_len]
            ppg_seg = ppg_125[start:start + window_len]

            is_valid, reason = evaluate_segment_sqa_v2(ecg_seg, ppg_seg, fs=fs)

            if is_valid:
                passed_segments += 1
                ppg_aligned, _ = align_ppg_phase_cross_correlation(ecg_seg, ppg_seg, fs=fs)
                feat_7ch = extract_7channel_features(ecg_seg, ppg_aligned, fs=fs)
                sbp_val, dbp_val = self.predict_segment(feat_7ch)
                sbp_preds.append(sbp_val)
                dbp_preds.append(dbp_val)
                print(f" [Segmen #{total_segments:02d}] ({t_start_sec:4.1f}s - {t_end_sec:4.1f}s) -> [OK] LOLOS SQA (SBP: {sbp_val:.1f}, DBP: {dbp_val:.1f})")
            else:
                reason_key = reason.split('(')[0].strip()
                rejections[reason_key] = rejections.get(reason_key, 0) + 1
                print(f" [Segmen #{total_segments:02d}] ({t_start_sec:4.1f}s - {t_end_sec:4.1f}s) -> [REJECT] DITOLAK: {reason}")

            start += stride_len

        rejected_segments = total_segments - passed_segments
        print("----------------------------------------------------------")
        print(f" [RINGKASAN SQA] Total Segmen 10s Dibuat  : {total_segments} segmen")
        print(f" [RINGKASAN SQA] Segmen Lolos (Valid)     : {passed_segments} segmen ({passed_segments/max(total_segments,1)*100:.1f}%)")
        print(f" [RINGKASAN SQA] Segmen Ditolak (Noise)    : {rejected_segments} segmen ({rejected_segments/max(total_segments,1)*100:.1f}%)")
        if rejections:
            print(" [RINGKASAN SQA] Detail Alasan Penolakan Segmen:")
            for reason_key, count in rejections.items():
                print(f"   * {reason_key:<32} : {count} segmen ({count/max(total_segments,1)*100:.1f}%)")
        print("==========================================================\n")

        # Susun string ringkasan alasan penolakan untuk popup GUI
        reasons_summary = ", ".join([f"{k} ({v} segmen)" for k, v in rejections.items()]) if rejections else "Noise/Artefak Sinyal"

        if passed_segments == 0:
            return {
                "sqa_passed": False,
                "passed_segments": 0,
                "rejected_segments": total_segments,
                "total_segments": total_segments,
                "sbp": 120.0,
                "dbp": 80.0,
                "rejections": rejections,
                "sqa_error": f"Tidak ada segmen 10s yang lolos SQA.\nAlasan Penolakan: {reasons_summary}.\nSilakan pastikan sensor terpasang dengan baik dan lakukan pengambilan data ulang."
            }

        # Mengambil rerata (atau median) dari seluruh segmen yang lolos SQA
        sbp_avg = float(np.clip(np.median(sbp_preds), 70.0, 220.0))
        dbp_avg = float(np.clip(np.median(dbp_preds), 40.0, 140.0))

        return {
            "sqa_passed": True,
            "passed_segments": passed_segments,
            "rejected_segments": rejected_segments,
            "total_segments": total_segments,
            "sbp": round(sbp_avg, 1),
            "dbp": round(dbp_avg, 1),
            "rejections": rejections
        }
