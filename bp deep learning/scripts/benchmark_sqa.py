import os
import time
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, butter, sosfiltfilt, iirnotch, filtfilt, welch, find_peaks, correlate

# Let's import the exact functions from dump or define them cleanly
def sanitize_raw_record(record):
    cleaned_rec = record.copy()
    for sig_type in ['ecg', 'ppg']:
        sig = record[sig_type]
        t = record[f'{sig_type}_time']
        valid_mask = ~(np.isnan(sig) | np.isinf(sig) | np.isnan(t) | np.isinf(t))
        sig, t = sig[valid_mask], t[valid_mask]
        if len(sig) == 0:
            return None
        df_temp = pd.DataFrame({'time': t, 'sig': sig})
        df_temp = df_temp.drop_duplicates(subset=['time']).sort_values(by='time')
        t_clean = df_temp['time'].values
        sig_clean = df_temp['sig'].values
        dt = np.diff(t_clean)
        if len(dt) == 0 or np.any(dt <= 0):
            return None
        median_dt = np.median(dt)
        std_dt = np.std(dt)
        fs_estimated = 1.0 / median_dt
        cleaned_rec[sig_type] = sig_clean
        cleaned_rec[f'{sig_type}_time'] = t_clean
        cleaned_rec[f'fs_{sig_type}'] = fs_estimated
        cleaned_rec[f'jitter_{sig_type}'] = std_dt
    return cleaned_rec

def align_and_resample_record(record, fs_target=125):
    t_ecg, sig_ecg = record['ecg_time'], record['ecg']
    t_ppg, sig_ppg = record['ppg_time'], record['ppg']
    t_start = max(t_ecg[0], t_ppg[0])
    t_end = min(t_ecg[-1], t_ppg[-1])
    duration = t_end - t_start
    if duration < 10.0:
        return None
    mask_ecg = (t_ecg >= t_start) & (t_ecg <= t_end)
    mask_ppg = (t_ppg >= t_start) & (t_ppg <= t_end)
    sig_ecg_cut = sig_ecg[mask_ecg]
    sig_ppg_cut = sig_ppg[mask_ppg]
    fs_ecg_in = record['fs_ecg']
    fs_ppg_in = record['fs_ppg']
    up_ecg = int(round(fs_target * 1000))
    down_ecg = int(round(fs_ecg_in * 1000))
    ecg_125 = resample_poly(sig_ecg_cut, up=up_ecg, down=down_ecg)
    up_ppg = int(round(fs_target * 1000))
    down_ppg = int(round(fs_ppg_in * 1000))
    ppg_125 = resample_poly(sig_ppg_cut, up=up_ppg, down=down_ppg)
    min_len = min(len(ecg_125), len(ppg_125))
    ecg_125 = ecg_125[:min_len]
    ppg_125 = ppg_125[:min_len]
    t_common = np.linspace(0, min_len / fs_target, min_len)
    return {
        'recording_id': record['recording_id'],
        'patient_id': record['patient_id'],
        'ecg': ecg_125,
        'ppg': ppg_125,
        'time': t_common,
        'fs': fs_target,
        'duration_sec': min_len / fs_target,
        'sbp': record['sbp'],
        'dbp': record['dbp']
    }

def notch_filter_zero_phase(sig, freq=50.0, fs=125.0, Q=30.0):
    nyq = 0.5 * fs
    if freq >= nyq:
        return sig
    b, a = iirnotch(w0=freq, Q=Q, fs=fs)
    return filtfilt(b, a, sig)

def bandpass_filter_zero_phase(sig, lowcut, highcut, fs=125.0, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, sig)

def preprocess_ecg_clean(raw_ecg, fs=125.0):
    ecg_notch = notch_filter_zero_phase(raw_ecg, freq=50.0, fs=fs, Q=30.0)
    return bandpass_filter_zero_phase(ecg_notch, lowcut=0.5, highcut=40.0, fs=fs, order=4)

def preprocess_ppg_clean(raw_ppg, fs=125.0, invert_ppg=True):
    ppg_sig = -raw_ppg if invert_ppg else raw_ppg
    ppg_notch = notch_filter_zero_phase(ppg_sig, freq=50.0, fs=fs, Q=30.0)
    return bandpass_filter_zero_phase(ppg_notch, lowcut=0.5, highcut=15.0, fs=fs, order=3)

from scipy.stats import skew, kurtosis
def compute_ppg_skewness(ppg_seg):
    return skew(ppg_seg)

def compute_ppg_kurtosis(ppg_seg):
    return kurtosis(ppg_seg, fisher=False)

def compute_spectral_snr(sig_seg, fs=125.0, f_low=0.5, f_high=8.0):
    freqs, psd = welch(sig_seg, fs=fs, nperseg=int(2 * fs))
    total_power = np.sum(psd) + 1e-8
    band_power = np.sum(psd[(freqs >= f_low) & (freqs <= f_high)])
    return band_power / total_power

def evaluate_segment_sqa_v2(ecg_seg, ppg_seg, fs=125.0, hr_diff_max=5.0):
    if np.isnan(ecg_seg).any() or np.isnan(ppg_seg).any():
        return False, "NaN/Inf Found"
    if np.std(ecg_seg) < 1e-4 or np.std(ppg_seg) < 1e-4:
        return False, "Flatline Signal"
    ecg_peaks, _ = find_peaks(ecg_seg, distance=int(0.4 * fs), prominence=0.4 * np.std(ecg_seg))
    if len(ecg_peaks) < 4:
        return False, f"Puncak ECG Terlalu Sedikit ({len(ecg_peaks)})"
    ppg_peaks, _ = find_peaks(ppg_seg, distance=int(0.4 * fs), prominence=0.25 * np.std(ppg_seg))
    if len(ppg_peaks) < 4:
        return False, f"Puncak PPG Terlalu Sedikit ({len(ppg_peaks)})"
    hr_ecg = 60.0 / np.mean(np.diff(ecg_peaks) / fs)
    hr_ppg = 60.0 / np.mean(np.diff(ppg_peaks) / fs)
    if not (40.0 <= hr_ecg <= 180.0) or not (40.0 <= hr_ppg <= 180.0):
        return False, f"HR Luar Batas (ECG: {hr_ecg:.1f}, PPG: {hr_ppg:.1f})"
    if abs(hr_ecg - hr_ppg) > hr_diff_max:
        return False, f"HR Mismatch (ECG: {hr_ecg:.1f} vs PPG: {hr_ppg:.1f})"
    s_sqa = compute_ppg_skewness(ppg_seg)
    k_sqa = compute_ppg_kurtosis(ppg_seg)
    if s_sqa <= -0.1:
        return False, f"PPG Skewness Invalid ({s_sqa:.2f})"
    if k_sqa < 1.8:
        return False, f"PPG Kurtosis Invalid ({k_sqa:.2f})"
    snr_ratio = compute_spectral_snr(ppg_seg, fs=fs, f_low=0.5, f_high=8.0)
    if snr_ratio < 0.50:
        return False, f"PPG Low Spectral SNR ({snr_ratio*100:.1f}%)"
    return True, "VALID_SQA"

t0 = time.time()
# test loading 5 subjects
BASE_PATH = r'G:\My Drive\TriaGO_Project\Dataset'
for i in range(1, 6):
    sid = f"subject{i:03d}"
    df_e = pd.read_csv(os.path.join(BASE_PATH, 'ECG', f"{sid}_ecg.csv"))
    df_p = pd.read_csv(os.path.join(BASE_PATH, 'PPG', f"{sid}_ppg.csv"))
    rec = {
        'recording_id': sid, 'patient_id': sid,
        'ecg': df_e['ECG_Raw (V)'].values, 'ecg_time': df_e['Time (s)'].values,
        'ppg': df_p['PPG_IR'].values, 'ppg_time': df_p['Time (s)'].values,
        'sbp': 120.0, 'dbp': 80.0
    }
    srec = sanitize_raw_record(rec)
    arec = align_and_resample_record(srec, 125)
    ecg_c = preprocess_ecg_clean(arec['ecg'], 125)
    ppg_c = preprocess_ppg_clean(arec['ppg'], 125)
    # segment
    win = 10 * 125
    stride = 2 * 125
    st = 0
    val_cnt = 0
    while st + win <= len(ecg_c):
        ok, r = evaluate_segment_sqa_v2(ecg_c[st:st+win], ppg_c[st:st+win], 125)
        if ok:
            val_cnt += 1
        st += stride
    print(f"{sid}: len={len(ecg_c)}, valid segments={val_cnt}")
print(f"5 subjects processed in {time.time()-t0:.2f}s")
