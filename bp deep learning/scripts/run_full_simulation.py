import os
import time
import json
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, butter, sosfiltfilt, iirnotch, filtfilt, welch, find_peaks, correlate
from sklearn.model_selection import GroupShuffleSplit

BASE_PATH = r'G:\My Drive\TriaGO_Project\Dataset'
ECG_DIR = os.path.join(BASE_PATH, 'ECG')
PPG_DIR = os.path.join(BASE_PATH, 'PPG')
METADATA_DIR = os.path.join(BASE_PATH, 'metadata')
TARGET_PATH = os.path.join(METADATA_DIR, 'target.csv')
MAPPING_PATH = os.path.join(METADATA_DIR, 'subject_mapping.csv')

gt_df = pd.read_csv(TARGET_PATH)
map_df = pd.read_csv(MAPPING_PATH)

# Replicate bp.ipynb Cell 3
if 'subject_id' in map_df.columns and 'patient_id' in map_df.columns:
    gt_df = gt_df.merge(map_df[['subject_id', 'patient_id']], on='subject_id', how='left')
    gt_df['true_patient_id'] = gt_df['patient_id'].fillna(gt_df['subject_id'])
else:
    gt_df['true_patient_id'] = gt_df['subject_id']

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

# Load and process all records
print("Processing records...")
dataset_segments = []
window_len = int(10 * 125)
stride_len = int(2 * 125)

for idx, row in gt_df.iterrows():
    sid = row['subject_id']
    true_pid = row['true_patient_id']
    sbp_val = row['SBP']
    dbp_val = row['DBP']

    if pd.isna(sbp_val) or pd.isna(dbp_val) or sbp_val <= 0 or dbp_val <= 0:
        continue

    ecg_path = os.path.join(ECG_DIR, f"{sid}_ecg.csv")
    ppg_path = os.path.join(PPG_DIR, f"{sid}_ppg.csv")
    if not (os.path.exists(ecg_path) and os.path.exists(ppg_path)):
        continue

    df_ecg = pd.read_csv(ecg_path)
    df_ppg = pd.read_csv(ppg_path)
    if df_ecg.empty or df_ppg.empty:
        continue

    rec = {
        'recording_id': sid, 'patient_id': true_pid,
        'ecg': df_ecg['ECG_Raw (V)'].values, 'ecg_time': df_ecg['Time (s)'].values,
        'ppg': df_ppg['PPG_IR'].values, 'ppg_time': df_ppg['Time (s)'].values,
        'sbp': float(sbp_val), 'dbp': float(dbp_val)
    }

    s_rec = sanitize_raw_record(rec)
    if s_rec is None:
        continue
    a_rec = align_and_resample_record(s_rec, fs_target=125)
    if a_rec is None:
        continue

    ecg_clean = preprocess_ecg_clean(a_rec['ecg'], fs=125)
    ppg_clean = preprocess_ppg_clean(a_rec['ppg'], fs=125, invert_ppg=True)

    start = 0
    while start + window_len <= len(ecg_clean):
        ecg_seg = ecg_clean[start:start+window_len]
        ppg_seg = ppg_clean[start:start+window_len]
        is_sqa_valid, _ = evaluate_segment_sqa_v2(ecg_seg, ppg_seg, fs=125.0)
        if is_sqa_valid:
            dataset_segments.append({
                'recording_id': sid,
                'patient_id': true_pid,
                'sbp': a_rec['sbp'],
                'dbp': a_rec['dbp']
            })
        start += stride_len

print(f"Total segments passing SQA: {len(dataset_segments)}")
seg_df = pd.DataFrame(dataset_segments)
print(f"Unique subjects with valid segments: {seg_df['patient_id'].nunique()}")

# Outlier filtering as in Cell 17:
valid_mask = (seg_df['sbp'] >= 70) & (seg_df['sbp'] <= 200) & (seg_df['dbp'] >= 40) & (seg_df['dbp'] <= 120)
seg_clean = seg_df[valid_mask].copy()
print(f"Segments after outlier filtering: {len(seg_clean)}")
print(f"Unique subjects after outlier filtering: {seg_clean['patient_id'].nunique()}")

# Now run find_balanced_subject_split
X_dummy = np.zeros((len(seg_clean), 1))
y_sbp = seg_clean['sbp'].values
y_dbp = seg_clean['dbp'].values
groups = seg_clean['patient_id'].values

best_score = float('inf')
best_splits = None
best_seed = 42
test_size = 0.15
val_size = 0.15
adjusted_val_size = val_size / (1.0 - test_size)

for seed in range(100):
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(X_dummy, y_sbp, groups))

    X_tv, y_sbp_tv, y_dbp_tv, groups_tv = X_dummy[train_val_idx], y_sbp[train_val_idx], y_dbp[train_val_idx], groups[train_val_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=seed)
    train_rel_idx, val_rel_idx = next(gss_val.split(X_tv, y_sbp_tv, groups_tv))

    tr_idx = train_val_idx[train_rel_idx]
    va_idx = train_val_idx[val_rel_idx]
    te_idx = test_idx

    diff_sbp_val = abs(np.mean(y_sbp[tr_idx]) - np.mean(y_sbp[va_idx]))
    diff_sbp_test = abs(np.mean(y_sbp[tr_idx]) - np.mean(y_sbp[te_idx]))
    diff_dbp_val = abs(np.mean(y_dbp[tr_idx]) - np.mean(y_dbp[va_idx]))
    diff_dbp_test = abs(np.mean(y_dbp[tr_idx]) - np.mean(y_dbp[te_idx]))

    total_imbalance_score = diff_sbp_val + diff_sbp_test + diff_dbp_val + diff_dbp_test

    if total_imbalance_score < best_score:
        best_score = total_imbalance_score
        best_seed = seed
        best_splits = {
            'train_sids': set(groups[tr_idx]),
            'val_sids': set(groups[va_idx]),
            'test_sids': set(groups[te_idx])
        }

print(f"Best seed: {best_seed}, best score: {best_score:.4f}")
print(f"Train subjects: {len(best_splits['train_sids'])}")
print(f"Val subjects: {len(best_splits['val_sids'])}")
print(f"Test subjects: {len(best_splits['test_sids'])}")

# Save split results to inspect
with open(r"scratch\gemastik\split_simulation.json", "w") as f:
    json.dump({
        'best_seed': best_seed,
        'best_score': best_score,
        'train_sids': sorted(list(best_splits['train_sids'])),
        'val_sids': sorted(list(best_splits['val_sids'])),
        'test_sids': sorted(list(best_splits['test_sids']))
    }, f, indent=2)
