import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, butter, sosfiltfilt, iirnotch, filtfilt, welch, find_peaks, correlate
from sklearn.model_selection import GroupShuffleSplit

# Direct implementation of bp.ipynb pipeline
BASE_PATH = r'G:\My Drive\TriaGO_Project\Dataset'
ECG_DIR = os.path.join(BASE_PATH, 'ECG')
PPG_DIR = os.path.join(BASE_PATH, 'PPG')
METADATA_DIR = os.path.join(BASE_PATH, 'metadata')
TARGET_PATH = os.path.join(METADATA_DIR, 'target.csv')
MAPPING_PATH = os.path.join(METADATA_DIR, 'subject_mapping.csv')

gt_df = pd.read_csv(TARGET_PATH)
map_df = pd.read_csv(MAPPING_PATH)

# Cell 3 logic
if 'subject_id' in map_df.columns and 'patient_id' in map_df.columns:
    gt_df = gt_df.merge(map_df[['subject_id', 'patient_id']], on='subject_id', how='left')
    gt_df['true_patient_id'] = gt_df['patient_id'].fillna(gt_df['subject_id'])
else:
    gt_df['true_patient_id'] = gt_df['subject_id']

print(f"Loaded gt_df with {len(gt_df)} rows. true_patient_id unique: {gt_df['true_patient_id'].nunique()}")

# Quick check of which subjects exist and are valid
records_info = []
for idx, row in gt_df.iterrows():
    sub_id = row['subject_id']
    true_pid = row['true_patient_id']
    sbp_val = row['SBP']
    dbp_val = row['DBP']

    if pd.isna(sbp_val) or pd.isna(dbp_val) or sbp_val <= 0 or dbp_val <= 0:
        continue

    ecg_path = os.path.join(ECG_DIR, f"{sub_id}_ecg.csv")
    ppg_path = os.path.join(PPG_DIR, f"{sub_id}_ppg.csv")

    if not (os.path.exists(ecg_path) and os.path.exists(ppg_path)):
        continue

    records_info.append({
        'subject_id': sub_id,
        'true_patient_id': true_pid,
        'sbp': float(sbp_val),
        'dbp': float(dbp_val)
    })

print(f"Subjects passing Stage 1 (file exist & valid target): {len(records_info)}")
