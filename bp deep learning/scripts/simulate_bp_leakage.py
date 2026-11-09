import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Let's inspect bp.ipynb's splitting function:
# Cell 17 in bp.ipynb:
# sanitize_target_outliers(X, y_sbp, y_dbp, groups, sbp_range=(70, 200), dbp_range=(40, 120))
# find_balanced_subject_split(X, y_sbp, y_dbp, groups, test_size=0.15, val_size=0.15, max_searches=100)

mapping_path = r"G:\My Drive\TriaGO_Project\Dataset\metadata\subject_mapping.csv"
target_path = r"G:\My Drive\TriaGO_Project\Dataset\metadata\target.csv"
ecg_dir = r"G:\My Drive\TriaGO_Project\Dataset\ECG"
ppg_dir = r"G:\My Drive\TriaGO_Project\Dataset\PPG"

gt_df = pd.read_csv(target_path)
map_df = pd.read_csv(mapping_path)

# Look at Cell 3 logic:
if 'subject_id' in map_df.columns and 'patient_id' in map_df.columns:
    gt_df = gt_df.merge(map_df[['subject_id', 'patient_id']], on='subject_id', how='left')
    gt_df['true_patient_id'] = gt_df['patient_id'].fillna(gt_df['subject_id'])
else:
    # This branch is taken because 'patient_id' is NOT in map_df!
    gt_df['true_patient_id'] = gt_df['subject_id']

print("true_patient_id unique count:", gt_df['true_patient_id'].nunique())
print("Total rows:", len(gt_df))
