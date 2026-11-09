# -*- coding: utf-8 -*-
"""Triage_XGBoost Training & ONNX Export Script (Tersinkronisasi dengan Notebook & GUI).

Modul ini melatih model XGBoost dengan 11 Fitur Hemodinamik & Vital (termasuk Pain Score)
menggunakan hasil hyperparameter tuning Optuna (Depth 6, LR 0.021, n_estimators 400, sample weighting)
serta mengekspor model ke ONNX untuk inferensi GUI TriaGO.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import onnxruntime as rt
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

warnings.filterwarnings('ignore')

MODEL_DIR = Path(__file__).resolve().parent
ONNX_FILENAME = MODEL_DIR / "triage_xgboost_model.onnx"

if "KAGGLE_API_TOKEN" not in os.environ:
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_fc19df71ba18eaa47482e266ccf79521"

print("=" * 70)
print("[INFO] MEMULAI PELATIHAN TUNED XGBOOST (11 FITUR DENGAN PAIN SCORE) & ONNX")
print("=" * 70)

# -------------------------------------------------------------------------
# 1. LOAD DATASET (Kaggle / Local / Fallback)
# -------------------------------------------------------------------------
def load_triage_dataset():
    local_csv = MODEL_DIR / "train.csv"
    if local_csv.exists():
        print(f"[INFO] Membaca dataset dari file lokal: {local_csv}")
        return pd.read_csv(local_csv)

    try:
        import kagglehub
        path = kagglehub.competition_download('triagegeist')
        csv_path = Path(path) / "train.csv"
        if csv_path.exists():
            print(f"[OK] Dataset berhasil diunduh via kagglehub: {csv_path}")
            return pd.read_csv(csv_path)
    except Exception as e:
        print(f"[INFO] Kagglehub fallback ({e}). Menggunakan dataset sampel sintetis...")

    # Fallback Data Generator
    np.random.seed(42)
    n_samples = 8000
    gcs = np.random.choice([15, 14, 13, 12, 10, 8, 5, 3], size=n_samples, p=[0.70, 0.10, 0.08, 0.04, 0.03, 0.02, 0.02, 0.01])
    spo2 = np.random.uniform(85.0, 100.0, size=n_samples)
    rr = np.random.uniform(10.0, 32.0, size=n_samples)
    hr = np.random.uniform(50.0, 130.0, size=n_samples)
    sbp = np.random.uniform(80.0, 170.0, size=n_samples)
    dbp = sbp * np.random.uniform(0.55, 0.75, size=n_samples)
    temp = np.random.uniform(35.5, 39.5, size=n_samples)
    pain = np.random.choice(range(1, 11), size=n_samples)

    acuity = []
    for i in range(n_samples):
        if gcs[i] <= 8 or spo2[i] < 90.0 or sbp[i] < 85.0:
            acuity.append(1)  # Resusitasi (Level 1)
        elif gcs[i] in [13, 14] or rr[i] > 26.0 or spo2[i] <= 93.0 or sbp[i] > 160.0 or pain[i] >= 8:
            acuity.append(2)  # Darurat (Level 2)
        elif hr[i] > 110.0 or temp[i] > 38.5 or (rr[i] > 24.0) or (rr[i] > 20.0 and (hr[i] > 90.0 or spo2[i] < 96.0)):
            acuity.append(3)  # Darurat (Level 3)
        elif hr[i] > 85.0 or rr[i] > 20.0:
            acuity.append(4)  # Non-Darurat (Level 4)
        else:
            acuity.append(5)  # Non-Darurat (Level 5)

    return pd.DataFrame({
        'temperature_c': temp, 'spo2': spo2, 'respiratory_rate': rr,
        'heart_rate': hr, 'systolic_bp': sbp, 'diastolic_bp': dbp,
        'gcs_total': gcs, 'pain_score': pain, 'triage_acuity': acuity
    })


df_raw = load_triage_dataset()

# -------------------------------------------------------------------------
# 2. PEMETAAN LABEL & PRA-PEMROSESAN (Identik Notebook)
# -------------------------------------------------------------------------
used_cols = [
    'temperature_c', 'spo2', 'respiratory_rate',
    'heart_rate', 'systolic_bp', 'diastolic_bp', 'gcs_total', 'pain_score'
]
target_col = 'triage_acuity'

df = df_raw[used_cols + [target_col]].copy()
df['triage_acuity'] = df['triage_acuity'] - 1
df['pain_score'] = df['pain_score'].replace(-1, np.nan)
mapping_triage = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
df['target'] = df['triage_acuity'].map(mapping_triage)

df = df.dropna(subset=used_cols).reset_index(drop=True)
df[used_cols] = df[used_cols].astype('float32')
df['target'] = df['target'].astype('int8')

# -------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (11 Fitur)
# -------------------------------------------------------------------------
def bp_based_feature(data_in):
    data = data_in.copy()
    data['shock_index'] = data['heart_rate'] / np.maximum(data['systolic_bp'], 1.0)
    data['pulse_pressure'] = data['systolic_bp'] / np.maximum(data['diastolic_bp'], 1.0)
    data['MAP'] = data['diastolic_bp'] + (data['pulse_pressure'] / 3.0)
    return data

X = df[used_cols].copy()
y = df['target'].copy()

X_fe = bp_based_feature(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_fe, y, test_size=0.10, random_state=42, stratify=y
)

# -------------------------------------------------------------------------
# 4. TUNED HYPERPARAMETERS OPTUNA (Triage_XGBoost)
# -------------------------------------------------------------------------
tuned_params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'tree_method': 'hist',
    'random_state': 42,
    'n_estimators': 400,
    'max_depth': 6,
    'learning_rate': 0.020982541647416003,
    'subsample': 0.6564711887550883,
    'colsample_bytree': 0.6931045329589705,
    'min_child_weight': 4.959158170141056,
    'gamma': 0.9320543375631457,
    'reg_alpha': 0.13378122792956854,
    'reg_lambda': 4.157053539666323,
    'n_jobs': -1,
}

weights = {0: 2.6981530587562035, 1: 1.0, 2: 1.3126068634020642}
sample_weights = compute_sample_weight(class_weight=weights, y=y_train)

print("\n[INFO] Melatih Model XGBoost dengan Hyperparameter Tuned...")
final_model = XGBClassifier(**tuned_params)
final_model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = final_model.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)
val_f1 = f1_score(y_val, y_pred, average='macro')

print("\n" + "=" * 70)
print(f"[INFO] HASIL EVALUASI MODEL (AKURASI VALIDASI: {val_acc*100:.2f}%, MACRO F1: {val_f1*100:.2f}%)")
print("=" * 70)
print(classification_report(y_val, y_pred, target_names=['Resusitasi (0)', 'Darurat (1)', 'Non-Darurat (2)'], digits=4))

# -------------------------------------------------------------------------
# 5. EKSPOR ARTEFAK ONNX
# -------------------------------------------------------------------------
print("\n[INFO] Mengekspor Model ONNX 11-Fitur (Termasuk Pain Score)...")
n_features = X_train.shape[1]
initial_type = [('float_input', FloatTensorType([None, n_features]))]

booster = final_model.get_booster()
original_feature_names = booster.feature_names
booster.feature_names = None

try:
    onnx_model = convert_xgboost(final_model, initial_types=initial_type, target_opset=13)
    with open(ONNX_FILENAME, "wb") as f:
        f.write(onnx_model.SerializeToString())
    size_kb = os.path.getsize(ONNX_FILENAME) / 1024
    print(f"[OK] BERHASIL: File ONNX Tersimpan di '{ONNX_FILENAME}' ({size_kb:.2f} KB)")
finally:
    booster.feature_names = original_feature_names

# -------------------------------------------------------------------------
# 6. VERIFIKASI UJI COBA INFERENSI ONNX
# -------------------------------------------------------------------------
print("\n=== UJI VERIFIKASI ONNX (SAMPLE VITAL + PAIN SCORE) ===")
sample_patient = pd.DataFrame([{
    'temperature_c': 36.5,
    'spo2': 98.0,
    'respiratory_rate': 18.0,
    'heart_rate': 75.0,
    'systolic_bp': 120.0,
    'diastolic_bp': 80.0,
    'gcs_total': 15.0,
    'pain_score': 3.0,
}])
sample_fe = bp_based_feature(sample_patient)
sample_onnx = sample_fe.values.astype(np.float32)

sess = rt.InferenceSession(str(ONNX_FILENAME), providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
onnx_outputs = sess.run(None, {input_name: sample_onnx})

raw_prob = onnx_outputs[1] if len(onnx_outputs) > 1 else onnx_outputs[0]
if isinstance(raw_prob, list) and isinstance(raw_prob[0], dict):
    prob_vec = np.array(list(raw_prob[0].values()))
else:
    prob_vec = np.squeeze(np.array(raw_prob))

pred_class = int(np.argmax(prob_vec))
labels = {0: 'RESUSITASI', 1: 'DARURAT', 2: 'NON-DARURAT'}

print(f"Hasil Klasifikasi ONNX: [{labels[pred_class]}] (Confidence: {prob_vec[pred_class]:.4f})")
print("Probabilitas per Kelas [Resusitasi, Darurat, Non-Darurat]:", prob_vec.round(4))
