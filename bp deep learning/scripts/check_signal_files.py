import os
import hashlib
import pandas as pd
from pathlib import Path

mapping_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\subject_mapping.csv")
target_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\target.csv")
ppg_dir = Path(r"G:\My Drive\TriaGO_Project\Dataset\PPG")
ecg_dir = Path(r"G:\My Drive\TriaGO_Project\Dataset\ECG")

df_map = pd.read_csv(mapping_path)
df_tgt = pd.read_csv(target_path)
df = pd.merge(df_map, df_tgt, on="subject_id")

def file_hash(path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

results = []
for idx, row in df.iterrows():
    sid = row["subject_id"]
    ppg_f = ppg_dir / f"{sid}_ppg.csv"
    ecg_f = ecg_dir / f"{sid}_ecg.csv"
    results.append({
        "subject_id": sid,
        "original_name": row["original_name"],
        "timestamp": row["timestamp"],
        "ppg_exists": ppg_f.exists(),
        "ecg_exists": ecg_f.exists(),
        "ppg_size": ppg_f.stat().st_size if ppg_f.exists() else 0,
        "ecg_size": ecg_f.stat().st_size if ecg_f.exists() else 0,
        "ppg_hash": file_hash(ppg_f),
        "ecg_hash": file_hash(ecg_f)
    })

res_df = pd.DataFrame(results)
print("Total rows:", len(res_df))
print("Missing PPG:", len(res_df[~res_df["ppg_exists"]]))
print("Missing ECG:", len(res_df[~res_df["ecg_exists"]]))

# Check duplicate hashes
ppg_dups = res_df[res_df.duplicated(subset=["ppg_hash"], keep=False)]
print(f"Rows sharing PPG hashes: {len(ppg_dups)} across {ppg_dups['ppg_hash'].nunique()} unique hashes")

ecg_dups = res_df[res_df.duplicated(subset=["ecg_hash"], keep=False)]
print(f"Rows sharing ECG hashes: {len(ecg_dups)} across {ecg_dups['ecg_hash'].nunique()} unique hashes")
