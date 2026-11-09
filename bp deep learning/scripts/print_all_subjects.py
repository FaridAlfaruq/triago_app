import re
import pandas as pd
from pathlib import Path

mapping_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\subject_mapping.csv")
target_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\target.csv")

df_map = pd.read_csv(mapping_path)
df_tgt = pd.read_csv(target_path)
df = pd.merge(df_map, df_tgt, on="subject_id")

with open(r"scratch\gemastik\all_subjects_dump.txt", "w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        f.write(f"{row['subject_id']:10s} | map: {str(row['original_name']):20s} | tgt: {str(row['subject_name']):20s} | time: {row['timestamp']} | SBP: {row['SBP']} DBP: {row['DBP']} HR: {row['HR']}\n")
print(f"Written {len(df)} lines to all_subjects_dump.txt")
