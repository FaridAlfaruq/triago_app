import pandas as pd
from pathlib import Path

mapping_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\subject_mapping.csv")
target_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\target.csv")

df_map = pd.read_csv(mapping_path)
df_tgt = pd.read_csv(target_path)

print(f"Mapping shape: {df_map.shape}, columns: {list(df_map.columns)}")
print(f"Target shape: {df_tgt.shape}, columns: {list(df_tgt.columns)}")
print(df_map.head(10))
print(df_map.tail(10))
