import re
import pandas as pd
from pathlib import Path

mapping_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\subject_mapping.csv")
target_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\target.csv")

df_map = pd.read_csv(mapping_path)
df_tgt = pd.read_csv(target_path)

df = pd.merge(df_map, df_tgt, on="subject_id")

def clean_name(name):
    if not isinstance(name, str):
        return ""
    # Strip leading digits and underscores or spaces, e.g. "002_Farid" -> "Farid"
    # Also strip trailing/leading whitespace and lowercase for comparison
    c = re.sub(r"^\d+[\s_-]*", "", name.strip())
    return c.strip().lower()

df["cleaned_map_name"] = df["original_name"].apply(clean_name)
df["cleaned_tgt_name"] = df["subject_name"].apply(clean_name)

print("Merged shape:", df.shape)
print("Names where cleaned_map_name != cleaned_tgt_name:")
mismatch = df[df["cleaned_map_name"] != df["cleaned_tgt_name"]]
print(mismatch[["subject_id", "original_name", "subject_name"]])

# Check value counts of cleaned names
counts = df["cleaned_map_name"].value_counts()
duplicates = counts[counts > 1]
print("\nDuplicate cleaned names in mapping:")
print(duplicates)
