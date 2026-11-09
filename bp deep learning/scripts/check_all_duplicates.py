import re
import difflib
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
    c = re.sub(r"^\d+[\s_-]*", "", name.strip())
    # remove prefixes like 'bu ', 'pak ', 'kak '
    c = re.sub(r"^(bu|pak|kak)\s+", "", c, flags=re.IGNORECASE)
    return c.strip().lower()

df["base_name"] = df["original_name"].apply(clean_name)

# Normalize common spelling variants
def normalize_spelling(name):
    # olyvia -> olivia, mey -> mei, nasywa -> nashwa
    s = name
    s = s.replace("olyvia", "olivia")
    s = s.replace("mey", "mei")
    s = s.replace("nasywa", "nashwa")
    return s

df["norm_name"] = df["base_name"].apply(normalize_spelling)

print("Unique raw original_names:", df["original_name"].nunique())
print("Unique base_names:", df["base_name"].nunique())
print("Unique norm_names:", df["norm_name"].nunique())

counts = df["norm_name"].value_counts()
print("\n--- Name counts (>= 2) ---")
print(counts[counts >= 2])
