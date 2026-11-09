import re
import pandas as pd
from pathlib import Path

mapping_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\subject_mapping.csv")
target_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\target.csv")

df_map = pd.read_csv(mapping_path)
df_tgt = pd.read_csv(target_path)
df = pd.merge(df_map, df_tgt, on="subject_id")

def clean(n):
    c = re.sub(r"^\d+[\s_-]*", "", str(n).strip())
    c = re.sub(r"^(bu|pak|kak)\s+", "", c, flags=re.IGNORECASE)
    return c.strip().lower()

df["cname"] = df["original_name"].apply(clean)

set1 = df[df["subject_id"] <= "subject044"]
set2 = df[df["subject_id"] > "subject044"]

print("Overlap between 001..044 and 045..123 by exact cleaned name:")
common = set(set1["cname"]).intersection(set(set2["cname"]))
print(common)

# Check fuzzy overlap
import difflib
for n1 in set(set1["cname"]):
    for n2 in set(set2["cname"]):
        sim = difflib.SequenceMatcher(None, n1, n2).ratio()
        if sim >= 0.8:
            print(f"Fuzzy match (sim={sim:.2f}): '{n1}' (001..044) vs '{n2}' (045..123)")
