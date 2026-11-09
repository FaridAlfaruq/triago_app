import re
import pandas as pd
from pathlib import Path
import hashlib

mapping_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\subject_mapping.csv")
target_path = Path(r"G:\My Drive\TriaGO_Project\Dataset\metadata\target.csv")
ppg_dir = Path(r"G:\My Drive\TriaGO_Project\Dataset\PPG")
ecg_dir = Path(r"G:\My Drive\TriaGO_Project\Dataset\ECG")

df_map = pd.read_csv(mapping_path)
df_tgt = pd.read_csv(target_path)
df = pd.merge(df_map, df_tgt, on="subject_id")

def get_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8192):
            h.update(c)
    return h.hexdigest()

df["ppg_hash"] = [get_hash(ppg_dir / f"{s}_ppg.csv") for s in df["subject_id"]]
df["ecg_hash"] = [get_hash(ecg_dir / f"{s}_ecg.csv") for s in df["subject_id"]]

def clean_name(n):
    c = re.sub(r"^\d+[\s_-]*", "", str(n).strip())
    c = re.sub(r"^(bu|pak|kak)\s+", "", c, flags=re.IGNORECASE)
    return c.strip().lower()

df["clean_name"] = df["original_name"].apply(clean_name)

# Group by ppg_hash first
hash_groups = {}
for idx, r in df.iterrows():
    h = r["ppg_hash"]
    if h not in hash_groups:
        hash_groups[h] = []
    hash_groups[h].append(r["subject_id"])

multi_hash_groups = {h: sids for h, sids in hash_groups.items() if len(sids) > 1}
print(f"Total multi-hash groups: {len(multi_hash_groups)}")
all_dups_in_hash = [sid for sids in multi_hash_groups.values() for sid in sids]
print(f"Total rows in multi-hash groups: {len(all_dups_in_hash)}")

# Now examine remaining subjects not in multi-hash groups
remaining = df[~df["subject_id"].isin(all_dups_in_hash)]
print(f"Remaining subjects: {len(remaining)}")

# Check if any remaining subjects share names or similar names
rem_name_counts = remaining["clean_name"].value_counts()
print("Remaining names with count > 1:")
print(rem_name_counts[rem_name_counts > 1])

# Check across all 123 subjects: if we group by human identity:
# Let's inspect the names in remaining:
for n, c in rem_name_counts.items():
    if c > 1:
        print(remaining[remaining["clean_name"] == n][["subject_id", "original_name", "timestamp", "SBP", "DBP", "HR"]])
