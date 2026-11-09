import re
import pandas as pd
from datetime import datetime
import check_signal_files as csf

df = pd.merge(csf.res_df, csf.df_tgt, on="subject_id")

def clean_name(n):
    c = str(n).strip().lower()
    c = re.sub(r"^\d+[\s_-]*", "", c)
    c = re.sub(r"^(bu|pak|kak)\s+", "", c)
    c = c.replace("olyvia", "olivia").replace("mey", "mei").replace("nasywa", "nashwa")
    return c.strip()

df["norm_name"] = df["original_name"].apply(clean_name)

# Reassign subject094 to Zana
df.loc[df["subject_id"] == "subject094", "norm_name"] = "zana"

norm_grouped = df.groupby("norm_name")
for name, grp in norm_grouped:
    unique_hashes = grp["ppg_hash"].unique()
    if len(unique_hashes) > 1:
        print(f"\nPerson: {name} (recordings count={len(unique_hashes)}, rows={len(grp)}):")
        for idx, r in grp.iterrows():
            print(f"  {r['subject_id']} | orig: {r['original_name']:15s} | time: {r['timestamp']} | hash: {r['ppg_hash'][:8]} | SBP: {r['SBP']}, DBP: {r['DBP']}, HR: {r['HR']}")
