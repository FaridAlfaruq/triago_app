import pandas as pd
from pathlib import Path

# Load results from check_signal_files
import check_signal_files as csf

res_df = csf.res_df
df_tgt = csf.df_tgt
df = pd.merge(res_df, df_tgt[["subject_id", "SBP", "DBP", "HR", "RR", "SpO2", "Glucose"]], on="subject_id")

grouped = df.groupby("ppg_hash")
print(f"Total groups sharing PPG hash: {sum(len(g) > 1 for _, g in grouped)}")

for h, g in grouped:
    if len(g) > 1:
        sids = list(g["subject_id"])
        names = list(g["original_name"])
        times = list(g["timestamp"])
        sbps = list(g["SBP"])
        dbps = list(g["DBP"])
        hrs = list(g["HR"])
        print(f"Hash {h[:8]} (count={len(g)}):")
        for i in range(len(g)):
            print(f"   {sids[i]} | name: {names[i]} | time: {times[i]} | SBP: {sbps[i]}, DBP: {dbps[i]}, HR: {hrs[i]}")
