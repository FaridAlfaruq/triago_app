import pandas as pd
from datetime import datetime

import check_signal_files as csf

df = pd.merge(csf.res_df, csf.df_tgt, on="subject_id")

# Let's inspect the suspicious pairs:
# 1. 016_Olyvia vs Olivia
# 2. 021_Nashwa vs Nasywa
# 3. Bu Nabila vs Nabila
# 4. Inas vs Zana (subject094)
# 5. Mey vs Mei

pairs_to_check = [
    ("subject015", "subject110", "Olyvia vs Olivia"),
    ("subject020", "subject108", "Nashwa vs Nasywa"),
    ("subject082", "subject106", "Bu Nabila vs Nabila"),
    ("subject056", "subject094", "Zana vs Inas(094)"),
    ("subject058", "subject069", "Mey vs Mei"),
]

for s1, s2, desc in pairs_to_check:
    r1 = df[df["subject_id"] == s1].iloc[0]
    r2 = df[df["subject_id"] == s2].iloc[0]
    print(f"=== {desc} ===")
    print(f"  {s1}: name={r1['original_name']}, time={r1['timestamp']}, SBP={r1['SBP']}, DBP={r1['DBP']}, HR={r1['HR']}, ppg_hash={r1['ppg_hash'][:8]}")
    print(f"  {s2}: name={r2['original_name']}, time={r2['timestamp']}, SBP={r2['SBP']}, DBP={r2['DBP']}, HR={r2['HR']}, ppg_hash={r2['ppg_hash'][:8]}")
