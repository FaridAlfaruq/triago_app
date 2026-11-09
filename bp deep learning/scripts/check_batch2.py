import pandas as pd
import check_signal_files as csf

df = pd.merge(csf.res_df, csf.df_tgt, on="subject_id")

# Find all subjects in 77..123
batch2 = df[df["subject_id"] >= "subject077"]

print(f"Total in subject077..123: {len(batch2)}")

# Categorize each row in batch2:
# Is it a duplicate of an earlier subject (< subject077)?
copied = []
unique_new = []

for idx, r in batch2.iterrows():
    # find if ppg_hash exists in earlier subjects
    earlier = df[(df["subject_id"] < "subject077") & (df["ppg_hash"] == r["ppg_hash"])]
    if len(earlier) > 0:
        match = earlier.iloc[0]
        copied.append((r["subject_id"], r["original_name"], match["subject_id"], match["original_name"], r["timestamp"] == match["timestamp"]))
    else:
        unique_new.append((r["subject_id"], r["original_name"], r["timestamp"], r["SBP"], r["DBP"], r["HR"]))

print(f"\nExact duplicate rows in subject077..123: {len(copied)}")
for c in copied:
    print(f"  {c[0]} ({c[1]}) == {c[2]} ({c[3]}) | Same timestamp: {c[4]}")

print(f"\nUnique/New rows in subject077..123: {len(unique_new)}")
for u in unique_new:
    print(f"  {u[0]} ({u[1]}): time={u[2]}, SBP={u[3]}, DBP={u[4]}, HR={u[5]}")
