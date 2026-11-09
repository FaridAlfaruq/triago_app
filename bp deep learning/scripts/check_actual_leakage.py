import json
import pandas as pd
import check_signal_files as csf

with open(r"scratch\gemastik\split_simulation.json", "r") as f:
    splits = json.load(f)

tr_set = set(splits['train_sids'])
va_set = set(splits['val_sids'])
te_set = set(splits['test_sids'])

df = pd.merge(csf.res_df, csf.df_tgt, on="subject_id")

# Hash groups of duplicates
grouped = df.groupby("ppg_hash")
leak_cases = []

for h, g in grouped:
    if len(g) > 1:
        sids = list(g["subject_id"])
        # Check where each sid is in the split
        locs = {}
        for s in sids:
            if s in tr_set:
                locs[s] = 'Train'
            elif s in va_set:
                locs[s] = 'Val'
            elif s in te_set:
                locs[s] = 'Test'
            else:
                locs[s] = 'Filtered_Out'
        
        assigned_splits = set(loc for loc in locs.values() if loc != 'Filtered_Out')
        is_leaked = len(assigned_splits) > 1
        leak_cases.append({
            'hash': h[:8],
            'name': g.iloc[0]['original_name'],
            'sids': sids,
            'locs': locs,
            'is_leaked': is_leaked,
            'splits_involved': list(assigned_splits)
        })

print(f"Total duplicate groups: {len(leak_cases)}")
leaked = [c for c in leak_cases if c['is_leaked']]
print(f"Total duplicate groups crossing Train/Val/Test: {len(leaked)}")
for l in leaked:
    print(f"Group {l['name']} (hash {l['hash']}): {l['locs']} -> CROSS-SPLIT: {l['splits_involved']}")

within_same = [c for c in leak_cases if not c['is_leaked'] and len(set(loc for loc in c['locs'].values() if loc != 'Filtered_Out')) == 1]
print(f"\nDuplicate groups assigned to same split (data redundancy): {len(within_same)}")
for w in within_same:
    print(f"Group {w['name']} (hash {w['hash']}): {w['locs']}")

filtered_out = [c for c in leak_cases if len(set(loc for loc in c['locs'].values() if loc != 'Filtered_Out')) == 0]
print(f"\nDuplicate groups completely filtered out: {len(filtered_out)}")
for f_out in filtered_out:
    print(f"Group {f_out['name']} (hash {f_out['hash']}): {f_out['locs']}")
