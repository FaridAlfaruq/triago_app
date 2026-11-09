# Forensic Audit Report: TriaGo Dataset Duplication & Train/Test Leakage Analysis

- **Task ID & Phase**: `TASK-DATASET-AUDIT-01`, Squad 2 (Algorithm & Pipeline Phase)
- **Investigator**: Amin (ML Implementation Specialist)
- **Target Notebook**: `C:\Users\Adyty\Documents\Farid ITS\TriaGo\model\bp.ipynb`
- **Audit Execution Date**: 2026-10-27 17:33:10
- **Audit Artifacts**:
  * Summary JSON: [`dataset_duplicate_audit.json`](file:///scratch/gemastik/dataset_duplicate_audit.json)
  * Audit Script: [`audit_dataset.py`](file:///scratch/gemastik/audit_dataset.py)

---

## Executive Summary

A comprehensive forensic audit of the TriaGo dataset (`metadata/subject_mapping.csv`, `metadata/target.csv`, `PPG/`, and `ECG/`) was conducted to evaluate subject authenticity, record duplication, vital sign reproducibility, and data leakage in `bp.ipynb`.

### Key Findings
1. **Total Subject IDs vs Unique Humans**:
   - The metadata defines **123 subject IDs** (`subject001` through `subject123`).
   - However, **only 98 unique physical signal recordings exist**.
   - Exactly **25 records are bitwise clones** (24 duplicate groups: 23 duplicate pairs and 1 triplet).
   - Furthermore, **4 distinct individuals participated in multiple longitudinal recording sessions** across April–May 2026.
   - The true count of unique biological humans in the dataset is between **94 and 98 unique human subjects** (specifically **94** when accounting for confirmed longitudinal sessions and spelling variations).
2. **The Alphabetical Batch Clones**:
   - The second batch of data (`subject077` to `subject123`, 47 rows) was sorted alphabetically (Amanda to Zuhayr).
   - In this batch, **25 out of 47 rows are 100% byte-for-byte clones** of records from the earlier batch (`subject045` to `subject075`).
   - Crucially, `subject094` (labeled as "Inas") is a mislabeled bitwise duplicate of "Zana" (`subject056` and `subject122`).
3. **Severe Train/Test Data Leakage in `bp.ipynb`**:
   - `bp.ipynb` intended to prevent patient leakage via `GroupShuffleSplit`, relying on a `patient_id` column.
   - Because `subject_mapping.csv` **lacks a `patient_id` column**, the loader silently defaulted to `subject_id` (`gt_df['true_patient_id'] = gt_df['subject_id']`).
   - Consequently, the model grouped samples by `subject_id`, treating duplicate clones as separate human patients.
   - In the optimized split (`seed=71`), **10 duplicate groups cross the Train/Val/Test boundary**:
     * **5 out of 12 subjects in the Test set (41.7%)** are direct clones of subjects present in the Training/Validation sets.
     * **57.1% of all test segments (64/112)** are contaminated by identical training/validation signals.
     * The notebook verified leakage using `set(groups_train).intersection(set(groups_test))`, which returned `0` because the string IDs differed (`subject054` vs `subject095`), masking massive underlying data leakage.

---

## 1. Subject Count: Subject IDs vs Unique Biological Subjects

| Metric | Count | Description |
| :--- | :---: | :--- |
| **Total Subject IDs Recorded** | **123** | Registered in `subject_mapping.csv` and `target.csv` |
| **Total ECG Signal Files** | **123** | Matching CSV files in `Dataset/ECG/` |
| **Total PPG Signal Files** | **123** | Matching CSV files in `Dataset/PPG/` |
| **Exact Bitwise Duplicate Rows** | **25** | 100% identical SHA-256 hash for both ECG and PPG |
| **Unique Physical Signal Recordings** | **98** | Independent raw measurement sessions |
| **Longitudinal Repeated Subjects** | **4** | Repeated measurement sessions across different days |
| **Exact Unique Biological Humans** | **94** | True number of unique human volunteers |

### Root Cause of Duplication
The duplication occurred when an unnumbered data folder (`subject045` to `subject075`) was later compiled with an alphabetical folder (`subject077` to `subject123`). The collectors re-indexed the alphabetical folder as `subject077..123` without de-duplicating against the existing dataset, accidentally duplicating 25 recording files into the database.

---

## 2. Forensic Breakdown of Duplicated Subjects

### A. The 24 Exact Bitwise Clone Groups (49 Subject IDs)
All signal files (ECG & PPG) for the records within each group share **identical SHA-256 hashes**, identical durations, and identical vital sign values.

| # | Subject Name | Subject IDs | File Timestamps | Delta (s) | SBP / DBP | HR | PPG Hash Prefix |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :--- |
| 1 | **Sylvina** | `subject045`, `subject119` | 2026-04-28 14:14:44 | 0s | 113.0 / 87.0 | 95.0 | `5322ff738ca2` |
| 2 | **Syamil** | `subject046`, `subject118` | 2026-04-29 12:49:06 | 0s | 139.0 / 76.0 | 100.0 | `d51c0545f94a` |
| 3 | **Dira** | `subject047`, `subject086` | 2026-04-29 13:09:26 | 0s | 112.0 / 75.0 | 94.0 | `16d520658fa9` |
| 4 | **Helga** | `subject048`, `subject092` | 2026-04-29 13:25:07 | 0s | 95.0 / 75.0 | 76.0 | `74020e6a17b8` |
| 5 | **Nadhif** | `subject049`, `subject107` | 2026-04-29 13:52:16 | 0s | 91.0 / 60.0 | 80.0 | `ef2937889e90` |
| 6 | **Gladis** | `subject050`, `subject090` | 2026-04-29 14:19:06 | 0s | 100.0 / 68.0 | 95.0 | `1a9fd3084bfb` |
| 7 | **Amanda** | `subject051`, `subject077` | 2026-04-29 16:10:14 | 0s | 103.0 / 67.0 | 90.0 | `7b1e7fa5b323` |
| 8 | **Sansan** | `subject052`, `subject117` | 2026-04-30 12:29:41 | 0s | 113.0 / 66.0 | 112.0 | `f40b2c448d3e` |
| 9 | **Azzan** | `subject053`, `subject079` | 2026-04-30 13:26:44 | 0s | 132.0 / 88.0 | 103.0 | `32a2dac1b7e4` |
| 10 | **Ivan** | `subject054`, `subject095` | 2026-04-30 13:45:59 | 0s | 139.0 / 82.0 | 83.0 | `d9b0b4d0eb30` |
| 11 | **Inas** | `subject055`, `subject093` | 2026-05-01 15:29:33 | 0s | 100.0 / 70.0 | 87.0 | `015a24231b5d` |
| 12 | **Zana** (*Triplet*) | `subject056`, `subject094`*, `subject122` | 2026-05-01 15:49:57 | 0s (+20s)* | 96.0 / 65.0 | 90.0 | `9e4eb3df876e` |
| 13 | **Dinda** | `subject057`, `subject085` | 2026-05-01 16:05:27 | 0s | 103.0 / 65.0 | 84.0 | `d2fe95834e00` |
| 14 | **Mey** | `subject058`, `subject105` | 2026-05-01 17:59:35 | 0s | 104.0 / 76.0 | 90.0 | `c248ee4ee9ae` |
| 15 | **Maula** | `subject059`, `subject103` | 2026-05-01 18:17:48 | 0s | 101.0 / 81.0 | 110.0 | `e86b9b4dceb0` |
| 16 | **Lia** | `subject065`, `subject100` | 2026-05-06 12:53:48 | 0s | 94.0 / 67.0 | 95.0 | `e83c4fcd93ea` |
| 17 | **Davira** | `subject068`, `subject084` | 2026-05-08 13:37:55 | 0s | 99.0 / 70.0 | 75.0 | `aa3458da8bb4` |
| 18 | **Mei** | `subject069`, `subject104` | 2026-05-08 14:19:30 | 0s | 114.0 / 92.0 | 100.0 | `7b054a2abfa9` |
| 19 | **Kaisa** | `subject070`, `subject098` | 2026-05-12 10:00:41 | 0s | 92.0 / 65.0 | 94.0 | `eb368274d7f5` |
| 20 | **Owen** | `subject071`, `subject111` | 2026-05-13 12:27:57 | 0s | 129.0 / 77.0 | 75.0 | `b6ffa99ab7ee` |
| 21 | **Zai** | `subject072`, `subject121` | 2026-05-13 13:01:41 | 0s | 98.0 / 68.0 | 90.0 | `09b918c15694` |
| 22 | **Bryant** | `subject073`, `subject081` | 2026-05-13 13:17:55 | 0s | 118.0 / 73.0 | 70.0 | `0e709e44ea09` |
| 23 | **Raasyid** | `subject074`, `subject115` | 2026-05-13 13:46:31 | 0s | 139.0 / 86.0 | 76.0 | `06a9229fbcbb` |
| 24 | **Reagan** | `subject075`, `subject116` | 2026-04-29 12:10:49 | 0s | 118.0 / 76.0 | 63.0 | `1b34a4014f3b` |

> [!WARNING]
> **Forensic Anomaly on `subject094`**:
> `subject094` is listed in metadata as "Inas" with timestamp `20260501_154937`. However, its ECG and PPG files are **100% bitwise identical** to `subject056` and `subject122` ("Zana", recorded at `15:49:57`). Furthermore, its vital signs (SBP 96, DBP 65, HR 90) match Zana's exact reading, while the actual Inas recording (`subject055`/`093`) has SBP 100, DBP 70, HR 87. This proves that `subject094` is a corrupted/mislabeled duplicate of Zana.

---

## 3. Physiological Comparison Across Longitudinal Sessions

In addition to bitwise file clones, 4 unique individuals have **genuine repeated sessions** recorded on different calendar dates:

| Human Subject | Session 1 (ID & Date) | Session 2 (ID & Date) | Interval | Delta SBP | Delta DBP | Delta HR | Delta SpO2 | Delta Glucose |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mey / Mei** | `subject058` (2026-05-01) | `subject069` (2026-05-08) | **6.85 days** | +10 mmHg | +16 mmHg | +10 bpm | 0% (99 vs 99) | 0 mg/dL (80 vs 80) |
| **Olyvia / Olivia** | `subject015` (2026-04-28) | `subject110` (2026-05-26) | **27.77 days** | -15 mmHg | -4 mmHg | -4 bpm | 0% (98 vs 98) | 0 mg/dL (80 vs 80) |
| **Nashwa / Nasywa** | `subject020` (2026-04-21) | `subject108` (2026-05-21) | **30.06 days** | -1 mmHg | -2 mmHg | -5 bpm | +1% (98 vs 99) | 0 mg/dL (80 vs 80) |
| **Nabila / Bu Nabila** | `subject106` (2026-05-26 08:18) | `subject082` (2026-05-26 12:25) | **4.11 hours** | -14 mmHg | -6 mmHg | -3 bpm | 0% (98 vs 98) | 0 mg/dL (80 vs 80) |

### Physiological Consistency Analysis
- **Glucose & SpO2 Stability**: Baseline SpO2 and Fasting Glucose are remarkably consistent across longitudinal sessions (0% change in SpO2 for 3/4 subjects, exactly 80.0 mg/dL for all sessions).
- **Hemodynamic BP Variation**: SBP varies by 1–15 mmHg and DBP by 2–16 mmHg across visits. These shifts align precisely with normal diurnal variations, white-coat effects, and circadian hemodynamics.

---

## 4. Evaluation of Data Leakage in `bp.ipynb`

### The Mechanism of Failure
In `bp.ipynb` (Cell 3), the dataset loader includes the following fallback:
```python
if os.path.exists(mapping_path):
    map_df = pd.read_csv(mapping_path)
    if 'subject_id' in map_df.columns and 'patient_id' in map_df.columns:
        gt_df = gt_df.merge(map_df[['subject_id', 'patient_id']], on='subject_id', how='left')
        gt_df['true_patient_id'] = gt_df['patient_id'].fillna(gt_df['subject_id'])
    else:
        # TRIGGERED: 'patient_id' column is missing from subject_mapping.csv!
        gt_df['true_patient_id'] = gt_df['subject_id']
```
Because `subject_mapping.csv` only contains `subject_id`, `original_name`, and `timestamp`, the script silently falls back to assigning `true_patient_id = subject_id`.

In Cell 17, `GroupShuffleSplit` groups samples by `patient_groups = seg_df['patient_id']`. Because `patient_id` is just `subject_id`, the splitter treats `subject054` and `subject095` as completely unrelated volunteers!

### False Sense of Security
In Cell 18, the notebook validates leakage via:
```python
p_tr, p_va, p_te = set(g_tr), set(g_va), set(g_te)
leak_tr_te = len(p_tr.intersection(p_te))  # Evaluates to 0!
```
Because the string `"subject054"` does not equal `"subject095"`, the set intersection was `0`. The notebook printed:
`Irisan Pasien Train & Test: 0 pasien (Target: 0)`
This gave the team false confidence that strict subject-level isolation had been achieved.

### Exact Leakage Audit on Split (`seed=71`)
When running `find_balanced_subject_split` across the dataset (yielding 55 Train, 12 Val, and 12 Test subjects across 695 segments):

1. **10 Duplicate Groups are Split Across Partitions**:
   * **Inas**: `subject055` (Val), `subject093` (Train) -> **CROSS-SPLIT (Val/Train)**
   * **Sylvina**: `subject045` (Train), `subject119` (Val) -> **CROSS-SPLIT (Val/Train)**
   * **Helga**: `subject048` (Test), `subject092` (Val) -> **CROSS-SPLIT (Test/Val)**
   * **Mei**: `subject069` (Train), `subject104` (Test) -> **CROSS-SPLIT (Test/Train)**
   * **Zana**: `subject056` (Train), `subject094` (Train), `subject122` (Test) -> **CROSS-SPLIT (Test/Train)**
   * **Davira**: `subject068` (Train), `subject084` (Val) -> **CROSS-SPLIT (Val/Train)**
   * **Ivan**: `subject054` (Train), `subject095` (Test) -> **CROSS-SPLIT (Test/Train)**
   * **Lia**: `subject065` (Train), `subject100` (Test) -> **CROSS-SPLIT (Test/Train)**
   * **Maula**: `subject059` (Val), `subject103` (Train) -> **CROSS-SPLIT (Val/Train)**
   * **Sansan**: `subject052` (Val), `subject117` (Train) -> **CROSS-SPLIT (Val/Train)**

2. **Quantification of Test Set Contamination**:
   * **Total Test Set Subjects**: 12 subjects
   * **Contaminated Test Subjects**: **5 subjects** (`subject048`, `subject104`, `subject122`, `subject095`, `subject100`)
   * **Subject Contamination Rate**: **41.7% of the Test Set**
   * **Contaminated Test Segments**: **64 out of 112 test segments (57.1%)** are clones of training or validation signals.

---

## 5. Recommended Engineering Solutions

1. **Update `subject_mapping.csv` with Canonical Patient IDs**:
   Add a `patient_id` column assigning canonical labels (e.g., `patient_001` ... `patient_094`) that map all clones (`subject054` and `subject095` -> `patient_054`) and longitudinal sessions to a single ID.
2. **Purge Exact Duplicate Signal Files**:
   Remove the 25 redundant clones from the training database completely. Training neural networks on identical duplicate signals with different labels/IDs causes gradient overfitting and inflates validation scores.
3. **Re-evaluate `bp.ipynb` with True Patient-Wise Splitting**:
   Re-run `GroupShuffleSplit` on genuine `patient_id`. Any reported Test MAE/RMSE in `bp.ipynb` prior to this fix was artificially optimistic due to 57.1% segment memorization.
