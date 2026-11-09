# Independent Verification & Metric Audit Sign-off Report

- **Task ID**: `TASK-AUDIT-VERIFY-01`
- **Auditor**: Asep (Independent QA & Metric Auditor)
- **Target Artifacts Audited**:
  * [`dataset_duplicate_audit.json`](file:///scratch/gemastik/dataset_duplicate_audit.json)
  * [`dataset_duplicate_audit.md`](file:///scratch/gemastik/dataset_duplicate_audit.md)
  * [`audit_dataset.py`](file:///scratch/gemastik/audit_dataset.py)
  * [`bp.ipynb`](file:///C:/Users/Adyty/Documents/Farid%20ITS/TriaGo/model/bp.ipynb)
- **Verification Script**: [`verify_dataset_audit.py`](file:///scratch/gemastik/verify_dataset_audit.py)
- **Summary JSON**: [`asep_audit_summary.json`](file:///scratch/gemastik/asep_audit_summary.json)
- **Date**: 2026-10-27

---

## 1. Executive Verdict: PASS_WITH_LIMITATIONS

Amin's forensic audit findings regarding dataset duplication, the `subject094` mislabeling anomaly, and the data leakage mechanism in `bp.ipynb` are **independently verified and mathematically confirmed**.

### Verdict Justification
- **Duplicate Records & Physical Signals**: **CONFIRMED PASS**. Exactly 25 redundant bitwise clone files across 24 groups (23 pairs, 1 triplet), yielding 98 unique recordings from 123 registered IDs.
- **`subject094` Anomaly**: **CONFIRMED PASS**. Signal SHA-256 hashes and hemodynamics match Zana (`subject056` and `subject122`), confirming mislabeling under "Inas".
- **Unique Humans**: **CONFIRMED PASS**. Exactly 4 longitudinal repeat subjects across different dates, establishing a true ground truth of 94 unique biological humans.
- **`bp.ipynb` Leakage Quantification**: **CONFIRMED PASS**. Seed 71 produces 10 cross-split duplicate groups, contaminating 5 of 12 test subjects (41.7%) and 64 of 112 test segments (57.14%).
- **Limitation / Discrepancy Flagged**: **LIMITATION NOTED**. In [`audit_dataset.py`](file:///scratch/gemastik/audit_dataset.py#L519-L526) and Section 3 of [`dataset_duplicate_audit.md`](file:///scratch/gemastik/dataset_duplicate_audit.md#L98-L108), Amin hardcoded markdown table entries `0% (99 vs 99)` SpO2 and `0 mg/dL (80 vs 80)` Glucose, claiming invariant vitals across sessions. Actual ground truth from `target.csv` demonstrates genuine physiological fluctuation across longitudinal visits.

---

## 2. Independent Audit Matrix

| Forensic Item | Amin Claim | Asep Verified Value | Status | Evidence / Location |
| :--- | :---: | :---: | :---: | :--- |
| **Total Subject IDs** | 123 | **123** | PASS | `metadata/subject_mapping.csv`, `metadata/target.csv` |
| **Unique Signal Recordings** | 98 | **98** | PASS | Full SHA-256 file sweep of 246 raw CSV files |
| **Bitwise Duplicate Files** | 25 | **25** | PASS | 24 duplicate groups (23 pairs + 1 triplet) |
| **`subject094` Identity** | Mislabeled Zana | **Mislabeled Zana** | PASS | SHA-256 `9e4eb3df...` matches `subject056`/`122` |
| **Longitudinal Individuals** | 4 | **4** | PASS | Mey/Mei, Olivia, Nashwa, Nabila |
| **Unique Biological Humans** | 94 | **94** | PASS | 98 unique recordings - 4 longitudinal repeats |
| **`bp.ipynb` Loader Fallback** | Triggered | **Triggered** | PASS | [`bp.ipynb`](file:///C:/Users/Adyty/Documents/Farid%20ITS/TriaGo/model/bp.ipynb#Cell3): `patient_id` missing |
| **Optimal Split Seed** | 71 | **71** | PASS | Imbalance penalty minimization over 100 seeds |
| **Cross-Split Duplicate Groups**| 10 | **10** | PASS | 10 groups split across Train, Val, and Test |
| **Test Subject Contamination** | 5 / 12 (41.7%) | **5 / 12 (41.7%)** | PASS | Clones present: `048`, `095`, `100`, `104`, `122` |
| **Test Segment Contamination** | 64 / 112 (57.1%)| **64 / 112 (57.1%)** | PASS | 64 test segments bitwise match train/val data |
| **Longitudinal SpO2/Glucose** | Fixed (99% / 80) | **Variable** | LIMITATION | Amin hardcoded markdown template values |

---

## 3. Step-by-Step Technical Audit Findings

### Step 1: Bitwise Clone Hashing (25 Redundant Files)
- Re-hashed all 123 ECG and 123 PPG signal files using SHA-256.
- ECG and PPG files exhibit 100% paired duplication consistency: every subject with an identical PPG hash also shares an identical ECG hash.
- **Group Distribution**:
  * 23 pairs (46 subject IDs)
  * 1 triplet (`subject056`, `subject094`, `subject122` = 3 subject IDs)
  * Total redundant files: $23 \times 1 + 1 \times 2 = 25$ records.
  * Unique recordings remaining: $123 - 25 = 98$.

### Step 2: The `subject094` Anomaly Proof
A critical inspection was conducted on `subject094`:
- **Registration**: In `subject_mapping.csv`, `subject094` is assigned the name `"Inas"` with timestamp `20260501_154937`.
- **PPG SHA-256**: `9e4eb3df876e...` (Exact match with `subject056` and `subject122` ["Zana"], differing from `subject055`/`093` ["Inas"] `015a24237e22...`).
- **ECG SHA-256**: `60d0574fdeeb...` (Exact match with `subject056` and `subject122` ["Zana"], differing from `subject055`/`093` ["Inas"] `0408ef251f94...`).
- **Hemodynamics**: `target.csv` records SBP 96.0 mmHg, DBP 65.0 mmHg, HR 90.0 bpm for `subject094`. This matches Zana (`subject056`/`122`) exactly, whereas true Inas (`subject055`/`093`) has SBP 100.0, DBP 70.0, HR 87.0.
- **Conclusion**: `subject094` was inadvertently created by re-recording/copying Zana's session file 20 seconds before `15:49:57` and mislabeling it as Inas.

### Step 3: Longitudinal Subjects & Vitals Ground Truth Correction
The dataset contains 4 individuals who participated in multiple valid recording sessions on different dates. While Amin correctly identified the 4 subjects, an audit of `target.csv` reveals that his markdown report hardcoded template values.

#### Audit Comparison Table:
| Subject | Session 1 (ID / Date) | Session 2 (ID / Date) | Interval | Amin Reported SpO2 / Gluc | Ground Truth SpO2 / Gluc (`target.csv`) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Mey / Mei** | `subject058` (2026-05-01) | `subject069` (2026-05-08) | 6.85 d | 0% (99 vs 99) \| 0 mg/dL (80 vs 80) | **96% vs 100% (+4%) \| 106 vs 101 (-5 mg/dL)** |
| **Olyvia / Olivia** | `subject015` (2026-04-28) | `subject110` (2026-05-26) | 27.77 d | 0% (98 vs 98) \| 0 mg/dL (80 vs 80) | **95% vs 99% (+4%) \| 114 vs 85 (-29 mg/dL)** |
| **Nashwa / Nasywa** | `subject020` (2026-04-21) | `subject108` (2026-05-21) | 30.06 d | +1% (98 vs 99) \| 0 mg/dL (80 vs 80) | **97% vs 100% (+3%) \| 132 vs 109 (-23 mg/dL)** |
| **Nabila / Bu Nabila** | `subject106` (2026-05-26 08:18) | `subject082` (2026-05-26 12:25) | 4.11 h | 0% (98 vs 98) \| 0 mg/dL (80 vs 80) | **99% vs 97% (-2%) \| 87 vs 91 (+4 mg/dL)** |

- **Root Cause of Error in Amin's Report**: In `audit_dataset.py` lines 519-526, formatted print strings were replaced by static markdown text during initial template construction. The actual values in `dataset_duplicate_audit.json` under `longitudinal_candidates` are correct.
- **Auditor Note**: Vitals show standard intra-individual biological variability (glucose varying by 4–29 mg/dL). True biological human count is confirmed as **94**.

### Step 4: `bp.ipynb` Leakage Mechanism & Split Verification
- **Vulnerability in Cell 3**:
  ```python
  if 'subject_id' in map_df.columns and 'patient_id' in map_df.columns:
      ...
  else:
      # SILENT TRIGGER: 'patient_id' is absent from subject_mapping.csv
      gt_df['true_patient_id'] = gt_df['subject_id']
  ```
- **Vulnerability in Cell 18**:
  `GroupShuffleSplit` received `groups = clean_seg['subject_id']`. Because string identifiers differed for duplicate records (`subject054` vs `subject095`), the splitter placed duplicates into opposing splits.
- **False Validation in Cell 19**:
  `set(g_tr).intersection(set(g_te))` evaluated to `0`, producing false assurance.
- **Empirical Split Verification (`seed=71`)**:
  * Total active subjects passing SQA & outlier filtering: **79 subjects** (55 Train, 12 Val, 12 Test).
  * 10 duplicate groups split across partition boundaries.
  * **Test set subjects**: 12 subjects total.
  * **Contaminated test subjects**: 5 subjects (`subject048`, `subject095`, `subject100`, `subject104`, `subject122`), representing **41.67% of the test cohort**.
  * **Test set segments**: 112 segments.
  * **Contaminated test segments**: **64 out of 112 segments (57.14%)** are exact duplicates of samples seen by the model in training or validation.

---

## 4. Auditor Sign-off & Recommendations

### Action Plan for Squad 2:
1. **Patch Metadata (`metadata/subject_mapping.csv`)**:
   Add a `patient_id` column assigning unique canonical IDs (`patient_001` to `patient_094`) that consolidate all 25 clones and 4 longitudinal sessions.
2. **Purge 25 Redundant Signal Records**:
   Exclude the 25 duplicate rows from preprocessing and training pipelines to prevent gradient overfitting and memorization.
3. **Re-run `bp.ipynb` Baseline Evaluation**:
   Re-train and evaluate models with strict `patient_id` GroupShuffleSplit. Acknowledge in documentation that previously reported baseline MAE/RMSE metrics were artificially deflated by 57.1% test data leakage.
