# Independent QA & Metric Audit Report: TASK-TL-KIM-BAEK-02

- **Auditor**: Asep (Independent Verification & Metric Auditor)
- **Target**: `scratch/gemastik/run_transfer_learning.py` & `scratch/gemastik/Bp revisi/transfer_learning_summary.json`
- **Execution Timestamp**: 2026-10-28T02:19:00+07:00
- **Overall Verdict**: **PASS**

---

## Gate 1: Zero Leakage & Disjoint Anchor Exclusion — PASS
- **Cross-Validation Partitioning**: 3-fold `GroupKFold` verified subject-disjoint via explicit set intersection assertion (`len(set(groups[trn_idx]).intersection(set(groups[tst_idx]))) == 0`, [run_transfer_learning.py:129](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/gemastik/run_transfer_learning.py#L129)).
- **Target Normalization**: Z-score parameters (`s_mean`, `s_std`, `d_mean`, `d_std`) are strictly computed on training folds (`y_sbp[trn_idx]`, `y_dbp[trn_idx]`, lines 130–131, 179–180). Zero test label leakage.
- **Disjoint Calibration Evaluation**: 1-point calibration offset computed exclusively using each test subject's initial anchor segment `gi[0]`. Evaluated exclusively on subsequent segments `gi[1:]` (`N=415`, line 211). All 57 subject anchor segments strictly excluded from evaluation mask (`eval_mask`, lines 212, 220).

## Gate 2: Signal Processing & SQI Integrity — PASS
- **Purity Check**: `apply_tight_sqi(X, fs=125)` in [train_calbased.py:36-70](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/gemastik/remote_jobs/pulsedb_calbased/train_calbased.py#L36-L70) operates exclusively on raw sensor arrays (ECG ch0, PPG ch1) checking physiological HR bounds, R-to-systolic peak count coherence, IBI CV regularity, and Welch spectral cardiac power purity. Zero label usage.
- **Retention Count**: Preserves exactly 472 clean 10s segments across 57 unique subjects (`assert n_samples == 472 and n_subjects == 57`, line 100).

## Gate 3: Metric Recalculation & Non-Trivial Variance — PASS
- **Execution Time**: Single fast verification run completed in **4.29 seconds** (< 10s cap).
- **Exact Metric Verification**:
  - **Transfer Learning (Uncalibrated, N=472)**:
    - SBP: MAE = 11.76 mmHg, STD = 14.57 mmHg, BHS Grade D (≤5: 25.6%, ≤10: 47.9%, ≤15: 67.6%)
    - DBP: MAE = 7.07 mmHg, STD = 8.54 mmHg, BHS Grade C (≤5: 40.5%, ≤10: 75.8%, ≤15: 93.2%)
    - AAMI Standard: FAIL
  - **Transfer Learning (1-Point Calibrated, N=415 Disjoint)**:
    - SBP: MAE = 4.90 mmHg, STD = 6.22 mmHg, BHS Grade B (≤5: 59.8%, ≤10: 87.2%, ≤15: 97.3%)
    - DBP: MAE = 2.30 mmHg, STD = 3.22 mmHg, BHS Grade A (≤5: 89.4%, ≤10: 98.1%, ≤15: 100.0%)
    - AAMI Standard: **PASS** (SBP & DBP both meet MAE ≤ 5.0 mmHg, STD ≤ 8.0 mmHg)
  - **Comparison Conditions Verified**:
    - Scratch (Cal-1pt): SBP MAE = 3.67 (Grade A), DBP MAE = 2.25 (Grade A), AAMI PASS
    - Zero-Shot (Cal-1pt): SBP MAE = 8.80 (Grade D), DBP MAE = 5.49 (Grade B), AAMI FAIL

## Gate 4: Engineering Standards & Resource Envelope — PASS
- **Summary JSON Size**: `transfer_learning_summary.json` is **1000 bytes** (< 1024 bytes cap).
- **Stdout Output Volume**: Exactly **15 lines** (≤ 15 lines cap).
- **No Artifact Pollution**: Zero `plt.savefig()` calls, zero matplotlib dependencies in execution path.
