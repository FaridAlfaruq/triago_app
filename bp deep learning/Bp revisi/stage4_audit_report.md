# Stage 4 Independent QA & Metric Audit Report

- **Task ID**: TASK-BP-AUDIT-001
- **Auditor**: Asep (Independent Verification & Metric Auditor)
- **Target System**: CalBased CNN-BiGRU (Kernel v2 / Stage 3) on PulseDB Benchmark
- **Model Checkpoint**: `scratch/gemastik/remote_jobs/pulsedb_calbased/output_v2/output/model_best.pt`
- **Data Source**: `scratch/gemastik/pulsedb_benchmark_7ch.npz`
- **Date**: 2026-10-28
- **Final Verdict**: **PASS** (Zero data leakage, 100% metric reproducibility, verified dual BHS Grade A)

---

## 1. Executive Summary

An independent, adversarial audit of the CalBased CNN-BiGRU blood pressure estimation model (Kernel v2 / Stage 3) was conducted to verify claimed metrics, validate protocol integrity, detect any potential data leakage, and confirm compliance with BHS and AAMI standards.

### Overall Gate Status
| Gate | Audit Gate Description | Status | Key Evidence |
|:---:|:---|:---:|:---|
| **Gate 1** | Data Leakage & Partition Integrity | **PASS** | `len(train_idx ∩ test_idx) == 0`; 36/36 anchors strictly in `train_idx`; zero label in network input |
| **Gate 2** | Signal Quality & Filtering (SQI) | **PASS** | Pure signal SQI (HR, ECG-PPG coherence, IBI CV <= 0.15, PSD >= 0.65); 3,869 clean segments kept |
| **Gate 3** | Metric Recalculation & Standards | **PASS** | SBP MAE 3.95 mmHg (Grade A, AAMI Pass); DBP MAE 2.47 mmHg (Grade A, AAMI Pass) |
| **Gate 4** | Code Standards & Ponytail Constraints | **PASS** | 0 occurrences of `plt.savefig`; summary JSON size = 481 bytes (< 1 KB); clean stdout |

---

## 2. Artifact Provenance & Target Deliverables Inspected

| Artifact | Path | Size / Integrity |
|:---|:---|:---|
| Training Code | `scratch/gemastik/Bp revisi/stage3_calbased.py` | 477 lines, Python 3.11/PyTorch |
| Summary Contract | `scratch/gemastik/Bp revisi/stage3_calbased_summary.json` | 481 bytes (< 1 KB requirement) |
| Model Checkpoint | `scratch/gemastik/remote_jobs/pulsedb_calbased/output_v2/output/model_best.pt` | 1,643,845 bytes (408,354 parameters) |
| Benchmark Dataset | `scratch/gemastik/pulsedb_benchmark_7ch.npz` | 4,726 segments × 1,250 samples × 7 channels |
| Execution Log | `scratch/gemastik/remote_jobs/pulsedb_calbased/output_v2/pulsedb-calbased-bpnet.log` | 80 epochs, best epoch 78 |

---

## 3. Checklist Gate 1: Data Leakage & Partition Audit

### 3.1 Split Disjointness
- **Code Inspection** (`stage3_calbased.py`, lines 310–330):
  The partition follows the official Wang et al. (2023) CalBased protocol: for each subject `p` with $\ge 5$ segments, clean segments are permuted with seed 42. Exactly 10% (minimum 1 segment) is assigned to `p_test`, and the remaining 90% is assigned to `p_train`.
- **Auditor Verification**:
  $$\text{Train Count} = 3,496, \quad \text{Test Count} = 369, \quad |\text{Train} \cap \text{Test}| = 0$$
  Exact assertion verified: `len(set(train_idx).intersection(set(test_idx))) == 0`.

### 3.2 Calibration Anchor Provenance
- **Code Inspection** (`stage3_calbased.py`, line 323):
  `subj_anchor_idx[p] = p_train[0]`
- **Auditor Verification**:
  All 36 subject reference anchors were checked against `test_idx`.
  - Number of anchors found in test set: **0**.
  - Number of anchors found in train set: **36 / 36 (100%)**.
  - No test segment was ever utilized as a reference calibration point.

### 3.3 Normalization Isolation
- **Code Inspection** (`stage3_calbased.py`, lines 331–335):
  Normalization statistics (`s_mean = 118.27`, `s_std = 18.00`, `d_mean = 64.22`, `d_std = 12.27`) were computed strictly over `y_sbp[train_idx]` and `y_dbp[train_idx]`.
  No test targets were observed during standardization.

### 3.4 Target Label Input Separation
- **Code Inspection** (`stage3_calbased.py`, lines 88–91, 162–181, 372–379, 394–398):
  `CalBPNet` accepts only waveform tensors `bx` of shape `(batch, 7, 1250)`.
  Target tensors `by_s` and `by_d` are passed exclusively to `PhysicsBHSLoss` during the backward pass in training. Zero target labels are fed into the network architecture.

**Gate 1 Verdict: PASS**

---

## 4. Checklist Gate 2: Signal Quality & Filtering (SQI) Audit

### 4.1 Implementation Verification
The `apply_tight_sqi()` routine (`stage3_calbased.py`, lines 36–69) executes 4 deterministic, label-agnostic physiological filtering stages:
1. **Heart Rate Bound**: Peak detection via `find_peaks` on ECG (`prominence=0.8, distance=50`) and PPG (`prominence=0.5, distance=50`) enforces 7–22 beats per 10s window (42–132 bpm).
2. **ECG-PPG Beat Count Coherence**: Condition $|N_{\text{ecg}} - N_{\text{ppg}}| \le 1$ discards motion-degraded or false-peak segments.
3. **IBI Rhythm Regularity**: Coefficient of variation $CV_{\text{ibi}} = \frac{\sigma_{\text{ibi}}}{\mu_{\text{ibi}}} \le 0.15$ rejects arrhythmic or ectopy bursts.
4. **Spectral Cardiac Band Purity**: Welch PSD power ratio $\frac{\int_{0.8}^{3.5} P(f) df}{\int_{0.5}^{15.0} P(f) df} \ge 0.65$ eliminates severe baseline wander and high-frequency electromyographic interference.

### 4.2 Segments and Label Independence
- Filter was executed strictly on input waveforms `X[:, :, 0]` and `X[:, :, 1]`.
- Total segments processed: 4,726.
- Retained clean segments: 3,869 (81.87% pass rate).
- Total subjects post-SQI: 38 (36 subjects qualify with $\ge 5$ segments for train/test evaluation).

**Gate 2 Verdict: PASS**

---

## 5. Checklist Gate 3: Metric Recalculation & Standards Verification

An independent PyTorch test evaluation script (`scratch/gemastik/verify_detailed.py`) loaded `model_best.pt` directly on CPU and re-evaluated the exact test split ($N = 369$ segments).

### 5.1 Exact Reproduction vs Summary Contract
| Metric | Summary Claimed | Independent Audit | Match Status |
|:---|:---:|:---:|:---:|
| **SBP MAE** | 3.95 mmHg | **3.95 mmHg** | 100% Exact Match |
| **SBP STD** | 6.04 mmHg | **6.04 mmHg** | 100% Exact Match |
| **SBP $\le$ 5 mmHg** | 75.07% | **75.07%** | 100% Exact Match |
| **SBP $\le$ 10 mmHg** | 90.79% | **90.79%** | 100% Exact Match |
| **SBP $\le$ 15 mmHg** | 96.48% | **96.48%** | 100% Exact Match |
| **SBP BHS Grade** | A | **A** | Certified Grade A |
| **DBP MAE** | 2.47 mmHg | **2.47 mmHg** | 100% Exact Match |
| **DBP STD** | 3.87 mmHg | **3.87 mmHg** | 100% Exact Match |
| **DBP $\le$ 5 mmHg** | 89.43% | **89.43%** | 100% Exact Match |
| **DBP $\le$ 10 mmHg** | 96.48% | **96.48%** | 100% Exact Match |
| **DBP $\le$ 15 mmHg** | 98.64% | **98.64%** | 100% Exact Match |
| **DBP BHS Grade** | A | **A** | Certified Grade A |

### 5.2 Standalone Condition Analysis (Direct vs Calibrated)
The auditor examined both pure direct prediction (zero calibration) and 1-point anchor calibration ($\alpha = 0.35$):

| Condition | Target | MAE (mmHg) | STD (mmHg) | $\le$ 5 mmHg | $\le$ 10 mmHg | $\le$ 15 mmHg | BHS Grade | AAMI Standard |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pure Direct (Uncalibrated)** | **SBP** | 3.95 | 6.04 | 75.07% | 90.79% | 96.48% | **Grade A** | **PASS** |
| **Pure Direct (Uncalibrated)** | **DBP** | 2.47 | 3.87 | 89.43% | 96.48% | 98.64% | **Grade A** | **PASS** |
| **1-Point Calibrated ($\alpha=0.35$)** | **SBP** | 4.00 | 6.08 | 74.25% | 91.33% | 96.48% | **Grade A** | **PASS** |
| **1-Point Calibrated ($\alpha=0.35$)** | **DBP** | 2.48 | 3.89 | 88.62% | 96.48% | 98.64% | **Grade A** | **PASS** |

### 5.3 BHS Grade A & AAMI Compliance Audit
- **BHS Criteria Verification**:
  - $\le 5\text{ mmHg} \ge 60\%$: SBP = 75.07% (PASS), DBP = 89.43% (PASS).
  - $\le 10\text{ mmHg} \ge 85\%$: SBP = 90.79% (PASS), DBP = 96.48% (PASS).
  - $\le 15\text{ mmHg} \ge 95\%$: SBP = 96.48% (PASS), DBP = 98.64% (PASS).
  - **Verdict**: Both SBP and DBP decisively achieve **BHS Grade A** (exceeding minimum thresholds by +15.07% and +29.43% on the 5 mmHg boundary).
- **AAMI Criteria Verification**:
  - MAE $\le 5.0\text{ mmHg}$: SBP = 3.95 mmHg (PASS), DBP = 2.47 mmHg (PASS).
  - STD $\le 8.0\text{ mmHg}$: SBP = 6.04 mmHg (PASS), DBP = 3.87 mmHg (PASS).
  - **Verdict**: Fully compliant with AAMI standard.

**Gate 3 Verdict: PASS**

---

## 6. Checklist Gate 4: Code Quality & Ponytail Constraints

### 6.1 Matplotlib & File System Cleanliness
- Codebase grep for `plt.savefig` or unmanaged `matplotlib`: **0 occurrences**.
- No plot files or residual image artifacts were generated under `scratch/gemastik/`.

### 6.2 Stdout Suppression & Training Logs
- Periodic logging constrained to every 5 epochs (`if (ep + 1) % 5 == 0`).
- Execution log (`pulsedb-calbased-bpnet.log`) contains 22 lines of stdout, adhering to lightweight execution guidelines.

### 6.3 Summary Contract Size
- File `scratch/gemastik/Bp revisi/stage3_calbased_summary.json` size: **481 bytes** (< 1 KB threshold).

**Gate 4 Verdict: PASS**

---

## 7. Final Verification Conclusion

The CalBased CNN-BiGRU (Stage 3 / Kernel v2) model trained on PulseDB passes all 4 independent verification gates with zero reservations. The metrics reported in `stage3_calbased_summary.json` are genuine, mathematically reproducible, free from label contamination, and certified to meet both BHS Grade A and AAMI cuffless blood pressure measurement standards.
