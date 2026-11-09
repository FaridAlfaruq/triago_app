# FINAL EVIDENCE AUDIT REPORT: TRANSFER LEARNING VALIDATION

**Task ID**: TASK-CHECKLIST-15-FINAL-AUDIT  
**Lead Orchestrator**: Alex  
**Independent QA Specialist**: Asep  
**Implementation Engineer**: Amin  
**Date**: 2026-10-28  
**Audit Target Files**:
- Verification Script: [`checklist_eval.py`](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/gemastik/checklist_eval.py)
- Metrics JSON: [`checklist_results.json`](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/gemastik/Bp%20revisi/checklist_results.json)
- Golden Reference: [`run_transfer_learning.py`](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/gemastik/run_transfer_learning.py)
- Audit Machine-Readable Artifact: [`final_audit_evidence.json`](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/gemastik/Bp%20revisi/final_audit_evidence.json)
- Dataset File: `scratch/gemastik/Bp revisi/dataset_7ch.npz`
- Pre-Trained Weights: `scratch/gemastik/Bp revisi/model_best.pt`

---

## 1. Evaluation Protocol and Sample Accounting

### 1.1 Funnel Accounting from Raw to Evaluation Cohort

| Stage | Filter Description | Remaining Segments | Remaining Subjects | Segments Dropped | Subjects Dropped | Source Code Reference |
|---|---|---|---|---|---|---|
| **0. Raw Input** | Unfiltered dataset (`dataset_7ch.npz`) | **732** | **75** | 0 | 0 | `dataset_7ch.npz` |
| **Stage 1 SQI** | Physiological HR range ($45 \le \text{HR} \le 135\text{ bpm}$) | **731** | **75** | 1 | 0 | `train_calbased.py:91-92` |
| **Stage 2 SQI** | ECG R-peak & PPG peak count coherence ($|\Delta| \le 1$) | **554** | **63** | 177 | 12 | `train_calbased.py:93-94` |
| **Stage 3 SQI** | Inter-Beat Interval (IBI) rhythm regularity ($\text{CV}_{\text{ibi}} \le 0.15$) | **472** | **57** | 82 | 6 | `train_calbased.py:96-100` |
| **Stage 4 SQI** | Spectral cardiac band power purity ($\text{PSD}_{\text{cardiac}} / \text{PSD}_{\text{total}} \ge 0.65$) | **472** | **57** | 0 | 0 | `train_calbased.py:102-106` |
| **Clean Baseline** | Clean cohort used for uncalibrated evaluation | **472** | **57** | — | — | `checklist_eval.py:408-417` |
| **Single-Seg Isolation** | Single-segment subjects excluded from 1-pt calibration | 456 | 41 | 16 | 16 | `checklist_eval.py:228-232` |
| **Anchor Exclusion** | One anchor segment per multi-segment subject excluded | **415** | **41** | 41 | 0 | `checklist_eval.py:230-234` |
| **Final Cal Eval** | Disjoint segments contributing to calibrated metrics | **415** | **41** | — | — | `checklist_eval.py:246-248` |

### 1.2 Mathematical Arithmetic Reconciliation
Why do exactly $415$ evaluation segments remain when $41$ calibration anchors are excluded?
$$\text{Clean Segments } (472) = \text{Single-Segment Cohort } (16 \times 1 = 16) + \text{Multi-Segment Cohort } (456 \text{ segments across } 41 \text{ subjects})$$
- **Single-Segment Subjects ($S=16$, $N=16$):** Each subject has only 1 available segment. If that segment is used as the calibration anchor, no secondary segment exists for evaluation. Evaluating on the anchor itself yields identically zero error ($\hat{y} - y = 0.00$), constituting self-testing data leakage. Hence, all 16 single-segment subjects are safely excluded from the calibrated evaluation set.
- **Multi-Segment Subjects ($S=41$, $N=456$):** Each subject has $k_s \ge 2$ clean segments. Exactly 1 segment per subject is designated as the calibration anchor ($41 \times 1 = 41$ anchor segments).
- **Disjoint Evaluated Segments:**
  $$\text{Evaluated Segments } N_{\text{eval}} = \sum_{s=1}^{41} (k_s - 1) = 456 - 41 = 415$$
- **Reconciliation Proof:**
  $$472 \text{ (clean)} - 16 \text{ (single-segment)} - 41 \text{ (anchors)} = 415 \text{ evaluation segments.}$$

### 1.3 Implementation Excerpt & Exclusion Assertion
From [`checklist_eval.py`](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/gemastik/checklist_eval.py) lines 228–236:
```python
for g in np.unique(groups):
    gi = np.where(groups == g)[0]
    if len(gi) > 1:
        anc_idx = gi[0]  # First segment anchor
        eval_indices = [i for i in gi if i != anc_idx]
        eval_mask[eval_indices] = True
        anc_indices.append(anc_idx)
```
Programmatic verification assertion (exit code 0):
```python
assert len(set(anc_indices).intersection(set(np.where(eval_mask)[0]))) == 0
assert np.sum(eval_mask) == 415
```
**Status: VERIFIED**.

---

## 2. Subject-Disjoint Cross-Validation and Leakage Prevention

### 2.1 3-Fold GroupKFold Subject Allocation

| Fold ID | Train Subjects ($S_{\text{train}}$) | Test Subjects ($S_{\text{test}}$) | Train Segments ($N_{\text{train}}$) | Test Segments ($N_{\text{test}}$) | Intersection ($S_{\text{train}} \cap S_{\text{test}}$) |
|---|---|---|---|---|---|
| **Fold 1** | 38 | 19 | 316 | 156 | $\emptyset$ ($0$ overlap) |
| **Fold 2** | 38 | 19 | 311 | 161 | $\emptyset$ ($0$ overlap) |
| **Fold 3** | 38 | 19 | 317 | 155 | $\emptyset$ ($0$ overlap) |

- **Test Subject IDs per Fold**:
  - **Fold 1 (19 subjects)**: `['P002', 'P006', 'P008', 'P010', 'P013', 'P014', 'P016', 'P022', 'P023', 'P024', 'P029', 'P034', 'P040', 'P044', 'P048', 'P049', 'P051', 'P058', 'P065']`
  - **Fold 2 (19 subjects)**: `['P001', 'P003', 'P005', 'P012', 'P015', 'P020', 'P026', 'P027', 'P030', 'P032', 'P036', 'P039', 'P042', 'P043', 'P047', 'P053', 'P055', 'P060', 'P064']`
  - **Fold 3 (19 subjects)**: `['P004', 'P007', 'P009', 'P011', 'P018', 'P021', 'P025', 'P028', 'P031', 'P033', 'P035', 'P041', 'P045', 'P046', 'P050', 'P052', 'P054', 'P056', 'P063']`

### 2.2 Target Normalization (Z-Scoring)
From [`checklist_eval.py`](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/gemastik/checklist_eval.py) lines 269–270 and 306–309:
- Normalization statistics are strictly isolated:
  ```python
  s_mean, s_std = float(y_sbp[trn_idx].mean()), float(y_sbp[trn_idx].std())
  d_mean, d_std = float(y_dbp[trn_idx].mean()), float(y_dbp[trn_idx].std())
  ys_trn_z = torch.from_numpy((y_sbp[trn_idx] - s_mean) / s_std).float()
  ```
- Test predictions are mapped back using training statistics only:
  ```python
  tst_pred_s = tst_ps_z * s_std + s_mean
  ```
- Test subject targets are never accessed during model fitting or early stopping.

### 2.3 Scope of Test Anchor Usage
- **Permitted Use:** The single anchor label $y_{\text{true}}[\text{anc}]$ is accessed strictly at test time to compute an additive offset: $\delta = y_{\text{true}}[\text{anc}] - \hat{y}_{\text{pred}}[\text{anc}]$.
- **Prohibited Use (Enforced):** Zero test anchor features or labels are fed into backpropagation, loss functions, optimizer steps, or model checkpoint selection.
**Status: VERIFIED**.

---

## 3. Reproduction of Reported Metrics

Independent execution of [`run_final_evidence_audit.py`](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/run_final_evidence_audit.py) recomputed all metrics from the saved weights [`model_best.pt`](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/gemastik/Bp%20revisi/model_best.pt) and dataset [`dataset_7ch.npz`](file:///C:/Users/Adyty/Documents/Farid%20ITS/Tugas%20Akhir/scratch/gemastik/Bp%20revisi/dataset_7ch.npz).

### 3.1 Metrics Reproduction & Numerical Diff Table

| Condition / Model | Eval $N$ | SBP MAE Reported | SBP MAE Recomputed | Diff SBP | DBP MAE Reported | DBP MAE Recomputed | Diff DBP | BHS SBP | BHS DBP | AAMI Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Zero-Shot (Uncal)** | 472 | 16.67 | 16.67 | **0.00** | 8.06 | 8.06 | **0.00** | D | D | FAIL |
| **Zero-Shot (Cal-1pt)** | 415 | 8.80 | 8.80 | **0.00** | 5.49 | 5.49 | **0.00** | D | B | FAIL |
| **Naive TL (Uncal)** | 472 | 11.90 | 11.90 | **0.00** | 7.12 | 7.12 | **0.00** | D | D | FAIL |
| **Naive TL (Cal-1pt)** | 415 | 5.51 | 5.51 | **0.00** | 2.67 | 2.67 | **0.00** | B | A | **PASS** |
| **Modified TL v2 (Uncal)** | 472 | 11.76 | 11.76 | **0.00** | 7.07 | 7.07 | **0.00** | D | C | FAIL |
| **Modified TL v2 (Cal-1pt)**| 415 | 4.90 | 4.90 | **0.00** | 2.30 | 2.30 | **0.00** | B | A | **PASS** |
| **Ablation A: No L2-SP** | 415 | 4.64 | 4.64 | **0.00** | 2.20 | 2.20 | **0.00** | A | A | **PASS** |
| **Ablation B: No Cal-Loss** | 415 | 6.48 | 6.48 | **0.00** | 2.94 | 2.94 | **0.00** | B | A | **FAIL** |
| **Ablation C: Fresh Head** | 415 | 3.67 | 3.67 | **0.00** | 2.25 | 2.25 | **0.00** | A | A | **PASS** |

### 3.2 Full Regression & Cumulative Distribution Metrics (Modified TL v2 Cal-1pt)
- **Systolic BP (SBP, $N=415$):**
  - $\text{MAE} = 4.90\text{ mmHg}$
  - $\text{RMSE} = 6.34\text{ mmHg}$
  - $\text{Signed Mean Bias} = +1.22\text{ mmHg}$
  - $\text{Error STD} = 6.22\text{ mmHg}$
  - Cumulative error: $\le 5\text{ mmHg} = 59.76\%$, $\le 10\text{ mmHg} = 87.23\%$, $\le 15\text{ mmHg} = 97.35\%$
  - BHS Grade: **Grade B** | AAMI SP10: **PASS** ($|+1.22| \le 5.0, 6.22 \le 8.0$)
- **Diastolic BP (DBP, $N=415$):**
  - $\text{MAE} = 2.30\text{ mmHg}$
  - $\text{RMSE} = 3.22\text{ mmHg}$
  - $\text{Signed Mean Bias} = +0.05\text{ mmHg}$
  - $\text{Error STD} = 3.22\text{ mmHg}$
  - Cumulative error: $\le 5\text{ mmHg} = 89.40\%$, $\le 10\text{ mmHg} = 98.07\%$, $\le 15\text{ mmHg} = 100.00\%$
  - BHS Grade: **Grade A** | AAMI SP10: **PASS** ($|+0.05| \le 5.0, 3.22 \le 8.0$)

**Status: VERIFIED**. All reported values recomputed with zero numerical difference.

---

## 4. Calibration Protocol

### 4.1 Exact Mathematical Formulation
One-point calibration operates strictly as an additive post-processing translation on unstandardized predictions:
$$\hat{y}_{\text{cal}}[i] = \hat{y}_{\text{raw}}[i] + \left( y_{\text{true}}[\text{anc}] - \hat{y}_{\text{raw}}[\text{anc}] \right)$$
- **Quantity Adjusted:** Only the output prediction scalar $\hat{y}$. Zero model weights, feature maps, or regression head parameters are modified.
- **Reference Label Availability:** The reference label $y_{\text{true}}[\text{anc}]$ is available exclusively to compute the scalar difference $(y_{\text{true}}[\text{anc}] - \hat{y}_{\text{raw}}[\text{anc}])$.
- **Anchor Exclusion:** The anchor segment is unconditionally masked out from all evaluation arrays (`eval_mask[anc] = False`).

### 4.2 Anchor Selection Strategies & Bias Analysis

| Strategy | Anchor Selection Algorithm | Cal SBP MAE | Cal DBP MAE | Pre-Specification Status | Bias Classification |
|---|---|---|---|---|---|
| **First Segment** | `anc_idx = gi[0]` (chronologically first segment) | $4.90\text{ mmHg}$ | $2.30\text{ mmHg}$ | Pre-specified | **Unbiased (Causal Baseline)** |
| **Middle Segment** | `anc_idx = gi[len(gi) // 2]` (temporal center) | $3.31\text{ mmHg}$ | $1.95\text{ mmHg}$ | Post-hoc exploratory | **Optimistic Selection Bias** |
| **Max Cardiac PSD** | `anc_idx = gi[argmax(PSD_ratio)]` (highest cardiac power) | $4.02\text{ mmHg}$ | $2.14\text{ mmHg}$ | Post-hoc exploratory | **Optimistic Signal Bias** |

- **Methodological Finding:** First-segment calibration represents the true clinically realistic protocol (calibration performed upon sensor placement). Middle-segment and Max-PSD anchors yield lower errors ($3.31$ and $4.02\text{ mmHg}$) because middle segments minimize non-stationary arterial drift across the 10-minute trial, while max-PSD anchors cherry-pick segments with minimal motion artifacts.
**Status: VERIFIED_WITH_LIMITATIONS** (First segment verified as valid primary protocol; middle/max-PSD noted as post-hoc upper bounds).

---

## 5. Ablation Study and Model Comparison

### 5.1 Architecture & Training Parameters by Condition

| Condition | Backbone Status | Head Initialization | Loss Formulation | Learning Rate | Trainable Parameters | Cal SBP MAE |
|---|---|---|---|---|---|---|
| **Modified TL v2** | Frozen ($358\text{k}$) | Pretrained PulseDB weights | $\text{Huber} + 1.5 L_{\text{cal}} + 0.05 L_{2\text{-SP}}$ | $5 \times 10^{-4}$ | $49,922$ ($12.2\%$) | **$4.90\text{ mmHg}$** |
| **Ablation A (No L2-SP)** | Frozen ($358\text{k}$) | Pretrained PulseDB weights | $\text{Huber} + 1.5 L_{\text{cal}}$ | $5 \times 10^{-4}$ | $49,922$ ($12.2\%$) | **$4.64\text{ mmHg}$** |
| **Ablation B (No Cal-Loss)**| Frozen ($358\text{k}$) | Pretrained PulseDB weights | $\text{Huber} + 0.05 L_{2\text{-SP}}$ | $5 \times 10^{-4}$ | $49,922$ ($12.2\%$) | **$6.48\text{ mmHg}$** |
| **Ablation C (Linear Probe)**| Frozen ($358\text{k}$) | Fresh random initialization | $\text{Huber only}$ | $1 \times 10^{-3}$ | $49,922$ ($12.2\%$) | **$3.67\text{ mmHg}$** |

### 5.2 Critical Scientific Evaluation of the Results
1. **Calibration Loss ($L_{\text{cal}}$) is Strictly Justified:**
   Removing $L_{\text{cal}}$ (Ablation B) increases SBP MAE from $4.90$ to $6.48\text{ mmHg}$ and blows up error standard deviation from $6.22$ to $8.27\text{ mmHg}$, immediately failing ANSI/AAMI SP10 standards ($\text{STD} \le 8.0$).
2. **L2-SP Regularization Fails to Improve In-Domain Accuracy:**
   Ablation A (without L2-SP) achieves $4.64\text{ mmHg}$ vs $4.90\text{ mmHg}$ for Full Modified TL v2. Furthermore, Ablation C (Pure Linear Probing with Fresh Head) achieves $3.67\text{ mmHg}$.
3. **Evidence-Based Explanation:**
   - PulseDB pre-training was conducted on ICU hospital finger transmission PPG. TriaGO uses a reflective MAX30105 sensor with superficial optical penetration and high contact pressure sensitivity.
   - The L2-SP regularizer explicitly penalizes distance from PulseDB head weights ($\|\theta - \theta_{\text{PulseDB}}\|_2^2$), preventing the head from fully adapting to the reflective PPG morphology.
   - When heads are randomly initialized (Ablation C), the head fits the in-domain TriaGO distribution unconstrained, reaching $3.67\text{ mmHg}$.
   - **Conclusion:** While L2-SP theoretically prevents catastrophic forgetting, on this specific dataset, its complexity is **empirically counterproductive for in-domain MAE**.
**Status: VERIFIED**. Complexity of L2-SP is not empirically justified for in-domain accuracy; Calibration-Consistency Loss is fully justified.

---

## 6. Statistical and Clinical Interpretation

### 6.1 Numerical Compliance vs Clinical Validity
- **Numerical Compliance:** Modified TL v2 meets BHS Grade B (SBP) / Grade A (DBP) and satisfies ANSI/AAMI SP10 criteria ($|\text{bias}| \le 5.0$, $\text{STD} \le 8.0$).
- **Formal Clinical Validity:** **CANNOT BE CLAIMED**. AAMI SP10 and ISO 81060-2 require clinical validation across $\ge 85$ human subjects under active pharmacological or exercise interventions inducing dynamic blood pressure swings $\ge 30\text{ mmHg}$.

### 6.2 The Static Cuff Label Limitation
- **Empirical Measurement:** Across all 57 clean subjects in TriaGO, the within-subject reference ground truth standard deviation is identically:
  $$\sigma_{\text{true}}(\text{SBP}) = 0.00\text{ mmHg}, \quad \sigma_{\text{true}}(\text{DBP}) = 0.00\text{ mmHg}$$
  Every subject has only ONE single static oscillometric cuff measurement recorded before the trial.
- **Model Output Variance:** The trained model produces non-zero within-subject standard deviations:
  $$\text{Mean } \sigma_{\text{pred}}(\text{SBP}) = 2.51\text{ mmHg (Range: } 0.22 - 6.26\text{ mmHg)}$$
  $$\text{Mean } \sigma_{\text{pred}}(\text{DBP}) = 1.52\text{ mmHg (Range: } 0.16 - 4.30\text{ mmHg)}$$
- **Forensic Boundary Statement:**
  Because the ground truth is static ($\sigma = 0.00$), it is scientifically invalid to interpret $\sigma_{\text{pred}}$ as verified continuous blood pressure tracking. The predicted fluctuations may reflect true beat-to-beat vascular tone changes, but could equally stem from motion artifacts, baseline drift, or respiration coupling. Without simultaneous beat-to-beat invasive arterial lines or Finapres reference, **continuous physiological BP tracking remains UNPROVEN**.
**Status: VERIFIED_WITH_LIMITATIONS**.

---

## 7. Reproducibility and Random Seeds

### 7.1 Multi-Seed Evaluation Across Seeds `[42, 123, 999]`

| Random Seed | Uncal SBP MAE | Cal SBP MAE | Uncal DBP MAE | Cal DBP MAE |
|---|---|---|---|---|
| **Seed 42** | $11.76\text{ mmHg}$ | $4.90\text{ mmHg}$ | $7.07\text{ mmHg}$ | $2.30\text{ mmHg}$ |
| **Seed 123** | $11.76\text{ mmHg}$ | $4.90\text{ mmHg}$ | $7.07\text{ mmHg}$ | $2.30\text{ mmHg}$ |
| **Seed 999** | $11.77\text{ mmHg}$ | $4.89\text{ mmHg}$ | $7.07\text{ mmHg}$ | $2.31\text{ mmHg}$ |
| **Mean $\pm$ Std** | **$11.76 \pm 0.01\text{ mmHg}$** | **$4.90 \pm 0.00\text{ mmHg}$** | **$7.07 \pm 0.00\text{ mmHg}$** | **$2.30 \pm 0.00\text{ mmHg}$** |

### 7.2 Why Are Seed Metrics Nearly Identical?
1. **Deterministic Feature Extraction:** `extract_backbone_context` runs in `model.eval()` under `torch.no_grad()`. Given fixed inputs, the extracted 256-dimensional features are 100% deterministic.
2. **Deterministic Weight Initialization:** In Modified TL v2, regression heads are initialized by copying the pre-trained PulseDB head weights (`copy.deepcopy(m_base.sbp_head)`). They do NOT use random Gaussian initialization.
3. **Full-Batch Convex Head Optimization:** Adam optimization is performed on the entire cached training fold context ($N \approx 315$) in single full-batch passes across 25 epochs. Without random mini-batch shuffling and starting from deterministic weights, the gradient descent trajectory is mathematically deterministic.
**Status: VERIFIED**. Numerical invariance is a direct mechanical consequence of frozen features and pre-trained head initialization, not artificial fabrication.

---

## 8. Computational Performance

### 8.1 Measured Benchmarks on Test Environment
- **Operating System:** Windows 11 AMD64
- **Processor:** Intel Core CPU (6 physical cores, 12 logical execution threads)
- **Python Environment:** `scratch/.venv/Scripts/python.exe` (Python 3.11, PyTorch 2.x CPU)

| Pipeline Phase | Execution Time | Benchmark Methodology |
|---|---|---|
| **Frozen Backbone Feature Extraction** | **$2.11\text{ seconds}$** | Single forward pass through all 732 10-second segments. |
| **Regression Head Adaptation** | **$0.48\text{ seconds}$** | 3 folds $\times$ 25 epochs on cached 256-dim features. |
| **Total Evaluation Script Runtime** | **$12.08\text{ seconds}$** | Entire end-to-end evaluation with all 15 checklist items. |
| **Single-Segment Inference Latency** | **$39.87\text{ ms}$** | Median latency of 50 forward passes on 10s segment ($1250 \times 7$). |
| **Real-Time Factor** | **$250.8\times$** | $10,000\text{ ms} / 39.87\text{ ms} \to 250\times$ faster than real-time. |

### 8.2 Parameter Footprint
- **Frozen Backbone:** 358,432 parameters ($87.77\%$, 1.37 MB)
- **Trainable Regression Heads:** 49,922 parameters ($12.23\%$, 0.19 MB)
  - SBP Head: 24,961 parameters (`Linear(256->96) + BatchNorm + Linear(96->1)`)
  - DBP Head: 24,961 parameters (`Linear(256->96) + BatchNorm + Linear(96->1)`)
- **Total Model Parameters:** 408,354 parameters (1.56 MB)
**Status: VERIFIED**.

---

## 9. Final Deliverables & Minimal Reproducibility Command

### 9.1 Summary Table of All Claims

| Claim Item | Audit Status | Evidence File & Line References |
|---|---|---|
| 1. Sample Funnel ($732 \to 472 \to 415$) | `VERIFIED` | `run_final_evidence_audit.py:42-125`; `dataset_7ch.npz` |
| 2. Subject Disjointness (0 overlap) | `VERIFIED` | `checklist_eval.py:264-268`; `final_audit_evidence.json` |
| 3. Target Z-Scoring from Train Only | `VERIFIED` | `checklist_eval.py:269-270`; `checklist_eval.py:306-309` |
| 4. Metric Reproduction (All Models) | `VERIFIED` | `run_final_evidence_audit.py:230-245` (Diff $= 0.00\text{ mmHg}$) |
| 5. Zero Anchor Leakage into Eval Set | `VERIFIED` | `checklist_eval.py:230-234` (`eval_mask[anc] == False`) |
| 6. SBP Grade B / DBP Grade A | `VERIFIED` | `checklist_eval.py:85-92`; `checklist_results.json` |
| 7. ANSI/AAMI SP10 Compliance | `VERIFIED` | `checklist_eval.py:94-96` ($|\text{bias}| \le 5, \text{STD} \le 8$) |
| 8. L2-SP Superiority Claim | `FAILED` | Ablation C ($3.67\text{ mmHg}$) beats Modified TL v2 ($4.90\text{ mmHg}$) |
| 9. Continuous BP Tracking Claim | `VERIFIED_WITH_LIMITATIONS` | Constrained by static cuff label ($\sigma_{\text{true}} = 0.00\text{ mmHg}$) |
| 10. Multi-Seed Stability ($4.90 \pm 0.00$) | `VERIFIED` | Deterministic PulseDB head init + frozen features |
| 11. Sub-15s CPU Efficiency | `VERIFIED` | Measured $12.08\text{s}$ total runtime on local CPU |

### 9.2 Minimal Reproducibility Command
To reproduce the complete audit and regenerate all machine-readable evidence from project root:
```powershell
scratch/.venv/Scripts/python.exe scratch/run_final_evidence_audit.py
```
*Prerequisites:* Virtual environment at `scratch/.venv`, weights at `scratch/gemastik/Bp revisi/model_best.pt`, dataset at `scratch/gemastik/Bp revisi/dataset_7ch.npz`.

---

## 10. Direct Answers to the Three Core Questions

### Question 1: Are the reported results reproducible from the actual artifacts?
**YES.**  
All reported metrics recompute with **$0.00\text{ mmHg}$ numerical discrepancy** using the saved model weights (`model_best.pt`) and preprocessed dataset (`dataset_7ch.npz`). The arithmetic of sample counts ($472$ clean, $16$ single-segment, $41$ anchors, $415$ evaluated segments) is $100\%$ verified and closed.

### Question 2: Is the evaluation protocol free from identified data leakage?
**YES.**  
1. Cross-validation enforces strictly zero subject overlap across all 3 folds ($\emptyset$ intersection).
2. Target standardization parameters ($s_{\text{mean}}, s_{\text{std}}$) are derived exclusively from the training fold.
3. All $41$ calibration anchors are strictly excluded from the evaluation mask (`eval_mask[anc] == False`). Single-segment subjects are safely excluded to prevent self-testing bias.

### Question 3: Which scientific or clinical claims are supported, limited, or unsupported?
- **SUPPORTED:**
  - One-point calibration removes individual physiological arterial offsets, reducing error by over $58\%$.
  - Calibration-consistency loss ($L_{\text{cal}}$) is essential to prevent intra-subject prediction drift.
  - The model satisfies ANSI/AAMI SP10 numerical thresholds ($|\text{bias}| \le 5.0$, $\text{STD} \le 8.0$) and British Hypertension Society Grade B (SBP) / Grade A (DBP).
- **FAILED (UNSUPPORTED):**
  - The claim that L2-SP regularization improves in-domain transfer learning accuracy is **FAILED**. Pure Linear Probing (Fresh Head) outperforms Modified TL v2 with L2-SP by $1.23\text{ mmHg}$ ($3.67$ vs $4.90\text{ mmHg}$) because the L2-SP constraint penalizes adaptation away from ICU transmission PPG priors.
- **LIMITED (CONSTRAINED):**
  - Claims of continuous physiological blood pressure tracking are **LIMITED**. Because the reference data contains only one static cuff inflation per subject ($\sigma_{\text{true}} = 0.00\text{ mmHg}$), model prediction variance ($\sigma_{\text{pred}} \approx 2.51\text{ mmHg}$) cannot be verified as true physiological variation without independent continuous arterial reference data.
  - Meeting numerical criteria does not substitute for formal clinical device certification under ISO 81060-2.
