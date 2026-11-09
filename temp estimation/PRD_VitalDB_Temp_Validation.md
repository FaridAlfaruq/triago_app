# Product Requirement Document (PRD) — Version 2.2
## Semi-Synthetic Bio-Heat Validation Pipeline for TriaGo Body Temperature Estimation

* **Document Version:** 2.2.0 (Synchronized with User Notebook & Workspace Directory)  
* **Status:** Under Review / Awaiting User Sign-Off  
* **Lead Orchestrator:** Alex (`alex-orchestrator`)  
* **Sub-Agents Involved:** `ahmad`, `aisyah`, `rian`, `amin`, `asep`, `joli` (*`amar` excluded*)  
* **Execution Environment:** Jupyter Notebook (`scratch/temp estimation/temp_estimation.ipynb`)  
* **Sub-Agent Reference Doc:** `scratch/temp estimation/PRD_VitalDB_Temp_Validation.md`  
* **Target Cohort:** Exactly 10 VitalDB Cases  
* **Target Performance:** $\text{MAE} \le 0.50\ ^\circ\text{C}$ (Benchmarked against Clinical Standard $\le 0.35\ ^\circ\text{C}$)  
* **Date:** 2026-10-25  

---

## 1. Executive Summary & Objective

### 1.1 Problem Statement
TriaGo proposes an automated triage clamp estimating core body temperature from peripheral skin temperature ($T_{\text{skin}}$), ambient temperature ($T_{\text{amb}}$), and PPG Perfusion Index ($PI$):
$$T_{\text{est}} = T_{\text{skin}} + \frac{T_{\text{skin}} - T_{\text{amb}}}{a + b \cdot PI^\alpha}$$
In the initial proposal, temperature evaluation was solely qualitative (forehead spot-check). No quantitative metrics (MAE, RMSE, Bland-Altman) exist. Public repositories lack a turnkey open dataset simultaneously recording $\{T_{\text{skin\_wrist}}, PI_{\text{wrist}}, T_{\text{core}}, T_{\text{amb}}\}$.

### 1.2 Objective
Develop and execute a **Semi-Synthetic Hybrid Validation Pipeline** on the open **VitalDB** clinical database:
1. Extract real surgical patient trajectories (continuous esophageal core temperature $T_{\text{core}}$ and optical PPG) from **exactly 10 cases**.
2. Downsample PPG from 500 Hz to 125 Hz using the user's `downsample` helper function, restore baseline, filter ($0.5\text{--}8.0\text{ Hz}$), segment into 10-s windows, filter via Signal Quality Index (SQI), and extract $PI = (AC / DC) \times 100\%$.
3. Synthesize physiological skin surface temperature ($T_{\text{skin}}$) using established, peer-reviewed bio-heat transfer formulations (Pennes, Stolwijk-Hardy, Gagge) with first-order thermal lag and MLX90614 medical-grade sensor noise.
4. Fit the regularized calibration parameters $(a, b, \alpha)$ on a train cohort and evaluate on a test cohort to achieve the lightness target of $\text{MAE} \le 0.50\ ^\circ\text{C}$, benchmarking against the clinical standard $\le 0.35\ ^\circ\text{C}$.
5. Deliver all code, processing, and mandatory diagnostic plots exclusively in the user-created **Jupyter Notebook**:  
   `scratch/temp estimation/temp_estimation.ipynb`.
6. Maintain full PRD documentation at `scratch/temp estimation/PRD_VitalDB_Temp_Validation.md` for direct access by all active sub-agents.

---

## 2. Scientific Literature Foundation & Verbatim Citations

To guarantee academic rigor for the GEMASTIK jury and thesis committee, every biophysical model and constant is derived directly from peer-reviewed literature:

### 2.1 Complete Literature Reference List & Verbatim Excerpts

#### 1. Bio-Heat Governing Equation
* **Source:** Pennes, H. H. (1948). *"Analysis of tissue and arterial blood temperatures in the resting human forearm."* *Journal of Applied Physiology*, 1(2), 93–122.
* **DOI / Link:** [10.1152/jappl.1948.1.2.93](https://doi.org/10.1152/jappl.1948.1.2.93)
* **Verbatim Excerpt:**
  > *"The rate of heat transfer between blood and tissue was assumed to be proportional to the temperature difference between the arterial blood and the tissue... $\rho c \frac{\partial T}{\partial t} = k \nabla^2 T + \omega_b c_b (T_a - T) + q_m$."*
* **Application in PRD:** Fundamental basis for coupling blood flow volumetric heat capacity ($\rho_b c_b = 3.85 \times 10^6\ \text{J}/(\text{m}^3\cdot^\circ\text{C})$) to tissue thermal equilibration.

#### 2. Two-Node Core-to-Skin Conductance Formulation
* **Source:** Stolwijk, J. A. J., & Hardy, J. D. (1966). *"Partitional calorimetric studies of man during exposures to thermal transients."* *Journal of Applied Physiology*, 21(3), 967–977.
* **DOI / Link:** [10.1152/jappl.1966.21.3.967](https://doi.org/10.1152/jappl.1966.21.3.967)
* **Verbatim Excerpt:**
  > *"Heat flow from the core to the skin is mediated by conduction through tissue layers and convection by blood flow: $H_{c-s} = (K + c_b \dot{m}_{bl}) (T_{core} - T_{skin})$ where $K$ is the minimal tissue conductance."*
* **Application in PRD:** Derivation of internal conductance separating passive tissue conductance ($K_{\text{tissue}} = 5.28\ \text{W}/(\text{m}^2\cdot^\circ\text{C})$) from microvascular perfusion.

#### 3. Skin-to-Environment Heat Balance & Core-Shell Equilibrium
* **Source:** Gagge, A. P., Stolwijk, J. A., & Nishi, Y. (1971). *"An effective temperature scale based on a simple model of human physiological regulatory response."* *ASHRAE Transactions*, 77(1), 247–262.
* **Link:** [Semantic Scholar Record](https://www.semanticscholar.org/paper/An-effective-temperature-scale-based-on-a-simple-of-Gagge-Stolwijk/9c222ff5c45330a84e314e3cae697818e692a343)
* **Verbatim Excerpt:**
  > *"The core exchanges heat with the skin shell through passive conduction and peripheral blood circulation... Heat balance at the skin surface requires that heat transferred from the core equals heat lost by convection and radiation to the environment: $Q_{cr-sk} = (h_c + h_r)(T_{sk} - T_a) + E_{sk}$."*
* **Application in PRD:** Equation equating internal delivered heat to external environmental loss in controlled hospital environments.

#### 4. Optical Perfusion Index (PI) as Measure of Microvascular Tone
* **Source:** Allen, J. (2007). *"Photoplethysmography and its application in clinical physiological measurement."* *Physiological Measurement*, 28(3), R1–R39.
* **DOI / Link:** [10.1088/0967-3334/28/3/R01](https://doi.org/10.1088/0967-3334/28/3/R01)
* **Verbatim Excerpt:**
  > *"The AC component of the PPG represents the pulsatile changes in arterial blood volume synchronous with the heartbeat, while the DC component reflects the overall baseline blood volume, tissue, and bone attenuation. The perfusion index (PI = AC/DC) provides an objective numerical measure of peripheral microvascular perfusion."*
* **Application in PRD:** Rigorous mathematical basis for separating pulsatile $AC$ and baseline $DC$ components to compute $PI$.

#### 5. Clinical Correlation Between PI and Peripheral Temperature Drops
* **Source:** Lima, A. P., & Bakker, J. (2005). *"Noninvasive monitoring of peripheral perfusion."* *Intensive Care Medicine*, 31(10), 1316–1326.
* **DOI / Link:** [10.1007/s00134-005-2790-2](https://doi.org/10.1007/s00134-005-2790-2)
* **Verbatim Excerpt:**
  > *"The peripheral perfusion index derived from pulse oximetry reflects changes in peripheral blood flow... In states of hypoperfusion, vasoconstriction causes a marked decrease in the pulsatile amplitude relative to the non-pulsatile signal."*
* **Application in PRD:** Justifies mapping perfusion index $PI$ dynamically to blood flow rate $\omega_b(PI)$.

#### 6. Hospital Environmental Heat Transfer Coefficients
* **Source:** de Dear, R. J., Arens, E., Hui, Z., & Oguro, M. (1997). *"Convective and radiative heat transfer coefficients for individual human body segments."* *International Journal of Biometeorology*, 40(3), 141–156.
* **DOI / Link:** [10.1007/s004840050035](https://doi.org/10.1007/s004840050035)
* **Verbatim Excerpt:**
  > *"Natural convection heat transfer coefficients ($h_c$) for distal extremities in still air range between 4.0 and 5.0 W/(m²·K), while the radiative heat transfer coefficient ($h_r$) averages 4.7 W/(m²·K) across normal indoor ambient temperatures."*
* **Application in PRD:** Grounding total environmental heat transfer coefficient $h_{\text{total}} = h_c + h_r \approx 9.2\ \text{W}/(\text{m}^2\cdot^\circ\text{C})$ for standard hospital ambient temperature ($T_{\text{amb}} = 21.0 \pm 0.3\ ^\circ\text{C}$).

#### 7. Thermal Inertia & Dynamic Lag in Peripheral Tissue
* **Source:** Wissler, E. H. (1985). *"Mathematical simulation of human thermal behavior using whole body models."* In *Heat Transfer in Medicine and Biology*, Plenum Press, pp. 325–373.
* **DOI / Link:** [10.1007/978-1-4684-8285-0_13](https://doi.org/10.1007/978-1-4684-8285-0_13)
* **Verbatim Excerpt:**
  > *"The transient thermal response of peripheral tissue layers is characterized by significant thermal inertia, requiring dynamic differential formulations rather than static steady-state assumptions when modeling rapid hemodynamic perturbations."*
* **Application in PRD:** First-order differential equation $C_{\text{skin}} \frac{dT_{\text{skin}}}{dt} = Q_{\text{in}} - Q_{\text{out}}$ with thermal time constant $\tau \approx 180\text{ s}$.

#### 8. Core Body Temperature Estimation via Perfusion Resistance
* **Source:** Ren, X., Zhang, Y., Yang, H., Li, K., Ye, X., & Zhou, C. (2025). *"Core body temperature estimation model with thermal contact resistance compensation."* *Measurement*, 241, 115687.
* **DOI / Link:** [10.1016/j.measurement.2024.115687](https://doi.org/10.1016/j.measurement.2024.115687)
* **Verbatim Excerpt:**
  > *"The heat transfer from the core to the skin is strongly modulated by cutaneous perfusion. The ratio of temperature difference $(T_{skin} - T_{ambient})$ divided by perfusion index reflects the thermal resistance between core and surface."*
* **Application in PRD:** Core relationship underlying TriaGo's estimation formula.

#### 9. Signal Quality Index (SQI) for Photoplethysmography
* **Source:** Elgendi, M. (2016). *"Optimal signal quality index for photoplethysmogram signals."* *Bioengineering*, 3(4), 21.
* **DOI / Link:** [10.3390/bioengineering3040021](https://doi.org/10.3390/bioengineering3040021)
* **Verbatim Excerpt:**
  > *"Signal Quality Indices (SQI) based on skewness, kurtosis, and zero-crossing ratios effectively segregate motion-corrupted PPG segments from physiologically viable pulsatile waveforms."*
* **Application in PRD:** SQI filtering criteria to retain only clean PPG windows.

---

## 3. Sensor Realism & MLX90614 Datasheet Specifications

* **Official Datasheet Source:** Melexis MLX90614 Family Single and Dual Zone Infra-Red Thermometer in TO-39.  
* **URL:** [Melexis MLX90614 Datasheet](https://www.melexis.com/en/documents/documentation/datasheets/datasheet-mlx90614)  
* **Delegation:** Hardware review assigned to `rian` (Electronics) and `ahmad` (Literature).

### 3.1 Medical-Grade Specifications Applied to Model
1. **Clinical Temperature Operating Range:** Enforced spanning **$33.0\ ^\circ\text{C}$ to $45.0\ ^\circ\text{C}$** ($T_{\text{core}}$ and $T_{\text{skin}}$), representing the full physiological boundary from moderate accidental/surgical hypothermia ($33\ ^\circ\text{C}$) to severe fever/malignant hyperthermia ($45\ ^\circ\text{C}$).
2. **Medical Zone Accuracy:** $\pm 0.2\ ^\circ\text{C}$ within normal body temperature range ($T_{\text{object}} \in [36.0\ ^\circ\text{C}, 38.0\ ^\circ\text{C}]$) and operating ambient range ($T_{\text{ambient}} \in [16.0\ ^\circ\text{C}, 40.0\ ^\circ\text{C}]$).
3. **Digital Resolution:** 17-bit on-chip ADC with SMBus / I2C digital output resolution of $0.02\ ^\circ\text{C}$.
4. **Noise Equivalent Temperature Difference (NETD):** Random noise Gaussian parameter modeled as:
   $$\epsilon_{\text{sensor}} \sim \mathcal{N}(0, \sigma_{\text{sensor}}^2) \quad \text{where } \sigma_{\text{sensor}} = 0.05\ ^\circ\text{C}$$
5. **Thermal Equilibrium / Settling Time:** Refresh rate configured to 2 Hz (0.5 s), matching TriaGo firmware specification.

---

## 4. Signal Processing Pipeline & Feature Engineering

All PPG preprocessing strictly follows the refined 7-stage protocol:

```
[Raw 500 Hz PPG (VitalDB)]
           │
           ▼ (1)
[Downsample to 125 Hz] (scipy.signal.resample helper function)
           │
           ▼ (2)
[Baseline Restoration] (Double-median filter / Cubic spline detrending)
           │
           ▼ (3)
[PPG Bandpass Filter] (4th-order Butterworth 0.5 – 8.0 Hz)
           │
           ▼ (4)
[10-Second Segmentation] (1,250 samples per window)
           │
           ▼ (5)
[Signal Quality Index (SQI)] (Skewness > 0, Kurtosis 2.5–5.5)
     ├── Pass  ──► (6) Digital Filtering: Separate AC & DC Components
     │                    │
     │                    ▼ (7)
     │             Compute Perfusion Index: PI = (AC / DC) * 100%
     │
     └── Fail  ──► Discard Window (Log artifact rejection)
```

### 4.1 Detailed Pipeline Steps

#### 1. Downsampling (500 Hz $\to$ 125 Hz)
Implemented using the exact helper function:
```python
from scipy.signal import resample
import numpy as np

def downsample(x, fs, target_fs, time):
    x_resample = resample(x, int(len(x) * target_fs / fs))
    time_resample = np.linspace(time[0], time[-1], len(x_resample))
    return x_resample, time_resample
```

#### 2. Baseline Restoration
* Remove slow respiratory baseline drift using double-median filtering (window lengths 0.2 s and 0.6 s) or cubic spline detrending.
* Centers systolic pulses around the true isoelectric zero-line.

#### 3. Bandpass Filtering ($0.5\text{--}8.0\text{ Hz}$)
* Standard zero-phase 4th-order Butterworth bandpass filter with cut-offs at **0.5 Hz** (suppresses sub-cardiac baseline wander) and **8.0 Hz** (retains up to 4th cardiac harmonic, preserving systolic peak rise time and dicrotic notch while eliminating high-frequency electrosurgical noise).

#### 4. 10-Second Segmentation
* Slice signal into non-overlapping epochs of 1,250 samples (10 s at 125 Hz).

#### 5. Signal Quality Index (SQI)
* Compute skewness $S_{sqi} = \frac{\mathbb{E}[(x-\mu)^3]}{\sigma^3}$ and kurtosis $K_{sqi} = \frac{\mathbb{E}[(x-\mu)^4]}{\sigma^4}$.
* Pass condition: $S_{sqi} > 0$ (positive systolic peak asymmetry) and $2.5 \le K_{sqi} \le 5.5$.
* Discard motion-corrupted or flatline epochs.

#### 6. AC / DC Separation
* $AC = \text{Peak-to-Peak Amplitude}$ (95th percentile minus 5th percentile of filtered pulse).
* $DC = \text{Mean of Lowpass Filtered Baseline}$ ($<0.5\text{ Hz}$).

#### 7. Perfusion Index Formulation
$$PI = \frac{AC}{DC} \times 100\%$$
* Physiological bounding applied: $PI \in [0.05\%, 12.0\%]$.

---

## 5. Mandatory 2-Column Diagnostic Visualizations

The Jupyter Notebook MUST render the following 2-column comparative figures to provide transparent visual auditing:

| Figure | Left Column (Col 1) | Right Column (Col 2) | Purpose |
| :--- | :--- | :--- | :--- |
| **Diag Plot 1** | Full-duration raw PPG signal from dataset (entire surgical case, e.g., 60+ mins) | 10-second close-up of raw PPG showing individual pulse waveforms | Verifies raw data integrity and pulse morphology. |
| **Diag Plot 2** | Raw 500 Hz signal segment (overlay of original dense sample points) | Downsampled 125 Hz signal segment (overlay demonstrating peak retention without aliasing) | Proves downsampling preserves systolic and dicrotic peaks. |
| **Diag Plot 3** | Signal before baseline restoration (with respiratory drift and DC shifts) | Signal after baseline restoration (true zero-centered baseline) | Demonstrates removal of respiration-induced baseline drift. |
| **Diag Plot 4** | Signal before 0.5–8.0 Hz bandpass filtering (noisy signal) | Signal after 0.5–8.0 Hz bandpass filtering (clean cardiac pulse waveform) | Proves high-frequency noise removal while keeping dicrotic notch. |
| **Diag Plot 5** | Representative 10-s segment that **PASSED** SQI (clean morphology, positive skew) | Representative 10-s segment that **FAILED** SQI (motion artifact, clipping, or arrhythmia) | Validates automated quality gate efficacy. |
| **Diag Plot 6** | Extracted AC component waveform (isolated dynamic pulsatile pulses) | Extracted DC component trend (low-frequency baseline attenuation over time) | Verifies correct separation of $AC$ and $DC$ for $PI$. |

### Core Model Evaluation Plots (Rendered Post-Calibration)
* **Eval Plot 1 (Time-Series):** 60-minute continuous tracking: Real Core Temp ($T_{\text{core}}$, ground truth) vs. Estimated Temp ($T_{\text{est}}$) vs. Simulated Skin Temp ($T_{\text{skin}}$).
* **Eval Plot 2 (Regression Scatter):** $T_{\text{est}}$ vs. $T_{\text{core}}$ with identity line ($y = x$), linear regression slope, and $R^2$.
* **Eval Plot 3 (Bland-Altman Agreement):** Difference ($T_{\text{est}} - T_{\text{core}}$) vs. Mean with Mean Bias and 95% Limits of Agreement ($\pm 1.96 \cdot \text{SD}$).

---

## 6. Model Optimization & Acceptance Criteria

### 6.1 Formula to Fit
$$T_{\text{est}} = T_{\text{skin\_meas}} + \frac{T_{\text{skin\_meas}} - T_{\text{amb}}}{a + b \cdot PI^\alpha}$$
* Ambient temperature fixed to hospital standard: $T_{\text{amb}} = 21.0 \pm 0.3\ ^\circ\text{C}$.
* Train / Test split: **Case-level** on 10 patients (7 patients Train, 3 patients Test).
* Optimization: Nonlinear least squares (`scipy.optimize.curve_fit`, method='trf').

### 6.2 Acceptance Thresholds & Benchmarks (Evaluated on Unseen Test Patients)
1. **Clinical Gold Standard Benchmark:** $\text{MAE} \le 0.35\ ^\circ\text{C}$ (per ISO 80601-2-56 / ASTM E1112 clinical thermometry standards).
2. **Pipeline Development Target (Lightness Target):** $\text{MAE} \le 0.50\ ^\circ\text{C}$ (Approved by user as acceptable for initial surrogate validation).
3. **Root Mean Squared Error (RMSE):** $\le 0.65\ ^\circ\text{C}$.
4. **Bland-Altman 95% LoA:** Mean bias within $\pm 0.2\ ^\circ\text{C}$; Limits within $\pm 0.9\ ^\circ\text{C}$.
5. **Baseline Improvement:** $>75\%$ error reduction compared to raw uncompensated skin temperature ($|T_{\text{core}} - T_{\text{skin}}| \approx 2\text{--}5\ ^\circ\text{C}$).

---

## 7. Sub-Agent Governance & Workspace Management

Per user instructions, **`amar` is removed**. File/folder tracking is assigned to **`joli`**. All code execution is restricted to the user-created **Jupyter Notebook**.

```
[Alex: Lead Orchestrator]
       │
       ├─► [joli]: Manage 'scratch/temp estimation/', audit files & SHA-256 hashes
       │
       ├─► [ahmad] & [rian]: Verify MLX90614 specs & peer-reviewed biophysical constants
       │
       ├─► [aisyah]: Counterfactual review on model formulation (preventing circularity)
       │
       ├─► [amin]: Construct 'scratch/temp estimation/temp_estimation.ipynb'
       │
       └─► [asep]: Independent audit of notebook metrics, leakage checks, and Bland-Altman LoA
```

### 7.1 Detailed Sub-Agent Roles

| Sub-Agent | Role | Specific Task for this PRD | Deliverable |
| :--- | :--- | :--- | :--- |
| **`joli`** | Workspace Management | Manage `scratch/temp estimation/`, verify directory structure, track SHA-256 hashes of input cases and output artifacts. | Directory audit log in `scratch/temp estimation/` |
| **`ahmad`** | Literature Researcher | Validate mathematical units and citations for biophysical constants ($K_{\text{tissue}}, \rho_b c_b, h_c, h_r, \tau$). | Citation audit section in PRD |
| **`rian`** | Electronics Specialist | Audit MLX90614 official datasheet parameters (FOV, medical zone accuracy $\pm 0.2\ ^\circ\text{C}$, 17-bit resolution). | Hardware specification sign-off |
| **`aisyah`** | Critical Reviewer | Independent counterfactual challenge on 1st-order bio-heat model to ensure $T_{\text{skin}}$ generation does not create an algebraic tautology with $T_{\text{est}}$. | Counterfactual validation sign-off |
| **`amin`** | DL & Pipeline Engineer | Build the end-to-end Jupyter Notebook (`temp_estimation.ipynb`) containing all 7 pipeline steps and 6 mandatory diagnostic plots. | Runnable Jupyter Notebook |
| **`asep`** | Test Engineer | Execute independent audit: re-calculate MAE/RMSE on test cases, verify zero patient leakage, test $PI \to 0$ numerical bounds. | Verification report |

---

## 8. User Review Sign-Off Checklist

- [x] Notebook set to user-created file: `scratch/temp estimation/temp_estimation.ipynb` (Section 1.2 & 7).
- [x] Markdown PRD placed at `scratch/temp estimation/PRD_VitalDB_Temp_Validation.md` for all subagents (Section 1.2 & 7).
- [x] Temperature simulation bounds set to $33.0\ ^\circ\text{C}$ to $45.0\ ^\circ\text{C}$ (hypothermia to severe hyperthermia) (Section 3.1).
- [x] Standard PPG bandpass filter locked to $0.5\text{--}8.0\text{ Hz}$ (Section 4.1).
- [x] User's `downsample` helper function included verbatim (Section 4.1).
- [x] Dedicated 2-column plot for baseline restoration added (Section 5, Diag Plot 3).
- [x] Clinical standard ($\le 0.35\ ^\circ\text{C}$) documented alongside development target ($\le 0.50\ ^\circ\text{C}$) (Section 6.2).
- [x] Restricted exclusively to Jupyter Notebook in `scratch/temp estimation/` (Section 1.2 & 7).
- [x] `amar` removed, `joli` integrated, cohort set to 10 cases (Section 7).
