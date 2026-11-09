# Primary Rules & Instructions

> [!IMPORTANT]
> 1. **Never download or make anything that is not explicitly instructed.**
> 2. **Apply Ponytail principles** in making code (`C:\Users\Adyty\.gemini\config\plugins\ponytail\skills\ponytail\SKILL.md`):
>    - Minimal dependencies and standard library first.
>    - Zero speculative abstractions and zero boilerplate (YAGNI).
>    - Shortest working code wins.
>    - Local files only; never pull or download remote data without direct instruction.
>    - Only render signal plots via `plt.show()`; never save PNG/image files to disk unless explicitly told.
> 3. **Workspace Organization & File Placement Rules**:
>    - Root directory (`scratch/gemastik/`): Holds ONLY foundational references (`project_notes.md`, proposal PDF).
>    - `scripts/`: Holds all Python automation, forensic audit, preprocessing, and test scripts (`.py`). No loose scripts in root.
>    - `reports/`: Holds finalized analytical findings, audit sign-offs, and summary contracts (`.md`, `.json`).
>    - `logs/`: Holds stdout dumps, intermediate debug traces, and inspect logs (`.txt`, `.log`).
>    - Specialized experiment directories (e.g. `Bp revisi/`, `vitaldb_rr_test/`): Must remain isolated and self-contained.

---

# Project Technical Notes: Master Reference & System Architecture

Technical reference and complete audit matrix for the latest 29-page proposal document (`Gemastik para pencari berkah.pdf`, updated 2026-10-29 00:02:53) and the `TriaGo` system implementation.

---

## 1. System Overview & Executive Summary

* **Project Name**: **TriaGo** (Emergency Triage & Vital Signs Multi-Sensor System).
* **Team**: Para Pencari Berkah (Biomedical Engineering, Institut Teknologi Sepuluh Nopember).
* **Clinical Purpose**: Solve Emergency Department (IGD) overcrowding, reduce nurse cognitive fatigue, eliminate triage subjectivity, and automate patient bed allocation in under 1 minute.
* **Form Factor**: Two ergonomic arm/hand clamps resembling standard ECG limb clamps:
  * **Right Clamp**: Negative ECG lead, MLX90614 infrared thermopile, MAX30105 optical PPG sensor.
  * **Left Clamp**: Positive ECG lead and Driven Right Leg (DRL) ground.
* **Core Computing**: STM32 Black Pill (ARM Cortex-M4 acquisition @ 400 Hz) $\rightarrow$ USB Serial $\rightarrow$ Raspberry Pi 5 (Quad-core Cortex-A76, 16 GB RAM) running TFLite & ONNX runtimes $\rightarrow$ 10" IPS LCD touchscreen (1280x800).
* **Connectivity & Cloud**: MQTT over TLS (QoS 1) to HiveMQ Cloud $\rightarrow$ Flask Backend $\rightarrow$ MySQL Database $\rightarrow$ WebSocket (Socket.IO) to real-time IGD Web Dashboard deployed at `https://bme.its.ac.id/informatics/TriaGo/` via Nginx reverse proxy.

---

## 2. Master Parameter Pipelines & Mathematical Formulations

### 2.1 Heart Rate (HR) Pipeline
* **Sensor**: MikroE ECG Click (MCP609 IC, analog filter stages, DRL circuit).
* **Sampling**: STM32 ADC (12-bit) at $400\text{ Hz}$ raw, downsampled to $125\text{ Hz}$.
* **Conditioning**: $50\text{ Hz}$ IIR notch filter ($Q = 30$), double median filter cascade ($0.2\text{ s}$ and $0.6\text{ s}$ baseline detrending), $35\text{ Hz}$ 3rd-order Butterworth lowpass, Savitzky-Golay smoothing ($w=11, p=2$).
* **QRS Detection**: Pan-Tompkins algorithm ($5 - 18\text{ Hz}$ BPF, 5-point derivative kernel, squaring, flat-top window $60\text{ ms}$, MWI $150\text{ ms}$, adaptive dual thresholds with slope-based T-wave rejection and search-back).
* **Formula (Eq. 2.1)**:
  $$HR = \frac{60}{\overline{RR}_{\text{interval}}}$$
* **Validation**: BIDMC dataset (53 subjects), $\text{MAE} = 0.695\text{ BPM}$, Accuracy $= 99.15\%$ (AAMI EC13 compliant).

### 2.2 Respiratory Rate (RR) Pipeline
* **Sensors**: Multimodal cross-derivation from ECG R-peaks and PPG pulse envelopes.
* **Signal Conditioning**: Extracts Amplitude Modulation (AM), Frequency Modulation (FM), and Baseline Wander (BW). PCHIP interpolation at $4.0\text{ Hz}$, filtered in physiological band ($0.167 - 0.70\text{ Hz} = 10 - 42\text{ breaths/min}$).
* **Frequency Analysis**: Welch Power Spectral Density (PSD) weighted by Respiratory Quality Index (RQI) to isolate dominant frequency $f_{peak}$.
* **Formula (Eq. 2.2)**:
  $$RR = 60 \times f_{peak}$$
* **Validation**: BIDMC dataset (53 subjects), $\text{MAE} = 0.96\text{ RPM}$, Accuracy $= 92.32\%$ (ISO 80601-2-87 compliant).

### 2.3 Oxygen Saturation (SpO2) Pipeline
* **Sensor**: MAX30105 dual-wavelength reflective optical sensor (Red $660\text{ nm}$, IR $880\text{ nm}$, I2C).
* **Conditioning**: Signal inversion ($x_{inv} = -x$), local trough detection + `CubicSpline` baseline isolation, pulsatile AC extraction filtered via 2nd-order Butterworth ($0.5 - 15.0\text{ Hz}$) and Savitzky-Golay ($w=11, p=2$).
* **Formulas**:
  * Perfusion Index (Eq. 2.6): $PI = \frac{AC}{DC} \times 100\%$
  * Ratio-of-Ratios (Eq. 2.3): $R = \frac{AC_{red} / DC_{red}}{AC_{ir} / DC_{ir}}$
  * **Linear Calibration Curve (Eq. 2.4 - MASTER)**:
    $$SpO_2(\%) = 107.83 - 25 \cdot R$$
    *(Linear calibration prevents the ~98% saturation ceiling seen in quadratic curves and preserves hypoxemia detection sensitivity).*
* **Validation**: The University of Sydney dataset (22 subjects), $\text{MAE} = 1.08\%$, $\text{RMSE} = 1.47\%$, Accuracy $= 98.89\%$ (ISO 80601-2-61 compliant).
* **Divergence Flag**: Codebase (`processing_data.py:504`) still contains legacy quadratic curve ($3.0694 R^2 - 5.1491 R + 99.7943$).

### 2.4 Core Body Temperature (CBT) Pipeline
* **Sensors**: Melexis MLX90614 (IR Thermopile: object temp $T_s$ + ambient temp $T_{amb}$ @ $2\text{ Hz}$), MAX30105 ($PI$).
* **Bioheat Formula (Eq. 2.5)**:
  $$T_c = T_s + \frac{T_s - T_{amb}}{a + b \times PI} + C_{offset}$$
  * Parameters: $a = 0.82991$, $b = 0.97126$, $C_{offset} = 1.0649^\circ\text{C}$.
* **Validation**: VitalDB 100 cases (80 train / 20 test), test $\text{MAE} = 0.237^\circ\text{C}$ (80% pass), overall $\text{MAE} = 0.250^\circ\text{C}$ (83% pass, ISO 80601-2-56 compliant).
* **Divergence Flag**: TriaGo GUI (`loading_page.py:160`) uses linear bioheat adaptive formula: $T_c = T_s + k_{env\_adaptive}(T_s - T_{amb}) + C_{offset}$.

### 2.5 Cuffless Blood Pressure (BP) Pipeline
* **Architecture**: Deep learning hybrid 1D-CNN + Bidirectional GRU (BiGRU) + Multi-Head Self-Attention (MHSA) with separate regression heads for SBP and DBP.
* **Input Window**: 7-channel multimodal window ($10\text{ s}$ at $125\text{ Hz} = 1250\text{ samples}$): ECG Lead I, PPG Red, PPG IR, PPG Green, VPG (1st derivative), APG (2nd derivative), PTT temporal features.
* **Training & Calibration**: Transfer learning from PulseDB benchmark to TriaGo primary dataset (94 unique biological humans, aged 18–22) with 1-point anchor calibration.
* **Verified Performance (Proposal Table 4.1 & `transfer_learning_summary.json`)**:
  * **Systolic (SBP)**: $\text{MAE} = 4.90\text{ mmHg}$, $\text{SD} = 6.22\text{ mmHg}$ ($\le 5\text{ mmHg}: 59.76\%$, $\le 10\text{ mmHg}: 87.23\%$, $\le 15\text{ mmHg}: 97.35\%$) $\rightarrow$ **BHS Grade B**.
  * **Diastolic (DBP)**: $\text{MAE} = 2.30\text{ mmHg}$, $\text{SD} = 3.22\text{ mmHg}$ ($\le 5\text{ mmHg}: 89.40\%$, $\le 10\text{ mmHg}: 98.07\%$, $\le 15\text{ mmHg}: 100.00\%$) $\rightarrow$ **BHS Grade A**.
* **Deployment Target**: TensorFlow Lite on Raspberry Pi 5.

### 2.6 Automated Triage Classification (XGBoost)
* **Standard**: Permenkes RI No. 47 Tahun 2018 (3 target classes: Resusitasi / Kritis, Darurat, Non-Darurat; Hitam/DOA excluded).
* **Features**: Vital signs (HR, RR, SpO2, CBT, SBP, DBP) + Consciousness (GCS) + Pain Score.
* **Safety Rules**: Hard-coded escalation overrides for severe hypoxemia ($SpO_2 < 90\%$) or impaired consciousness ($GCS \le 8$).
* **Comparative Benchmarks (Table 4.2)**:
  * Linear Regression: Acc $= 88.48\%$, F1 $= 89.40\%$, Latency $= 0.0005\text{ ms/sample}$
  * Support Vector Machine: Acc $= 89.11\%$, F1 $= 89.92\%$, Latency $= 0.3280\text{ ms/sample}$
  * Random Forest: Acc $= 88.45\%$, F1 $= 89.29\%$, Latency $= 0.0042\text{ ms/sample}$
  * Decision Tree: Acc $= 76.13\%$, F1 $= 77.74\%$, Latency $= 0.0037\text{ ms/sample}$
  * **XGBoost (Selected)**: Acc $= 90.67\%$, F1 $= 91.42\%$, Latency $= 0.0037\text{ ms/sample}$
* **Per-Class Clinical Diagnostics (Table 4.3)**:
  * **Resusitasi**: Sensitivity $= 0.925$, Specificity $= 0.998$, PPV $= 0.948$, NPV $= 0.997$
  * **Darurat**: Sensitivity $= 0.878$, Specificity $= 0.887$, PPV $= 0.897$, NPV $= 0.866$
  * **Non-Darurat**: Sensitivity $= 0.883$, Specificity $= 0.890$, PPV $= 0.858$, NPV $= 0.910$
* **Deployment Target**: ONNX Runtime on Raspberry Pi 5 (`TriageOnnxModel.predict`).

---

## 3. IoT Architecture & Automated Bed Management (Section 4.5)

* **Transport Layer**: MQTT over TLS with QoS 1 published to HiveMQ Cloud broker.
* **Backend Processing**: Flask backend subscribes to MQTT topics, validates payloads, enforces deduplication, and stores records in MySQL.
* **Bed Allocation Logic**:
  * **Resusitasi (Merah)**: Beds 01 – 03
  * **Darurat (Kuning)**: Beds 04 – 09
  * **Non-Darurat (Hijau)**: Beds 10 – 12
  * **Overflow Rule**: When primary zones are full, Red and Yellow patients re-route automatically to open beds in other zones; Green patients enter a waiting state.
* **Feedback Loop to Device**: Flask backend publishes allocated bed number and triage zone back to Raspberry Pi 5 via MQTT, updating the TriaGo screen instantly without nurse intervention.
* **Hospital Dashboard**: Real-time push via WebSocket (Socket.IO):
  * **Meja Perawat**: Nurse intake desk view to append patient demographic identity and handle bed discharge.
  * **Monitor IGD**: High-visibility wall-mount dashboard showing color-coded bed map and response-time countdown timers.
* **Live Deployment URL**: `https://bme.its.ac.id/informatics/TriaGo/` (served via Nginx reverse proxy with HTTPS).

---

## 4. Competitive Solutions & Economic Value (Section 4.6, Table 4.5)

| Instrument | Physiological Parameters | Triage Engine | Price (IDR) | Value Comparison |
| :--- | :--- | :--- | :--- | :--- |
| **Mindray VS 9** | HR, RR, SpO₂, NIBP (cuff), Temp | Early Warning Score (EWS) | Rp37.975.000 | Commercial hospital standard; pneumatic cuff |
| **Welch Allyn Connex Spot Monitor** | RR, SpO₂, NIBP (cuff), Temp | Early Warning Score (EWS) | Rp38.127.600 | High-end clinical standard; pneumatic cuff |
| **GE CARESCAPE V100** | HR, RR, SpO₂, NIBP (cuff), Temp | None (spot check only) | Rp20.720.000 | Legacy vital monitor; no triage logic |
| **TriaGo (Gemastik)** | **HR, RR, SpO₂, NIBP (cuffless), Temp** | **3-Class Permenkes AI + Bed Alloc** | **Rp15.000.000** | **< 40% cost of commercial EWS units; cuffless AI** |

---

## 5. Hardware Thermal & Power Performance (Section 4.7)

* **Power Source**: Single 20,000 mAh PowerBank supplying both Raspberry Pi 5 (5V) and 10" IPS LCD (12V/5V).
* **Thermal Envelope**:
  * **Idle Temperature**: Raspberry Pi 5 runs at steady **$33^\circ\text{C}$**.
  * **Active Computing Temperature**: During concurrent high-speed acquisition (400 Hz), digital filtering, TFLite CNN+BiGRU inference, and ONNX classification, peak temperature reaches only **$37^\circ\text{C}$**.
* **Operational Battery Life**:
  * Measured system power consumption is **~9.5W**.
  * Continuous battery runtime is verified at **$5.5\text{ to } 6.0\text{ hours}$**, fulfilling hospital shift readiness requirements.

---

## 6. Expert Medical Validation (Section 4.8)

* **Expert Consultant**: **dr. Saskia Diva Kamila** (General Practitioner, Medical Centre ITS).
* **Clinical Evaluation**:
  * Validated that TriaGo directly targets acute ED pain points: surging intake volume, nurse mental fatigue, and triage classification subjectivity.
  * Emphasized that automated bed management significantly shortens patient boarding delays.
  * Commended the dual-clamp design for enabling one-minute multi-vital acquisition without cumbersome multiple cords and cuffs.

---

## 7. Master Synchronization & Divergence Audit

| Subsystem / Pipeline | Proposal Spec (2026-10-29) | Codebase Implementation | Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Heart Rate** | Eq. 2.1 ($60 / \overline{RR}$) | `processing_data/processing_data.py:246` | **100% Synced** | Pan-Tompkins 125 Hz |
| **Respiratory Rate** | Eq. 2.2 ($60 \times f_{peak}$) | `processing_data/processing_data.py:322` | **100% Synced** | Multimodal AM/FM/BW + Welch |
| **SpO2 Calibration** | Eq. 2.4 ($107.83 - 25 \cdot R$) | `processing_data/processing_data.py:504` | **DIVERGED** | Codebase runs legacy quadratic fit |
| **CBT Temperature** | Eq. 2.5 ($T_s + \frac{\Delta T}{a+b\cdot PI} + C$) | `GUI/loading_page.py:160` | **DIVERGED** | Codebase runs $k_{env}$ adaptive linear fit |
| **Cuffless BP** | 1D-CNN+BiGRU+MHSA (Table 4.1) | `Bp revisi/checklist_eval.py` | **100% Synced** | Transfer learning matches BHS A/B |
| **XGBoost Triage** | 3-Class Permenkes (Table 4.2) | `model/deployment_inference.py:16` | **100% Synced** | ONNX runtime inference |
| **IoT Architecture** | MQTT HiveMQ + Socket.IO | Backend scripts & GUI worker | **100% Synced** | Fully aligned across all chapters |
| **Bed Management** | Beds 01-03 (Red), 04-09 (Yellow), 10-12 (Green) | Dashboard Backend | **100% Synced** | Integrated into `bme.its.ac.id` |
