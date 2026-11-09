# Comprehensive Writing, Stylistic, and Editorial Audit Report
**Target Document**: `Gemastik para pencari berkah.pdf` (TriaGo Proposal, Gemastik XIX 2026, 29 Pages)  
**Auditor**: Amar (Technical Writer & Thesis Specialist, Squad 3)  
**Task ID**: `TASK-WRITING-AUDIT-01`  
**Date**: 2026-10-29  
**Reference Standards**: EYD V (Kemendikdasmen RI), Pedoman Penulisan Ilmiah Gemastik XIX, IEEE Citation Standard, Humanizer Skill v3.1.0  

---

## 1. Executive Summary & Audit Scorecard

An exhaustive, multi-layered forensic audit was executed across the full extracted proposal text (`scratch/gemastik/logs/proposal_extracted_text.txt`), individual chapter dumps (`bab1_new.txt` through `bab5_new.txt`), and PDF typographic font properties via PyMuPDF.

| Audit Dimension | Status | Severity | Primary Findings Summary |
| :--- | :---: | :---: | :--- |
| **1. Typographical & Spelling** | **FAILED** | **HIGH** | 12 critical typos (`melakuakan`, `meskipul`, `prakti`, `performma`, `dihilangan`, `menyuruh`, `dijalakan`, `Spesifitas`, `rata- rata`, etc.). |
| **2. EYD V / PUEBI Adherence** | **FAILED** | **HIGH** | Pervasive upright foreign terms in PDF (`real-time`, `backend`, `broker`, `noise`, `ground`, etc.); hyphenation errors in bound morphemes (`antar-`, `non-`). |
| **3. Figures & Tables Alignment** | **FAILED** | **CRITICAL** | Complete domino off-by-one table desynchronization in Bab 4; misdirected figure callouts (Gambar 4.6 & 4.7 duplicate callouts); Bab IV heading omitted in Daftar Isi. |
| **4. Scientific Numbers & Units** | **FAILED** | **CRITICAL** | Decimal comma vs dot schism (Chapters 1–2 & 4.3.3–4.3.6 use comma; 4.3.7–4.3.8 & Table 4.4 use dot); XGBoost metric contradiction (text says 88.20%/89.83% vs table/abstract 90.67%/91.42%). |
| **5. Tone & Humanizer Tells** | **FLAGGED** | **MEDIUM** | Classic AI staging patterns (§1 *Not X but Y*); forced repetitive SDG padding (SDG 3, 8, 9, 12 repeated in 4 separate sections); pleonastic and tautological sentences. |
| **6. Citations & References** | **FLAGGED** | **MEDIUM** | Ref [2] includes institutional affiliations as co-authors; Ref [25] is a marketing paper cited for hospital IoT; web scraper artifacts in Ref [12]; uncapitalized proper nouns in Ref [6]. |

---

## 2. Typographical & Lexical Error Inventory

The audit identified 14 distinct spelling and typographical defects. Every entry has been mapped to its chapter, page, and exact context:

| No | Location | Extracted Text | Corrected Form | Category & Diagnosis |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **Bab 2, Hal. 10** | `...sehingga pesan tetap terkirim meskipul jaringan sempat terputus.` | `meskipun` | Critical typo: consonant substitution (`l` for `n`). |
| 2 | **Bab 2, Hal. 11** | `...menerima data hasil triase dengan melakuakan subscribe...` | `melakukan` | Critical typo: vowel intrusion (`ua` for `u`). |
| 3 | **Bab 3, Hal. 12** | `...menggunakan model deep learning dijalakan di Raspberry Pi 5...` | `dijalankan` | Critical typo: omitted nasal consonant (`n`). |
| 4 | **Bab 3, Hal. 12** | `...menggunakan Tensor Flow Lite, sedangkan model XGboost...` | `TensorFlow Lite`, `XGBoost` | Software branding errors: unneeded space in TensorFlow; lowercase `b` in XGBoost. |
| 5 | **Bab 3, Hal. 12** | `Pembacaan data digital up to 18-bit ADC...` | `hingga 18-bit ADC` | Untranslated English fragment in Indonesian specification table. |
| 6 | **Bab 4, Hal. 19** | `...masih ada dihilangan melalui penerapan bandpass filter...` | `dihilangkan` | Critical typo: omitted consonant (`k`). |
| 7 | **Bab 4, Hal. 20** | `Jika dilihat secara menyuruh pada 100 kasus...` | `menyeluruh` | Critical semantic error: `menyuruh` (to command) substituted for `menyeluruh` (overall). |
| 8 | **Bab 4, Hal. 22** | `Tabel 4.2 Perbandingan performma model machine learning.` | `performa` | Critical typo in table title: duplicated consonant (`mm`). |
| 9 | **Bab 4, Hal. 22** | `Xtreme Gradien Boost` | `eXtreme Gradient Boosting` | Hybrid spelling error in table row: Indonesian `Gradien` mixed with English `Boost`. |
| 10 | **Bab 4, Hal. 22** | `Spesifitas` (Header Kolom Tabel 4.3) | `Spesifisitas` | Morphological error: omitted syllable `-si-` in statistical metric. |
| 11 | **Bab 4, Hal. 25** | `Sistem yang dikembangkan meminimalisir penggunaan...` | `meminimalkan` / `meminimalisasi` | Non-standard loanword: KBBI and EYD V do not recognize `-isir` suffix. |
| 12 | **Bab 5, Hal. 27** | `...dikembangkan dalam bentuk dua penjepit prakti menyerupai...` | `praktis` | Critical typo: dropped terminal consonant (`s`). |
| 13 | **Bab 1, Hal. 5** | `...membutuhkan rata- rata 4 menit 11 detik...` | `rata-rata` | Typographic error: extraneous space inside hyphenated reduplication. |
| 14 | **Abstrak, Hal. 2** | `...akurasi 90,67% dan F1- Score 91,42%...` | `F1-Score` | Spacing error: extraneous space after hyphen. |

---

## 3. PUEBI & Indonesian Academic Writing Standards (EYD V)

### 3.1. Foreign Term Italicization (PyMuPDF Font Inspection)
Under EYD V Bab III Huruf E, foreign words, technical terms, and phrases not yet absorbed into Indonesian must be italicized (*huruf miring*). Inspection of PDF font spans revealed severe inconsistencies between `TimesNewRomanPS-ItalicMT` and upright `TimesNewRomanPSMT`:

| Term | Total Appearances | Italic Count | Upright Count | Defective Instances in PDF |
| :--- | :---: | :---: | :---: | :--- |
| `real-time` | 10 | 9 | **1** | Hal. 2 (Abstrak): `...dashboard IGD secara real-time. Hasil...` is upright. |
| `backend` | 7 | 3 | **4** | Hal. 11 (2.5.2) & Hal. 13 (3.1.2): `Backend berbasis Flask...` and `oleh backend Flask...` are upright. |
| `broker` | 4 | 0 | **4** | Hal. 10, 13, 17: `ke broker HiveMQ Cloud` is completely upright throughout. |
| `ground` | 3 | 3 | 0 | Correctly italicized in PDF, but should be evaluated whether to use Indonesian term *pembumian* / *ditanahkan*. |
| `noise` | 2 | 0 | **2** | Hal. 18 (4.2.3): `terdapat noise frekuensi tinggi` & `Masalah noise ini` are upright (should be *noise* or *derau*). |
| `bandpass` | 3 | 2 | **1** | Hal. 19 (4.3.2): `penerapan bandpass filter` is upright. |
| `baseline` | 3 | 2 | **1** | Hal. 19 (4.3.2): `kembali ke baseline aslinya` is upright. |
| `cut-off` / `cut off` | 3 | 0 | **3** | Hal. 8, 18, 19: `cut-off 5 – 18 Hz`, `cut off 35 Hz` are upright and spelled inconsistently. |
| `hybrid` | 1 | 0 | **1** | Hal. 14 (3.2.1): `dibangun secara hybrid` is upright (should be *hibrida* or italic *hybrid*). |
| `decision tree` | 1 | 0 | **1** | Hal. 14 (3.2.2): `berbasis decision tree` is upright. |
| `bed layout` | 1 | 0 | **1** | Hal. 16 (3.3): `(bed layout)` is upright. |

### 3.2. Bound Elements (*Unsur Terikat*)
Under EYD V Bab II Huruf D, bound combining elements such as `antar-` and `non-` must be written joined with the base word without spaces or hyphens (except when followed by capitalized words or acronyms):
- **Defects Found**:
  1. `antar-petugas` (Hal. 5, Bab 1) $\rightarrow$ MUST be `antarpetugas`.
  2. `antar-komponen` (Hal. 13, Bab 3) $\rightarrow$ MUST be `antarkomponen`.
  3. `antar detak` (Hal. 9, Bab 2) $\rightarrow$ MUST be `antardetak`.
  4. `antar puncak` (Hal. 8, Bab 2; Hal. 18, Bab 4) $\rightarrow$ MUST be `antarpuncak`.
  5. `antar siklus` (Hal. 14, Bab 3) $\rightarrow$ MUST be `antarsiklus`.
  6. `antar-segmen` (Hal. 14, Bab 3) $\rightarrow$ MUST be `antarsegmen`.
  7. `non-invasif` (Hal. 5, 7, 10, 12, 16) $\rightarrow$ MUST be `noninvasif`.
  8. `non-linier` (Hal. 20, Bab 4) $\rightarrow$ MUST be `nonlinear` (KBBI standard).

### 3.3. Capitalization and Punctuation
1. **Inconsistent Triage Capitalization**: Permenkes triage classes are capitalized as official categories (`Resusitasi`, `Darurat`, `Non-Darurat`), but on page 9 it is written as `kategori hitam` (lowercase), and on page 14 as `darurat dan non-darurat` (lowercase). Standardize to Title Case for all defined triage classes.
2. **Missing Oxford Serial Commas**: In perincian $\ge 3$ items, EYD V requires a comma before the conjunction:
   - Hal. 2: `...sensor suhu inframerah MLX90614 dan modul...` $\rightarrow$ add comma.
   - Hal. 5: `...suhu tubuh, denyut jantung, laju pernapasan, saturasi oksigen dan tekanan darah.` $\rightarrow$ add comma before `dan`.
   - Hal. 18: `...(b) Dokumentasi pengukuran dan (c) Termo Gun` $\rightarrow$ add comma before `dan`.
3. **Punctuation in Captions**: Captions end inconsistently: Gambar 3.6, Gambar 5.1, and Tabel 4.5 have no period, while Gambar 4.1 has a trailing dot after the number (`Gambar 4.1. Pengambilan...`). Standardize across all captions.

---

## 4. Figure & Table Numbering and In-Text Callout Discrepancies

### 4.1. Critical Domino Table Desynchronization in Chapter 4
There is an off-by-one domino callout failure across the entire second half of Chapter 4:

```
[In-Text Callout L0230]: "tabel 4.3"  ----MISDIRECTED----> [Actual Caption L0242]: "Tabel 4.2 Perbandingan performma..."
[In-Text Callout L0232]: "Tabel 4.4"  ----MISDIRECTED----> [Actual Caption L0269]: "Tabel 4.3 Performa setiap tingkat triase"
[In-Text Callout L0319]: "Tabel 4.5"  ----MISDIRECTED----> [Actual Caption L0320]: "Tabel 4.4. Hasil performa masing-masing..."
[In-Text Callout L0394]: "Tabel 4.5"  ====DUPLICATE======> [Actual Caption L0398]: "Tabel 4.5. Perbandingan dengan alat..."
```

**Detailed Discrepancy Breakdown**:
1. **L0230 Callout**: The text states `...seperti yang ditunjukkan pada tabel 4.3. dengan akurasi sebesar 88.20%...`. The table immediately following is **Tabel 4.2** (Machine Learning Comparison).
2. **L0232 Callout**: The text states `...performa model per tingkat triase ditampilkan pada Tabel 4.4. Evaluasi metrik...`. The table immediately following is **Tabel 4.3** (Per-Class Triage Diagnostics).
3. **L0319 Callout**: The text states `...secara lengkap pada Tabel 4.5 berikut.` The table immediately following is **Tabel 4.4** (Overall System Parameter Performance).
4. **L0394 Callout**: The text again states `...dapat dilihat pada Tabel 4.5 berikut.`, referencing **Tabel 4.5** (Commercial Competitor Price & Feature Comparison). Thus, `Tabel 4.5` is called twice in the prose to refer to two completely different tables!

### 4.2. Chapter 4 Figure Misdirection
1. **L0128 Callout**: Text refers to `sebagaimana ditunjukkan pada Gambar 4.6. Frekuensi dominan (f_peak) yang diperoleh...`. However, the figure right below it at L0136 is **Gambar 4.7 Spektrum Daya (PSD) domain frekuensi sinyal respirasi**. The text accidentally duplicates `Gambar 4.6` (which was already used for Pan-Tompkins R-peak detection at L0121).
2. **L0149 Callout**: Text refers to `Hasil dari pemisahan komponen AC dan DC ditunjukkan pada Gambar 4.7 yang memisahkan...`. However, the figure right below it at L0156 is **Gambar 4.8 Perbedaan komponen DC dan AC sinyal PPG**.

### 4.3. Preliminary Pages (Daftar Isi) Deficiencies
1. **Missing Chapter IV Header**: In the Table of Contents (`DAFTAR ISI`, Hal. 3–4), section 3.3 is followed immediately by section 4.1. The major heading **`BAB IV ANALISIS FUNGSIONAL`** (or `BAB IV HASIL DAN PEMBAHASAN`) is completely omitted from the TOC list.
2. **Omission of Lists of Figures and Tables**: The proposal provides no `DAFTAR GAMBAR` and no `DAFTAR TABEL`, despite containing 18 distinct figures and 7 tables.

---

## 5. Scientific Numbers, Decimals, Units, and Mathematical Symbols

### 5.1. The Decimal Comma vs Decimal Dot Schism
Under EYD V Bab III Huruf G, standard Indonesian academic prose requires the comma (`,`) as the decimal separator and the period (`.`) as the thousands separator. The proposal exhibits a sharp schism where parts of the text adhere to Indonesian convention, while sections generated or copied from code outputs retain raw Python decimal dots:

- **Indonesian Comma Usage (Sections 1.1, 2.2, 4.3.3–4.3.6, 4.7)**:
  `99,15%`, `0,695 BPM`, `92,32%`, `0,96 RPM`, `98,89%`, `1,08%`, `1,47%`, `0,237 ℃`, `0,250 ℃`, `5,5 jam`.
- **English Dot Usage (Sections 4.3.7, 4.3.8, Table 4.1, Table 4.2, Table 4.3)**:
  `4.90 mmHg`, `2.3 mmHg`, `6.22 mmHg`, `3.22 mmHg`, `88.20%`, `89.83%`, `0.925`, `0.998`, `0.948`, `0.878`, `0.897`, `0.883`, `0.910`.
- **Intra-Table Hybrid Clashing (Tabel 4.4, Hal. 23)**:
  Within the very same table, row 1–3 use English dots (`0.695 BPM`, `0.96 RPM`, `1.08%`), while row 4–6 use Indonesian commas (`0,250 ℃`, `68,8% / 83,9% / 96,4%`, `73,2% / 90,2% / 95,5%`).

### 5.2. XGBoost Metric Contradiction
A severe factual contradiction exists within Section 4.3.8 regarding the primary triage classification model performance:
- **Prose Claim (Hal. 22, L0230-0231)**: `...dengan akurasi sebesar 88.20% dan F1 Score sebesar 89.83%.`
- **Table 4.2 Claim (Hal. 22, L0265-0266)**: `Xtreme Gradien Boost | Akurasi: 0.9067 (90.67%) | F1-Skor: 0.9142 (91.42%)`.
- **Abstrak Claim (Hal. 2, L0024-0025)**: `Model klasifikasi XGBoost mencapai akurasi 90,67% dan F1-Score 91,42%`.
- **Resolution**: The prose in 4.3.8 contains legacy draft numbers (88.20% / 89.83%) that directly contradict both the summary table and the abstract. It must be updated to **90,67% dan 91,42%**.

### 5.3. Unit and Symbol Standardizations
1. **Degree Celsius Symbol**: Bab 2 and 3 use `°C` (degree sign U+00B0 + letter C). Bab 4 and Section 4.7 switch to the single unicode character `℃` (U+2103). Standardize on `°C` with an ISO standard non-breaking space (e.g. `37 °C`).
2. **Missing Unit Spaces**: `±0.5°C` (Hal. 12) $\rightarrow$ `±0,5 °C`; `2.4GHz` (Hal. 10) $\rightarrow$ `2,4 GHz`.
3. **Pleonastic Modifiers**: The phrase `MAE hanya 0,695 BPM saja` combines `hanya` and `saja`. Choose one: `MAE hanya 0,695 BPM` or `MAE sebesar 0,695 BPM`.
4. **Precision Inconsistency**: In L0188-0189, SBP MAE is given to two decimals (`4.90 mmHg`), while DBP MAE is given to one decimal (`2.3 mmHg`). Format symmetrically as `4,90 mmHg` and `2,30 mmHg`.

---

## 6. Tone, Flow, and Humanizer Audit

Applying the criteria from the `humanizer` skill (Wikipedia Signs of AI Writing):

### 6.1. §1 "Not X but Y" AI Staging Constructs
AI models frequently stage contrast to inflate importance without adding information:
- **Hal. 14 (3.2.2)**:
  *Current*: `Model tidak hanya dirancang secara pintar menggunakan algoritma XGBoost saja, tetapi juga dilengkapi dengan algoritma yang dapat meningkatkan kewaspadaan model...`
  *Humanized Rewrite*: `Selain klasifikasi XGBoost, model dilengkapi aturan eskalasi berbasis batas kritis saturasi oksigen dan skor GCS.`
- **Hal. 25 (4.8)**:
  *Current*: `Sistem yang dikembangkan tidak hanya membantu tenaga medis dalam mengekstraksi parameter fisiologis utama saja, melainkan juga memberikan rekomendasi keputusan triase...`
  *Humanized Rewrite*: `Sistem mengekstrak tanda vital utama sekaligus memberikan rekomendasi tingkat triase dan alokasi kasur IGD.`

### 6.2. §13 Inflated Significance & Boilerplate SDG Padding
The text forcibly injects the Sustainable Development Goals (SDGs 3, 8, 9, 12) into four separate sections: Abstrak (Hal. 2), Bab 1.1 (Hal. 5), Bab 1.3 (Hal. 6), and Bab 5.1 (Hal. 27). This repetitive grant-writing boilerplate dilutes the technical rigor of the engineering proposal. Keep the detailed SDG mapping in Section 1.3 (Urgensi dan Manfaat) and streamline the repetitive mentions in Bab 1.1 and Bab 5.1.

### 6.3. Tautology and Pleonastic Constructions
- **Hal. 2 (Abstrak)**: `TriaGo mengkategorikan triase berdasarkan Permenkes RI No. 47 Tahun 2018 yang mengkategorikan pasien...` $\rightarrow$ Tautology (`mengkategorikan... yang mengkategorikan`). Revise to: `TriaGo mengadopsi standar Permenkes RI No. 47 Tahun 2018 untuk mengklasifikasikan pasien...`.
- **Hal. 10 (2.4.1)**: `...yang bekerja dengan prinsip kerja inframerah...` $\rightarrow$ Pleonasm (`bekerja dengan prinsip kerja`). Revise to: `...yang bekerja berdasarkan radiasi inframerah...`.
- **Hal. 12 (3.1.2)**: `...bekerja secara cepat dengan hasil yang tetap optimal, walaupun dijalankan di Raspberry Pi 5.` $\rightarrow$ Raspberry Pi 5 was already established as the subject two clauses earlier.
- **Hal. 14 (3.2.2)**: `Model klasifikasi tingkat triase yang dibangun secara otomatis dibangun menggunakan...` $\rightarrow$ Repeated predicate (`dibangun... dibangun`). Revise to: `Model klasifikasi triase dibangun secara otomatis menggunakan...`.
- **Hal. 25 (4.7)**: `Dalam kondisi idle alat tidak digunakan, perangkat Raspberry Pi bekerja dengan suhu 33 ℃ dalam kondisi aktif dan tidak digunakan.` $\rightarrow$ Triple repetition of `tidak digunakan / idle`. Revise to: `Pada kondisi siaga (idle), suhu operasional Raspberry Pi 5 stabil pada 33 °C.`

---

## 7. Citation & Reference Consistency Audit (IEEE Style)

The 25 references in `DAFTAR PUSTAKA` were audited against standard IEEE styling:

| Ref No | Defect Description | Location in Text & Reference List | Remediation Action |
| :---: | :--- | :--- | :--- |
| **[2]** | **Affiliations Erroneously Parsed as Authors** | `A. Budiyanto Adi Putra, A. Adhiwijaya, E. Yustilawati, P. S. Studi, F. Kedokteran dan Ilmu Kesehatan, and U. Alauddin Makassar...` | Institutional metadata (`Program Studi`, `Fakultas Kedokteran...`, `UIN Alauddin Makassar`) was accidentally copied into the author string. Strip affiliations; leave only real authors: `A. Budiyanto Adi Putra, A. Adhiwijaya, and E. Yustilawati`. |
| **[6]** | **Uncapitalized Proper Nouns & Double Punctuation** | `“Validation of the cipto triage method: A single-centre study from indonesia.,” 2020...` | Capitalize proper nouns: `Cipto` and `Indonesia`. Remove double punctuation `.,"` $\rightarrow$ `,”`. |
| **[12]** | **Webpage Scraping Title Artifact** | `Oxygen Saturation - StatPearls - NCBI Bookshelf (nih.gov)` | Strip browser title artifact ` - NCBI Bookshelf (nih.gov)`; cite cleanly as `StatPearls`. |
| **[21]** | **Typo in Datasheet Title** | `“Datasheet for MLX90614 I Melexis.”` | The letter `I` was substituted for the pipe character `\|`. Correct to `MLX90614 \| Melexis`. |
| **[25]** | **Thematic Mismatch (Marketing Paper Cited for IoT)** | P. Rajan, *"Integrating IoT Analytics into Marketing Decision Making: A Smart Data-Driven Approach"* | Cited on page 10 for basic IoT definitions. Citing a marketing paper in a medical triage proposal weakens academic credibility. Replace with an authoritative biomedical/healthcare IoT reference (e.g. IEEE Internet of Things Journal or standard health informatics literature). |
| **[1]** | **Substandard Government Law Citation** | `“JDIH Kementerian Kesehatan RI.”` | Format properly under IEEE government document standard: `Kementerian Kesehatan Republik Indonesia, “Peraturan Menteri Kesehatan Republik Indonesia Nomor 47 Tahun 2018 tentang Pelayanan Kegawatdaruratan,” Jakarta, 2018.` |

---

## 8. Actionable Editorial Revision Checklist

To bring the TriaGo proposal to 100% academic compliance and competition-grade readiness for Gemastik XIX:

- [ ] **Step 1: Fix Chapter 4 Table Numbering Domino**:
  - Change line 230: `pada tabel 4.3.` $\rightarrow$ `pada Tabel 4.2`.
  - Change line 232: `pada Tabel 4.4.` $\rightarrow$ `pada Tabel 4.3`.
  - Change line 319: `pada Tabel 4.5 berikut.` $\rightarrow$ `pada Tabel 4.4 berikut.`
  - Keep line 394: `pada Tabel 4.5 berikut.` $\rightarrow$ matches `Tabel 4.5`.
- [ ] **Step 2: Fix Chapter 4 Figure Callouts**:
  - Change line 128: `pada Gambar 4.6.` $\rightarrow$ `pada Gambar 4.7.` (Spektrum Daya PSD).
  - Change line 149: `pada Gambar 4.7` $\rightarrow$ `pada Gambar 4.8` (Komponen AC/DC).
- [ ] **Step 3: Resolve XGBoost Metric Conflict**:
  - Change line 230-231: `88.20% dan F1 Score sebesar 89.83%` $\rightarrow$ `90,67% dan F1-Score sebesar 91,42%`.
- [ ] **Step 4: Standardize Decimal Notation**:
  - Replace all decimal periods in prose and tables with Indonesian decimal commas (`4,90 mmHg`, `2,30 mmHg`, `6,22 mmHg`, `3,22 mmHg`, `0,925`, `0,998`, etc.).
- [ ] **Step 5: Apply Typo Corrections**:
  - Replace `melakuakan` $\rightarrow$ `melakukan`, `meskipul` $\rightarrow$ `meskipun`, `prakti` $\rightarrow$ `praktis`, `performma` $\rightarrow$ `performa`, `dihilangan` $\rightarrow$ `dihilangkan`, `menyuruh` $\rightarrow$ `menyeluruh`, `dijalakan` $\rightarrow$ `dijalankan`, `Spesifitas` $\rightarrow$ `Spesifisitas`.
- [ ] **Step 6: Enforce EYD V Bound Elements & Italicization**:
  - Merge `antar-` compounds (`antarkomponen`, `antarpetugas`, `antarpuncak`, etc.) and `non-` (`noninvasif`, `nonlinear`).
  - Italicize all upright foreign terms in the PDF source (`real-time`, `backend`, `broker`, `noise`, `bandpass filter`, etc.).
- [ ] **Step 7: Repair DAFTAR ISI & References**:
  - Insert `BAB IV ANALISIS FUNGSIONAL ................. 17` into `DAFTAR ISI`.
  - Fix Ref [2] authors (remove university affiliations).
  - Replace Ref [25] with a credible medical IoT publication.
