# Milestone 2: Exploratory Data Analysis (EDA) Report

**Project Title:** LegalLens — NLP-Based Classifier for Indian Court Judgments  
**Date:** October 11, 2026  
**Auditor / NLP Engineer:** Antigravity (Implementation Agent)  
**Dataset Under Analysis:** Jud-IPL (`case_files_total.csv`)  
**EDA Execution Script:** `src/eda.py`  
**Interactive Notebook:** `notebooks/01_dataset_eda.ipynb`  
**Generated Metrics:** `outputs/metrics/eda_metrics.json`  
**Figures Produced:**
- `outputs/figures/class_wise_length_distribution.png`
- `outputs/figures/duplicate_group_sizes.png`
- `outputs/figures/top_bigrams_by_class.png`

---

## 1. Executive Summary & Overview

Milestone 2 conducted an in-depth empirical exploration of the audited Jud-IPL dataset. The core objective was to analyze the legal text characteristics, resolve duplicate clusters, perform forensic inspection of label conflicts, extract discriminative vocabulary, and establish an evidence-based data cleaning and splitting policy before any modeling begins.

### Key EDA Discoveries:
1. **Total Usable Records:** Out of 53,446 raw rows, **41,376 judgments** have both a confirmed category (`civil` or `criminal`) and non-empty text.
2. **Class Imbalance:** 32,941 Civil (79.61%) vs. 8,435 Criminal (20.39%), confirming a **~3.91 : 1 class imbalance ratio**.
3. **Duplicate Footprint:** 7,329 distinct duplicate clusters contain 14,980 rows (7,651 excess duplicate copies). 99.97% (7,327 clusters) are label-consistent.
4. **Conflicting Label Forensic Resolution:** Exactly **2 duplicate clusters** have conflicting labels. Both were traced to specific causes:
   - *Conflict 1 (4,083 words):* Arms Act conviction, unambiguously Criminal; one row was mislabeled `civil`.
   - *Conflict 2 (844 words):* Article 32 Constitutional Writ Petition seeking police prosecution, demonstrating institutional registry label ambiguity.
5. **Class Length Asymmetry:** Civil judgments are on average **~35% longer** in median word count than Criminal judgments (2,856 median words vs. 2,123 median words).
6. **Domain Vocabulary Separation:** Bigram and unigram distributions reveal **stark legal topic differentiation** (`income tax`, `sales tax`, `writ petition` vs. `sessions judge`, `penal code`, `dying declaration`), proving high feasibility for TF-IDF linear baselines.

---

## 2. Dataset Scale & Filtering Waterfall

| Pipeline Stage | Row Count | Percentage of Raw | Excluded Reasons |
|---|---|---|---|
| **Raw Input CSV** | **53,446** | 100.00% | Pristine input file |
| *Row 0 Ghost Artifact* | -1 | 0.002% | Empty index-0 row with all `NaN`s |
| *Unlabeled / Missing Category* | -9,833 | 18.40% | Ambiguous or missing `case_category` |
| **Total Labeled Records** | **43,612** | 81.60% | Valid Civil & Criminal cases |
| *Empty `judgement` Text* | -2,236 | 4.18% | Empty strings or `NaN` judgment text |
| **Valid Non-Empty Corpus** | **41,376** | 77.42% | Valid input corpus for EDA |

### Class Breakdown (Pre-Deduplication)
- **Civil:** 32,941 (79.61%)
- **Criminal:** 8,435 (20.39%)
- **Imbalance Ratio:** **3.91 : 1** (Civil to Criminal)

---

## 3. Duplicate Cluster Analysis

### Cluster Characteristics
Across the 41,376 valid cases, we computed SHA-256 hashes of the full judgment texts:
- **Unique Text Hashes:** **33,725**
- **Total Duplicate Rows:** **14,980 rows** (spread across 7,329 clusters)
- **Excess Duplicate Copies:** **7,651 copies** (~18.5% of the valid text dataset)

### Distribution of Cluster Sizes

| Cluster Size (Copies of Same Text) | Number of Distinct Clusters | Total Rows Represented | Percentage of Clusters |
|---|---|---|---|
| **2 copies** | 7,041 | 14,082 | 96.07% |
| **3 copies** | 255 | 765 | 3.48% |
| **4 copies** | 32 | 128 | 0.44% |
| **5 copies** | 1 | 5 | 0.01% |
| **Total** | **7,329** | **14,980** | **100.00%** |

### Duplicate Consistency
- **Same-Label Clusters:** **7,327 clusters** (99.97%). All repeated copies have identical categories.
- **Conflicting-Label Clusters:** **2 clusters** (0.03%).

### Root Cause of Duplication:
Inspection confirms these are **identical court judgment exports** resulting from scraper/database ingestion where companion appeals, co-accused petitions, or batch-disposed matters were exported as separate rows with separate file names (`name` column) but identical full judgment text.

---

## 4. Forensic Deep-Dive: The Two Conflicting-Label Cases

### Case 1: Arms Act & Explosives Conviction (`Conflict_1`)
- **Text Hash:** `6461ba2266a5a53441bbaccfc8debf76d5273d08f3cd5574e2e7602d1ab65dd6`
- **Document Length:** 23,244 characters, 4,083 words
- **Records in Cluster:** 2 rows
  1. Row 23298 (`162668847.txt`): Labeled **`criminal`**, case type: `appeal`
  2. Row 24211 (`1800491.txt`): Labeled **`civil`**, case type: `appeal`
- **Opening Text Excerpt:**
  > *"criminal appeal number 941 of 1998 has been preferred by mohd. ishtiaq and criminal appeal number 307 of 1998 has been preferred by aslam parwez against the judgment and order dated 2.2.1998 of addl. designated court delhi by which the former was convicted under section 25 of the arms act and section 5 of explosives substances act and was sentenced to two years ri and a fine of rs.1,000/- under the first count..."*
- **Root Cause & Findings:**
  This case is an open-and-shut Criminal Appeal arising from designated terror/arms court convictions under the Arms Act and Explosive Substances Act. The opening sentence begins with *"criminal appeal number 941 of 1998"*. File `1800491.txt` was erroneously miscataloged as `"civil"` during automated ingestion.
- **Recommendation:** Remove this cluster (2 rows) to eliminate contradictory supervision without introducing manual relabeling subjectivity.

---

### Case 2: Article 32 Police Prosecution Writ (`Conflict_2`)
- **Text Hash:** `b9009ed43e842ac80283317e56d3e18d006ffd42cfa75787da4263bbf1e2f756`
- **Document Length:** 4,943 characters, 844 words
- **Records in Cluster:** 4 rows
  1. Row 8793 (`1945538.txt`): Labeled **`civil`**, case type: `writ petition`
  2. Row 13125 (`191219719.txt`): Labeled **`criminal`**, case type: `writ petition`
  3. Row 34565 (`119448908.txt`): Labeled **`civil`**, case type: `writ petition`
  4. Row 38728 (`605378.txt`): Labeled **`civil`**, case type: `writ petition`
- **Opening Text Excerpt:**
  > *"this writ petition has been filed under article 32 of the constitution seeking the following reliefs - a direct the respondent number2 to properly investigate the subject matter of the present writ petition, regarding the abduction, harassment, illegal detention and other serious criminal offences committed by the high officials as stated in the writ petition and to punish the wrongdoers in accordance with the various provisions of the indian penal code..."*
- **Root Cause & Findings:**
  This is an Article 32 Constitutional Writ Petition seeking police investigation of criminal offenses (abduction, illegal detention). In Indian Supreme Court registry practice, writ petitions involving criminal investigations are sometimes registered on the civil writ roster and sometimes on the criminal writ roster, creating inherent institutional ambiguity.
- **Recommendation:** Remove this cluster (4 rows). It represents genuine jurisdictional ambiguity and label divergence.

---

## 5. Document Length Analysis: Civil vs. Criminal Asymmetry

| Metric | Overall | Civil Cases | Criminal Cases | Comparison |
|---|---|---|---|---|
| **Minimum Words** | 1 | 1 | 1 | Both contain short orders |
| **25th Percentile (p25)** | 1,385 words | 1,486 words | 1,083 words | Civil is 37% longer at p25 |
| **Median (p50)** | **2,695 words** | **2,856 words** | **2,123 words** | **Civil is 34.5% longer at median** |
| **Mean Words** | 2,955 words | 3,079 words | 2,471 words | Civil is 24.6% longer on mean |
| **Standard Deviation** | 1,817 words | 1,830 words | 1,679 words | Wide variance across court docket |
| **75th Percentile (p75)** | 4,703 words | 5,049 words | 3,603 words | Civil is 40.1% longer at p75 |
| **90th Percentile (p90)** | 5,595 words | 5,607 words | 5,477 words | Approaching export cap |
| **95th Percentile (p95)** | 5,696 words | 5,703 words | 5,659 words | Truncation plateau |
| **Maximum Words** | 6,591 words | 6,591 words | 6,145 words | Capped at ~32.7 KB text |

### Substantive Interpretation:
- **Civil cases are substantially longer:** Civil appeals often litigate intricate property deeds, multi-layered statutory rules (e.g., Land Acquisition Act, Income Tax Act, Service Rules), and detailed lower-court commercial records.
- **Criminal cases are more concise:** Criminal judgments focus more directly on the immediate trial record, eyewitness reliability, post-mortem findings, and ingredients of the specific penal charge.
- **Export Truncation Impact:** 19.4% of judgments reach the 32,765-character database limit and end abruptly. However, with an average of 2,400 to 3,000 words preserved, the document text contains ample statutory citations and terminology for bag-of-words and TF-IDF representation.

---

## 6. Vocabulary & Discriminative Legal N-Grams

Extracted from a representative stratified corpus sample (5,000 judgments, 14.78M tokens, 86,950 unique terms):

### Top Discriminative Unigrams (Domain Stopwords Filtered)

| Rank | Civil Domain Unigrams | Criminal Domain Unigrams | Contrast Interpretation |
|---|---|---|---|
| 1 | `government` (23,789) | `accused` (14,084) | Civil litigations frequently cite state authorities vs. accused persons |
| 2 | `land` (16,802) | `evidence` (9,477) | Land disputes vs. evidentiary trial scrutiny |
| 3 | `service` (13,880) | `deceased` (5,764) | Civil service disputes vs. homicide/death inquiries |
| 4 | `tax` (13,502) | `prosecution` (5,058) | Revenue & tax appeals vs. criminal prosecution |
| 5 | `rules` (13,104) | `trial` (4,901) | Administrative rules vs. criminal trial procedure |
| 6 | `provisions` (13,272) | `offence` (4,672) | Statutory provisions vs. penal offences |
| 7 | `property` (12,980) | `police` (4,118) | Civil ownership vs. law enforcement actions |
| 8 | `company` (10,412) | `witnesses` (3,468) | Corporate / commercial law vs. trial witnesses |

### Top Discriminative Bigrams

| Civil Legal Bigrams | Criminal Legal Bigrams |
|---|---|
| `writ petition` (6,632) | `accused persons` (1,733) |
| `income tax` (4,544) | `sessions judge` (1,318) |
| `division bench` (4,194) | `police station` (1,109) |
| `central government` (3,473) | `penal code` (1,050) |
| `article constitution` (3,422) | `criminal procedure` (1,029) |
| `sales tax` (1,856) | `dying declaration` (736) |
| `special leave` (3,375) | `eye witnesses` (666) |
| `civil number` (2,231) | `rigorous imprisonment` (568) |

> [!TIP]
> **Key Modeling Takeaway:** The vocabulary divergence between Civil and Criminal cases in Indian court judgments is **exceptionally pronounced**. Feature representations combining **Unigrams + Bigrams** with **TF-IDF sublinear scaling** will possess outstanding discriminative power.

---

## 7. Recommended Data Cleaning & Splitting Policy (Milestone 3)

Based on these empirical findings, we recommend the following four-step cleaning policy:

1. **Prune Missing and Empty Records:**
   - Drop the 9,834 unannotated rows and 2,236 empty-text rows.
2. **Eliminate Conflicting-Label Clusters:**
   - Drop the 6 records belonging to `Conflict_1` (2 rows) and `Conflict_2` (4 rows) to eliminate label contamination.
3. **Deduplicate to Distinct Unique Texts:**
   - Retain exactly **1 representative instance per unique text hash**.
   - This prevents duplicate judgments from leaking across splits.
   - Clean deduplicated dataset size: **33,723 distinct judgments**.
     - **Civil:** 26,117 (77.45%)
     - **Criminal:** 7,606 (22.55%)
     - **Imbalance Ratio:** **3.43 : 1**
4. **Stratified Split:**
   - Partition the 33,723 deduplicated cases into **Train (80% = 26,978)**, **Validation (10% = 3,372)**, and **Test (10% = 3,373)** using stratified random sampling with seed `42`.
   - Freeze the Test set completely.
