# Milestone 1: Jud-IPL Dataset Acquisition & Audit Report

**Project Title:** LegalLens — NLP-Based Classifier for Indian Court Judgments  
**Date:** October 11, 2026  
**Auditor:** Antigravity (Implementation Agent)  
**Dataset Under Audit:** Jud-IPL (`case_files_total.csv`)  
**Audit Script:** `src/data_audit.py`  
**Generated Metrics:** `outputs/metrics/dataset_audit_metrics.json`  
**Visualizations:** `outputs/figures/raw_class_distribution.png`, `outputs/figures/document_length_distribution.png`

---

## 1. Executive Summary & Verdict

We completed an empirical audit of the raw Jud-IPL dataset file (`case_files_total.csv`, 838.44 MB) placed under `data/raw/`. 

**Core Verdict:**  
The dataset is **viable and well-suited** for the whole-judgment binary classification task (**Civil vs. Criminal**), but **requires careful data hygiene** before modeling. Crucially, several published claims on the dataset card do not match the real file contents.

### Critical Discrepancies Discovered:
1. **Row Count Discrepancy:** The published dataset card reports 42,342 cases. The actual file contains **53,446 raw rows** (26.2% more than published).
2. **Missing Label Representation:** The published summary describes an explicit `"Undetermined"` string label. In the actual CSV, **0 rows** have the string `"Undetermined"`; ambiguous/undetermined cases are encoded as **missing (`NaN`)** across 9,834 rows.
3. **Ghost Artifact Row:** Row 0 is an empty artifact where every feature column is `NaN` and `Unnamed: 0` is `0`.
4. **Empty Judgment Texts:** **3,303 rows** contain empty/null judgment text in the raw file (including **2,236 rows** within the labeled Civil/Criminal subset).
5. **Significant Duplicate Footprint:** **7,651 duplicate judgment texts** exist within the valid Civil/Criminal subset (~18.5% of non-empty records). Additionally, **2 text entries have conflicting labels** (identical text assigned to both Civil and Criminal).
6. **Hard Length Truncation Cap:** The maximum character length in the raw file is exactly **32,765 characters** (matching the $2^{15}-1$ integer limit), indicating that the original data collectors truncated judgment text during database export.

---

## 2. File & Schema Specification

| Property | Value / Measurement |
|---|---|
| **File Name** | `case_files_total.csv` |
| **Path** | `data/raw/case_files_total.csv` |
| **File Size** | **879,162,942 bytes (~838.44 MB)** |
| **Format** | Standard UTF-8 Comma-Separated Values (CSV) |
| **Raw Total Rows** | **53,446** |
| **Raw Total Columns** | **10** |

### Column Schema & Data Types

| # | Column Name | Pandas Inferred Type | Description | Role in LegalLens |
|---|---|---|---|---|
| 0 | `Unnamed: 0` | `int64` | Original sequential index | Ignored / Row index |
| 1 | `name` | `object` (`str`) | File identifier (e.g. `74634371.txt`) | Document ID |
| 2 | `case_category` | `object` (`str`) | Legal category (`civil`, `criminal`, `NaN`) | **Primary Target Label** |
| 3 | `case_type` | `object` (`str`) | Procedural type (e.g. `appeal`, `writ petition`) | Metadata (Leakage Risk) |
| 4 | `case_info` | `object` (`str`) | Case procedural header / bench info | Metadata (Leakage Risk) |
| 5 | `judgement` | `object` (`str`) | Full text of Supreme Court judgment | **Primary Model Input** |
| 6 | `tokens` | `object` (`str`) | Pre-tokenized count/representation | Redundant |
| 7 | `sentences` | `object` (`str`) | Pre-sentence count/representation | Redundant |
| 8 | `label` | `object` (`str`) | Outcome prediction target (`Accepted`, `Rejected`, `Other`) | **Strict Exclude (Different Task)** |
| 9 | `proof_sentence` | `object` (`str`) | Final operative order sentence (e.g. `appeal dismissed.`) | **Strict Exclude (Outcome Leakage)** |

---

## 3. Label Distribution & Target Construction

### Raw `case_category` Value Counts

| Category Value | Raw Count | Percentage | Usability Status |
|---|---|---|---|
| `civil` | **34,543** | 64.63% | Usable Target |
| `criminal` | **9,069** | 16.97% | Usable Target |
| `NaN` (Missing / Undetermined) | **9,834** | 18.40% | Excluded (Unlabeled / Ambiguous) |
| `undetermined` (string) | **0** | 0.00% | N/A |
| Other invalid strings | **0** | 0.00% | N/A |
| **Total** | **53,446** | **100.00%** | |

### Primary Dataset Construction (Civil vs. Criminal)

For the primary whole-judgment binary classification experiment:
- **Eligible Labeled Rows:** **43,612**
- **Excluded Rows:** **9,834** (18.40%)
  - *Row 0 ghost artifact:* 1 row
  - *Missing/Undetermined category:* 9,833 rows
- **Class Balance (Labeled Subset):**
  - **Civil:** 34,543 (79.21%)
  - **Criminal:** 9,069 (20.79%)
  - **Class Imbalance Ratio:** **3.81 : 1** (Civil to Criminal)

> [!NOTE]
> When filtering further for rows with **non-empty judgment text**:
> - **Total Non-Empty Labeled Cases:** **41,376**
> - **Civil:** 32,941 (79.61%)
> - **Criminal:** 8,435 (20.39%)
> - **Imbalance Ratio:** **3.90 : 1**
> 
> Because of this ~4:1 class imbalance, **Macro-F1 and per-class Recall** must serve as the primary evaluation metrics rather than raw Accuracy.

---

## 4. Text Quality & Document Length Audit

### Missingness & Extreme Lengths
- **Empty Judgment Texts:** **3,303** across raw data (including **2,236** in the labeled Civil/Criminal subset). These empty records must be pruned during dataset cleaning.
- **Extremely Short Texts (< 10 words):** **40 cases**.
- **Short Texts (< 50 words):** **202 cases**.

### Document Length Percentiles (Calculated on Non-Empty Texts)

| Metric | Word Count | Character Count |
|---|---|---|
| **Minimum** | 1 | 1 |
| **Median** | **2,310 words** | **13,509 chars** |
| **Mean** | **2,690 words** | **15,729 chars** |
| **90th Percentile** | **5,565 words** | **32,765 chars** |
| **95th Percentile** | **5,678 words** | **32,765 chars** |
| **Maximum** | **6,591 words** | **32,765 chars** |

### Key Document Length Observations:
1. **Truncation Plateau:** The 90th percentile, 95th percentile, and maximum character length are all exactly **32,765 characters**. This confirms that documents longer than ~32.7 KB were truncated at source when exported into this CSV.
2. **Implications for TF-IDF Baselines:** A median of 2,310 words is a rich, robust context length for n-gram and TF-IDF representations. Whole-document TF-IDF vectors will have ample legal terminology signal.
3. **Implications for Transformers:** If a BERT-style model (512 token context window) is evaluated later, it will capture only ~10% to 15% of the median judgment text unless chunking or hierarchical aggregation is applied.

---

## 5. Duplicate & Data Contamination Audit

### Identification of Duplicates
- **Unique Document IDs (`name`):** 44,677 unique out of 53,446 rows (8,768 duplicate IDs).
- **Unique Judgment Texts (Global):** 41,006 unique across all 50,143 non-empty texts (9,137 duplicate text entries).
- **Duplicate Texts in Labeled Subset:** **7,651 duplicate texts** exist among the 41,376 valid cases.

### Critical Split Contamination Risk:
If a standard randomized train/validation/test split is applied without text deduplication or document clustering, **thousands of duplicate judgment texts will leak across train and test sets**, yielding artificially inflated test performance.

### Cross-Category Label Noise (Conflicting Duplicates):
We detected **2 text entries where the exact same judgment text is assigned conflicting category labels**:
1. **Conflict 1 (23,246 chars):** Labeled `criminal` in `162668847.txt` and `civil` in `1800491.txt`.
2. **Conflict 2 (4,944 chars):** Labeled `criminal` in `191219719.txt` and `civil` in `1945538.txt`, `119448908.txt`, and `605378.txt`.

**Decision:** Both conflicting text instances must be removed completely during preprocessing to eliminate contradictory supervision.

---

## 6. Target Leakage Audit

We systematically reviewed all non-text columns for potential target or outcome leakage:

| Field Name | Type / Values | Leakage Assessment & Policy |
|---|---|---|
| `judgement` | Raw text | **Primary Input.** Cleaned of metadata headers where appropriate. |
| `case_type` | `appeal` (27,027), `writ petition` (18,845), `special leave petition` (3,916), etc. | **Potential Proxy Leakage.** While appeals and petitions exist in both civil and criminal jurisdictions, certain petitions (e.g. habeas corpus, tax references) strongly correlate with one branch. **Policy:** Exclude from input feature vector for core whole-judgment NLP baseline. |
| `case_info` | Text procedural headers (e.g., `order 1.\n`, judge bench names) | **Metadata Risk.** Contains procedural headers and bench designations. **Policy:** Exclude from modeling. |
| `label` | `Accepted` (21,196), `Rejected` (23,913), `Other` (8,335) | **Task Contamination.** Outcome prediction target from CloLex. Unrelated to legal case category. **Policy:** Strictly excluded. |
| `proof_sentence`| Operative disposal sentences (e.g., `appeal dismissed.`) | **Outcome Leakage.** Directly summarizes the final judicial disposition. **Policy:** Strictly excluded. |
| `tokens`, `sentences` | Preprocessed token / sentence counts | Redundant derivative fields. **Policy:** Strictly excluded. |

---

## 7. Representative Examples

### 1. Civil Case Sample (`74634371.txt`, 5,762 words)
> *"thus spake benjamin franklin in his letter of november 13, 1789 to jean baptiste leroy. to tax the dead is a contradiction in terms. tax laws are made by the living to tax the living. what survives the dead person is what is left behind in the form of such persons property. this appeal raises questions as to whether the dead persons property, in the form of his or her estate, can be taxed without..."*

### 2. Criminal Case Sample (`746119.txt`, 2,383 words)
> *"this appeal by special leave is against the judgment and order dated 28-4-1971 of the high court of punjab and haryana in criminal appeal number 1310 of 1969 confirming the conviction of the appellant for an offence under section 5 2 of the prevention of corruption act, read with section 161 of the indian penal code. the appellant was sentenced by the special judge to undergo two and a half years..."*

### 3. Excluded Missing Category Sample (`745920.txt`, 1,401 words)
> *"this appeal by special leave is directed against the judgment of the high court refusing to direct the income-tax appellate tribunal to state a case under section 66 2 of the indian income-tax act, 1922, hereinafter referred to as the act. the only question which falls for consideration is whether a referable question of law arises or not out of the order of the appellate tribunal, b-bench, calcu..."*

---

## 8. Recommendations for Next Milestones

1. **Lock Jud-IPL as Primary Dataset:** The dataset provides **41,376 valid, non-empty Indian Supreme Court judgments** with strong semantic differentiation between Civil and Criminal law.
2. **Deduplication Strategy (Milestone 3):**
   - Eliminate Row 0 and all 2,236 empty-text rows.
   - Remove the 2 cross-category conflicting duplicates.
   - Deduplicate judgments on text hash so that only distinct legal judgments enter the train/val/test split, preventing cross-split contamination.
3. **Stratification & Class Weights (Milestones 3 & 4):**
   - Apply stratified splitting (80/10/10 or 70/15/15) to maintain the ~4:1 Civil-to-Criminal ratio across Train, Validation, and Test partitions.
   - Utilize balanced class weighting (`class_weight='balanced'`) in Logistic Regression and Linear SVM to ensure robust minority class (Criminal) recall.
4. **Evaluation Strategy (Milestone 6):**
   - Standardize on **Macro-F1** as the headline metric to prevent Civil class dominance from masking Criminal errors.
