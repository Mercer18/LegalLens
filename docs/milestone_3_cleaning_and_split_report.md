# Milestone 3: Data Cleaning & Stratified Splitting Report

**Project Title:** LegalLens — NLP-Based Classifier for Indian Court Judgments  
**Date:** October 11, 2026  
**Auditor / NLP Engineer:** Antigravity (Implementation Agent)  
**Execution Modules:** `src/preprocessing.py`, `src/dataset_split.py`  
**Test Suite:** `tests/test_preprocessing.py`, `tests/test_dataset_split.py` (7/7 Passed)  
**Generated Metrics:** `outputs/metrics/cleaning_metrics.json`, `outputs/metrics/split_metrics.json`  
**Persisted Data Artifacts:**
- Cleaned Corpus: `data/interim/cleaned_judgments.parquet` (and `.csv`)
- Train Partition: `data/processed/train.parquet` (and `.csv`)
- Validation Partition: `data/processed/val.parquet` (and `.csv`)
- Test Partition: `data/processed/test.parquet` (and `.csv`)
- Frozen Test Membership: `data/processed/frozen_test_ids.json`
- Split Metadata: `data/processed/split_metadata.json`

---

## 1. Executive Summary & Verification

Milestone 3 successfully executed the approved data hygiene, deduplication, and contamination-free partitioning pipeline. 

### Key Deliverables & Results:
1. **Pristine Cleaning Waterfall:** Pruned 9,834 unannotated rows and 2,236 empty-text rows from the raw 53,446 input cases.
2. **Conflicting-Label Cluster Pruning:** Verified and removed all 6 records belonging to the two known conflicting duplicate clusters (`Conflict_1` and `Conflict_2`).
3. **100% Duplicate Cluster Consistency Verification:** Verified that every remaining duplicate cluster in the dataset has strictly one unanimous, consistent class label across all copies before retaining representatives.
4. **Final Clean Corpus Scale:** Deduplicated to **33,717 distinct, unique court judgments** (26,867 Civil, 6,850 Criminal, 3.92 : 1 imbalance).
5. **Stratified 80/10/10 Partitioning:** Partitioned with `random_state=42` into **26,973 Train**, **3,372 Validation**, and **3,372 Test** samples with exact class ratio conservation.
6. **Zero-Contamination Verification:** Verified zero text hash overlap and zero ID overlap across all partitions ($0$ train-val, $0$ train-test, $0$ val-test).
7. **Test Set Freezing:** Persisted exact test sample IDs, text hashes, and metadata in `data/processed/frozen_test_ids.json` to guarantee immutable test evaluations.

---

## 2. Data Cleaning Waterfall & Row Accounting

| Pipeline Stage | Rows Remaining | Excluded Rows | Reason / Action |
|---|---|---|---|
| **Raw Input CSV** | **53,446** | - | Original `case_files_total.csv` |
| *Unannotated / Missing Target* | 43,612 | 9,834 | Dropped rows with missing `case_category` (incl. empty Row 0 artifact) |
| *Empty `judgement` Text* | 41,376 | 2,236 | Dropped rows where text is null or whitespace-only |
| *Conflicting Duplicate Clusters* | 41,370 | 6 | Dropped all 6 rows belonging to `Conflict_1` (2 rows) and `Conflict_2` (4 rows) |
| *Consistent Duplicate Copies* | **33,717** | 7,653 | Retained exactly 1 representative judgment per unique text hash |
| **Final Cleaned Deduplicated Corpus** | **33,717** | **19,729 total** | **Pristine modeling corpus** |

### Note on Whitespace Normalization Impact:
In Milestone 2, preliminary text hashing on `.strip()` identified 33,723 distinct texts. In Milestone 3, internal whitespace and newline normalization (`re.sub(r"\s+", " ", text)`) discovered 6 additional pairs of identical judgments that differed only by line-wrap artifacts. These 6 clusters were verified 100% label-consistent and merged, yielding **33,717 true unique judgments**.

---

## 3. Duplicate Removal & Label Consistency Verification

- **Total Rows in Duplicate Clusters Prior to Pruning:** 14,980 rows (7,329 clusters).
- **Conflicting Duplicate Clusters:** Exactly 2 clusters (6 rows pruned).
- **Remaining Duplicate Clusters Audited:** 7,327 clusters.
- **Consistency Verification Result:** **100.00% consistent**. Every single one of the 7,327 duplicate clusters has exactly 1 unanimous label.
- **Representative Selection:** Deterministically kept the first occurrence by original CSV row index (`Unnamed: 0`).
- **Excess Duplicate Copies Removed:** **7,653 rows** eliminated from the training/evaluation pool.

---

## 4. Final Clean Corpus Class Distribution

| Category | Clean Document Count | Proportion | Imbalance Ratio |
|---|---|---|---|
| **Civil** | **26,867** | **79.68%** | 3.92 : 1 |
| **Criminal** | **6,850** | **20.32%** | Minority class |
| **Total** | **33,717** | **100.00%** | |

---

## 5. Stratified Split Accounting (80 / 10 / 10)

Split parameters: `random_state=42`, `stratify=category`, `TRAIN=0.80`, `VAL=0.10`, `TEST=0.10`.

| Partition | Total Documents | Civil Documents | Civil % | Criminal Documents | Criminal % | Ratio (Civ : Crim) |
|---|---|---|---|---|---|---|
| **Train (80%)** | **26,973** | 21,493 | 79.68% | 5,480 | 20.32% | 3.92 : 1 |
| **Validation (10%)** | **3,372** | 2,687 | 79.69% | 685 | 20.31% | 3.92 : 1 |
| **Test (10%)** | **3,372** | 2,687 | 79.69% | 685 | 20.31% | 3.92 : 1 |
| **Total (100%)** | **33,717** | **26,867** | **79.68%** | **6,850** | **20.32%** | **3.92 : 1** |

### Mathematical Contamination Audit:
- $\text{Train Hashes} \cap \text{Val Hashes} = \emptyset$ (**0 overlapping documents**)
- $\text{Train Hashes} \cap \text{Test Hashes} = \emptyset$ (**0 overlapping documents**)
- $\text{Val Hashes} \cap \text{Test Hashes} = \emptyset$ (**0 overlapping documents**)
- $\text{Train IDs} \cap \text{Val IDs} = \emptyset$ (**0 overlapping IDs**)
- $\text{Train IDs} \cap \text{Test IDs} = \emptyset$ (**0 overlapping IDs**)
- $\text{Val IDs} \cap \text{Test IDs} = \emptyset$ (**0 overlapping IDs**)

---

## 6. Test Set Freezing & Persistence

The test partition is frozen and locked in `data/processed/frozen_test_ids.json`.  
Key metadata stored:
- `test_size`: 3,372
- `random_seed`: 42
- `test_class_distribution`: 2,687 Civil, 685 Criminal
- Complete sorted lists of `frozen_test_ids` and `frozen_test_hashes`

Future modeling milestones (Milestones 4–8) will read this exact test membership, preventing any accidental data leakage, test set reassignment, or evaluation contamination.

---

## 7. Model Input Policy Enforcement

All metadata fields (`case_type`, `case_info`, `label`, `proof_sentence`, `tokens`, `sentences`, row indices) have been completely stripped from the modeling datasets. The persisted partitions contain strictly:
- `text`: Clean normalized court judgment text.
- `category`: Target class label (`civil` or `criminal`).
- `name` & `text_hash`: Unique document identifiers for auditability and verification.
