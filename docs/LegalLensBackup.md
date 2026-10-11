# LegalLens: Persistent Project State & Handoff Document (LegalLensBackup)

> **Document Purpose:**  
> This file is the single source of truth and living memory for **LegalLens**. Any autonomous coding agent, collaborator, or new session can read this document to understand the full project history, verified empirical statistics, architectural decisions, and immediate next steps without needing prior conversation context.

---

# 1. PROJECT IDENTITY

- **Project Name:** LegalLens
- **Full Title:** LegalLens — NLP-Based Classifier for Indian Court Judgments
- **Academic Context:** 2-Person B.Tech Data Science & AI Project
- **Course Code & Title:** DSE 4150 Natural Language Processing (NLP)
- **Repository:** `https://github.com/Mercer18/LegalLens`
- **Current Git Branch:** `feature/milestone-1-dataset-audit`
- **Base Branch:** `main` (clean, verified, synced at commit `0b3c349`)
- **Current Milestone:** **Final Test Evaluation (Complete)**
- **Overall Status:** Final unbiased test evaluation complete (Accuracy 93.59%, Macro-F1 0.9061, Criminal Recall 91.68%); ready for Streamlit deployment or project wrap-up.

---

# 2. PROJECT OBJECTIVE

LegalLens is an NLP system engineered to take the full text of an Indian court judgment as input and classify the whole judgment into its primary legal case category:
- **Civil**
- **Criminal**

### Scope Boundaries (What LegalLens is NOT):
- **NOT** a sentence-level rhetorical-role classifier (e.g., Facts, Arguments, Ratio Decidendi).
- **NOT** a court judgment outcome predictor (Accepted vs. Rejected appeals).
- **NOT** a court ranking or judicial performance evaluator.
- **NOT** an automated legal advisor or AI lawyer.
- **NOT** a predictive system for how a judge *should* decide a case.

---

# 3. DATASET SPECIFICATION (Jud-IPL)

- **Dataset Name:** Jud-IPL — Indian Legal Judgement Prediction Dataset
- **Canonical Source:** Kaggle (`bhavyabj/jud-ipl-indian-legal-judgement-prediction-dataset`, redirected from `kaggle.com/dsv/14441669`)
- **Associated Literature:** *CloLex: Exploring Closed Lexicals for LegalTech* (Jain, Harde & Jain, 2024, Springer)
- **License:** CC0: Public Domain
- **Local File Path:** `data/raw/case_files_total.csv` (879,162,942 bytes / 838.44 MB, git-ignored)

### Verified Empirical Dataset Statistics (Milestone 1 Audit)
*Note: These statistics were independently computed on the actual CSV and supersede published claims.*

| Property | Published Claim | Actual Verified Measurement | Discrepancy Note |
|---|---|---|---|
| **Raw Row Count** | 42,342 | **53,446** | Actual file contains +11,104 additional rows |
| **Row 0 Integrity** | Not mentioned | **Empty artifact** | Row index 0 has all columns `NaN` |
| **Label `civil`** | ~28,213 | **34,543** | Stored lowercased (`civil`) |
| **Label `criminal`** | ~7,363 | **9,069** | Stored lowercased (`criminal`) |
| **Label `undetermined`** | Explicit string | **0 string occurrences** | Represented purely as `NaN` (9,834 rows) |
| **Total Labeled Rows** | ~35,576 | **43,612** | Valid Civil + Criminal rows |
| **Empty `judgement` Text** | 0 | **3,303 total** (2,236 in labeled) | Must be pruned during cleaning |
| **Valid Cases with Text** | 42,342 | **41,376** | 32,941 Civil (79.61%), 8,435 Criminal (20.39%) |
| **Class Imbalance** | ~3.8 : 1 | **3.90 : 1** (Civil to Criminal) | Requires Macro-F1 & balanced class weights |
| **Document Length Median** | Not reported | **2,310 words** (13,509 characters) | Rich document-level legal context |
| **Document Length Mean** | Not reported | **2,690 words** (15,729 characters) | Substantial document length |
| **Max Character Length** | Uncapped | **32,765 characters** ($2^{15}-1$) | Text was capped at export by creators |
| **Unique Judgment Texts** | 42,342 | **33,725** (among 41,376 valid) | **7,651 duplicate text instances** (~18.5%) |
| **Conflicting Labels** | None reported | **2 text instances** | Identical text labeled as both Civil & Criminal |

---

# 4. DATA AUDIT HISTORY

- **2026-10-11 11:15:** Milestone 0 completed. Workspace inspected (Python 3.13.12, scikit-learn 1.8.0, pandas 3.0.2, NVIDIA RTX 5050 Laptop GPU, 95.8 GB free disk space). Scaffolding committed.
- **2026-10-11 12:25:** Teammate research report reviewed (`frndswork/Indian_Legal_Case_Classification_Dataset_Report.docx`). Jud-IPL provisionally approved; NyayaAnumana (15.1 GB) rejected due to task mismatch.
- **2026-10-11 12:49:** Dataset file `data/raw/case_files_total.csv` (838.44 MB) acquired.
- **2026-10-11 12:54:** Milestone 1 audit script `src/data_audit.py` executed. Discovered 53,446 raw rows (vs 42,342 claimed), 9,834 missing category rows, 2,236 empty texts in labeled subset, 7,651 duplicate texts, 2 cross-category conflicting duplicates, and 32,765 character truncation ceiling. All metrics recorded in `outputs/metrics/dataset_audit_metrics.json` and `docs/milestone_1_dataset_audit.md`.

---

# 5. NLP PIPELINE DESIGN

```
Indian Court Judgment (Full Text)
                ↓
    Text Cleaning & Normalization
  (Strip headers, whitespace normalization, preserve legal terms/citations/sections)
                ↓
            Tokenization
                ↓
        N-Gram Extraction (Unigrams & Bigrams)
                ↓
    TF-IDF Vector Representation (sublinear_tf=True, min_df, max_features)
                ↓
      Supervised Classification Baseline
  (Majority Baseline → Logistic Regression [balanced] → Linear SVM [balanced])
                ↓
    Predicted Legal Category (Civil vs Criminal)
                ↓
      Rigorous Evaluation & Error Analysis
  (Macro-F1, Per-Class Precision/Recall, Confusion Matrix)
                ↓
        Streamlit Web Interface (Final Milestone)
```

---

# 6. NLP CONCEPTS TRACKER

### Concepts Actively Used / Planned:
- **Text Classification:** Supervised document-level binary categorization.
- **Tokenization:** Word-level tokenization preserving statutory punctuation (e.g., `s. 302`, `art. 226`).
- **Domain-Aware Normalization:** Whitespace/case normalization; selective non-destructive cleaning.
- **Stopword Handling:** Domain-specific review (preserving negations like `not`, `no`, `never` and critical legal prepositions).
- **N-Grams:** Word unigrams and bigrams (e.g., `writ petition`, `penal code`, `special leave`).
- **TF-IDF:** Sublinear term frequency weighting with inverse document frequency.
- **Evaluation Metrics:** Macro-F1, Weighted-F1, Precision, Recall, Confusion Matrix.

### Concepts Intentionally Omitted (Justified Viva Defense):
- **Aggressive Stemming/Lemmatization:** Destroys subtle statutory distinctions (e.g., `murder` vs `murderer`, `suit` vs `suited`).
- **Spelling Correction:** High risk of corrupting specialized legal Latin terms (e.g., `mandamus`, `certiorari`, `suo motu`).
- **POS Tagging / Parsing (CFG/CKY):** Unnecessary computational overhead for document-level legal topic categorization.
- **Finite State Automata (FSA/DFA):** Regex pattern matching suffices for header cleanup.

---

# 7. MODELING PLAN & HIERARCHY

1. **Majority Baseline Classifier (`DummyClassifier`):** Predicts constant `civil` (~79.6% baseline accuracy, ~0.44 Macro-F1).
2. **TF-IDF + Logistic Regression:** L2 regularized, `class_weight='balanced'`, reproducible random seed (42).
3. **TF-IDF + Linear Support Vector Machine (`LinearSVC`):** `class_weight='balanced'`, dual=False.
4. **Multinomial Naive Bayes (Optional benchmark):** With Laplace smoothing.
5. **Transformer / InLegalBERT (Conditional Milestone 8):** Only pursued if classical baselines demonstrate error patterns that contextual representations can resolve, and if compute/length permits.

---

# 8. DATA LEAKAGE & DATA QUALITY CONTROLS

| Field / Phenomenon | Risk Type | Mitigation Policy | Status |
|---|---|---|---|
| `label` (`Accepted`/`Rejected`) | Task Contamination | Strictly excluded from modeling | Enforced |
| `proof_sentence` | Direct Outcome Leakage | Strictly excluded from modeling | Enforced |
| `case_type` (`appeal`, `writ petition`) | Metadata Proxy Leakage | Excluded from input features | Enforced |
| `case_info` (bench/judges) | Metadata Leakage | Excluded from input features | Enforced |
| 7,651 Duplicate Texts | Cross-Split Contamination | Deduplication policy being defined in Milestone 2 | Under Review |
| 2 Conflicting Texts | Label Noise | Inspected in Milestone 2; recommended removal | Under Review |
| 32,765 Char Truncation | Truncation Ceiling | Acknowledged; documents remain 2,300+ words median | Documented |
| 2,236 Empty Judgments | Missing Input | Pruned during dataset cleaning (Milestone 3) | Planned |

---

# 9. EXPERIMENT LOG

## Experiment 1: Majority Class Baseline
- **Date:** 2026-10-11
- **Architecture:** `DummyClassifier(strategy='most_frequent')`
- **Features:** None (predicts constant majority class `civil`)
- **Evaluation Partition:** Validation Set (3,372 judgments: 2,687 Civil, 685 Criminal)
- **Validation Results:**
  - Accuracy: 0.7969
  - Macro-F1: 0.4435 (Macro-Precision: 0.3984, Macro-Recall: 0.5000)
  - Civil Recall: 1.0000 | Civil F1: 0.8869
  - Criminal Recall: 0.0000 | Criminal F1: 0.0000
- **Confusion Matrix:** True Civil Pred Civil: 2,687 | True Crim Pred Crim: 0 | False Negatives: 685
- **Observation:** Confirms that Accuracy is misleading in imbalanced legal data; 79.69% accuracy fails to identify a single criminal case.

## Experiment 2: TF-IDF + Logistic Regression (Default Weights)
- **Date:** 2026-10-11
- **Architecture:** `LogisticRegression(C=1.0, class_weight=None, solver='lbfgs', random_state=42)`
- **Features:** TF-IDF unigrams + bigrams `(1, 2)`, max_features=30,000, sublinear_tf=True, min_df=3, max_df=0.90
- **Validation Results:**
  - Accuracy: 0.9267
  - Macro-F1: 0.8895
  - Criminal Recall: 0.8526 | Criminal Precision: 0.8000 | Criminal F1: 0.8254
  - Civil Recall: 0.9457 | Civil Precision: 0.9618 | Civil F1: 0.9536
- **Confusion Matrix:** True Civil Pred Civil: 2,541 (146 FP) | True Crim Pred Crim: 584 (101 FN)

## Experiment 3: TF-IDF + Logistic Regression (Balanced Weights)
- **Date:** 2026-10-11
- **Architecture:** `LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', random_state=42)`
- **Features:** Same TF-IDF (30,000 unigrams + bigrams)
- **Validation Results:**
  - Accuracy: 0.9208
  - Macro-F1: 0.8886
  - Criminal Recall: **0.9431** | Criminal Precision: 0.7391 | Criminal F1: 0.8287
  - Civil Recall: 0.9151 | Civil Precision: 0.9844 | Civil F1: 0.9485
- **Confusion Matrix:** True Civil Pred Civil: 2,459 (228 FP) | True Crim Pred Crim: 646 (39 FN)
- **Observation:** `class_weight='balanced'` boosted Criminal Recall from 85.26% to 94.31% (dropping false negatives from 101 to 39).

## Experiment 4: TF-IDF + Linear SVM (Default Weights)
- **Date:** 2026-10-11
- **Architecture:** `LinearSVC(C=1.0, class_weight=None, dual='auto', random_state=42)`
- **Features:** Same TF-IDF (30,000 unigrams + bigrams)
- **Validation Results:**
  - Accuracy: **0.9404**
  - Macro-F1: 0.9100
  - Criminal Recall: 0.8847 | Criminal Precision: **0.8324** | Criminal F1: 0.8577
  - Civil Recall: **0.9546** | Civil Precision: 0.9701 | Civil F1: 0.9623
- **Confusion Matrix:** True Civil Pred Civil: 2,565 (122 FP) | True Crim Pred Crim: 606 (79 FN)

## Experiment 5: TF-IDF + Linear SVM (Balanced Weights) — [TOP VALIDATION MODEL]
- **Date:** 2026-10-11
- **Architecture:** `LinearSVC(C=1.0, class_weight='balanced', dual='auto', random_state=42)`
- **Features:** Same TF-IDF (30,000 unigrams + bigrams)
- **Validation Results:**
  - Accuracy: 0.9386
  - **Macro-F1:** **0.9107** (Best overall)
  - **Criminal Recall:** **0.9343** (640 / 685 identified, only 45 FN)
  - **Criminal Precision:** 0.7980
  - **Criminal F1:** **0.8608** (Best overall)
  - **Civil Recall:** 0.9397 | Civil Precision: **0.9825** | Civil F1: 0.9606
- **Confusion Matrix:** True Civil Pred Civil: 2,525 (162 FP) | True Crim Pred Crim: 640 (45 FN)
- **Artifact Saved:** `models/linear_svm_balanced.joblib`

---

# 10. FILE & CODEBASE MAP

```
LegalLens/
├── .gitignore                          # Ignore rules for raw data, models, caches, and research notes
├── README.md                           # Project introduction, architecture diagram, and setup instructions
├── requirements.txt                    # Core verified dependencies (pandas, scikit-learn, streamlit, pytest)
├── docs/
│   ├── LegalLensBackup.md              # Living persistent memory / handoff document
│   ├── milestone_1_dataset_audit.md    # Detailed audit report on raw Jud-IPL dataset
│   ├── milestone_2_eda_report.md       # Comprehensive EDA report
│   ├── milestone_3_cleaning_and_split_report.md  # Data cleaning and split report
│   └── milestone_4_baseline_and_classical_models.md # Baseline and classical models benchmark report
├── data/
│   ├── raw/                            # Pristine raw CSV (case_files_total.csv, git-ignored)
│   ├── interim/                        # Cleaned / deduplicated data artifacts (git-ignored)
│   └── processed/                      # Final train/val/test splits & frozen test IDs (git-ignored)
├── notebooks/
│   └── 01_dataset_eda.ipynb            # Interactive EDA notebook
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Project paths, random seeds, schema column constants
│   ├── data_loader.py                  # Validated dataset loading utility
│   ├── data_audit.py                   # Milestone 1 audit script
│   ├── eda.py                          # Milestone 2 exploratory data analysis script
│   ├── preprocessing.py                # Milestone 3 cleaning & deduplication module
│   ├── dataset_split.py                # Milestone 3 contamination-free stratified splitting & test freezing
│   ├── features.py                     # Milestone 4 TF-IDF vectorization module
│   ├── models.py                       # Milestone 4 baseline, Logistic Regression, Linear SVM factory
│   ├── train.py                        # Milestone 4 training & benchmarking orchestrator
│   ├── evaluate.py                     # Milestone 4 metrics, reports, and confusion matrix plotter
│   └── predict.py                      # Unified prediction pipeline (Milestone 9)
├── app/
│   └── streamlit_app.py                # Streamlit UI (Milestone 10)
├── tests/
│   ├── __init__.py                     # Pytest suite
│   ├── test_preprocessing.py           # Unit tests for text normalization & hashing
│   ├── test_dataset_split.py           # Unit tests for split shapes, zero contamination & frozen test
│   └── test_models.py                  # Unit tests for model pipelines & inference sanity
└── outputs/
    ├── figures/                        # Generated PNG charts & confusion matrices (git-ignored)
    ├── reports/                        # Classification reports (git-ignored)
    └── metrics/                        # JSON metrics summaries (git-ignored)
```

---

# 11. GIT HISTORY & CHECKPOINTS

- **`0b3c349` (main):** `feat: initialize LegalLens project scaffolding and structure`
  - 15 files, initial commit on `main`, pushed to `origin/main`.
- **`8b27cac` (`feature/milestone-1-dataset-audit`):** `feat(audit): implement reproducible Jud-IPL dataset audit and report`
  - Created `src/config.py`, `src/data_audit.py`, `docs/milestone_1_dataset_audit.md`, updated `.gitignore`.
- **`639eecc` (`feature/milestone-1-dataset-audit`):** `docs: create persistent project state document LegalLensBackup.md`
  - Created `docs/LegalLensBackup.md` as living handoff document.
- **`3a6d47c` (`feature/milestone-1-dataset-audit`):** `feat(eda): implement Milestone 2 EDA pipeline, forensic analysis, notebook, and report`
  - Created `src/eda.py`, `notebooks/01_dataset_eda.ipynb`, `docs/milestone_2_eda_report.md`, updated backup.
- **`1a53bc2` (`feature/milestone-1-dataset-audit`):** `feat(cleaning): implement Milestone 3 data cleaning, deduplication, and stratified 80/10/10 split`
  - Created `src/preprocessing.py`, `src/dataset_split.py`, `tests/test_preprocessing.py`, `tests/test_dataset_split.py`, `docs/milestone_3_cleaning_and_split_report.md`.

---

# 12. MILESTONE STATUS

- [x] **Milestone 0: Scaffolding & Environment Setup** (Complete)
- [x] **Milestone 1: Dataset Acquisition & Audit** (Complete)
- [x] **Milestone 2: Exploratory Data Analysis (EDA)** (Complete)
- [x] **Milestone 3: Data Cleaning & Stratified Splitting** (Complete)
- [x] **Milestone 4: Majority Baseline & Logistic Regression & Linear SVM** (Complete)
- [ ] **Milestone 5: Model Comparison & Tuning / Naive Bayes** (Next)
- [ ] **Milestone 6: Error Analysis**
- [ ] **Milestone 7: Optional Transformer Experiment**
- [ ] **Milestone 8: Final Prediction Pipeline**
- [ ] **Milestone 9: Streamlit Application**
- [ ] **Milestone 10: Final Test Set Evaluation & Viva Defense Preparation**

---

# 13. DECISION LOG

1. **2026-10-11 — Primary Task Selection:**
   - *Decision:* Binary whole-judgment classification (Civil vs. Criminal).
2. **2026-10-11 — Dataset Selection:**
   - *Decision:* Jud-IPL adopted as official primary dataset.
3. **2026-10-11 — Feature Input Boundary:**
   - *Decision:* Only the `judgement` text is provided to the classifier.
4. **2026-10-11 — Primary Metric:**
   - *Decision:* Standardize on Macro-F1 and per-class Recall.
5. **2026-10-11 — Duplicate Pruning & Deduplication Policy (Milestone 2):**
   - *Decision:* Prune the 2 conflicting-label clusters (6 rows) and deduplicate to exactly 1 unique judgment text per hash before train/validation/test splitting.
6. **2026-10-11 — Whitespace Normalization Impact (Milestone 3):**
   - *Decision:* Applied internal whitespace/newline collapsing `re.sub(r"\s+", " ", text)`, which merged 6 additional formatting-duplicate pairs into verified consistent clusters, resulting in exactly **33,717 unique clean judgments**.
7. **2026-10-11 — Stratified Split & Test Freezing (Milestone 3):**
   - *Decision:* Partitioned into 80/10/10 (26,973 Train / 3,372 Val / 3,372 Test) with `random_state=42`, and froze test IDs/hashes in `data/processed/frozen_test_ids.json`.
8. **2026-10-11 — Model Benchmark Selection (Milestone 4):**
   - *Decision:* Selected **TF-IDF + Linear SVM (Balanced)** as the leading validation architecture (**0.9107 Macro-F1, 93.43% Criminal Recall, 93.97% Civil Recall**). Logistic Regression (Balanced) retained as strong probabilistic alternative (0.8886 Macro-F1, 94.31% Criminal Recall).
9. **2026-10-11 — Validation Error Analysis & Final Candidate Selection (Milestone 5):**
   - *Decision:* Completed error analysis on the 207 validation errors of balanced Linear SVM. Demonstrated that truncation does not degrade accuracy, while longer complex judgments have slightly higher confusion. Isolated 4 dominant legal failure modes (Contempt overlap, Police misconduct writs, Anti-corruption administrative overlap, Quasi-criminal commercial offenses).
   - *Targeted Tuning Outcome:* Expanding vocabulary to 40,000 features improves validation Macro-F1 to **0.9148** and Criminal Recall to **93.72%** (Accuracy 94.16%). Recommended **Linear SVM (Balanced, C=1.0, 40,000 features, 1-2 ngrams)** as the final model candidate for LegalLens.
10. **2026-10-11 — Final Prediction Pipeline Implementation (Milestone 6):**
   - *Decision:* Implemented `src/predict.py` packaging the locked M5 pipeline: TF-IDF (1-2 ngrams, 40,000 features, sublinear_tf=True) + LinearSVC (class_weight='balanced', C=1.0).
   - *Feature Integrity Guarantee:* Strict rejection of metadata; pipeline accepts raw judgment text only. Exact `normalize_legal_text` applied at inference time. Decision score and calibrated probabilities returned.
11. **2026-10-11 — Final Unbiased Hold-Out Test Evaluation:**
   - *Decision:* Evaluated locked pipeline exactly once against the frozen test partition (`test.parquet`, $N = 3,372$).
   - *Integrity Verification:* Test IDs match `frozen_test_ids.json` 100%. Train/test and val/test text-hash overlap = 0. Model artifact was NOT refitted or retuned after seeing test data.
   - *Test Results:* **Accuracy: 93.59%**, **Macro-F1: 0.9061**, **Weighted-F1: 0.9375**, **Criminal Recall: 91.68%**, **Criminal Precision: 79.80%**, **Civil Recall: 94.08%**, **Civil Precision: 97.79%**. Confirms minimal generalization delta (-0.87% Macro-F1 vs validation).

---

# 14. PROBLEMS & RESOLUTIONS LOG

- **Problem 1:** Dataset unauthenticated download blocked (Kaggle redirect to login).
  - *Resolution:* User provided `case_files_total.csv` directly into `data/raw/`. File integrity verified (879,162,942 bytes).
- **Problem 2:** Discrepancy between published row count (42,342) and raw CSV rows (53,446).
  - *Resolution:* Streamed full file in audit; traced difference to 9,834 unannotated rows and 2,236 empty texts. Verified true usable count is 41,376.
- **Problem 3:** 2 duplicate text instances found with conflicting class labels.
  - *Resolution:* Pruned all 6 records belonging to both conflicting clusters in Milestone 3. Verified remaining 7,327 duplicate clusters are 100.0% label-consistent.
- **Problem 4:** Potential cross-split contamination from 7,653 duplicate judgment copies.
  - *Resolution:* Deduplicated to 33,717 distinct judgments prior to splitting. Verified zero hash or ID overlap across Train, Val, and Test.
- **Problem 5:** Windows console CP1252 character encoding error when printing Unicode arrow (`→`) in terminal scripts.
  - *Resolution:* Replaced Unicode arrows with ASCII `->` across print statements in `src/error_analysis.py`.
- **Problem 6:** Seaborn 0.13 categorical palette assignment deprecation warning and boolean mapping.
  - *Resolution:* Re-mapped boolean flags to explicit categorical string labels (`"Correct"`, `"Misclassified"`) and assigned `hue` parameter in seaborn boxplots.
- **Problem 7:** Handling invalid or empty input gracefully in production prediction interface.
  - *Resolution:* Added input type checks and validation in `predict_judgment` returning structured failure metadata without throwing unhandled exceptions.

---

# 15. CURRENT STATE SNAPSHOT

- **Current Milestone:** Final Test Evaluation Complete; ready for Streamlit Deployment.
- **Current Branch:** `feature/milestone-1-dataset-audit`
- **Last Completed Action:** Executed one-shot final test evaluation on frozen hold-out partition (`test.parquet`, 3,372 cases) via `src/final_evaluation.py`. Generated `outputs/metrics/final_test_metrics.json`, `outputs/figures/final_test_confusion_matrix.png`, and report `docs/final_test_evaluation_report.md`. Verified all 18 unit tests passing.
- **Last Verified Result:** Final Hold-out Test Performance: Accuracy = 93.59%, Macro-F1 = 0.9061, Criminal Recall = 91.68%, Civil Recall = 94.08%. Zero data contamination verified. Locked model pipeline remained untouched without post-test modifications.
- **Current Task:** Awaiting user and reviewer review of Final Test Evaluation.
- **Current Blocker:** Awaiting approval before proceeding to Streamlit UI development.
- **Next Action:** Upon approval, implement interactive Streamlit web application in `app/streamlit_app.py` for live judgment classification and analysis.
- **Files Recently Changed:** `src/final_evaluation.py`, `tests/test_predict.py`, `docs/final_test_evaluation_report.md`, `docs/LegalLensBackup.md`, `outputs/metrics/final_test_metrics.json`, `outputs/figures/final_test_confusion_matrix.png`.
- **Working Tree Status:** Clean, all tests passing.



---

# 16. HANDOFF INSTRUCTIONS

## IF THIS PROJECT IS HANDED TO A NEW AGENT:
1. **Read this entire file first.** It reflects the exact real state of the workspace, not assumptions.
2. **Inspect the current Git branch and status** using `git status`. Ensure you remain on the active feature branch.
3. **Do not repeat completed work.** Milestones 0 and 1 are verified and audited.
4. **Do not discard previous decisions without justification.**
5. **Continue directly from the `CURRENT STATE SNAPSHOT`.**
6. **Maintain this backup continuously.** Update `docs/LegalLensBackup.md` after every meaningful action or milestone transition.
7. **Never silently overwrite experiment history.**
8. **Keep raw data and large generated models out of Git.**
