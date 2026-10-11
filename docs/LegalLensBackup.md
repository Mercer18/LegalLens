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
- **Current Milestone:** **Milestone 2: Exploratory Data Analysis (EDA)**
- **Overall Status:** Milestone 0 (Scaffolding) and Milestone 1 (Audit) complete; Milestone 2 active.

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

*(No models trained yet. Experiments will be sequentially appended in Milestones 4–8).*

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
│   └── milestone_2_eda_report.md       # Comprehensive EDA report (in progress)
├── data/
│   ├── raw/                            # Pristine raw CSV (case_files_total.csv, git-ignored)
│   ├── interim/                        # Cleaned / deduplicated data artifacts (git-ignored)
│   └── processed/                      # Final train/val/test splits (git-ignored)
├── notebooks/
│   └── 01_dataset_eda.ipynb            # Interactive EDA notebook
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Project paths, random seeds, schema column constants
│   ├── data_loader.py                  # Validated dataset loading utility
│   ├── data_audit.py                   # Milestone 1 audit script
│   ├── eda.py                          # Milestone 2 exploratory data analysis script
│   ├── preprocessing.py                # Text cleaning & normalization (Milestone 3)
│   ├── dataset_split.py                # Contamination-free stratified splitting (Milestone 3)
│   ├── features.py                     # TF-IDF vectorization (Milestone 4)
│   ├── models.py                       # Baseline and classical ML classifiers (Milestone 4/5)
│   ├── train.py                        # Training pipeline orchestrator (Milestone 4/5)
│   ├── evaluate.py                     # Metrics and error analysis (Milestone 6/7)
│   └── predict.py                      # Unified prediction pipeline (Milestone 9)
├── app/
│   └── streamlit_app.py                # Streamlit UI (Milestone 10)
├── tests/
│   └── __init__.py                     # Pytest suite
└── outputs/
    ├── figures/                        # Generated PNG charts (git-ignored)
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

---

# 12. MILESTONE STATUS

- [x] **Milestone 0: Scaffolding & Environment Setup** (Complete)
- [x] **Milestone 1: Dataset Acquisition & Audit** (Complete)
- [x] **Milestone 2: Exploratory Data Analysis (EDA)** (Complete)
- [ ] **Milestone 3: Data Cleaning & Stratified Splitting** (Pending Milestone 2 Review)
- [ ] **Milestone 4: Majority Baseline & Logistic Regression**
- [ ] **Milestone 5: TF-IDF & Linear SVM**
- [ ] **Milestone 6: Model Comparison & Evaluation**
- [ ] **Milestone 7: Error Analysis**
- [ ] **Milestone 8: Optional Transformer Experiment**
- [ ] **Milestone 9: Final Prediction Pipeline**
- [ ] **Milestone 10: Streamlit Application**
- [ ] **Milestone 11: Comprehensive Tests, Documentation & Viva Preparation**

---

# 13. DECISION LOG

1. **2026-10-11 — Primary Task Selection:**
   - *Decision:* Binary whole-judgment classification (Civil vs. Criminal).
   - *Reason:* Well-defined legal distinction, robust dataset availability, directly connected to course objectives.
   - *Rejected:* Sentence-level rhetorical roles (not whole-judgment) and outcome prediction (unreliable labels / different task).
2. **2026-10-11 — Dataset Selection:**
   - *Decision:* Jud-IPL adopted as official primary dataset.
   - *Reason:* Only available Indian corpus with verified whole-judgment Civil/Criminal annotations.
   - *Rejected:* NyayaAnumana (15.1 GB, outcome prediction only, no category target).
3. **2026-10-11 — Feature Input Boundary:**
   - *Decision:* Only the `judgement` text is provided to the classifier.
   - *Reason:* Prevents procedural metadata leakage (`case_type`, `case_info`, `proof_sentence`).
4. **2026-10-11 — Primary Metric:**
   - *Decision:* Standardize on Macro-F1 and per-class Recall.
   - *Reason:* Mitigates ~3.9 : 1 Civil dominance from inflating accuracy while hiding Criminal misclassifications.
5. **2026-10-11 — Duplicate Pruning & Deduplication Policy (Milestone 2):**
   - *Decision:* Prune the 2 conflicting-label clusters (6 rows) and deduplicate to exactly 1 unique judgment text per hash (33,723 distinct judgments) before train/validation/test splitting.
   - *Reason:* Prevents severe cross-split data leakage/contamination and eliminates conflicting supervisory signals.

---

# 14. PROBLEMS & RESOLUTIONS LOG

- **Problem 1:** Dataset unauthenticated download blocked (Kaggle redirect to login).
  - *Resolution:* User provided `case_files_total.csv` directly into `data/raw/`. File integrity verified (879,162,942 bytes).
- **Problem 2:** Discrepancy between published row count (42,342) and raw CSV rows (53,446).
  - *Resolution:* Streamed full file in audit; traced difference to 9,834 unannotated rows and 2,236 empty texts. Verified true usable count is 41,376.
- **Problem 3:** 2 duplicate text instances found with conflicting class labels.
  - *Resolution:* Forensic analysis in Milestone 2 showed Conflict 1 is an Arms Act conviction mislabeled civil in one row; Conflict 2 is an Art 32 police investigation writ with institutional registry ambiguity. Recommended dropping both clusters (6 rows).

---

# 15. CURRENT STATE SNAPSHOT

- **Current Milestone:** Milestone 2 — Exploratory Data Analysis (EDA) Complete; ready for Milestone 3 (Cleaning & Splitting).
- **Current Branch:** `feature/milestone-1-dataset-audit`
- **Last Completed Action:** Milestone 2 EDA executed (`src/eda.py`), notebook created (`notebooks/01_dataset_eda.ipynb`), full report generated (`docs/milestone_2_eda_report.md`), metrics saved (`outputs/metrics/eda_metrics.json`), and figures produced (`outputs/figures/`).
- **Last Verified Result:** 33,723 unique clean judgments identified (26,117 Civil, 7,606 Criminal, 3.43:1 imbalance). High discriminative signal confirmed in unigram/bigram domain vocabulary.
- **Current Task:** Awaiting user & ChatGPT review of Milestone 2 EDA findings and approval of the recommended cleaning policy.
- **Current Blocker:** Awaiting confirmation to proceed to Milestone 3.
- **Next Action:** Implement Milestone 3 data cleaning (`src/preprocessing.py`) and stratified train/val/test splitting (`src/dataset_split.py`).
- **Files Recently Changed:** `src/eda.py`, `notebooks/01_dataset_eda.ipynb`, `docs/milestone_2_eda_report.md`, `docs/LegalLensBackup.md`.
- **Working Tree Status:** Untracked/modified EDA deliverables ready to be committed.


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
