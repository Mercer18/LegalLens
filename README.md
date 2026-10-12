# LegalLens

**An NLP-Based Classifier for Indian Court Judgments**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Tests: 22 Passed](https://img.shields.io/badge/tests-22%20passed-brightgreen.svg)](tests/)

---

## Quick Start

Launch the interactive classifier locally in three commands:

```bash
git clone https://github.com/Mercer18/LegalLens.git
cd LegalLens
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Run the complete test suite:

```bash
pytest -v tests/
```

---

## 1. Project Overview

**LegalLens** is an applied Natural Language Processing (NLP) system designed to perform whole-document binary classification of Indian court judgments into two primary jurisdictions:

- **Civil**
- **Criminal**

Indian court judgments are notoriously long, linguistically complex, and structurally dense documents that cite multi-disciplinary precedent across both substantive and procedural law. LegalLens automates whole-judgment routing directly from raw judgment text, bypassing reliance on external procedural metadata or outcome annotations.

### Input & Output Specification
- **Input:** Raw Indian court judgment text (Supreme Court or High Court appellate decisions, trial orders, or operative excerpts).
- **Output:** Predicted jurisdiction (`Civil` vs. `Criminal`), signed hyperplane decision score, estimated confidence score, and text length metrics.

### Academic Scope & Strict Boundaries
> [!IMPORTANT]
> **What LegalLens is NOT:**
> - **NOT legal advice** or a substitute for qualified legal professionals.
> - **NOT an AI lawyer** or judicial decision-making engine.
> - **NOT case outcome prediction** (e.g., predicting appeal acceptance or dismissal).
> - **NOT sentence-level rhetorical-role labeling** (e.g., facts vs. arguments vs. ratio decidendi).

---

## 2. Key Objectives & Results

- **Leakage-Free NLP Architecture:** Clean and partition court records without cross-split contamination.
- **Handling Severe Class Imbalance:** Mitigate the natural ~4:1 Civil-to-Criminal imbalance using cost-sensitive class balancing.
- **Interpretable Classical Machine Learning:** Build high-performing, explainable linear baselines without opaque multi-billion-parameter overhead.
- **Immutable Hold-Out Evaluation:** Evaluate the final locked candidate exactly once on a frozen test set.
- **Production Web Deployment:** Deliver a self-contained, deployment-ready Streamlit interface.

### Headline Benchmark Summary

| Metric | Validation Set ($N=3,372$) | Unseen Held-Out Test Set ($N=3,372$) |
| :--- | :---: | :---: |
| **Accuracy** | 94.16% | **93.59%** |
| **Macro-F1** | 0.9148 | **0.9061** |
| **Weighted-F1** | 0.9431 | **0.9375** |
| **Civil F1-Score** | 0.9626 | **0.9590** |
| **Criminal F1-Score** | 0.8670 | **0.8533** |
| **Criminal Recall (Minority)** | 93.72% | **91.68%** |

*Macro-F1 remained above 0.90 on the unseen test set.*

---

## 3. Dataset & Empirical Audit (Jud-IPL)

LegalLens was developed using the **Jud-IPL** corpus (*CloLex: Exploring Closed Lexicals for LegalTech*, Jain et al., 2024, CC0 Public Domain). An empirical audit of the raw corpus file (`data/raw/case_files_total.csv`, 838.44 MB) uncovered important differences from published summaries:

| Dimension | Published Summary | Empirical Audit Measurement | Engineering Action |
| :--- | :---: | :---: | :--- |
| **Total Rows** | 42,342 | **53,446** | Ingested full raw CSV stream |
| **Missing Category** | "Undetermined" string | **9,834 `NaN` rows** | Filtered during data cleaning |
| **Empty Text Rows** | 0 | **2,236 rows (in labeled subset)** | Dropped null/blank records |
| **Duplicate Texts** | None reported | **7,651 duplicate copies** | Deduplicated via SHA-256 hash |
| **Conflicting Labels** | None reported | **2 clusters (6 rows)** | Fully pruned to preserve ground truth |
| **Text Length Cap** | Uncapped | **32,765 chars ($2^{15}-1$)** | Excel/database export truncation artifact |

> [!NOTE]
> The raw 838 MB dataset is excluded from version control via `.gitignore`. Downstream analysis showed that no validation cases were identified as truncated under the 32,765-character criterion, so truncation could not be evaluated as a source of validation errors.  
> Detailed audit report: [`docs/milestone_1_dataset_audit.md`](docs/milestone_1_dataset_audit.md)

---

## 4. Data Cleaning & Stratified Splitting

To ensure reproducibility and zero data leakage, a four-stage data cleaning waterfall was executed:

```
Raw CSV Records (53,446)
       ↓ Drop unannotated/NaN category rows (-9,834)
Labeled Subset (43,612)
       ↓ Drop null/empty judgment texts (-2,236)
Non-Empty Valid Cases (41,376)
       ↓ Drop 2 conflicting-label duplicate clusters (-6)
Consistent Candidate Pool (41,370)
       ↓ Deduplicate via exact SHA-256 hash (-7,653 copies)
Final Clean Unique Corpus (33,717 cases)
```

### Final Corpus Demographics
- **Total Unique Judgments:** 33,717
- **Civil:** 26,867 (79.68%)
- **Criminal:** 6,850 (20.32%)
- **Class Imbalance Ratio:** 3.92 : 1

### Stratified Split (80 / 10 / 10)
Partitioned with `random_state=42` and strict class stratification:
- **Training Set:** 26,973 judgments (21,493 Civil / 5,480 Criminal)
- **Validation Set:** 3,372 judgments (2,687 Civil / 685 Criminal)
- **Held-Out Test Set:** 3,372 judgments (2,687 Civil / 685 Criminal) — **Frozen in [`data/processed/frozen_test_ids.json`](data/processed/frozen_test_ids.json)**

*Contamination Check: Verified 0 text-hash and 0 ID overlap across all pairwise partition intersections.*  
Detailed cleaning report: [`docs/milestone_3_cleaning_and_split_report.md`](docs/milestone_3_cleaning_and_split_report.md)

---

## 5. NLP Feature Engineering Pipeline

Whole-judgment classification relies entirely on domain-aware text representations of the `judgement` body:

1. **Text Normalization:**
   - Standardizes internal whitespace, tabs, and newlines.
   - Converts text to lowercase while preserving statutory references (e.g., `section 302`, `article 226`) and legal punctuation.
2. **Stopword Strategy with Preserved Negations:**
   - Standard English stopwords are filtered, but critical statutory negations are strictly preserved:
     ```python
     LEGAL_PRESERVED_TOKENS = {"not", "no", "nor", "neither", "never", "without", "against"}
     ```
3. **TF-IDF Vectorization:**
   - **N-Gram Range:** `(1, 2)` (captures key legal phrases like `writ petition`, `penal code`, `special judge`, `bail application`).
   - **Vocabulary Capacity:** `40,000` features.
   - **Sublinear Term Frequency:** `sublinear_tf=True` (applies $1 + \log(\text{tf})$ to prevent long judgments from dominating due to repeated citations).
   - **Document Frequency Cutoffs:** `min_df=3` (prunes hapax legomena/OCR errors), `max_df=0.90` (removes boilerplate headers).

---

## 6. Model Exploration & Validation Benchmarks

Multiple classical architectures were trained on the training partition ($N=26,973$) and benchmarked on the validation partition ($N=3,372$):

| Model | Class Weight | Macro-F1 | Criminal Recall | Criminal Precision | Civil Recall | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (40k feats, locked candidate)** | **Balanced** | **0.9148** | **93.72%** | **80.65%** | **94.27%** | **94.16%** |
| Linear SVM (30k feats, C=1.0) | Balanced | 0.9107 | 93.43% | 79.80% | 93.97% | 93.86% |
| Linear SVM (30k feats, C=1.0) | Default | 0.9100 | 88.47% | 83.24% | 95.46% | 94.04% |
| Logistic Regression (30k feats, C=1.0) | Balanced | 0.8886 | 94.31% | 73.91% | 91.51% | 92.08% |
| Logistic Regression (30k feats, C=1.0) | Default | 0.8895 | 85.26% | 80.00% | 94.57% | 92.67% |
| Majority Class Baseline | Most Frequent | 0.4435 | 0.00% | 0.00% | 100.00% | 79.69% |

### Why Linear SVM with Balanced Weights?
Default models suffer from minority class under-detection in imbalanced settings. Balanced class weighting (`class_weight='balanced'`) inversely weights classes by their frequency, protecting Criminal Recall (>93%) while maintaining high precision. Expanding features from 30,000 to 40,000 boosted validation Macro-F1 from 0.9107 to 0.9148.

Detailed modeling and error analysis reports:  
- [`docs/milestone_4_baseline_and_classical_models.md`](docs/milestone_4_baseline_and_classical_models.md)  
- [`docs/milestone_5_error_analysis_and_model_selection.md`](docs/milestone_5_error_analysis_and_model_selection.md)

---

## 7. Final Hold-Out Test Evaluation

The locked model pipeline was evaluated **once** on the frozen unseen test partition ($N=3,372$):

```
Locked Configuration:
TF-IDF (1–2 ngrams, 40,000 features, sublinear_tf=True, custom legal stopwords)
+ LinearSVC (C=1.0, class_weight='balanced', loss='squared_hinge', random_state=42)
```

### Final Confusion Matrix (Test Set, $N = 3,372$)

```
                          PREDICTED CIVIL       PREDICTED CRIMINAL       TOTAL
ACTUAL CIVIL                   2,528                    159              2,687
ACTUAL CRIMINAL                   57                    628                685
TOTAL                          2,585                    787              3,372
```

- **True Positives (Criminal):** 628
- **True Negatives (Civil):** 2,528
- **False Criminals (True Civil $\to$ Pred Criminal):** 159 (5.92% error rate on Civil)
- **False Civils (True Criminal $\to$ Pred Civil):** 57 (8.32% error rate on Criminal)

### Validation vs. Test Generalization Comparison

| Metric | Validation Set | Held-Out Test Set | Delta (Test – Val) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 94.16% | **93.59%** | -0.57% |
| **Macro-F1** | 0.9148 | **0.9061** | -0.87% |
| **Weighted-F1** | 0.9431 | **0.9375** | -0.56% |
| **Criminal Recall** | 93.72% | **91.68%** | -2.04% |
| **Criminal Precision** | 80.65% | **79.80%** | -0.85% |
| **Criminal F1** | 0.8670 | **0.8533** | -1.37% |
| **Civil Recall** | 94.27% | **94.08%** | -0.19% |
| **Civil Precision** | 98.36% | **97.79%** | -0.57% |
| **Civil F1** | 0.9626 | **0.9590** | -0.36% |

> [!NOTE]
> *The small validation-to-test performance drop indicates good generalization and no evidence of substantial overfitting.*  
> Detailed test report: [`docs/final_test_evaluation_report.md`](docs/final_test_evaluation_report.md)

---

## 8. Prediction Interface & Architecture

Inference is encapsulated in `src/predict.py`:

```
Raw Court Judgment Text
          ↓
normalize_legal_text() (Whitespace & citation cleaning)
          ↓
TfidfVectorizer.transform() (Fitted 40k vocabulary)
          ↓
LinearSVC.decision_function() (Hyperplane distance)
          ↓
Sigmoid Calibration: p = 1 / (1 + exp(-decision_score))
          ↓
Structured Output Dictionary
```

### Python API Example

```python
from src.predict import predict_judgment

judgment_text = """
This criminal appeal is directed against the judgment and order of conviction 
passed by the learned Sessions Judge under Section 302 of the Indian Penal Code. 
The appellant was sentenced to rigorous imprisonment for life. Bail application was rejected.
"""

result = predict_judgment(judgment_text)
print(result)
```

**Returned Payload:**
```json
{
  "predicted_class": "Criminal",
  "category": "criminal",
  "decision_score": 1.7098,
  "confidence_prob": 0.8468,
  "probability_criminal": 0.8468,
  "probability_civil": 0.1532,
  "word_count": 42,
  "char_count": 274,
  "is_valid": true,
  "status": "success"
}
```

### Important Technical Caveat: Estimated Confidence
`LinearSVC` optimizes a maximum-margin hyperplane (squared hinge loss) rather than log-loss, producing an uncalibrated geometric decision margin. The application calculates **"Estimated Confidence"** via a standard sigmoid transformation of that margin. It is a heuristic confidence indicator, **not a formally calibrated Bayesian probability**.

---

## 9. Streamlit Interactive Application

The interactive web application is located at `app/streamlit_app.py`.

```bash
streamlit run app/streamlit_app.py
```

### Key UI Features:
1. **Direct Text Entry:** Paste complete judgment text or operative paragraphs into a multi-line input area.
2. **Pre-Loaded Example Judgments:** Includes three domain-accurate case studies:
   - *Criminal Case:* Murder conviction and life imprisonment appeal under IPC 302.
   - *Civil Case:* Land suit seeking specific performance of an agreement to sell.
   - *Civil Case:* Motor Accidents Claims Tribunal (MACT) road accident compensation appeal.
3. **Transparent Decision Dashboard:** Displays the predicted category badge, signed decision score, estimated confidence percentage, and word/character counts.
4. **Decision Distribution Bar:** Displays relative decision weights between Civil and Criminal.
5. **Academic Disclaimer:** Prominently highlights the research boundaries and non-legal-advice status.
6. **Graceful Validation:** Rejects empty or whitespace-only inputs without crashing.

---

## 10. Repository Structure

```
LegalLens/
├── .gitignore               # Excludes raw data, intermediate caches, and models (whitelisting final pipeline)
├── README.md                # Project documentation and technical overview
├── requirements.txt         # Runtime & test dependencies (Python 3.13 verified)
├── app/
│   └── streamlit_app.py     # Streamlit demonstration web interface
├── data/
│   ├── raw/                 # case_files_total.csv (git-ignored, 838 MB)
│   ├── interim/             # Cleaned Parquet/CSV artifacts (git-ignored)
│   └── processed/           # Train/Val/Test Parquets & frozen_test_ids.json
├── docs/                    # Systematic technical milestone reports
│   ├── LegalLensBackup.md   # Living project memory & decision log
│   ├── milestone_1_dataset_audit.md
│   ├── milestone_2_eda_report.md
│   ├── milestone_3_cleaning_and_split_report.md
│   ├── milestone_4_baseline_and_classical_models.md
│   ├── milestone_5_error_analysis_and_model_selection.md
│   └── final_test_evaluation_report.md
├── models/
│   └── final_linear_svm_pipeline.joblib  # Locked production pipeline artifact (2.05 MB, Git tracked)
├── notebooks/
│   └── 01_dataset_eda.ipynb # Reproducible exploratory data analysis notebook
├── outputs/
│   ├── figures/             # Confusion matrices and coefficient plots
│   └── metrics/             # JSON evaluation benchmarks & audit metrics
├── src/
│   ├── config.py            # Global paths, seeds, and column constants
│   ├── data_audit.py        # Dataset audit streaming script
│   ├── preprocessing.py     # Text cleaning & SHA-256 deduplication
│   ├── dataset_split.py     # Stratified 80/10/10 data splitting
│   ├── features.py          # TF-IDF feature builder
│   ├── models.py            # Baseline, Logistic Regression, Linear SVM definitions
│   ├── train.py             # Reproducible training & benchmark pipeline
│   ├── evaluate.py          # Metrics computation & confusion matrix generation
│   ├── error_analysis.py    # Validation diagnostics & targeted tuning
│   ├── predict.py           # Unified inference pipeline
│   └── final_evaluation.py   # Unbiased hold-out test evaluation script
└── tests/                   # 22 automated unit tests across pipeline
    ├── test_dataset_split.py
    ├── test_error_analysis.py
    ├── test_models.py
    ├── test_predict.py
    ├── test_preprocessing.py
    └── test_streamlit_app.py
```

---

## 11. Streamlit Community Cloud Deployment

LegalLens is configured for deployment on **Streamlit Community Cloud**:

1. **Repository:** `Mercer18/LegalLens`
2. **Branch:** `main`
3. **Main file path:** `app/streamlit_app.py`
4. **Python Version:** `3.11`, `3.12`, or `3.13`
5. **Model Artifact Availability:**
   - The production artifact `models/final_linear_svm_pipeline.joblib` (2.05 MB) is whitelisted in `.gitignore` and tracked by Git.
   - Streamlit Cloud clones the model artifact directly; no external S3/GCS buckets, LFS, or secrets are required.
   - Large raw datasets remain excluded from the deployment footprint.

---

## 12. Systematic Milestone History

LegalLens was developed through an incremental, milestone-driven workflow:

- **M0 Scaffolding (`0b3c349`):** Repository architecture, dependencies, and environment setup.
- **M1 Dataset Audit (`8b27cac`):** Raw Jud-IPL file inspection, uncovering row counts and length caps.
- **Living Memory Setup (`639eecc`):** Established `LegalLensBackup.md` as the persistent single source of truth.
- **M2 Exploratory Data Analysis (`3a6d47c`):** Duplicate cluster forensics and text length distributions.
- **M3 Data Cleaning & Split (`1a53bc2`):** Deduplication to 33,717 unique cases, 80/10/10 stratified split, test freezing.
- **M4 Baseline Model Experiments (`111e15b`):** Majority baseline, Logistic Regression, and Linear SVM benchmarks.
- **M5 Error Analysis & Tuning (`1164c16`):** Document length vs. error analysis, failure modes, 40k feature tuning.
- **M6 Prediction Pipeline (`93762b6`):** Inference interface in `src/predict.py` and unit testing.
- **Final Hold-Out Test Evaluation (`03a073b`):** One-shot evaluation on frozen test set (Accuracy: 93.59%, Macro-F1: 0.9061).
- **M7 Streamlit Demo & Deployment (`3cbac30`):** Interactive UI, deployment configuration, and test suite.
- **Main Branch Merge (`d428221`):** Complete milestone history merged to `main` via PR #1.

---

## 13. License & Academic Attribution

This project is released under the **MIT License**.

If you reference this work or dataset, please cite the underlying corpus:
```bibtex
@inproceedings{jain2024clolex,
  title={CloLex: Exploring Closed Lexicals for LegalTech},
  author={Jain, Bhavya and Harde, Shweta and Jain, Aditya},
  booktitle={Proceedings of the International Conference on Natural Language Processing},
  year={2024},
  publisher={Springer}
}
```
