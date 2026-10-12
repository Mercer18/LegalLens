# LegalLens: NLP-Based Classifier for Indian Court Judgments

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Project Overview

**LegalLens** is an NLP system engineered to classify full Indian court judgments into appropriate legal case categories. 

The primary end-to-end classification pipeline:
```
Indian Court Judgment (Full Text)
                ↓
    Text Extraction / Loading
                ↓
    Legal-Aware NLP Preprocessing
                ↓
        Feature Representation
                ↓
        Classification Model
                ↓
      Predicted Legal Category
```

This project focuses on **whole-judgment category classification**, preserving critical legal semantics (statutory references, citations, legal terms, negations) rather than aggressively stripping them.

---

## 2. Repository Structure

```
LegalLens/
├── .gitignore               # Comprehensive ignore rules (datasets, models, caches)
├── README.md                # Project documentation and roadmap
├── requirements.txt         # Core dependencies
├── data/
│   ├── raw/                 # Original unmodified dataset files (git-ignored)
│   ├── interim/             # Audited / cleaned intermediate artifacts (git-ignored)
│   └── processed/           # Stratified train/val/test splits (git-ignored)
├── notebooks/               # Exploratory data analysis notebooks
│   └── 01_dataset_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py            # Reproducible paths, random seeds, schema constants
│   ├── data_loader.py       # Dataset ingestion & format validation
│   ├── data_audit.py        # Missing values, duplicate detection, leakage checks
│   ├── preprocessing.py     # Domain-aware text normalization & tokenization
│   ├── dataset_split.py     # Leakage-free stratified train/val/test splits
│   ├── features.py          # TF-IDF & feature extraction
│   ├── models.py            # Baseline, Logistic Regression, Linear SVM
│   ├── train.py             # Reproducible training pipeline
│   ├── evaluate.py          # Evaluation metrics (Macro-F1, Confusion Matrix, etc.)
│   └── predict.py           # Unified inference pipeline
├── app/
│   └── streamlit_app.py     # Thin interactive UI presenting model predictions
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_preprocessing.py
├── outputs/
│   ├── figures/             # Confusion matrices, EDA plots
│   ├── reports/             # Per-class classification reports
│   └── metrics/             # JSON evaluation logs
├── models/                  # Serialized model pipelines (.joblib, git-ignored)
└── docs/                    # Design decisions, dataset audit notes, report drafts
```

---

## 3. Verified Empirical Performance (Held-Out Frozen Test Set)

The final model pipeline was evaluated **exactly once** on the held-out frozen test set ($N = 3,372$ judgments, 10% stratified split) under strict zero-leakage conditions:

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **93.59%** |
| **Macro-F1** | **90.61%** |
| **Civil F1-Score** | **95.90%** |
| **Criminal F1-Score** | **85.33%** |
| **Civil Recall** | **94.08%** |
| **Criminal Recall** | **91.68%** |

- **Architecture:** TF-IDF (1–2 ngrams, 40,000 features, sublinear TF) + Linear Support Vector Machine (`LinearSVC`, `class_weight='balanced'`, `C=1.0`).

---

## 4. Setup, Local Run & Testing

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mercer18/LegalLens.git
   cd LegalLens
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run unit test suite (22/22 tests passing):**
   ```bash
   pytest -v tests/
   ```

4. **Launch the Streamlit Web Application locally:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## 5. Deployment Guide: Streamlit Community Cloud

LegalLens is configured for deployment directly from GitHub to **Streamlit Community Cloud**:

1. **Repository & Branch:**
   - **Repository:** `Mercer18/LegalLens`
   - **Branch:** `main` (or active feature branch)
2. **Application Configuration in Streamlit Cloud Dashboard:**
   - **Main file path:** `app/streamlit_app.py`
   - **Python version:** `3.11` or `3.12` or `3.13`
3. **Artifact Resolution:**
   - The production model pipeline (`models/final_linear_svm_pipeline.joblib`, 2.05 MB) is tracked and included directly in the repository via `.gitignore` whitelisting.
   - Raw datasets, interim parquets, and intermediate models remain completely ignored.
   - Zero external APIs, secrets, or cloud storage buckets are required.
4. **Click Deploy:**
   - Streamlit Cloud automatically installs dependencies from `requirements.txt` and starts the app.
