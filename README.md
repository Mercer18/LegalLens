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

## 3. Development Roadmap

We are following an incremental milestone strategy:

- **Milestone 0: Workspace & Environment Inspection** (Completed)
- **Milestone 1: Dataset Audit** (Pending dataset selection confirmation)
- **Milestone 2: Exploratory Data Analysis (EDA)**
- **Milestone 3: Data Cleaning & Stratified Splitting**
- **Milestone 4: Majority Baseline & Logistic Regression**
- **Milestone 5: TF-IDF & Linear SVM**
- **Milestone 6: Model Comparison & Evaluation**
- **Milestone 7: Error Analysis**
- **Milestone 8: Optional Transformer Experiment**
- **Milestone 9: Final Prediction Pipeline**
- **Milestone 10: Streamlit Application**
- **Milestone 11: Comprehensive Tests, Documentation & Viva Preparation**

---

## 4. Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd LegalLens
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
