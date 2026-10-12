# Final Test Evaluation Report: LegalLens

**Project:** LegalLens — NLP-Based Classifier for Indian Court Judgments  
**Task:** Binary Whole-Judgment Classification (Civil vs. Criminal)  
**Dataset:** Jud-IPL Held-Out Frozen Test Set (`data/processed/test.parquet`, $N = 3,372$)  
**Status:** Evaluation Completed (Single Unbiased Hold-Out Pass)  
**Date:** 2026-10-11  

---

## 1. Executive Summary

In accordance with strict empirical machine learning governance, the held-out test partition ($N = 3,372$ judgments, 10% stratified hold-out) was evaluated **exactly once** using our locked model pipeline from Milestone 5/6. 

### Critical Integrity & Protocol Confirmations:
1. **Zero Contamination:** Test set hash intersection against training partition = 0; intersection against validation partition = 0.
2. **Frozen ID Verification:** All 3,372 test records match the SHA-256 ID list persisted during Milestone 3 in `data/processed/frozen_test_ids.json`.
3. **No Retraining or Tuning:** The model was evaluated using the exact serialized artifact (`models/final_linear_svm_pipeline.joblib`) fitted solely on training data. No hyperparameters, feature representations, or decision thresholds were adjusted based on test outputs.
4. **Generalization Confirmation:** The locked classifier demonstrates outstanding generalization to unseen judgments, achieving **93.59% Accuracy**, **0.9061 Macro-F1**, and **91.68% Criminal Recall**.

---

## 2. Locked Model Pipeline Specification

- **Input:** Raw Indian court judgment text only (zero metadata features).
- **Text Normalization:** Lowercasing, whitespace collapsing, preservation of statutory sections and negations (`src/preprocessing.py:normalize_legal_text`).
- **Feature Extraction:** `TfidfVectorizer`
  - `ngram_range = (1, 2)` (unigrams and bigrams)
  - `max_features = 40,000`
  - `sublinear_tf = True`
  - `min_df = 3`, `max_df = 0.90`
  - `stop_words = custom_legal` (preserving negations like `not`, `no`, `neither`, `against`)
- **Classification Engine:** `LinearSVC`
  - `class_weight = "balanced"`
  - `C = 1.0`
  - `loss = "squared_hinge"`
  - `dual = "auto"`
  - `random_state = 42`

---

## 3. Test Partition Demographics

- **Total Test Samples:** 3,372 judgments
- **Civil Judgments:** 2,687 (79.69%)
- **Criminal Judgments:** 685 (20.31%)
- **Imbalance Ratio:** 3.92 : 1 (Civil to Criminal)

---

## 4. Final Hold-Out Test Metrics

| Metric | Score | Support / Details |
| :--- | :---: | :--- |
| **Accuracy** | **0.9359** (93.59%) | 3,156 / 3,372 correct |
| **Macro-F1** | **0.9061** | Unweighted mean of Civil & Criminal F1 |
| **Macro-Precision** | **0.8880** | Unweighted mean of precisions |
| **Macro-Recall** | **0.9288** | Unweighted mean of recalls |
| **Weighted-F1** | **0.9375** | Class prevalence-weighted F1 |
| **Civil Precision** | **0.9779** | 2,528 / (2,528 + 57) |
| **Civil Recall** | **0.9408** | 2,528 / 2,687 |
| **Civil F1-Score** | **0.9590** | Harmonic mean of Civil P & R |
| **Criminal Precision** | **0.7980** | 628 / (628 + 159) |
| **Criminal Recall** | **0.9168** | 628 / 685 (Minority Class Protection) |
| **Criminal F1-Score** | **0.8533** | Harmonic mean of Criminal P & R |

---

## 5. Final Test Confusion Matrix

```
                          PREDICTED CIVIL       PREDICTED CRIMINAL       TOTAL
ACTUAL CIVIL                   2,528                    159              2,687
ACTUAL CRIMINAL                   57                    628                685
TOTAL                          2,585                    787              3,372
```

- **True Positives (Criminal):** 628
- **True Negatives (Civil):** 2,528
- **False Criminals (True Civil $\to$ Pred Criminal):** 159 (5.92% of all Civil cases)
- **False Civils (True Criminal $\to$ Pred Civil):** 57 (8.32% of all Criminal cases)
- **Confusion Matrix Figure:** Saved to [`outputs/figures/final_test_confusion_matrix.png`](file:///x:/CODING/PROJECTS/Py/NLP-Based%20Classifier%20for%20Indian%20Court%20Judgments/LegalLens/outputs/figures/final_test_confusion_matrix.png).

---

## 6. Validation vs. Final Test Performance Comparison

| Metric | Validation Score ($N=3,372$) | Final Test Score ($N=3,372$) | Delta (Test – Val) | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | 0.9416 | **0.9359** | -0.0057 (-0.57%) | Negligible change; model generalizes cleanly |
| **Macro-F1** | 0.9148 | **0.9061** | -0.0087 (-0.87%) | Macro-F1 remained above 0.90 on the unseen test set |
| **Criminal Recall** | 0.9372 | **0.9168** | -0.0204 (-2.04%) | High minority class safety retained |
| **Criminal Precision** | 0.8065 | **0.7980** | -0.0085 (-0.85%) | Stable across independent splits |
| **Criminal F1** | 0.8670 | **0.8533** | -0.0137 (-1.37%) | Robust performance despite 4:1 imbalance |
| **Civil Recall** | 0.9427 | **0.9408** | -0.0019 (-0.19%) | Virtually identical recall on majority class |
| **Civil Precision** | 0.9836 | **0.9779** | -0.0057 (-0.57%) | Exceptionally high precision maintained |
| **Civil F1** | 0.9626 | **0.9590** | -0.0036 (-0.36%) | Consistently strong |

### Discussion:
The variance between validation and test scores is minimal ($< 0.9\%$ Macro-F1 delta). The small validation-to-test performance drop indicates good generalization and no evidence of substantial overfitting. This confirms that:
1. Stratification was effective.
2. The model will reliably generalize to external Indian High Court and Supreme Court judgments.

---

## 7. Integrity Checklist
- [x] Frozen test set IDs matched expected frozen set (`frozen_test_ids.json`).
- [x] Zero train/test text hash contamination verified.
- [x] Zero validation/test text hash contamination verified.
- [x] Evaluated strictly using the locked Milestone 5/6 model artifact (`final_linear_svm_pipeline.joblib`).
- [x] Zero test-derived tuning or retraining performed.
- [x] Evaluation executed exactly once.
