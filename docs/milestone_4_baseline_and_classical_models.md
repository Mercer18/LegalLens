# Milestone 4: Majority Baseline & Classical Models Benchmark Report

**Project Title:** LegalLens — NLP-Based Classifier for Indian Court Judgments  
**Date:** October 11, 2026  
**Auditor / NLP Engineer:** Antigravity (Implementation Agent)  
**Modules Implemented:** `src/features.py`, `src/models.py`, `src/evaluate.py`, `src/train.py`  
**Test Suite:** `tests/test_models.py` (9/9 Passed across suite)  
**Metrics Artifact:** `outputs/metrics/milestone_4_model_metrics.json`  
**Saved Model Pipelines:** `models/logistic_regression_balanced.joblib`, `models/linear_svm_balanced.joblib`  
**Evaluation Set:** Validation Set (3,372 judgments: 2,687 Civil, 685 Criminal)  
*(Note: Test Set remains frozen and completely untouched).*

---

## 1. Executive Summary & Headline Results

Milestone 4 trained, evaluated, and compared the **Majority Baseline**, **TF-IDF + Logistic Regression**, and **TF-IDF + Linear Support Vector Machine (Linear SVM)** on the held-out validation set.

### Headline Comparison Table (Ranked by Macro-F1)

| Model Architecture | Class Weight | Macro-F1 | Criminal Recall | Criminal Precision | Criminal F1 | Civil Recall | Civil F1 | Accuracy | Weighted-F1 |
|---|---|---|---|---|---|---|---|---|---|
| **Linear SVM** *(Best Overall)* | **Balanced** | **0.9107** | **0.9343** | **0.7980** | **0.8608** | **0.9397** | **0.9606** | **0.9386** | **0.9403** |
| **Linear SVM** | Default | 0.9100 | 0.8847 | 0.8324 | 0.8577 | 0.9546 | 0.9623 | 0.9404 | 0.9411 |
| **Logistic Regression** | Default | 0.8895 | 0.8526 | 0.8000 | 0.8254 | 0.9457 | 0.9536 | 0.9267 | 0.9276 |
| **Logistic Regression** | **Balanced** | 0.8886 | **0.9431** | 0.7391 | 0.8287 | 0.9151 | 0.9485 | 0.9208 | 0.9242 |
| **Majority Baseline** | Most Frequent | 0.4435 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8869 | 0.7969 | 0.7068 |

---

## 2. Experimental Methodology & Rigor

### Leakage-Free Design:
- **TF-IDF Fit Boundary:** Fitted strictly on the 26,973 training documents. Validation texts were transformed using the frozen training vocabulary.
- **Frozen Test Set:** The test partition (3,372 cases) was **not accessed, fitted, or evaluated**.
- **Features Used:** Strictly the clean court `judgement` text. No metadata (`case_type`, `case_info`, `label`, `proof_sentence`, etc.) was supplied.
- **Reproducibility:** Seed `42` enforced across splits, vectorization, and model initializations.

---

## 3. Exact Model & Vectorizer Configurations

### TF-IDF Configuration (`src/features.py`)
- **N-Gram Range:** Unigrams and Bigrams `(1, 2)`
- **Max Features:** `30,000` terms
- **Sublinear Term Frequency:** `sublinear_tf=True` (applies $1 + \log(\text{tf})$ to prevent long documents from over-weighting repeated words)
- **Document Frequency Cutoffs:** `min_df=3` (prunes rare hapax legomena), `max_df=0.90` (prunes ubiquitous legal boilerplate)
- **Stopwords:** Standard English stopwords, with **explicit preservation of legal negations** (`not`, `no`, `nor`, `neither`, `never`, `without`, `against`).
- **Matrix Dimension:** Train: `(26,973, 30,000)`, Val: `(3,372, 30,000)`

### Classifier Hyperparameters (`src/models.py`)
1. **Majority Baseline:** `DummyClassifier(strategy='most_frequent')`
2. **Logistic Regression:**
   - Solver: `lbfgs`
   - Regularization: L2, $C=1.0$
   - Max Iterations: 1,000
   - Compared: `class_weight=None` vs. `class_weight='balanced'`
3. **Linear SVM:**
   - Architecture: `LinearSVC`
   - Loss: Squared hinge, $C=1.0$
   - Dual: `auto`
   - Compared: `class_weight=None` vs. `class_weight='balanced'`

---

## 4. Confusion Matrix Analysis (Validation Set: 3,372 Total)

```
                       Predicted Civil    Predicted Criminal
Majority Baseline:
  True Civil (2,687)         2,687                  0
  True Criminal (685)          685                  0

Logistic Regression (Default):
  True Civil (2,687)         2,541                146
  True Criminal (685)          101                584

Logistic Regression (Balanced):
  True Civil (2,687)         2,459                228
  True Criminal (685)           39                646

Linear SVM (Default):
  True Civil (2,687)         2,565                122
  True Criminal (685)           79                606

Linear SVM (Balanced):
  True Civil (2,687)         2,525                162
  True Criminal (685)           45                640
```

---

## 5. Key Analytical Takeaways for Viva Defense

1. **Failure of Naive Accuracy:**  
   The Majority Baseline demonstrates why accuracy is misleading in imbalanced legal NLP. An accuracy of **79.69%** masks a catastrophic failure to identify a single criminal case (**0% Criminal Recall, 0.00 Macro-F1**).
2. **Impact of Class Weighting:**  
   In both architectures, applying `class_weight='balanced'` dramatically suppresses false negatives in the minority class:
   - Logistic Regression Criminal Recall increases from **85.26% to 94.31%** (false negatives drop from 101 to 39).
   - Linear SVM Criminal Recall increases from **88.47% to 93.43%** (false negatives drop from 79 to 45).
3. **Linear SVM Superiority:**  
   Linear SVM (Balanced) provides the best balance between precision and recall, achieving the highest **Macro-F1 (0.9107)**, **93.86% Accuracy**, **93.43% Criminal Recall**, and **93.97% Civil Recall**.
4. **Viability of Classical NLP:**  
   Achieving **> 0.91 Macro-F1** with TF-IDF and classical linear models validates our earlier EDA finding: the legal terminology contrast between Civil and Criminal jurisprudence in Indian judgments is immensely powerful.
