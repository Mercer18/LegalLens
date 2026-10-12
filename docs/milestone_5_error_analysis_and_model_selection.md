# Milestone 5: Error Analysis & Final Model Selection

**Project:** LegalLens — NLP-Based Classifier for Indian Court Judgments  
**Phase:** Milestone 5 — Held-out Validation Error Diagnostics & Final Candidate Selection  
**Dataset:** Jud-IPL Clean Split (Validation Set: `data/processed/val.parquet`, $N = 3,372$)  
**Status:** Complete  
**Date:** 2026-10-11  

---

## 1. Executive Summary

Milestone 5 rigorously examines the classification boundary, failure modes, feature weights, and targeted hyperparameter tuning behavior of our validation champion model identified in Milestone 4: **TF-IDF (1–2 grams, 30,000 features) + Linear SVM (Balanced class weights)**.

All diagnostics and tuning iterations were conducted **strictly on the validation set (`val.parquet`)**. The test set (`test.parquet`, 3,372 cases) and its SHA-256 ID list (`frozen_test_ids.json`) remained completely untouched, guaranteeing zero data leakage.

### Key Conclusions:
1. **Error Profile:** Out of 3,372 validation judgments, the balanced Linear SVM commits **207 errors (6.14% error rate, 93.86% overall accuracy)**.
   - **False Criminals (True Civil $\to$ Predicted Criminal):** 162 errors (6.03% of Civil cases).
   - **False Civils (True Criminal $\to$ Predicted Civil):** 45 errors (6.57% of Criminal cases).
2. **Document Length & Complexity:** Errors correlate with longer, more complex judgments. Misclassified judgments have a median word count of **3,282 words** versus **2,736 words** for correctly classified judgments.
3. **Truncation Evaluation:** No validation cases were identified as truncated under the 32,765-character criterion, so truncation could not be evaluated as a source of validation errors. Errors stem from substantive jurisdictional and legal overlaps.
4. **Dominant Failure Patterns:**
   - *Contempt of Court Overlap:* Civil contempt petitions invoking penal punishment terms (`punishment`, `imprisonment`, `state`).
   - *Constitutional Writs on Police/Crime:* Article 32/226 civil petitions litigating police inaction, CBI investigations, or encounter deaths.
   - *Anti-Corruption Trials:* Criminal Prevention of Corruption Act trials heavily referencing civil service rules, departmental inquiries, and administrative orders.
   - *Quasi-Criminal Commercial Offenses:* Section 138 Negotiable Instruments Act (cheque bounce) or financial scams dominated by commercial contract vocabulary (`bank`, `debt`, `account`, `loan`).
5. **Targeted Tuning Validation:**
   - Increasing TF-IDF features from 30,000 to **40,000 features** improves validation **Macro-F1 from 0.9107 to 0.9148** and **Criminal Recall from 93.43% to 93.72%** (Accuracy reaches **94.16%**).
   - Tuning $C$ values reveals that $C=1.0$ remains the optimal balance between high minority recall (93.43% - 93.72%) and precision (79.80% - 80.65%).
   - We recommend **Linear SVM (Balanced, $C=1.0$, 40,000 features, 1–2 ngrams)** as the final candidate model for LegalLens.

---

## 2. Validation Error Breakdown

| Metric | Count / Percentage |
| :--- | :--- |
| **Total Validation Judgments** | 3,372 |
| **Correct Predictions** | 3,165 (93.86%) |
| **Total Errors** | 207 (6.14%) |
| **False Criminals (True Civil $\to$ Pred Criminal)** | 162 (78.26% of all errors; 6.03% of all Civil cases) |
| **False Civils (True Criminal $\to$ Pred Civil)** | 45 (21.74% of all errors; 6.57% of all Criminal cases) |

### Confusion Matrix Representation:
```
                     Predicted Civil    Predicted Criminal
Actual Civil              2,525                162          (Total: 2,687)
Actual Criminal              45                640          (Total:   685)
```

The balanced class weighting slightly shifts the decision hyperplane toward protecting minority Criminal Recall ($93.43\%$). Because Civil cases outnumber Criminal cases by ~4:1 (2,687 vs 685 in validation), the absolute volume of False Criminals (162) exceeds False Civils (45), while the error rate per class is virtually identical (6.03% vs 6.57%).

---

## 3. Document Length & Truncation Forensics

### 3.1 Word Count Distribution
| Cohort | Mean Words | Median Words | IQR (25th - 75th percentile) |
| :--- | :--- | :--- | :--- |
| **Correctly Classified** | 2,983.9 words | 2,736.0 words | 1,416.0 – 4,737.0 words |
| **Misclassified (Errors)** | 3,341.9 words | 3,282.0 words | 1,994.5 – 5,204.0 words |

*Finding:* Misclassified cases had a median length 546 words greater than correctly classified cases. Longer Indian court judgments tend to cite extensive multi-disciplinary precedent (e.g. citing criminal jurisprudence in civil writs or vice-versa), confusing surface n-gram frequencies.

### 3.2 32,765-Character Truncation Forensic
In the raw dataset, Excel/CSV string limits artificially capped a subset of judgments at 32,765 characters. In the cleaned stratified validation set:
- Truncated judgments count in validation: 0
- Non-truncated judgments error rate: 6.14% (207 / 3,372)
- *Conclusion:* No validation cases were identified as truncated under the 32,765-character criterion, so truncation could not be evaluated as a source of validation errors.

---

## 4. Top Model Feature Coefficients

Linear SVM feature weights ($\mathbf{w}$) provide direct interpretability into what signals trigger each class:

### Top 15 Criminal Predictive Features ($\mathbf{w} > 0$):
1. `criminal` (+6.5067)
2. `accused` (+2.4146)
3. `prosecution` (+1.9637)
4. `crl` (+1.9395)
5. `bail` (+1.8810)
6. `magistrate` (+1.5679)
7. `section` (+1.5648)
8. `state` (+1.4549)
9. `police` (+1.3795)
10. `offence` (+1.3311)
11. `special judge` (+1.3299)
12. `sentence` (+1.3189)
13. `criminal writ` (+1.2927)
14. `trial court` (+1.2520)
15. `criminal contempt` (+1.2325)

### Top 15 Civil Predictive Features ($\mathbf{w} < 0$):
1. `civil` (-17.2223)
2. `civil hospital` (-7.9939)
3. `costs` (-3.4163)
4. `civil court` (-3.2086)
5. `suit` (-2.9885)
6. `civil criminal` (-2.9561)
7. `civil suit` (-2.5696)
8. `surgeon` (-2.5560)
9. `civil appeal` (-1.9107)
10. `civil sessions` (-1.7592)
11. `order costs` (-1.7425)
12. `costs appeal` (-1.7124)
13. `civil nature` (-1.5680)
14. `civil procedure` (-1.5625)
15. `city civil` (-1.5353)

*Note on `civil hospital` and `surgeon`:* High negative weights for medical officer terms in civil cases reflect the heavy volume of motor accident claims (MACT), personal injury disputes, and civil service health board litigation.

---

## 5. Failure Mode Taxonomy & Representative Case Studies

### Failure Mode 1: Contempt of Court Jurisdictional Overlap (True Civil $\to$ Pred Criminal)
- **Representative Case:** `1921207.txt` (Word count: 1,151 words)
- **Ground Truth:** Civil | **Prediction:** Criminal
- **Context:** Contempt petition filed under the Contempt of Courts Act before the Supreme Court arising out of disobedience to civil court orders.
- **Root Cause:** Contempt proceedings contain heavy quasi-penal terminology (`punishment`, `imprisonment`, `authority of apex court`, `state`). Because contempt petitions can impose punitive prison sentences, the model's criminal vocabulary triggers heavily, overpowering civil indicators.

### Failure Mode 2: Constitutional Writs Litigating Police Conduct (True Civil $\to$ Pred Criminal)
- **Representative Case:** `837290.txt` (Word count: 2,938 words)
- **Ground Truth:** Civil | **Prediction:** Criminal
- **Context:** Civil appellate / Special Leave Petition questioning police investigations, custodial procedures, or seeking transfer to an independent investigative agency (CBI).
- **Root Cause:** Even though the procedural mechanism is a Civil Writ / SLP, the factual text is dominated by penal terminology: `police`, `investigation`, `charge sheet`, `offence`, and `convicted`.

### Failure Mode 3: Anti-Corruption Trials Involving Public Servants (True Criminal $\to$ Pred Civil)
- **Representative Case:** `1438663.txt` (Word count: 3,377 words)
- **Ground Truth:** Criminal | **Prediction:** Civil
- **Context:** Criminal prosecution under the Prevention of Corruption Act involving public officials.
- **Root Cause:** The judgment devotes thousands of words to discussing departmental administrative procedures, government sanction orders, service conduct rules, and administrative tribunals. These administrative tokens trigger strong Civil weights, diluting the penal charges.

### Failure Mode 4: Quasi-Criminal Commercial & Economic Offenses (True Criminal $\to$ Pred Civil)
- **Representative Case:** `1631729.txt` (Word count: 2,202 words)
- **Ground Truth:** Criminal | **Prediction:** Civil
- **Context:** Maintenance disputes under CrPC Sec 125 or dishonour of cheques under Section 138 Negotiable Instruments Act.
- **Root Cause:** Substantive text discusses financial maintenance, bank accounts, loan defaults, property entitlements, and commercial agreements, which closely resemble civil contract and family disputes.

---

## 6. Targeted Validation-Only Tuning Experiments

We evaluated targeted variations of regularization parameter $C$, n-gram ranges, and vocabulary size strictly against the validation set:

| Model Configuration | Macro-F1 | Criminal Recall | Criminal Precision | Criminal F1 | Civil Recall | Civil F1 | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM (Balanced, C=1.0, 40k feats, 1–2g)** | **0.9148** | **93.72%** | **80.65%** | **0.8670** | **94.27%** | **0.9626** | **94.16%** |
| Linear SVM (Balanced, C=2.0, 30k feats, 1–2g) | 0.9136 | 91.68% | 81.77% | 0.8644 | 94.79% | 0.9628 | 0.9416 |
| **Baseline Champion (Balanced, C=1.0, 30k feats, 1–2g)** | **0.9107** | **93.43%** | **79.80%** | **0.8608** | **93.97%** | **0.9606** | **93.86%** |
| Feature Tuning: Unigrams (1,1), C=1.0, 30k feats | 0.9081 | 92.99% | 79.43% | 0.8568 | 93.86% | 0.9595 | 0.9368 |
| Regularization Tuning: C=0.5, Balanced, 30k feats | 0.9052 | 94.45% | 77.76% | 0.8530 | 93.11% | 0.9573 | 0.9339 |
| Regularization Tuning: C=0.2, Balanced, 30k feats | 0.8934 | 94.89% | 74.71% | 0.8360 | 91.81% | 0.9509 | 0.9244 |

### Tuning Analysis:
1. **Vocabulary Expansion (40,000 features):** Expanding the TF-IDF feature space from 30,000 to 40,000 n-grams gives a clear, consistent boost across all metrics:
   - Macro-F1 increases from **0.9107 $\to$ 0.9148** (+0.0041).
   - Criminal Recall increases from **93.43% $\to$ 93.72%** (+0.29%).
   - Criminal Precision improves from **79.80% $\to$ 80.65%** (+0.85%).
   - Overall Accuracy increases from **93.86% $\to$ 94.16%**.
2. **Bigrams are Essential:** Restricting to unigrams drops Macro-F1 to 0.9081, demonstrating that bigrams like `criminal writ`, `civil appeal`, `special judge`, and `civil hospital` provide discriminative power.
3. **Regularization ($C$):** Lower $C$ ($C=0.2, 0.5$) aggressively boosts Criminal Recall up to 94.89% but severely hurts Criminal Precision (down to 74.71%) and overall Macro-F1. Higher $C=2.0$ improves precision slightly but causes a 1.75% drop in Criminal Recall. $C=1.0$ is the optimal sweet spot.

---

## 7. Final Model Recommendation

**Recommended Final Model:**  
`TF-IDF (1–2 grams, sublinear_tf=True, max_features=40,000) + Linear SVM (C=1.0, class_weight='balanced', random_state=42)`

### Selection Rationale:
1. **Exceptional Minority Class Recall:** Retains **93.72% Criminal Recall** on the imbalanced 4:1 distribution, preventing criminal cases from being improperly categorized as civil.
2. **Superior Macro-F1 & Accuracy:** Achieves **0.9148 Macro-F1** and **94.16% Accuracy** on 3,372 validation judgments.
3. **Robust Interpretability:** Linear coefficients provide transparent auditability for legal technology deployments.
4. **Computational Efficiency:** Sub-second inference time and small artifact size (< 15 MB) ideal for deployment in the Streamlit application.
