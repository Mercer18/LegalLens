"""
LegalLens Milestone 5: Error Analysis & Targeted Model Tuning Module
Analyzes validation errors from the balanced Linear SVM:
1. Error breakdown by class (False Positives vs False Negatives).
2. Document length and 32,765-character truncation impact.
3. Feature coefficient inspection & legal linguistic root cause extraction.
4. Representative case study extraction with concise previews.
5. Targeted validation-only tuning experiments (C values, n-gram ranges, features).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    COL_CLEAN_TARGET,
    COL_CLEAN_TEXT,
    COL_ID,
    COL_TEXT_HASH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_CIVIL,
    TARGET_CRIMINAL,
    TRAIN_FILE,
    VAL_FILE,
)
from src.evaluate import compute_metrics, plot_and_save_confusion_matrix
from src.features import build_tfidf_vectorizer
from src.models import build_linear_svm


def run_error_analysis_and_tuning() -> Dict[str, Any]:
    print("=" * 70)
    print("LEGAL LENS: MILESTONE 5 ERROR ANALYSIS & TARGETED TUNING")
    print("=" * 70)

    # 1. Load Data & Validation Model
    print(f"[*] Loading training data from: {TRAIN_FILE}")
    df_train = pd.read_parquet(TRAIN_FILE)
    print(f"[*] Loading validation data from: {VAL_FILE}")
    df_val = pd.read_parquet(VAL_FILE)

    champion_path = MODELS_DIR / "linear_svm_balanced.joblib"
    if not champion_path.exists():
        raise FileNotFoundError(f"Champion model not found at {champion_path}. Run train.py first.")

    print(f"[*] Loading validation champion model from: {champion_path}")
    pipeline: Pipeline = joblib.load(champion_path)
    vectorizer = pipeline.named_steps["tfidf"]
    clf: LinearSVC = pipeline.named_steps["clf"]

    # 2. Generate Validation Predictions & Isolate Errors
    print("[*] Generating validation predictions...")
    y_val = df_val[COL_CLEAN_TARGET].tolist()
    y_pred = pipeline.predict(df_val[COL_CLEAN_TEXT].tolist())

    df_eval = df_val.copy()
    df_eval["y_true"] = y_val
    df_eval["y_pred"] = y_pred
    df_eval["is_correct"] = df_eval["y_true"] == df_eval["y_pred"]
    df_eval["char_len"] = df_eval[COL_CLEAN_TEXT].str.len()
    df_eval["word_len"] = df_eval[COL_CLEAN_TEXT].apply(lambda t: len(str(t).split()))
    df_eval["is_truncated"] = df_eval["char_len"] >= 32760

    total_val = len(df_eval)
    correct_count = int(df_eval["is_correct"].sum())
    error_count = total_val - correct_count
    error_rate = error_count / total_val

    # Error Types:
    # False Criminal (True Civil, Pred Criminal)
    fp_mask = (df_eval["y_true"] == TARGET_CIVIL) & (df_eval["y_pred"] == TARGET_CRIMINAL)
    df_fp = df_eval[fp_mask]

    # False Civil (True Criminal, Pred Civil)
    fn_mask = (df_eval["y_true"] == TARGET_CRIMINAL) & (df_eval["y_pred"] == TARGET_CIVIL)
    df_fn = df_eval[fn_mask]

    print(f"[*] Total Validation Samples: {total_val:,}")
    print(f"[*] Correct Predictions: {correct_count:,} ({correct_count/total_val*100:.2f}%)")
    print(f"[*] Total Misclassifications: {error_count:,} ({error_rate*100:.2f}%)")
    print(f"    - True Civil -> Predicted Criminal (False Criminal): {len(df_fp):,} ({len(df_fp)/2687*100:.2f}% of Civil)")
    print(f"    - True Criminal -> Predicted Civil (False Civil): {len(df_fn):,} ({len(df_fn)/685*100:.2f}% of Criminal)")

    # 3. Document Length & Truncation Correlation
    print("[*] Analyzing document length and truncation correlations...")
    len_correct = df_eval[df_eval["is_correct"]]["word_len"]
    len_errors = df_eval[~df_eval["is_correct"]]["word_len"]

    trunc_overall = int(df_eval["is_truncated"].sum())
    trunc_in_errors = int(df_eval[~df_eval["is_correct"]]["is_truncated"].sum())
    trunc_error_rate = trunc_in_errors / trunc_overall if trunc_overall > 0 else 0
    non_trunc_error_rate = (error_count - trunc_in_errors) / (total_val - trunc_overall)

    length_analysis = {
        "correct_cases": {
            "median_words": float(len_correct.median()),
            "mean_words": round(float(len_correct.mean()), 2),
            "p25_words": float(len_correct.quantile(0.25)),
            "p75_words": float(len_correct.quantile(0.75)),
        },
        "error_cases": {
            "median_words": float(len_errors.median()),
            "mean_words": round(float(len_errors.mean()), 2),
            "p25_words": float(len_errors.quantile(0.25)),
            "p75_words": float(len_errors.quantile(0.75)),
        },
        "truncation_analysis": {
            "total_truncated_in_val": trunc_overall,
            "pct_val_truncated": round(trunc_overall / total_val * 100, 2),
            "truncated_errors_count": trunc_in_errors,
            "error_rate_among_truncated": round(trunc_error_rate * 100, 2),
            "error_rate_among_non_truncated": round(non_trunc_error_rate * 100, 2),
            "is_disproportionate": bool(trunc_error_rate > non_trunc_error_rate * 1.25),
        }
    }

    print(f"    Correct Median Words: {length_analysis['correct_cases']['median_words']} | "
          f"Error Median Words: {length_analysis['error_cases']['median_words']}")
    print(f"    Truncated Error Rate: {length_analysis['truncation_analysis']['error_rate_among_truncated']}% vs "
          f"Non-Truncated: {length_analysis['truncation_analysis']['error_rate_among_non_truncated']}%")

    # 4. Top Feature Coefficients
    print("[*] Extracting top feature coefficients from Linear SVM...")
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_[0]  # Positive = Criminal, Negative = Civil

    # Top Criminal Predictors
    top_crim_idx = np.argsort(coef)[-20:][::-1]
    top_crim_features = [(feature_names[i], round(float(coef[i]), 4)) for i in top_crim_idx]

    # Top Civil Predictors
    top_civ_idx = np.argsort(coef)[:20]
    top_civ_features = [(feature_names[i], round(float(coef[i]), 4)) for i in top_civ_idx]

    # 5. Extract Representative Error Case Studies
    print("[*] Extracting representative error case studies across failure modes...")
    representative_errors = []

    # Case 1: True Civil -> Pred Criminal (Contempt of Court / Criminal Contempt)
    contempt_sample = df_fp[df_fp[COL_CLEAN_TEXT].str.contains(r"\bcontempt\b", regex=True)]
    if not contempt_sample.empty:
        r = contempt_sample.iloc[0]
        representative_errors.append({
            "case_id": str(r[COL_ID]),
            "ground_truth": "civil",
            "prediction": "criminal",
            "failure_pattern": "Contempt of Court Jurisdictional Overlap",
            "word_count": int(r["word_len"]),
            "text_preview": str(r[COL_CLEAN_TEXT])[:350] + "...",
            "explanation": (
                "Civil contempt proceedings arising from disobedience of civil orders frequently invoke "
                "criminal terminology ('punishment', 'imprisonment', 'contempt of courts act'), causing the model to misclassify."
            ),
        })

    # Case 2: True Civil -> Pred Criminal (Constitutional Writ with Police Investigation allegations)
    police_writ = df_fp[df_fp[COL_CLEAN_TEXT].str.contains(r"\bpolice\b|\binvestigation\b", regex=True)]
    if not police_writ.empty:
        r = police_writ.iloc[0]
        representative_errors.append({
            "case_id": str(r[COL_ID]),
            "ground_truth": "civil",
            "prediction": "criminal",
            "failure_pattern": "Article 32/226 Writ Involving Criminal Investigations",
            "word_count": int(r["word_len"]),
            "text_preview": str(r[COL_CLEAN_TEXT])[:350] + "...",
            "explanation": (
                "A civil writ petition litigating police inaction or seeking CBI investigation contains heavy penal vocabulary "
                "('police', 'investigation', 'offence'), overriding civil procedural markers."
            ),
        })

    # Case 3: True Criminal -> Pred Civil (Prevention of Corruption / Statutory Service Rules)
    corruption_sample = df_fn[df_fn[COL_CLEAN_TEXT].str.contains(r"\bprevention of corruption\b|\bgovernment\b", regex=True)]
    if not corruption_sample.empty:
        r = corruption_sample.iloc[0]
        representative_errors.append({
            "case_id": str(r[COL_ID]),
            "ground_truth": "criminal",
            "prediction": "civil",
            "failure_pattern": "Anti-Corruption & Public Servant Disciplinary Overlap",
            "word_count": int(r["word_len"]),
            "text_preview": str(r[COL_CLEAN_TEXT])[:350] + "...",
            "explanation": (
                "Criminal corruption trials against civil servants discuss departmental inquiries, service rules, and administrative orders, "
                "which heavily triggers civil revenue/government features."
            ),
        })

    # Case 4: True Criminal -> Pred Civil (Company / Economic Offenses / Cheque Bounce)
    cheque_sample = df_fn[df_fn[COL_CLEAN_TEXT].str.contains(r"\bcheque\b|\bcompany\b|\bcommercial\b", regex=True)]
    if not cheque_sample.empty:
        r = cheque_sample.iloc[0]
        representative_errors.append({
            "case_id": str(r[COL_ID]),
            "ground_truth": "criminal",
            "prediction": "civil",
            "failure_pattern": "Quasi-Criminal Commercial & Economic Offenses",
            "word_count": int(r["word_len"]),
            "text_preview": str(r[COL_CLEAN_TEXT])[:350] + "...",
            "explanation": (
                "Section 138 Negotiable Instruments Act or corporate financial crimes discuss banking transactions, commercial agreements, and debts, "
                "biasing the classifier toward Civil."
            ),
        })

    # 6. Targeted Hyperparameter Tuning (Validation Only)
    print("\n" + "=" * 70)
    print("[*] Running targeted validation-only tuning experiments...")
    X_train_raw = df_train[COL_CLEAN_TEXT].tolist()
    y_train = df_train[COL_CLEAN_TARGET].tolist()
    X_val_raw = df_val[COL_CLEAN_TEXT].tolist()

    tuning_results = []

    # Baseline configuration reference
    baseline_metrics = compute_metrics(y_val, y_pred, model_name="Current Champion: Linear SVM (Balanced, C=1.0, 30k 1-2g)")
    tuning_results.append({
        "experiment": "Baseline (C=1.0, Balanced, 30k features, ngram (1,2))",
        "macro_f1": baseline_metrics["macro_f1"],
        "criminal_recall": baseline_metrics["criminal_metrics"]["recall"],
        "criminal_precision": baseline_metrics["criminal_metrics"]["precision"],
        "criminal_f1": baseline_metrics["criminal_metrics"]["f1"],
        "civil_recall": baseline_metrics["civil_metrics"]["recall"],
        "civil_f1": baseline_metrics["civil_metrics"]["f1"],
        "accuracy": baseline_metrics["accuracy"],
    })

    # Tuning Experiment A: Regularization parameter C in Linear SVM (C=0.2, C=0.5, C=2.0)
    # Using existing fitted TF-IDF matrix for speed
    X_train_tfidf = vectorizer.transform(X_train_raw)
    X_val_tfidf = vectorizer.transform(X_val_raw)

    for c_val in [0.2, 0.5, 2.0]:
        print(f"    - Testing Linear SVM (Balanced, C={c_val})...")
        svm_c = LinearSVC(C=c_val, class_weight="balanced", dual="auto", random_state=RANDOM_SEED, max_iter=2000)
        svm_c.fit(X_train_tfidf, y_train)
        pred_c = svm_c.predict(X_val_tfidf)
        m = compute_metrics(y_val, pred_c, model_name=f"Linear SVM (C={c_val}, Balanced)")
        tuning_results.append({
            "experiment": f"Tuning C: Linear SVM (C={c_val}, Balanced, 30k features, ngram (1,2))",
            "macro_f1": m["macro_f1"],
            "criminal_recall": m["criminal_metrics"]["recall"],
            "criminal_precision": m["criminal_metrics"]["precision"],
            "criminal_f1": m["criminal_metrics"]["f1"],
            "civil_recall": m["civil_metrics"]["recall"],
            "civil_f1": m["civil_metrics"]["f1"],
            "accuracy": m["accuracy"],
        })

    # Tuning Experiment B: Unigrams only vs Unigram+Bigram
    print("    - Testing TF-IDF Unigrams-only (1,1)...")
    vec_unigram = build_tfidf_vectorizer(ngram_range=(1, 1), max_features=30000)
    X_train_uni = vec_unigram.fit_transform(X_train_raw)
    X_val_uni = vec_unigram.transform(X_val_raw)
    svm_uni = build_linear_svm(class_weight="balanced", C=1.0)
    svm_uni.fit(X_train_uni, y_train)
    pred_uni = svm_uni.predict(X_val_uni)
    m_uni = compute_metrics(y_val, pred_uni, model_name="Linear SVM (Unigrams only (1,1), C=1.0)")
    tuning_results.append({
        "experiment": "Feature Tuning: Unigrams only (1,1), C=1.0, 30k features",
        "macro_f1": m_uni["macro_f1"],
        "criminal_recall": m_uni["criminal_metrics"]["recall"],
        "criminal_precision": m_uni["criminal_metrics"]["precision"],
        "criminal_f1": m_uni["criminal_metrics"]["f1"],
        "civil_recall": m_uni["civil_metrics"]["recall"],
        "civil_f1": m_uni["civil_metrics"]["f1"],
        "accuracy": m_uni["accuracy"],
    })

    # Tuning Experiment C: Expanded vocabulary capacity (max_features=40,000)
    print("    - Testing TF-IDF with 40,000 features...")
    vec_40k = build_tfidf_vectorizer(ngram_range=(1, 2), max_features=40000)
    X_train_40k = vec_40k.fit_transform(X_train_raw)
    X_val_40k = vec_40k.transform(X_val_raw)
    svm_40k = build_linear_svm(class_weight="balanced", C=1.0)
    svm_40k.fit(X_train_40k, y_train)
    pred_40k = svm_40k.predict(X_val_40k)
    m_40k = compute_metrics(y_val, pred_40k, model_name="Linear SVM (40k features, ngram (1,2), C=1.0)")
    tuning_results.append({
        "experiment": "Feature Tuning: 40,000 features, ngram (1,2), C=1.0",
        "macro_f1": m_40k["macro_f1"],
        "criminal_recall": m_40k["criminal_metrics"]["recall"],
        "criminal_precision": m_40k["criminal_metrics"]["precision"],
        "criminal_f1": m_40k["criminal_metrics"]["f1"],
        "civil_recall": m_40k["civil_metrics"]["recall"],
        "civil_f1": m_40k["civil_metrics"]["f1"],
        "accuracy": m_40k["accuracy"],
    })

    # Sort tuning results by Macro-F1
    tuning_results.sort(key=lambda x: x["macro_f1"], reverse=True)

    # Save Metrics first
    summary_payload = {
        "error_overview": {
            "total_validation_samples": total_val,
            "correct_predictions": correct_count,
            "total_errors": error_count,
            "error_rate_pct": round(error_rate * 100, 2),
            "false_criminal_count": len(df_fp),
            "false_criminal_pct_of_civil": round(len(df_fp) / 2687 * 100, 2),
            "false_civil_count": len(df_fn),
            "false_civil_pct_of_crim": round(len(df_fn) / 685 * 100, 2),
        },
        "document_length_and_truncation": length_analysis,
        "top_feature_coefficients": {
            "top_criminal_features": top_crim_features[:15],
            "top_civil_features": top_civ_features[:15],
        },
        "representative_error_case_studies": representative_errors,
        "tuning_benchmark_results": tuning_results,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "milestone_5_tuning_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[*] Saved Milestone 5 tuning & error metrics to: {metrics_path}")

    # Generate Visualizations
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_error_visualizations(df_eval, top_crim_features, top_civ_features)

    print("=" * 70)
    print("MILESTONE 5 ERROR ANALYSIS & TUNING COMPLETE")
    print("=" * 70)

    return summary_payload


def generate_error_visualizations(
    df_eval: pd.DataFrame,
    top_crim_features: List[Tuple[str, float]],
    top_civ_features: List[Tuple[str, float]],
) -> None:
    # Figure 1: Length Distribution Correct vs Error
    plt.figure(figsize=(11, 4.5))

    df_eval_plot = df_eval.copy()
    df_eval_plot["Outcome"] = df_eval_plot["is_correct"].map({True: "Correct", False: "Misclassified"})

    plt.subplot(1, 2, 1)
    sns.boxplot(
        data=df_eval_plot,
        x="Outcome",
        y="word_len",
        palette={"Correct": "#1b9e77", "Misclassified": "#d95f02"},
        showfliers=False,
    )
    plt.title("Document Word Count: Correct vs. Error", fontsize=11, fontweight="bold")
    plt.ylabel("Word Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    trunc_err_pct = (df_eval.groupby("is_truncated")["is_correct"].apply(lambda s: (~s).mean()) * 100)
    bars = plt.bar(["Non-Truncated", "Truncated (32,765 chars)"], trunc_err_pct.values, color=["#2b5c8f", "#e7298a"], edgecolor="black", alpha=0.85)
    plt.title("Error Rate: Truncated vs. Non-Truncated", fontsize=11, fontweight="bold")
    plt.ylabel("Error Rate (%)")
    plt.ylim(0, max(trunc_err_pct.values) * 1.3)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "error_length_and_truncation_distribution.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"[*] Saved figure: {fig1_path}")

    # Figure 2: Top SVM Feature Coefficients
    plt.figure(figsize=(13, 6))

    plt.subplot(1, 2, 1)
    civ_words = [f[0] for f in top_civ_features[:12]]
    civ_weights = [abs(f[1]) for f in top_civ_features[:12]]
    plt.barh(civ_words[::-1], civ_weights[::-1], color="#2b5c8f", edgecolor="black", alpha=0.85)
    plt.title("Top Discriminative Coefficients: Civil", fontsize=11, fontweight="bold")
    plt.xlabel("Absolute SVM Weight")

    plt.subplot(1, 2, 2)
    crim_words = [f[0] for f in top_crim_features[:12]]
    crim_weights = [f[1] for f in top_crim_features[:12]]
    plt.barh(crim_words[::-1], crim_weights[::-1], color="#d95f02", edgecolor="black", alpha=0.85)
    plt.title("Top Discriminative Coefficients: Criminal", fontsize=11, fontweight="bold")
    plt.xlabel("SVM Weight")

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "top_model_coefficients.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"[*] Saved figure: {fig2_path}")


if __name__ == "__main__":
    run_error_analysis_and_tuning()
