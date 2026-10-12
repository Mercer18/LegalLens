"""
LegalLens Training & Benchmark Orchestration Module (Milestone 4)
Orchestrates Phase 1 through Phase 4:
- Majority Baseline
- TF-IDF + Logistic Regression (Default vs Balanced)
- TF-IDF + Linear SVM (Default vs Balanced)
- Comprehensive Validation Metric Evaluation & Persistence
"""

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    COL_CLEAN_TARGET,
    COL_CLEAN_TEXT,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    TRAIN_FILE,
    VAL_FILE,
)
from src.evaluate import compute_metrics, plot_and_save_confusion_matrix
from src.features import build_tfidf_vectorizer
from src.models import (
    build_linear_svm,
    build_logistic_regression,
    build_majority_baseline,
)


def run_milestone_4_benchmarks() -> Dict[str, Any]:
    """
    Executes the complete Milestone 4 modeling benchmark on the validation set.
    """
    print("=" * 70)
    print("LEGAL LENS: MILESTONE 4 CLASSICAL MODEL BENCHMARKS")
    print("=" * 70)

    # 1. Load Data
    print(f"[*] Loading training partition from: {TRAIN_FILE}")
    df_train = pd.read_parquet(TRAIN_FILE)
    print(f"[*] Loading validation partition from: {VAL_FILE}")
    df_val = pd.read_parquet(VAL_FILE)

    X_train, y_train = df_train[COL_CLEAN_TEXT].tolist(), df_train[COL_CLEAN_TARGET].tolist()
    X_val, y_val = df_val[COL_CLEAN_TEXT].tolist(), df_val[COL_CLEAN_TARGET].tolist()

    print(f"[*] Training samples: {len(X_train):,} | Validation samples: {len(X_val):,}")

    # 2. Fit TF-IDF Vectorizer Strictly on Training Set
    print("[*] Fitting TF-IDF Vectorizer (unigrams + bigrams, max_features=30,000, sublinear_tf=True)...")
    start_tfidf = time.time()
    vectorizer = build_tfidf_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    tfidf_time = time.time() - start_tfidf
    print(f"[*] TF-IDF fitting complete in {tfidf_time:.2f}s! Matrix shapes: Train {X_train_tfidf.shape}, Val {X_val_tfidf.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Model Experiments Container
    experiments = {}
    comparison_table = []

    # -------------------------------------------------------------
    # Experiment 1: Majority Class Baseline
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[1/5] Training Majority Baseline Classifier...")
    start_t = time.time()
    dummy = build_majority_baseline()
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_val)
    dummy_time = time.time() - start_t

    dummy_metrics = compute_metrics(y_val, dummy_pred, model_name="Majority Baseline")
    dummy_metrics["train_eval_time_sec"] = round(dummy_time, 2)
    plot_and_save_confusion_matrix(y_val, dummy_pred, "Majority Baseline")
    experiments["majority_baseline"] = dummy_metrics

    print(f"    Accuracy: {dummy_metrics['accuracy']:.4f} | Macro-F1: {dummy_metrics['macro_f1']:.4f} | "
          f"Crim Recall: {dummy_metrics['criminal_metrics']['recall']:.4f}")

    # -------------------------------------------------------------
    # Experiment 2: TF-IDF + Logistic Regression (Default Weights)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[2/5] Training TF-IDF + Logistic Regression (Default Weights)...")
    start_t = time.time()
    lr_default = build_logistic_regression(class_weight=None)
    lr_default.fit(X_train_tfidf, y_train)
    lr_def_pred = lr_default.predict(X_val_tfidf)
    lr_def_time = time.time() - start_t

    lr_def_metrics = compute_metrics(y_val, lr_def_pred, model_name="Logistic Regression (Default)")
    lr_def_metrics["train_eval_time_sec"] = round(lr_def_time, 2)
    plot_and_save_confusion_matrix(y_val, lr_def_pred, "Logistic Regression Default")
    experiments["lr_default"] = lr_def_metrics

    print(f"    Accuracy: {lr_def_metrics['accuracy']:.4f} | Macro-F1: {lr_def_metrics['macro_f1']:.4f} | "
          f"Crim Recall: {lr_def_metrics['criminal_metrics']['recall']:.4f}")

    # -------------------------------------------------------------
    # Experiment 3: TF-IDF + Logistic Regression (Balanced Weights)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[3/5] Training TF-IDF + Logistic Regression (Balanced Weights)...")
    start_t = time.time()
    lr_balanced = build_logistic_regression(class_weight="balanced")
    lr_balanced.fit(X_train_tfidf, y_train)
    lr_bal_pred = lr_balanced.predict(X_val_tfidf)
    lr_bal_time = time.time() - start_t

    lr_bal_metrics = compute_metrics(y_val, lr_bal_pred, model_name="Logistic Regression (Balanced)")
    lr_bal_metrics["train_eval_time_sec"] = round(lr_bal_time, 2)
    plot_and_save_confusion_matrix(y_val, lr_bal_pred, "Logistic Regression Balanced")
    experiments["lr_balanced"] = lr_bal_metrics

    # Save Pipeline Artifact
    lr_pipeline = Pipeline([("tfidf", vectorizer), ("clf", lr_balanced)])
    joblib.dump(lr_pipeline, MODELS_DIR / "logistic_regression_balanced.joblib")

    print(f"    Accuracy: {lr_bal_metrics['accuracy']:.4f} | Macro-F1: {lr_bal_metrics['macro_f1']:.4f} | "
          f"Crim Recall: {lr_bal_metrics['criminal_metrics']['recall']:.4f}")

    # -------------------------------------------------------------
    # Experiment 4: TF-IDF + Linear SVM (Default Weights)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[4/5] Training TF-IDF + Linear SVM (Default Weights)...")
    start_t = time.time()
    svm_default = build_linear_svm(class_weight=None)
    svm_default.fit(X_train_tfidf, y_train)
    svm_def_pred = svm_default.predict(X_val_tfidf)
    svm_def_time = time.time() - start_t

    svm_def_metrics = compute_metrics(y_val, svm_def_pred, model_name="Linear SVM (Default)")
    svm_def_metrics["train_eval_time_sec"] = round(svm_def_time, 2)
    plot_and_save_confusion_matrix(y_val, svm_def_pred, "Linear SVM Default")
    experiments["svm_default"] = svm_def_metrics

    print(f"    Accuracy: {svm_def_metrics['accuracy']:.4f} | Macro-F1: {svm_def_metrics['macro_f1']:.4f} | "
          f"Crim Recall: {svm_def_metrics['criminal_metrics']['recall']:.4f}")

    # -------------------------------------------------------------
    # Experiment 5: TF-IDF + Linear SVM (Balanced Weights)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[5/5] Training TF-IDF + Linear SVM (Balanced Weights)...")
    start_t = time.time()
    svm_balanced = build_linear_svm(class_weight="balanced")
    svm_balanced.fit(X_train_tfidf, y_train)
    svm_bal_pred = svm_balanced.predict(X_val_tfidf)
    svm_bal_time = time.time() - start_t

    svm_bal_metrics = compute_metrics(y_val, svm_bal_pred, model_name="Linear SVM (Balanced)")
    svm_bal_metrics["train_eval_time_sec"] = round(svm_bal_time, 2)
    plot_and_save_confusion_matrix(y_val, svm_bal_pred, "Linear SVM Balanced")
    experiments["svm_balanced"] = svm_bal_metrics

    # Save Pipeline Artifact
    svm_pipeline = Pipeline([("tfidf", vectorizer), ("clf", svm_balanced)])
    joblib.dump(svm_pipeline, MODELS_DIR / "linear_svm_balanced.joblib")

    print(f"    Accuracy: {svm_bal_metrics['accuracy']:.4f} | Macro-F1: {svm_bal_metrics['macro_f1']:.4f} | "
          f"Crim Recall: {svm_bal_metrics['criminal_metrics']['recall']:.4f}")

    # -------------------------------------------------------------
    # Comparative Results Summary
    # -------------------------------------------------------------
    for m_key, m_val in experiments.items():
        comparison_table.append({
            "model": m_val["model_name"],
            "macro_f1": m_val["macro_f1"],
            "criminal_recall": m_val["criminal_metrics"]["recall"],
            "criminal_precision": m_val["criminal_metrics"]["precision"],
            "criminal_f1": m_val["criminal_metrics"]["f1"],
            "civil_recall": m_val["civil_metrics"]["recall"],
            "civil_f1": m_val["civil_metrics"]["f1"],
            "accuracy": m_val["accuracy"],
            "weighted_f1": m_val["weighted_f1"],
        })

    # Sort by Macro-F1 descending
    comparison_table.sort(key=lambda x: x["macro_f1"], reverse=True)

    summary_payload = {
        "benchmark_dataset": "Jud-IPL (Validation Set)",
        "train_size": len(X_train),
        "val_size": len(X_val),
        "tfidf_configuration": {
            "ngram_range": [1, 2],
            "max_features": 30000,
            "min_df": 3,
            "max_df": 0.90,
            "sublinear_tf": True,
            "stop_words": "custom_legal (preserved negations)",
        },
        "comparison_table": comparison_table,
        "experiments": experiments,
    }

    metrics_file = METRICS_DIR / "milestone_4_model_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\n[*] Full benchmark metrics saved to: {metrics_file}")

    print("=" * 70)
    print("MILESTONE 4 BENCHMARKS COMPLETE")
    print("=" * 70)

    return summary_payload


if __name__ == "__main__":
    run_milestone_4_benchmarks()
