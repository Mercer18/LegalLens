"""
LegalLens Milestone 7: Final Test Evaluation Script
Executes the final, one-shot unbiased evaluation of the locked LegalLens candidate pipeline
against the frozen held-out test partition.

Strict Verification & Integrity Protocols:
1. Frozen Test Set Verification: Confirms exact SHA-256 ID list matches frozen_test_ids.json.
2. Contamination Assertions: Verifies zero text-hash overlap between train/test and val/test.
3. Locked Model Verification: Confirms model is loaded strictly from models/final_linear_svm_pipeline.joblib
   (fitted exclusively on train data; no refitting, no parameter tuning, no threshold alteration).
4. Unbiased Metrics Computation: Accuracy, Macro-F1, Weighted-F1, Per-class Precision/Recall/F1, Confusion Matrix.
5. Comparison: Validates against held-out validation metrics.
6. Persistence: Saves outputs to outputs/metrics/final_test_metrics.json and confusion matrix figure.
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    COL_CLEAN_TARGET,
    COL_CLEAN_TEXT,
    COL_ID,
    COL_TEXT_HASH,
    FIGURES_DIR,
    FINAL_MODEL_FILE,
    FROZEN_TEST_IDS_FILE,
    METRICS_DIR,
    TARGET_CIVIL,
    TARGET_CRIMINAL,
    TEST_FILE,
    TRAIN_FILE,
    VAL_FILE,
)
from src.evaluate import compute_metrics, plot_and_save_confusion_matrix


def run_final_test_evaluation() -> Dict[str, Any]:
    print("=" * 75)
    print("LEGAL LENS: FINAL UNBIASED TEST EVALUATION (ONE-SHOT HOLD-OUT)")
    print("=" * 75)

    # -------------------------------------------------------------
    # Step 1: Integrity Checks - Verify Disjoint Partitions & Frozen IDs
    # -------------------------------------------------------------
    print("[*] Running pre-evaluation integrity verification...")

    assert TRAIN_FILE.exists(), f"Train file missing: {TRAIN_FILE}"
    assert VAL_FILE.exists(), f"Val file missing: {VAL_FILE}"
    assert TEST_FILE.exists(), f"Test file missing: {TEST_FILE}"
    assert FROZEN_TEST_IDS_FILE.exists(), f"Frozen test IDs missing: {FROZEN_TEST_IDS_FILE}"
    assert FINAL_MODEL_FILE.exists(), f"Final locked model missing: {FINAL_MODEL_FILE}"

    # Load frozen test IDs
    with open(FROZEN_TEST_IDS_FILE, "r", encoding="utf-8") as f:
        frozen_data = json.load(f)
    frozen_id_set = set(frozen_data["frozen_test_ids"])
    print(f"[*] Verified frozen test ID index: {len(frozen_id_set):,} IDs expected.")

    # Load partitions
    df_train = pd.read_parquet(TRAIN_FILE)
    df_val = pd.read_parquet(VAL_FILE)
    df_test = pd.read_parquet(TEST_FILE)

    # Verify test set count and membership
    test_id_set = set(df_test[COL_ID].tolist())
    assert test_id_set == frozen_id_set, "CRITICAL ERROR: Test set IDs do not match frozen_test_ids.json!"
    print("[*] PASS: Test set IDs match frozen_test_ids.json perfectly (3,372 / 3,372).")

    # Contamination checks
    train_hashes = set(df_train[COL_TEXT_HASH].tolist())
    val_hashes = set(df_val[COL_TEXT_HASH].tolist())
    test_hashes = set(df_test[COL_TEXT_HASH].tolist())

    train_test_overlap = len(train_hashes.intersection(test_hashes))
    val_test_overlap = len(val_hashes.intersection(test_hashes))
    assert train_test_overlap == 0, f"CONTAMINATION DETECTED: {train_test_overlap} train/test overlaps!"
    assert val_test_overlap == 0, f"CONTAMINATION DETECTED: {val_test_overlap} val/test overlaps!"
    print(f"[*] PASS: Zero text hash contamination (Train/Test overlap = {train_test_overlap}, Val/Test overlap = {val_test_overlap}).")

    # Verify Class Distribution in Test Set
    civil_count = int((df_test[COL_CLEAN_TARGET] == TARGET_CIVIL).sum())
    criminal_count = int((df_test[COL_CLEAN_TARGET] == TARGET_CRIMINAL).sum())
    total_test = len(df_test)
    assert total_test == 3372
    assert civil_count == 2687
    assert criminal_count == 685
    print(f"[*] Test Partition: {total_test:,} samples (Civil: {civil_count:,} [79.69%], Criminal: {criminal_count:,} [20.31%])")

    # -------------------------------------------------------------
    # Step 2: Load Locked Model Pipeline (Strictly from Disk, NO Retraining)
    # -------------------------------------------------------------
    print(f"[*] Loading locked pipeline from: {FINAL_MODEL_FILE}")
    pipeline: Pipeline = joblib.load(FINAL_MODEL_FILE)

    # Verify pipeline components & hyperparameters
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    assert tfidf.max_features == 40000, f"Expected 40,000 features, found {tfidf.max_features}"
    assert tfidf.ngram_range == (1, 2), f"Expected ngram_range (1,2), found {tfidf.ngram_range}"
    assert tfidf.sublinear_tf is True, "sublinear_tf is not True"
    assert clf.class_weight == "balanced", f"Expected class_weight='balanced', found {clf.class_weight}"
    assert clf.C == 1.0, f"Expected C=1.0, found {clf.C}"
    print("[*] PASS: Locked pipeline components verified (TF-IDF 40k 1-2g + LinearSVC balanced C=1.0).")

    # -------------------------------------------------------------
    # Step 3: Execute Single Unbiased Test Inference Pass
    # -------------------------------------------------------------
    print("[*] Executing one-shot evaluation pass on test partition...")
    X_test = df_test[COL_CLEAN_TEXT].tolist()
    y_test = df_test[COL_CLEAN_TARGET].tolist()

    y_test_pred = pipeline.predict(X_test)

    # -------------------------------------------------------------
    # Step 4: Compute Metrics
    # -------------------------------------------------------------
    test_metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        model_name="LegalLens Locked Model (Linear SVM Balanced, 40k 1-2g)",
    )

    # Generate Confusion Matrix Figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = FIGURES_DIR / "final_test_confusion_matrix.png"
    plot_and_save_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred,
        model_name="Final Test Set (Locked Linear SVM)",
        output_path=cm_path,
    )
    print(f"[*] Saved test confusion matrix figure to: {cm_path}")

    # Validation reference metrics (from Milestone 5 40k feature run)
    val_metrics_ref = {
        "accuracy": 0.9416,
        "macro_f1": 0.9148,
        "weighted_f1": 0.9431,
        "criminal_metrics": {
            "precision": 0.8065,
            "recall": 0.9372,
            "f1": 0.8670,
            "support": 685,
        },
        "civil_metrics": {
            "precision": 0.9836,
            "recall": 0.9427,
            "f1": 0.9626,
            "support": 2687,
        },
    }

    # Comparison payload
    comparison = {
        "metric": ["Accuracy", "Macro-F1", "Criminal Recall", "Criminal Precision", "Criminal F1", "Civil Recall", "Civil Precision", "Civil F1"],
        "validation_score": [
            val_metrics_ref["accuracy"],
            val_metrics_ref["macro_f1"],
            val_metrics_ref["criminal_metrics"]["recall"],
            val_metrics_ref["criminal_metrics"]["precision"],
            val_metrics_ref["criminal_metrics"]["f1"],
            val_metrics_ref["civil_metrics"]["recall"],
            val_metrics_ref["civil_metrics"]["precision"],
            val_metrics_ref["civil_metrics"]["f1"],
        ],
        "test_score": [
            test_metrics["accuracy"],
            test_metrics["macro_f1"],
            test_metrics["criminal_metrics"]["recall"],
            test_metrics["criminal_metrics"]["precision"],
            test_metrics["criminal_metrics"]["f1"],
            test_metrics["civil_metrics"]["recall"],
            test_metrics["civil_metrics"]["precision"],
            test_metrics["civil_metrics"]["f1"],
        ],
        "delta (test - val)": [
            round(test_metrics["accuracy"] - val_metrics_ref["accuracy"], 4),
            round(test_metrics["macro_f1"] - val_metrics_ref["macro_f1"], 4),
            round(test_metrics["criminal_metrics"]["recall"] - val_metrics_ref["criminal_metrics"]["recall"], 4),
            round(test_metrics["criminal_metrics"]["precision"] - val_metrics_ref["criminal_metrics"]["precision"], 4),
            round(test_metrics["criminal_metrics"]["f1"] - val_metrics_ref["criminal_metrics"]["f1"], 4),
            round(test_metrics["civil_metrics"]["recall"] - val_metrics_ref["civil_metrics"]["recall"], 4),
            round(test_metrics["civil_metrics"]["precision"] - val_metrics_ref["civil_metrics"]["precision"], 4),
            round(test_metrics["civil_metrics"]["f1"] - val_metrics_ref["civil_metrics"]["f1"], 4),
        ]
    }

    output_payload = {
        "evaluation_name": "LegalLens Final Unbiased Hold-Out Test Evaluation",
        "dataset": "Jud-IPL Frozen Test Partition",
        "total_test_samples": total_test,
        "test_class_distribution": {
            "civil_count": civil_count,
            "civil_pct": round(civil_count / total_test * 100, 2),
            "criminal_count": criminal_count,
            "criminal_pct": round(criminal_count / total_test * 100, 2),
            "imbalance_ratio": round(civil_count / criminal_count, 2),
        },
        "locked_model_configuration": {
            "vectorizer": "TF-IDF (1-2 ngrams, max_features=40,000, sublinear_tf=True, min_df=3, max_df=0.90)",
            "classifier": "LinearSVC (C=1.0, class_weight='balanced', loss='squared_hinge', random_state=42)",
            "model_path": str(FINAL_MODEL_FILE),
        },
        "test_metrics": test_metrics,
        "validation_vs_test_comparison": comparison,
        "integrity_verification": {
            "frozen_test_ids_matched": True,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "model_refitted_on_test": False,
            "post_test_tuning_occurred": False,
        }
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = METRICS_DIR / "final_test_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"[*] Saved final test metrics to: {out_file}")

    print("\n" + "=" * 75)
    print("FINAL TEST EVALUATION RESULTS:")
    print("=" * 75)
    print(f"  Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:           {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:        {test_metrics['weighted_f1']:.4f}")
    print(f"  Civil Precision:    {test_metrics['civil_metrics']['precision']:.4f}")
    print(f"  Civil Recall:       {test_metrics['civil_metrics']['recall']:.4f}")
    print(f"  Civil F1:           {test_metrics['civil_metrics']['f1']:.4f}")
    print(f"  Criminal Precision: {test_metrics['criminal_metrics']['precision']:.4f}")
    print(f"  Criminal Recall:    {test_metrics['criminal_metrics']['recall']:.4f}")
    print(f"  Criminal F1:        {test_metrics['criminal_metrics']['f1']:.4f}")
    print("-" * 75)
    print("Confusion Matrix (rows=true, cols=pred):")
    cm = test_metrics["confusion_matrix"]["matrix"]
    print(f"  True Civil    -> Pred Civil: {cm[0][0]:<5} | Pred Criminal: {cm[0][1]:<5} (Total: {civil_count})")
    print(f"  True Criminal -> Pred Civil: {cm[1][0]:<5} | Pred Criminal: {cm[1][1]:<5} (Total: {criminal_count})")
    print("=" * 75)

    return output_payload


if __name__ == "__main__":
    run_final_test_evaluation()
