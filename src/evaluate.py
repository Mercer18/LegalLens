"""
LegalLens Evaluation Module (Milestone 4)
Computes rigorous, leak-free classification metrics:
Macro-F1, Per-class Precision/Recall/F1, Accuracy, Weighted-F1, and Confusion Matrices.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.config import FIGURES_DIR, TARGET_CIVIL, TARGET_CRIMINAL


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics on validation/test predictions.
    Focuses on Macro-F1 and Criminal Recall as headline performance indicators.
    """
    labels = [TARGET_CIVIL, TARGET_CRIMINAL]

    # Headline Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-Class Precision, Recall, F1
    p_class, r_class, f_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Confusion Matrix: [[TN, FP], [FN, TP]] -> rows are true, cols are pred
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    clf_report = classification_report(
        y_true, y_pred, labels=labels, target_names=["Civil", "Criminal"], output_dict=True, zero_division=0
    )
    clf_report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=["Civil", "Criminal"], zero_division=0
    )

    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "civil_metrics": {
            "precision": round(float(p_class[0]), 4),
            "recall": round(float(r_class[0]), 4),
            "f1": round(float(f_class[0]), 4),
            "support": int(support[0]),
        },
        "criminal_metrics": {
            "precision": round(float(p_class[1]), 4),
            "recall": round(float(r_class[1]), 4),
            "f1": round(float(f_class[1]), 4),
            "support": int(support[1]),
        },
        "confusion_matrix": {
            "labels": ["Civil", "Criminal"],
            "matrix": cm.tolist(),
            "true_civil_pred_civil": int(cm[0][0]),
            "true_civil_pred_crim": int(cm[0][1]),
            "true_crim_pred_civil": int(cm[1][0]),
            "true_crim_pred_crim": int(cm[1][1]),
        },
        "classification_report_dict": clf_report,
        "classification_report_str": clf_report_text,
    }


def plot_and_save_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    model_name: str,
    output_path: Optional[Path] = None,
) -> Path:
    """Renders and saves a clean confusion matrix heatmap."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('+', '_')}.png"
        output_path = FIGURES_DIR / filename

    labels = [TARGET_CIVIL, TARGET_CRIMINAL]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=["Pred Civil", "Pred Criminal"],
        yticklabels=["True Civil", "True Criminal"],
        cbar=False,
        annot_kws={"size": 12, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path
