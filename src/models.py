"""
LegalLens Model Definitions Module (Milestone 4)
Defines classical baseline and linear classification architectures:
1. Majority Class Baseline (DummyClassifier)
2. Logistic Regression (L2 regularization, balanced & default class weighting)
3. Linear Support Vector Machine (LinearSVC, balanced & default class weighting)
"""

from typing import Optional, Union
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from src.config import RANDOM_SEED


def build_majority_baseline() -> DummyClassifier:
    """Constructs a majority class dummy baseline classifier."""
    return DummyClassifier(strategy="most_frequent")


def build_logistic_regression(
    class_weight: Optional[Union[str, dict]] = "balanced",
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = RANDOM_SEED,
) -> LogisticRegression:
    """Constructs an L2-regularized Logistic Regression classifier."""
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
        n_jobs=-1,
    )


def build_linear_svm(
    class_weight: Optional[Union[str, dict]] = "balanced",
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = RANDOM_SEED,
) -> LinearSVC:
    """Constructs a Linear Support Vector Machine (LinearSVC)."""
    return LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        dual="auto",
    )
