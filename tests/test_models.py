"""
Unit tests for Milestone 4 model artifacts and inference sanity.
"""

from pathlib import Path
import joblib
import pytest
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR


def test_saved_model_pipelines_exist():
    lr_path = MODELS_DIR / "logistic_regression_balanced.joblib"
    svm_path = MODELS_DIR / "linear_svm_balanced.joblib"
    assert lr_path.exists(), f"Missing LR model pipeline: {lr_path}"
    assert svm_path.exists(), f"Missing SVM model pipeline: {svm_path}"


def test_saved_model_inference_sanity():
    svm_path = MODELS_DIR / "linear_svm_balanced.joblib"
    pipeline = joblib.load(svm_path)
    assert isinstance(pipeline, Pipeline)

    sample_criminal = "criminal appeal accused convicted under section 302 indian penal code police station sessions judge"
    sample_civil = "civil writ petition income tax assessment sales tax revenue land acquisition act"

    pred_crim = pipeline.predict([sample_criminal])[0]
    pred_civ = pipeline.predict([sample_civil])[0]

    assert pred_crim == "criminal", f"Expected criminal, got {pred_crim}"
    assert pred_civ == "civil", f"Expected civil, got {pred_civ}"
