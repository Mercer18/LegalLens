"""
Unit tests for LegalLens Streamlit Application (Milestone 7)
Verifies:
1. Streamlit app module can be cleanly imported without syntax errors or broken dependencies.
2. Production model pipeline can be successfully loaded and executed within the Streamlit context.
3. Model caching and execution helper produces valid schema matching the UI expectations.
4. Deployment readiness: all necessary packages from requirements.txt are importable.
"""

from pathlib import Path
import pytest
from app.streamlit_app import get_cached_pipeline, ROOT_DIR
from src.predict import predict_judgment
from src.config import FINAL_MODEL_FILE


def test_app_root_dir_resolution():
    """Verifies that ROOT_DIR in streamlit_app.py correctly resolves to the project root."""
    assert ROOT_DIR.exists()
    assert (ROOT_DIR / "src").exists()
    assert (ROOT_DIR / "app").exists()
    assert (ROOT_DIR / "models").exists()


def test_app_pipeline_caching():
    """Verifies that get_cached_pipeline loads the locked production artifact."""
    assert FINAL_MODEL_FILE.exists(), f"Production model missing: {FINAL_MODEL_FILE}"
    pipe = get_cached_pipeline()
    assert pipe is not None
    assert "tfidf" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_app_sample_predictions_compatibility():
    """Verifies that sample text predictions produce expected UI fields."""
    pipe = get_cached_pipeline()
    sample_text = (
        "This civil regular second appeal arises out of a suit for specific performance of an agreement to sell "
        "immovable ancestral property. The trial court decreed the suit with costs."
    )
    result = predict_judgment(sample_text, pipeline=pipe)
    
    assert result["is_valid"] is True
    assert result["predicted_class"] == "Civil"
    assert "decision_score" in result
    assert "confidence_prob" in result
    assert "word_count" in result
    assert "char_count" in result
    assert "probability_civil" in result
    assert "probability_criminal" in result
    assert isinstance(result["decision_score"], float)
    assert 0.0 <= result["confidence_prob"] <= 1.0


def test_streamlit_deployment_dependencies_importable():
    """Verifies that all core deployment dependencies required by requirements.txt can be imported."""
    import streamlit
    import sklearn
    import joblib
    import numpy
    import pandas
    import scipy

    assert streamlit.__version__ is not None
    assert sklearn.__version__ is not None
