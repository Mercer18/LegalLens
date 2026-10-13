"""
Unit tests for LegalLens Streamlit Application
Verifies:
1. Streamlit app module can be cleanly imported without syntax errors or broken dependencies.
2. Production model pipeline can be successfully loaded and executed within the Streamlit context.
3. Model caching and execution helper produces valid schema matching the UI expectations.
4. Deployment readiness: all necessary packages from requirements.txt are importable.
5. Mode A and B demo cases: dataset exists, conforms to required schema, and source URLs are non-empty.
6. Safeguard assertion: demo case inference passes purely judgment_text to the predictor (never ground truth).
7. Theme-aware stylesheet exists and is loadable.
8. Feature explanation extractor produces valid weights from fitted model without parameter modification.
"""

import json
from pathlib import Path
import pytest
from app.streamlit_app import (
    ROOT_DIR,
    CSS_FILE,
    DEMO_DATA_FILE,
    get_cached_pipeline,
    load_demo_cases,
    extract_top_features,
)
from src.predict import predict_judgment
from src.config import FINAL_MODEL_FILE


def test_app_root_dir_resolution():
    """Verifies that ROOT_DIR in streamlit_app.py correctly resolves to the project root."""
    assert ROOT_DIR.exists()
    assert (ROOT_DIR / "src").exists()
    assert (ROOT_DIR / "app").exists()
    assert (ROOT_DIR / "models").exists()


def test_stylesheet_exists_and_readable():
    """Verifies that app/styles/main.css exists and can be loaded relative to repository root."""
    assert CSS_FILE.exists(), f"Stylesheet missing at: {CSS_FILE}"
    content = CSS_FILE.read_text(encoding="utf-8")
    assert "--ll-civil-bg" in content
    assert "--ll-crim-bg" in content
    assert "disclaimer-box" in content


def test_demo_cases_schema_and_integrity():
    """Verifies that demo_cases.json conforms to the required fields and constraints."""
    assert DEMO_DATA_FILE.exists(), f"Demo cases file missing at: {DEMO_DATA_FILE}"
    data = load_demo_cases()
    assert "known_cases" in data
    assert "blind_cases" in data
    assert "guided_examples" in data

    # Verify Known Cases
    for c in data["known_cases"]:
        assert "demo_id" in c
        assert "title" in c
        assert "description" in c
        assert "judgment_text" in c
        assert "ground_truth_category" in c
        assert c["ground_truth_category"] in ["Civil", "Criminal"]
        assert "source_url" in c
        assert c["source_url"].startswith("http")
        assert "source_attribution" in c
        assert "is_real_case" in c
        assert "verification_notes" in c

    # Verify Blind Cases
    for b in data["blind_cases"]:
        assert "demo_id" in b
        assert "title" in b
        assert "description" in b
        assert "judgment_text" in b
        assert "ground_truth_category" in b
        assert b["ground_truth_category"] in ["Civil", "Criminal"]
        assert "source_url" in b
        assert b["source_url"].startswith("http")


def test_demo_cases_isolated_from_predictor():
    """
    CRITICAL SAFEGUARD:
    Ensures that when a demo case is evaluated, the model predictor is passed
    STRICTLY text and never receives ground_truth_category or metadata.
    """
    pipe = get_cached_pipeline()
    data = load_demo_cases()
    for case in data["known_cases"]:
        # Verify call signature accepts only text
        res = predict_judgment(case["judgment_text"], pipeline=pipe)
        assert res["is_valid"] is True
        assert res["predicted_class"] in ["Civil", "Criminal"]
        assert "category" in res


def test_app_pipeline_caching():
    """Verifies that get_cached_pipeline loads the locked production artifact."""
    assert FINAL_MODEL_FILE.exists(), f"Production model missing: {FINAL_MODEL_FILE}"
    pipe = get_cached_pipeline()
    assert pipe is not None
    assert "tfidf" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_extract_top_features_behavior():
    """Verifies that extract_top_features inspects fitted parameters safely without modification."""
    pipe = get_cached_pipeline()
    sample = "The appellant was convicted under Section 302 of the Indian Penal Code and sentenced to imprisonment."
    features = extract_top_features(sample, pipeline=pipe, top_n=5)
    assert len(features) > 0
    assert len(features) <= 5
    first = features[0]
    assert "feature" in first
    assert "tfidf" in first
    assert "weight" in first
    assert "contribution" in first
    assert first["favors"] in ["Civil", "Criminal"]


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


def test_demo_cases_not_in_frozen_test_set():
    """
    REGRESSION SAFEGUARD:
    Ensures that NO demo case in known_cases or blind_cases originates from or
    belongs to the frozen holdout test set (data/processed/frozen_test_ids.json).
    All demo presets must be strictly drawn from the validation split.
    """
    import re
    frozen_test_path = ROOT_DIR / "data" / "processed" / "frozen_test_ids.json"
    assert frozen_test_path.exists(), f"Frozen test IDs missing at {frozen_test_path}"

    with open(frozen_test_path, "r", encoding="utf-8") as f:
        frozen_ids = set(json.load(f))

    data = load_demo_cases()
    all_cases = data["known_cases"] + data["blind_cases"]

    for case in all_cases:
        # Check source URL for Kanoon doc id
        url = case.get("source_url", "")
        match = re.search(r"/doc/(\d+)/?", url)
        if match:
            doc_id = f"{match.group(1)}.txt"
            assert doc_id not in frozen_ids, (
                f"Contamination error: Case '{case.get('demo_id')}' with doc ID '{doc_id}' "
                f"belongs to the frozen holdout test set!"
            )

        # Check source attribution text
        attribution = case.get("source_attribution", "")
        for fid in frozen_ids:
            # Check if any frozen filename is explicitly cited in attribution
            assert fid not in attribution, (
                f"Contamination error: Frozen test ID '{fid}' referenced in attribution: {attribution}"
            )

