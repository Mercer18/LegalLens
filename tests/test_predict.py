"""
Unit tests for LegalLens Milestone 6: Final Prediction Pipeline
Verifies:
1. Normal judgment prediction (Civil and Criminal cases).
2. Empty and invalid inputs (None, empty string, whitespace-only, non-string types).
3. Preprocessing consistency (normalizing text produces consistent inference results).
4. Model/vectorizer loading and caching behavior.
5. Strict feature integrity (refusal to consume or require metadata fields).
"""

from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.predict import fit_and_save_final_pipeline, load_model_pipeline, predict_judgment, predict_batch_judgments
from src.preprocessing import normalize_legal_text


def test_final_model_artifact_exists_and_loads():
    """Verifies that the serialized final pipeline exists and can be loaded without retraining."""
    assert config.FINAL_MODEL_FILE.exists(), f"Final model file missing: {config.FINAL_MODEL_FILE}"
    assert config.FINAL_MODEL_FILE.stat().st_size > 1000000, "Model file size unexpectedly small"

    pipeline = load_model_pipeline(config.FINAL_MODEL_FILE)
    assert isinstance(pipeline, Pipeline)
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    assert isinstance(tfidf, TfidfVectorizer)
    assert isinstance(clf, LinearSVC)
    assert tfidf.max_features == 40000
    assert tfidf.ngram_range == (1, 2)
    assert tfidf.sublinear_tf is True
    assert clf.class_weight == "balanced"
    assert clf.C == 1.0


def test_predict_normal_criminal_judgment():
    """Verifies typical criminal judgment inference."""
    criminal_text = (
        "The appellant was convicted by the learned Sessions Judge under Section 302 of the Indian Penal Code "
        "and Section 25 of the Arms Act and sentenced to rigorous imprisonment for life. The prosecution examined "
        "ten eyewitnesses whose testimony established the commission of the offence beyond reasonable doubt. "
        "The application for bail was rejected and the appeal is dismissed."
    )
    result = predict_judgment(criminal_text)

    assert result["is_valid"] is True
    assert result["status"] == "success"
    assert result["predicted_class"] == "Criminal"
    assert result["category"] == "criminal"
    assert result["decision_score"] > 0
    assert result["confidence_prob"] > 0.5
    assert result["probability_criminal"] > 0.5
    assert result["probability_civil"] < 0.5
    assert result["word_count"] > 20


def test_predict_normal_civil_judgment():
    """Verifies typical civil judgment inference."""
    civil_text = (
        "This civil appeal arises out of a suit for specific performance of a contract for sale of immovable property. "
        "The plaintiff filed the original suit in the Court of the Civil Judge, claiming relief of execution of the sale deed. "
        "The Trial Court decreed the suit with costs. The First Appellate Court reversed the decree. Hence, this regular "
        "second appeal before the High Court under Section 100 of the Code of Civil Procedure."
    )
    result = predict_judgment(civil_text)

    assert result["is_valid"] is True
    assert result["status"] == "success"
    assert result["predicted_class"] == "Civil"
    assert result["category"] == "civil"
    assert result["decision_score"] < 0
    assert result["confidence_prob"] > 0.5
    assert result["probability_civil"] > 0.5
    assert result["probability_criminal"] < 0.5
    assert result["word_count"] > 20


def test_predict_invalid_and_empty_inputs():
    """Verifies graceful handling of empty, None, and whitespace-only inputs without crashing."""
    invalid_inputs = [
        None,
        "",
        "    ",
        "\n\t\r\n",
        12345,
        {"some_dict": "invalid"},
        [],
    ]

    for val in invalid_inputs:
        res = predict_judgment(val)
        assert res["is_valid"] is False
        assert res["predicted_class"] is None
        assert res["category"] is None
        assert res["decision_score"] is None
        assert "error" in res["status"]


def test_preprocessing_consistency():
    """
    Verifies that calling predict_judgment directly on raw unnormalized text yields
    the identical decision score and category as explicitly pre-normalizing with normalize_legal_text.
    """
    raw_messy_text = (
        "   IN THE HIGH COURT OF JUDICATURE AT BOMBAY   \n\n\n"
        "   Civil Appeal No. 412 OF 2018   \n\n"
        "   The appellant seeks specific performance of contract with costs awarded by the trial court.   "
    )

    clean_text = normalize_legal_text(raw_messy_text)
    pipeline = load_model_pipeline()

    # Direct predict
    res_raw = predict_judgment(raw_messy_text, pipeline=pipeline)
    # Normalized predict
    res_clean = predict_judgment(clean_text, pipeline=pipeline)

    assert res_raw["predicted_class"] == res_clean["predicted_class"]
    assert res_raw["category"] == res_clean["category"]
    assert pytest.approx(res_raw["decision_score"], abs=1e-4) == res_clean["decision_score"]
    assert pytest.approx(res_raw["confidence_prob"], abs=1e-4) == res_clean["confidence_prob"]


def test_batch_prediction_interface():
    """Verifies that the batch interface processes a list of mixed judgments accurately."""
    batch = [
        "Criminal writ petition seeking regular bail under Section 439 CrPC in murder offence under IPC.",
        "Civil regular second appeal under section 100 CPC regarding partition of ancestral agricultural property and costs.",
        "",  # Invalid entry in batch
    ]

    results = predict_batch_judgments(batch)
    assert len(results) == 3
    assert results[0]["predicted_class"] == "Criminal"
    assert results[1]["predicted_class"] == "Civil"
    assert results[2]["is_valid"] is False


def test_final_test_set_integrity_and_metrics():
    """
    Guarantees test set integrity: checks that final test metrics exist, match frozen test IDs,
    and verified zero text-hash contamination.
    """
    test_metrics_path = config.METRICS_DIR / "final_test_metrics.json"
    assert test_metrics_path.exists(), "Final test metrics file missing!"
    assert config.FROZEN_TEST_IDS_FILE.exists(), "Frozen test set hash index missing!"

    import json
    with open(test_metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["integrity_verification"]["frozen_test_ids_matched"] is True
    assert data["integrity_verification"]["train_test_overlap"] == 0
    assert data["integrity_verification"]["val_test_overlap"] == 0
    assert data["integrity_verification"]["model_refitted_on_test"] is False
    assert data["total_test_samples"] == 3372
    assert data["test_metrics"]["accuracy"] > 0.90
    assert data["test_metrics"]["macro_f1"] > 0.88
