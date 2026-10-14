"""
Unit tests for app.nlp_utils
Verifies:
1. Linear SVM feature attribution computes mathematically sound contributions.
2. Indian statutory entity extractor parses sections, acts, and procedural codes.
3. Linguistic diagnostics (TTR, sentence length, vocabulary overlap) compute accurately.
"""

import pytest
from app.nlp_utils import (
    extract_top_features,
    extract_statutory_entities,
    compute_text_diagnostics,
)
from app.streamlit_app import get_cached_pipeline


@pytest.fixture(scope="module")
def pipeline():
    return get_cached_pipeline()


def test_feature_attribution_criminal_sample(pipeline):
    text = (
        "The appellant was convicted by the learned Sessions Judge under Section 302 IPC "
        "and sentenced to rigorous imprisonment for life. Bail was rejected during trial. "
        "The criminal appeal is dismissed."
    )
    result = extract_top_features(text, pipeline, top_n=5)
    assert "criminal_features" in result
    assert "civil_features" in result
    assert len(result["criminal_features"]) > 0

    crim_words = [token for token, score in result["criminal_features"]]
    # Words like 'convicted', 'bail', 'criminal' should have positive contribution
    assert any(w in crim_words for w in ["criminal", "convicted", "bail", "appellant"])
    # All criminal features should have positive weights
    for _, score in result["criminal_features"]:
        assert score > 0


def test_feature_attribution_civil_sample(pipeline):
    text = (
        "This civil regular second appeal arises out of a suit for specific performance "
        "of an agreement to sell immovable property. The trial court decreed the suit with costs."
    )
    result = extract_top_features(text, pipeline, top_n=5)
    assert len(result["civil_features"]) > 0

    civ_words = [token for token, score in result["civil_features"]]
    # Words like 'civil', 'suit', 'decreed' should have negative contribution
    assert any(w in civ_words for w in ["civil", "suit", "decreed", "appeal", "costs"])
    # All civil features should have negative weights
    for _, score in result["civil_features"]:
        assert score < 0


def test_feature_attribution_empty_text(pipeline):
    result = extract_top_features("", pipeline)
    assert result == {"criminal_features": [], "civil_features": []}


def test_statutory_entity_extraction():
    text = (
        "Conviction under Section 302 of the Indian Penal Code and Section 25 of the Arms Act. "
        "Application under Section 437 CrPC was rejected. Later filed petition under Article 226 of Constitution of India."
    )
    entities = extract_statutory_entities(text)
    assert len(entities) >= 3

    citation_texts = [e[0] for e in entities]
    assert any("302" in c for c in citation_texts)
    assert any("Arms Act" in c for c in citation_texts)
    assert any("226" in c for c in citation_texts)


def test_statutory_entity_extraction_empty():
    assert extract_statutory_entities("") == []
    assert extract_statutory_entities("No legal terms here.") == []


def test_compute_text_diagnostics(pipeline):
    text = (
        "The appellant was convicted by the learned Sessions Judge under Section 302 IPC. "
        "The prosecution examined ten eyewitnesses. The criminal appeal is dismissed."
    )
    diagnostics = compute_text_diagnostics(text, pipeline)

    assert diagnostics["sentence_count"] == 3
    assert diagnostics["word_count"] > 10
    assert 0.0 < diagnostics["ttr"] <= 100.0
    assert diagnostics["avg_sentence_len"] > 0
    assert diagnostics["active_vocab_count"] > 0
