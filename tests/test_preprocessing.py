"""
Tests for text preprocessing and normalization functions.
"""

import pytest

from src.preprocessing import compute_sha256, normalize_legal_text


def test_normalize_legal_text_preserves_legal_entities():
    sample = "  Section 302 of the IPC, read with  Article 226   of the Constitution. Not guilty! \n\n\t "
    normalized = normalize_legal_text(sample)
    # Check whitespace collapsed
    assert "  " not in normalized
    assert "\n" not in normalized
    assert "\t" not in normalized
    # Check lowercase
    assert normalized.startswith("section 302")
    # Check legal entities and negations preserved
    assert "article 226" in normalized
    assert "not guilty!" in normalized


def test_normalize_empty_and_none():
    assert normalize_legal_text("") == ""
    assert normalize_legal_text(None) == ""


def test_compute_sha256_deterministic():
    t1 = "criminal appeal number 100"
    t2 = "criminal appeal number 100"
    assert compute_sha256(t1) == compute_sha256(t2)
    assert len(compute_sha256(t1)) == 64
    assert compute_sha256("") == ""
