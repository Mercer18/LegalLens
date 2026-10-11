import pytest
import json
from pathlib import Path
from src import config

def test_milestone_5_metrics_exist():
    metrics_path = config.METRICS_DIR / "milestone_5_tuning_metrics.json"
    assert metrics_path.exists(), f"Milestone 5 metrics file missing: {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "error_overview" in data
    assert "document_length_and_truncation" in data
    assert "top_feature_coefficients" in data
    assert "representative_error_case_studies" in data
    assert "tuning_benchmark_results" in data

def test_milestone_5_figures_exist():
    fig1 = config.FIGURES_DIR / "error_length_and_truncation_distribution.png"
    fig2 = config.FIGURES_DIR / "top_model_coefficients.png"
    assert fig1.exists(), f"Figure missing: {fig1}"
    assert fig2.exists(), f"Figure missing: {fig2}"
    assert fig1.stat().st_size > 1000
    assert fig2.stat().st_size > 1000
