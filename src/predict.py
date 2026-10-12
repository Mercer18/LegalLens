"""
LegalLens Milestone 6: Final Prediction Pipeline
Provides a clean, reusable inference interface for Indian court judgment classification.

Architecture (Locked Milestone 5 Candidate):
- Input: Raw Indian court judgment text ONLY.
- Feature Representation: Fitted TF-IDF (1-2 grams, max_features=40,000, sublinear_tf=True).
- Classification Engine: Linear Support Vector Machine (LinearSVC, class_weight='balanced', C=1.0).
- Decision Output: Predicted class ('Civil' / 'Criminal'), internal confidence/decision score,
  and inference execution metadata.

Strict Integrity Constraints:
- Zero data leakage: trained strictly on train.parquet.
- Does NOT touch or evaluate on test.parquet.
- No metadata inputs (case_type, case_info, label, proof_sentence, etc.).
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    COL_CLEAN_TARGET,
    COL_CLEAN_TEXT,
    FINAL_MODEL_FILE,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_CIVIL,
    TARGET_CRIMINAL,
    TRAIN_FILE,
)
from src.features import build_tfidf_vectorizer
from src.models import build_linear_svm
from src.preprocessing import normalize_legal_text


# Global cache for the loaded pipeline in memory
_MODEL_PIPELINE: Optional[Pipeline] = None


def fit_and_save_final_pipeline(
    train_path: Path = TRAIN_FILE,
    model_output_path: Path = FINAL_MODEL_FILE,
    force_refit: bool = False,
) -> Pipeline:
    """
    Fits the locked Milestone 5 candidate pipeline strictly on the training partition
    and serializes the pipeline to disk.
    
    If the artifact already exists and force_refit is False, loads and returns it.
    """
    if model_output_path.exists() and not force_refit:
        print(f"[*] Loading existing final model pipeline from: {model_output_path}")
        pipeline = joblib.load(model_output_path)
        return pipeline

    print("=" * 70)
    print("FITTING LOCKED CANDIDATE MODEL PIPELINE (TRAINING DATA ONLY)")
    print("=" * 70)
    print(f"[*] Training partition source: {train_path}")

    if not train_path.exists():
        raise FileNotFoundError(f"Training partition not found at: {train_path}")

    df_train = pd.read_parquet(train_path)
    X_train = df_train[COL_CLEAN_TEXT].tolist()
    y_train = df_train[COL_CLEAN_TARGET].tolist()

    print(f"[*] Training samples: {len(X_train):,}")
    print("[*] Configuration: TF-IDF (1-2g, max_features=40,000, sublinear_tf=True) + LinearSVC(balanced, C=1.0)")

    # Build exact locked configuration
    vectorizer = build_tfidf_vectorizer(
        ngram_range=(1, 2),
        max_features=40000,
        sublinear_tf=True,
    )
    clf = build_linear_svm(
        class_weight="balanced",
        C=1.0,
        random_state=RANDOM_SEED,
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])

    print("[*] Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"[*] Final model pipeline successfully saved to: {model_output_path}")
    print("=" * 70)

    return pipeline


def load_model_pipeline(model_path: Path = FINAL_MODEL_FILE) -> Pipeline:
    """
    Loads the fitted model pipeline from disk using a module-level cache.
    Ensures that inference loads the existing fitted pipeline rather than refitting.
    """
    global _MODEL_PIPELINE
    if _MODEL_PIPELINE is not None:
        return _MODEL_PIPELINE

    if not model_path.exists():
        # Build and save if not present yet
        _MODEL_PIPELINE = fit_and_save_final_pipeline(model_output_path=model_path)
    else:
        _MODEL_PIPELINE = joblib.load(model_path)

    return _MODEL_PIPELINE


def predict_judgment(
    text: Union[str, Any],
    pipeline: Optional[Pipeline] = None,
) -> Dict[str, Any]:
    """
    Classifies a single Indian court judgment text into 'Civil' or 'Criminal'.

    Args:
        text: Raw Indian court judgment text (string).
        pipeline: Optional pre-loaded pipeline. If None, loads via load_model_pipeline().

    Returns:
        Dict containing:
            - 'predicted_class': 'Civil' or 'Criminal'
            - 'category': 'civil' or 'criminal' (lowercase target representation)
            - 'decision_score': float decision function score from LinearSVC
              (positive favors 'criminal', negative favors 'civil')
            - 'confidence_prob': pseudo-probability derived via sigmoid calibration
            - 'word_count': number of words in normalized text
            - 'char_count': character count in normalized text
            - 'is_valid': boolean indicating whether input text was non-empty and valid
            - 'status': 'success' or error message

    Integrity Safeguards:
        - Rejects empty, whitespace-only, or None text gracefully.
        - Applies the exact domain-aware text normalization used during training.
        - Relies strictly on judgment text; accepts no other metadata fields.
    """
    if pipeline is None:
        pipeline = load_model_pipeline()

    # 1. Validation of Input
    if text is None or not isinstance(text, (str, bytes)):
        return {
            "predicted_class": None,
            "category": None,
            "decision_score": None,
            "confidence_prob": None,
            "word_count": 0,
            "char_count": 0,
            "is_valid": False,
            "status": "error: input text is None or not a string",
        }

    raw_text_str = str(text).strip()
    if not raw_text_str:
        return {
            "predicted_class": None,
            "category": None,
            "decision_score": None,
            "confidence_prob": None,
            "word_count": 0,
            "char_count": 0,
            "is_valid": False,
            "status": "error: input text is empty or whitespace-only",
        }

    # 2. Consistent Preprocessing (Identical to Training Pipeline)
    clean_text = normalize_legal_text(raw_text_str)
    if not clean_text:
        return {
            "predicted_class": None,
            "category": None,
            "decision_score": None,
            "confidence_prob": None,
            "word_count": 0,
            "char_count": 0,
            "is_valid": False,
            "status": "error: text produced empty string after normalization",
        }

    # 3. Model Inference
    # Classes are ['civil', 'criminal']
    pred_raw = pipeline.predict([clean_text])[0]
    decision_val = float(pipeline.decision_function([clean_text])[0])

    # Sigmoid calibration: p(criminal) = 1 / (1 + exp(-decision_score))
    # Decision score > 0 -> criminal, < 0 -> civil
    try:
        prob_criminal = 1.0 / (1.0 + math.exp(-decision_val))
    except OverflowError:
        prob_criminal = 1.0 if decision_val > 0 else 0.0

    prob_civil = 1.0 - prob_criminal

    predicted_class_display = "Criminal" if pred_raw == TARGET_CRIMINAL else "Civil"
    confidence = prob_criminal if pred_raw == TARGET_CRIMINAL else prob_civil

    words = clean_text.split()
    return {
        "predicted_class": predicted_class_display,
        "category": str(pred_raw),
        "decision_score": round(decision_val, 4),
        "confidence_prob": round(confidence, 4),
        "probability_criminal": round(prob_criminal, 4),
        "probability_civil": round(prob_civil, 4),
        "word_count": len(words),
        "char_count": len(clean_text),
        "is_valid": True,
        "status": "success",
    }


def predict_batch_judgments(
    texts: List[str],
    pipeline: Optional[Pipeline] = None,
) -> List[Dict[str, Any]]:
    """
    Batch prediction interface for multiple court judgments.
    """
    if pipeline is None:
        pipeline = load_model_pipeline()

    return [predict_judgment(t, pipeline=pipeline) for t in texts]


if __name__ == "__main__":
    print("[*] Initializing final model pipeline...")
    pipe = load_model_pipeline()
    sample_text = (
        "This criminal appeal is directed against the judgment and order of conviction passed by "
        "the learned Sessions Judge under Section 302 of the Indian Penal Code. The appellant was convicted "
        "and sentenced to undergo rigorous imprisonment for life. Bail application was rejected."
    )
    res = predict_judgment(sample_text, pipeline=pipe)
    print("\nSample Judgment Inference Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
