"""
NLP Utilities for LegalLens Application
Provides technical NLP capabilities:
1. Local Feature Attribution (Salient Legal N-Grams): Exact linear SVM margin contributions (x_j * w_j).
2. Domain-Aware Statutory & Entity Extraction: Pattern-based extraction of Indian statutory citations and acts.
3. Document Stylometry & Linguistic Diagnostics: Lexical diversity (TTR), sentence metrics, and vocabulary coverage.
"""

import re
from typing import Any, Dict, List, Tuple
from sklearn.pipeline import Pipeline
from src.preprocessing import normalize_legal_text


# High-precision statutory citation and legal act patterns for Indian jurisprudence
STATUTE_PATTERNS = [
    (
        r"(?:(?:Section|Sec\.?|s\.)\s*\d+[A-Za-z]?(?:\s*(?:and|,)\s*\d+[A-Za-z]?)*(?:\s+of\s+(?:the\s+)?)?"
        r"(?:IPC|Cr\.?P\.?C\.?|C\.?P\.?C\.?|Indian\s+Penal\s+Code|Arms\s+Act|Motor\s+Vehicles\s+Act|"
        r"Specific\s+Relief\s+Act|Evidence\s+Act|Contract\s+Act|Negotiable\s+Instruments\s+Act|NI\s+Act)?)",
        "Section / Provision",
    ),
    (
        r"(?:Order\s+[IVXLCDM]+(?:\s+Rule\s+\d+)?)",
        "CPC Order / Rule",
    ),
    (
        r"(?:Article\s+\d+[A-Za-z]?(?:\s*\(\d+\))?(?:\s+of\s+(?:the\s+)?)?(?:Constitution(?:\s+of\s+India)?)?)",
        "Constitutional Article",
    ),
    (
        r"(?:Indian\s+Penal\s+Code|Code\s+of\s+Criminal\s+Procedure|Code\s+of\s+Civil\s+Procedure|"
        r"Arms\s+Act|Motor\s+Vehicles\s+Act|Specific\s+Relief\s+Act|Evidence\s+Act|Contract\s+Act|"
        r"Negotiable\s+Instruments\s+Act|Prevention\s+of\s+Corruption\s+Act|NDPS\s+Act|"
        r"Transfer\s+of\s+Property\s+Act|Arbitration\s+and\s+Conciliation\s+Act|Hindu\s+Marriage\s+Act)",
        "Statutory Act / Code",
    ),
]


def extract_top_features(
    text: str,
    pipeline: Pipeline,
    top_n: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Computes exact local feature attribution using the fitted Linear SVM decision hyperplane.
    
    For a linear SVM, decision_score = sum(x_j * w_j) + b.
    Each term x_j * w_j represents the direct numerical contribution of n-gram feature j
    to the decision boundary.
    - Positive contribution (> 0): pushes the prediction toward Criminal.
    - Negative contribution (< 0): pushes the prediction toward Civil.

    Returns:
        Dict with 'criminal_features' and 'civil_features', each a list of (ngram, contribution).
    """
    if not text or not str(text).strip():
        return {"criminal_features": [], "civil_features": []}

    clean_text = normalize_legal_text(str(text))
    if not clean_text:
        return {"criminal_features": [], "civil_features": []}

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    vec = tfidf.transform([clean_text])
    feature_names = tfidf.get_feature_names_out()
    weights = clf.coef_[0]

    indices = vec.indices
    data = vec.data

    contributions = []
    for i, idx in enumerate(indices):
        token = feature_names[idx]
        contribution = float(data[i] * weights[idx])
        contributions.append((token, contribution))

    # Criminal features: highest positive contributions
    crim_features = [item for item in contributions if item[1] > 0]
    crim_features.sort(key=lambda x: x[1], reverse=True)

    # Civil features: most negative contributions
    civ_features = [item for item in contributions if item[1] < 0]
    civ_features.sort(key=lambda x: x[1])  # most negative first

    return {
        "criminal_features": crim_features[:top_n],
        "civil_features": civ_features[:top_n],
    }


def extract_statutory_entities(text: str) -> List[Tuple[str, str]]:
    """
    Identifies and categorizes statutory citations, procedural orders, and legislative acts
    referenced in Indian court judgments.

    Returns:
        List of (citation_text, category_label) tuples, deduplicated and sorted by appearance.
    """
    if not text or not str(text).strip():
        return []

    raw_text = str(text)
    matches = []

    for pat, category in STATUTE_PATTERNS:
        for m in re.finditer(pat, raw_text, re.IGNORECASE):
            raw = m.group(0).strip()
            cleaned = re.sub(r"\s+", " ", raw)
            # Filter noise and single short digits
            if cleaned and len(cleaned) >= 3 and not re.match(r"^(?:Section|Sec\.?|s\.)\s*$", cleaned, re.I):
                matches.append((cleaned, category, m.start(), m.end()))

    # Sort primarily by span start position, then by span length descending
    matches.sort(key=lambda x: (x[2], -(x[3] - x[2])))

    # Deduplicate overlapping sub-spans and repeated citations
    filtered = []
    for item in matches:
        span = (item[2], item[3])
        # If this span is a strict sub-span of an already accepted span, ignore it
        if not any(f[2] <= span[0] and span[1] <= f[3] and (f[3] - f[2]) > (span[1] - span[0]) for f in filtered):
            if not any(f[0].lower() == item[0].lower() for f in filtered):
                filtered.append(item)

    return [(f[0], f[1]) for f in filtered]


def compute_text_diagnostics(
    text: str,
    pipeline: Pipeline,
) -> Dict[str, Any]:
    """
    Calculates document stylometry and NLP lexical diagnostics:
    - Sentence count & average sentence length
    - Type-Token Ratio (TTR / Lexical Diversity)
    - Active model vocabulary overlap (features matched in the locked 40k vocabulary)
    """
    if not text or not str(text).strip():
        return {
            "sentence_count": 0,
            "word_count": 0,
            "unique_words": 0,
            "ttr": 0.0,
            "avg_sentence_len": 0.0,
            "active_vocab_count": 0,
        }

    raw_text = str(text).strip()
    sentences = [s.strip() for s in re.split(r"[\.\?\!]+", raw_text) if s.strip()]
    sentence_count = max(len(sentences), 1)

    clean_text = normalize_legal_text(raw_text)
    words = clean_text.split()
    word_count = len(words)

    unique_words = len(set(words))
    ttr = (unique_words / max(word_count, 1)) * 100.0
    avg_sentence_len = word_count / sentence_count

    # Measure how many active n-grams match the locked 40,000 TF-IDF features
    tfidf = pipeline.named_steps["tfidf"]
    vec = tfidf.transform([clean_text])
    active_vocab_count = len(vec.indices)

    return {
        "sentence_count": sentence_count,
        "word_count": word_count,
        "unique_words": unique_words,
        "ttr": round(ttr, 1),
        "avg_sentence_len": round(avg_sentence_len, 1),
        "active_vocab_count": active_vocab_count,
    }


def load_validation_cases(
    n_per_category: int = 5,
    seed: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Samples authentic Indian court judgments from the 3,372-record validation pool.
    Balances cases evenly between Criminal and Civil categories.
    Falls back to app/validation_pool.json if val.parquet is not present.
    """
    import json
    from pathlib import Path
    import pandas as pd
    from src.config import VAL_FILE

    cases: Dict[str, Dict[str, Any]] = {}
    
    if VAL_FILE.exists():
        df_val = pd.read_parquet(VAL_FILE)
        civ_df = df_val[df_val["category"] == "civil"]
        crim_df = df_val[df_val["category"] == "criminal"]
        
        sample_civ = civ_df.sample(min(n_per_category, len(civ_df)), random_state=seed)
        sample_crim = crim_df.sample(min(n_per_category, len(crim_df)), random_state=seed)
        sampled = pd.concat([sample_crim, sample_civ])
        
        for _, row in sampled.iterrows():
            cat = str(row["category"]).title()
            doc_id = str(row["name"]).replace(".txt", "")
            words = str(row["text"]).split()
            if len(words) > 400:
                text = " ".join(words[:400]) + "..."
            else:
                text = str(row["text"]).strip()
                
            label = f"[{cat}] Indian Kanoon Doc #{doc_id} ({len(words)} words)"
            cases[label] = {
                "text": text,
                "category": cat,
                "doc_id": doc_id,
                "total_words": len(words),
            }
        return cases

    # Fallback to pre-bundled JSON pool
    json_path = Path(__file__).resolve().parent / "validation_pool.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        import random
        if seed is not None:
            random.seed(seed)
        crim_items = [d for d in data if d["category"] == "Criminal"]
        civ_items = [d for d in data if d["category"] == "Civil"]
        
        selected = random.sample(crim_items, min(n_per_category, len(crim_items))) + \
                   random.sample(civ_items, min(n_per_category, len(civ_items)))
                   
        for item in selected:
            cat = item["category"]
            doc_id = item["doc_id"]
            words_count = item["total_words"]
            label = f"[{cat}] Indian Kanoon Doc #{doc_id} ({words_count} words)"
            cases[label] = {
                "text": item["text"],
                "category": cat,
                "doc_id": doc_id,
                "total_words": words_count,
            }
        return cases

    return cases
