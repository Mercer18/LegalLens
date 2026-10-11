"""
LegalLens Feature Extraction Module (Milestone 4)
Builds and fits TF-IDF vectorizers strictly on the training partition.
Guarantees zero data leakage into validation and test sets.
"""

from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# Substantive legal stopwords (standard English stopwords excluding negations)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Preserve critical negations
LEGAL_PRESERVED_TOKENS = {"not", "no", "nor", "neither", "never", "without", "against"}
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - LEGAL_PRESERVED_TOKENS)


def build_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 30000,
    min_df: int = 3,
    max_df: float = 0.90,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Constructs a TF-IDF vectorizer optimized for legal whole-judgment texts:
    - n-gram range (1, 2) captures key phrases like 'writ petition', 'penal code'.
    - sublinear_tf=True dampens the effect of very high frequency terms in long judgments.
    - min_df=3 filters rare hapax legomena/OCR typos.
    - max_df=0.90 filters corpus-wide universal boilerplate.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        stop_words=CUSTOM_STOP_WORDS,
        strip_accents="unicode",
        dtype="float32",
    )
