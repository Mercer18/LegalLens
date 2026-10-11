"""
LegalLens Milestone 2: Exploratory Data Analysis (EDA) Module
Performs deep-dive statistical analysis on Jud-IPL:
- Duplicate cluster analysis & distribution
- Forensic inspection of conflicting-label records
- Class-wise document length comparison
- Vocabulary & lexical diversity statistics
- Class-specific unigram, bigram, and trigram frequencies
- Text-quality artifact detection
"""

import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    COL_CATEGORY,
    COL_ID,
    COL_INDEX,
    COL_TEXT,
    COL_TYPE,
    FIGURES_DIR,
    METRICS_DIR,
    RAW_DATA_FILE,
    TARGET_CIVIL,
    TARGET_CRIMINAL,
    VALID_TARGETS,
)


def compute_sha256(text: str) -> str:
    """Returns SHA256 hex digest of string."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def run_eda(data_path: Path = RAW_DATA_FILE) -> Dict[str, Any]:
    """Runs complete Milestone 2 EDA pipeline and saves metrics and figures."""
    print("=" * 70)
    print("LEGAL LENS: MILESTONE 2 EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Load labeled Civil and Criminal cases with non-empty text
    print("[1/6] Loading and filtering raw dataset for labeled non-empty judgments...")
    df_raw = pd.read_csv(data_path, usecols=[COL_INDEX, COL_ID, COL_CATEGORY, COL_TYPE, COL_TEXT])
    raw_total = len(df_raw)

    # Clean category string
    df_raw["category_clean"] = df_raw[COL_CATEGORY].astype(str).str.strip().str.lower()
    df_labeled = df_raw[df_raw["category_clean"].isin(VALID_TARGETS)].copy()
    labeled_total = len(df_labeled)

    # Identify empty texts
    empty_mask = df_labeled[COL_TEXT].isna() | (df_labeled[COL_TEXT].astype(str).str.strip() == "")
    empty_count = int(empty_mask.sum())

    df_valid = df_labeled[~empty_mask].copy()
    df_valid["text_clean"] = df_valid[COL_TEXT].astype(str).str.strip()
    valid_total = len(df_valid)

    print(f"[*] Raw rows: {raw_total:,} | Labeled: {labeled_total:,} | Valid Non-Empty: {valid_total:,}")
    print(f"[*] Excluded: {raw_total - labeled_total:,} unlabelled/missing, {empty_count:,} empty text")

    # 2. Detailed Duplicate Cluster & Conflicting-Label Forensics
    print("[2/6] Performing duplicate group cluster & conflict analysis...")
    df_valid["text_hash"] = df_valid["text_clean"].apply(compute_sha256)
    df_valid["char_len"] = df_valid["text_clean"].str.len()
    df_valid["word_len"] = df_valid["text_clean"].apply(lambda t: len(t.split()))

    # Group by text hash
    hash_grouped = df_valid.groupby("text_hash")
    group_sizes = hash_grouped.size()
    duplicate_hashes = group_sizes[group_sizes > 1]
    num_duplicate_groups = len(duplicate_hashes)
    total_duplicate_rows = int(duplicate_hashes.sum())
    excess_duplicate_records = total_duplicate_rows - num_duplicate_groups

    size_distribution = duplicate_hashes.value_counts().sort_index().to_dict()
    size_dist_formatted = {f"group_size_{k}": int(v) for k, v in size_distribution.items()}

    # Check category consistency within each duplicate group
    hash_cats = hash_grouped["category_clean"].nunique()
    same_label_groups = int((hash_cats[hash_cats == 1] > 0).sum())
    conflicting_groups = int((hash_cats > 1).sum())

    # Deep-dive inspection of the conflicting cases
    conflicting_hashes = hash_cats[hash_cats > 1].index.tolist()
    conflicting_case_details = []

    for c_idx, c_hash in enumerate(conflicting_hashes):
        c_records = df_valid[df_valid["text_hash"] == c_hash]
        first_row = c_records.iloc[0]
        text_preview = first_row["text_clean"][:500]
        full_text_sample = first_row["text_clean"]

        records_info = []
        for _, r in c_records.iterrows():
            records_info.append({
                "id": str(r[COL_ID]),
                "row_index": int(r[COL_INDEX]),
                "category": str(r["category_clean"]),
                "case_type": str(r[COL_TYPE]) if pd.notna(r[COL_TYPE]) else "missing",
            })

        conflicting_case_details.append({
            "conflict_id": f"Conflict_{c_idx + 1}",
            "text_hash": c_hash,
            "char_length": int(first_row["char_len"]),
            "word_length": int(first_row["word_len"]),
            "total_occurrences": len(c_records),
            "records": records_info,
            "text_preview": text_preview,
            "text_header_check": text_preview[:150],
        })

    print(f"[*] Duplicate Groups: {num_duplicate_groups:,} | Total rows in dup groups: {total_duplicate_rows:,}")
    print(f"[*] Conflicting Groups: {conflicting_groups} | Same-label Groups: {num_duplicate_groups - conflicting_groups:,}")

    # 3. Class-wise Document Length Statistics
    print("[3/6] Computing class-wise document length distributions...")
    civil_df = df_valid[df_valid["category_clean"] == TARGET_CIVIL]
    crim_df = df_valid[df_valid["category_clean"] == TARGET_CRIMINAL]

    def get_length_stats(series: pd.Series) -> Dict[str, float]:
        return {
            "min": int(series.min()),
            "median": float(series.median()),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
            "p90": float(series.quantile(0.90)),
            "p95": float(series.quantile(0.95)),
            "max": int(series.max()),
        }

    length_stats = {
        "overall_words": get_length_stats(df_valid["word_len"]),
        "overall_chars": get_length_stats(df_valid["char_len"]),
        "civil_words": get_length_stats(civil_df["word_len"]),
        "civil_chars": get_length_stats(civil_df["char_len"]),
        "criminal_words": get_length_stats(crim_df["word_len"]),
        "criminal_chars": get_length_stats(crim_df["char_len"]),
    }

    # 4. Vocabulary & N-Gram Analysis (Sampling for memory safety & speed)
    print("[4/6] Conducting vocabulary and n-gram analysis (stratified sample of 5,000 cases)...")
    sample_seed = 42
    # Stratified sample of 5000 cases (4000 civil, 1000 criminal)
    sample_civ = civil_df.sample(n=min(4000, len(civil_df)), random_state=sample_seed)
    sample_crim = crim_df.sample(n=min(1000, len(crim_df)), random_state=sample_seed)
    sample_df = pd.concat([sample_civ, sample_crim], ignore_index=True)

    # Standard English stopwords to isolate substantive legal vocabulary
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    legal_stopwords = set(ENGLISH_STOP_WORDS).union({
        "court", "said", "shall", "state", "order", "appellant", "respondent",
        "case", "learned", "high", "counsel", "appeal", "held", "judgment",
        "act", "section", "view", "mr", "mrs", "shri", "honble", "justice"
    })

    def extract_ngrams(texts: pd.Series, n: int = 1, top_k: int = 25, remove_stop: bool = True) -> List[Tuple[str, int]]:
        counts = collections.Counter()
        for t in texts:
            # Simple whitespace/word regex
            words = re.findall(r"\b[a-zA-Z]{3,}\b", str(t).lower())
            if remove_stop:
                words = [w for w in words if w not in legal_stopwords]
            if n == 1:
                counts.update(words)
            elif n == 2:
                bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                counts.update(bigrams)
            elif n == 3:
                trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
                counts.update(trigrams)
        return counts.most_common(top_k)

    top_unigrams_overall = extract_ngrams(sample_df["text_clean"], n=1, top_k=25)
    top_unigrams_civil = extract_ngrams(sample_civ["text_clean"], n=1, top_k=25)
    top_unigrams_crim = extract_ngrams(sample_crim["text_clean"], n=1, top_k=25)

    top_bigrams_overall = extract_ngrams(sample_df["text_clean"], n=2, top_k=25)
    top_bigrams_civil = extract_ngrams(sample_civ["text_clean"], n=2, top_k=25)
    top_bigrams_crim = extract_ngrams(sample_crim["text_clean"], n=2, top_k=25)

    # Lexical Diversity (Type-Token Ratio on sample)
    total_tokens_sample = sum(len(str(t).split()) for t in sample_df["text_clean"])
    unique_tokens_sample = len(set(w for t in sample_df["text_clean"] for w in re.findall(r"\b[a-zA-Z]{2,}\b", str(t).lower())))
    ttr = (unique_tokens_sample / total_tokens_sample) if total_tokens_sample > 0 else 0

    vocab_stats = {
        "sample_size": len(sample_df),
        "sample_total_tokens": total_tokens_sample,
        "sample_unique_tokens": unique_tokens_sample,
        "type_token_ratio": round(ttr, 5),
        "top_unigrams_overall": [f"{k}: {v}" for k, v in top_unigrams_overall[:15]],
        "top_unigrams_civil": [f"{k}: {v}" for k, v in top_unigrams_civil[:15]],
        "top_unigrams_criminal": [f"{k}: {v}" for k, v in top_unigrams_crim[:15]],
        "top_bigrams_civil": [f"{k}: {v}" for k, v in top_bigrams_civil[:15]],
        "top_bigrams_criminal": [f"{k}: {v}" for k, v in top_bigrams_crim[:15]],
    }

    # 5. Text Quality & Formatting Artifacts Check
    print("[5/6] Checking formatting artifacts & text quality anomalies...")
    # Check uppercase proportion, digit proportion, newline counts, etc.
    sample_text = df_valid["text_clean"].sample(n=1000, random_state=42)
    has_html_tags = int(sample_text.str.contains(r"<[^>]+>", regex=True).sum())
    has_multiple_newlines = int(sample_text.str.contains(r"\n\s*\n\s*\n", regex=True).sum())
    ends_abruptly = int(sample_text.apply(lambda s: len(s) >= 32760 and not s.rstrip().endswith((".", "!", "?", "\""))).sum())

    text_quality_findings = {
        "sample_size_audited": 1000,
        "sample_with_html_tags": has_html_tags,
        "sample_with_excess_whitespace_newlines": has_multiple_newlines,
        "sample_ending_abruptly_at_cap": ends_abruptly,
        "capping_ceiling": 32765,
    }

    # Compile comprehensive EDA summary
    eda_summary = {
        "dataset_name": "Jud-IPL (case_files_total.csv)",
        "summary_counts": {
            "raw_total": raw_total,
            "labeled_total": labeled_total,
            "excluded_unlabeled": raw_total - labeled_total,
            "excluded_empty_text": empty_count,
            "valid_nonempty_total": valid_total,
            "civil_count": len(civil_df),
            "civil_percentage": round(len(civil_df) / valid_total * 100, 2),
            "criminal_count": len(crim_df),
            "criminal_percentage": round(len(crim_df) / valid_total * 100, 2),
            "imbalance_ratio": round(len(civil_df) / len(crim_df), 2),
        },
        "duplicate_analysis": {
            "unique_text_hashes": int(df_valid["text_hash"].nunique()),
            "total_duplicate_rows": total_duplicate_rows,
            "distinct_duplicate_groups": num_duplicate_groups,
            "excess_duplicate_records": excess_duplicate_records,
            "group_size_distribution": size_dist_formatted,
            "same_label_groups": same_label_groups,
            "conflicting_label_groups": conflicting_groups,
            "conflicting_case_forensics": conflicting_case_details,
        },
        "document_lengths": length_stats,
        "vocabulary_and_ngrams": vocab_stats,
        "text_quality_artifacts": text_quality_findings,
    }

    # Save metrics JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "eda_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"[*] Saved EDA metrics to: {metrics_path}")

    # 6. Generate Visual Charts
    print("[6/6] Generating EDA visualizations...")
    generate_eda_charts(df_valid, duplicate_hashes, top_bigrams_civil, top_bigrams_crim)

    print("=" * 70)
    print("EDA PIPELINE COMPLETE")
    print("=" * 70)
    return eda_summary


def generate_eda_charts(df_valid: pd.DataFrame, duplicate_hashes: pd.Series, 
                        top_bigrams_civil: List[Tuple[str, int]], 
                        top_bigrams_crim: List[Tuple[str, int]]) -> None:
    """Generates visual figures for EDA."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Class-wise Document Length Boxplot & KDE
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(
        data=df_valid, 
        x="category_clean", 
        y="word_len", 
        palette={"civil": "#2b5c8f", "criminal": "#d95f02"},
        showfliers=False
    )
    plt.title("Document Length by Class (IQR Without Outliers)", fontsize=11, fontweight="bold")
    plt.xlabel("Case Category", fontsize=10)
    plt.ylabel("Word Count", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    # KDE up to 6000 words
    for cat, col in [("civil", "#2b5c8f"), ("criminal", "#d95f02")]:
        subset = df_valid[df_valid["category_clean"] == cat]["word_len"]
        sns.kdeplot(subset, label=f"{cat.capitalize()} (Med: {subset.median():,.0f})", color=col, linewidth=2)
    plt.xlim(0, 6500)
    plt.title("Document Word Count Distribution (KDE)", fontsize=11, fontweight="bold")
    plt.xlabel("Word Count", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    f1_path = FIGURES_DIR / "class_wise_length_distribution.png"
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print(f"[*] Saved figure: {f1_path}")

    # Figure 2: Duplicate Group Sizes Distribution
    plt.figure(figsize=(8, 4.5))
    sizes = duplicate_hashes.value_counts().sort_index().head(6)
    bars = plt.bar([str(s) for s in sizes.index], sizes.values, color="#7570b3", edgecolor="black", alpha=0.85)
    plt.title("Distribution of Duplicate Group Cluster Sizes", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Duplicate Copies in Cluster", fontsize=10)
    plt.ylabel("Number of Distinct Clusters", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 30, f"{h:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    f2_path = FIGURES_DIR / "duplicate_group_sizes.png"
    plt.savefig(f2_path, dpi=200)
    plt.close()
    print(f"[*] Saved figure: {f2_path}")

    # Figure 3: Top Discriminative Bigrams (Civil vs Criminal)
    plt.figure(figsize=(13, 6))

    plt.subplot(1, 2, 1)
    civ_bgs = [b[0] for b in top_bigrams_civil[:12]]
    civ_cnts = [b[1] for b in top_bigrams_civil[:12]]
    plt.barh(civ_bgs[::-1], civ_cnts[::-1], color="#2b5c8f", edgecolor="black", alpha=0.85)
    plt.title("Top Substantive Bigrams: Civil Cases", fontsize=11, fontweight="bold")
    plt.xlabel("Term Frequency (Sampled)", fontsize=10)

    plt.subplot(1, 2, 2)
    crim_bgs = [b[0] for b in top_bigrams_crim[:12]]
    crim_cnts = [b[1] for b in top_bigrams_crim[:12]]
    plt.barh(crim_bgs[::-1], crim_cnts[::-1], color="#d95f02", edgecolor="black", alpha=0.85)
    plt.title("Top Substantive Bigrams: Criminal Cases", fontsize=11, fontweight="bold")
    plt.xlabel("Term Frequency (Sampled)", fontsize=10)

    plt.tight_layout()
    f3_path = FIGURES_DIR / "top_bigrams_by_class.png"
    plt.savefig(f3_path, dpi=200)
    plt.close()
    print(f"[*] Saved figure: {f3_path}")


if __name__ == "__main__":
    run_eda()
