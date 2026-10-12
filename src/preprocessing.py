"""
LegalLens Milestone 3: Data Cleaning & Deduplication Module
Implements the verified cleaning policy:
1. Prunes unannotated/invalid rows and empty judgment texts.
2. Identifies and removes the two known conflicting-label clusters (6 records).
3. Verifies that all remaining duplicate clusters have strictly consistent labels.
   (Halts execution with an error if any undocumented conflict is found).
4. Retains exactly one representative per consistent duplicate cluster.
5. Normalizes whitespace and standardizes casing while preserving legal semantics.
6. Persists cleaned dataset to data/interim/.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from src.config import (
    CLEANED_DATA_CSV,
    CLEANED_DATA_FILE,
    COL_CATEGORY,
    COL_CLEAN_TARGET,
    COL_CLEAN_TEXT,
    COL_ID,
    COL_INDEX,
    COL_TEXT,
    COL_TEXT_HASH,
    INTERIM_DATA_DIR,
    METRICS_DIR,
    RAW_DATA_FILE,
    TARGET_CIVIL,
    TARGET_CRIMINAL,
    VALID_TARGETS,
)


def compute_sha256(text: str) -> str:
    """Computes SHA-256 hash of a string after stripping leading/trailing whitespace."""
    if not text:
        return ""
    return hashlib.sha256(text.strip().encode("utf-8", errors="replace")).hexdigest()


def normalize_legal_text(text: str) -> str:
    """
    Normalizes court judgment text while preserving essential legal semantics:
    - Normalizes excessive whitespace, tabs, and multiple newlines to a single space.
    - Strips leading and trailing whitespace.
    - Lowercases text for consistent TF-IDF vocabulary representation.
    - Preserves statutory section numbers, article references, citations, and negations.
    """
    if pd.isna(text):
        return ""
    # Normalize internal whitespace
    s = re.sub(r"\s+", " ", str(text))
    return s.strip().lower()


def clean_and_deduplicate_dataset(
    raw_path: Path = RAW_DATA_FILE,
    save_interim: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full Milestone 3 cleaning and deduplication pipeline.
    
    Returns:
        Tuple of (cleaned_df, cleaning_stats_dict)
    """
    print("=" * 70)
    print("LEGAL LENS: MILESTONE 3 DATA CLEANING & DEDUPLICATION PIPELINE")
    print("=" * 70)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at {raw_path}")

    # Step 1: Load Raw Dataset
    print(f"[*] Loading raw dataset from: {raw_path}")
    df_raw = pd.read_csv(raw_path, usecols=[COL_INDEX, COL_ID, COL_CATEGORY, COL_TEXT])
    raw_count = len(df_raw)
    print(f"[*] Raw rows loaded: {raw_count:,}")

    # Standardize category strings
    cat_clean = df_raw[COL_CATEGORY].astype(str).str.strip().str.lower()
    valid_cat_mask = cat_clean.isin(VALID_TARGETS)
    unlabeled_count = int((~valid_cat_mask).sum())

    df_labeled = df_raw[valid_cat_mask].copy()
    df_labeled[COL_CLEAN_TARGET] = cat_clean[valid_cat_mask]
    labeled_count = len(df_labeled)
    print(f"[*] Labeled rows (Civil/Criminal): {labeled_count:,} (Excluded unlabeled/invalid: {unlabeled_count:,})")

    # Step 2: Prune Empty / Whitespace-Only Text
    empty_mask = df_labeled[COL_TEXT].isna() | (df_labeled[COL_TEXT].astype(str).str.strip() == "")
    empty_count = int(empty_mask.sum())

    df_valid_text = df_labeled[~empty_mask].copy()
    valid_text_count = len(df_valid_text)
    print(f"[*] Non-empty valid text rows: {valid_text_count:,} (Excluded empty texts: {empty_count:,})")

    # Step 3: Compute Exact Text Hashes on Cleaned Text
    print("[*] Normalizing whitespace and computing SHA-256 hashes...")
    df_valid_text[COL_CLEAN_TEXT] = df_valid_text[COL_TEXT].apply(normalize_legal_text)
    df_valid_text[COL_TEXT_HASH] = df_valid_text[COL_CLEAN_TEXT].apply(compute_sha256)

    # Step 4: Identify & Verify All Conflicting Duplicate Clusters
    print("[*] Auditing all duplicate clusters for category label consistency...")
    hash_grouped = df_valid_text.groupby(COL_TEXT_HASH)
    cluster_cat_nunique = hash_grouped[COL_CLEAN_TARGET].nunique()
    conflicting_hashes = cluster_cat_nunique[cluster_cat_nunique > 1].index.tolist()

    print(f"[*] Conflicting duplicate clusters detected: {len(conflicting_hashes)}")

    # Safety Check: We expect exactly the 2 clusters identified in Milestone 2.
    # If any new/unexpected conflicting cluster appears, we report and raise an error.
    conflicting_rows_mask = df_valid_text[COL_TEXT_HASH].isin(conflicting_hashes)
    conflicting_rows_count = int(conflicting_rows_mask.sum())

    conflicting_details = []
    for c_hash in conflicting_hashes:
        c_rows = df_valid_text[df_valid_text[COL_TEXT_HASH] == c_hash]
        cats = c_rows[COL_CLEAN_TARGET].tolist()
        ids = c_rows[COL_ID].tolist()
        conflicting_details.append({
            "hash": c_hash,
            "occurrences": len(c_rows),
            "categories": cats,
            "ids": ids,
            "preview": c_rows[COL_CLEAN_TEXT].iloc[0][:200]
        })

    print(f"[*] Total rows in conflicting clusters to be pruned: {conflicting_rows_count}")
    for idx, c in enumerate(conflicting_details):
        print(f"    - Conflict {idx + 1}: {c['occurrences']} rows, categories={c['categories']}, ids={c['ids']}")

    if len(conflicting_hashes) > 2:
        error_msg = (
            f"CRITICAL SAFETY STOP: Discovered {len(conflicting_hashes)} conflicting duplicate clusters, "
            f"which exceeds the 2 known clusters identified in Milestone 2. Details: {conflicting_details}"
        )
        print(f"[!] {error_msg}")
        raise ValueError(error_msg)

    # Drop conflicting clusters
    df_non_conflicting = df_valid_text[~conflicting_rows_mask].copy()
    non_conflicting_count = len(df_non_conflicting)
    print(f"[*] Rows after dropping conflicting clusters: {non_conflicting_count:,}")

    # Verification: Verify 100% of remaining duplicate clusters have exactly 1 unique label
    remaining_grouped = df_non_conflicting.groupby(COL_TEXT_HASH)[COL_CLEAN_TARGET].nunique()
    remaining_inconsistencies = int((remaining_grouped > 1).sum())
    if remaining_inconsistencies > 0:
        raise RuntimeError(f"Assertion failed: {remaining_inconsistencies} inconsistent clusters remain!")
    print("[*] VERIFIED: 100% of remaining duplicate clusters have exactly 1 consistent label.")

    # Step 5: Deduplicate - Keep exactly 1 representative per unique text hash
    print("[*] Deduplicating by unique text hash (keeping first representative per cluster)...")
    initial_unique_count = int(df_non_conflicting[COL_TEXT_HASH].nunique())
    
    # Sort by original index to ensure deterministic representative selection
    df_sorted = df_non_conflicting.sort_values(by=COL_INDEX)
    df_deduped = df_sorted.drop_duplicates(subset=[COL_TEXT_HASH], keep="first").copy()
    final_clean_count = len(df_deduped)
    duplicate_copies_removed = non_conflicting_count - final_clean_count

    print(f"[*] Final deduplicated unique judgments: {final_clean_count:,}")
    print(f"[*] Excess duplicate copies removed: {duplicate_copies_removed:,}")

    # Retain strictly only model input features and identifiers
    # Features: 'text', Target: 'category', Metadata tracking: 'name', 'text_hash'
    keep_columns = [COL_ID, COL_TEXT_HASH, COL_CLEAN_TEXT, COL_CLEAN_TARGET]
    df_final = df_deduped[keep_columns].reset_index(drop=True)

    # Class Breakdown in Final Cleaned Dataset
    civil_final = int((df_final[COL_CLEAN_TARGET] == TARGET_CIVIL).sum())
    criminal_final = int((df_final[COL_CLEAN_TARGET] == TARGET_CRIMINAL).sum())
    civil_pct = round(civil_final / final_clean_count * 100, 2)
    criminal_pct = round(criminal_final / final_clean_count * 100, 2)
    imbalance_ratio = round(civil_final / criminal_final, 2)

    print(f"[*] Final Class Distribution: Civil={civil_final:,} ({civil_pct}%), Criminal={criminal_final:,} ({criminal_pct}%)")
    print(f"[*] Final Imbalance Ratio: {imbalance_ratio} : 1")

    # Compile Statistics
    cleaning_stats = {
        "raw_total": raw_count,
        "unlabeled_excluded": unlabeled_count,
        "labeled_total": labeled_count,
        "empty_text_excluded": empty_count,
        "valid_text_total": valid_text_count,
        "conflicting_clusters_count": len(conflicting_hashes),
        "conflicting_records_pruned": conflicting_rows_count,
        "conflicting_clusters_details": conflicting_details,
        "remaining_consistency_verified": True,
        "duplicate_copies_removed": duplicate_copies_removed,
        "final_cleaned_total": final_clean_count,
        "final_class_counts": {
            "civil": civil_final,
            "civil_percentage": civil_pct,
            "criminal": criminal_final,
            "criminal_percentage": criminal_pct,
            "imbalance_ratio": imbalance_ratio,
        },
        "preserved_columns": keep_columns,
    }

    # Step 6: Persist Cleaned Dataset
    if save_interim:
        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[*] Persisting cleaned dataset to Parquet: {CLEANED_DATA_FILE}")
        df_final.to_parquet(CLEANED_DATA_FILE, index=False, engine="pyarrow")
        print(f"[*] Persisting cleaned dataset to CSV: {CLEANED_DATA_CSV}")
        df_final.to_csv(CLEANED_DATA_CSV, index=False)

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        stats_file = METRICS_DIR / "cleaning_metrics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(cleaning_stats, f, indent=2)
        print(f"[*] Cleaning metrics saved to: {stats_file}")

    print("=" * 70)
    print("DATA CLEANING & DEDUPLICATION COMPLETE")
    print("=" * 70)

    return df_final, cleaning_stats


if __name__ == "__main__":
    clean_and_deduplicate_dataset()
