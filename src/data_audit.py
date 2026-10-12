"""
LegalLens Dataset Audit Module (Milestone 1)
Audits the raw Jud-IPL dataset for schema compliance, label distribution,
duplicates, missing values, document length distributions, and leakage risks.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    COL_CATEGORY,
    COL_ID,
    COL_INDEX,
    COL_INFO,
    COL_LABEL,
    COL_PROOF,
    COL_SENTENCES,
    COL_TEXT,
    COL_TOKENS,
    COL_TYPE,
    FIGURES_DIR,
    METRICS_DIR,
    RAW_DATA_FILE,
    TARGET_CIVIL,
    TARGET_CRIMINAL,
    VALID_TARGETS,
)


def compute_text_hash(text: Any) -> str:
    """Computes SHA-256 hash of a string, returning empty string for null/empty."""
    if pd.isna(text):
        return ""
    text_str = str(text).strip()
    if not text_str:
        return ""
    return hashlib.sha256(text_str.encode("utf-8", errors="replace")).hexdigest()


def run_dataset_audit(data_path: Path = RAW_DATA_FILE) -> Dict[str, Any]:
    """
    Executes a comprehensive, memory-conscious audit of the raw dataset.
    Returns audit metrics dictionary and generates summary figures.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {data_path}")

    file_size_bytes = data_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"[*] Starting audit on: {data_path}")
    print(f"[*] File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

    # Phase 1: Load metadata columns first to check schema, duplicates, and labels
    meta_cols = [COL_INDEX, COL_ID, COL_CATEGORY, COL_TYPE, COL_INFO, COL_LABEL, COL_PROOF]
    print("[*] Reading metadata columns...")
    df_meta = pd.read_csv(data_path, usecols=lambda c: c != COL_TEXT and c != COL_TOKENS and c != COL_SENTENCES)
    raw_row_count = len(df_meta)
    print(f"[*] Total raw rows: {raw_row_count:,}")

    # Inspect all column names in the CSV
    sample_df = pd.read_csv(data_path, nrows=1)
    all_columns = sample_df.columns.tolist()

    # Column datatypes from meta
    col_dtypes = {col: str(df_meta[col].dtype) for col in df_meta.columns}
    col_dtypes[COL_TEXT] = "object"

    # Analyze row 0 (which was observed to have NaNs in earlier sample)
    row_0_all_nan = bool(df_meta.iloc[0].drop(labels=[COL_INDEX], errors="ignore").isna().all())

    # ID Analysis
    id_series = df_meta[COL_ID]
    id_null_count = int(id_series.isna().sum())
    unique_ids = int(id_series.dropna().nunique())
    duplicate_id_count = int(id_series.dropna().duplicated().sum())

    # case_category Analysis
    cat_series = df_meta[COL_CATEGORY].astype(str).str.strip().str.lower()
    # Replace 'nan' string back with NaN
    cat_series = cat_series.replace({"nan": np.nan, "none": np.nan, "": np.nan})

    raw_category_counts_raw = df_meta[COL_CATEGORY].value_counts(dropna=False).to_dict()
    # Format dict keys to string
    raw_category_counts = {str(k): int(v) for k, v in raw_category_counts_raw.items()}

    civil_count = int((cat_series == TARGET_CIVIL).sum())
    criminal_count = int((cat_series == TARGET_CRIMINAL).sum())
    undetermined_count = int((cat_series == "undetermined").sum())
    missing_category_count = int(cat_series.isna().sum())

    known_labels = {TARGET_CIVIL, TARGET_CRIMINAL, "undetermined"}
    other_invalid_mask = cat_series.notna() & (~cat_series.isin(known_labels))
    other_invalid_counts = cat_series[other_invalid_mask].value_counts().to_dict()

    # Primary Dataset Filtering Metrics
    valid_binary_mask = cat_series.isin(VALID_TARGETS)
    usable_rows = int(valid_binary_mask.sum())
    excluded_rows = raw_row_count - usable_rows

    civil_pct = (civil_count / usable_rows * 100) if usable_rows > 0 else 0
    criminal_pct = (criminal_count / usable_rows * 100) if usable_rows > 0 else 0
    imbalance_ratio = (civil_count / criminal_count) if criminal_count > 0 else 0

    print(f"[*] Category breakdown: Civil={civil_count:,}, Criminal={criminal_count:,}, "
          f"Undetermined={undetermined_count:,}, Missing={missing_category_count:,}")
    print(f"[*] Usable Civil/Criminal: {usable_rows:,} (Civil: {civil_pct:.2f}%, Criminal: {criminal_pct:.2f}%)")

    # Phase 2: Stream judgment text in chunks to evaluate lengths, text duplicates, and empty strings
    print("[*] Streaming 'judgement' text for duplicate hashing and length statistics...")
    chunk_size = 5000
    text_hashes = []
    char_lengths = []
    word_lengths = []
    empty_text_indices = []
    short_text_indices = []  # < 50 words
    extremely_short_indices = []  # < 10 words

    chunk_idx = 0
    # Also track representative examples
    rep_examples: Dict[str, Dict[str, Any]] = {
        "civil": {},
        "criminal": {},
        "undetermined": {},
        "missing_category": {},
    }

    for chunk in pd.read_csv(data_path, usecols=[COL_INDEX, COL_ID, COL_CATEGORY, COL_TEXT], chunksize=chunk_size):
        for idx, row in chunk.iterrows():
            c_val = str(row[COL_CATEGORY]).strip().lower() if pd.notna(row[COL_CATEGORY]) else "missing"
            t_val = row[COL_TEXT]
            r_id = row[COL_ID]

            # Populate representative example if not yet filled
            if c_val == "civil" and not rep_examples["civil"] and pd.notna(t_val) and len(str(t_val).split()) > 100:
                rep_examples["civil"] = {
                    "id": str(r_id),
                    "category": "civil",
                    "preview": str(t_val)[:400].replace("\n", " ") + "...",
                    "word_count": len(str(t_val).split())
                }
            elif c_val == "criminal" and not rep_examples["criminal"] and pd.notna(t_val) and len(str(t_val).split()) > 100:
                rep_examples["criminal"] = {
                    "id": str(r_id),
                    "category": "criminal",
                    "preview": str(t_val)[:400].replace("\n", " ") + "...",
                    "word_count": len(str(t_val).split())
                }
            elif c_val == "undetermined" and not rep_examples["undetermined"] and pd.notna(t_val):
                rep_examples["undetermined"] = {
                    "id": str(r_id),
                    "category": "undetermined",
                    "preview": str(t_val)[:400].replace("\n", " ") + "...",
                    "word_count": len(str(t_val).split())
                }
            elif c_val == "missing" and not rep_examples["missing_category"] and pd.notna(t_val):
                rep_examples["missing_category"] = {
                    "id": str(r_id),
                    "category": "missing",
                    "preview": str(t_val)[:400].replace("\n", " ") + "...",
                    "word_count": len(str(t_val).split())
                }

            if pd.isna(t_val) or not str(t_val).strip():
                empty_text_indices.append(int(idx))
                text_hashes.append("")
                char_lengths.append(0)
                word_lengths.append(0)
            else:
                s = str(t_val).strip()
                c_len = len(s)
                w_len = len(s.split())
                h = hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()

                text_hashes.append(h)
                char_lengths.append(c_len)
                word_lengths.append(w_len)

                if w_len < 10:
                    extremely_short_indices.append(int(idx))
                if w_len < 50:
                    short_text_indices.append(int(idx))

        chunk_idx += 1
        print(f"    ... processed chunk {chunk_idx * chunk_size:,} rows")

    # Document Length Statistics (excluding empty texts)
    non_zero_words = [w for w in word_lengths if w > 0]
    non_zero_chars = [c for c in char_lengths if c > 0]

    word_stats = {
        "min": int(np.min(non_zero_words)) if non_zero_words else 0,
        "median": float(np.median(non_zero_words)) if non_zero_words else 0,
        "mean": float(np.mean(non_zero_words)) if non_zero_words else 0,
        "p90": float(np.percentile(non_zero_words, 90)) if non_zero_words else 0,
        "p95": float(np.percentile(non_zero_words, 95)) if non_zero_words else 0,
        "max": int(np.max(non_zero_words)) if non_zero_words else 0,
    }

    char_stats = {
        "min": int(np.min(non_zero_chars)) if non_zero_chars else 0,
        "median": float(np.median(non_zero_chars)) if non_zero_chars else 0,
        "mean": float(np.mean(non_zero_chars)) if non_zero_chars else 0,
        "p90": float(np.percentile(non_zero_chars, 90)) if non_zero_chars else 0,
        "p95": float(np.percentile(non_zero_chars, 95)) if non_zero_chars else 0,
        "max": int(np.max(non_zero_chars)) if non_zero_chars else 0,
    }

    # Text Duplicate Analysis
    hash_series = pd.Series([h for h in text_hashes if h])
    unique_text_count = int(hash_series.nunique())
    duplicate_text_count = int(hash_series.duplicated().sum())

    # Cross-Category Text Duplication Check (same text assigned to different categories)
    df_eval = pd.DataFrame({
        "hash": text_hashes,
        "category": cat_series,
        "id": df_meta[COL_ID]
    })
    # Filter to non-empty hashes and valid civil/criminal cases
    valid_hashes = df_eval[df_eval["hash"] != ""]
    cat_per_hash = valid_hashes.groupby("hash")["category"].nunique()
    conflicting_hashes = cat_per_hash[cat_per_hash > 1].count()

    # Exact Duplicate Rows Check across all metadata columns
    exact_meta_dupes = int(df_meta.duplicated().sum())

    # Leakage Analysis
    # Check what columns exist and what unique values they take
    leakage_columns = {}
    for col in [COL_TYPE, COL_LABEL, COL_INFO, COL_PROOF]:
        if col in df_meta.columns:
            n_unique = int(df_meta[col].nunique(dropna=True))
            top_vals = df_meta[col].value_counts(dropna=False).head(5).to_dict()
            leakage_columns[col] = {
                "unique_count": n_unique,
                "null_count": int(df_meta[col].isna().sum()),
                "sample_values": {str(k): int(v) for k, v in top_vals.items()}
            }

    # Cross-tabulation between case_type and case_category
    case_type_crosstab = {}
    if COL_TYPE in df_meta.columns:
        ct_tab = pd.crosstab(df_meta[COL_TYPE].fillna("MISSING"), cat_series.fillna("MISSING"))
        case_type_crosstab = ct_tab.head(10).to_dict()

    audit_summary = {
        "dataset_file": str(data_path.name),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "raw_row_count": raw_row_count,
        "all_columns": all_columns,
        "column_datatypes": col_dtypes,
        "row_0_all_nan": row_0_all_nan,
        "id_audit": {
            "total_ids": len(id_series),
            "null_ids": id_null_count,
            "unique_ids": unique_ids,
            "duplicate_ids": duplicate_id_count,
        },
        "category_audit": {
            "raw_value_counts": raw_category_counts,
            "civil_count": civil_count,
            "criminal_count": criminal_count,
            "undetermined_count": undetermined_count,
            "missing_count": missing_category_count,
            "other_invalid_counts": other_invalid_counts,
        },
        "primary_dataset_construction": {
            "usable_binary_rows": usable_rows,
            "excluded_rows": excluded_rows,
            "exclusion_breakdown": {
                "row_0_nan_artifact": 1 if row_0_all_nan else 0,
                "undetermined": undetermined_count,
                "missing_category": missing_category_count - (1 if row_0_all_nan else 0),
                "other_invalid": sum(other_invalid_counts.values()),
            },
            "civil_count": civil_count,
            "civil_percentage": round(civil_pct, 2),
            "criminal_count": criminal_count,
            "criminal_percentage": round(criminal_pct, 2),
            "imbalance_ratio_civil_to_criminal": round(imbalance_ratio, 2),
        },
        "text_quality_audit": {
            "empty_text_count": len(empty_text_indices),
            "extremely_short_less_10_words": len(extremely_short_indices),
            "short_less_50_words": len(short_text_indices),
            "unique_judgment_texts": unique_text_count,
            "duplicate_judgment_texts": duplicate_text_count,
            "conflicting_cross_category_texts": int(conflicting_hashes),
        },
        "document_length_statistics": {
            "words": word_stats,
            "characters": char_stats,
        },
        "representative_examples": rep_examples,
        "leakage_audit": {
            "columns_inspected": leakage_columns,
            "type_category_crosstab_sample": case_type_crosstab,
        },
    }

    # Save metrics JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = METRICS_DIR / "dataset_audit_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"[*] Audit metrics saved to {metrics_file}")

    # Generate Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_audit_figures(audit_summary, word_lengths)

    return audit_summary


def generate_audit_figures(summary: Dict[str, Any], word_lengths: List[int]) -> None:
    """Generates visual charts for the dataset audit."""
    print("[*] Generating audit visualizations...")

    # Figure 1: Class Distribution
    plt.figure(figsize=(8, 5))
    cats = ["Civil", "Criminal", "Undetermined", "Missing"]
    ca = summary["category_audit"]
    counts = [ca["civil_count"], ca["criminal_count"], ca["undetermined_count"], ca["missing_count"]]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"]

    bars = plt.bar(cats, counts, color=colors, edgecolor="black", alpha=0.85)
    plt.title("Jud-IPL Raw Class Distribution (Case Category)", fontsize=13, fontweight="bold")
    plt.ylabel("Number of Documents", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 500, f"{h:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "raw_class_distribution.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"[*] Saved figure: {fig1_path}")

    # Figure 2: Document Length Distribution (Word count up to 95th percentile for readable visualization)
    non_zero_words = [w for w in word_lengths if w > 0]
    p95 = summary["document_length_statistics"]["words"]["p95"]
    capped_words = [min(w, p95 * 1.5) for w in non_zero_words]

    plt.figure(figsize=(9, 5))
    plt.hist(capped_words, bins=50, color="#1b9e77", edgecolor="black", alpha=0.8)
    median_val = summary["document_length_statistics"]["words"]["median"]
    mean_val = summary["document_length_statistics"]["words"]["mean"]

    plt.axvline(median_val, color="red", linestyle="--", linewidth=2, label=f"Median ({median_val:,.0f} words)")
    plt.axvline(mean_val, color="blue", linestyle=":", linewidth=2, label=f"Mean ({mean_val:,.0f} words)")
    plt.axvline(p95, color="orange", linestyle="-.", linewidth=2, label=f"95th Pct ({p95:,.0f} words)")

    plt.title("Distribution of Judgment Length (Word Counts)", fontsize=13, fontweight="bold")
    plt.xlabel("Word Count per Judgment (capped at 1.5x 95th percentile)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "document_length_distribution.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"[*] Saved figure: {fig2_path}")


if __name__ == "__main__":
    run_dataset_audit()
