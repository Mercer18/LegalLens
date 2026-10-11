"""
LegalLens Milestone 3: Stratified Dataset Splitting & Freezing Module
Splits the deduplicated Jud-IPL corpus into a contamination-free,
stratified 80/10/10 Train, Validation, and Test partition.
Freezes and persists test set membership to guarantee zero test-leakage across future experiments.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEANED_DATA_FILE,
    COL_CLEAN_TARGET,
    COL_CLEAN_TEXT,
    COL_ID,
    COL_TEXT_HASH,
    FROZEN_TEST_IDS_FILE,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SPLIT_METADATA_FILE,
    TARGET_CIVIL,
    TARGET_CRIMINAL,
    TEST_CSV,
    TEST_FILE,
    TEST_RATIO,
    TRAIN_CSV,
    TRAIN_FILE,
    TRAIN_RATIO,
    VAL_CSV,
    VAL_FILE,
    VAL_RATIO,
)


def create_stratified_split(
    cleaned_data_path: Path = CLEANED_DATA_FILE,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Partitions the cleaned deduplicated corpus into stratified 80/10/10 splits.
    Persists train/val/test partitions, freezes test set membership, and saves verification metadata.
    """
    print("=" * 70)
    print("LEGAL LENS: MILESTONE 3 STRATIFIED DATASET SPLIT & FREEZING")
    print("=" * 70)

    if not cleaned_data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_data_path}. Run preprocessing first.")

    print(f"[*] Loading cleaned dataset from: {cleaned_data_path}")
    df = pd.read_parquet(cleaned_data_path)
    total_samples = len(df)
    print(f"[*] Total clean deduplicated judgments to split: {total_samples:,}")

    # Ensure required columns are present
    required_cols = [COL_ID, COL_TEXT_HASH, COL_CLEAN_TEXT, COL_CLEAN_TARGET]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' not found in cleaned dataset.")

    # Step 1: Stratified split into Train (80%) and Temp (20%)
    temp_ratio = VAL_RATIO + TEST_RATIO  # 0.20
    print(f"[*] Splitting Train ({int(TRAIN_RATIO*100)}%) vs Temp ({int(temp_ratio*100)}%) with seed {random_state}...")
    df_train, df_temp = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df[COL_CLEAN_TARGET],
        random_state=random_state,
        shuffle=True,
    )

    # Step 2: Stratified split of Temp into Val (10% of total) and Test (10% of total)
    # Since Temp is 20% of total, a 50/50 split of Temp yields 10% and 10%
    val_share_of_temp = VAL_RATIO / temp_ratio  # 0.50
    print(f"[*] Splitting Temp into Validation (50%) and Test (50%) with seed {random_state}...")
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_share_of_temp),
        stratify=df_temp[COL_CLEAN_TARGET],
        random_state=random_state,
        shuffle=True,
    )

    train_count = len(df_train)
    val_count = len(df_val)
    test_count = len(df_test)

    # Step 3: Zero-Contamination Verification Checks
    print("[*] Running zero-contamination mathematical verification...")
    # 1. Total row conservation
    assert train_count + val_count + test_count == total_samples, "Split sizes do not sum to total samples!"

    # 2. Text hash disjointness (Zero duplicate leak across partitions)
    train_hashes = set(df_train[COL_TEXT_HASH])
    val_hashes = set(df_val[COL_TEXT_HASH])
    test_hashes = set(df_test[COL_TEXT_HASH])

    leak_train_val = len(train_hashes.intersection(val_hashes))
    leak_train_test = len(train_hashes.intersection(test_hashes))
    leak_val_test = len(val_hashes.intersection(test_hashes))

    assert leak_train_val == 0, f"Contamination detected! {leak_train_val} shared texts between Train and Val!"
    assert leak_train_test == 0, f"Contamination detected! {leak_train_test} shared texts between Train and Test!"
    assert leak_val_test == 0, f"Contamination detected! {leak_val_test} shared texts between Val and Test!"
    print("[*] VERIFIED: Zero text hash overlap across Train, Validation, and Test partitions.")

    # 3. ID disjointness
    train_ids = set(df_train[COL_ID])
    val_ids = set(df_val[COL_ID])
    test_ids = set(df_test[COL_ID])

    assert len(train_ids.intersection(val_ids)) == 0, "Shared IDs between Train and Val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Shared IDs between Train and Test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Shared IDs between Val and Test!"
    print("[*] VERIFIED: Zero document ID overlap across Train, Validation, and Test partitions.")

    # Class distribution helper
    def get_distribution(subset: pd.DataFrame) -> Dict[str, Any]:
        c_count = int((subset[COL_CLEAN_TARGET] == TARGET_CIVIL).sum())
        k_count = int((subset[COL_CLEAN_TARGET] == TARGET_CRIMINAL).sum())
        total = len(subset)
        return {
            "total": total,
            "civil_count": c_count,
            "civil_pct": round(c_count / total * 100, 2),
            "criminal_count": k_count,
            "criminal_pct": round(k_count / total * 100, 2),
            "imbalance_ratio": round(c_count / k_count, 2) if k_count > 0 else 0,
        }

    train_dist = get_distribution(df_train)
    val_dist = get_distribution(df_val)
    test_dist = get_distribution(df_test)
    total_dist = get_distribution(df)

    print(f"[*] Train partition: {train_count:,} ({train_count/total_samples*100:.2f}%) | "
          f"Civil: {train_dist['civil_count']:,} ({train_dist['civil_pct']}%), Crim: {train_dist['criminal_count']:,} ({train_dist['criminal_pct']}%)")
    print(f"[*] Val partition:   {val_count:,} ({val_count/total_samples*100:.2f}%) | "
          f"Civil: {val_dist['civil_count']:,} ({val_dist['civil_pct']}%), Crim: {val_dist['criminal_count']:,} ({val_dist['criminal_pct']}%)")
    print(f"[*] Test partition:  {test_count:,} ({test_count/total_samples*100:.2f}%) | "
          f"Civil: {test_dist['civil_count']:,} ({test_dist['civil_pct']}%), Crim: {test_dist['criminal_count']:,} ({test_dist['criminal_pct']}%)")

    # Step 4: Persist Frozen Test Set Membership
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    frozen_test_payload = {
        "description": "LegalLens Frozen Test Set Membership (Milestone 3)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_state,
        "test_size": test_count,
        "test_proportion": round(test_count / total_samples, 4),
        "test_class_distribution": test_dist,
        "frozen_test_ids": sorted(list(test_ids)),
        "frozen_test_hashes": sorted(list(test_hashes)),
    }

    with open(FROZEN_TEST_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(frozen_test_payload, f, indent=2)
    print(f"[*] Frozen test set membership saved to: {FROZEN_TEST_IDS_FILE}")

    # Step 5: Persist Split Parquet & CSV Files
    print(f"[*] Saving Train partition to Parquet ({TRAIN_FILE}) and CSV ({TRAIN_CSV})...")
    df_train.to_parquet(TRAIN_FILE, index=False, engine="pyarrow")
    df_train.to_csv(TRAIN_CSV, index=False)

    print(f"[*] Saving Validation partition to Parquet ({VAL_FILE}) and CSV ({VAL_CSV})...")
    df_val.to_parquet(VAL_FILE, index=False, engine="pyarrow")
    df_val.to_csv(VAL_CSV, index=False)

    print(f"[*] Saving Test partition to Parquet ({TEST_FILE}) and CSV ({TEST_CSV})...")
    df_test.to_parquet(TEST_FILE, index=False, engine="pyarrow")
    df_test.to_csv(TEST_CSV, index=False)

    # Step 6: Compile Metadata & Metrics
    split_summary = {
        "dataset_name": "Jud-IPL Clean Deduplicated Corpus",
        "random_seed": random_state,
        "split_ratios_target": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "total_samples": total_samples,
        "contamination_checks": {
            "train_val_hash_overlap": leak_train_val,
            "train_test_hash_overlap": leak_train_test,
            "val_test_hash_overlap": leak_val_test,
            "is_clean_and_disjoint": True,
        },
        "distributions": {
            "total": total_dist,
            "train": train_dist,
            "val": val_dist,
            "test": test_dist,
        },
        "frozen_test_file": str(FROZEN_TEST_IDS_FILE.name),
        "saved_files": {
            "train": [str(TRAIN_FILE.name), str(TRAIN_CSV.name)],
            "val": [str(VAL_FILE.name), str(VAL_CSV.name)],
            "test": [str(TEST_FILE.name), str(TEST_CSV.name)],
        }
    }

    with open(SPLIT_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    print(f"[*] Split metadata saved to: {SPLIT_METADATA_FILE}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "split_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    print(f"[*] Split metrics saved to: {metrics_path}")

    print("=" * 70)
    print("STRATIFIED DATASET SPLIT & FREEZING COMPLETE")
    print("=" * 70)

    return df_train, df_val, df_test, split_summary


if __name__ == "__main__":
    create_stratified_split()
