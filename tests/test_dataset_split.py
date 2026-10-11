"""
Tests for dataset splits, stratification proportions, and test freezing.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    COL_CLEAN_TARGET,
    COL_CLEAN_TEXT,
    COL_ID,
    COL_TEXT_HASH,
    FROZEN_TEST_IDS_FILE,
    SPLIT_METADATA_FILE,
    TEST_FILE,
    TRAIN_FILE,
    VAL_FILE,
)


def test_processed_splits_exist():
    assert TRAIN_FILE.exists(), f"Train file not found: {TRAIN_FILE}"
    assert VAL_FILE.exists(), f"Val file not found: {VAL_FILE}"
    assert TEST_FILE.exists(), f"Test file not found: {TEST_FILE}"
    assert FROZEN_TEST_IDS_FILE.exists(), f"Frozen test file not found: {FROZEN_TEST_IDS_FILE}"
    assert SPLIT_METADATA_FILE.exists(), f"Split metadata not found: {SPLIT_METADATA_FILE}"


def test_split_sizes_and_columns():
    df_train = pd.read_parquet(TRAIN_FILE)
    df_val = pd.read_parquet(VAL_FILE)
    df_test = pd.read_parquet(TEST_FILE)

    total_records = len(df_train) + len(df_val) + len(df_test)
    assert total_records == 33717, f"Expected 33,717 records, got {total_records}"
    assert len(df_train) == 26973
    assert len(df_val) == 3372
    assert len(df_test) == 3372

    expected_cols = {COL_ID, COL_TEXT_HASH, COL_CLEAN_TEXT, COL_CLEAN_TARGET}
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        assert expected_cols.issubset(set(df.columns)), f"Missing columns in {name}"
        assert df[COL_CLEAN_TEXT].isna().sum() == 0, f"Null text in {name}"
        assert (df[COL_CLEAN_TEXT].str.strip() == "").sum() == 0, f"Empty text in {name}"


def test_zero_contamination_across_splits():
    df_train = pd.read_parquet(TRAIN_FILE)
    df_val = pd.read_parquet(VAL_FILE)
    df_test = pd.read_parquet(TEST_FILE)

    train_hashes = set(df_train[COL_TEXT_HASH])
    val_hashes = set(df_val[COL_TEXT_HASH])
    test_hashes = set(df_test[COL_TEXT_HASH])

    assert len(train_hashes.intersection(val_hashes)) == 0, "Train-Val text overlap detected!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Train-Test text overlap detected!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Val-Test text overlap detected!"


def test_frozen_test_set_integrity():
    with open(FROZEN_TEST_IDS_FILE, "r", encoding="utf-8") as f:
        frozen_data = json.load(f)

    df_test = pd.read_parquet(TEST_FILE)
    frozen_hashes = set(frozen_data["frozen_test_hashes"])
    actual_test_hashes = set(df_test[COL_TEXT_HASH])

    assert frozen_data["test_size"] == 3372
    assert frozen_hashes == actual_test_hashes, "Frozen test hashes do not match test partition!"
