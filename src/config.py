"""
LegalLens Configuration & Constants
Defines reproducible random seeds, project directory paths, column names,
and file destinations for processed data and split partitions.
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "case_files_total.csv"
CLEANED_DATA_FILE = INTERIM_DATA_DIR / "cleaned_judgments.parquet"
CLEANED_DATA_CSV = INTERIM_DATA_DIR / "cleaned_judgments.csv"

# Split Destination Files
TRAIN_FILE = PROCESSED_DATA_DIR / "train.parquet"
VAL_FILE = PROCESSED_DATA_DIR / "val.parquet"
TEST_FILE = PROCESSED_DATA_DIR / "test.parquet"

TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

FROZEN_TEST_IDS_FILE = PROCESSED_DATA_DIR / "frozen_test_ids.json"
SPLIT_METADATA_FILE = PROCESSED_DATA_DIR / "split_metadata.json"

# Output Directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"

MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_FILE = MODELS_DIR / "final_linear_svm_pipeline.joblib"
DOCS_DIR = PROJECT_ROOT / "docs"

# Reproducibility
RANDOM_SEED = 42

# Column Constants in Jud-IPL
COL_ID = "name"
COL_INDEX = "Unnamed: 0"
COL_CATEGORY = "case_category"
COL_TYPE = "case_type"
COL_INFO = "case_info"
COL_TEXT = "judgement"
COL_LABEL = "label"
COL_PROOF = "proof_sentence"
COL_TOKENS = "tokens"
COL_SENTENCES = "sentences"

# Modeling Clean Columns
COL_CLEAN_TEXT = "text"
COL_CLEAN_TARGET = "category"
COL_TEXT_HASH = "text_hash"

# Valid Target Labels
TARGET_CIVIL = "civil"
TARGET_CRIMINAL = "criminal"
VALID_TARGETS = [TARGET_CIVIL, TARGET_CRIMINAL]

# Split Ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
