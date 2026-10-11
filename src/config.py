"""
LegalLens Configuration & Constants
Defines reproducible random seeds, project directory paths, and column names.
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

# Output Directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"

MODELS_DIR = PROJECT_ROOT / "models"
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

# Valid Target Labels
TARGET_CIVIL = "civil"
TARGET_CRIMINAL = "criminal"
VALID_TARGETS = [TARGET_CIVIL, TARGET_CRIMINAL]
