from pathlib import Path

# =========================================================
# Base paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

EDA_REPORT_DIR = REPORTS_DIR / "eda"
MODEL_REPORT_DIR = REPORTS_DIR / "model"


# =========================================================
# Files
# =========================================================

DATA_PATH = RAW_DATA_DIR / "fraud_oracle.csv"

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
PREPROCESSED_DATA_PATH = ARTIFACTS_DIR / "preprocessed_data.pkl"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
RAW_FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "raw_feature_columns.json"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


# =========================================================
# Project settings
# =========================================================

TARGET = "FraudFound_P"

ID_COLUMNS = [
    "PolicyNumber"
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


# =========================================================
# Modeling settings
# =========================================================

DEFAULT_THRESHOLD = 0.5

RISK_LEVEL_THRESHOLDS = {
    "low": 0.30,
    "medium": 0.70
}


# =========================================================
# Create required folders
# =========================================================

for folder in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    EDA_REPORT_DIR,
    MODEL_REPORT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)