from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models directory
MODEL_DIR = PROJECT_ROOT / "models"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Default file paths
RAW_DATA_FILE = RAW_DATA_DIR / "AAPL.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "AAPL_features.csv"

# Model artifact
MODEL_FILE = MODEL_DIR / "model_v1.pkl"