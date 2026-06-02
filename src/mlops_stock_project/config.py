from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models Directory
MODEL_DIR = PROJECT_ROOT / "models"

# Reports Directory

REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"

# Multi-Stock Configuration
TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "AMD",
    "NFLX",
    "INTC",
]

# Default Dataset Paths
COMBINED_RAW_DATA_FILE = (
    RAW_DATA_DIR / "combined_stock_data.csv"
)

PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR / "stock_features.csv"
)

# Model Artifact
MODEL_FILE = MODEL_DIR / "model_v1.pkl"