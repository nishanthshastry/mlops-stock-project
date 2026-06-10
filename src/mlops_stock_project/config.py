from pathlib import Path

# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# DATA DIRECTORIES
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


# MODEL DIRECTORY
MODEL_DIR = PROJECT_ROOT / "models"


# REPORT DIRECTORIES
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_MONITORING_DIR = REPORTS_DIR / "monitoring"

REPORTS_BACKTESTING_DIR = REPORTS_DIR / "backtesting"

REPORTS_EXPLAINABILITY_DIR = REPORTS_DIR / "explainability"


# MLFLOW
MLFLOW_DB_FILE = PROJECT_ROOT / "mlflow.db"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"


# DVC
DVC_DIR = PROJECT_ROOT / ".dvc"


# API CONFIG
API_HOST = "0.0.0.0"

API_PORT = 8080


# GLOBAL CONFIG
RANDOM_STATE = 42

TEST_SIZE = 0.2

N_SPLITS = 5


# DATA COLLECTION CONFIG
START_DATE = "2015-01-01"

END_DATE = None

# MULTI-SECTOR STOCK UNIVERSE
#
# Expanded beyond technology stocks to improve
# generalization and address sector concentration bias.
#
# Professor feedback:
# "The model may simply be learning that tech follows
# the broader market."
#

TECH_STOCKS = [
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

HEALTHCARE_STOCKS = [
    "JNJ",
    "PFE",
    "MRK",
    "ABBV",
]

FINANCIAL_STOCKS = [
    "JPM",
    "BAC",
    "GS",
]

CONSUMER_STOCKS = [
    "WMT",
    "COST",
    "PG",
]

ENERGY_STOCKS = [
    "XOM",
    "CVX",
]

# Master stock universe
TICKERS = (
    TECH_STOCKS + HEALTHCARE_STOCKS + FINANCIAL_STOCKS + CONSUMER_STOCKS + ENERGY_STOCKS
)


# MARKET INDICATORS
MARKET_INDICATORS = [
    "SPY",
    "QQQ",
    "^VIX",
]


# DATASET FILES
COMBINED_RAW_DATA_FILE = RAW_DATA_DIR / "combined_stock_data.csv"

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "stock_features.csv"


# MODEL FILES
MODEL_FILE = MODEL_DIR / "model_v1.pkl"

MODEL_BACKUP_FILE = MODEL_DIR / "model_backup.pkl"


# MONITORING CONFIG
DRIFT_PSI_THRESHOLD = 0.25

DRIFT_KS_THRESHOLD = 0.05

RETRAIN_SEVERITY_THRESHOLD = 0.35


# BACKTEST CONFIG
INITIAL_CAPITAL = 10000

TRANSACTION_COST = 0.001

TRADING_DAYS_PER_YEAR = 252


# CREATE DIRECTORIES
DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    REPORTS_DIR,
    REPORTS_FIGURES_DIR,
    REPORTS_MONITORING_DIR,
    REPORTS_BACKTESTING_DIR,
    REPORTS_EXPLAINABILITY_DIR,
    MLRUNS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

TARGET_RETURN_THRESHOLD = 0.01
