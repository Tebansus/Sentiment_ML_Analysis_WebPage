import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Environment
ENV = os.getenv("ENV", "dev")  # 'dev', 'test', or 'prod'

# Database configurations
DB_CONFIG = {
    "dev": {
        "host": os.getenv("DEV_DB_HOST", "localhost"),
        "port": os.getenv("DEV_DB_PORT", "5432"),
        "database": os.getenv("DEV_DB_NAME", "sentiment_analysis_dev"),
        "user": os.getenv("DEV_DB_USER", "postgres"),
        "password": os.getenv("DEV_DB_PASSWORD", "postgres"),
    },
    "test": {
        "host": os.getenv("TEST_DB_HOST", "localhost"),
        "port": os.getenv("TEST_DB_PORT", "5432"),
        "database": os.getenv("TEST_DB_NAME", "sentiment_analysis_test"),
        "user": os.getenv("TEST_DB_USER", "postgres"),
        "password": os.getenv("TEST_DB_PASSWORD", "postgres"),
    },
    "prod": {
        "host": os.getenv("PROD_DB_HOST", "localhost"),
        "port": os.getenv("PROD_DB_PORT", "5432"),
        "database": os.getenv("PROD_DB_NAME", "sentiment_analysis_prod"),
        "user": os.getenv("PROD_DB_USER", "postgres"),
        "password": os.getenv("PROD_DB_PASSWORD", ""),
    },
}

# API configurations
API_CONFIG = {
    "dev": {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "debug": True,
    },
    "test": {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,
        "debug": False,
    },
    "prod": {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,
        "debug": False,
    },
}

# Twitter API credentials
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# Alpha Vantage API credentials
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# ML Model settings
MODEL_DIR = BASE_DIR / "models"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
SEQUENCE_LENGTH = 15  # Lookback window size for LSTM
PREDICTION_WINDOWS = [1, 3, 7]  # Days to predict ahead

# Stocks to track
TRACKED_STOCKS = ["INTC", "AMD", "NVDA"]

# Data collection settings
DATA_COLLECTION_INTERVAL = int(os.getenv("DATA_COLLECTION_INTERVAL", "60"))  # minutes

# Get active configuration based on environment
def get_db_config():
    return DB_CONFIG[ENV]

def get_api_config():
    return API_CONFIG[ENV] 