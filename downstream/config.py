"""
Configuration file for Thematic Sentiment Analysis & Event Study Pipeline

Modify these settings to customize pipeline behavior.
"""

# ============================================================================
# SENTIMENT ANALYSIS SETTINGS
# ============================================================================

# FinBERT Model Configuration
SENTIMENT_MODEL = 'yiyanghkust/finbert-tone'  # HuggingFace model name
BATCH_SIZE = 16  # Number of sentences to process at once (reduce if memory issues)
USE_GPU = False  # Set to True if you have a CUDA-capable GPU

# Sentiment Aggregation
AGGREGATION_STRATEGY = 'simple'  # Currently uses simple count method
# Future options: 'simple', 'weighted', 'confidence_weighted'

# Output Options
OUTPUT_CSV_PER_THEME = True  # Generate separate CSV file for each theme
CSV_OUTPUT_DIR = 'sentiment_analysis_output'  # Directory for theme-specific CSVs

# ============================================================================
# EVENT STUDY SETTINGS
# ============================================================================

# Event Window Parameters (in trading days)
EVENT_WINDOW_START = -10  # Days before earnings call (negative number)
EVENT_WINDOW_END = 10     # Days after earnings call (positive number)

# Estimation Window Parameters
ESTIMATION_WINDOW = 100  # Number of days for estimation period
GAP = 50                 # Gap between estimation window and event window

# Data Quality Requirements
MIN_OBSERVATIONS = 70  # Minimum non-missing return observations required

# Return Model Selection
# Options: 'm' (Market Model), 'ff' (Fama-French 3-Factor),
#          'ffm' (Fama-French + Momentum), 'madj' (Market-Adjusted)
MODEL = 'm'

# Output Format
# Options: 'df' (DataFrame dict), 'csv', 'json', 'xls', 'print'
EVENT_STUDY_OUTPUT_FORMAT = 'csv'

# ============================================================================
# PORTFOLIO SORTS SETTINGS
# ============================================================================

# Portfolio Construction
WEIGHTING = 'value'  # 'value' for value-weighted, 'equal' for equal-weighted
PORTFOLIO_DAYS = 90  # Number of trading days to track portfolio returns

# Sentiment Buckets
NUM_SENTIMENT_BUCKETS = 3  # Number of buckets to sort firms into (default: terciles)
BUCKET_NAMES = ['Low', 'Medium', 'High']  # Names for buckets (low to high sentiment)

# Data Source
USE_LOCAL_CRSP = False  # Set True to use local CSV instead of WRDS query
LOCAL_CRSP_FILE = 'data/crsp_dsf.csv'  # Path to local CRSP data (if USE_LOCAL_CRSP=True)

# ============================================================================
# WRDS CONNECTION SETTINGS
# ============================================================================

# WRDS connection will use .pgpass file by default
# Create ~/.pgpass with: wrds-pgdata.wharton.upenn.edu:9737:wrds:USERNAME:PASSWORD
# Or set these environment variables:
# WRDS_USERNAME, WRDS_PASSWORD

# Connection timeout (seconds)
WRDS_TIMEOUT = 300

# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================

# Logging Level
# Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# Output Directories
RESULTS_DIR = 'results'
EVENT_STUDY_DIR = 'results/event_study'
PORTFOLIO_DIR = 'results/portfolio'
LOGS_DIR = 'logs'

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Parallel Processing
USE_MULTIPROCESSING = False  # Set True for parallel processing (experimental)
NUM_WORKERS = 4              # Number of worker processes (if USE_MULTIPROCESSING=True)

# Memory Management
MAX_MEMORY_GB = 8  # Maximum memory to use (rough limit, not enforced)

# ============================================================================
# ADVANCED SETTINGS (Change only if you know what you're doing)
# ============================================================================

# FinBERT Settings
MAX_SEQUENCE_LENGTH = 512  # BERT maximum sequence length
TRUNCATION = True          # Truncate long sentences

# Event Study Settings
REMOVE_OUTLIERS = False    # Remove extreme returns (experimental)
OUTLIER_THRESHOLD = 0.50   # Threshold for outlier removal (50% daily return)

# Portfolio Sorts Settings
REBALANCE_DAILY = True     # Rebalance portfolios daily vs. buy-and-hold
MIN_FIRMS_PER_BUCKET = 5   # Minimum firms required in each bucket

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    assert EVENT_WINDOW_START < 0, "EVENT_WINDOW_START must be negative"
    assert EVENT_WINDOW_END > 0, "EVENT_WINDOW_END must be positive"
    assert ESTIMATION_WINDOW > 0, "ESTIMATION_WINDOW must be positive"
    assert GAP >= 0, "GAP must be non-negative"
    assert MIN_OBSERVATIONS > 0, "MIN_OBSERVATIONS must be positive"
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert PORTFOLIO_DAYS > 0, "PORTFOLIO_DAYS must be positive"
    assert MODEL in ['m', 'ff', 'ffm', 'madj'], f"Invalid MODEL: {MODEL}"
    assert WEIGHTING in ['value', 'equal'], f"Invalid WEIGHTING: {WEIGHTING}"
    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
