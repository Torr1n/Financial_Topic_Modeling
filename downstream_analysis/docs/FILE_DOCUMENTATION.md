# File Documentation

## Core Pipeline Files

### `thematic_sentiment_analyzer.py`

#### Purpose
Analyzes sentiment of thematic clusters using FinBERT-tone model.

#### Input
- **Thematic analysis JSON** with themes, firm contributions (including PERMNOs), and sentences
- Format:
  ```json
  {
    "themes": [
      {
        "theme_id": "theme_001",
        "firm_contributions": [
          {
            "firm_name": "Apple Inc.",
            "permno": 14593,
            "earnings_call_date": "2023-01-28",
            "sentences": [{"text": "...", "speaker": "CEO"}]
          }
        ]
      }
    ]
  }
  ```

#### Output
- **sentiment_scores.csv** with columns:
  - `permno`: CRSP identifier
  - `edate`: Event date (MM/DD/YYYY)
  - `sentiment`: Score from -1.0 to +1.0
  - `firm_name`, `theme_name`, `theme_id`, `n_sentences`

#### Key Functions

**`analyze_themes(thematic_output, aggregation_strategy, permno_mapping, output_csv, csv_directory)`**
- Main entry point for sentiment analysis
- Processes all themes and returns organized results
- Can output separate CSV files per theme if `output_csv=True`

**`compute_sentence_sentiments(sentences_data)`**
- Runs FinBERT on list of sentences
- Returns SentimentResult objects with label + confidence

**`aggregate_firm_theme_sentiment(sentiment_results, sentences_data, strategy)`**
- Aggregates sentence-level sentiments to firm-theme level
- Formula: `(positive_count - negative_count) / total_count`

#### Dependencies
- `transformers` (FinBERT model)
- `torch` (PyTorch)
- `pandas`, `numpy`

---

### `event_study.py` (NEW: Batched Approach)

#### Purpose
Conducts batched event studies with 14 covariates + sentiment using optimized WRDS queries.

#### Architecture Update
- **OLD:** Called once per theme (20-30 WRDS queries)
- **NEW:** Called once for all themes (3 WRDS queries total, 90% reduction!)
- Regression logic moved to `run_pipeline.py` and `run_event_study.py`

#### Key Classes
- `ThematicES`: Main event study class with batched processing

#### Input
- **List of events** across ALL themes:
  ```python
  [
    {
      "permno": 14593,
      "edate": "01/28/2023",
      "sentiment": 0  # Placeholder, actual sentiment merged later per theme
    },
    ...
  ]
  ```

#### Output
DataFrame with:
- `permno`, `edate`, `datadate`: Identifiers and dates
- `ticker`, `comnam`: Company info
- **14 Covariates:** ROA, leverage, capex, R&D, sales growth, firm size, cash, tangibility, employee change, stock volatility, stock return, MTB, earnings surprise
- **CAR Metrics:** `cret`, `car`, `bhar`
- `sentiment`: Merged at pipeline level per theme

#### Key Methods

**`wrdsPull()`**
- Single batched SQL query for all unique (permno, edate) combinations
- Queries:
  - Compustat Quarterly (`fundq`): Quarterly financials
  - Compustat Annual (`funda`): Annual sales, employees
  - CRSP Monthly (`msf`): Price, returns, shares for volatility calculation
  - IBES (`statsum_epsus`): Analyst estimates
  - Linking tables: `ccmxpf_linktable`, `stocknames`
- Saves intermediate results: `wrdsquery.csv`, `crspquery.csv`

**`calculateFactors()`**
- Computes 14 control covariates:
  1. Return_on_Assets = net_income / total_assets
  2. Book_Leverage = (debt + liabilities) / (debt + liabilities + equity)
  3. Capital_Expenditures = capex / total_assets
  4. Research_and_Development = R&D / total_assets
  5. Sales_Growth = sales / sales_lagged
  6. Firm_Size = log(sales)
  7. Cash = cash / total_assets
  8. Asset_Tangibility = PPE / total_assets
  9. Delta_Employee_Change = Δ(employees) / total_assets
  10. Stock_Volatility = sqrt(12-month rolling sum of squared returns)
  11. Stock_Return = 3-month buy-and-hold return
  12. Market_to_Book = (market_cap + debt - cash) / total_assets
  13. Earnings_Surprise = (actual - median_estimate) × price

**`calculateCovariatesAndCAR()`**
- Calls `EventStudy` module to compute CAR
- Merges covariates with CAR
- **Returns DataFrame** (does NOT run regression)
- Handles NaN/infinite values with dropna

**`doAll()`**
- Orchestrates full pipeline: wrdsPull → calculateFactors → calculateCovariatesAndCAR
- Returns complete DataFrame ready for per-theme regression

#### Duplicate Event Handling
- Same (permno, edate) can appear in multiple themes with different sentiments
- Solution:
  1. Deduplicate for WRDS query (save API calls)
  2. Track theme-specific sentiments separately
  3. Merge theme-specific sentiment after batched processing
  4. Creates multiple regression rows per unique event (one per theme)

#### Dependencies
- `wrds`: WRDS connection
- `pandas`, `numpy`: Data manipulation
- `event_study_module.py`: CAR calculation (called internally)
- `statsmodels`: NOT used here (moved to pipeline level)

---

### `event_study_module.py`

#### Purpose
Low-level CAR calculation using market-adjusted returns (called by `event_study.py`).

#### Input
- **List of events** with format:
  ```python
  [
    {
      "permno": 14593,
      "edate": "01/28/2023"  # MM/DD/YYYY format
    },
    ...
  ]
  ```

#### Output
Dictionary with one key DataFrame:
- **`event_date`**: Cumulative returns at event date
  - Columns: `permno`, `edate`, `cret`, `car`, `bhar`

#### Key Functions

**`eventstudy(data, model='madj', output='df')`**

**Parameters:**
- `data`: List of events (see above)
- `model`: Return model (default: 'madj' = market-adjusted)
  - `'madj'`: Market-Adjusted (most commonly used in pipeline)
  - `'m'`: Market Model
  - `'ff'`: Fama-French 3-Factor
  - `'ffm'`: Fama-French + Momentum
- `output`: 'df' (default)

**WRDS Queries Executed:**
- `crsp_a_stock.dsf`: Daily stock returns
- `crsp_a_stock.dsi`: Trading calendar
- `ff_all.factors_daily`: Fama-French factors (for market return)
- `crsp_a_stock.dsedelist`: Delisting returns

#### Metrics Calculated

**Cumulative Abnormal Return (CAR):**
```
AR_t = Actual_Return_t - Market_Return_t  (market-adjusted)
CAR = Σ AR_t  (summed over event window)
```

**Buy-and-Hold Abnormal Return (BHAR):**
```
BHAR = Π(1 + Actual_t) - Π(1 + Market_t)
```

#### Dependencies
- `wrds` (WRDS connection)
- `pandas`, `numpy`

---

### `portfolio_sorts.py`

#### Purpose
Sorts firms into sentiment-based portfolios and tracks returns over 90 days.

#### Input
- **Dictionary of events** with `permno`, `edate`, `sentiment`
- **CRSP data** (either from WRDS or local CSV)

#### Output
**`portfolio_returns.csv`** with columns:
- `bucket`: 'Low', 'Medium', or 'High' sentiment
- `days_from_event`: 0 to 90 trading days
- `vw_return`: Daily value-weighted return
- `cumulative_return`: Cumulative return since day 1

#### Key Functions

**`crspreturns()`**
- Loads CRSP returns data from WRDS or local CSV
- Merges with sentiment scores
- Filters to 90 days post-event
- Calculates `days_from_event` for each observation

**`compute_portfolio_returns()`**
- Splits firms into sentiment terciles each day
- Calculates value-weighted returns:
  ```python
  weight_i = market_cap_i / Σ(market_cap_bucket)
  portfolio_return = Σ(return_i × weight_i)
  ```
- Computes cumulative returns starting from day 1

#### Portfolio Construction

**Tercile Assignment (Daily):**
- Bottom 33% of sentiment scores → Low bucket
- Middle 33% → Medium bucket
- Top 33% → High bucket

**Value Weighting:**
```python
market_cap = |price| × shares_outstanding
weight_i = market_cap_i / Σ(market_cap_all_firms_in_bucket)
```

**Cumulative Return Calculation:**
```python
# Day 0 = event date = 0% return (baseline)
# Day 1+ = compounded returns
cumulative_return_t = Π(1 + daily_return_s) - 1  for s = 1 to t
```

#### Dependencies
- `pandas`, `numpy`
- `pathlib`

---

## Helper Files

### `permno_mapper.py` (DEPRECATED for this pipeline)

This file is included for reference but is NOT used in the simplified pipeline since PERMNOs are already in the input JSON.

If you need to generate PERMNO mappings from Capital IQ data, see the SQL queries in README.md.

---

## Configuration Files

### `config.py`

Central configuration for all pipeline parameters.

**Sentiment Analysis:**
```python
SENTIMENT_MODEL = 'yiyanghkust/finbert-tone'
BATCH_SIZE = 16
USE_GPU = False
```

**Event Study:**
```python
EVENT_WINDOW_START = -10
EVENT_WINDOW_END = 10
ESTIMATION_WINDOW = 100
GAP = 50
MIN_OBSERVATIONS = 70
MODEL = 'm'  # 'm', 'ff', 'ffm', 'madj'
```

**Portfolio Sorts:**
```python
WEIGHTING = 'value'  # 'value' or 'equal'
PORTFOLIO_DAYS = 90
```

---

## Runner Scripts

### `run_pipeline.py`

Main orchestrator that runs all three stages sequentially with batched event study.

**Usage:**
```bash
python run_pipeline.py \
    --themes_file data/earnings_call_themes.json \
    --output_dir results/
```

**Process:**
1. Validates input file format
2. Runs sentiment analysis → `sentiment_scores.csv`
3. Runs batched event study:
   - Deduplicates events by (permno, edate)
   - Single WRDS query for all themes (90% reduction in API calls)
   - Calculates 14 covariates + CAR
   - Runs per-theme OLS regressions
   - Generates regression significance summary
4. Runs portfolio sorts:
   - Per-theme portfolio returns
   - Combined portfolio analysis
   - Time series chart (PNG)
5. Generates summary report

**Key Functions:**

**`run_batched_event_study(all_events, wrds_conn)`**
- Runs event study ONCE for all events across all themes
- Returns DataFrame with covariates + CAR for all unique events

**`run_regression_for_theme(theme_id, theme_name, theme_results_df, results_dir)`**
- Runs OLS regression: `CAR ~ sentiment + 14 covariates`
- Saves regression table (.txt) and results (.csv)
- Returns RegressionResult object

**`create_regression_significance_summary(regression_results, output_dir)`**
- Ranks themes by sentiment coefficient p-value
- Generates CSV and TXT summaries
- Shows statistical significance levels

**`create_portfolio_time_series_chart(combined_portfolio, output_dir)`**
- Creates professional PNG chart (300 DPI)
- Three colored lines for Low/Medium/High sentiment
- Summary statistics embedded in chart

---

### Stage-Specific Runners

**`run_event_study.py`** (Standalone Event Study)
```bash
python run_event_study.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/event_study/
```

**Features:**
- Uses same batched event study approach as main pipeline
- Includes `run_batched_event_study()`, `run_regression_for_theme()`, `create_regression_significance_summary()`
- Outputs:
  - Per-theme regression results and tables
  - Regression significance summary (CSV + TXT)

**`run_portfolio_sorts.py`** (Standalone Portfolio Sorts)
```bash
python run_portfolio_sorts.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/portfolio_sorts/ \
    --weighting value  # or 'equal'
```

**Features:**
- Includes `create_portfolio_time_series_chart()` function
- Outputs:
  - Per-theme portfolio returns
  - Combined portfolio returns (aggregated)
  - Portfolio time series chart (PNG)

---

## Data Files

### Input: `earnings_call_themes.json`

Required format:
```json
{
  "themes": [
    {
      "theme_id": "theme_001",
      "theme_name": "Supply Chain Resilience",
      "firm_contributions": [
        {
          "firm_name": "Apple Inc.",
          "permno": 14593,
          "earnings_call_date": "2023-01-28",
          "sentences": [
            {
              "text": "We continue to navigate supply chain challenges effectively.",
              "speaker": "CEO"
            }
          ]
        }
      ]
    }
  ]
}
```

**Critical fields:**
- `permno`: Must be at firm_contribution level (not theme level)
- `earnings_call_date`: YYYY-MM-DD format
- `sentences`: List with at least one sentence containing `text` field

### Output Files

See README.md section "Output Files" for detailed documentation of all output formats.

---

## Error Handling

### Common Errors and Solutions

**KeyError: 'permno'**
- Cause: PERMNO not in firm_contribution
- Solution: Add PERMNO at firm_contribution level in JSON

**"No CRSP data for PERMNO XXXXX"**
- Cause: Invalid PERMNO or insufficient trading history
- Solution: Verify PERMNO in WRDS, check temporal matching

**"Out of memory"**
- Cause: Large theme file or high batch size
- Solution: Reduce BATCH_SIZE in config.py or process themes in batches

**"WRDS connection timeout"**
- Cause: Large query or network issues
- Solution: Process smaller batches, check network/firewall

---

## Testing

### Validate Input Format

```python
import json

with open('data/themes.json') as f:
    data = json.load(f)

# Check structure
assert 'themes' in data
theme = data['themes'][0]
assert 'theme_id' in theme
assert 'firm_contributions' in theme

# Check firm contribution
contrib = theme['firm_contributions'][0]
assert 'permno' in contrib, "Missing PERMNO!"
assert 'earnings_call_date' in contrib
assert 'sentences' in contrib
assert len(contrib['sentences']) > 0

print("✓ Input format validated")
```

### Test WRDS Connection

```python
import wrds

db = wrds.Connection()
print("✓ Connected to WRDS")

# Test CRSP access
result = db.raw_sql("SELECT COUNT(*) FROM crsp_a_stock.dsf LIMIT 1")
print("✓ Can access CRSP")

# Test Fama-French access
result = db.raw_sql("SELECT COUNT(*) FROM ff_all.factors_daily LIMIT 1")
print("✓ Can access Fama-French factors")

db.close()
```

### Test FinBERT Model

```python
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

# Test sentence
test_text = "Our revenue grew significantly this quarter."
inputs = tokenizer(test_text, return_tensors='pt')
outputs = model(**inputs)

print("✓ FinBERT model loaded successfully")
```

---

## Performance Optimization

### Batch Processing

For large datasets (1000+ firms):

```python
# Process themes in batches
import json

with open('data/large_themes.json') as f:
    all_data = json.load(f)

batch_size = 100
themes = all_data['themes']

for i in range(0, len(themes), batch_size):
    batch = themes[i:i+batch_size]
    batch_data = {'themes': batch}

    # Process batch...
    print(f"Processed batch {i//batch_size + 1}/{len(themes)//batch_size + 1}")
```

### GPU Acceleration

```python
# In config.py
USE_GPU = True  # If you have CUDA-capable GPU

# Check GPU availability
import torch
print("CUDA available:", torch.cuda.is_available())
```

### Caching WRDS Queries

```python
# Save CRSP data locally for repeated experiments
import pandas as pd

# First run: Download and save
crsp_data = portfolio_sorter.CRSPQuery
crsp_data.to_csv('data/crsp_cache.csv', index=False)

# Subsequent runs: Load from file
crsp_data = pd.read_csv('data/crsp_cache.csv')
```

---

## Additional Notes

- All dates in input JSON should be YYYY-MM-DD format
- Event study expects dates in MM/DD/YYYY format (conversion handled automatically)
- PERMNOs must be integers
- Sentiment scores will be in range [-1.0, +1.0]
- Event study requires ~170 trading days of history per event
- Portfolio sorts require 90 trading days of post-event data
