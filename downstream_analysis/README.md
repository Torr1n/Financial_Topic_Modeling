# Thematic Sentiment Analysis & Event Study Pipeline

## Overview

This is a **self-contained pipeline** for analyzing sentiment in earnings call transcripts and conducting event studies on stock returns. The pipeline requires just TWO inputs:

1. **Earnings Call Themes JSON** (themes with firm contributions, sentences, and PERMNOs)
2. **WRDS Connection** (for querying stock returns data)

The pipeline will:
1. **Compute sentiment scores** for each firm-theme combination using FinBERT
2. **Run event studies** by querying WRDS for stock returns and calculating abnormal returns
3. **Generate portfolio sorts** to analyze returns by sentiment terciles

**IMPORTANT: This pipeline requires:**
- Input JSON with PERMNOs already mapped (at the earnings call level, to account for temporal linkdate variations)
- WRDS account with access to CRSP and Fama-French data

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Input Requirements](#input-requirements)
4. [Data Sources](#data-sources)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [File Documentation](#file-documentation)
7. [Output Files](#output-files)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- **WRDS account with access to:**
  - CRSP Daily Stock File (`crsp_a_stock.dsf`)
  - CRSP Daily Stock Index (`crsp_a_stock.dsi`)
  - Fama-French Factors (`ff_all.factors_daily`)
  - CRSP Delisting File (`crsp_a_stock.dsedelist`)

### Installation

```bash
# 1. Install required packages
pip install -r requirements.txt

# 2. Configure WRDS credentials
# Create a .pgpass file in your home directory
# Format: wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD
chmod 600 ~/.pgpass
```

### Running the Pipeline

```bash
# Run the complete pipeline
python run_pipeline.py \
    --themes_file data/earnings_call_themes.json \
    --output_dir results/
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Sentiment Analysis                           │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Input: Themes JSON (with PERMNOs, NO sentiment)   │ │
│  │ Process: FinBERT sentiment scoring                │ │
│  │ Output: Sentiment scores per firm-theme-date      │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Event Study (Batched + Per-Theme Regression) │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Input: Sentiment scores with PERMNOs              │ │
│  │ Data Source: WRDS (Compustat, CRSP, IBES)         │ │
│  │ Process:                                          │ │
│  │   1. Deduplicate events (permno, edate)           │ │
│  │   2. Single batched WRDS query (all themes)       │ │
│  │   3. Calculate covariates + CAR for all events    │ │
│  │   4. Run per-theme OLS regressions                │ │
│  │      (CAR ~ sentiment + 14 covariates)            │ │
│  │ Output:                                           │ │
│  │   - Per-theme regression tables                   │ │
│  │   - Regression significance summary (ranked)      │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Portfolio Sorts                               │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Input: Sentiment scores                           │ │
│  │ Data Source: WRDS (CRSP daily returns)            │ │
│  │ Process: Sort into sentiment terciles             │ │
│  │ Output:                                           │ │
│  │   - Portfolio returns by theme (CSV)              │ │
│  │   - Combined portfolio analysis (all themes)      │ │
│  │   - Time series chart (PNG visualization)         │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Input Requirements

### 1. Earnings Call Themes JSON (with PERMNOs)

**CRITICAL REQUIREMENTS:**
- Contains thematic clustering results BEFORE sentiment analysis
- **Includes PERMNO for each firm contribution** (at earnings call level)
- PERMNOs must be temporally correct (matching linkdates for that specific earnings call date)
- NO sentiment scores included - pipeline computes them

**Format:** JSON file with thematic clustering results

**Required Structure:**
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
            },
            {
              "text": "Our inventory management has improved significantly.",
              "speaker": "CFO"
            }
          ]
        },
        {
          "firm_name": "Microsoft Corporation",
          "permno": 10107,
          "earnings_call_date": "2023-01-25",
          "sentences": [
            {
              "text": "Supply chain constraints are easing across our product lines.",
              "speaker": "CEO"
            }
          ]
        }
      ]
    }
  ]
}
```

**Key Fields:**

- `theme_id`: Unique identifier for theme
- `theme_name`: Descriptive name for the investment theme
- **`permno`** (CRITICAL): CRSP permanent company identifier
  - Must be at the **firm_contribution level** (not theme level)
  - Must be the correct PERMNO for that specific earnings call date
  - Accounts for temporal variations in company linkdates
  - Example: A company acquired in 2023 may have different PERMNO before/after acquisition
- `earnings_call_date`: Date in YYYY-MM-DD format
- `firm_name`: Company name (for logging/reference)
- `sentences`: List of sentence dictionaries
  - `text`: Sentence text (will be scored by FinBERT)
  - `speaker`: Optional speaker role (CEO, CFO, etc.)

**Why PERMNO Must Be at Earnings Call Level:**

PERMNOs can change over time due to:
- Mergers and acquisitions
- Corporate restructuring
- Exchange listing changes

**Example:**
```json
{
  "firm_name": "Company X",
  "permno": 12345,
  "earnings_call_date": "2022-06-15"  // PERMNO 12345 valid on this date
}
{
  "firm_name": "Company X",
  "permno": 98765,
  "earnings_call_date": "2023-06-15"  // PERMNO changed to 98765 after merger
}
```

**How to Generate PERMNOs from WRDS:**

If you have Capital IQ transcripts data:

```sql
-- Query to get PERMNOs matched to earnings call dates
WITH transcript_dates AS (
  SELECT
    t.transcriptid,
    t.companyid,
    c.companyname,
    t.mostimportantdateutc::date as earnings_call_date
  FROM ciq.ciqtranscript t
  JOIN ciq.ciqcompany c ON t.companyid = c.companyid
  WHERE t.mostimportantdateutc >= '2023-01-01'
),
gvkeys AS (
  SELECT
    td.companyid,
    td.companyname,
    td.earnings_call_date,
    wg.gvkey
  FROM transcript_dates td
  JOIN ciq.wrds_gvkey wg ON td.companyid = wg.companyid
),
permnos AS (
  SELECT
    g.companyname,
    g.earnings_call_date,
    ccm.lpermno as permno
  FROM gvkeys g
  JOIN crsp.ccmxpf_linktable ccm ON g.gvkey = ccm.gvkey
  WHERE ccm.linktype IN ('LU', 'LC')
    AND ccm.linkprim IN ('P', 'C')
    AND g.earnings_call_date >= ccm.linkdt
    AND g.earnings_call_date <= COALESCE(ccm.linkenddt, '9999-12-31')
)
SELECT * FROM permnos;
```

This ensures each earnings call gets the correct PERMNO valid on that specific date.

**What Happens to This File:**

1. Pipeline loads themes and firm contributions
2. Extracts sentences and PERMNOs (already provided!)
3. For each firm-theme, FinBERT analyzes all sentences
4. Sentences are scored as positive/negative/neutral
5. Scores are aggregated: `(positive_count - negative_count) / total_count`
6. Result: sentiment score between -1.0 and +1.0, paired with provided PERMNO

---

## Data Sources

### WRDS Data Requirements

The pipeline queries WRDS for the following data:

#### Event Study Stage

**NEW: Batched Event Study Approach**
- The pipeline deduplicates events by (permno, edate) to minimize WRDS API calls
- Single batched query retrieves all data for all unique events across all themes
- Reduces WRDS queries from 20-30 down to ~3 (90% reduction!)
- Handles duplicate events with different sentiment scores per theme correctly

**Compustat Quarterly (`comp_na_daily_all.fundq`):**
- Quarterly financials: EPS, net income, total assets, debt, equity
- Capital expenditures, cash, R&D, PPE
- **Used for:** Calculating 14 control covariates for regression

**Compustat Annual (`comp_na_daily_all.funda`):**
- Annual sales, employees
- **Used for:** Sales growth and employee change covariates

**CRSP Monthly Stock File (`crsp_a_stock.msf`):**
- Monthly returns, prices, shares outstanding
- **Used for:** Stock volatility and return covariates

**IBES Summary Statistics (`tr_ibes.statsum_epsus`):**
- Analyst EPS estimates (mean, median)
- Actual EPS
- **Used for:** Earnings surprise covariate

**CRSP-Compustat Link Table (`crsp.ccmxpf_linktable`):**
- Links PERMNO to GVKEY
- **Used for:** Merging CRSP and Compustat data

**CRSP Stock Names (`crsp.stocknames`):**
- Ticker symbols and company names
- **Used for:** Labeling and validation

**Event Study Module (called by batched event study):**
- CRSP Daily Stock File (`crsp_a_stock.dsf`)
- Fama-French Daily Factors (`ff_all.factors_daily`)
- CRSP Delisting File (`crsp_a_stock.dsedelist`)
- **Used for:** CAR calculation (market-adjusted returns)

#### Portfolio Sorts Stage

**CRSP Daily Stock File (`crsp_a_stock.dsf`):**
- Daily returns for 90 trading days after each earnings call
- Market cap = price × shares outstanding
- **Used for:** Post-event portfolio performance tracking

**All data retrieved directly from WRDS - no local CSV files needed!**

---

## Step-by-Step Guide

### Step 1: Prepare Your Input File

1. **Run your thematic clustering pipeline** on earnings call transcripts
2. **Add PERMNOs to each firm contribution:**
   - Use WRDS linking tables (shown above)
   - Ensure PERMNO is valid on the earnings_call_date
   - Store at firm_contribution level (not theme level)
3. **Save as JSON** in `data/` directory
4. **Verify format:**
   ```python
   import json
   with open('data/themes.json') as f:
       data = json.load(f)
       contrib = data['themes'][0]['firm_contributions'][0]
       assert 'permno' in contrib, "Missing PERMNO!"
       assert 'earnings_call_date' in contrib, "Missing date!"
       assert 'sentences' in contrib, "Missing sentences!"
       print("✓ Format validated")
   ```

### Step 2: Configure WRDS Access

```bash
# Create .pgpass file
echo "wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD" > ~/.pgpass
chmod 600 ~/.pgpass

# Test connection
python -c "import wrds; db = wrds.Connection(); print('✓ Connected'); db.close()"
```

### Step 3: Configure Pipeline Settings

Edit `config.py`:

```python
# Sentiment Analysis
SENTIMENT_MODEL = 'yiyanghkust/finbert-tone'
BATCH_SIZE = 16
USE_GPU = False  # Set True if you have a GPU

# Event Study
EVENT_WINDOW_START = -10
EVENT_WINDOW_END = 10
ESTIMATION_WINDOW = 100
GAP = 50
MIN_OBSERVATIONS = 70
MODEL = 'm'  # 'm'=Market, 'ff'=Fama-French, 'ffm'=FF+Momentum

# Portfolio Sorts
WEIGHTING = 'value'  # 'value' or 'equal'
PORTFOLIO_DAYS = 90
```

### Step 4: Run the Pipeline

**Complete Pipeline:**
```bash
python run_pipeline.py \
    --themes_file data/earnings_call_themes.json \
    --output_dir results/
```

**Individual Stages:**

```bash
# Option 1: Run complete pipeline (all 3 stages)
python run_pipeline.py \
    --themes_file data/earnings_call_themes.json \
    --output_dir results/

# Option 2: Run only Event Study (on existing sentiment scores)
python run_event_study.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/event_study/

# Option 3: Run only Portfolio Sorts (on existing sentiment scores)
python run_portfolio_sorts.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/portfolio_sorts/

# Option 4: Run only Portfolio Sorts with equal weighting
python run_portfolio_sorts.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/portfolio_sorts/ \
    --weighting equal
```

**Note:** Options 2-4 allow you to re-run event studies or portfolio sorts on existing sentiment scores without recomputing sentiment

### Step 5: Analyze Results

Results in timestamped folders: `results/run_YYYYMMDD_HHMMSS/`
- `sentiment_scores.csv`: Firm-level sentiment scores (all themes)
- `event_study/by_theme/`: Per-theme regression results and CAR
- `portfolio_sorts/by_theme/`: Per-theme portfolio returns by sentiment tercile
- `pipeline_summary.txt`: Summary statistics

---

## File Documentation

### Core Pipeline Files

#### 1. `thematic_sentiment_analyzer.py`

**Purpose:** Computes sentiment scores from thematic analysis using FinBERT-tone

**Key Classes:**
- `ThematicSentimentAnalyzer`: Main sentiment analysis class

**Key Methods:**

**`analyze_themes(thematic_output)`**
- **Input:** Thematic analysis JSON dictionary (with PERMNOs, NO sentiment)
- **Process:**
  1. Loads themes and firm contributions
  2. Extracts sentences and PERMNOs from each contribution
  3. Scores each sentence using FinBERT (positive/negative/neutral)
  4. Aggregates sentence scores to firm-theme level
- **Output:** Dictionary with sentiment scores + PERMNOs

**`compute_sentence_sentiments(sentences_data)`**
- **Input:** List of sentence dictionaries with 'text' field
- **Process:** Runs FinBERT in batches to classify sentiment
- **Output:** List of SentimentResult objects (label + confidence)

**`aggregate_firm_theme_sentiment(sentiment_results, sentences_data, strategy)`**
- **Input:** List of sentence-level sentiment results
- **Process:** Counts positive/negative/neutral sentences
- **Formula:**
  ```python
  sentiment_score = (positive_count - negative_count) / total_count
  ```
- **Output:** Single sentiment score from -1.0 to +1.0
  - -1.0 = all sentences negative
  - 0.0 = balanced or neutral
  - +1.0 = all sentences positive

**Dependencies:**
- `transformers`: FinBERT model (yiyanghkust/finbert-tone)
- `torch`: PyTorch backend
- `pandas`, `numpy`: Data manipulation

---

#### 2. `event_study.py` (Batched Event Study)

**Purpose:** Conducts event studies with 14 covariates + sentiment using batched WRDS queries

**Key Classes:**
- `ThematicES`: Main event study class with batched processing

**Architecture Changes:**
- **OLD:** Called once per theme (20-30 WRDS queries)
- **NEW:** Called once for all themes (3 WRDS queries total, 90% reduction!)
- Regression logic moved to pipeline level for per-theme analysis

**Key Methods:**

**`wrdsPull()`**
- Queries WRDS for all unique (permno, edate) combinations
- Single batched SQL query retrieves:
  - Compustat fundamentals (quarterly + annual)
  - CRSP monthly data (for volatility/return covariates)
  - IBES analyst estimates (for earnings surprise)
- Saves intermediate results: `wrdsquery.csv`, `crspquery.csv`

**`calculateFactors()`**
- Computes 14 control covariates:
  1. Return on Assets
  2. Book Leverage
  3. Capital Expenditures
  4. Research & Development
  5. Sales Growth
  6. Firm Size (log sales)
  7. Cash
  8. Asset Tangibility
  9. Delta Employee Change
  10. Stock Volatility (12-month rolling)
  11. Stock Return (3-month buy-and-hold)
  12. Market-to-Book
  13. Earnings Surprise
  14. (Sentiment added at regression stage)

**`calculateCovariatesAndCAR()`**
- Calls `EventStudy` module to compute CAR
- Merges covariates with CAR results
- **Returns DataFrame** (does NOT run regression)
- Handles missing/infinite values with dropna

**`doAll()`**
- Orchestrates: wrdsPull → calculateFactors → calculateCovariatesAndCAR
- Returns complete DataFrame ready for per-theme regression

**Duplicate Event Handling:**
- Same (permno, edate) can appear in multiple themes with different sentiments
- Deduplicates for WRDS query efficiency
- Tracks theme-specific sentiments separately
- Creates multiple regression rows per unique event (one per theme)

**Dependencies:**
- `wrds`: WRDS connection
- `pandas`, `numpy`: Data manipulation
- `event_study_module.py`: CAR calculation (called internally)

---

#### 3. `event_study_module.py`

**Purpose:** Low-level CAR calculations using different methodologies

**Key Classes:**
- `EventStudy`: Market-adjusted return calculation

**Key Methods:**

**`eventstudy(data, model='madj', output='df')`**

- **Input:**
  - `data`: List of events with `permno` (int), `edate` (MM/DD/YYYY)
  - `model`: Return model (default: 'madj' = market-adjusted)
  - `output`: Output format ('df', 'csv', 'json', 'xls')

- **Process:**
  1. Query CRSP daily returns
  2. Calculate market-adjusted returns
  3. Aggregate to CAR

- **Output:** Dictionary with:
  - `event_date`: DataFrame with `permno`, `edate`, `car`, `bhar`

**Event Study Metrics:**
- **Cumulative Abnormal Return (CAR):** Market-adjusted cumulative return
- **Buy-and-Hold Abnormal Return (BHAR):** Compounded abnormal return

**Dependencies:**
- `wrds`: WRDS Python library
- `pandas`, `numpy`: Data manipulation

---

#### 4. `portfolio_sorts.py`

**Purpose:** Sorts firms into sentiment-based portfolios and tracks returns

**Key Classes:**
- `PortfolioSorts`: Portfolio sorting with WRDS data

**Key Methods:**

**`crspreturns()`**
- **Input:** Dictionary of events with `permno`, `edate`, `sentiment`
- **Data Source:** Queries WRDS CRSP or loads from local CSV
- **Process:**
  1. Extract unique PERMNOs
  2. Query daily returns for 90 days after each event
  3. Merge with sentiment scores
  4. Filter to only post-event data
- **Output:** Populates `self.CRSPQuery` DataFrame

**`compute_portfolio_returns()`**
- **Input:** None (uses `self.CRSPQuery`)
- **Process:**
  1. Split firms into sentiment terciles (Low/Medium/High)
  2. For each tercile and each day:
     - Calculate value-weighted return
     - weight = market_cap / Σ(market_cap)
  3. Calculate cumulative returns from day 1
- **Output:** DataFrame with:
  - `bucket`: 'Low', 'Medium', or 'High'
  - `days_from_event`: 0-90
  - `vw_return`: Daily value-weighted return
  - `cumulative_return`: Cumulative return since day 1

**Portfolio Construction:**

**Tercile Assignment:**
- Sort firms by sentiment score each day
- Bottom 33%: Low sentiment
- Middle 33%: Medium sentiment
- Top 33%: High sentiment

**Value Weighting:**
```python
market_cap = |price| × shares_outstanding
weight_i = market_cap_i / Σ(market_cap_bucket)
portfolio_return = Σ(stock_return_i × weight_i)
```

**Cumulative Returns:**
```python
# Day 0 = 0% (event date baseline)
# Day 1+ = compounded returns
cumulative_t = Π(1 + daily_return_s) - 1  for s = 1 to t
```

**Dependencies:**
- `pandas`, `numpy`: Data manipulation
- `pathlib`: File handling

---

## Output Files

### Sentiment Analysis Output

#### `sentiment_scores.csv`

**Description:** Firm-level sentiment scores from FinBERT analysis

**Columns:**
- `permno`: CRSP permanent identifier (from input JSON)
- `edate`: Event date in MM/DD/YYYY format
- `sentiment`: Sentiment score from -1.0 to +1.0
  - Formula: `(positive_count - negative_count) / total_sentences`
- `firm_name`: Company name
- `theme_name`: Associated theme
- `theme_id`: Theme identifier
- `n_sentences`: Number of sentences analyzed

**Example:**
```csv
permno,edate,sentiment,firm_name,theme_name,theme_id,n_sentences
14593,01/28/2023,0.2500,Apple Inc.,Supply Chain,theme_001,20
10107,01/25/2023,-0.1250,Microsoft Corporation,Supply Chain,theme_001,16
12490,02/02/2023,0.4000,Alphabet Inc.,AI Innovation,theme_002,25
```

**Interpretation:**
- Apple: 25% more positive than negative sentences
- Microsoft: 12.5% more negative than positive sentences
- Alphabet: 40% more positive than negative sentences

---

### Event Study Outputs

#### Per-Theme Regression Results

**`{theme_id}_{theme_name}_event_study_results.csv`**

**Description:** Complete dataset used for regression (per theme)

**Columns:**
- `permno`, `edate`, `datadate`: Identifiers and dates
- `ticker`, `comnam`: Company identifiers
- **Covariates (14):**
  - `Return_on_Assets`
  - `Book_Leverage`
  - `Capital_Expenditures`
  - `Research_and_Development`
  - `Sales_Growth`
  - `Firm_Size`
  - `Cash`
  - `Asset_Tangibility`
  - `Delta_Employee_Change`
  - `Stock_Volatility`
  - `Stock_Return`
  - `Market_to_Book`
  - `Earnings_Surprise`
  - `sentiment` (theme-specific)
- **Dependent Variables:**
  - `cret`: Cumulative return
  - `car`: Cumulative abnormal return
  - `bhar`: Buy-and-hold abnormal return

**`{theme_id}_{theme_name}_regression_table.txt`**

**Description:** OLS regression output (statsmodels summary)

**Regression Model:**
```
CAR ~ sentiment + Return_on_Assets + Book_Leverage + Capital_Expenditures +
      Research_and_Development + Sales_Growth + Firm_Size + Cash +
      Asset_Tangibility + Delta_Employee_Change + Stock_Volatility +
      Stock_Return + Market_to_Book + Earnings_Surprise
```

**Example Output:**
```
                            OLS Regression Results
==============================================================================
Dep. Variable:                    car   R-squared:                       0.245
Model:                            OLS   Adj. R-squared:                  0.198
Method:                 Least Squares   F-statistic:                     5.21
...

                         coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------
const                 0.0012      0.005      0.240      0.811      -0.009       0.011
sentiment             0.0234      0.008      2.925      0.004       0.008       0.039
Return_on_Assets      0.0456      0.023      1.983      0.048       0.001       0.090
...
```

#### Regression Significance Summary

**`regression_significance_summary.csv`**

**Description:** All themes ranked by sentiment coefficient p-value

**Columns:**
- `rank`: Ranking (1 = most significant)
- `theme_id`, `theme_name`: Theme identifiers
- `sentiment_coef`: Coefficient on sentiment variable
- `sentiment_pvalue`: P-value for sentiment coefficient
- `significance_level`: '***' (p<0.01), '**' (p<0.05), '*' (p<0.10), '' (not significant)
- `r_squared`: R² of regression
- `n_obs`: Number of observations
- `f_statistic`: F-statistic
- `f_pvalue`: P-value for F-statistic

**`regression_significance_summary.txt`**

**Description:** Formatted table with summary statistics

**Example:**
```
=== Event Study Regression Significance Summary ===

Themes Ranked by Sentiment Coefficient Significance

Rank  Theme ID    Theme Name                    Sentiment Coef  P-value   Sig    R²     N
----  ----------  ----------------------------  --------------  --------  -----  -----  ---
1     theme_003   AI Innovation                     0.0345      0.001    ***    0.312  145
2     theme_007   Supply Chain Resilience           0.0234      0.004    **     0.245  132
3     theme_012   ESG Initiatives                   0.0189      0.023    *      0.198  98
...

Summary Statistics:
- Significant at p < 0.01: 3 themes (15.0%)
- Significant at p < 0.05: 5 themes (25.0%)
- Significant at p < 0.10: 8 themes (40.0%)
```

---

### Portfolio Sorts Output

#### Per-Theme Portfolio Returns

**`{theme_id}_{theme_name}_portfolio_returns.csv`**

**Description:** Value-weighted portfolio returns by sentiment tercile (per theme)

**Columns:**
- `bucket`: 'Low', 'Medium', or 'High'
- `days_from_event`: 0-90 trading days
- `vw_return`: Daily value-weighted return
- `cumulative_return`: Cumulative return since day 1

**Example:**
```csv
bucket,days_from_event,vw_return,cumulative_return
Low,0,0.0000,0.0000
Low,90,-0.0045,-0.0823
High,0,0.0000,0.0000
High,90,0.0015,0.0567
```

**Interpretation:**
- Low sentiment: -8.23% over 90 days
- High sentiment: +5.67% over 90 days
- **Long-Short Spread:** 13.90% (demonstrates sentiment signals alpha!)

#### Combined Portfolio Analysis

**`combined_portfolio_returns.csv`**

**Description:** Aggregated portfolio returns across all themes

**Columns:**
- `bucket`: 'Low', 'Medium', or 'High'
- `days_from_event`: 0-90 trading days
- `vw_return`: Daily value-weighted return
- `cumulative_return`: Cumulative return since day 1

**Use Case:** Understand overall portfolio strategy performance across all themes

#### Portfolio Time Series Chart

**`portfolio_time_series_chart.png`**

**Description:** Visual representation of cumulative returns by sentiment bucket

**Features:**
- High-quality PNG image (300 DPI, publication-ready)
- Three colored lines:
  - Red: Low sentiment portfolio
  - Orange: Medium sentiment portfolio
  - Green: High sentiment portfolio
- X-axis: Days from event (0-90)
- Y-axis: Cumulative return (%)
- Summary statistics in text box:
  - Final cumulative returns for each bucket
  - Long-short spread (High - Low)
- Grid, zero line, and professional formatting

**Example:**
![Portfolio Time Series](docs/example_portfolio_chart.png)

**Interpretation:**
- Visualize portfolio divergence over time
- Identify when sentiment signal kicks in
- Compare low vs high sentiment performance trajectory

---

## Troubleshooting

### Common Issues

#### 1. Missing PERMNO in Themes File

**Problem:** "KeyError: 'permno'" when running pipeline

**Solution:**
```python
# Verify format
import json
with open('data/themes.json') as f:
    data = json.load(f)
    contrib = data['themes'][0]['firm_contributions'][0]
    print('Has permno:', 'permno' in contrib)
    print('Sample permno:', contrib.get('permno'))
```

PERMNOs must be at `firm_contribution` level, not theme level!

#### 2. Invalid PERMNOs

**Problem:** Event study returns no results for certain firms

**Solution:**

Test PERMNO in WRDS:
```python
import wrds
db = wrds.Connection()
query = """
SELECT permno, date, ret
FROM crsp_a_stock.dsf
WHERE permno = 14593
AND date >= '2022-08-01'
AND date <= '2023-03-01'
LIMIT 10
"""
result = db.raw_sql(query)
print("CRSP data available:", not result.empty)
```

#### 3. Temporal PERMNO Mismatches

**Problem:** Some events fail even though PERMNO seems valid

**Cause:** PERMNO may be valid NOW but not on the earnings call date

**Solution:** Ensure PERMNO-to-date matching in input:

```sql
-- Correct way to get temporally-matched PERMNOs
SELECT
  companyname,
  earnings_call_date,
  ccm.lpermno as permno
FROM your_transcripts t
JOIN ciq.wrds_gvkey wg ON t.companyid = wg.companyid
JOIN crsp.ccmxpf_linktable ccm ON wg.gvkey = ccm.gvkey
WHERE ccm.linktype IN ('LU', 'LC')
  AND ccm.linkprim IN ('P', 'C')
  AND earnings_call_date >= ccm.linkdt  -- CRITICAL: Date-aware matching
  AND earnings_call_date <= COALESCE(ccm.linkenddt, '9999-12-31')
```

#### 4. FinBERT Model Errors

**Problem:** "Error loading transformers model"

**Solution:**
```bash
# Reinstall transformers
pip install --upgrade transformers torch

# Download model manually
python -c "from transformers import BertForSequenceClassification; BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')"

# Force CPU mode if GPU issues
# In config.py: USE_GPU = False
```

#### 5. Memory Errors

**Problem:** Out of memory during processing

**Solution:**

Reduce batch size:
```python
# In config.py
BATCH_SIZE = 8  # Instead of 16
```

Process themes in batches:
```python
import json

with open('data/large_themes.json') as f:
    all_themes = json.load(f)

# Process 100 themes at a time
for i in range(0, len(all_themes['themes']), 100):
    batch = {'themes': all_themes['themes'][i:i+100]}
    # Run pipeline on batch...
```

---

## Advanced Usage

### Custom Event Windows

```python
# In config.py, adjust for research questions

# Short-term reaction (-1 to +1)
EVENT_WINDOW_START = -1
EVENT_WINDOW_END = 1

# Long-term drift (-10 to +30)
EVENT_WINDOW_START = -10
EVENT_WINDOW_END = 30
```

### Equal-Weighted Portfolios

```python
# In config.py
WEIGHTING = 'equal'  # Each stock gets equal weight
```

### Fama-French Models

```python
# In config.py
MODEL = 'ff'   # Fama-French 3-factor
MODEL = 'ffm'  # Fama-French + Momentum
MODEL = 'madj' # Market-adjusted (no regression)
```

---

## Summary of Data Flow

```
INPUT FILE (YOU PROVIDE)
└── earnings_call_themes.json
    ├── Contains: themes, firm contributions, PERMNOs, sentences
    ├── PERMNOs: At earnings call level (temporally matched)
    └── NO sentiment scores yet

↓

STAGE 1: SENTIMENT ANALYSIS
├── Load themes with PERMNOs (already provided!)
├── Run FinBERT on sentences
├── Aggregate to firm-theme level
└── Output: sentiment_scores.csv

↓

STAGE 2: EVENT STUDY
├── Read sentiment_scores.csv (has PERMNOs)
├── Query WRDS CRSP data:
│   ├── crsp_a_stock.dsf (returns)
│   ├── ff_all.factors_daily (factors)
│   └── crsp_a_stock.dsedelist (delistings)
├── Run OLS regressions
├── Calculate abnormal returns
└── Outputs: event_stats.csv, event_window.csv, event_date.csv

↓

STAGE 3: PORTFOLIO SORTS
├── Read sentiment_scores.csv
├── Query WRDS for 90-day returns
├── Sort into sentiment terciles
├── Calculate value-weighted returns
└── Output: portfolio_returns.csv

FINAL OUTPUTS
├── sentiment_scores.csv
├── event_stats.csv
├── event_window.csv
├── event_date.csv
└── portfolio_returns.csv
```

**KEY SIMPLIFICATION:** PERMNOs are already in the input JSON (temporally matched to earnings call dates). No PERMNO lookup needed in pipeline!



# Usage Guide: Running Individual Pipeline Components


This guide shows how to run different parts of the pipeline independently.

## Important Update: Batched Event Study

**The pipeline now uses a batched event study approach:**
- Single WRDS query for all unique (permno, edate) combinations across all themes
- Reduces WRDS API calls by ~90% (from 20-30 queries to ~3)
- Handles duplicate events with different sentiment scores per theme correctly
- Runs per-theme OLS regressions at pipeline level after batched data retrieval

## Typical Workflow

### 1. Run Complete Pipeline (First Time)

Run all three stages together to get initial results:

```bash
python run_pipeline.py \
    --themes_file data/themes_with_permnos.json \
    --output_dir results/
```

**Output:** `results/run_YYYYMMDD_HHMMSS/` containing:
- `sentiment_scores.csv` (all themes)
- `sentiment_analysis_output/` (per-theme details)
- `event_study/by_theme/` (per-theme regressions + results)
- `regression_significance_summary.csv` and `.txt` (ranked themes)
- `portfolio_sorts/by_theme/` (per-theme returns)
- `combined_portfolio_returns.csv` (aggregated across themes)
- `portfolio_time_series_chart.png` (visual chart)
- `pipeline_summary.txt`

---

### 2. Re-run Event Study Only (Skip Sentiment)

If you want to re-run event studies on the same sentiment scores (e.g., different time window or covariates):

```bash
python run_event_study.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/event_study_rerun/
```

**Why?**
- Sentiment computation is slow (FinBERT inference)
- Event study is faster (just WRDS queries + regression)
- Allows experimentation with event study parameters

**Output:** `results/event_study_rerun/run_YYYYMMDD_HHMMSS/`
- `by_theme/`
  - `{theme_id}_{theme_name}_event_study_results.csv` (regression data)
  - `{theme_id}_{theme_name}_regression_table.txt` (OLS output)
- `regression_significance_summary.csv` (ranked themes)
- `regression_significance_summary.txt` (formatted summary)

---

### 3. Re-run Portfolio Sorts Only (Skip Sentiment)

If you want to try different portfolio weighting schemes:

```bash
# Value-weighted (default)
python run_portfolio_sorts.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/portfolio_value_weighted/

# Equal-weighted
python run_portfolio_sorts.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/portfolio_equal_weighted/ \
    --weighting equal
```

**Why?**
- Compare value-weighted vs equal-weighted returns
- Re-run on different time periods
- Test different sentiment tercile cutoffs (would require code modification)

**Output:** `results/portfolio_*/run_YYYYMMDD_HHMMSS/`
- `by_theme/`
  - `{theme_id}_{theme_name}_portfolio_returns.csv` (per-theme)
- `combined_portfolio_returns.csv` (aggregated across all themes)
- `portfolio_time_series_chart.png` (visual chart, 300 DPI PNG)

---

## Command Line Flags

### `run_pipeline.py` (Complete Pipeline)

```bash
python run_pipeline.py \
    --themes_file <path>         # Required: Input JSON with themes+PERMNOs
    --output_dir <path>          # Optional: Base output dir (default: results/)
    --skip_sentiment             # Skip sentiment analysis (NOT IMPLEMENTED YET)
    --skip_event_study           # Skip event study stage
    --skip_portfolio             # Skip portfolio sorts stage
```

### `run_event_study.py` (Event Study Only)

```bash
python run_event_study.py \
    --sentiment_file <path>      # Required: Path to sentiment_scores.csv
    --output_dir <path>          # Optional: Output dir (default: results/event_study)
```

### `run_portfolio_sorts.py` (Portfolio Sorts Only)

```bash
python run_portfolio_sorts.py \
    --sentiment_file <path>      # Required: Path to sentiment_scores.csv
    --output_dir <path>          # Optional: Output dir (default: results/portfolio_sorts)
    --weighting {value,equal}    # Optional: Weighting method (default: value)
```

---

## Example Workflows

### Workflow A: Full Pipeline, Then Experiment with Weighting

```bash
# Step 1: Run complete pipeline
python run_pipeline.py --themes_file data/themes_with_permnos.json --output_dir results/

# Step 2: Check which run to use
ls -lt results/  # Find latest: run_20260102_154500

# Step 3: Try equal-weighted portfolios
python run_portfolio_sorts.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/portfolio_equal_weighted/ \
    --weighting equal

# Step 4: Compare results
# - results/run_20260102_154500/portfolio_sorts/by_theme/  (value-weighted)
# - results/portfolio_equal_weighted/run_*/by_theme/       (equal-weighted)
```

### Workflow B: Multiple Event Studies with Different Configurations

```bash
# Step 1: Run pipeline once to get sentiment
python run_pipeline.py --themes_file data/themes_with_permnos.json --output_dir results/

# Step 2: Run event study with default settings
python run_event_study.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/event_study_default/

# Step 3: Modify event_study.py to use different window or covariates
# (requires code changes in pipeline/event_study.py)

# Step 4: Re-run event study with modified settings
python run_event_study.py \
    --sentiment_file results/run_20260102_154500/sentiment_scores.csv \
    --output_dir results/event_study_modified/

# Step 5: Compare regression results
# - results/event_study_default/run_*/by_theme/*_regression_table.txt
# - results/event_study_modified/run_*/by_theme/*_regression_table.txt
```

---

## Tips

1. **Timestamped folders preserve all runs** - You never lose previous results
2. **Use descriptive output_dir names** - Helps identify what parameters were used
3. **sentiment_scores.csv is portable** - Can copy to different machines and re-run analysis
4. **Check theme_id and theme_name columns** - Scripts auto-detect and group by theme
5. **WRDS connection required** - Both standalone scripts still need WRDS access for data

---

## Troubleshooting

**Q: Can I run sentiment analysis only?**
A: There's currently no standalone script for this. Use `run_pipeline.py` and then ignore event_study/portfolio_sorts folders.

**Q: How do I know which sentiment_scores.csv file to use?**
A: Check `results/` for timestamped run folders. Use the most recent or the one with your desired sentiment computation.

**Q: Can I change the sentiment tercile cutoffs for portfolio sorts?**
A: Yes, but requires modifying `pipeline/portfolio_sorts.py` (currently uses 33rd/67th percentiles hardcoded).

**Q: Can I run on a subset of themes?**
A: Yes - filter `sentiment_scores.csv` to specific theme_ids before passing to standalone scripts.


---

## Credits

This pipeline implements:
- **Sentiment Analysis:** FinBERT-tone for financial text classification
- **Event Studies:** Batched approach with 14 covariates + sentiment regression
- **Portfolio Sorts:** Sentiment-based long-short strategy evaluation

Designed for academic research in finance and NLP.

---

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and recent updates.

**Current Version:** 2.0.0 (January 2026)

**Recent Major Updates:**
- Batched event study approach (90% reduction in WRDS API calls)
- Regression significance summary across all themes
- Professional portfolio time series charts (PNG visualization)

**Last Updated:** January 3, 2026
