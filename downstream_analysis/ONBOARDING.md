# Downstream Analysis Onboarding Guide

## What This Does

This pipeline takes **earnings call themes** (from the upstream topic modeling) and answers a research question:

> **Does the sentiment of what executives say about a theme predict stock returns?**

## The Three Stages

```
themes_with_permnos.json
         │
         ▼
┌─────────────────────────────────┐
│  STAGE 1: Sentiment Analysis    │  No credentials needed
│  (FinBERT scores sentences)     │
│  Output: sentiment_scores.csv   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  STAGE 2: Event Study           │  Requires WRDS
│  (Does sentiment predict CAR?)  │
│  Output: regression tables      │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  STAGE 3: Portfolio Sorts       │  Requires WRDS
│  (Trading strategy backtest)    │
│  Output: portfolio returns      │
└─────────────────────────────────┘
```

## Stage-by-Stage Breakdown

### Stage 1: Sentiment Analysis
- **What it does**: Runs FinBERT (a finance-tuned BERT model) on every sentence from earnings calls
- **How it scores**: Each sentence gets labeled positive/negative/neutral, then aggregated per firm-theme as `(positive_count - negative_count) / total_count`
- **Output**: `sentiment_scores.csv` with columns: `permno, edate, sentiment, firm_name, theme_id`
- **No external data needed** - runs entirely locally

### Stage 2: Event Study
- **What it does**: Tests if sentiment predicts abnormal stock returns around earnings calls
- **Key concept - CAR**: Cumulative Abnormal Return = actual return minus expected return (based on market model)
- **The regression**: `CAR ~ sentiment + 14 control variables`
- **Control variables**: ROA, leverage, capex, R&D, sales growth, firm size, cash, tangibility, employee change, stock volatility, stock return, market-to-book, earnings surprise
- **Output**: Per-theme regression tables showing if sentiment coefficient is significant

### Stage 3: Portfolio Sorts
- **What it does**: Simulates a trading strategy based on sentiment
- **Method**: Sort firms into terciles (Low/Medium/High sentiment), track returns for 90 days
- **Output**: Cumulative return chart showing if high-sentiment stocks outperform low-sentiment stocks

## What is WRDS?

**WRDS (Wharton Research Data Services)** is the standard academic finance database. It contains:
- **CRSP**: Stock prices and returns for all US public companies
- **Compustat**: Company financials (balance sheets, income statements)
- **IBES**: Analyst earnings estimates
- **Fama-French**: Market risk factors

You need university access to use it. Create an account at: https://wrds-www.wharton.upenn.edu

## Setup

### 1. Install Dependencies
```bash
cd downstream_analysis
pip install -r requirements.txt
```

### 2. Configure WRDS (for Stages 2 & 3)

Option A - Create `~/.pgpass` file:
```
wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD
```
Then run: `chmod 600 ~/.pgpass`

Option B - Environment variables:
```bash
export WRDS_USERNAME=your_username
export WRDS_PASSWORD=your_password
```

### 3. Prepare Input Data

Your input JSON must have this structure:
```json
{
  "themes": [
    {
      "theme_id": "theme_001",
      "theme_name": "Supply Chain Resilience",
      "firm_contributions": [
        {
          "firm_id": "AAPL",
          "firm_name": "Apple Inc.",
          "permno": 14593,
          "earnings_call_date": "2023-01-28",
          "sentences": [
            {"text": "We've diversified our supply chain...", "speaker": "CEO"}
          ]
        }
      ]
    }
  ]
}
```

**Critical fields**:
- `permno`: CRSP identifier (must be correct for the date - companies change PERMNOs during M&A)
- `earnings_call_date`: The actual date of the earnings call
- `sentences.text`: The text to analyze for sentiment

## Running the Pipeline

### Full Pipeline (all 3 stages)
```bash
python run_pipeline.py \
    --themes_file ../data/themes_for_sentiment_with_permnos.json \
    --output_dir results/
```

### Sentiment Only (no WRDS needed)
```bash
python run_pipeline.py \
    --themes_file ../data/themes_for_sentiment_with_permnos.json \
    --output_dir results/ \
    --skip_event_study \
    --skip_portfolio
```

### Event Study Only (on existing sentiment scores)
```bash
python run_event_study.py \
    --sentiment_file results/run_YYYYMMDD/sentiment_scores.csv \
    --output_dir results/event_study/
```

### Portfolio Sorts Only
```bash
python run_portfolio_sorts.py \
    --sentiment_file results/run_YYYYMMDD/sentiment_scores.csv \
    --output_dir results/portfolio/
```

## Output Structure

```
results/run_20260119_153000/
├── sentiment_scores.csv              # All firm-theme sentiment scores
├── sentiment_analysis_output/        # Per-theme sentiment CSVs
├── event_study/
│   ├── by_theme/                     # Per-theme regression results
│   │   ├── theme_001_..._regression_table.txt
│   │   └── theme_001_..._event_study_results.csv
│   ├── regression_significance_summary.csv
│   └── regression_significance_summary.txt
├── portfolio_sorts/
│   ├── by_theme/                     # Per-theme portfolio returns
│   ├── combined_all_themes_portfolio_returns.csv
│   └── portfolio_time_series_chart.png
└── pipeline_summary.txt
```

## Interpreting Results

### Event Study Results
Look at `regression_significance_summary.txt`:
- **Sentiment coefficient > 0**: Positive sentiment → higher returns
- **p-value < 0.05**: Statistically significant relationship
- **R²**: How much variance the model explains

### Portfolio Results
Look at `portfolio_time_series_chart.png`:
- If High sentiment line > Low sentiment line → sentiment predicts returns
- The spread between High and Low is your potential trading profit

## File Reference

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Main orchestrator - runs all 3 stages |
| `config.py` | Configuration settings |
| `pipeline/thematic_sentiment_analyzer.py` | Stage 1: FinBERT sentiment |
| `pipeline/event_study.py` | Stage 2: Covariates + regression |
| `pipeline/event_study_module.py` | Stage 2: Core CAR calculation |
| `pipeline/portfolio_sorts.py` | Stage 3: Portfolio construction |

## Common Issues

### "No WRDS connection"
- Check your `~/.pgpass` file or environment variables
- Make sure you have an active WRDS account

### "No PERMNO mapping for firm X"
- Your input JSON is missing the `permno` field for that firm
- PERMNOs must be looked up from WRDS for the specific date (they change during M&A)

### "Event study returned no results"
- The firm may not have enough trading data around the earnings call date
- WRDS requires 70+ days of returns in the estimation window

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        run_pipeline.py                           │
│  - Loads themes JSON                                             │
│  - Orchestrates 3 stages                                         │
│  - Manages WRDS connection (shared across stages)                │
│  - Generates summary reports                                     │
└──────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ thematic_       │  │ event_study.py  │  │ portfolio_      │
│ sentiment_      │  │ + event_study_  │  │ sorts.py        │
│ analyzer.py     │  │   module.py     │  │                 │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ - Load FinBERT  │  │ - Query WRDS    │  │ - Query CRSP    │
│ - Score each    │  │ - Calc 14 covs  │  │ - Sort terciles │
│   sentence      │  │ - Compute CAR   │  │ - Track returns │
│ - Aggregate to  │  │ - Run OLS per   │  │ - Calc cumul.   │
│   firm-theme    │  │   theme         │  │   returns       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```
