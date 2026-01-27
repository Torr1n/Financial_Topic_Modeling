# Downstream Analysis

Analyzes earnings call themes to test if sentiment predicts stock returns. Takes themes from the cloud pipeline and runs sentiment analysis, event studies, and portfolio sorts.

## What It Does

1. **Sentiment Analysis** - Uses FinBERT to score the sentiment of each firm's contribution to a theme
2. **Event Study** - Calculates abnormal returns (CAR) around earnings calls and regresses on sentiment
3. **Portfolio Sorts** - Groups firms into Low/Medium/High sentiment buckets and tracks cumulative returns

## Requirements

- Python 3.8+
- WRDS account (for event study and portfolio sorts)

```bash
pip install -r requirements.txt
```

## WRDS Credentials

Create a `.env` file in the project root:

```
WRDS_USERNAME=your_username
WRDS_PASSWORD=your_password
```

Or create `~/.wrds/wrds.cfg`:

```ini
[wrds]
wrds_username=your_username
```

## Usage

### Full Pipeline

```bash
# Run all stages (sentiment → event study → portfolio sorts)
python cli.py run --themes data/themes.json --output results/
```

### Individual Stages

```bash
# Sentiment only (fast, no WRDS needed)
python cli.py run --themes data/themes.json --stages sentiment

# Event study on existing sentiment file
python cli.py event-study --sentiment-file results/sentiment.csv --output results/

# Portfolio sorts on existing sentiment file
python cli.py portfolio --sentiment-file results/sentiment.csv --output results/
```

### Stage Options

| Stage | Needs WRDS | Time | Description |
|-------|------------|------|-------------|
| `sentiment` | No | ~1 min | FinBERT scoring |
| `event_study` | Yes | ~5 min | CAR calculation + regression |
| `portfolio` | Yes | ~5 min | Tercile portfolio returns |

## Input Format

The `--themes` JSON file should have this structure (output from cloud pipeline):

```json
{
  "themes": [
    {
      "theme_id": "theme_001",
      "theme_name": "Technology Innovation",
      "firm_contributions": [
        {
          "firm_id": "AAPL",
          "firm_name": "Apple Inc",
          "permno": 14593,
          "earnings_call_date": "2023-01-15",
          "sentences": [
            {"text": "We are excited about our new products", "speaker": "CEO"}
          ]
        }
      ]
    }
  ]
}
```

## Output

```
results/run_YYYYMMDD_HHMMSS/
├── sentiment_scores.csv          # Sentiment per firm-theme-date
├── event_study/
│   ├── by_theme/
│   │   ├── theme_001_*_event_study_results.csv
│   │   └── theme_001_*_regression_table.txt
│   └── regression_significance_summary.csv
└── portfolio_sorts/
    └── combined_portfolio_returns.csv
```

### Key Output Files

**sentiment_scores.csv**
```
permno,edate,sentiment,theme_id,theme_name
14593,01/15/2023,0.85,theme_001,Technology Innovation
```

**regression_significance_summary.csv** - Ranks themes by whether sentiment predicts returns:
```
Theme_ID,Theme_Name,Sentiment_Coef,p_Value,Significance
theme_001,Technology Innovation,0.045,0.012,*
```

**combined_portfolio_returns.csv** - Cumulative returns by sentiment bucket:
```
bucket,days_from_event,cumulative_return
High,30,0.025
Low,30,-0.012
```

## Architecture

```
cli.py                    # Main entry point
├── src/
│   ├── thematic_sentiment_analyzer.py   # FinBERT sentiment
│   ├── event_study.py                   # WRDS data orchestration
│   ├── event_study_module.py            # CAR/BHAR calculations
│   ├── portfolio_sorts.py               # Tercile portfolios
│   ├── utils.py                         # Shared utilities
│   └── wrds_connection.py               # WRDS connection manager
└── tests/                # 34 unit tests
```

## Running Tests

```bash
python -m pytest tests/ -v
```
