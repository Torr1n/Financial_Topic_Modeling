# Downstream Analysis

Sentiment analysis, event studies, and portfolio sorts on earnings call themes.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set WRDS credentials in .env (project root)
WRDS_USERNAME=your_username
WRDS_PASSWORD=your_password

# Run full pipeline
python cli.py run --themes data/themes.json --output results/

# Run sentiment only (no WRDS needed)
python cli.py run --themes data/themes.json --stages sentiment

# Run event study on existing sentiment
python cli.py event-study --sentiment-file results/sentiment.csv

# Run portfolio sorts on existing sentiment
python cli.py portfolio --sentiment-file results/sentiment.csv
```

## Pipeline Stages

1. **Sentiment** - FinBERT scores for each firm-theme (no WRDS)
2. **Event Study** - Abnormal returns (CAR) around earnings calls (needs WRDS)
3. **Portfolio Sorts** - Returns by sentiment terciles (needs WRDS)

## Output

```
results/
├── sentiment_scores.csv
├── event_study/
│   ├── by_theme/*.csv
│   └── regression_significance_summary.csv
└── portfolio_sorts/
    └── combined_portfolio_returns.csv
```
