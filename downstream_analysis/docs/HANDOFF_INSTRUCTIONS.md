# Handoff Instructions

## What You're Receiving

This is a **self-contained pipeline** for analyzing sentiment in earnings call transcripts and conducting event studies on stock returns. Everything you need to run the complete analysis is in this directory.

---

## Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
cd handoff_package
pip install -r requirements.txt
```

### 2. Configure WRDS Access

```bash
# Create .pgpass file with your WRDS credentials
echo "wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD" > ~/.pgpass
chmod 600 ~/.pgpass

# Test connection
python -c "import wrds; db = wrds.Connection(); print('Connected!'); db.close()"
```

### 3. Run Example Pipeline

```bash
# Run on example data (small test)
python run_pipeline.py --themes_file data/example_themes.json --output_dir results/
```

This will:
- Compute sentiment scores using FinBERT
- Run event study on WRDS CRSP data
- Generate portfolio sorts by sentiment
- Save all results to `results/run_YYYYMMDD_HHMMSS/` directory (timestamped)

---

## Directory Structure

```
handoff_package/
├── README.md                    # Main documentation (START HERE!)
├── HANDOFF_INSTRUCTIONS.md     # This file
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration settings
├── run_pipeline.py             # Main pipeline runner
│
├── pipeline/                   # Core pipeline modules
│   ├── thematic_sentiment_analyzer.py   # FinBERT sentiment analysis
│   ├── event_study.py                   # Batched event study with 14 covariates
│   ├── event_study_module.py            # CAR calculation module (called by event_study.py)
│   └── portfolio_sorts.py               # Portfolio sorts (per-theme)
│
├── data/                       # Input data & cached WRDS CSVs
│   ├── example_themes.json     # Example input file
│   ├── compustat_fundq.csv     # Cached Compustat quarterly data
│   ├── compustat_funda.csv     # Cached Compustat annual data
│   ├── crsp_msf.csv            # Cached CRSP monthly stock file
│   ├── ccmxpf_linktable.csv    # CRSP-Compustat link table
│   └── ...                     # Other cached WRDS data
│
├── docs/                       # Additional documentation
│   └── FILE_DOCUMENTATION.md   # Detailed file documentation
│
├── results/                    # Output directory with timestamped runs
│   ├── run_20260102_154500/    # Each run gets a timestamp
│   ├── run_20260102_160000/    # Preserves all historical runs
│   └── run_20260102_173000/    # Easy to compare results
│
└── logs/                       # Log files (created on first run)
```

---

## Input Requirements

You need ONE input file: **Earnings Call Themes JSON**

**CRITICAL:** This file must include:
1. Themes with firm contributions
2. **PERMNOs** at the firm_contribution level (not theme level!)
3. PERMNOs temporally matched to earnings call dates
4. Sentences with text (NO sentiment scores - pipeline computes them)

**Example format:**
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
            {"text": "We continue to navigate challenges...", "speaker": "CEO"}
          ]
        }
      ]
    }
  ]
}
```

See `data/example_themes.json` for a complete example.

---

## How to Get PERMNOs from WRDS

If you have Capital IQ transcripts data, use this SQL query:

```sql
-- Get PERMNOs matched to earnings call dates
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

This ensures each earnings call gets the correct PERMNO valid on that date.

---

## Running the Pipeline

### Complete Pipeline (All Stages)

```bash
python run_pipeline.py \
    --themes_file data/your_themes.json \
    --output_dir results/
```

### Individual Stages

**1. Sentiment Analysis Only:**
```bash
python run_pipeline.py \
    --themes_file data/your_themes.json \
    --skip_event_study \
    --skip_portfolio
```

**2. Event Study Only (requires existing sentiment_scores.csv):**
```bash
python run_pipeline.py \
    --themes_file data/your_themes.json \
    --skip_sentiment \
    --skip_portfolio
```

**3. Portfolio Sorts Only (requires existing sentiment_scores.csv):**
```bash
python run_pipeline.py \
    --themes_file data/your_themes.json \
    --skip_sentiment \
    --skip_event_study
```

---

## Configuration

All settings are in `config.py`. Key parameters:

**Sentiment Analysis:**
```python
BATCH_SIZE = 16      # Reduce if memory issues
USE_GPU = False      # Set True if you have GPU
```

**Event Study:**
```python
EVENT_WINDOW_START = -10  # Days before earnings call
EVENT_WINDOW_END = 10     # Days after
ESTIMATION_WINDOW = 100   # Days for estimation
GAP = 50                  # Gap between windows
MODEL = 'm'               # 'm', 'ff', 'ffm', 'madj'
```

**Portfolio Sorts:**
```python
WEIGHTING = 'value'      # 'value' or 'equal'
PORTFOLIO_DAYS = 90      # Days to track
```

---

## Output Files

**NEW:** The pipeline now creates timestamped run directories to preserve all historical results!

Each run creates a folder: `results/run_YYYYMMDD_HHMMSS/` containing:

### Complete Output Structure

```
results/
└── run_20260102_154500/                    # Timestamped run folder
    ├── pipeline_summary.txt                 # Summary report with key statistics
    ├── sentiment_scores.csv                 # All sentiment scores (all themes)
    │
    ├── sentiment_analysis_output/           # Per-theme sentiment details
    │   ├── theme_0_*.csv
    │   ├── theme_1_*.csv
    │   └── ...
    │
    ├── event_study/                         # Batched event study with regressions
    │   ├── by_theme/
    │   │   ├── 0_theme_name_event_study_results.csv  # Regression data
    │   │   ├── 0_theme_name_regression_table.txt     # OLS output
    │   │   └── ...
    │   ├── regression_significance_summary.csv       # Ranked themes (CSV)
    │   └── regression_significance_summary.txt       # Ranked themes (formatted)
    │
    └── portfolio_sorts/                     # Per-theme + combined portfolio sorts
        ├── by_theme/
        │   ├── 0_theme_name_portfolio_returns.csv
        │   └── ...
        ├── combined_portfolio_returns.csv   # Aggregated across all themes
        └── portfolio_time_series_chart.png  # Visual chart (300 DPI)
```

### Key Files

**`sentiment_scores.csv`** - Firm-level sentiment scores from FinBERT
Columns: `permno`, `edate`, `sentiment`, `firm_name`, `theme_name`, `theme_id`, `n_sentences`

**`*_event_study_results.csv`** - Complete regression dataset (per theme)
Columns: `permno`, `edate`, `car`, `sentiment`, + 14 covariates (ROA, leverage, etc.)

**`*_regression_table.txt`** - OLS regression output (per theme)
Regression: `CAR ~ sentiment + 14 covariates`

**`regression_significance_summary.txt`** - Themes ranked by sentiment p-value
Shows which themes have strongest sentiment signals

**`*_portfolio_returns.csv`** - Portfolio returns by sentiment tercile (per theme)
Columns: `bucket` (Low/Medium/High), `days_from_event`, `vw_return`, `cumulative_return`

**`combined_portfolio_returns.csv`** - Aggregated portfolio returns (all themes)
Same format as per-theme, but aggregated across all themes

**`portfolio_time_series_chart.png`** - Visual chart of portfolio performance
High-quality PNG showing cumulative returns by sentiment bucket over 90 days

**`pipeline_summary.txt`** - Summary of all themes with statistics

### Benefits of Timestamped Runs

- **Preserve history**: Never lose previous results
- **Easy comparison**: Compare results across different input data or configurations
- **Reproducibility**: Each run is completely self-contained
- **Re-analysis**: Can run additional analysis on any saved run's data

---

## Common Issues & Solutions

### 1. Missing PERMNO in JSON

**Error:** `KeyError: 'permno'`

**Solution:** Verify PERMNO is at firm_contribution level:
```python
import json
with open('data/themes.json') as f:
    data = json.load(f)
    contrib = data['themes'][0]['firm_contributions'][0]
    print('Has permno:', 'permno' in contrib)
```

### 2. WRDS Connection Fails

**Error:** "Connection refused" or authentication error

**Solution:**
```bash
# Check .pgpass file
cat ~/.pgpass
chmod 600 ~/.pgpass

# Test connection
python -c "import wrds; wrds.Connection()"
```

### 3. Event Study Returns No Results

**Problem:** Empty output from event study

**Solutions:**
- Check PERMNO validity in WRDS
- Ensure events have ~170 days of history
- Lower MIN_OBSERVATIONS in config.py
- Verify date formats (YYYY-MM-DD in JSON, MM/DD/YYYY in event study)

### 4. Memory Errors

**Problem:** Out of memory during FinBERT

**Solutions:**
```python
# In config.py
BATCH_SIZE = 8  # Reduce from 16
USE_GPU = False  # Force CPU mode
```

Or process themes in batches:
```python
# Split large JSON file into smaller chunks
# Run pipeline separately on each chunk
```

---

## Testing the Pipeline

### 1. Test with Example Data

```bash
# Run on provided example (3 themes, 4 firms)
python run_pipeline.py \
    --themes_file data/example_themes.json \
    --output_dir test_results/
```

Should complete in ~5 minutes and produce:
- `test_results/sentiment_scores.csv` (6 rows)
- `test_results/event_study/` (3 files)
- `test_results/portfolio_returns.csv`

### 2. Validate Your Input

```python
import json

# Load and check structure
with open('data/your_themes.json') as f:
    data = json.load(f)

theme = data['themes'][0]
contrib = theme['firm_contributions'][0]

# Verify required fields
assert 'permno' in contrib
assert 'earnings_call_date' in contrib
assert 'sentences' in contrib
assert len(contrib['sentences']) > 0

print("✓ Input format valid")
```

### 3. Test WRDS Access

```python
import wrds

db = wrds.Connection()

# Test CRSP access
result = db.raw_sql("SELECT COUNT(*) FROM crsp_a_stock.dsf LIMIT 1")
print("✓ CRSP access:", not result.empty)

# Test Fama-French access
result = db.raw_sql("SELECT COUNT(*) FROM ff_all.factors_daily LIMIT 1")
print("✓ FF access:", not result.empty)

db.close()
```

---

## Next Steps

### 1. Read the Documentation

- **README.md**: Complete pipeline documentation (start here!)
- **docs/FILE_DOCUMENTATION.md**: Detailed module documentation
- **config.py**: All configurable settings

### 2. Prepare Your Input

- Run your thematic clustering pipeline
- Add PERMNOs using WRDS linking tables (SQL query above)
- Save as JSON in `data/` directory

### 3. Run the Pipeline

```bash
python run_pipeline.py \
    --themes_file data/your_themes.json \
    --output_dir results/
```

### 4. Analyze Results

- Review `results/sentiment_scores.csv`
- Check event study statistics in `results/event_study/`
- Examine portfolio returns in `results/portfolio_returns.csv`
- Read `results/pipeline_summary.txt` for key findings

---

## Support Resources

### Documentation
- Main README: `README.md`
- File docs: `docs/FILE_DOCUMENTATION.md`
- Config reference: `config.py` (well-commented)

### Example Files
- Input example: `data/example_themes.json`
- Configuration: `config.py`

### External Resources
- WRDS Setup: https://wrds-www.wharton.upenn.edu/
- FinBERT: https://huggingface.co/yiyanghkust/finbert-tone
- Event Studies: MacKinlay (1997)

---

## What's NOT Included

This pipeline does NOT include:
- Thematic clustering (you need to run that separately)
- PERMNO mapping logic (PERMNOs must be in input JSON)
- WRDS account (you need your own)
- Stock price data (queried from WRDS in real-time)

---

## Summary

**Input:** Themes JSON with PERMNOs
**Process:** Sentiment → Event Study → Portfolio Sorts
**Output:** CSV files with results
**Time:** ~10-30 minutes for 100-500 firms

**Key Requirements:**
1. Python 3.8+
2. WRDS account (CRSP + FF access)
3. Themes JSON with PERMNOs at earnings call level

**Start Command:**
```bash
python run_pipeline.py --themes_file data/your_themes.json
```

Good luck with your research!

---

## Version Information

**Pipeline Version:** 2.0.0
**Last Updated:** January 3, 2026

**What's New in v2.0:**
- Batched event study (90% fewer WRDS API calls)
- Regression significance summary across all themes
- Portfolio time series chart (PNG visualization)
- See [CHANGELOG.md](CHANGELOG.md) for complete details
