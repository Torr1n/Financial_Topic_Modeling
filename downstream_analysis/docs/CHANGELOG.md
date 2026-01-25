# Changelog

All notable changes to the handoff package are documented in this file.

## [2.0.0] - 2026-01-03

### Major Architecture Refactoring

#### Batched Event Study Implementation
- **BREAKING CHANGE:** Event study now uses batched approach instead of per-theme queries
- Reduced WRDS API calls by ~90% (from 20-30 queries down to 3)
- Single batched query for all unique (permno, edate) combinations across all themes
- Handles duplicate events with different sentiment scores per theme correctly
- Event study regression logic moved from `pipeline/event_study.py` to pipeline-level (`run_pipeline.py`, `run_event_study.py`)

**Impact:**
- Significantly faster execution (especially for large theme sets)
- Lower WRDS API usage (reduces timeouts and rate limiting)
- Maintains accuracy: same results as per-theme approach

#### Event Study Changes

**`pipeline/event_study.py`:**
- Renamed `ES()` method to `calculateCovariatesAndCAR()`
- Removed regression code and visualization methods
- `doAll()` now returns DataFrame instead of running regression
- Class focuses solely on data retrieval and covariate calculation

**`run_pipeline.py` and `run_event_study.py`:**
- Added `run_batched_event_study()` function
  - Deduplicates events by (permno, edate)
  - Tracks theme-specific sentiments separately
  - Merges sentiment back after batched processing
- Added `run_regression_for_theme()` function
  - Runs OLS regression: `CAR ~ sentiment + 14 covariates`
  - Saves regression table (.txt) and results (.csv)
- Added `create_regression_significance_summary()` function
  - Ranks all themes by sentiment coefficient p-value
  - Generates CSV and TXT summaries
  - Shows statistical significance levels (p<0.01, p<0.05, p<0.10)

### New Features

#### 1. Regression Significance Summary
- **Files:** `regression_significance_summary.csv` and `regression_significance_summary.txt`
- Ranks themes by sentiment coefficient statistical significance
- Provides easy identification of most predictive themes
- Includes summary statistics (% significant at various levels)
- Machine-readable CSV + human-readable formatted TXT

**Example Output:**
```
Rank  Theme ID    Theme Name              Sentiment Coef  P-value   Sig    R²     N
----  ----------  ----------------------  --------------  --------  -----  -----  ---
1     theme_003   AI Innovation                0.0345     0.001    ***    0.312  145
2     theme_007   Supply Chain Resilience      0.0234     0.004    **     0.245  132
...
```

#### 2. Portfolio Time Series Chart
- **File:** `portfolio_time_series_chart.png`
- Professional matplotlib visualization (300 DPI, publication-ready)
- Three colored lines:
  - Red: Low sentiment portfolio
  - Orange: Medium sentiment portfolio
  - Green: High sentiment portfolio
- Summary statistics embedded in chart
- Shows cumulative returns over 90 days from event
- Replaces previous ASCII text-based chart

**Features:**
- Grid, zero line, and professional formatting
- Clear axis labels and legend
- Summary text box with final returns and long-short spread
- Graceful handling of missing matplotlib dependency

### Improvements

#### Documentation Updates
- Updated `README.md` with batched event study architecture
- Updated `USAGE_GUIDE.md` with new output files
- Updated `HANDOFF_INSTRUCTIONS.md` with complete output structure
- Updated `docs/FILE_DOCUMENTATION.md` with detailed batched approach documentation
- All documentation now reflects:
  - Batched event study approach
  - 14 covariates + sentiment regression model
  - New output files (regression significance summary, portfolio chart)

#### Output Structure
- Clearer organization of per-theme vs aggregated results
- All event study outputs in `event_study/` directory
- All portfolio sort outputs in `portfolio_sorts/` directory
- New summary files at appropriate levels

### Technical Details

#### Duplicate Event Handling
**Problem:** Same (permno, edate) can appear in multiple themes with different sentiments

**Solution:**
1. Deduplicate for WRDS query (save API calls)
2. Track `theme_event_mapping`: {theme_id: [(permno, edate, sentiment), ...]}
3. After batched event study, merge theme-specific sentiments
4. Create separate regression rows for each theme

**Example:**
- Event (permno=12345, edate='2023-01-15') in Theme A with sentiment=+0.8
- Same event in Theme B with sentiment=-0.3
- Batched query calculates covariates + CAR once
- Two regression rows created with different sentiments

#### Regression Model
```
CAR ~ const + sentiment + Return_on_Assets + Book_Leverage +
      Capital_Expenditures + Research_and_Development + Sales_Growth +
      Firm_Size + Cash + Asset_Tangibility + Delta_Employee_Change +
      Stock_Volatility + Stock_Return + Market_to_Book + Earnings_Surprise
```

**14 Covariates:**
1. Return on Assets
2. Book Leverage
3. Capital Expenditures
4. Research & Development
5. Sales Growth
6. Firm Size (log sales)
7. Cash
8. Asset Tangibility
9. Delta Employee Change
10. Stock Volatility (12-month)
11. Stock Return (3-month)
12. Market-to-Book
13. Earnings Surprise
14. (+ sentiment as key explanatory variable)

### Files Changed

**Core Modules:**
- `pipeline/event_study.py`: Major refactoring (regression logic removed)
- `run_pipeline.py`: Added batched event study, regression, summary, and chart functions
- `run_event_study.py`: Added batched approach for standalone script
- `run_portfolio_sorts.py`: Added chart generation function

**Documentation:**
- `README.md`: Updated architecture, data sources, file documentation, output files
- `USAGE_GUIDE.md`: Updated workflow and output descriptions
- `HANDOFF_INSTRUCTIONS.md`: Updated directory structure and key files
- `docs/FILE_DOCUMENTATION.md`: Comprehensive updates for batched approach
- `CHANGELOG.md`: New file documenting all changes

### Backward Compatibility

**Breaking Changes:**
- Event study no longer runs regression internally
- Regression tables now generated at pipeline level
- Per-theme event study no longer available (use batched approach)

**Migration Guide:**
If you have custom code calling `ThematicES`:
```python
# OLD (no longer works)
study = ThematicES(events)
study.doAll()  # This used to run regression

# NEW (v2.0.0)
study = ThematicES(events)
results_df = study.doAll()  # Returns DataFrame
# Run regression separately at pipeline level
```

### Performance Benchmarks

**WRDS API Calls (20 themes, 500 events):**
- Old approach: ~30 queries (one per theme + CRSP/Compustat)
- New approach: ~3 queries (batched)
- **Improvement:** 90% reduction

**Execution Time (estimated):**
- Old approach: ~15-20 minutes
- New approach: ~5-7 minutes
- **Improvement:** 60-65% faster

### Dependencies

No new dependencies added. Existing dependencies:
- `wrds`
- `pandas`, `numpy`
- `statsmodels` (for OLS regression)
- `matplotlib` (optional, for portfolio chart; gracefully handles absence)
- `transformers`, `torch` (for FinBERT sentiment analysis)

---

## [1.0.0] - 2026-01-02

### Initial Release

- Sentiment analysis with FinBERT
- Per-theme event studies with WRDS
- Per-theme portfolio sorts
- Timestamped output directories
- Standalone scripts for individual stages
- Combined portfolio analysis
- Pipeline summary generation

---

**Version Format:** [Major.Minor.Patch]
- **Major:** Breaking changes, architecture changes
- **Minor:** New features, non-breaking enhancements
- **Patch:** Bug fixes, documentation updates
