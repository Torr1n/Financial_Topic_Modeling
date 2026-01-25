# Downstream Analysis Integration & Refactor Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate `downstream_analysis/` into the main repo structure, refactor for maintainability, and clean up cruft.

**Architecture:** Keep downstream as a separate module (runs independently from cloud pipeline), but with proper module structure, shared config patterns, and unified documentation. Connected via JSON export from cloud pipeline.

**Tech Stack:** Python 3.8+, FinBERT, WRDS, statsmodels, pandas, GitHub Actions

---

## Phase 1: Cleanup & Delete Cruft

### Task 1: Delete stale handoff_package

The `results/handoff_package/` directory (15MB) contains duplicate code from the main files. This is stale and should be removed.

**Files:**
- Delete: `downstream_analysis/results/handoff_package/` (entire directory)

**Step 1: Verify contents are duplicates**

```bash
diff downstream_analysis/run_pipeline.py downstream_analysis/results/handoff_package/run_pipeline.py | head -20
```

Expected: Files are identical or handoff_package is older version.

**Step 2: Delete the directory**

```bash
rm -rf downstream_analysis/results/handoff_package/
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove stale handoff_package duplicate"
```

---

### Task 2: Clean up results directory structure

Keep only one example run for reference, gitignore the rest.

**Files:**
- Modify: `downstream_analysis/.gitignore` (create if doesn't exist)
- Delete: Old result directories (keep one as example)

**Step 1: Create .gitignore for results**

Create `downstream_analysis/.gitignore`:
```
# Ignore all results except one example
results/*
!results/.gitkeep
!results/example_run/
```

**Step 2: Rename one run as example (for documentation)**

```bash
mv downstream_analysis/results/full_run_20260103 downstream_analysis/results/example_run
rm -rf downstream_analysis/results/run_20260103_EventStudy
rm -rf downstream_analysis/results/run_20260103_PortfolioSorts
touch downstream_analysis/results/.gitkeep
```

**Step 3: Commit**

```bash
git add downstream_analysis/.gitignore downstream_analysis/results/
git commit -m "chore: clean up results directory, keep one example"
```

---

## Phase 2: Restructure Directory Layout

### Task 3: Flatten module structure

Move from nested `downstream_analysis/pipeline/` to cleaner `downstream/src/` structure.

**Files:**
- Rename: `downstream_analysis/` → `downstream/`
- Move: `downstream_analysis/pipeline/*.py` → `downstream/src/`
- Create: `downstream/src/__init__.py`

**Step 1: Rename top-level directory**

```bash
mv downstream_analysis downstream
```

**Step 2: Restructure internals**

```bash
mkdir -p downstream/src
mv downstream/pipeline/*.py downstream/src/
rmdir downstream/pipeline
touch downstream/src/__init__.py
```

**Step 3: Update imports in run files**

In `downstream/run_pipeline.py`, `downstream/run_event_study.py`, `downstream/run_portfolio_sorts.py`:

Change:
```python
from pipeline.thematic_sentiment_analyzer import ThematicSentimentAnalyzer
from pipeline.event_study import ThematicES
from pipeline.portfolio_sorts import PortfolioSorts
```

To:
```python
from src.thematic_sentiment_analyzer import ThematicSentimentAnalyzer
from src.event_study import ThematicES
from src.portfolio_sorts import PortfolioSorts
```

**Step 4: Verify imports work**

```bash
cd downstream && python -c "from src.thematic_sentiment_analyzer import ThematicSentimentAnalyzer; print('OK')"
```

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: flatten downstream directory structure"
```

---

## Phase 3: Deduplicate Code

### Task 4: Extract shared functions to utils module

The functions `create_regression_significance_summary` and `run_portfolio_sorts_for_theme` are duplicated across run_*.py files.

**Files:**
- Create: `downstream/src/utils.py`
- Modify: `downstream/run_pipeline.py`
- Modify: `downstream/run_event_study.py`
- Modify: `downstream/run_portfolio_sorts.py`

**Step 1: Create utils.py with shared functions**

Create `downstream/src/utils.py`:

```python
"""Shared utility functions for downstream analysis pipeline."""

import os
import pandas as pd
from typing import Dict, List, Any, Optional


def create_regression_significance_summary(
    event_study_models: Dict[str, Any],
    themes: List[Dict],
    output_dir: str
) -> pd.DataFrame:
    """
    Create summary table of regression results ranked by sentiment coefficient p-value.

    Args:
        event_study_models: Dict mapping theme_id to statsmodels RegressionResults
        themes: List of theme dicts with theme_id and theme_name
        output_dir: Directory to write summary files

    Returns:
        DataFrame with regression summary statistics
    """
    summary_rows = []

    for theme in themes:
        theme_id = theme.get('theme_id', '')
        theme_name = theme.get('theme_name', '')
        if isinstance(theme_name, list):
            theme_name = theme_name[0] if theme_name else ''

        model = event_study_models.get(theme_id)
        if model is None:
            continue

        try:
            # Extract sentiment coefficient stats
            if 'sentiment' in model.params.index:
                coef = model.params['sentiment']
                pval = model.pvalues['sentiment']
                tstat = model.tvalues['sentiment']

                summary_rows.append({
                    'theme_id': theme_id,
                    'theme_name': theme_name,
                    'sentiment_coef': coef,
                    'sentiment_tstat': tstat,
                    'sentiment_pval': pval,
                    'r_squared': model.rsquared,
                    'r_squared_adj': model.rsquared_adj,
                    'n_obs': int(model.nobs),
                    'significant_5pct': pval < 0.05,
                    'significant_10pct': pval < 0.10
                })
        except Exception as e:
            print(f"Warning: Could not extract stats for {theme_id}: {e}")

    if not summary_rows:
        print("No valid regression results to summarize")
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('sentiment_pval')

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'regression_significance_summary.csv')
    txt_path = os.path.join(output_dir, 'regression_significance_summary.txt')

    summary_df.to_csv(csv_path, index=False)

    # Write formatted text summary
    with open(txt_path, 'w') as f:
        f.write("REGRESSION SIGNIFICANCE SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total themes analyzed: {len(summary_df)}\n")
        f.write(f"Significant at 5%: {summary_df['significant_5pct'].sum()}\n")
        f.write(f"Significant at 10%: {summary_df['significant_10pct'].sum()}\n\n")

        f.write("Top 10 by p-value:\n")
        f.write("-" * 80 + "\n")
        for _, row in summary_df.head(10).iterrows():
            sig = "***" if row['significant_5pct'] else ("*" if row['significant_10pct'] else "")
            f.write(f"{row['theme_name'][:40]:<40} coef={row['sentiment_coef']:>8.4f} "
                   f"p={row['sentiment_pval']:>6.4f} {sig}\n")

    print(f"Regression summary saved to {csv_path}")
    return summary_df


def run_portfolio_sorts_for_theme(
    theme_id: str,
    theme_name: str,
    events: List[Dict],
    results_dir: str,
    wrds_conn: Optional[Any] = None,
    weighting: str = "value"
) -> Optional[pd.DataFrame]:
    """
    Run portfolio sorts for a single theme.

    Args:
        theme_id: Theme identifier
        theme_name: Human-readable theme name
        events: List of event dicts with permno, edate, sentiment
        results_dir: Directory to write results
        wrds_conn: Optional shared WRDS connection
        weighting: 'value' or 'equal' weighted portfolios

    Returns:
        DataFrame with portfolio returns or None if failed
    """
    from src.portfolio_sorts import PortfolioSorts

    if not events:
        print(f"No events for theme {theme_id}")
        return None

    print(f"\nRunning portfolio sorts for: {theme_name}")
    print(f"  Events: {len(events)}")

    try:
        ps = PortfolioSorts(events, wrds_connection=wrds_conn, weighting=weighting)
        ps.crspreturns()
        returns_df = ps.compute_portfolio_returns()

        if returns_df is not None and not returns_df.empty:
            # Add theme identifiers
            returns_df['theme_id'] = theme_id
            returns_df['theme_name'] = theme_name

            # Save to file
            theme_dir = os.path.join(results_dir, 'by_theme')
            os.makedirs(theme_dir, exist_ok=True)

            safe_name = "".join(c if c.isalnum() else "_" for c in theme_name[:30])
            output_path = os.path.join(theme_dir, f'{theme_id}_{safe_name}_portfolio_returns.csv')
            returns_df.to_csv(output_path, index=False)
            print(f"  Saved to {output_path}")

            return returns_df
    except Exception as e:
        print(f"  Error: {e}")

    return None
```

**Step 2: Update run_pipeline.py to use utils**

Replace the duplicated functions with imports:

```python
from src.utils import create_regression_significance_summary, run_portfolio_sorts_for_theme
```

Delete the local definitions of these functions (approximately lines 257-400 and 602-700).

**Step 3: Update run_event_study.py to use utils**

```python
from src.utils import create_regression_significance_summary
```

Delete local definition (approximately lines 106-200).

**Step 4: Update run_portfolio_sorts.py to use utils**

```python
from src.utils import run_portfolio_sorts_for_theme
```

Delete local definition (approximately lines 353-450).

**Step 5: Test imports**

```bash
cd downstream && python -c "from src.utils import create_regression_significance_summary, run_portfolio_sorts_for_theme; print('OK')"
```

**Step 6: Commit**

```bash
git add downstream/src/utils.py downstream/run_*.py
git commit -m "refactor: extract shared functions to utils module"
```

---

### Task 5: Fix hardcoded dates in portfolio_sorts.py

The CRSP query uses hardcoded dates (2023-01-01 to 2023-07-01). Should derive from event dates.

**Files:**
- Modify: `downstream/src/portfolio_sorts.py`

**Step 1: Update crspreturns() method**

Replace lines 39-41:
```python
        # Fixed date range: 1/1/2023 to 7/1/2023 (6 months)
        start_date = '2023-01-01'
        end_date = '2023-07-01'
```

With dynamic date calculation:
```python
        # Derive date range from event dates
        edates = [pd.to_datetime(event["edate"]) for event in self.dictionary]
        min_edate = min(edates)
        max_edate = max(edates)

        # Start from earliest event, end 120 days after latest event
        start_date = min_edate.strftime('%Y-%m-%d')
        end_date = (max_edate + pd.Timedelta(days=120)).strftime('%Y-%m-%d')
```

**Step 2: Test the change**

```bash
cd downstream && python -c "
from src.portfolio_sorts import PortfolioSorts
events = [{'permno': 14593, 'edate': '2024-01-15', 'sentiment': 0.5}]
ps = PortfolioSorts(events)
# Just verify no syntax errors - actual WRDS call would need credentials
print('Syntax OK')
"
```

**Step 3: Commit**

```bash
git add downstream/src/portfolio_sorts.py
git commit -m "fix: derive CRSP date range from event dates instead of hardcoding"
```

---

## Phase 4: Consolidate Entry Points

### Task 6: Create unified CLI entry point

Replace three separate run_*.py with one CLI that can run all stages or individual ones.

**Files:**
- Create: `downstream/cli.py`
- Keep: `downstream/run_pipeline.py` (for backwards compatibility, but thin wrapper)

**Step 1: Create cli.py**

```python
#!/usr/bin/env python3
"""
Downstream Analysis Pipeline CLI

Run sentiment analysis, event studies, and portfolio sorts on theme data.

Usage:
    python -m downstream.cli --themes data/themes.json --output results/
    python -m downstream.cli --themes data/themes.json --stages sentiment event_study
    python -m downstream.cli --sentiment-file results/sentiment.csv --stages portfolio
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import config


def load_themes(themes_file: str) -> dict:
    """Load themes JSON file."""
    with open(themes_file, 'r') as f:
        return json.load(f)


def run_sentiment_stage(themes: dict, output_dir: str) -> str:
    """Run sentiment analysis, return path to sentiment CSV."""
    from src.thematic_sentiment_analyzer import ThematicSentimentAnalyzer

    print("\n" + "=" * 60)
    print("STAGE 1: SENTIMENT ANALYSIS")
    print("=" * 60)

    analyzer = ThematicSentimentAnalyzer(themes)
    results = analyzer.analyze_all_themes()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    sentiment_path = os.path.join(output_dir, 'sentiment_scores.csv')
    results.to_csv(sentiment_path, index=False)
    print(f"Sentiment scores saved to {sentiment_path}")

    return sentiment_path


def run_event_study_stage(
    themes: dict,
    sentiment_df,
    output_dir: str,
    wrds_conn=None
) -> dict:
    """Run event study regressions, return models dict."""
    from src.event_study import ThematicES
    from src.utils import create_regression_significance_summary

    print("\n" + "=" * 60)
    print("STAGE 2: EVENT STUDY")
    print("=" * 60)

    es_output_dir = os.path.join(output_dir, 'event_study')
    os.makedirs(es_output_dir, exist_ok=True)

    models = {}
    theme_list = themes.get('themes', [])

    for theme in theme_list:
        theme_id = theme.get('theme_id', '')
        theme_name = theme.get('theme_name', '')
        if isinstance(theme_name, list):
            theme_name = theme_name[0] if theme_name else ''

        # Filter sentiment data for this theme
        theme_sentiment = sentiment_df[sentiment_df['theme_id'] == theme_id]
        if theme_sentiment.empty:
            print(f"Skipping {theme_id}: no sentiment data")
            continue

        print(f"\nProcessing: {theme_name}")

        try:
            es = ThematicES(
                theme_sentiment.to_dict('records'),
                wrds_connection=wrds_conn
            )
            model = es.run_regression()
            if model is not None:
                models[theme_id] = model
        except Exception as e:
            print(f"  Error: {e}")

    # Create summary
    create_regression_significance_summary(models, theme_list, es_output_dir)

    return models


def run_portfolio_stage(
    themes: dict,
    sentiment_df,
    output_dir: str,
    wrds_conn=None
) -> None:
    """Run portfolio sorts analysis."""
    from src.utils import run_portfolio_sorts_for_theme
    import pandas as pd

    print("\n" + "=" * 60)
    print("STAGE 3: PORTFOLIO SORTS")
    print("=" * 60)

    ps_output_dir = os.path.join(output_dir, 'portfolio_sorts')
    os.makedirs(ps_output_dir, exist_ok=True)

    all_returns = []
    theme_list = themes.get('themes', [])

    for theme in theme_list:
        theme_id = theme.get('theme_id', '')
        theme_name = theme.get('theme_name', '')
        if isinstance(theme_name, list):
            theme_name = theme_name[0] if theme_name else ''

        # Build events list from sentiment data
        theme_sentiment = sentiment_df[sentiment_df['theme_id'] == theme_id]
        events = theme_sentiment.to_dict('records')

        returns_df = run_portfolio_sorts_for_theme(
            theme_id, theme_name, events, ps_output_dir,
            wrds_conn=wrds_conn, weighting=config.WEIGHTING
        )

        if returns_df is not None:
            all_returns.append(returns_df)

    # Combine all theme returns
    if all_returns:
        combined = pd.concat(all_returns, ignore_index=True)
        combined_path = os.path.join(ps_output_dir, 'combined_portfolio_returns.csv')
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined returns saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run downstream analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--themes', '-t',
        help='Path to themes JSON file (from cloud export)'
    )
    parser.add_argument(
        '--sentiment-file', '-s',
        help='Path to existing sentiment CSV (skip sentiment stage)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results',
        help='Output directory (default: results)'
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['sentiment', 'event_study', 'portfolio', 'all'],
        default=['all'],
        help='Which stages to run (default: all)'
    )

    args = parser.parse_args()

    # Validate inputs
    if 'all' in args.stages or 'sentiment' in args.stages:
        if not args.themes:
            parser.error("--themes required for sentiment stage")

    if ('event_study' in args.stages or 'portfolio' in args.stages) and 'sentiment' not in args.stages and 'all' not in args.stages:
        if not args.sentiment_file:
            parser.error("--sentiment-file required when skipping sentiment stage")

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Determine stages to run
    stages = set(args.stages)
    if 'all' in stages:
        stages = {'sentiment', 'event_study', 'portfolio'}

    # Load themes if provided
    themes = None
    if args.themes:
        themes = load_themes(args.themes)
        print(f"Loaded {len(themes.get('themes', []))} themes")

    # Initialize WRDS connection (shared across stages)
    wrds_conn = None
    if 'event_study' in stages or 'portfolio' in stages:
        try:
            import wrds
            wrds_conn = wrds.Connection()
            print("WRDS connection established")
        except Exception as e:
            print(f"Warning: Could not connect to WRDS: {e}")
            print("Event study and portfolio stages will be skipped")
            stages -= {'event_study', 'portfolio'}

    # Run stages
    sentiment_df = None

    if 'sentiment' in stages:
        sentiment_path = run_sentiment_stage(themes, output_dir)
        import pandas as pd
        sentiment_df = pd.read_csv(sentiment_path)
    elif args.sentiment_file:
        import pandas as pd
        sentiment_df = pd.read_csv(args.sentiment_file)
        print(f"Loaded existing sentiment from {args.sentiment_file}")

    if 'event_study' in stages and sentiment_df is not None:
        run_event_study_stage(themes, sentiment_df, output_dir, wrds_conn)

    if 'portfolio' in stages and sentiment_df is not None:
        run_portfolio_stage(themes, sentiment_df, output_dir, wrds_conn)

    # Cleanup
    if wrds_conn:
        wrds_conn.close()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

**Step 2: Update run_pipeline.py to be thin wrapper**

Replace entire file with:

```python
#!/usr/bin/env python3
"""
Legacy entry point - redirects to cli.py

For new usage, prefer: python -m downstream.cli
"""
import sys
from cli import main

if __name__ == '__main__':
    print("Note: Consider using 'python -m downstream.cli' for new invocations")
    main()
```

**Step 3: Test CLI**

```bash
cd downstream && python cli.py --help
```

**Step 4: Commit**

```bash
git add downstream/cli.py downstream/run_pipeline.py
git commit -m "refactor: add unified CLI entry point"
```

---

## Phase 5: Add Tests

### Task 7: Create basic test structure for downstream

**Files:**
- Create: `tests/downstream/__init__.py`
- Create: `tests/downstream/test_utils.py`
- Create: `tests/downstream/test_config.py`

**Step 1: Create test directory**

```bash
mkdir -p tests/downstream
touch tests/downstream/__init__.py
```

**Step 2: Create test_config.py**

```python
"""Tests for downstream config validation."""
import sys
sys.path.insert(0, 'downstream')

import config


def test_config_validation():
    """Test that config validates successfully."""
    # Should not raise
    config.validate_config()


def test_event_window_constraints():
    """Test event window parameter constraints."""
    assert config.EVENT_WINDOW_START < 0
    assert config.EVENT_WINDOW_END > 0
    assert config.ESTIMATION_WINDOW > 0


def test_valid_model_options():
    """Test MODEL is a valid option."""
    assert config.MODEL in ['m', 'ff', 'ffm', 'madj']


def test_valid_weighting_options():
    """Test WEIGHTING is a valid option."""
    assert config.WEIGHTING in ['value', 'equal']
```

**Step 3: Create test_utils.py**

```python
"""Tests for downstream utils module."""
import sys
import os
import tempfile
sys.path.insert(0, 'downstream')

import pandas as pd
from unittest.mock import MagicMock


def test_create_regression_summary_empty():
    """Test summary handles empty input."""
    from src.utils import create_regression_significance_summary

    with tempfile.TemporaryDirectory() as tmpdir:
        result = create_regression_significance_summary({}, [], tmpdir)
        assert result.empty


def test_create_regression_summary_with_data():
    """Test summary with mock regression results."""
    from src.utils import create_regression_significance_summary

    # Create mock model
    mock_model = MagicMock()
    mock_model.params = pd.Series({'sentiment': 0.05, 'intercept': 0.01})
    mock_model.pvalues = pd.Series({'sentiment': 0.03, 'intercept': 0.001})
    mock_model.tvalues = pd.Series({'sentiment': 2.1, 'intercept': 3.5})
    mock_model.rsquared = 0.15
    mock_model.rsquared_adj = 0.12
    mock_model.nobs = 100

    themes = [{'theme_id': 'theme_001', 'theme_name': 'Test Theme'}]
    models = {'theme_001': mock_model}

    with tempfile.TemporaryDirectory() as tmpdir:
        result = create_regression_significance_summary(models, themes, tmpdir)

        assert len(result) == 1
        assert result.iloc[0]['theme_id'] == 'theme_001'
        assert result.iloc[0]['sentiment_coef'] == 0.05
        assert result.iloc[0]['significant_5pct'] == True

        # Check files created
        assert os.path.exists(os.path.join(tmpdir, 'regression_significance_summary.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'regression_significance_summary.txt'))
```

**Step 4: Run tests**

```bash
pytest tests/downstream/ -v
```

**Step 5: Commit**

```bash
git add tests/downstream/
git commit -m "test: add basic downstream tests"
```

---

## Phase 6: Documentation

### Task 8: Create architecture documentation

**Files:**
- Create: `docs/ARCHITECTURE.md`
- Modify: `docs/PIPELINE_INTEGRATION.md` (update paths)

**Step 1: Create ARCHITECTURE.md**

```markdown
# Financial Topic Modeling - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FINANCIAL TOPIC MODELING                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐         ┌─────────────────────────────────┐   │
│  │   Cloud Pipeline    │         │     Downstream Analysis          │   │
│  │   (Topic Modeling)  │         │     (Sentiment & Event Study)    │   │
│  ├─────────────────────┤         ├─────────────────────────────────┤   │
│  │ • Ingest transcripts│         │ • FinBERT sentiment scoring     │   │
│  │ • BERTopic clusters │  JSON   │ • CAR regression (14 covariates)│   │
│  │ • Cross-firm themes │ ──────► │ • Portfolio sorts (terciles)    │   │
│  │ • PostgreSQL storage│         │ • WRDS market data              │   │
│  └─────────────────────┘         └─────────────────────────────────┘   │
│           │                                    │                        │
│           │                                    │                        │
│  ┌────────▼────────┐              ┌───────────▼───────────┐            │
│  │  cloud/src/     │              │  downstream/src/      │            │
│  │  ├── pipeline/  │              │  ├── sentiment.py     │            │
│  │  ├── database/  │              │  ├── event_study.py   │            │
│  │  └── export/    │              │  └── portfolio.py     │            │
│  └─────────────────┘              └───────────────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
Financial_Topic_Modeling/
├── cloud/                      # Topic modeling pipeline (AWS)
│   ├── src/
│   │   ├── pipeline/           # Unified processing pipeline
│   │   ├── database/           # PostgreSQL models & repository
│   │   ├── export/             # JSON export for downstream
│   │   ├── firm_processor.py   # Per-firm topic extraction
│   │   ├── theme_aggregator.py # Cross-firm theme clustering
│   │   └── interfaces.py       # Abstract contracts
│   ├── config/                 # YAML configuration
│   └── scripts/                # Utility scripts
│
├── downstream/                 # Sentiment & event study (WRDS)
│   ├── src/
│   │   ├── thematic_sentiment_analyzer.py  # FinBERT scoring
│   │   ├── event_study.py                  # CAR regression
│   │   ├── event_study_module.py           # Core event study logic
│   │   ├── portfolio_sorts.py              # Tercile portfolios
│   │   └── utils.py                        # Shared utilities
│   ├── cli.py                  # Main entry point
│   ├── config.py               # Configuration
│   └── results/                # Output directory
│
├── tests/
│   ├── unit/                   # Cloud pipeline tests
│   ├── integration/            # End-to-end tests
│   └── downstream/             # Downstream tests
│
├── docs/
│   ├── ARCHITECTURE.md         # This file
│   ├── PIPELINE_INTEGRATION.md # How to connect pipelines
│   └── plans/                  # Implementation plans
│
└── legacy/                     # Old MVP code (reference only)
```

## Data Flow

### 1. Topic Modeling (Cloud Pipeline)

```
Earnings Transcripts (CSV/WRDS)
        │
        ▼
┌───────────────────────────────┐
│ Data Ingestion                │
│ • Parse transcript text       │
│ • Extract sentences           │
│ • Identify speakers           │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ Firm-Level Processing         │
│ • Generate embeddings         │
│ • BERTopic clustering         │
│ • LLM topic summaries         │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ Theme Aggregation             │
│ • Re-cluster firm topics      │
│ • Validate (min firms, etc)   │
│ • LLM theme descriptions      │
└───────────────────────────────┘
        │
        ▼
PostgreSQL (Theme → Topic → Sentence → Firm)
```

### 2. Export Bridge

```
PostgreSQL
    │
    ▼
┌───────────────────────────────┐
│ export_for_downstream.py      │
│ • Query theme hierarchy       │
│ • WRDS PERMNO lookup          │
│ • Restructure for downstream  │
└───────────────────────────────┘
    │
    ▼
themes_for_sentiment.json
```

### 3. Downstream Analysis

```
themes_for_sentiment.json
        │
        ▼
┌───────────────────────────────┐
│ Stage 1: Sentiment            │
│ • FinBERT on each sentence    │
│ • Aggregate to firm-theme     │
│ Output: sentiment_scores.csv  │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ Stage 2: Event Study          │
│ • Query WRDS (CRSP/Compustat) │
│ • Calculate CAR               │
│ • OLS: CAR ~ sentiment + covs │
│ Output: regression_summary    │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ Stage 3: Portfolio Sorts      │
│ • Sort by sentiment tercile   │
│ • Track 90-day returns        │
│ • Value-weighted portfolios   │
│ Output: portfolio_returns     │
└───────────────────────────────┘
```

## Key Design Decisions

1. **Separate Pipelines**: Cloud and downstream run independently, connected by JSON. This allows different compute requirements (GPU for BERTopic, WRDS access for event study).

2. **PERMNO at Export**: CRSP identifiers are resolved during export, not downstream. This ensures temporal accuracy (PERMNOs change during M&A).

3. **Shared WRDS Connection**: Downstream stages share one WRDS connection to minimize API calls.

4. **Config-Driven**: Both pipelines use configuration files for hyperparameters, not hardcoded values.

## Running the Pipelines

### Cloud Pipeline
```bash
cd cloud
python -m src.pipeline.unified_pipeline --config config/production.yaml
```

### Export to Downstream
```bash
python -m cloud.src.export.export_for_downstream \
    --db-url postgresql://user:pass@host/db \
    --output downstream/data/themes.json
```

### Downstream Analysis
```bash
cd downstream
python cli.py --themes data/themes.json --output results/
```

Or individual stages:
```bash
python cli.py --themes data/themes.json --stages sentiment
python cli.py --sentiment-file results/sentiment.csv --stages event_study portfolio
```
```

**Step 2: Update PIPELINE_INTEGRATION.md paths**

Update file paths from `downstream_analysis/` to `downstream/`.

**Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md docs/PIPELINE_INTEGRATION.md
git commit -m "docs: add architecture documentation, update paths"
```

---

## Phase 7: GitHub Actions (Optional)

### Task 9: Create workflow for downstream analysis

**Files:**
- Create: `.github/workflows/downstream.yml`

**Step 1: Create workflow file**

```yaml
name: Downstream Analysis

on:
  workflow_dispatch:
    inputs:
      themes_file:
        description: 'Path to themes JSON in repo or S3 URL'
        required: true
        type: string
      stages:
        description: 'Stages to run (sentiment, event_study, portfolio, all)'
        required: false
        default: 'all'
        type: string

  # Trigger on themes file push (optional)
  # push:
  #   paths:
  #     - 'downstream/data/themes*.json'

jobs:
  analyze:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r downstream/requirements.txt

      - name: Configure WRDS credentials
        env:
          WRDS_USERNAME: ${{ secrets.WRDS_USERNAME }}
          WRDS_PASSWORD: ${{ secrets.WRDS_PASSWORD }}
        run: |
          mkdir -p ~/.pgpass
          echo "wrds-pgdata.wharton.upenn.edu:9737:wrds:$WRDS_USERNAME:$WRDS_PASSWORD" > ~/.pgpass
          chmod 600 ~/.pgpass

      - name: Run downstream analysis
        working-directory: downstream
        run: |
          python cli.py \
            --themes "${{ github.event.inputs.themes_file }}" \
            --stages ${{ github.event.inputs.stages }} \
            --output results/

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: downstream-results-${{ github.run_id }}
          path: downstream/results/
          retention-days: 30
```

**Step 2: Add secrets documentation**

Add to README or docs:
```markdown
## GitHub Actions Setup

To run downstream analysis via GitHub Actions:

1. Add repository secrets:
   - `WRDS_USERNAME`: Your WRDS username
   - `WRDS_PASSWORD`: Your WRDS password

2. Trigger manually from Actions tab, or push themes JSON to trigger automatically.
```

**Step 3: Commit**

```bash
git add .github/workflows/downstream.yml
git commit -m "ci: add GitHub Actions workflow for downstream analysis"
```

---

## Phase 8: Final Cleanup

### Task 10: Update root-level files

**Files:**
- Modify: `CLAUDE.md` (update paths)
- Modify: `.gitignore` (add downstream patterns)

**Step 1: Update CLAUDE.md**

Update references from `downstream_analysis/` to `downstream/`.

**Step 2: Update .gitignore**

Add:
```
# Downstream results (keep example only)
downstream/results/*
!downstream/results/.gitkeep
!downstream/results/example_run/

# Data files
downstream/data/*.json
!downstream/data/example_themes.json
```

**Step 3: Final commit**

```bash
git add CLAUDE.md .gitignore
git commit -m "chore: update root config for downstream integration"
```

---

## Summary Checklist

- [ ] Phase 1: Delete handoff_package, clean results
- [ ] Phase 2: Rename directory, flatten structure
- [ ] Phase 3: Extract utils, fix hardcoded dates
- [ ] Phase 4: Create unified CLI
- [ ] Phase 5: Add basic tests
- [ ] Phase 6: Write architecture docs
- [ ] Phase 7: GitHub Actions workflow
- [ ] Phase 8: Update root files

**Estimated tasks: 10**
**Estimated commits: 10**
