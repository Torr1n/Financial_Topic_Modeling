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
