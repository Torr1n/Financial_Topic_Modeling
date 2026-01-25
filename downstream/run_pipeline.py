"""
Main Pipeline Runner for Thematic Sentiment Analysis & Event Study

This script orchestrates the complete pipeline:
1. Sentiment Analysis (FinBERT)
2. Event Study per theme (WRDS CRSP + Fama-French + Covariates + Sentiment Regression)
3. Portfolio Sorts per theme (Sentiment Terciles)

Usage:
    python run_pipeline.py --themes_file data/themes.json --output_dir results/

Author: Team 2 COMM386I
Date: January 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import configuration
import config

# Import pipeline modules
from src.thematic_sentiment_analyzer import ThematicSentimentAnalyzer
from src.event_study import ThematicES
from src.portfolio_sorts import PortfolioSorts

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(config.LOGS_DIR) / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def validate_input_file(themes_file):
    """Validate the themes JSON file format."""
    logger.info(f"Validating input file: {themes_file}")

    with open(themes_file, 'r') as f:
        data = json.load(f)

    # Check required structure
    assert 'themes' in data, "Missing 'themes' key in JSON"
    assert len(data['themes']) > 0, "No themes found in JSON"

    # Check first theme structure
    theme = data['themes'][0]
    assert 'theme_id' in theme, "Missing 'theme_id' in theme"
    assert 'firm_contributions' in theme, "Missing 'firm_contributions' in theme"
    assert len(theme['firm_contributions']) > 0, "No firm contributions found"

    # Check first firm contribution structure
    contrib = theme['firm_contributions'][0]
    required_fields = ['permno', 'earnings_call_date', 'sentences']
    for field in required_fields:
        assert field in contrib, f"Missing required field '{field}' in firm contribution"

    assert len(contrib['sentences']) > 0, "No sentences found in firm contribution"
    assert 'text' in contrib['sentences'][0], "Missing 'text' field in sentence"

    logger.info(f"✓ Input file validated successfully")
    logger.info(f"  - {len(data['themes'])} themes")
    logger.info(f"  - {len(theme['firm_contributions'])} firms in first theme")

    return data


def run_sentiment_analysis(themes_data, output_dir):
    """
    Run sentiment analysis on themes.

    Args:
        themes_data: Loaded themes JSON data
        output_dir: Directory for output files

    Returns:
        Dictionary with sentiment results by theme
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: SENTIMENT ANALYSIS")
    logger.info("=" * 80)

    # Initialize analyzer
    analyzer = ThematicSentimentAnalyzer(
        model_name=config.SENTIMENT_MODEL,
        batch_size=config.BATCH_SIZE,
        use_gpu=config.USE_GPU
    )

    # Run analysis
    results = analyzer.analyze_themes(
        thematic_output=themes_data,
        aggregation_strategy=config.AGGREGATION_STRATEGY,
        permno_mapping=None,  # Not needed - PERMNOs in JSON
        output_csv=config.OUTPUT_CSV_PER_THEME,
        csv_directory=str(output_dir / config.CSV_OUTPUT_DIR)
    )

    # Save main sentiment scores
    sentiment_df = pd.DataFrame(results['all_events'])
    if not sentiment_df.empty and 'metadata' in sentiment_df.columns:
        metadata_df = pd.json_normalize(sentiment_df['metadata'])
        sentiment_df = pd.concat([
            sentiment_df[['permno', 'edate', 'sentiment']],
            metadata_df
        ], axis=1)

    output_file = output_dir / 'sentiment_scores.csv'
    sentiment_df.to_csv(output_file, index=False)
    logger.info(f"✓ Sentiment scores saved to {output_file}")
    logger.info(f"  - {len(sentiment_df)} events analyzed")

    return results


def run_batched_event_study(all_events_across_themes, wrds_conn=None):
    """
    Run event study ONCE for all events across all themes.

    This batched approach reduces WRDS API calls from 10+ (one per theme) to just 1.

    Args:
        all_events_across_themes: List of ALL event dictionaries with permno, edate, sentiment
        wrds_conn: Shared WRDS connection

    Returns:
        DataFrame with covariates, CAR, and sentiment for all events
    """
    logger.info("\n" + "="*80)
    logger.info("BATCHED EVENT STUDY (ALL THEMES COMBINED)")
    logger.info("="*80)
    logger.info(f"Total events across all themes: {len(all_events_across_themes)}")

    try:
        # Initialize and run event study with ALL events
        logger.info("\nRunning batched event study with covariates...")
        study = ThematicES(all_events_across_themes, wrds_connection=wrds_conn)

        # Run the complete analysis (pulls WRDS data once, calculates factors, computes CAR)
        try:
            results_df = study.doAll()
        except Exception as e:
            # If WRDS query fails, rollback the connection so it can be reused
            if wrds_conn:
                try:
                    wrds_conn.rollback()
                    logger.warning(f"Rolled back WRDS connection after error")
                except Exception as rollback_error:
                    logger.warning(f"Failed to rollback WRDS connection: {rollback_error}")
            raise e

        if results_df is not None and not results_df.empty:
            logger.info(f"✓ Batched event study complete - {len(results_df)} events processed")
            return results_df
        else:
            logger.error("Batched event study returned no results")
            return None

    except Exception as e:
        logger.error(f"Batched event study failed: {e}", exc_info=True)
        return None


def run_regression_for_theme(theme_id, theme_name, theme_results_df, results_dir):
    """
    Run OLS regression for a single theme on batched event study results.

    Args:
        theme_id: Theme identifier
        theme_name: Theme name (string)
        theme_results_df: DataFrame filtered to this theme's events (from batched results)
        results_dir: Directory to save results

    Returns:
        OLS regression model or None if failed
    """
    # Ensure theme_name is a string
    if isinstance(theme_name, list):
        theme_name = theme_name[0] if len(theme_name) > 0 else "unknown_theme"

    logger.info(f"\n{'='*60}")
    logger.info(f"Regression for: {theme_name} ({theme_id})")
    logger.info(f"Events: {len(theme_results_df)}")
    logger.info(f"{'='*60}")

    try:
        # Define independent variables (covariates + sentiment) and dependent variable (CAR)
        y = theme_results_df["car"].astype(float)
        x = theme_results_df[[
            'sentiment', 'Return_on_Assets', 'Book_Leverage',
            'Capital_Expenditures', 'Research_and_Development', 'Sales_Growth',
            'Firm_Size', 'Cash', 'Asset_Tangibility', 'Delta_Employee_Change',
            'Stock_Volatility', 'Stock_Return', 'Market_to_Book', 'Earnings_Surprise'
        ]].astype(float)
        x = sm.add_constant(x)  # add B(0)

        # Run OLS regression
        model = sm.OLS(y, x).fit()

        # Create theme-specific directory
        theme_dir = results_dir / 'by_theme'
        theme_dir.mkdir(parents=True, exist_ok=True)

        # Save event study results for this theme
        csv_path = theme_dir / f'{theme_id}_{theme_name.replace(" ", "_")}_event_study_results.csv'
        theme_results_df.to_csv(csv_path, index=False)
        logger.info(f"  ✓ Event study results saved to {csv_path}")

        # Save regression table
        regression_path = theme_dir / f'{theme_id}_{theme_name.replace(" ", "_")}_regression_table.txt'
        with open(regression_path, 'w') as f:
            f.write(model.summary().as_text())
        logger.info(f"  ✓ Regression table saved to {regression_path}")

        # Print regression summary
        sentiment_coef = model.params.get('sentiment', np.nan)
        sentiment_pval = model.pvalues.get('sentiment', np.nan)
        sig = ''
        if sentiment_pval < 0.001:
            sig = '***'
        elif sentiment_pval < 0.01:
            sig = '**'
        elif sentiment_pval < 0.05:
            sig = '*'

        logger.info(f"  Sentiment coefficient: {sentiment_coef:+.6f}{sig} (p={sentiment_pval:.4f})")
        logger.info(f"  R-squared: {model.rsquared:.4f}")
        logger.info(f"✓ Regression complete for {theme_name}")

        # Create a simple object to hold the model (to match existing interface)
        class RegressionResult:
            def __init__(self, model, results_df):
                self.model = model
                self.Results = results_df

        return RegressionResult(model, theme_results_df)

    except Exception as e:
        logger.error(f"Regression failed for {theme_name}: {e}", exc_info=True)
        return None


def create_regression_significance_summary(event_study_models, themes, output_dir):
    """
    Create a summary table ranking themes by sentiment coefficient significance.

    Args:
        event_study_models: Dictionary of {theme_id: RegressionResult}
        themes: Dictionary of theme data (for theme names)
        output_dir: Directory to save the summary

    Returns:
        DataFrame with regression results ranked by significance
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING REGRESSION SIGNIFICANCE SUMMARY")
    logger.info("="*80)

    if not event_study_models:
        logger.warning("No event study models to summarize")
        return None

    # Collect regression statistics for all themes
    summary_data = []

    for theme_id, result_obj in event_study_models.items():
        if result_obj is None or not hasattr(result_obj, 'model') or result_obj.model is None:
            continue

        model = result_obj.model
        theme_name_raw = themes[theme_id]['theme_name']
        theme_name = theme_name_raw[0] if isinstance(theme_name_raw, list) else theme_name_raw

        # Extract sentiment coefficient statistics
        sentiment_coef = model.params.get('sentiment', np.nan)
        sentiment_se = model.bse.get('sentiment', np.nan)
        sentiment_tstat = model.tvalues.get('sentiment', np.nan)
        sentiment_pval = model.pvalues.get('sentiment', np.nan)

        # Determine significance level
        sig_stars = ''
        if sentiment_pval < 0.001:
            sig_stars = '***'
        elif sentiment_pval < 0.01:
            sig_stars = '**'
        elif sentiment_pval < 0.05:
            sig_stars = '*'

        # Model fit statistics
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        n_obs = int(model.nobs)

        summary_data.append({
            'Theme_ID': theme_id,
            'Theme_Name': theme_name,
            'Sentiment_Coef': sentiment_coef,
            'Std_Error': sentiment_se,
            't_Statistic': sentiment_tstat,
            'p_Value': sentiment_pval,
            'Significance': sig_stars,
            'R_Squared': r_squared,
            'Adj_R_Squared': adj_r_squared,
            'N_Observations': n_obs
        })

    if not summary_data:
        logger.warning("No valid regression results to summarize")
        return None

    # Create DataFrame and sort by p-value (most significant first)
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('p_Value')

    # Save as CSV
    csv_path = output_dir / 'regression_significance_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved CSV summary to {csv_path}")

    # Create formatted text summary
    txt_lines = []
    txt_lines.append("="*100)
    txt_lines.append("SENTIMENT COEFFICIENT SIGNIFICANCE SUMMARY (RANKED BY P-VALUE)")
    txt_lines.append("="*100)
    txt_lines.append("")
    txt_lines.append("This table ranks themes by the statistical significance of their sentiment coefficient")
    txt_lines.append("in predicting cumulative abnormal returns (CAR).")
    txt_lines.append("")
    txt_lines.append("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    txt_lines.append("")
    txt_lines.append("-"*100)

    # Header
    header = f"{'Rank':<6} {'Theme Name':<50} {'Coef':>10} {'p-value':>10} {'Sig':>5} {'N':>6} {'R²':>8}"
    txt_lines.append(header)
    txt_lines.append("-"*100)

    # Data rows
    for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
        theme_name = row['Theme_Name'][:47] + '...' if len(row['Theme_Name']) > 50 else row['Theme_Name']

        line = f"{rank:<6} {theme_name:<50} {row['Sentiment_Coef']:>+10.6f} {row['p_Value']:>10.4f} {row['Significance']:>5} {row['N_Observations']:>6} {row['R_Squared']:>8.4f}"
        txt_lines.append(line)

    txt_lines.append("-"*100)
    txt_lines.append("")

    # Summary statistics
    txt_lines.append("SUMMARY STATISTICS:")
    txt_lines.append("")
    txt_lines.append(f"  Total themes analyzed: {len(summary_df)}")
    txt_lines.append(f"  Significant at p<0.05: {(summary_df['p_Value'] < 0.05).sum()} ({100*(summary_df['p_Value'] < 0.05).sum()/len(summary_df):.1f}%)")
    txt_lines.append(f"  Significant at p<0.01: {(summary_df['p_Value'] < 0.01).sum()} ({100*(summary_df['p_Value'] < 0.01).sum()/len(summary_df):.1f}%)")
    txt_lines.append(f"  Significant at p<0.001: {(summary_df['p_Value'] < 0.001).sum()} ({100*(summary_df['p_Value'] < 0.001).sum()/len(summary_df):.1f}%)")
    txt_lines.append("")
    txt_lines.append(f"  Positive coefficients: {(summary_df['Sentiment_Coef'] > 0).sum()} ({100*(summary_df['Sentiment_Coef'] > 0).sum()/len(summary_df):.1f}%)")
    txt_lines.append(f"  Negative coefficients: {(summary_df['Sentiment_Coef'] < 0).sum()} ({100*(summary_df['Sentiment_Coef'] < 0).sum()/len(summary_df):.1f}%)")
    txt_lines.append("")
    txt_lines.append(f"  Mean R²: {summary_df['R_Squared'].mean():.4f}")
    txt_lines.append(f"  Mean |coefficient|: {summary_df['Sentiment_Coef'].abs().mean():.6f}")
    txt_lines.append("")

    # Top 3 most significant
    txt_lines.append("TOP 3 MOST SIGNIFICANT THEMES:")
    txt_lines.append("")
    for rank, (_, row) in enumerate(summary_df.head(3).iterrows(), 1):
        txt_lines.append(f"  {rank}. {row['Theme_Name']}")
        txt_lines.append(f"     Coefficient: {row['Sentiment_Coef']:+.6f}{row['Significance']}")
        txt_lines.append(f"     p-value: {row['p_Value']:.6f}")
        txt_lines.append(f"     t-statistic: {row['t_Statistic']:+.4f}")
        txt_lines.append(f"     R²: {row['R_Squared']:.4f} (N={row['N_Observations']})")
        txt_lines.append("")

    txt_lines.append("="*100)
    txt_lines.append("")
    txt_lines.append("INTERPRETATION GUIDE:")
    txt_lines.append("")
    txt_lines.append("  • Positive coefficient: Higher sentiment → Higher abnormal returns")
    txt_lines.append("  • Negative coefficient: Higher sentiment → Lower abnormal returns")
    txt_lines.append("  • p-value < 0.05: Statistically significant relationship (95% confidence)")
    txt_lines.append("  • R²: Proportion of variance in CAR explained by the model (0-1 scale)")
    txt_lines.append("  • t-statistic: Coefficient divided by standard error (larger |t| = stronger evidence)")
    txt_lines.append("")
    txt_lines.append("="*100)

    # Save text summary
    txt_path = output_dir / 'regression_significance_summary.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(txt_lines))
    logger.info(f"✓ Saved text summary to {txt_path}")

    # Print top 5 to console
    logger.info("\nTop 5 Most Significant Themes:")
    logger.info("-" * 60)
    for rank, (_, row) in enumerate(summary_df.head(5).iterrows(), 1):
        logger.info(f"{rank}. {row['Theme_Name'][:50]}")
        logger.info(f"   Coefficient: {row['Sentiment_Coef']:+.6f}{row['Significance']} (p={row['p_Value']:.4f})")

    logger.info("="*80)

    return summary_df


def create_portfolio_time_series_chart(combined_portfolio, output_dir):
    """
    Create line chart showing cumulative returns by sentiment bucket over time.

    Args:
        combined_portfolio: DataFrame with portfolio returns over time
        output_dir: Directory to save the chart
    """
    if combined_portfolio is None or combined_portfolio.empty:
        logger.warning("No portfolio data to chart")
        return

    logger.info("\nCreating portfolio time series chart...")

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - skipping chart generation")
        logger.warning("Install with: pip install matplotlib")
        return

    # Get data for each bucket
    buckets = ['Low', 'Medium', 'High']
    bucket_data = {}
    max_day = 0

    for bucket in buckets:
        bucket_df = combined_portfolio[combined_portfolio['bucket'] == bucket].copy()
        bucket_df = bucket_df.sort_values('days_from_event')
        # Limit to 90 days
        bucket_df = bucket_df[bucket_df['days_from_event'] <= 90]
        bucket_data[bucket] = bucket_df
        if not bucket_df.empty:
            max_day = max(max_day, bucket_df['days_from_event'].max())

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors for each bucket
    colors = {'Low': '#d62728', 'Medium': '#ff7f0e', 'High': '#2ca02c'}
    labels = {'Low': 'Low Sentiment', 'Medium': 'Medium Sentiment', 'High': 'High Sentiment'}

    # Plot each bucket
    for bucket in buckets:
        bucket_df = bucket_data[bucket]
        if not bucket_df.empty:
            ax.plot(
                bucket_df['days_from_event'],
                bucket_df['cumulative_return'] * 100,  # Convert to percentage
                label=labels[bucket],
                color=colors[bucket],
                linewidth=2.5,
                marker='o',
                markersize=4,
                alpha=0.8
            )

    # Customize plot
    ax.set_xlabel('Days from Event', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Portfolio Cumulative Returns by Sentiment Bucket\n(Combined Across All Themes)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    # Legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # Format axes
    ax.tick_params(labelsize=10)

    # Add summary text box with final returns
    summary_text = "Final Returns (Day {}):\n".format(max_day)
    for bucket in buckets:
        bucket_df = bucket_data[bucket]
        if not bucket_df.empty:
            final_return = bucket_df[bucket_df['days_from_event'] == max_day]['cumulative_return'].values
            if len(final_return) > 0:
                summary_text += f"{labels[bucket]}: {final_return[0]*100:+.2f}%\n"

    # Calculate and add spread
    if not bucket_data['High'].empty and not bucket_data['Low'].empty:
        high_final = bucket_data['High'][bucket_data['High']['days_from_event'] == max_day]['cumulative_return'].values
        low_final = bucket_data['Low'][bucket_data['Low']['days_from_event'] == max_day]['cumulative_return'].values
        if len(high_final) > 0 and len(low_final) > 0:
            spread = (high_final[0] - low_final[0]) * 100
            summary_text += f"\nHigh-Low Spread: {spread:+.2f}%"

    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, summary_text.strip(), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=props, family='monospace')

    # Tight layout
    plt.tight_layout()

    # Save chart as PNG
    chart_path = output_dir / 'portfolio_time_series_chart.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved portfolio time series chart to {chart_path}")


def create_combined_portfolio_analysis(portfolio_results, output_dir):
    """
    Create combined portfolio analysis across all themes.

    Aggregates portfolio returns from all themes to show overall performance.
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING COMBINED PORTFOLIO ANALYSIS")
    logger.info("="*80)

    if not portfolio_results:
        return None

    # Combine all theme portfolio returns
    all_returns = []
    for theme_id, returns_df in portfolio_results.items():
        if returns_df is not None and not returns_df.empty:
            returns_copy = returns_df.copy()
            returns_copy['theme_id'] = theme_id
            all_returns.append(returns_copy)

    if not all_returns:
        return None

    combined_df = pd.concat(all_returns, ignore_index=True)

    # Calculate combined returns for each bucket and day (average across themes)
    combined_returns = []

    for bucket in ['Low', 'Medium', 'High']:
        bucket_data = combined_df[combined_df['bucket'] == bucket]
        unique_days = sorted(bucket_data['days_from_event'].unique())
        unique_days = [d for d in unique_days if 0 <= d <= 90]

        for day in unique_days:
            day_data = bucket_data[bucket_data['days_from_event'] == day]
            if not day_data.empty:
                combined_returns.append({
                    'bucket': bucket,
                    'days_from_event': day,
                    'vw_return': day_data['vw_return'].mean(),
                    'cumulative_return': day_data['cumulative_return'].mean(),
                    'n_themes': day_data['theme_id'].nunique()
                })

    combined_portfolio = pd.DataFrame(combined_returns)

    # Save combined portfolio returns
    combined_path = output_dir / 'combined_all_themes_portfolio_returns.csv'
    combined_portfolio.to_csv(combined_path, index=False)
    logger.info(f"✓ Saved combined portfolio returns to {combined_path}")

    # Print summary
    logger.info("\nCombined Portfolio Performance (All Themes):")
    for bucket in ['Low', 'Medium', 'High']:
        bucket_data = combined_portfolio[combined_portfolio['bucket'] == bucket]
        if not bucket_data.empty:
            max_day = min(bucket_data['days_from_event'].max(), 90)
            final_return = bucket_data[bucket_data['days_from_event'] == max_day]['cumulative_return'].values[0]
            logger.info(f"  {bucket} Sentiment: {final_return:+.4%} ({max_day} days)")

    # Calculate spread
    high_data = combined_portfolio[combined_portfolio['bucket'] == 'High']
    low_data = combined_portfolio[combined_portfolio['bucket'] == 'Low']
    if not high_data.empty and not low_data.empty:
        max_day = min(high_data['days_from_event'].max(), 90)
        high_return = high_data[high_data['days_from_event'] == max_day]['cumulative_return'].values[0]
        low_return = low_data[low_data['days_from_event'] == max_day]['cumulative_return'].values[0]
        logger.info(f"  High-Low Spread: {high_return - low_return:+.4%}")
    logger.info("="*80)

    return combined_portfolio


def run_portfolio_sorts_for_theme(theme_id, theme_name, events, results_dir, wrds_conn=None):
    """
    Run portfolio sorts for a single theme.

    Args:
        theme_id: Theme identifier
        theme_name: Theme name (string or list - will be converted to string)
        events: List of event dictionaries with permno, edate, sentiment
        results_dir: Directory to save results
        wrds_conn: Shared WRDS connection

    Returns:
        DataFrame with portfolio returns or None if failed
    """
    # Ensure theme_name is a string
    if isinstance(theme_name, list):
        theme_name = theme_name[0] if len(theme_name) > 0 else "unknown_theme"

    logger.info(f"\n  Running portfolio sorts for {theme_name}...")

    # Extract only the required fields
    portfolio_inputs = []
    for event in events:
        portfolio_inputs.append({
            'permno': event['permno'],
            'edate': event['edate'],
            'sentiment': event['sentiment']
        })

    try:
        # Initialize portfolio sorts
        portfolio = PortfolioSorts(
            portfolio_inputs,
            wrds_connection=wrds_conn,
            weighting=config.WEIGHTING
        )

        # Pull CRSP returns data (will auto-download from WRDS to CSV)
        portfolio.crspreturns()

        # Compute portfolio returns
        portfolio_returns = portfolio.compute_portfolio_returns()

        if portfolio_returns is not None:
            # Create theme-specific directory
            theme_dir = results_dir / 'by_theme'
            theme_dir.mkdir(parents=True, exist_ok=True)

            # Save portfolio returns
            csv_path = theme_dir / f'{theme_id}_{theme_name.replace(" ", "_")}_portfolio_returns.csv'
            portfolio_returns.to_csv(csv_path, index=False)

            logger.info(f"  ✓ Portfolio sorts complete for {theme_name}")
            return portfolio_returns
        else:
            logger.warning(f"  Portfolio sorts produced no results for {theme_name}")
            return None

    except Exception as e:
        logger.error(f"  Portfolio sorts failed for {theme_name}: {e}", exc_info=True)
        return None


def generate_summary_report(sentiment_results, event_study_models, portfolio_results, output_dir):
    """Generate summary report of pipeline results."""
    logger.info("=" * 80)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("=" * 80)

    report = []
    report.append("=" * 80)
    report.append("PIPELINE SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    # Sentiment Analysis Summary
    all_events = sentiment_results['all_events']
    sentiments = [e['sentiment'] for e in all_events]

    report.append("SENTIMENT ANALYSIS:")
    report.append(f"  Total themes: {len(sentiment_results['by_theme'])}")
    report.append(f"  Total events: {len(all_events)}")
    report.append(f"  Mean sentiment: {np.mean(sentiments):+.4f}")
    report.append(f"  Std sentiment: {np.std(sentiments):.4f}")
    report.append(f"  Positive events: {sum(1 for s in sentiments if s > 0)}")
    report.append(f"  Negative events: {sum(1 for s in sentiments if s < 0)}")
    report.append("")

    # Event Study Summary (per theme)
    if event_study_models:
        report.append(f"EVENT STUDIES (BY THEME):")
        report.append(f"  Themes processed: {len(event_study_models)}")
        report.append("")

        for theme_id, model in event_study_models.items():
            theme_name_raw = sentiment_results['by_theme'][theme_id]['theme_name']
            theme_name = theme_name_raw[0] if isinstance(theme_name_raw, list) else theme_name_raw
            if hasattr(model, 'model') and model.model is not None:
                sentiment_coef = model.model.params.get('sentiment', np.nan)
                sentiment_pval = model.model.pvalues.get('sentiment', np.nan)
                sig = ''
                if sentiment_pval < 0.001:
                    sig = '***'
                elif sentiment_pval < 0.01:
                    sig = '**'
                elif sentiment_pval < 0.05:
                    sig = '*'

                report.append(f"  {theme_name}:")
                report.append(f"    Sentiment coefficient: {sentiment_coef:+.6f}{sig}")
                report.append(f"    p-value: {sentiment_pval:.4f}")
        report.append("")

    # Portfolio Sorts Summary (per theme)
    if portfolio_results:
        report.append(f"PORTFOLIO SORTS (BY THEME):")
        report.append(f"  Themes processed: {len(portfolio_results)}")
        report.append("")

        for theme_id, portfolio_df in portfolio_results.items():
            theme_name = sentiment_results['by_theme'][theme_id]['theme_name']
            report.append(f"  {theme_name}:")

            for bucket in ['Low', 'Medium', 'High']:
                bucket_data = portfolio_df[portfolio_df['bucket'] == bucket]
                if not bucket_data.empty:
                    max_day = bucket_data['days_from_event'].max()
                    final_return = bucket_data[bucket_data['days_from_event'] == max_day]['cumulative_return'].values[0]
                    report.append(f"    {bucket} Sentiment: {final_return:+.4%}")

            # Calculate spread
            high_data = portfolio_df[portfolio_df['bucket'] == 'High']
            low_data = portfolio_df[portfolio_df['bucket'] == 'Low']
            if not high_data.empty and not low_data.empty:
                max_day = high_data['days_from_event'].max()
                high_return = high_data[high_data['days_from_event'] == max_day]['cumulative_return'].values[0]
                low_return = low_data[low_data['days_from_event'] == max_day]['cumulative_return'].values[0]
                spread = high_return - low_return
                report.append(f"    High-Low Spread: {spread:+.4%}")
            report.append("")

    report.append("=" * 80)
    report.append("Pipeline completed successfully!")
    report.append("=" * 80)

    # Print report
    report_text = "\n".join(report)
    print("\n" + report_text)

    # Save report
    report_file = output_dir / 'pipeline_summary.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    logger.info(f"✓ Summary report saved to {report_file}")


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='Run complete thematic sentiment analysis pipeline')
    parser.add_argument('--themes_file', type=str, required=True,
                        help='Path to earnings call themes JSON file (with PERMNOs)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results (default: results/)')
    parser.add_argument('--skip_sentiment', action='store_true',
                        help='Skip sentiment analysis (use existing sentiment_scores.csv)')
    parser.add_argument('--skip_event_study', action='store_true',
                        help='Skip event study')
    parser.add_argument('--skip_portfolio', action='store_true',
                        help='Skip portfolio sorts')

    args = parser.parse_args()

    # Create timestamped output directory
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config.LOGS_DIR).mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Validate configuration
    config.validate_config()

    # Initialize variables
    sentiment_results = None
    event_study_models = {}
    portfolio_results = {}
    wrds_connection = None

    try:
        # Stage 1: Sentiment Analysis
        if not args.skip_sentiment:
            themes_data = validate_input_file(args.themes_file)
            sentiment_results = run_sentiment_analysis(themes_data, output_dir)
        else:
            logger.info("Skipping sentiment analysis...")
            # Load from existing CSV and reconstruct by_theme structure
            logger.error("--skip_sentiment requires loading existing sentiment results, not yet implemented")
            sys.exit(1)

        # Establish WRDS connection for stages 2 and 3
        if not args.skip_event_study or not args.skip_portfolio:
            logger.info("Establishing WRDS connection...")
            import wrds
            wrds_connection = wrds.Connection()
            logger.info("✓ WRDS connection established")

        # Create event study output directory
        event_study_dir = output_dir / 'event_study'
        event_study_dir.mkdir(parents=True, exist_ok=True)

        # Create portfolio sorts output directory
        portfolio_dir = output_dir / 'portfolio_sorts'
        portfolio_dir.mkdir(parents=True, exist_ok=True)

        # Process themes
        themes = sentiment_results.get('by_theme', {})
        logger.info(f"\nProcessing {len(themes)} themes...")

        # Stage 2: BATCHED Event Study (all themes together)
        batched_event_study_results = None
        if not args.skip_event_study:
            # Collect ALL events across all themes and deduplicate by (permno, edate)
            logger.info("\nCollecting events across all themes...")

            unique_events_dict = {}  # {(permno, edate): {'event': {...}, 'theme_sentiments': [(theme_id, sentiment), ...]}}
            theme_event_mapping = {}  # {theme_id: [(permno, edate, sentiment), ...]}

            for theme_id, theme_data in themes.items():
                events = theme_data['events']
                theme_event_mapping[theme_id] = []

                if events:
                    for event in events:
                        permno = event['permno']
                        edate = event['edate']
                        sentiment = event['sentiment']
                        key = (permno, edate)

                        # Store for theme mapping (with sentiment - needed for filtering later)
                        theme_event_mapping[theme_id].append((permno, pd.to_datetime(edate), sentiment))

                        # Deduplicate for WRDS query
                        if key not in unique_events_dict:
                            unique_events_dict[key] = {
                                'event': {'permno': permno, 'edate': edate, 'sentiment': 0},  # placeholder
                                'theme_sentiments': []
                            }

                        # Track all (theme_id, sentiment) pairs for this event
                        unique_events_dict[key]['theme_sentiments'].append((theme_id, sentiment))

            # Extract unique events for batched query
            unique_events = [item['event'] for item in unique_events_dict.values()]

            total_events = sum(len(events) for events in theme_event_mapping.values())
            logger.info(f"  Total events across themes: {total_events}")
            logger.info(f"  Unique (permno, edate) pairs: {len(unique_events)}")
            if total_events > len(unique_events):
                logger.info(f"  ✓ Deduplication will save {total_events - len(unique_events)} redundant queries ({100*(total_events - len(unique_events))/total_events:.1f}%)")

            if unique_events:
                # Run batched event study ONCE for all UNIQUE events
                batched_event_study_results = run_batched_event_study(unique_events, wrds_connection)

                # Run per-theme regressions on batched results
                if batched_event_study_results is not None:
                    logger.info("\n" + "="*80)
                    logger.info("RUNNING PER-THEME REGRESSIONS")
                    logger.info("="*80)

                    # Ensure edate is datetime for merging
                    batched_event_study_results['edate'] = pd.to_datetime(
                        batched_event_study_results['edate']
                    )

                    for theme_id, theme_data in themes.items():
                        theme_name_raw = theme_data['theme_name']
                        theme_name = theme_name_raw[0] if isinstance(theme_name_raw, list) else theme_name_raw
                        events = theme_data['events']

                        if events and theme_id in theme_event_mapping:
                            # Get this theme's events with their specific sentiments
                            theme_events = theme_event_mapping[theme_id]  # [(permno, edate, sentiment), ...]

                            # Create a DataFrame for this theme's events with their sentiments
                            theme_events_df = pd.DataFrame(theme_events, columns=['permno', 'edate', 'theme_sentiment'])

                            # Filter batched results to this theme's (permno, edate) pairs
                            theme_permno_dates = [(e[0], e[1]) for e in theme_events]  # Just (permno, edate)

                            # Create filter mask for (permno, edate)
                            mask = batched_event_study_results.apply(
                                lambda row: (row['permno'], row['edate']) in theme_permno_dates,
                                axis=1
                            )
                            theme_results = batched_event_study_results[mask].copy()

                            if not theme_results.empty:
                                # Merge with theme-specific sentiments
                                # Drop the placeholder sentiment column from batched results
                                theme_results = theme_results.drop(columns=['sentiment'], errors='ignore')

                                # Merge with theme-specific sentiment
                                theme_results = theme_results.merge(
                                    theme_events_df,
                                    on=['permno', 'edate'],
                                    how='left'
                                )

                                # Rename theme_sentiment to sentiment
                                theme_results = theme_results.rename(columns={'theme_sentiment': 'sentiment'})

                                # Run regression for this theme
                                study_object = run_regression_for_theme(
                                    theme_id, theme_name, theme_results, event_study_dir
                                )
                                if study_object is not None:
                                    event_study_models[theme_id] = study_object
                            else:
                                logger.warning(f"No batched results found for theme {theme_name}")

                    # Create regression significance summary across all themes
                    if event_study_models:
                        create_regression_significance_summary(
                            event_study_models, themes, event_study_dir
                        )

        # Stage 3: Portfolio Sorts (per theme - still separate because uses different data)
        if not args.skip_portfolio:
            logger.info("\n" + "="*80)
            logger.info("RUNNING PORTFOLIO SORTS")
            logger.info("="*80)

            for theme_id, theme_data in themes.items():
                theme_name_raw = theme_data['theme_name']
                theme_name = theme_name_raw[0] if isinstance(theme_name_raw, list) else theme_name_raw
                events = theme_data['events']

                if events:
                    portfolio_df = run_portfolio_sorts_for_theme(
                        theme_id, theme_name, events, portfolio_dir, wrds_connection
                    )
                    if portfolio_df is not None:
                        portfolio_results[theme_id] = portfolio_df
                else:
                    logger.warning(f"Skipping {theme_name} - no events")

        # Create combined portfolio analysis if we have multiple themes
        if not args.skip_portfolio and len(portfolio_results) > 1:
            combined_portfolio = create_combined_portfolio_analysis(portfolio_results, portfolio_dir)

            # Create time series chart
            if combined_portfolio is not None:
                create_portfolio_time_series_chart(combined_portfolio, portfolio_dir)

        # Generate summary report
        generate_summary_report(sentiment_results, event_study_models, portfolio_results, output_dir)

        logger.info("\n✓ Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Close WRDS connection
        if wrds_connection:
            wrds_connection.close()
            logger.info("WRDS connection closed")


if __name__ == "__main__":
    main()
