"""
Standalone Portfolio Sorts Runner

This script runs portfolio sorts on pre-computed sentiment scores.
Use this to re-run portfolio analysis without recomputing sentiment.

Usage:
    python run_portfolio_sorts.py --sentiment_file results/run_*/sentiment_scores.csv --output_dir results/portfolio_sorts_rerun/

Author: Team 2 COMM386I
Date: January 2026
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent / 'pipeline'))

# Import configuration
import config

# Import pipeline modules
from portfolio_sorts import PortfolioSorts

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(config.LOGS_DIR) / f'portfolio_sorts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def load_sentiment_scores(sentiment_file):
    """Load sentiment scores from CSV file."""
    logger.info(f"Loading sentiment scores from {sentiment_file}")

    df = pd.read_csv(sentiment_file)

    # Validate required columns
    required_cols = ['permno', 'edate', 'sentiment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in sentiment file: {missing_cols}")

    logger.info(f"✓ Loaded {len(df)} sentiment scores")

    return df


def group_by_theme(df):
    """Group sentiment scores by theme."""
    if 'theme_id' in df.columns and 'theme_name' in df.columns:
        # Group by theme
        themes = {}
        for theme_id in df['theme_id'].unique():
            theme_df = df[df['theme_id'] == theme_id]
            theme_name = theme_df['theme_name'].iloc[0]
            if isinstance(theme_name, list):
                theme_name = theme_name[0] if len(theme_name) > 0 else "unknown_theme"

            events = []
            for _, row in theme_df.iterrows():
                events.append({
                    'permno': row['permno'],
                    'edate': row['edate'],
                    'sentiment': row['sentiment']
                })

            themes[theme_id] = {
                'theme_name': theme_name,
                'events': events
            }

        logger.info(f"✓ Found {len(themes)} themes")
        return themes
    else:
        # Single theme (all events)
        logger.info("No theme columns found - treating as single theme")
        events = []
        for _, row in df.iterrows():
            events.append({
                'permno': row['permno'],
                'edate': row['edate'],
                'sentiment': row['sentiment']
            })

        return {
            'all': {
                'theme_name': 'all_themes',
                'events': events
            }
        }


def create_portfolio_time_series_chart(combined_portfolio, output_dir):
    """
    Create line chart showing cumulative returns by sentiment bucket over time.
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

    Args:
        portfolio_results: Dictionary of {theme_id: portfolio_returns_df}
        output_dir: Directory to save combined results

    Returns:
        DataFrame with combined portfolio returns
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING COMBINED PORTFOLIO ANALYSIS")
    logger.info("="*80)

    if not portfolio_results:
        logger.warning("No portfolio results to combine")
        return None

    # Combine all theme portfolio returns
    all_returns = []
    for theme_id, returns_df in portfolio_results.items():
        if returns_df is not None and not returns_df.empty:
            returns_copy = returns_df.copy()
            returns_copy['theme_id'] = theme_id
            all_returns.append(returns_copy)

    if not all_returns:
        logger.warning("No valid portfolio returns to combine")
        return None

    combined_df = pd.concat(all_returns, ignore_index=True)

    # Calculate combined returns for each bucket and day
    # Average across all themes
    combined_returns = []

    for bucket in ['Low', 'Medium', 'High']:
        bucket_data = combined_df[combined_df['bucket'] == bucket]

        # Get unique days (limit to 90)
        unique_days = sorted(bucket_data['days_from_event'].unique())
        unique_days = [d for d in unique_days if 0 <= d <= 90]

        for day in unique_days:
            day_data = bucket_data[bucket_data['days_from_event'] == day]

            if not day_data.empty:
                # Average cumulative return across all themes for this day
                avg_cumulative_return = day_data['cumulative_return'].mean()
                avg_daily_return = day_data['vw_return'].mean()
                n_themes = day_data['theme_id'].nunique()

                combined_returns.append({
                    'bucket': bucket,
                    'days_from_event': day,
                    'vw_return': avg_daily_return,
                    'cumulative_return': avg_cumulative_return,
                    'n_themes': n_themes
                })

    combined_portfolio = pd.DataFrame(combined_returns)

    # Save combined portfolio returns
    combined_path = output_dir / 'combined_all_themes_portfolio_returns.csv'
    combined_portfolio.to_csv(combined_path, index=False)
    logger.info(f"✓ Saved combined portfolio returns to {combined_path}")

    # Create performance summary
    summary_data = []

    for bucket in ['Low', 'Medium', 'High']:
        bucket_data = combined_portfolio[combined_portfolio['bucket'] == bucket]

        if not bucket_data.empty:
            # Get returns at key horizons
            horizons = [30, 60, 90]
            row_data = {'Sentiment_Bucket': bucket}

            for horizon in horizons:
                horizon_data = bucket_data[bucket_data['days_from_event'] == horizon]
                if not horizon_data.empty:
                    cum_return = horizon_data['cumulative_return'].values[0]
                    row_data[f'{horizon}d_Return_%'] = cum_return * 100
                else:
                    row_data[f'{horizon}d_Return_%'] = np.nan

            # Final return (max day available, up to 90)
            max_day = min(bucket_data['days_from_event'].max(), 90)
            final_data = bucket_data[bucket_data['days_from_event'] == max_day]
            if not final_data.empty:
                row_data['Final_Return_%'] = final_data['cumulative_return'].values[0] * 100
                row_data['Final_Day'] = max_day

            summary_data.append(row_data)

    # Calculate High-Low spread
    if len(summary_data) == 3:
        high_row = summary_data[2]  # High is last
        low_row = summary_data[0]   # Low is first

        spread_row = {'Sentiment_Bucket': 'High-Low Spread'}
        for key in high_row.keys():
            if key != 'Sentiment_Bucket' and key != 'Final_Day':
                if key in low_row and not pd.isna(high_row[key]) and not pd.isna(low_row[key]):
                    spread_row[key] = high_row[key] - low_row[key]
                else:
                    spread_row[key] = np.nan

        summary_data.append(spread_row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'combined_all_themes_performance_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"✓ Saved combined performance summary to {summary_path}")

    # Print summary
    logger.info("\nCombined Portfolio Performance (All Themes):")
    logger.info("-" * 60)
    for bucket in ['Low', 'Medium', 'High']:
        bucket_data = combined_portfolio[combined_portfolio['bucket'] == bucket]
        if not bucket_data.empty:
            max_day = min(bucket_data['days_from_event'].max(), 90)
            final_return = bucket_data[bucket_data['days_from_event'] == max_day]['cumulative_return'].values[0]
            logger.info(f"  {bucket} Sentiment: {final_return:+.4%} ({max_day} days)")

    # Calculate and print spread
    high_data = combined_portfolio[combined_portfolio['bucket'] == 'High']
    low_data = combined_portfolio[combined_portfolio['bucket'] == 'Low']
    if not high_data.empty and not low_data.empty:
        max_day = min(high_data['days_from_event'].max(), 90)
        high_return = high_data[high_data['days_from_event'] == max_day]['cumulative_return'].values[0]
        low_return = low_data[low_data['days_from_event'] == max_day]['cumulative_return'].values[0]
        spread = high_return - low_return
        logger.info(f"  High-Low Spread: {spread:+.4%}")

    logger.info("="*80)

    return combined_portfolio


def run_portfolio_sorts_for_theme(theme_id, theme_name, events, results_dir, wrds_conn=None):
    """
    Run portfolio sorts for a single theme.
    """
    # Ensure theme_name is a string
    if isinstance(theme_name, list):
        theme_name = theme_name[0] if len(theme_name) > 0 else "unknown_theme"

    logger.info(f"\n  Running portfolio sorts for {theme_name}...")

    try:
        # Initialize portfolio sorts
        portfolio = PortfolioSorts(
            events,
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


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Run portfolio sorts on pre-computed sentiment scores')
    parser.add_argument('--sentiment_file', type=str, required=True,
                        help='Path to sentiment_scores.csv file')
    parser.add_argument('--output_dir', type=str, default='results/portfolio_sorts',
                        help='Output directory for portfolio results (default: results/portfolio_sorts)')
    parser.add_argument('--weighting', type=str, default='value', choices=['value', 'equal'],
                        help='Portfolio weighting method: value (market-cap weighted) or equal (default: value)')

    args = parser.parse_args()

    # Override config weighting if specified
    if args.weighting:
        config.WEIGHTING = args.weighting

    # Create timestamped output directory
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config.LOGS_DIR).mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Weighting method: {config.WEIGHTING}")

    # Validate configuration
    config.validate_config()

    # Initialize variables
    portfolio_results = {}
    wrds_connection = None

    try:
        # Load sentiment scores
        sentiment_df = load_sentiment_scores(args.sentiment_file)

        # Group by theme
        themes = group_by_theme(sentiment_df)

        # Establish WRDS connection
        logger.info("Establishing WRDS connection...")
        import wrds
        wrds_connection = wrds.Connection()
        logger.info("✓ WRDS connection established")

        # Process each theme
        logger.info(f"\nProcessing {len(themes)} themes...")

        for theme_id, theme_data in themes.items():
            theme_name = theme_data['theme_name']
            events = theme_data['events']

            if events:
                portfolio_df = run_portfolio_sorts_for_theme(
                    theme_id, theme_name, events, output_dir, wrds_connection
                )
                if portfolio_df is not None:
                    portfolio_results[theme_id] = portfolio_df
            else:
                logger.warning(f"Skipping {theme_name} - no events")

        # Create combined portfolio analysis across all themes
        if len(portfolio_results) > 1:
            combined_portfolio = create_combined_portfolio_analysis(portfolio_results, output_dir)

            # Create time series chart
            if combined_portfolio is not None:
                create_portfolio_time_series_chart(combined_portfolio, output_dir)
        else:
            logger.info("\nSkipping combined analysis - only 1 theme (no need to combine)")
            combined_portfolio = None

        # Generate summary
        logger.info("\n" + "="*80)
        logger.info(f"PORTFOLIO SORTS SUMMARY (PER THEME)")
        logger.info("="*80)
        logger.info(f"Total themes processed: {len(portfolio_results)}")
        logger.info(f"Weighting method: {config.WEIGHTING}")
        logger.info(f"Output directory: {output_dir}")

        # Print returns summary
        for theme_id, portfolio_df in portfolio_results.items():
            theme_name = themes[theme_id]['theme_name']
            logger.info(f"\n  {theme_name}:")

            for bucket in ['Low', 'Medium', 'High']:
                bucket_data = portfolio_df[portfolio_df['bucket'] == bucket]
                if not bucket_data.empty:
                    max_day = bucket_data['days_from_event'].max()
                    final_return = bucket_data[bucket_data['days_from_event'] == max_day]['cumulative_return'].values[0]
                    logger.info(f"    {bucket} Sentiment: {final_return:+.4%}")

            # Calculate spread
            high_data = portfolio_df[portfolio_df['bucket'] == 'High']
            low_data = portfolio_df[portfolio_df['bucket'] == 'Low']
            if not high_data.empty and not low_data.empty:
                max_day = high_data['days_from_event'].max()
                high_return = high_data[high_data['days_from_event'] == max_day]['cumulative_return'].values[0]
                low_return = low_data[low_data['days_from_event'] == max_day]['cumulative_return'].values[0]
                spread = high_return - low_return
                logger.info(f"    High-Low Spread: {spread:+.4%}")

        logger.info("="*80)

        logger.info("\n✓ Portfolio sorts completed successfully!")

    except Exception as e:
        logger.error(f"Portfolio sorts failed with error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Close WRDS connection
        if wrds_connection:
            wrds_connection.close()
            logger.info("WRDS connection closed")


if __name__ == "__main__":
    main()
