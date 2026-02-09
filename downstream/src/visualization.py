"""Visualization functions for downstream analysis pipeline.

This module contains functions for creating charts, summary tables, and
visualizations from event study and portfolio analysis results.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_regression_significance_summary(
    event_study_models: Dict[str, Any],
    themes: Dict[str, Dict],
    output_dir: Path
) -> Optional[pd.DataFrame]:
    """
    Create a summary table ranking themes by sentiment coefficient significance.

    Generates both CSV and formatted text outputs summarizing regression results
    across all themes, sorted by p-value (most significant first).

    Args:
        event_study_models: Dictionary of {theme_id: RegressionResult} where
            RegressionResult has a .model attribute containing statsmodels OLS results.
        themes: Dictionary of theme data {theme_id: {'theme_name': str, ...}}.
        output_dir: Path to directory for saving output files.

    Returns:
        DataFrame with regression results ranked by significance, or None if no
        valid models were provided.
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING REGRESSION SIGNIFICANCE SUMMARY")
    logger.info("=" * 80)

    if not event_study_models:
        logger.warning("No event study models to summarize")
        return None

    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect regression statistics for all themes
    summary_data = []

    for theme_id, result_obj in event_study_models.items():
        if result_obj is None or not hasattr(result_obj, 'model') or result_obj.model is None:
            continue

        model = result_obj.model
        theme_name_raw = themes.get(theme_id, {}).get('theme_name', theme_id)
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
    logger.info(f"Saved CSV summary to {csv_path}")

    # Create formatted text summary
    txt_lines = []
    txt_lines.append("=" * 100)
    txt_lines.append("SENTIMENT COEFFICIENT SIGNIFICANCE SUMMARY (RANKED BY P-VALUE)")
    txt_lines.append("=" * 100)
    txt_lines.append("")
    txt_lines.append("This table ranks themes by the statistical significance of their sentiment coefficient")
    txt_lines.append("in predicting cumulative abnormal returns (CAR).")
    txt_lines.append("")
    txt_lines.append("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    txt_lines.append("")
    txt_lines.append("-" * 100)

    # Header
    header = f"{'Rank':<6} {'Theme Name':<50} {'Coef':>10} {'p-value':>10} {'Sig':>5} {'N':>6} {'R2':>8}"
    txt_lines.append(header)
    txt_lines.append("-" * 100)

    # Data rows
    for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
        theme_name = row['Theme_Name'][:47] + '...' if len(row['Theme_Name']) > 50 else row['Theme_Name']

        line = f"{rank:<6} {theme_name:<50} {row['Sentiment_Coef']:>+10.6f} {row['p_Value']:>10.4f} {row['Significance']:>5} {row['N_Observations']:>6} {row['R_Squared']:>8.4f}"
        txt_lines.append(line)

    txt_lines.append("-" * 100)
    txt_lines.append("")

    # Summary statistics
    txt_lines.append("SUMMARY STATISTICS:")
    txt_lines.append("")
    txt_lines.append(f"  Total themes analyzed: {len(summary_df)}")
    txt_lines.append(f"  Significant at p<0.05: {(summary_df['p_Value'] < 0.05).sum()} ({100 * (summary_df['p_Value'] < 0.05).sum() / len(summary_df):.1f}%)")
    txt_lines.append(f"  Significant at p<0.01: {(summary_df['p_Value'] < 0.01).sum()} ({100 * (summary_df['p_Value'] < 0.01).sum() / len(summary_df):.1f}%)")
    txt_lines.append(f"  Significant at p<0.001: {(summary_df['p_Value'] < 0.001).sum()} ({100 * (summary_df['p_Value'] < 0.001).sum() / len(summary_df):.1f}%)")
    txt_lines.append("")
    txt_lines.append(f"  Positive coefficients: {(summary_df['Sentiment_Coef'] > 0).sum()} ({100 * (summary_df['Sentiment_Coef'] > 0).sum() / len(summary_df):.1f}%)")
    txt_lines.append(f"  Negative coefficients: {(summary_df['Sentiment_Coef'] < 0).sum()} ({100 * (summary_df['Sentiment_Coef'] < 0).sum() / len(summary_df):.1f}%)")
    txt_lines.append("")
    txt_lines.append(f"  Mean R-squared: {summary_df['R_Squared'].mean():.4f}")
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
        txt_lines.append(f"     R-squared: {row['R_Squared']:.4f} (N={row['N_Observations']})")
        txt_lines.append("")

    txt_lines.append("=" * 100)
    txt_lines.append("")
    txt_lines.append("INTERPRETATION GUIDE:")
    txt_lines.append("")
    txt_lines.append("  - Positive coefficient: Higher sentiment -> Higher abnormal returns")
    txt_lines.append("  - Negative coefficient: Higher sentiment -> Lower abnormal returns")
    txt_lines.append("  - p-value < 0.05: Statistically significant relationship (95% confidence)")
    txt_lines.append("  - R-squared: Proportion of variance in CAR explained by the model (0-1 scale)")
    txt_lines.append("  - t-statistic: Coefficient divided by standard error (larger |t| = stronger evidence)")
    txt_lines.append("")
    txt_lines.append("=" * 100)

    # Save text summary
    txt_path = output_dir / 'regression_significance_summary.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(txt_lines))
    logger.info(f"Saved text summary to {txt_path}")

    # Print top 5 to console
    logger.info("\nTop 5 Most Significant Themes:")
    logger.info("-" * 60)
    for rank, (_, row) in enumerate(summary_df.head(5).iterrows(), 1):
        logger.info(f"{rank}. {row['Theme_Name'][:50]}")
        logger.info(f"   Coefficient: {row['Sentiment_Coef']:+.6f}{row['Significance']} (p={row['p_Value']:.4f})")

    logger.info("=" * 80)

    return summary_df


def create_portfolio_time_series_chart(
    combined_portfolio: Optional[pd.DataFrame],
    output_dir: Path
) -> None:
    """
    Create line chart showing cumulative returns by sentiment bucket over time.

    Generates a PNG chart with separate lines for Low, Medium, and High sentiment
    portfolios, including a summary text box with final returns and spread.

    Args:
        combined_portfolio: DataFrame with columns: bucket, days_from_event,
            vw_return, cumulative_return. Can be None or empty.
        output_dir: Path to directory for saving the chart PNG.
    """
    if combined_portfolio is None or combined_portfolio.empty:
        logger.warning("No portfolio data to chart")
        return

    logger.info("\nCreating portfolio time series chart...")

    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
                summary_text += f"{labels[bucket]}: {final_return[0] * 100:+.2f}%\n"

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

    logger.info(f"Saved portfolio time series chart to {chart_path}")


def create_combined_portfolio_analysis(
    portfolio_results: Dict[str, pd.DataFrame],
    output_dir: Path
) -> Optional[pd.DataFrame]:
    """
    Create combined portfolio analysis across all themes.

    Aggregates portfolio returns from all themes to show overall performance
    by averaging cumulative returns across themes for each sentiment bucket
    and time horizon.

    Args:
        portfolio_results: Dictionary of {theme_id: portfolio_returns_df} where
            each DataFrame has columns: bucket, days_from_event, vw_return,
            cumulative_return.
        output_dir: Path to directory for saving output files.

    Returns:
        DataFrame with combined portfolio returns averaged across themes,
        or None if no valid results were provided.
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING COMBINED PORTFOLIO ANALYSIS")
    logger.info("=" * 80)

    if not portfolio_results:
        logger.warning("No portfolio results to combine")
        return None

    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    logger.info(f"Saved combined portfolio returns to {combined_path}")

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
    logger.info("=" * 80)

    return combined_portfolio


def plot_sentiment_by_theme(
    sentiment_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create box plot showing sentiment score distributions grouped by theme.

    Each theme gets one box on the x-axis, sorted by median sentiment from
    lowest (left) to highest (right). Boxes are colored red-ish for themes
    with negative median sentiment and green-ish for positive median sentiment.

    Args:
        sentiment_df: DataFrame with columns: theme_id, theme_name,
            sentiment_score (one row per firm-theme observation).
        output_dir: Path to directory for saving the chart PNG.
    """
    if sentiment_df is None or sentiment_df.empty:
        logger.warning("No sentiment data to plot")
        return

    logger.info("Creating sentiment-by-theme box plot...")

    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - skipping chart generation")
        logger.warning("Install with: pip install matplotlib")
        return

    # Compute median sentiment per theme for sorting and coloring
    theme_medians = (
        sentiment_df
        .groupby(['theme_id', 'theme_name'])['sentiment_score']
        .median()
        .reset_index()
        .rename(columns={'sentiment_score': 'median_sentiment'})
        .sort_values('median_sentiment')
    )

    # Ordered theme names (lowest median first)
    ordered_theme_names = theme_medians['theme_name'].tolist()
    ordered_theme_ids = theme_medians['theme_id'].tolist()

    # Build list of data arrays in sorted order
    data_by_theme = []
    for theme_id in ordered_theme_ids:
        theme_data = sentiment_df.loc[
            sentiment_df['theme_id'] == theme_id, 'sentiment_score'
        ].dropna().values
        data_by_theme.append(theme_data)

    # Assign colors based on median sign
    box_colors = [
        '#d96060' if med < 0 else '#5db85d'
        for med in theme_medians['median_sentiment'].values
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    bp = ax.boxplot(
        data_by_theme,
        labels=ordered_theme_names,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )

    # Color each box
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Neutral sentiment reference line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)

    ax.set_title('Sentiment Distribution by Theme', fontsize=14, fontweight='bold')
    ax.set_xlabel('Theme', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    chart_path = output_dir / 'sentiment_by_theme.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved sentiment-by-theme box plot to {chart_path}")


def plot_event_study_car(
    car_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create line plot of cumulative abnormal returns around earnings events.

    If the input contains a ``sentiment_bucket`` column the plot draws
    separate lines for Low, Medium, and High sentiment groups; otherwise a
    single aggregate line is shown. Shaded bands represent +/-1.96 standard
    errors (approximate 95% confidence interval).

    Args:
        car_df: DataFrame with columns: day (relative to event, e.g. -10 to
            +10), car_mean, car_se (standard error). Optionally includes
            sentiment_bucket (Low, Medium, High).
        output_dir: Path to directory for saving the chart PNG.
    """
    if car_df is None or car_df.empty:
        logger.warning("No CAR data to plot")
        return

    logger.info("Creating event study CAR plot...")

    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - skipping chart generation")
        logger.warning("Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    has_buckets = 'sentiment_bucket' in car_df.columns

    if has_buckets:
        bucket_colors = {'Low': '#d62728', 'Medium': '#ff7f0e', 'High': '#2ca02c'}
        bucket_labels = {'Low': 'Low Sentiment', 'Medium': 'Medium Sentiment', 'High': 'High Sentiment'}

        for bucket in ['Low', 'Medium', 'High']:
            bucket_data = car_df[car_df['sentiment_bucket'] == bucket].sort_values('day')
            if bucket_data.empty:
                continue

            days = bucket_data['day'].values
            car_mean = bucket_data['car_mean'].values
            car_se = bucket_data['car_se'].values
            color = bucket_colors[bucket]

            ax.plot(
                days, car_mean,
                label=bucket_labels[bucket],
                color=color,
                linewidth=2.0,
            )
            ax.fill_between(
                days,
                car_mean - 1.96 * car_se,
                car_mean + 1.96 * car_se,
                color=color,
                alpha=0.15,
            )
    else:
        car_df_sorted = car_df.sort_values('day')
        days = car_df_sorted['day'].values
        car_mean = car_df_sorted['car_mean'].values
        car_se = car_df_sorted['car_se'].values

        ax.plot(
            days, car_mean,
            label='CAR',
            color='#1f77b4',
            linewidth=2.0,
        )
        ax.fill_between(
            days,
            car_mean - 1.96 * car_se,
            car_mean + 1.96 * car_se,
            color='#1f77b4',
            alpha=0.15,
        )

    # Event date and zero-CAR reference lines
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)

    ax.set_title(
        'Cumulative Abnormal Returns Around Earnings Announcements',
        fontsize=14, fontweight='bold',
    )
    ax.set_xlabel('Days Relative to Event', fontsize=12, fontweight='bold')
    ax.set_ylabel('CAR (%)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    chart_path = output_dir / 'event_study_car.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved event study CAR plot to {chart_path}")


def plot_regression_forest(
    regression_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create a forest plot (horizontal dot-and-whisker) of regression coefficients.

    Each theme is a row on the y-axis with a dot at its sentiment coefficient
    and a horizontal line spanning the 95% confidence interval. Themes are
    sorted by coefficient value (most negative at bottom, most positive at
    top). Dots are filled dark when significant at p < 0.05 and hollow / light
    gray otherwise.

    Args:
        regression_df: DataFrame with columns: theme_name, coefficient,
            ci_lower, ci_upper, p_value.
        output_dir: Path to directory for saving the chart PNG.
    """
    if regression_df is None or regression_df.empty:
        logger.warning("No regression data to plot")
        return

    logger.info("Creating regression forest plot...")

    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - skipping chart generation")
        logger.warning("Install with: pip install matplotlib")
        return

    # Sort by coefficient (most negative at bottom so it reads naturally)
    df_sorted = regression_df.sort_values('coefficient').reset_index(drop=True)

    n_themes = len(df_sorted)
    fig_height = max(8, n_themes * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n_themes)

    for i, row in df_sorted.iterrows():
        significant = row['p_value'] < 0.05

        # Confidence interval whisker
        ax.plot(
            [row['ci_lower'], row['ci_upper']],
            [i, i],
            color='#333333' if significant else '#bbbbbb',
            linewidth=1.5,
            solid_capstyle='round',
        )

        # Coefficient dot
        ax.plot(
            row['coefficient'], i,
            marker='o',
            markersize=7,
            color='#1a1a1a' if significant else 'none',
            markeredgecolor='#1a1a1a' if significant else '#aaaaaa',
            markeredgewidth=1.5,
        )

    # Zero reference line
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_sorted['theme_name'].values, fontsize=10)
    ax.set_xlabel('Sentiment Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(
        'Sentiment Coefficient by Theme (with 95% CI)',
        fontsize=14, fontweight='bold',
    )
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    # Annotation in the bottom margin
    fig.text(
        0.5, 0.01,
        'Filled dots = significant at p<0.05',
        ha='center', fontsize=9, fontstyle='italic', color='#555555',
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    chart_path = output_dir / 'regression_forest_plot.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved regression forest plot to {chart_path}")
