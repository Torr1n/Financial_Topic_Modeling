"""Shared utility functions for downstream analysis pipeline."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the downstream pipeline.

    Sets up console handler and optionally a file handler with consistent formatting.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to log file. If provided, logs will also be written to file.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def load_sentiment_scores(sentiment_file: str) -> pd.DataFrame:
    """
    Load sentiment scores from a CSV file and validate required columns.

    Args:
        sentiment_file: Path to the CSV file containing sentiment scores

    Returns:
        DataFrame with sentiment scores

    Raises:
        ValueError: If required columns are missing from the CSV
    """
    logger.info(f"Loading sentiment scores from {sentiment_file}")

    df = pd.read_csv(sentiment_file)

    # Validate required columns
    required_cols = ['permno', 'edate', 'sentiment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in sentiment file: {missing_cols}")

    logger.info(f"Loaded {len(df)} sentiment scores")

    return df


def group_by_theme(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Group sentiment scores by theme.

    If theme_id and theme_name columns exist, groups data by theme.
    Otherwise creates a single 'all' group containing all events.

    Args:
        df: DataFrame with sentiment scores (must have permno, edate, sentiment columns)

    Returns:
        Dictionary mapping theme_id to {'theme_name': str, 'events': List[Dict]}
    """
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

        logger.info(f"Found {len(themes)} themes")
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


def run_batched_event_study(
    all_events_across_themes: List[Dict],
    wrds_conn: Optional[Any] = None
) -> Optional[pd.DataFrame]:
    """
    Run event study ONCE for all events across all themes.

    This batched approach reduces WRDS API calls from 10+ (one per theme) to just 1.

    Args:
        all_events_across_themes: List of ALL event dictionaries with permno, edate, sentiment
        wrds_conn: Shared WRDS connection

    Returns:
        DataFrame with covariates, CAR, and sentiment for all events, or None if failed
    """
    from src.event_study import ThematicES

    logger.info("=" * 80)
    logger.info("BATCHED EVENT STUDY (ALL THEMES COMBINED)")
    logger.info("=" * 80)
    logger.info(f"Total events across all themes: {len(all_events_across_themes)}")

    try:
        # Initialize and run event study with ALL events
        logger.info("Running batched event study with covariates...")
        study = ThematicES(all_events_across_themes, wrds_connection=wrds_conn)

        # Run the complete analysis (pulls WRDS data once, calculates factors, computes CAR)
        try:
            results_df = study.doAll()
        except Exception as e:
            # If WRDS query fails, rollback the connection so it can be reused
            if wrds_conn:
                try:
                    wrds_conn.rollback()
                    logger.warning("Rolled back WRDS connection after error")
                except Exception as rollback_error:
                    logger.warning(f"Failed to rollback WRDS connection: {rollback_error}")
            raise e

        if results_df is not None and not results_df.empty:
            logger.info(f"Batched event study complete - {len(results_df)} events processed")
            return results_df
        else:
            logger.error("Batched event study returned no results")
            return None

    except Exception as e:
        logger.error(f"Batched event study failed: {e}", exc_info=True)
        return None


def run_regression_for_theme(
    theme_id: str,
    theme_name: str,
    theme_results_df: pd.DataFrame,
    results_dir: Union[str, Path]
) -> Optional[Any]:
    """
    Run OLS regression for a single theme on batched event study results.

    Args:
        theme_id: Theme identifier
        theme_name: Theme name (string or list - will be converted to string)
        theme_results_df: DataFrame filtered to this theme's events (from batched results)
        results_dir: Directory to save results

    Returns:
        RegressionResult object with model and Results attributes, or None if failed
    """
    # Lazy import to avoid import errors if statsmodels has issues
    import statsmodels.api as sm

    # Ensure theme_name is a string
    if isinstance(theme_name, list):
        theme_name = theme_name[0] if len(theme_name) > 0 else "unknown_theme"

    # Ensure results_dir is a Path
    results_dir = Path(results_dir)

    logger.info("=" * 60)
    logger.info(f"Regression for: {theme_name} ({theme_id})")
    logger.info(f"Events: {len(theme_results_df)}")
    logger.info("=" * 60)

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
        safe_name = theme_name.replace(" ", "_")
        csv_path = theme_dir / f'{theme_id}_{safe_name}_event_study_results.csv'
        theme_results_df.to_csv(csv_path, index=False)
        logger.info(f"  Event study results saved to {csv_path}")

        # Save regression table
        regression_path = theme_dir / f'{theme_id}_{safe_name}_regression_table.txt'
        with open(regression_path, 'w') as f:
            f.write(model.summary().as_text())
        logger.info(f"  Regression table saved to {regression_path}")

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
        logger.info(f"Regression complete for {theme_name}")

        # Create a simple object to hold the model (to match existing interface)
        class RegressionResult:
            def __init__(self, model, results_df):
                self.model = model
                self.Results = results_df

        return RegressionResult(model, theme_results_df)

    except Exception as e:
        logger.error(f"Regression failed for {theme_name}: {e}", exc_info=True)
        return None


def run_portfolio_sorts_for_theme(
    theme_id: str,
    theme_name: str,
    events: List[Dict],
    results_dir: Union[str, Path],
    wrds_conn: Optional[Any] = None,
    weighting: str = "value"
) -> Optional[pd.DataFrame]:
    """
    Run portfolio sorts for a single theme.

    Args:
        theme_id: Theme identifier
        theme_name: Human-readable theme name (string or list - will be converted)
        events: List of event dicts with permno, edate, sentiment
        results_dir: Directory to write results
        wrds_conn: Optional shared WRDS connection
        weighting: 'value' or 'equal' weighted portfolios

    Returns:
        DataFrame with portfolio returns or None if failed
    """
    from src.portfolio_sorts import PortfolioSorts

    # Ensure theme_name is a string
    if isinstance(theme_name, list):
        theme_name = theme_name[0] if len(theme_name) > 0 else "unknown_theme"

    # Ensure results_dir is a Path
    results_dir = Path(results_dir)

    if not events:
        logger.warning(f"No events for theme {theme_id}")
        return None

    logger.info(f"Running portfolio sorts for: {theme_name}")
    logger.info(f"  Events: {len(events)}")

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
        ps = PortfolioSorts(
            portfolio_inputs,
            wrds_connection=wrds_conn,
            weighting=weighting
        )

        # Pull CRSP returns data (will auto-download from WRDS to CSV)
        ps.crspreturns()

        # Compute portfolio returns
        returns_df = ps.compute_portfolio_returns()

        if returns_df is not None and not returns_df.empty:
            # Add theme identifiers
            returns_df['theme_id'] = theme_id
            returns_df['theme_name'] = theme_name

            # Save to file
            theme_dir = results_dir / 'by_theme'
            theme_dir.mkdir(parents=True, exist_ok=True)

            safe_name = "".join(c if c.isalnum() else "_" for c in theme_name[:30])
            output_path = theme_dir / f'{theme_id}_{safe_name}_portfolio_returns.csv'
            returns_df.to_csv(output_path, index=False)
            logger.info(f"  Portfolio sorts complete - saved to {output_path}")

            return returns_df
        else:
            logger.warning(f"  Portfolio sorts produced no results for {theme_name}")
            return None

    except Exception as e:
        logger.error(f"  Portfolio sorts failed for {theme_name}: {e}", exc_info=True)
        return None


def create_regression_significance_summary(
    event_study_models: Dict[str, Any],
    themes: Dict[str, Dict[str, Any]],
    output_dir: Union[str, Path]
) -> Optional[pd.DataFrame]:
    """
    Create a summary table ranking themes by sentiment coefficient significance.

    Args:
        event_study_models: Dictionary of {theme_id: RegressionResult}
        themes: Dictionary of theme data {theme_id: {'theme_name': str, 'events': List}}
        output_dir: Directory to save the summary

    Returns:
        DataFrame with regression results ranked by significance, or None if no valid results
    """
    logger.info("=" * 80)
    logger.info("CREATING REGRESSION SIGNIFICANCE SUMMARY")
    logger.info("=" * 80)

    # Ensure output_dir is a Path
    output_dir = Path(output_dir)

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

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

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
    header = f"{'Rank':<6} {'Theme Name':<50} {'Coef':>10} {'p-value':>10} {'Sig':>5} {'N':>6} {'R^2':>8}"
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
    txt_lines.append(f"  Significant at p<0.05: {(summary_df['p_Value'] < 0.05).sum()} ({100*(summary_df['p_Value'] < 0.05).sum()/len(summary_df):.1f}%)")
    txt_lines.append(f"  Significant at p<0.01: {(summary_df['p_Value'] < 0.01).sum()} ({100*(summary_df['p_Value'] < 0.01).sum()/len(summary_df):.1f}%)")
    txt_lines.append(f"  Significant at p<0.001: {(summary_df['p_Value'] < 0.001).sum()} ({100*(summary_df['p_Value'] < 0.001).sum()/len(summary_df):.1f}%)")
    txt_lines.append("")
    txt_lines.append(f"  Positive coefficients: {(summary_df['Sentiment_Coef'] > 0).sum()} ({100*(summary_df['Sentiment_Coef'] > 0).sum()/len(summary_df):.1f}%)")
    txt_lines.append(f"  Negative coefficients: {(summary_df['Sentiment_Coef'] < 0).sum()} ({100*(summary_df['Sentiment_Coef'] < 0).sum()/len(summary_df):.1f}%)")
    txt_lines.append("")
    txt_lines.append(f"  Mean R^2: {summary_df['R_Squared'].mean():.4f}")
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
        txt_lines.append(f"     R^2: {row['R_Squared']:.4f} (N={row['N_Observations']})")
        txt_lines.append("")

    txt_lines.append("=" * 100)
    txt_lines.append("")
    txt_lines.append("INTERPRETATION GUIDE:")
    txt_lines.append("")
    txt_lines.append("  - Positive coefficient: Higher sentiment -> Higher abnormal returns")
    txt_lines.append("  - Negative coefficient: Higher sentiment -> Lower abnormal returns")
    txt_lines.append("  - p-value < 0.05: Statistically significant relationship (95% confidence)")
    txt_lines.append("  - R^2: Proportion of variance in CAR explained by the model (0-1 scale)")
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
