"""
Standalone Event Study Runner

This script runs event studies on pre-computed sentiment scores.
Use this to re-run event studies without recomputing sentiment.

Usage:
    python run_event_study.py --sentiment_file results/run_*/sentiment_scores.csv --output_dir results/event_study_rerun/

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
import statsmodels.api as sm

# Add pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent / 'pipeline'))

# Import configuration
import config

# Import pipeline modules
from event_study import ThematicES

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(config.LOGS_DIR) / f'event_study_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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


def create_regression_significance_summary(event_study_models, themes, output_dir):
    """
    Create a summary table ranking themes by sentiment coefficient significance.
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


def run_batched_event_study(all_events_across_themes, wrds_conn=None):
    """
    Run event study ONCE for all events across all themes.

    This batched approach reduces WRDS API calls from 10+ (one per theme) to just 1.
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


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Run event studies on pre-computed sentiment scores')
    parser.add_argument('--sentiment_file', type=str, required=True,
                        help='Path to sentiment_scores.csv file')
    parser.add_argument('--output_dir', type=str, default='results/event_study',
                        help='Output directory for event study results (default: results/event_study)')

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
    event_study_models = {}
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

        # Process themes with batched approach
        logger.info(f"\nProcessing {len(themes)} themes...")

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

                    # Store for theme mapping (with sentiment)
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
                    theme_name = theme_data['theme_name']
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
                                theme_id, theme_name, theme_results, output_dir
                            )
                            if study_object is not None:
                                event_study_models[theme_id] = study_object
                        else:
                            logger.warning(f"No batched results found for theme {theme_name}")

                # Create regression significance summary across all themes
                if event_study_models:
                    create_regression_significance_summary(
                        event_study_models, themes, output_dir
                    )

        # Generate summary
        logger.info("\n" + "="*80)
        logger.info(f"EVENT STUDY SUMMARY")
        logger.info("="*80)
        logger.info(f"Total themes processed: {len(event_study_models)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*80)

        logger.info("\n✓ Event study completed successfully!")

    except Exception as e:
        logger.error(f"Event study failed with error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Close WRDS connection
        if wrds_connection:
            wrds_connection.close()
            logger.info("WRDS connection closed")


if __name__ == "__main__":
    main()
