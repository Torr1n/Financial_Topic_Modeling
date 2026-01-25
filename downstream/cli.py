#!/usr/bin/env python3
"""
Downstream Analysis Pipeline CLI

Run sentiment analysis, event studies, and portfolio sorts on theme data.

Usage:
    python cli.py --themes data/themes.json --output results/
    python cli.py --themes data/themes.json --stages sentiment event_study
    python cli.py --sentiment-file results/sentiment.csv --stages portfolio
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_themes(themes_file: str) -> dict:
    """Load themes JSON file."""
    with open(themes_file, 'r') as f:
        return json.load(f)


def run_sentiment_stage(themes: dict, output_dir: str):
    """
    Run sentiment analysis stage.

    Args:
        themes: Themes dict from JSON
        output_dir: Directory to write results

    Returns:
        Path to sentiment CSV, sentiment DataFrame
    """
    import pandas as pd
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

    return sentiment_path, results


def run_event_study_stage(
    themes: dict,
    sentiment_df,
    output_dir: str,
    wrds_conn=None
) -> dict:
    """
    Run event study regressions.

    Args:
        themes: Themes dict from JSON
        sentiment_df: DataFrame with sentiment scores
        output_dir: Directory to write results
        wrds_conn: Optional shared WRDS connection

    Returns:
        Dict mapping theme_id to regression model
    """
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
    if models:
        create_regression_significance_summary(models, theme_list, es_output_dir)

    return models


def run_portfolio_stage(
    themes: dict,
    sentiment_df,
    output_dir: str,
    wrds_conn=None
) -> None:
    """
    Run portfolio sorts analysis.

    Args:
        themes: Themes dict from JSON
        sentiment_df: DataFrame with sentiment scores
        output_dir: Directory to write results
        wrds_conn: Optional shared WRDS connection
    """
    import pandas as pd
    from src.utils import run_portfolio_sorts_for_theme

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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages
  python cli.py --themes data/themes.json --output results/

  # Run only sentiment analysis (no WRDS needed)
  python cli.py --themes data/themes.json --stages sentiment

  # Run event study on existing sentiment
  python cli.py --sentiment-file results/sentiment.csv --stages event_study
        """
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

    if ('event_study' in args.stages or 'portfolio' in args.stages) and \
       'sentiment' not in args.stages and 'all' not in args.stages:
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
    import pandas as pd
    sentiment_df = None

    try:
        if 'sentiment' in stages:
            sentiment_path, sentiment_df = run_sentiment_stage(themes, output_dir)
        elif args.sentiment_file:
            sentiment_df = pd.read_csv(args.sentiment_file)
            print(f"Loaded existing sentiment from {args.sentiment_file}")

        if 'event_study' in stages and sentiment_df is not None:
            run_event_study_stage(themes, sentiment_df, output_dir, wrds_conn)

        if 'portfolio' in stages and sentiment_df is not None:
            run_portfolio_stage(themes, sentiment_df, output_dir, wrds_conn)

    finally:
        # Cleanup
        if wrds_conn:
            wrds_conn.close()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
