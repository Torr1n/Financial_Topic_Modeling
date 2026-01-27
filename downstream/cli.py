#!/usr/bin/env python3
"""
Downstream Analysis Pipeline CLI

Primary entry point for running sentiment analysis, event studies, and portfolio sorts
on theme data. This module provides subcommands for different analysis tasks.

Usage:
    python cli.py run --themes data/themes.json --output results/
    python cli.py run --themes data/themes.json --stages sentiment event_study
    python cli.py event-study --sentiment-file results/sentiment.csv --output results/
    python cli.py portfolio --sentiment-file results/sentiment.csv --output results/
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')  # Load from project root

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import config
from src.utils import (
    setup_logging,
    load_sentiment_scores,
    group_by_theme,
    run_batched_event_study,
    run_regression_for_theme,
    run_portfolio_sorts_for_theme,
    create_regression_significance_summary,
)
from src.visualization import (
    create_portfolio_time_series_chart,
    create_combined_portfolio_analysis,
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

    analyzer = ThematicSentimentAnalyzer()
    results = analyzer.analyze_themes(themes)

    # Convert all_events list to DataFrame
    sentiment_df = pd.DataFrame(results['all_events'])

    # Extract theme_id and theme_name from metadata if present
    if 'metadata' in sentiment_df.columns:
        sentiment_df['theme_id'] = sentiment_df['metadata'].apply(
            lambda x: x.get('theme_id') if isinstance(x, dict) else None
        )
        sentiment_df['theme_name'] = sentiment_df['metadata'].apply(
            lambda x: x.get('theme_name') if isinstance(x, dict) else None
        )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    sentiment_path = os.path.join(output_dir, 'sentiment_scores.csv')
    sentiment_df.to_csv(sentiment_path, index=False)
    print(f"Sentiment scores saved to {sentiment_path}")

    return sentiment_path, sentiment_df


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
    import pandas as pd
    from src.event_study import ThematicES

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
            # Prepare event study data (pull WRDS, calculate covariates and CAR)
            es = ThematicES(
                theme_sentiment.to_dict('records'),
                wrds_connection=wrds_conn
            )
            results_df = es.doAll()

            if results_df is not None and not results_df.empty:
                # Run regression on the prepared data
                model = run_regression_for_theme(
                    theme_id=theme_id,
                    theme_name=theme_name,
                    theme_results_df=results_df,
                    results_dir=es_output_dir
                )
                if model is not None:
                    models[theme_id] = model
            else:
                print(f"  No results for {theme_name}")
        except Exception as e:
            print(f"  Error: {e}")

    # Create summary
    if models:
        themes_dict = {t['theme_id']: t for t in theme_list}
        create_regression_significance_summary(models, themes_dict, Path(es_output_dir))

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


def cmd_run(args):
    """Execute the 'run' subcommand - full pipeline execution."""
    # Validate inputs
    stages = set(args.stages)
    if 'all' in stages:
        stages = {'sentiment', 'event_study', 'portfolio'}

    if 'sentiment' in stages:
        if not args.themes:
            print("Error: --themes required for sentiment stage")
            sys.exit(1)

    if ('event_study' in stages or 'portfolio' in stages) and \
       'sentiment' not in stages:
        if not args.sentiment_file:
            print("Error: --sentiment-file required when skipping sentiment stage")
            sys.exit(1)

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

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
            # Set PostgreSQL env vars so wrds library doesn't prompt
            username = os.environ.get('WRDS_USERNAME', 'torrin')
            password = os.environ.get('WRDS_PASSWORD', '')
            os.environ['PGUSER'] = username
            if password:
                os.environ['PGPASSWORD'] = password
            wrds_conn = wrds.Connection(wrds_username=username)
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


def cmd_event_study(args):
    """Execute the 'event-study' subcommand - standalone event study."""
    import pandas as pd

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load sentiment scores
    sentiment_df = load_sentiment_scores(args.sentiment_file)

    # Group by theme
    themes = group_by_theme(sentiment_df)
    print(f"Processing {len(themes)} themes...")

    # Initialize WRDS connection
    wrds_conn = None
    try:
        import wrds
        # Set PostgreSQL env vars so wrds library doesn't prompt
        username = os.environ.get('WRDS_USERNAME', 'torrin')
        password = os.environ.get('WRDS_PASSWORD', '')
        os.environ['PGUSER'] = username
        if password:
            os.environ['PGPASSWORD'] = password
        wrds_conn = wrds.Connection(wrds_username=username)
        print("WRDS connection established")
    except Exception as e:
        print(f"Error: Could not connect to WRDS: {e}")
        sys.exit(1)

    event_study_models = {}

    try:
        # Collect ALL events across all themes and deduplicate
        unique_events_dict = {}
        theme_event_mapping = {}

        for theme_id, theme_data in themes.items():
            events = theme_data['events']
            theme_event_mapping[theme_id] = []

            if events:
                for event in events:
                    permno = event['permno']
                    edate = event['edate']
                    sentiment = event['sentiment']
                    key = (permno, edate)

                    theme_event_mapping[theme_id].append((permno, pd.to_datetime(edate), sentiment))

                    if key not in unique_events_dict:
                        unique_events_dict[key] = {
                            'event': {'permno': permno, 'edate': edate, 'sentiment': 0},
                            'theme_sentiments': []
                        }
                    unique_events_dict[key]['theme_sentiments'].append((theme_id, sentiment))

        unique_events = [item['event'] for item in unique_events_dict.values()]

        if unique_events:
            # Run batched event study
            batched_results = run_batched_event_study(unique_events, wrds_conn)

            if batched_results is not None:
                batched_results['edate'] = pd.to_datetime(batched_results['edate'])

                for theme_id, theme_data in themes.items():
                    theme_name = theme_data['theme_name']
                    events = theme_data['events']

                    if events and theme_id in theme_event_mapping:
                        theme_events = theme_event_mapping[theme_id]
                        theme_events_df = pd.DataFrame(
                            theme_events, columns=['permno', 'edate', 'theme_sentiment']
                        )

                        theme_permno_dates = [(e[0], e[1]) for e in theme_events]
                        mask = batched_results.apply(
                            lambda row: (row['permno'], row['edate']) in theme_permno_dates,
                            axis=1
                        )
                        theme_results = batched_results[mask].copy()

                        if not theme_results.empty:
                            theme_results = theme_results.drop(columns=['sentiment'], errors='ignore')
                            theme_results = theme_results.merge(
                                theme_events_df, on=['permno', 'edate'], how='left'
                            )
                            theme_results = theme_results.rename(columns={'theme_sentiment': 'sentiment'})

                            study_object = run_regression_for_theme(
                                theme_id, theme_name, theme_results, Path(output_dir)
                            )
                            if study_object is not None:
                                event_study_models[theme_id] = study_object

                if event_study_models:
                    create_regression_significance_summary(
                        event_study_models, themes, Path(output_dir)
                    )

    finally:
        if wrds_conn:
            wrds_conn.close()

    print("\n" + "=" * 60)
    print("EVENT STUDY COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def cmd_portfolio(args):
    """Execute the 'portfolio' subcommand - standalone portfolio sorts."""
    import pandas as pd

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load sentiment scores
    sentiment_df = load_sentiment_scores(args.sentiment_file)

    # Group by theme
    themes = group_by_theme(sentiment_df)
    print(f"Processing {len(themes)} themes...")

    # Override weighting if specified
    weighting = args.weighting if args.weighting else config.WEIGHTING

    # Initialize WRDS connection
    wrds_conn = None
    try:
        import wrds
        # Set PostgreSQL env vars so wrds library doesn't prompt
        username = os.environ.get('WRDS_USERNAME', 'torrin')
        password = os.environ.get('WRDS_PASSWORD', '')
        os.environ['PGUSER'] = username
        if password:
            os.environ['PGPASSWORD'] = password
        wrds_conn = wrds.Connection(wrds_username=username)
        print("WRDS connection established")
    except Exception as e:
        print(f"Error: Could not connect to WRDS: {e}")
        sys.exit(1)

    portfolio_results = {}

    try:
        for theme_id, theme_data in themes.items():
            theme_name = theme_data['theme_name']
            events = theme_data['events']

            if events:
                portfolio_df = run_portfolio_sorts_for_theme(
                    theme_id, theme_name, events, Path(output_dir),
                    wrds_conn=wrds_conn, weighting=weighting
                )
                if portfolio_df is not None:
                    portfolio_results[theme_id] = portfolio_df

        # Create combined analysis if multiple themes
        if len(portfolio_results) > 1:
            combined_portfolio = create_combined_portfolio_analysis(
                portfolio_results, Path(output_dir)
            )
            if combined_portfolio is not None:
                create_portfolio_time_series_chart(combined_portfolio, Path(output_dir))

    finally:
        if wrds_conn:
            wrds_conn.close()

    print("\n" + "=" * 60)
    print("PORTFOLIO SORTS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Downstream Analysis Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with all stages
  python cli.py run --themes data/themes.json --output results/

  # Run only sentiment analysis (no WRDS needed)
  python cli.py run --themes data/themes.json --stages sentiment

  # Run standalone event study on existing sentiment
  python cli.py event-study --sentiment-file results/sentiment.csv --output results/

  # Run standalone portfolio sorts
  python cli.py portfolio --sentiment-file results/sentiment.csv --output results/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # 'run' subcommand - full pipeline
    run_parser = subparsers.add_parser(
        'run',
        help='Run the full downstream analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    run_parser.add_argument(
        '--themes', '-t',
        help='Path to themes JSON file (from cloud export)'
    )
    run_parser.add_argument(
        '--sentiment-file', '-s',
        help='Path to existing sentiment CSV (skip sentiment stage)'
    )
    run_parser.add_argument(
        '--output', '-o',
        default='results',
        help='Output directory (default: results)'
    )
    run_parser.add_argument(
        '--stages',
        nargs='+',
        choices=['sentiment', 'event_study', 'portfolio', 'all'],
        default=['all'],
        help='Which stages to run (default: all)'
    )
    run_parser.set_defaults(func=cmd_run)

    # 'event-study' subcommand
    es_parser = subparsers.add_parser(
        'event-study',
        help='Run standalone event study on pre-computed sentiment scores',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    es_parser.add_argument(
        '--sentiment-file', '-s',
        required=True,
        help='Path to sentiment_scores.csv file'
    )
    es_parser.add_argument(
        '--output', '-o',
        default='results/event_study',
        help='Output directory (default: results/event_study)'
    )
    es_parser.set_defaults(func=cmd_event_study)

    # 'portfolio' subcommand
    ps_parser = subparsers.add_parser(
        'portfolio',
        help='Run standalone portfolio sorts on pre-computed sentiment scores',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ps_parser.add_argument(
        '--sentiment-file', '-s',
        required=True,
        help='Path to sentiment_scores.csv file'
    )
    ps_parser.add_argument(
        '--output', '-o',
        default='results/portfolio_sorts',
        help='Output directory (default: results/portfolio_sorts)'
    )
    ps_parser.add_argument(
        '--weighting', '-w',
        choices=['value', 'equal'],
        default=None,
        help='Portfolio weighting method (default: from config)'
    )
    ps_parser.set_defaults(func=cmd_portfolio)

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Dispatch to appropriate command handler
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
