#!/usr/bin/env python3
"""
Thin wrapper for backward compatibility.

This script provides backward compatibility for the original run_pipeline.py interface.
All functionality has been consolidated into cli.py. This wrapper translates the
original CLI arguments to the new CLI format.

Usage (original interface preserved):
    python run_pipeline.py --themes_file data/themes.json --output_dir results/

For new projects, prefer using cli.py directly:
    python cli.py run --themes data/themes.json --output results/

Author: Team 2 COMM386I
Date: January 2026
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cli import cmd_run


def main():
    """Parse original CLI arguments and delegate to cli.py run command."""
    parser = argparse.ArgumentParser(
        description='Run complete thematic sentiment analysis pipeline',
        epilog='Note: This is a backward-compatibility wrapper. Use cli.py run for new projects.'
    )
    parser.add_argument(
        '--themes_file', type=str,
        help='Path to earnings call themes JSON file (with PERMNOs)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results',
        help='Output directory for results (default: results/)'
    )
    parser.add_argument(
        '--skip_sentiment', action='store_true',
        help='Skip sentiment analysis (use existing sentiment_scores.csv)'
    )
    parser.add_argument(
        '--skip_event_study', action='store_true',
        help='Skip event study'
    )
    parser.add_argument(
        '--skip_portfolio', action='store_true',
        help='Skip portfolio sorts'
    )

    args = parser.parse_args()

    # Convert to cli.py run command arguments
    class RunArgs:
        def __init__(self):
            self.themes = args.themes_file
            self.sentiment_file = None
            self.output = args.output_dir
            # Determine stages based on skip flags
            stages = []
            if not args.skip_sentiment:
                stages.append('sentiment')
            if not args.skip_event_study:
                stages.append('event_study')
            if not args.skip_portfolio:
                stages.append('portfolio')
            self.stages = stages if stages else ['all']

    run_args = RunArgs()

    # Validate that we have required inputs
    if 'sentiment' in run_args.stages and not run_args.themes:
        parser.error("--themes_file is required when running sentiment stage")

    # Delegate to cli.py
    cmd_run(run_args)


if __name__ == "__main__":
    main()
