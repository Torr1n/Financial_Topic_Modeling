#!/usr/bin/env python3
"""
Thin wrapper for backward compatibility.

This script provides backward compatibility for the original run_portfolio_sorts.py interface.
All functionality has been consolidated into cli.py. This wrapper translates the
original CLI arguments to the new CLI format.

Usage (original interface preserved):
    python run_portfolio_sorts.py --sentiment_file results/run_*/sentiment_scores.csv --output_dir results/portfolio_sorts_rerun/

For new projects, prefer using cli.py directly:
    python cli.py portfolio --sentiment-file results/sentiment.csv --output results/

Author: Team 2 COMM386I
Date: January 2026
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cli import cmd_portfolio


def main():
    """Parse original CLI arguments and delegate to cli.py portfolio command."""
    parser = argparse.ArgumentParser(
        description='Run portfolio sorts on pre-computed sentiment scores',
        epilog='Note: This is a backward-compatibility wrapper. Use cli.py portfolio for new projects.'
    )
    parser.add_argument(
        '--sentiment_file', type=str, required=True,
        help='Path to sentiment_scores.csv file'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results/portfolio_sorts',
        help='Output directory for portfolio results (default: results/portfolio_sorts)'
    )
    parser.add_argument(
        '--weighting', type=str, default=None, choices=['value', 'equal'],
        help='Portfolio weighting method: value (market-cap weighted) or equal (default: from config)'
    )

    args = parser.parse_args()

    # Convert to cli.py portfolio command arguments
    class PortfolioArgs:
        def __init__(self):
            self.sentiment_file = args.sentiment_file
            self.output = args.output_dir
            self.weighting = args.weighting

    ps_args = PortfolioArgs()

    # Delegate to cli.py
    cmd_portfolio(ps_args)


if __name__ == "__main__":
    main()
