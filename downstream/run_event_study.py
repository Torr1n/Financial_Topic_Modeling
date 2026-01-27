#!/usr/bin/env python3
"""
Thin wrapper for backward compatibility.

This script provides backward compatibility for the original run_event_study.py interface.
All functionality has been consolidated into cli.py. This wrapper translates the
original CLI arguments to the new CLI format.

Usage (original interface preserved):
    python run_event_study.py --sentiment_file results/run_*/sentiment_scores.csv --output_dir results/event_study_rerun/

For new projects, prefer using cli.py directly:
    python cli.py event-study --sentiment-file results/sentiment.csv --output results/

Author: Team 2 COMM386I
Date: January 2026
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cli import cmd_event_study


def main():
    """Parse original CLI arguments and delegate to cli.py event-study command."""
    parser = argparse.ArgumentParser(
        description='Run event studies on pre-computed sentiment scores',
        epilog='Note: This is a backward-compatibility wrapper. Use cli.py event-study for new projects.'
    )
    parser.add_argument(
        '--sentiment_file', type=str, required=True,
        help='Path to sentiment_scores.csv file'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results/event_study',
        help='Output directory for event study results (default: results/event_study)'
    )

    args = parser.parse_args()

    # Convert to cli.py event-study command arguments
    class EventStudyArgs:
        def __init__(self):
            self.sentiment_file = args.sentiment_file
            self.output = args.output_dir

    es_args = EventStudyArgs()

    # Delegate to cli.py
    cmd_event_study(es_args)


if __name__ == "__main__":
    main()
