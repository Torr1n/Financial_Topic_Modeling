#!/usr/bin/env python3
"""
Export themes from PostgreSQL to JSON format for downstream sentiment/event study pipeline.

This script bridges the cloud topic modeling pipeline to the downstream analysis by:
1. Querying themes with full hierarchy from PostgreSQL
2. Looking up PERMNOs via WRDS for each firm
3. Restructuring data into the format expected by downstream_analysis

Usage:
    python -m cloud.src.export.export_for_downstream \
        --db-url postgresql://user:pass@host:port/db \
        --output data/themes_for_sentiment.json

Output format matches downstream_analysis/run_pipeline.py expectations.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import wrds
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.database.models import Base, Firm, Theme, Topic, Sentence
from src.database.repository import DatabaseRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PermnoLookup:
    """WRDS PERMNO lookup with caching."""

    def __init__(self):
        self._cache: Dict[tuple, Optional[int]] = {}
        self._db: Optional[wrds.Connection] = None

    def connect(self):
        """Connect to WRDS."""
        if self._db is None:
            logger.info("Connecting to WRDS...")
            self._db = wrds.Connection()
            logger.info("WRDS connection established")

    def close(self):
        """Close WRDS connection."""
        if self._db is not None:
            self._db.close()
            self._db = None

    def lookup(self, ticker: str, date: datetime) -> Optional[int]:
        """
        Look up PERMNO for ticker on a given date.

        Args:
            ticker: Stock ticker symbol
            date: Date for lookup (PERMNO can change over time)

        Returns:
            PERMNO if found, None otherwise
        """
        if not ticker or not date:
            return None

        ticker = ticker.upper()
        date_str = date.strftime("%Y-%m-%d")
        cache_key = (ticker, date_str)

        if cache_key in self._cache:
            return self._cache[cache_key]

        self.connect()

        query = f"""
        SELECT permno
        FROM crsp.stocknames
        WHERE UPPER(ticker) = '{ticker}'
          AND namedt <= '{date_str}'
          AND nameendt >= '{date_str}'
        LIMIT 1
        """

        try:
            df = self._db.raw_sql(query)
            if not df.empty:
                permno = int(df.iloc[0]['permno'])
                self._cache[cache_key] = permno
                return permno
            else:
                self._cache[cache_key] = None
                return None
        except Exception as e:
            logger.warning(f"PERMNO lookup failed for {ticker} on {date_str}: {e}")
            self._cache[cache_key] = None
            return None

    def batch_lookup(self, lookups: List[tuple]) -> Dict[tuple, Optional[int]]:
        """
        Batch lookup PERMNOs for efficiency.

        Args:
            lookups: List of (ticker, date) tuples

        Returns:
            Dictionary mapping (ticker, date_str) -> permno
        """
        # Filter out already cached
        to_lookup = []
        for ticker, date in lookups:
            if ticker and date:
                ticker = ticker.upper()
                date_str = date.strftime("%Y-%m-%d")
                if (ticker, date_str) not in self._cache:
                    to_lookup.append((ticker, date_str))

        if not to_lookup:
            return self._cache

        self.connect()

        # Build batch query
        conditions = []
        for ticker, date_str in to_lookup:
            conditions.append(
                f"(UPPER(ticker) = '{ticker}' AND namedt <= '{date_str}' AND nameendt >= '{date_str}')"
            )

        if not conditions:
            return self._cache

        query = f"""
        SELECT DISTINCT ON (UPPER(ticker))
               UPPER(ticker) as ticker, permno, namedt, nameendt
        FROM crsp.stocknames
        WHERE {' OR '.join(conditions)}
        ORDER BY UPPER(ticker), namedt DESC
        """

        try:
            df = self._db.raw_sql(query)
            for _, row in df.iterrows():
                ticker = row['ticker']
                permno = int(row['permno'])
                # Add to cache for the ticker (approximate - uses latest valid period)
                for orig_ticker, date_str in to_lookup:
                    if orig_ticker == ticker:
                        self._cache[(ticker, date_str)] = permno
        except Exception as e:
            logger.warning(f"Batch PERMNO lookup failed: {e}")

        # Mark not-found as None
        for ticker, date_str in to_lookup:
            if (ticker, date_str) not in self._cache:
                self._cache[(ticker, date_str)] = None

        return self._cache


def export_themes_for_downstream(
    db_url: str,
    output_path: str,
    skip_permno_lookup: bool = False
) -> Dict[str, Any]:
    """
    Export themes from PostgreSQL to downstream JSON format.

    Args:
        db_url: PostgreSQL connection URL
        output_path: Path for output JSON file
        skip_permno_lookup: If True, skip WRDS lookup (for testing)

    Returns:
        The exported data structure
    """
    # Connect to database
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    repo = DatabaseRepository(session)

    # Initialize PERMNO lookup
    permno_lookup = PermnoLookup() if not skip_permno_lookup else None

    # Get all themes
    themes = repo.get_all_themes()
    logger.info(f"Found {len(themes)} themes to export")

    exported_themes = []

    for theme in themes:
        logger.info(f"Processing theme {theme.id}: {theme.name}")

        # Get full hierarchy for this theme
        hierarchy = repo.get_theme_with_hierarchy(theme.id)
        if not hierarchy:
            logger.warning(f"No hierarchy data for theme {theme.id}")
            continue

        # Group topics by firm to create firm_contributions
        firm_contributions_map = defaultdict(lambda: {
            "firm_name": None,
            "ticker": None,
            "earnings_call_date": None,
            "permno": None,
            "sentences": []
        })

        for topic in hierarchy.get("topics", []):
            firm_id = topic["firm_id"]
            firm_name = topic.get("firm_name")

            # Get full firm record for ticker and date
            firm = session.get(Firm, firm_id)
            if firm:
                firm_contributions_map[firm_id]["firm_name"] = firm.name
                firm_contributions_map[firm_id]["ticker"] = firm.ticker
                firm_contributions_map[firm_id]["earnings_call_date"] = firm.earnings_call_date

            # Add sentences from this topic
            for sentence in topic.get("sentences", []):
                firm_contributions_map[firm_id]["sentences"].append({
                    "text": sentence.get("raw_text", ""),
                    "speaker": sentence.get("speaker_type", "")
                })

        # Build firm_contributions list with PERMNO lookup
        firm_contributions = []
        for firm_id, contrib in firm_contributions_map.items():
            # Look up PERMNO
            permno = None
            if permno_lookup and contrib["ticker"] and contrib["earnings_call_date"]:
                permno = permno_lookup.lookup(
                    contrib["ticker"],
                    contrib["earnings_call_date"]
                )

            if permno is None and not skip_permno_lookup:
                logger.warning(
                    f"No PERMNO found for {contrib['firm_name']} "
                    f"({contrib['ticker']}) on {contrib['earnings_call_date']}"
                )

            # Format date as string
            date_str = None
            if contrib["earnings_call_date"]:
                date_str = contrib["earnings_call_date"].strftime("%Y-%m-%d")

            firm_contributions.append({
                "firm_name": contrib["firm_name"],
                "permno": permno,
                "earnings_call_date": date_str,
                "sentences": contrib["sentences"]
            })

        exported_themes.append({
            "theme_id": f"theme_{theme.id:03d}",
            "theme_name": theme.name,
            "firm_contributions": firm_contributions
        })

    # Close connections
    session.close()
    if permno_lookup:
        permno_lookup.close()

    # Build final output
    output_data = {
        "export_timestamp": datetime.now().isoformat(),
        "n_themes": len(exported_themes),
        "themes": exported_themes
    }

    # Write to file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"Exported {len(exported_themes)} themes to {output_path}")

    # Summary stats
    total_firms = sum(len(t["firm_contributions"]) for t in exported_themes)
    total_sentences = sum(
        len(fc["sentences"])
        for t in exported_themes
        for fc in t["firm_contributions"]
    )
    firms_with_permno = sum(
        1 for t in exported_themes
        for fc in t["firm_contributions"]
        if fc["permno"] is not None
    )

    logger.info(f"Summary:")
    logger.info(f"  - Themes: {len(exported_themes)}")
    logger.info(f"  - Firm contributions: {total_firms}")
    logger.info(f"  - Firms with PERMNO: {firms_with_permno}/{total_firms}")
    logger.info(f"  - Total sentences: {total_sentences}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Export themes to JSON for downstream sentiment/event study pipeline"
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--output",
        default="data/themes_for_sentiment.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--skip-permno",
        action="store_true",
        help="Skip WRDS PERMNO lookup (for testing)"
    )

    args = parser.parse_args()

    export_themes_for_downstream(
        db_url=args.db_url,
        output_path=args.output,
        skip_permno_lookup=args.skip_permno
    )


if __name__ == "__main__":
    main()
