#!/usr/bin/env python3
"""
Backfill earnings_call_date for existing Firm records using WRDS.

This script queries WRDS Compustat for earnings announcement dates (rdq)
and updates Firm records that have ticker + quarter but missing earnings_call_date.

Usage:
    python backfill_earnings_dates.py --db-url postgresql://user:pass@host:port/db

Requirements:
    - WRDS account with Compustat access
    - ~/.pgpass file with WRDS credentials OR WRDS_USERNAME/WRDS_PASSWORD env vars
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import wrds
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.models import Base, Firm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_quarter_to_compustat_format(quarter: str) -> str:
    """
    Convert quarter string to Compustat datacqtr format.

    Args:
        quarter: Quarter string like "2023Q1" or "2023-Q1"

    Returns:
        Compustat format like "2023Q1"
    """
    # Remove any dashes or spaces
    q = quarter.replace("-", "").replace(" ", "").upper()
    return q


def fetch_earnings_dates_from_wrds(tickers: list, quarters: list) -> dict:
    """
    Query WRDS Compustat for earnings announcement dates.

    Args:
        tickers: List of ticker symbols
        quarters: List of quarters in format "2023Q1"

    Returns:
        Dictionary mapping (ticker, quarter) -> earnings_date
    """
    logger.info(f"Connecting to WRDS to fetch earnings dates for {len(tickers)} tickers...")

    try:
        db = wrds.Connection()
    except Exception as e:
        logger.error(f"Failed to connect to WRDS: {e}")
        logger.info("Ensure you have a ~/.pgpass file with WRDS credentials or set WRDS_USERNAME/WRDS_PASSWORD")
        raise

    # Build the query - get earnings announcement dates from Compustat fundq
    # rdq = report date of quarterly earnings
    unique_tickers = list(set(t.upper() for t in tickers if t))
    unique_quarters = list(set(quarters))

    if not unique_tickers:
        logger.warning("No tickers provided")
        return {}

    # Format for SQL IN clause
    ticker_str = ", ".join(f"'{t}'" for t in unique_tickers)
    quarter_str = ", ".join(f"'{q}'" for q in unique_quarters)

    query = f"""
    SELECT tic, datacqtr, rdq
    FROM comp.fundq
    WHERE UPPER(tic) IN ({ticker_str})
      AND datacqtr IN ({quarter_str})
      AND rdq IS NOT NULL
    ORDER BY tic, datacqtr
    """

    logger.info(f"Querying WRDS for {len(unique_tickers)} tickers across {len(unique_quarters)} quarters...")

    try:
        df = db.raw_sql(query)
        db.close()
    except Exception as e:
        logger.error(f"WRDS query failed: {e}")
        raise

    # Build lookup dictionary
    result = {}
    for _, row in df.iterrows():
        key = (row['tic'].upper(), row['datacqtr'])
        result[key] = row['rdq']

    logger.info(f"Retrieved {len(result)} earnings dates from WRDS")
    return result


def backfill_earnings_dates(db_url: str, dry_run: bool = False):
    """
    Backfill earnings_call_date for Firm records missing the date.

    Args:
        db_url: PostgreSQL connection URL
        dry_run: If True, don't commit changes
    """
    # Connect to local database
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Get all firms with ticker and quarter but no earnings_call_date
    firms = session.query(Firm).filter(
        Firm.ticker.isnot(None),
        Firm.quarter.isnot(None),
        Firm.earnings_call_date.is_(None)
    ).all()

    if not firms:
        logger.info("No firms need backfilling (all have earnings_call_date or missing ticker/quarter)")
        return

    logger.info(f"Found {len(firms)} firms needing earnings_call_date backfill")

    # Collect tickers and quarters for batch WRDS query
    tickers = [f.ticker for f in firms]
    quarters = [parse_quarter_to_compustat_format(f.quarter) for f in firms]

    # Fetch earnings dates from WRDS
    earnings_dates = fetch_earnings_dates_from_wrds(tickers, quarters)

    # Update firms
    updated = 0
    not_found = 0

    for firm in firms:
        ticker = firm.ticker.upper() if firm.ticker else None
        quarter = parse_quarter_to_compustat_format(firm.quarter) if firm.quarter else None

        if not ticker or not quarter:
            continue

        key = (ticker, quarter)
        if key in earnings_dates:
            earnings_date = earnings_dates[key]
            if isinstance(earnings_date, str):
                earnings_date = datetime.strptime(earnings_date, "%Y-%m-%d")

            firm.earnings_call_date = earnings_date
            updated += 1
            logger.debug(f"Updated {firm.name} ({ticker}) {quarter}: {earnings_date}")
        else:
            not_found += 1
            logger.warning(f"No earnings date found for {firm.name} ({ticker}) {quarter}")

    logger.info(f"Updated {updated} firms, {not_found} not found in WRDS")

    if dry_run:
        logger.info("DRY RUN - rolling back changes")
        session.rollback()
    else:
        session.commit()
        logger.info("Changes committed to database")

    session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill earnings_call_date for Firm records using WRDS"
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="PostgreSQL connection URL (e.g., postgresql://user:pass@host:port/db)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't commit changes, just show what would be updated"
    )

    args = parser.parse_args()

    backfill_earnings_dates(args.db_url, args.dry_run)


if __name__ == "__main__":
    main()
