# Pipeline Integration: Cloud Topic Modeling → Downstream Analysis

This document describes how to connect the cloud topic modeling pipeline to the downstream sentiment/event study pipeline.

## Overview

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│  Cloud Pipeline (Topic Model)  │     │  Downstream Analysis            │
│  ─────────────────────────────  │     │  ─────────────────────────────  │
│  • Ingests transcripts         │     │  • FinBERT sentiment scoring    │
│  • Firm-level BERTopic         │ ──► │  • Event study (CAR regression) │
│  • Cross-firm themes           │     │  • Portfolio sorts by sentiment │
│  • Stores in PostgreSQL        │     │  • Requires JSON with PERMNOs   │
└─────────────────────────────────┘     └─────────────────────────────────┘
                                  │
                          Export Bridge
                    (cloud/src/export/export_for_downstream.py)
```

## Prerequisites

- PostgreSQL database with processed themes from cloud pipeline
- WRDS account with access to:
  - `crsp.stocknames` (PERMNO lookup)
  - `comp.fundq` (earnings dates, if backfilling)
- WRDS credentials via `~/.pgpass` or environment variables

## Step-by-Step Integration

### 1. Database Schema Update

The `Firm` model now includes `earnings_call_date`. If you have an existing database, run a migration or recreate tables:

```sql
ALTER TABLE firms ADD COLUMN earnings_call_date TIMESTAMP;
```

Or let SQLAlchemy handle it on next run with `Base.metadata.create_all()`.

### 2. Backfill Earnings Dates (if needed)

If existing Firm records are missing `earnings_call_date`, backfill from WRDS:

```bash
python cloud/scripts/backfill_earnings_dates.py \
    --db-url postgresql://user:pass@host:port/db \
    --dry-run  # Remove to commit changes
```

This queries WRDS `comp.fundq` for earnings announcement dates by ticker + quarter.

**Alternative:** If you have the original transcript CSV with `mostimportantdateutc`, use that instead (more accurate).

### 3. Export Themes to JSON

Run the bridge script to export themes with PERMNO lookups:

```bash
python -m cloud.src.export.export_for_downstream \
    --db-url postgresql://user:pass@host:port/db \
    --output downstream/data/themes_for_sentiment.json
```

Options:
- `--skip-permno`: Skip WRDS lookup (for testing without WRDS access)

### 4. Run Downstream Pipeline

```bash
cd downstream
python cli.py --themes data/themes_for_sentiment.json --output results/
```

Or run individual stages:
```bash
python cli.py --themes data/themes.json --stages sentiment
python cli.py --sentiment-file results/sentiment.csv --stages event_study portfolio
```

## Output JSON Format

The export script produces JSON matching downstream expectations:

```json
{
  "export_timestamp": "2026-01-19T15:30:00",
  "n_themes": 82,
  "themes": [
    {
      "theme_id": "theme_001",
      "theme_name": "Supply Chain Resilience",
      "firm_contributions": [
        {
          "firm_name": "Apple Inc.",
          "permno": 14593,
          "earnings_call_date": "2023-01-28",
          "sentences": [
            {"text": "We continue to navigate supply chain challenges.", "speaker": "CEO"},
            {"text": "Inventory management has improved significantly.", "speaker": "CFO"}
          ]
        }
      ]
    }
  ]
}
```

## Key Files

| File | Purpose |
|------|---------|
| `cloud/src/database/models.py` | Firm model with `earnings_call_date` |
| `cloud/scripts/backfill_earnings_dates.py` | One-time WRDS backfill script |
| `cloud/src/export/export_for_downstream.py` | Main bridge/export script |
| `downstream/cli.py` | Unified downstream CLI |
| `downstream/config.py` | Downstream configuration |
| `downstream/src/` | Core analysis modules |

## Troubleshooting

### WRDS Connection Issues
- Ensure `~/.pgpass` has: `wrds-pgdata.wharton.upenn.edu:9737:wrds:USERNAME:PASSWORD`
- Or set `WRDS_USERNAME` and `WRDS_PASSWORD` environment variables
- Run `chmod 600 ~/.pgpass`

### Missing PERMNOs
- Some tickers may not have CRSP coverage (ADRs, OTC stocks)
- Check ticker spelling matches CRSP conventions
- Verify earnings_call_date is within the stock's trading history

### Missing Earnings Dates
- Backfill script uses Compustat `rdq` (report date)
- Some firms may not be in Compustat - check quarterly date manually
- Consider using original transcript CSV if available
