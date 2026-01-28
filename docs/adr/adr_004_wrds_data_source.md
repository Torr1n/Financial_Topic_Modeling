# ADR-004: WRDS/Capital IQ Data Source

## Status
Accepted

## Date
2026-01-26

## Context

The Financial Topic Modeling pipeline currently ingests earnings call transcripts from a local CSV file (`transcripts_2023-01-01_to_2023-03-31_enriched.csv`). This CSV was originally downloaded from WRDS Capital IQ tables as a one-time export for the MVP.

For production processing of 8+ quarters of data, we need to:
1. Fetch transcripts dynamically from WRDS for arbitrary date ranges
2. Include CRSP PERMNO and Compustat GVKEY identifiers for downstream event studies
3. Maintain compatibility with the existing `DataConnector` interface

### Current Limitations
- CSV is static (single quarter, single snapshot in time)
- No PERMNO/GVKEY in current data (requires manual mapping for sentiment analysis)
- Cannot easily scale to multi-quarter processing
- Data refresh requires manual re-export from WRDS web interface

### Requirements from Sentiment Analysis
The sentiment analysis module (`sentiment_analysis/handoff_package/`) expects firm contributions with:
```json
{
  "firm_id": "374372246",
  "permno": 14593,           // REQUIRED
  "earnings_call_date": "2023-01-28",
  "sentences": [...]
}
```

Without PERMNO at ingestion time, a separate mapping step is required between topic modeling and sentiment analysis. This is error-prone and adds complexity.

## Decision

**Implement a `WRDSConnector` class that fetches transcripts directly from WRDS Capital IQ tables with PERMNO/GVKEY linking performed at ingestion time.**

### Key Tables
| Table | Library | Purpose |
|-------|---------|---------|
| `wrds_transcript_detail` | `ciq` | Denormalized transcript metadata (companyid, date, etc.) |
| `ciqtranscriptcomponent` | `ciq` | Transcript text segments (full componenttext) |
| `wrds_transcript_person` | `ciq` | Speaker info (speakertypename) |
| `wrds_gvkey` | `ciq` | Capital IQ → Compustat GVKEY |
| `ccmxpf_linktable` | `crsp` | Compustat GVKEY → CRSP PERMNO |

**Note**: The implementation uses WRDS denormalized views (`wrds_transcript_detail`, `wrds_transcript_person`) rather than raw Capital IQ tables (`ciqtranscript`, `ciqcompany`) because:
1. Raw tables lack direct `companyid` column in `ciqtranscript`
2. Denormalized views provide `companyid`, `companyname`, and `mostimportantdateutc` in one place
3. Speaker type requires joining through `wrds_transcript_person` (not direct column)

### Linking Strategy
```
companyid (Capital IQ)
    ↓ via ciq.wrds_gvkey
gvkey (Compustat)
    ↓ via crsp.ccmxpf_linktable
permno (CRSP)
```

### Link Quality Filters
Per WRDS documentation, only use high-quality links:
- `linktype IN ('LU', 'LC')` - Linked, Unique/Complete
- `linkprim IN ('P', 'C')` - Primary/Compustat primary link
- Date overlap: `earnings_call_date BETWEEN linkdt AND COALESCE(linkenddt, '9999-12-31')`

### Interface Design
```python
class WRDSConnector(DataConnector):
    """
    WRDS connector with CRSP/Compustat identifier enrichment.

    Returns TranscriptData with metadata containing:
        - permno: CRSP permanent number (for event studies)
        - gvkey: Compustat identifier (for fundamental data)
        - link_date: Date when linkage was valid
        - earnings_call_date: Transcript date
    """

    def fetch_transcripts(
        self,
        firm_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> TranscriptData:
        ...

    def get_available_firm_ids(self) -> List[str]:
        ...
```

## Consequences

### Positive
- **Direct WRDS access**: Fetch any quarter without manual CSV export
- **PERMNO at source**: No separate mapping step needed for sentiment analysis
- **Flexible queries**: Can filter by date range, firm list, event type
- **Data freshness**: Always get latest corrections from Capital IQ
- **Scalability**: Can easily process 100 quarters if needed

### Negative
- **WRDS dependency**: Requires active WRDS subscription and credentials
- **Network latency**: Queries take longer than local CSV (~10-30 seconds for 5000 firms)
- **Link coverage gaps**: Not all firms have PERMNO (~5-10% unlinked)
- **Schema coupling**: If WRDS changes table structure, connector needs updates

### Mitigations
- **Credentials**: Auto-configured via `_setup_wrds_auth()` (see Credential Management below)
- **Caching**: PERMNO linking done in SQL (single query, no separate cache needed)
- **Unlinked firms**: **SKIP entirely** (see decision note below)
- **Testing**: Mock WRDS responses for unit tests; use integration tests sparingly

### Credential Management

The connector automatically configures WRDS credentials with the following priority:

1. **Existing `.pgpass` file**: If `~/.pgpass` exists with WRDS entry and correct permissions (0600)
2. **Environment variables**: `WRDS_USERNAME` + `WRDS_PASSWORD` → auto-creates `.pgpass`
3. **AWS Secrets Manager**: Secret named `wrds-credentials` with JSON `{"username":"xxx","password":"xxx"}`
4. **Interactive prompt**: Fallback if no credentials found

**Why auto-create `.pgpass`?** The WRDS Python library (`wrds.Connection()`) does NOT recognize `WRDS_USERNAME`/`WRDS_PASSWORD` environment variables directly. It only recognizes:
- PostgreSQL standard env vars (`PGUSER`, `PGPASSWORD`) - security risk
- The `.pgpass` file with strict 0600 permissions
- The `wrds_username` constructor parameter (for username only)

**AWS Batch/Lambda**: The connector writes to `/tmp/.pgpass` when `/tmp` is writable (common in AWS) and sets `PGPASSFILE` env var, enabling non-interactive auth in ephemeral environments. If `/tmp` is not writable, it falls back to `~/.pgpass`.

```python
# Implementation in _setup_wrds_auth()
if username and password:
    pgpass_path = "/tmp/.pgpass" if os.access("/tmp", os.W_OK) else "~/.pgpass"
    with open(pgpass_path, "w") as f:
        f.write(f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{username}:{password}\n")
    os.chmod(pgpass_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    os.environ["PGPASSFILE"] = pgpass_path
```

### Decision Note: Skip Unlinked Firms

**Decision**: Firms without PERMNO linkage are **skipped entirely** from processing.

**Rationale** (per stakeholder input):
- Unlinked firms are typically international (non-US/Canada) companies
- Without PERMNO, we cannot derive covariates needed for event studies
- Sentiment analysis cannot be performed on these firms anyway
- Processing them would waste compute and create incomplete data

**Implementation**:
```python
# In WRDSConnector._build_transcript_data()
for firm_id, group in df.groupby("firm_id"):
    permno = group["permno"].iloc[0]
    if pd.isna(permno):
        logger.info(f"Skipping firm {firm_id} - no PERMNO linkage (likely non-US)")
        continue
    # ... process firm
```

**Expected coverage**: ~90-95% of US/Canada firms will have valid PERMNO linkage.

## Alternatives Considered

### 1. Continue with CSV exports
- **Pro**: No code changes needed
- **Con**: Manual process, doesn't scale, no PERMNO

### 2. Build local database replica
- **Pro**: Fast queries, offline access
- **Con**: Significant upfront work, data sync complexity, storage costs

### 3. Use WRDS Cloud (SAS)
- **Pro**: Native WRDS environment
- **Con**: Would require rewriting Python pipeline in SAS

## Implementation Notes

### Connection Pattern
```python
import wrds

class WRDSConnector(DataConnector):
    def __init__(self, connection: Optional[wrds.Connection] = None):
        self._conn = connection
        self._owns_connection = connection is None

    def _get_connection(self) -> wrds.Connection:
        if self._conn is None:
            self._conn = wrds.Connection()
        return self._conn

    def close(self):
        if self._owns_connection and self._conn:
            self._conn.close()
```

### SQL Query Template
```sql
-- Uses WRDS denormalized views for transcript metadata
-- Window function selects latest transcript per firm per date range
WITH latest_transcripts AS (
    SELECT
        td.companyid AS firm_id,
        td.companyname AS firm_name,
        td.transcriptid,
        td.mostimportantdateutc::date AS earnings_call_date,
        wg.gvkey,
        ROW_NUMBER() OVER (
            PARTITION BY td.companyid
            ORDER BY td.mostimportantdateutc DESC, td.transcriptid DESC
        ) AS rn
    FROM ciq.wrds_transcript_detail td
    LEFT JOIN ciq.wrds_gvkey wg ON td.companyid = wg.companyid
    WHERE td.mostimportantdateutc BETWEEN %(start_date)s AND %(end_date)s
      AND td.keydeveventtypeid = 48  -- Earnings calls only
      AND (%(firm_ids)s IS NULL OR td.companyid = ANY(%(firm_ids)s))
),
selected_transcripts AS (
    SELECT * FROM latest_transcripts WHERE rn = 1
),
with_components AS (
    SELECT
        st.firm_id::text AS firm_id,
        st.firm_name,
        st.transcriptid::text AS transcript_id,
        st.earnings_call_date,
        st.gvkey,
        tc.componenttext,
        tc.componentorder,
        COALESCE(tp.speakertypename, 'Unknown') AS speakertypename
    FROM selected_transcripts st
    JOIN ciq.ciqtranscriptcomponent tc ON st.transcriptid = tc.transcriptid
    LEFT JOIN ciq.wrds_transcript_person tp ON tc.transcriptid = tp.transcriptid
        AND tc.transcriptcomponentid = tp.transcriptcomponentid
),
with_permno AS (
    SELECT
        wc.*,
        ccm.lpermno AS permno,
        ccm.linkdt AS link_date
    FROM with_components wc
    LEFT JOIN crsp.ccmxpf_linktable ccm ON wc.gvkey = ccm.gvkey
        AND ccm.linktype IN ('LU', 'LC')
        AND ccm.linkprim IN ('P', 'C')
        AND wc.earnings_call_date >= ccm.linkdt
        AND wc.earnings_call_date <= COALESCE(ccm.linkenddt, '9999-12-31')
)
SELECT * FROM with_permno ORDER BY firm_id, componentorder;
```

### Multi-Transcript Rule
When a firm has multiple transcripts in the date range, only the **latest** is selected:
- `ROW_NUMBER()` partitioned by `companyid`, ordered by date DESC
- Tie-break: `transcriptid DESC` for deterministic ordering
- All components from the selected transcript are preserved

## References

- [WRDS Python Package](https://pypi.org/project/wrds/)
- [Capital IQ Transcripts Documentation](https://wrds-www.wharton.upenn.edu/pages/grid-items/capital-iq-transcripts/)
- [CRSP/Compustat Merged Documentation](https://wrds-www.wharton.upenn.edu/pages/classroom/using-crspcompustat-merged-database/)
- [Linking Capital IQ with Compustat](https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/linking-capital-iq-with-compustat/)
