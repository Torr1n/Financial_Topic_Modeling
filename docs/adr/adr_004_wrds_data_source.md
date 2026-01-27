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
| `ciqtranscript` | `ciq` | Master transcript metadata |
| `ciqtranscriptcomponent` | `ciq` | Transcript text segments |
| `ciqcompany` | `ciq` | Company names |
| `wrds_gvkey` | `ciq` | Capital IQ → Compustat GVKEY |
| `ccmxpf_linktable` | `crsp` | Compustat GVKEY → CRSP PERMNO |

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
- **Credentials**: Use environment variables (`WRDS_USERNAME`, `WRDS_PASSWORD`)
- **Caching**: Cache GVKEY→PERMNO linkage table at start of pipeline run
- **Unlinked firms**: **SKIP entirely** (see decision note below)
- **Testing**: Mock WRDS responses for unit tests; use integration tests sparingly

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
WITH transcript_data AS (
    SELECT
        t.companyid AS firm_id,
        c.companyname AS firm_name,
        t.transcriptid,
        t.mostimportantdateutc::date AS earnings_call_date,
        tc.componenttext,
        tc.componentorder,
        tc.speakertypename AS speaker_type
    FROM ciq.ciqtranscript t
    JOIN ciq.ciqcompany c ON t.companyid = c.companyid
    JOIN ciq.ciqtranscriptcomponent tc ON t.transcriptid = tc.transcriptid
    WHERE t.mostimportantdateutc BETWEEN %(start_date)s AND %(end_date)s
),
gvkey_link AS (
    SELECT td.*, wg.gvkey
    FROM transcript_data td
    LEFT JOIN ciq.wrds_gvkey wg ON td.firm_id = wg.companyid
),
permno_link AS (
    SELECT g.*,
           ccm.lpermno AS permno,
           ccm.linkdt AS link_date
    FROM gvkey_link g
    LEFT JOIN crsp.ccmxpf_linktable ccm ON g.gvkey = ccm.gvkey
        AND ccm.linktype IN ('LU', 'LC')
        AND ccm.linkprim IN ('P', 'C')
        AND g.earnings_call_date >= ccm.linkdt
        AND g.earnings_call_date <= COALESCE(ccm.linkenddt, '9999-12-31')
)
SELECT * FROM permno_link
ORDER BY firm_id, transcriptid, componentorder;
```

## References

- [WRDS Python Package](https://pypi.org/project/wrds/)
- [Capital IQ Transcripts Documentation](https://wrds-www.wharton.upenn.edu/pages/grid-items/capital-iq-transcripts/)
- [CRSP/Compustat Merged Documentation](https://wrds-www.wharton.upenn.edu/pages/classroom/using-crspcompustat-merged-database/)
- [Linking Capital IQ with Compustat](https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/linking-capital-iq-with-compustat/)
