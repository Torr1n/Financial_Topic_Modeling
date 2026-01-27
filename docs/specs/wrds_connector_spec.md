# WRDSConnector Interface Specification

## Overview

The `WRDSConnector` class provides direct access to WRDS Capital IQ transcript tables with automatic PERMNO/GVKEY linking for downstream event studies. It implements the `DataConnector` interface defined in `cloud/src/interfaces.py`.

## Module Location

`cloud/src/connectors/wrds_connector.py`

## Dependencies

```python
import wrds                    # WRDS Python package
import pandas as pd            # DataFrame handling
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from cloud.src.interfaces import DataConnector
from cloud.src.models import TranscriptData, FirmTranscriptData, TranscriptSentence
```

## Class Definition

```python
class WRDSConnector(DataConnector):
    """
    WRDS connector with CRSP/Compustat identifier enrichment.

    Fetches earnings call transcripts from Capital IQ tables and
    automatically links to CRSP PERMNO via Compustat GVKEY.

    Attributes:
        _conn: WRDS database connection
        _owns_connection: Whether this instance owns the connection
        _link_cache: Cached GVKEY → PERMNO mappings

    Usage:
        connector = WRDSConnector()
        data = connector.fetch_transcripts(
            firm_ids=["374372246", "24937"],
            start_date="2023-01-01",
            end_date="2023-03-31"
        )
        connector.close()

        # Or with context manager:
        with WRDSConnector() as connector:
            data = connector.fetch_transcripts(...)
    """
```

## Constructor

```python
def __init__(
    self,
    connection: Optional[wrds.Connection] = None,
    preload_links: bool = True,
) -> None:
    """
    Initialize the WRDS connector.

    Args:
        connection: Optional existing WRDS connection. If None, a new
                   connection will be created lazily on first query.
        preload_links: If True, cache the GVKEY → PERMNO linkage table
                      on first query for faster subsequent lookups.

    Environment Variables:
        WRDS_USERNAME: WRDS account username (if not using .pgpass)
        WRDS_PASSWORD: WRDS account password (if not using .pgpass)
    """
```

## Public Methods

### fetch_transcripts

```python
def fetch_transcripts(
    self,
    firm_ids: List[str],
    start_date: str,
    end_date: str,
) -> TranscriptData:
    """
    Fetch transcript sentences for specified firms and date range.

    Args:
        firm_ids: List of Capital IQ company IDs (companyid).
                 Pass empty list to fetch all firms in date range.
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)

    Returns:
        TranscriptData containing:
            - firms: Dict[str, FirmTranscriptData] keyed by firm_id
            - metadata: Dict with query parameters and stats

        Each FirmTranscriptData contains:
            - firm_id: Capital IQ companyid
            - firm_name: Company name
            - sentences: List[TranscriptSentence]
            - metadata: Dict with:
                - permno: CRSP PERMNO (int, non-null for returned firms)
                - gvkey: Compustat GVKEY (str)
                - link_date: Date of GVKEY-PERMNO link
                - earnings_call_date: Transcript date
                - transcript_id: WRDS transcript identifier

    Note:
        Firms without PERMNO linkage are **skipped** (not included in output).
        This is by design - unlinked firms cannot be used in downstream
        sentiment analysis/event studies. See ADR-004 for rationale.

    Raises:
        WRDSConnectionError: If unable to connect to WRDS
        WRDSQueryError: If SQL query fails
        ValueError: If date format is invalid

    Example:
        >>> connector = WRDSConnector()
        >>> data = connector.fetch_transcripts(
        ...     firm_ids=["374372246"],
        ...     start_date="2023-01-01",
        ...     end_date="2023-03-31"
        ... )
        >>> firm = data.firms["374372246"]
        >>> print(firm.metadata["permno"])  # 16431
        >>> print(len(firm.sentences))  # 523
    """
```

### get_available_firm_ids

```python
def get_available_firm_ids(self) -> List[str]:
    """
    List all firm IDs available in WRDS Capital IQ.

    Returns:
        List of Capital IQ companyid strings, sorted alphabetically.

    Example:
        >>> connector = WRDSConnector()
        >>> firms = connector.get_available_firm_ids()
        >>> len(firms)
        4872
    """
```

### close

```python
def close(self) -> None:
    """
    Close the WRDS connection if owned by this instance.

    Safe to call multiple times. If the connection was passed in
    via constructor, this is a no-op (caller owns the connection).
    """
```

### Context Manager Support

```python
def __enter__(self) -> "WRDSConnector":
    """Support for 'with' statement."""
    return self

def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Close connection on context exit."""
    self.close()
```

## Private Methods

### _get_connection

```python
def _get_connection(self) -> wrds.Connection:
    """
    Get or create WRDS connection.

    Returns:
        Active WRDS connection

    Raises:
        WRDSConnectionError: If connection fails
    """
```

### _load_link_cache

```python
def _load_link_cache(self) -> None:
    """
    Load GVKEY → PERMNO linkage table into memory.

    Caches the ccmxpf_linktable for faster lookups.
    Called automatically on first query if preload_links=True.
    """
```

### _execute_transcript_query

```python
def _execute_transcript_query(
    self,
    firm_ids: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Execute the main transcript query with PERMNO linking.

    Returns DataFrame with columns:
        - firm_id, firm_name, transcript_id
        - earnings_call_date, componenttext, componentorder
        - speaker_type, gvkey, permno, link_date
    """
```

### _build_transcript_data

```python
def _build_transcript_data(
    self,
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> TranscriptData:
    """
    Convert WRDS DataFrame to TranscriptData structure.

    Groups rows by firm_id and builds FirmTranscriptData objects
    with sentences and metadata.
    """
```

## SQL Queries

### Main Transcript Query

```sql
WITH transcript_base AS (
    SELECT
        t.companyid::text AS firm_id,
        c.companyname AS firm_name,
        t.transcriptid::text AS transcript_id,
        t.mostimportantdateutc::date AS earnings_call_date,
        tc.componenttext,
        tc.componentorder,
        COALESCE(tc.speakertypename, 'Unknown') AS speaker_type
    FROM ciq.ciqtranscript t
    JOIN ciq.ciqcompany c ON t.companyid = c.companyid
    JOIN ciq.ciqtranscriptcomponent tc ON t.transcriptid = tc.transcriptid
    WHERE t.mostimportantdateutc BETWEEN %(start_date)s AND %(end_date)s
      AND t.keydeveventtypeid = 48  -- Earnings calls only
      AND (%(firm_ids)s IS NULL OR t.companyid = ANY(%(firm_ids)s))
),
gvkey_link AS (
    SELECT
        tb.*,
        wg.gvkey
    FROM transcript_base tb
    LEFT JOIN ciq.wrds_gvkey wg
        ON tb.firm_id::bigint = wg.companyid
),
permno_link AS (
    SELECT
        g.*,
        ccm.lpermno AS permno,
        ccm.linkdt AS link_date
    FROM gvkey_link g
    LEFT JOIN crsp.ccmxpf_linktable ccm
        ON g.gvkey = ccm.gvkey
        AND ccm.linktype IN ('LU', 'LC')
        AND ccm.linkprim IN ('P', 'C')
        AND g.earnings_call_date >= ccm.linkdt
        AND g.earnings_call_date <= COALESCE(ccm.linkenddt, '9999-12-31')
)
SELECT * FROM permno_link
ORDER BY firm_id, transcript_id, componentorder;
```

### Available Firms Query

```sql
SELECT DISTINCT t.companyid::text AS firm_id
FROM ciq.ciqtranscript t
WHERE t.keydeveventtypeid = 48
  AND (%(start_date)s IS NULL OR t.mostimportantdateutc >= %(start_date)s)
  AND (%(end_date)s IS NULL OR t.mostimportantdateutc <= %(end_date)s)
ORDER BY firm_id;
```

## Data Structures

### TranscriptSentence (existing, from models.py)

```python
@dataclass
class TranscriptSentence:
    sentence_id: str           # "{firm_id}_{transcript_id}_{position:04d}"
    raw_text: str              # Original text
    cleaned_text: Optional[str] = None  # Preprocessed (populated later)
    speaker_type: str = "Unknown"
    position: int = 0
```

### FirmTranscriptData (existing, from models.py)

```python
@dataclass
class FirmTranscriptData:
    firm_id: str
    firm_name: str
    sentences: List[TranscriptSentence]
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata keys for WRDS:
    #   - permno: int or None
    #   - gvkey: str or None
    #   - link_date: date or None
    #   - earnings_call_date: date
    #   - transcript_id: str
```

### TranscriptData (existing, from models.py)

```python
@dataclass
class TranscriptData:
    firms: Dict[str, FirmTranscriptData]  # Keyed by firm_id
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata keys for WRDS:
    #   - source: "wrds"
    #   - start_date: str
    #   - end_date: str
    #   - total_firms: int
    #   - total_sentences: int
    #   - firms_with_permno: int
    #   - query_time_seconds: float
```

## Exception Classes

```python
class WRDSError(Exception):
    """Base exception for WRDS connector errors."""

class WRDSConnectionError(WRDSError):
    """Raised when unable to connect to WRDS."""

class WRDSQueryError(WRDSError):
    """Raised when SQL query fails."""
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WRDS_USERNAME` | Conditional | None | WRDS account username |
| `WRDS_PASSWORD` | Conditional | None | WRDS account password |

Note: If `~/.pgpass` is configured for WRDS, environment variables are not required.

### WRDS .pgpass Format

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD
```

## Testing Strategy

### Unit Tests (Mocked)

```python
# tests/unit/test_wrds_connector.py

class TestWRDSConnectorInit:
    """Test initialization and connection handling."""

class TestWRDSConnectorFetchTranscripts:
    """Test transcript fetching with mocked responses."""

class TestWRDSConnectorPermnoLinking:
    """Test GVKEY → PERMNO linking logic."""

class TestWRDSConnectorEdgeCases:
    """Test handling of missing data, unlinked firms, etc."""
```

### Integration Tests (Real WRDS)

```python
# tests/integration/test_wrds_integration.py

@pytest.mark.integration
@pytest.mark.requires_wrds
class TestWRDSIntegration:
    """Integration tests against real WRDS (10 firms max)."""

    def test_fetch_known_firms(self):
        """Fetch transcripts for known firms with expected PERMNO."""

    def test_date_range_filtering(self):
        """Verify date range is respected."""

    def test_permno_present_for_linked_firms(self):
        """Confirm PERMNO appears in metadata for linked firms."""
```

### Test Fixtures

```python
@pytest.fixture
def mock_wrds_response():
    """Mock WRDS DataFrame response for unit tests."""
    return pd.DataFrame({
        "firm_id": ["374372246", "374372246"],
        "firm_name": ["Lamb Weston Holdings, Inc.", "Lamb Weston Holdings, Inc."],
        "transcript_id": ["123456", "123456"],
        "earnings_call_date": [date(2023, 1, 5), date(2023, 1, 5)],
        "componenttext": ["Good morning everyone.", "Q1 was strong."],
        "componentorder": [1, 2],
        "speaker_type": ["Operator", "CEO"],
        "gvkey": ["123456", "123456"],
        "permno": [16431, 16431],
        "link_date": [date(2022, 1, 1), date(2022, 1, 1)],
    })
```

## Validation Criteria

- [ ] `WRDSConnector` implements `DataConnector` interface
- [ ] Unit tests pass with mocked WRDS responses
- [ ] Integration test fetches 10 firms successfully
- [ ] PERMNO appears in metadata for all linked firms
- [ ] Unlinked firms have `permno=None` (not error)
- [ ] Data structure matches existing `TranscriptData` format
- [ ] Query time < 30 seconds for 5000 firms

## References

- `cloud/src/interfaces.py` - DataConnector interface
- `cloud/src/models.py` - TranscriptData, FirmTranscriptData
- `cloud/src/connectors/local_csv.py` - Reference implementation
- `docs/adr/adr_004_wrds_data_source.md` - Design rationale
