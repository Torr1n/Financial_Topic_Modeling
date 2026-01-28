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

### _setup_wrds_auth

```python
def _setup_wrds_auth(self) -> Optional[str]:
    """
    Setup non-interactive WRDS authentication.

    Priority order:
    1. Check for existing .pgpass file with correct permissions
    2. Create .pgpass from environment variables (WRDS_USERNAME, WRDS_PASSWORD)
    3. Attempt AWS Secrets Manager (wrds-credentials secret)

    Returns:
        WRDS username if found, None to trigger interactive prompt
    """
```

### _get_connection

```python
def _get_connection(self) -> wrds.Connection:
    """
    Get or create WRDS connection.

    Calls _setup_wrds_auth() first to configure credentials.
    Passes wrds_username parameter if available to avoid username prompt.

    Returns:
        Active WRDS connection

    Raises:
        WRDSConnectionError: If connection fails
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

Uses WRDS denormalized views (`wrds_transcript_detail`, `wrds_transcript_person`) with multi-transcript handling (latest per firm):

```sql
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
      AND td.keydeveventtypeid = 48
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

### Available Firms Query

```sql
SELECT DISTINCT td.companyid::text AS firm_id
FROM ciq.wrds_transcript_detail td
WHERE td.keydeveventtypeid = 48
ORDER BY firm_id;
```

### WRDS Tables Used

| Table | Purpose |
|-------|---------|
| `ciq.wrds_transcript_detail` | Transcript metadata (companyid, companyname, mostimportantdateutc) |
| `ciq.ciqtranscriptcomponent` | Full transcript text (componenttext) |
| `ciq.wrds_transcript_person` | Speaker info (speakertypename) |
| `ciq.wrds_gvkey` | Capital IQ → Compustat GVKEY mapping |
| `crsp.ccmxpf_linktable` | GVKEY → PERMNO linking |

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

### Credential Management

The connector auto-configures credentials via `_setup_wrds_auth()` with the following priority:

| Priority | Method | Description |
|----------|--------|-------------|
| 1 | `.pgpass` file | Existing `~/.pgpass` with WRDS entry and 0600 permissions |
| 2 | Environment variables | `WRDS_USERNAME` + `WRDS_PASSWORD` → auto-creates `.pgpass` |
| 3 | AWS Secrets Manager | Secret `wrds-credentials` with JSON `{"username":"xxx","password":"xxx"}` |
| 4 | Interactive prompt | Fallback if no credentials found |

**Note**: The WRDS Python library does NOT recognize `WRDS_USERNAME`/`WRDS_PASSWORD` directly. The connector auto-creates a `.pgpass` file from these variables to enable non-interactive authentication.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WRDS_USERNAME` | Conditional | None | WRDS account username |
| `WRDS_PASSWORD` | Conditional | None | WRDS account password |

### AWS Secrets Manager (for Batch/Lambda)

Store credentials as a secret named `wrds-credentials`:
```json
{
  "username": "your_wrds_username",
  "password": "your_wrds_password"
}
```

The connector writes to `/tmp/.pgpass` in ephemeral environments and sets `PGPASSFILE` env var.

### WRDS .pgpass Format

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD
```

**Critical**: File permissions must be `0600` (owner read/write only). PostgreSQL ignores the file otherwise.

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

- [x] `WRDSConnector` implements `DataConnector` interface
- [x] Unit tests pass with mocked WRDS responses (36/36)
- [x] Integration tests pass against real WRDS (7/7)
- [x] PERMNO appears in metadata for all returned firms
- [x] Unlinked firms are **skipped** (not included in output, logged at WARNING)
- [x] Data structure matches existing `TranscriptData` format
- [x] Non-interactive credential management works (env vars, .pgpass, AWS Secrets Manager)

## References

- `cloud/src/interfaces.py` - DataConnector interface
- `cloud/src/models.py` - TranscriptData, FirmTranscriptData
- `cloud/src/connectors/local_csv.py` - Reference implementation
- `docs/adr/adr_004_wrds_data_source.md` - Design rationale
