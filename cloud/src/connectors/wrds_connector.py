"""
WRDSConnector - WRDS data connector for earnings call transcripts.

Fetches transcripts from WRDS Capital IQ tables with automatic
PERMNO/GVKEY linking for downstream event studies.

Design:
    - Lazy database connection initialization
    - PERMNO linking via GVKEY at ingestion time
    - Firms without PERMNO are skipped (logged)
    - Multi-transcript: selects latest transcript per firm per date range
    - SpaCy-based sentence splitting and NLP preprocessing
    - Filter Operator speaker type and single-word sentences
"""

import json
import logging
import os
import stat
from datetime import datetime
from typing import List, Optional, Set

import pandas as pd
import spacy

from cloud.src.interfaces import DataConnector
from cloud.src.models import TranscriptData, FirmTranscriptData, TranscriptSentence

logger = logging.getLogger(__name__)

# Custom stopwords for earnings call transcripts (common low-signal words)
CUSTOM_STOPWORDS = {"yes", "thank", "thanks", "question", "questions", "afternoon", "operator", "welcome"}


# =============================================================================
# Exception Classes
# =============================================================================


class WRDSError(Exception):
    """Base exception for WRDS connector errors."""
    pass


class WRDSConnectionError(WRDSError):
    """Raised when unable to connect to WRDS."""
    pass


class WRDSQueryError(WRDSError):
    """Raised when SQL query fails."""
    pass


# =============================================================================
# SQL Queries
# =============================================================================

# Main transcript query with PERMNO linking and multi-transcript handling
# Uses WRDS denormalized views: wrds_transcript_detail, wrds_transcript_person
# Window function at transcript level (NOT component level) to preserve all components
TRANSCRIPT_QUERY = """
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
"""

# Query for available firm IDs (no date filter per interface spec)
AVAILABLE_FIRMS_QUERY = """
SELECT DISTINCT td.companyid::text AS firm_id
FROM ciq.wrds_transcript_detail td
WHERE td.keydeveventtypeid = 48
ORDER BY firm_id;
"""

# Query for firm IDs in a date range WITH PERMNO linking (for prefetch)
# Only returns firms that have valid PERMNO links (same filters as TRANSCRIPT_QUERY)
FIRM_IDS_IN_RANGE_QUERY = """
WITH firms_in_range AS (
    SELECT DISTINCT
        td.companyid::text AS firm_id,
        wg.gvkey
    FROM ciq.wrds_transcript_detail td
    LEFT JOIN ciq.wrds_gvkey wg ON td.companyid = wg.companyid
    WHERE td.mostimportantdateutc BETWEEN %(start_date)s AND %(end_date)s
      AND td.keydeveventtypeid = 48
),
with_permno AS (
    SELECT DISTINCT
        fir.firm_id,
        ccm.lpermno AS permno
    FROM firms_in_range fir
    LEFT JOIN crsp.ccmxpf_linktable ccm ON fir.gvkey = ccm.gvkey
        AND ccm.linktype IN ('LU', 'LC')
        AND ccm.linkprim IN ('P', 'C')
)
SELECT firm_id FROM with_permno WHERE permno IS NOT NULL ORDER BY firm_id;
"""


# =============================================================================
# WRDSConnector Class
# =============================================================================


class WRDSConnector(DataConnector):
    """
    WRDS connector with CRSP/Compustat identifier enrichment.

    Fetches earnings call transcripts from Capital IQ tables and
    automatically links to CRSP PERMNO via Compustat GVKEY.

    Attributes:
        _conn: WRDS database connection (lazily initialized)
        _owns_connection: Whether this instance owns the connection
        _nlp: SpaCy NLP model for sentence splitting and preprocessing
        _stopwords: Combined stopwords set

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

    # Named entity types to filter out (names, dates, numbers)
    _FILTERED_ENTITY_TYPES = {"PERSON", "DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT"}

    def __init__(
        self,
        connection=None,
        preload_links: bool = True,
    ) -> None:
        """
        Initialize the WRDS connector.

        Args:
            connection: Optional existing WRDS connection. If None, a new
                       connection will be created lazily on first query.
            preload_links: If True, cache the GVKEY -> PERMNO linkage table
                          on first query for faster subsequent lookups.
                          (Currently unused - linking done in SQL)

        Environment Variables:
            WRDS_USERNAME: WRDS account username (if not using .pgpass)
            WRDS_PASSWORD: WRDS account password (if not using .pgpass)
        """
        self._conn = connection
        self._owns_connection = connection is None
        self._preload_links = preload_links

        # Load SpaCy model for sentence splitting and preprocessing
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            raise

        # Combine built-in stopwords with custom earnings call stopwords
        self._stopwords: Set[str] = self._nlp.Defaults.stop_words | CUSTOM_STOPWORDS

        logger.debug("WRDSConnector initialized (connection=%s)", "provided" if connection else "lazy")

    def __enter__(self) -> "WRDSConnector":
        """Support for 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close connection on context exit."""
        self.close()

    def _setup_wrds_auth(self) -> Optional[str]:
        """
        Setup non-interactive WRDS authentication.

        Priority order:
        1. Check for existing .pgpass file with correct permissions
        2. Create .pgpass from environment variables (WRDS_USERNAME, WRDS_PASSWORD)
        3. Attempt AWS Secrets Manager (if boto3 available)

        Returns:
            WRDS username if found, None to trigger interactive prompt
        """
        # Check if .pgpass already exists with correct permissions
        pgpass_default = os.path.expanduser("~/.pgpass")
        if os.path.exists(pgpass_default):
            # Verify it has WRDS entry and correct permissions
            try:
                file_stat = os.stat(pgpass_default)
                if file_stat.st_mode & 0o077 == 0:  # Only owner can read/write
                    with open(pgpass_default, "r") as f:
                        if "wrds-pgdata.wharton.upenn.edu" in f.read():
                            logger.info("Found existing .pgpass file with WRDS entry")
                            # Return username from env if available, else None
                            return os.environ.get("WRDS_USERNAME")
            except Exception as e:
                logger.debug(f"Could not read .pgpass: {e}")

        # Try environment variables
        username = os.environ.get("WRDS_USERNAME")
        password = os.environ.get("WRDS_PASSWORD")

        if username and password:
            # Create .pgpass file
            pgpass_content = f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{username}:{password}\n"

            # Use /tmp for AWS Lambda/Batch, otherwise home directory
            if os.access("/tmp", os.W_OK):
                pgpass_path = "/tmp/.pgpass"
            else:
                pgpass_path = pgpass_default

            try:
                with open(pgpass_path, "w") as f:
                    f.write(pgpass_content)

                # Set permissions to 0600 (required by PostgreSQL)
                os.chmod(pgpass_path, stat.S_IRUSR | stat.S_IWUSR)

                # Tell PostgreSQL where to find it
                os.environ["PGPASSFILE"] = pgpass_path

                logger.info(f"Created .pgpass file at {pgpass_path}")
                return username
            except Exception as e:
                logger.warning(f"Failed to create .pgpass: {e}")

        # Try AWS Secrets Manager (if in AWS environment)
        try:
            import boto3

            client = boto3.client("secretsmanager")
            response = client.get_secret_value(SecretId="wrds-credentials")
            credentials = json.loads(response["SecretString"])

            username = credentials["username"]
            password = credentials["password"]

            pgpass_path = "/tmp/.pgpass"
            pgpass_content = f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{username}:{password}\n"

            with open(pgpass_path, "w") as f:
                f.write(pgpass_content)

            os.chmod(pgpass_path, stat.S_IRUSR | stat.S_IWUSR)
            os.environ["PGPASSFILE"] = pgpass_path

            logger.info("Created .pgpass from AWS Secrets Manager")
            return username
        except Exception as e:
            logger.debug(f"AWS Secrets Manager not available: {e}")

        # Fall back to interactive
        logger.warning("No WRDS credentials found - will prompt interactively")
        return None

    def _get_connection(self):
        """
        Get or create WRDS connection.

        Authentication priority:
        1. Existing .pgpass file
        2. WRDS_USERNAME + WRDS_PASSWORD environment variables
        3. AWS Secrets Manager (wrds-credentials secret)
        4. Interactive prompt (fallback)

        Returns:
            Active WRDS connection

        Raises:
            WRDSConnectionError: If connection fails
        """
        if self._conn is not None:
            return self._conn

        # Import wrds lazily to avoid ImportError in environments without it
        try:
            import wrds
        except ImportError:
            raise WRDSConnectionError(
                "WRDS Python package not installed. Run: pip install wrds"
            )

        try:
            # Setup authentication
            username = self._setup_wrds_auth()

            # Connect (pass username if available to avoid username prompt)
            if username:
                self._conn = wrds.Connection(wrds_username=username)
            else:
                self._conn = wrds.Connection()

            logger.info("Connected to WRDS database")
            return self._conn
        except Exception as e:
            raise WRDSConnectionError(f"Failed to connect to WRDS: {e}")

    def _validate_date_format(self, date_str: str) -> None:
        """
        Validate date string is in YYYY-MM-DD format.

        Args:
            date_str: Date string to validate

        Raises:
            ValueError: If date format is invalid
        """
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD format."
            )

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences using SpaCy.

        Args:
            text: Text to split (may contain multiple sentences)

        Returns:
            List of sentence strings
        """
        if not text or pd.isna(text):
            return []

        doc = self._nlp(str(text))
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text with NLP pipeline: lowercase, lemmatize, NER removal, stopwords.

        Pipeline:
        1. Tokenize with SpaCy (includes NER)
        2. For each token: use lowercase lemma
        3. Filter out: stopwords, punctuation, spaces, named entities (PERSON, DATE, etc.)
        4. Return cleaned text

        Args:
            text: Input text

        Returns:
            Preprocessed text (lowercase, lemmatized, filtered)
        """
        if not text:
            return ""

        doc = self._nlp(text)

        # Build set of token indices that are part of filtered entity types
        entity_token_indices = set()
        for ent in doc.ents:
            if ent.label_ in self._FILTERED_ENTITY_TYPES:
                for i in range(ent.start, ent.end):
                    entity_token_indices.add(i)

        # Filter tokens: lowercase lemma, exclude stopwords/punct/space/entities
        filtered_tokens = []
        for i, token in enumerate(doc):
            # Skip punctuation, spaces, and entity tokens
            if token.is_punct or token.is_space or i in entity_token_indices:
                continue

            # Use lowercase lemma
            lemma = token.lemma_.lower()

            # Skip stopwords (check lemma form)
            if lemma in self._stopwords:
                continue

            # Skip purely numeric tokens
            if token.like_num:
                continue

            filtered_tokens.append(lemma)

        return " ".join(filtered_tokens)

    def _execute_transcript_query(
        self,
        firm_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Execute the main transcript query with PERMNO linking.

        Args:
            firm_ids: List of Capital IQ company IDs (empty = all firms)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with transcript components and PERMNO linking info

        Raises:
            WRDSQueryError: If SQL query fails
        """
        conn = self._get_connection()

        # Convert firm_ids to integers for WRDS (companyid is bigint)
        # Pass None if empty to trigger the "IS NULL" branch in SQL
        if firm_ids:
            firm_ids_int = [int(fid) for fid in firm_ids]
        else:
            firm_ids_int = None

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "firm_ids": firm_ids_int,
        }

        try:
            df = conn.raw_sql(TRANSCRIPT_QUERY, params=params)
            logger.info(f"WRDS query returned {len(df)} rows")
            return df
        except Exception as e:
            raise WRDSQueryError(f"WRDS query failed: {e}")

    def _build_transcript_data(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> TranscriptData:
        """
        Convert WRDS DataFrame to TranscriptData structure.

        Groups rows by firm_id and builds FirmTranscriptData objects.
        Skips firms without PERMNO linking.

        Args:
            df: DataFrame from WRDS query
            start_date: Start date (for metadata)
            end_date: End date (for metadata)

        Returns:
            TranscriptData with firms mapped to their sentences
        """
        if df.empty:
            return TranscriptData(firms={})

        firms_dict = {}
        skipped_firms = []

        for firm_id, firm_group in df.groupby("firm_id"):
            firm_id_str = str(firm_id)

            # Check PERMNO linking - skip if no PERMNO
            permno = firm_group["permno"].iloc[0]
            if pd.isna(permno):
                skipped_firms.append(firm_id_str)
                continue

            firm_name = firm_group["firm_name"].iloc[0]
            transcript_id = str(firm_group["transcript_id"].iloc[0])
            earnings_call_date = firm_group["earnings_call_date"].iloc[0]
            gvkey = firm_group["gvkey"].iloc[0]
            link_date = firm_group["link_date"].iloc[0]

            # Filter out Operator speaker type (low-signal content)
            firm_group = firm_group[
                firm_group["speakertypename"].str.lower() != "operator"
            ].copy()

            if firm_group.empty:
                logger.debug(f"Firm {firm_id_str} has no non-Operator content")
                continue

            # Sort by componentorder
            firm_group = firm_group.sort_values("componentorder")

            # Process components into sentences
            sentences = []
            sentence_position = 0

            for _, row in firm_group.iterrows():
                component_text = row.get("componenttext", "")
                if pd.isna(component_text) or not str(component_text).strip():
                    continue

                speaker_type = row.get("speakertypename", None)
                if pd.isna(speaker_type):
                    speaker_type = None

                # Split component into sentences
                component_sentences = self._split_into_sentences(str(component_text))

                for sent_text in component_sentences:
                    # Preprocess: lowercase, lemmatize, NER removal, stopwords
                    cleaned_text = self._preprocess_text(sent_text)

                    # Skip empty sentences after preprocessing
                    if not cleaned_text.strip():
                        continue

                    # Skip single-word sentences (low signal)
                    word_count = len(cleaned_text.split())
                    if word_count <= 1:
                        continue

                    # Generate sentence_id: {firm_id}_{transcript_id}_{position:04d}
                    sentence_id = f"{firm_id_str}_{transcript_id}_{sentence_position:04d}"

                    sentence = TranscriptSentence(
                        sentence_id=sentence_id,
                        raw_text=sent_text,  # Original for observability
                        cleaned_text=cleaned_text,  # Preprocessed for topic modeling
                        speaker_type=speaker_type,
                        position=sentence_position,
                    )
                    sentences.append(sentence)
                    sentence_position += 1

            if not sentences:
                logger.debug(f"Firm {firm_id_str} has no sentences after preprocessing")
                continue

            firms_dict[firm_id_str] = FirmTranscriptData(
                firm_id=firm_id_str,
                firm_name=firm_name,
                sentences=sentences,
                metadata={
                    "permno": int(permno),
                    "gvkey": str(gvkey) if pd.notna(gvkey) else None,
                    "link_date": link_date,
                    "earnings_call_date": earnings_call_date,
                    "transcript_id": transcript_id,
                    "n_components": len(firm_group),
                    "date_range": f"{start_date} to {end_date}",
                },
            )

        # Log skipped firms
        if skipped_firms:
            logger.warning(
                f"Skipped {len(skipped_firms)} firms without PERMNO: {skipped_firms[:5]}{'...' if len(skipped_firms) > 5 else ''}"
            )

        logger.info(f"Built TranscriptData for {len(firms_dict)} firms")
        return TranscriptData(firms=firms_dict)

    def fetch_transcripts(
        self,
        firm_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> TranscriptData:
        """
        Fetch transcript sentences for specified firms and date range.

        Preprocessing pipeline:
        1. Filter out Operator speaker_type rows
        2. Split components into sentences
        3. For each sentence: lowercase, lemmatize, remove named entities, remove stopwords
        4. Filter out sentences with <= 1 word remaining
        5. Skip firms without PERMNO linking

        Args:
            firm_ids: List of Capital IQ company IDs.
                     Pass empty list to fetch all firms in date range.
            start_date: YYYY-MM-DD format (inclusive)
            end_date: YYYY-MM-DD format (inclusive)

        Returns:
            TranscriptData with firms mapped to their preprocessed sentences.
            Firms without PERMNO are NOT included (logged and skipped).

        Raises:
            WRDSConnectionError: If unable to connect to WRDS
            WRDSQueryError: If SQL query fails
            ValueError: If date format is invalid
        """
        # Validate date formats
        self._validate_date_format(start_date)
        self._validate_date_format(end_date)

        logger.info(
            f"Fetching transcripts for {len(firm_ids) if firm_ids else 'all'} firm IDs, "
            f"{start_date} to {end_date}"
        )

        # Execute WRDS query
        df = self._execute_transcript_query(firm_ids, start_date, end_date)

        # Build TranscriptData structure
        return self._build_transcript_data(df, start_date, end_date)

    def get_available_firm_ids(self) -> List[str]:
        """
        List all firm IDs available in WRDS Capital IQ.

        Note: No date filters per interface spec.

        Returns:
            Sorted list of unique firm IDs (as strings)
        """
        conn = self._get_connection()

        try:
            df = conn.raw_sql(AVAILABLE_FIRMS_QUERY)
            firm_ids = df["firm_id"].unique().tolist()
            return sorted(firm_ids)
        except Exception as e:
            raise WRDSQueryError(f"Failed to fetch available firm IDs: {e}")

    def get_firm_ids_in_range(self, start_date: str, end_date: str) -> List[str]:
        """
        Get firm IDs with transcripts in date range that have valid PERMNO links.

        This is a lightweight query for prefetch to discover firms without
        loading full transcript data. Only returns firms that will pass the
        PERMNO filter in fetch_transcripts().

        Note: This is NOT part of the DataConnector interface - it's a helper
        for WRDSPrefetcher to avoid loading entire quarter into memory.

        Args:
            start_date: YYYY-MM-DD format (inclusive)
            end_date: YYYY-MM-DD format (inclusive)

        Returns:
            Sorted list of firm IDs with valid PERMNO links
        """
        self._validate_date_format(start_date)
        self._validate_date_format(end_date)

        conn = self._get_connection()

        params = {"start_date": start_date, "end_date": end_date}

        try:
            df = conn.raw_sql(FIRM_IDS_IN_RANGE_QUERY, params=params)
            firm_ids = df["firm_id"].unique().tolist()
            logger.info(f"Found {len(firm_ids)} firms with PERMNO in date range")
            return sorted(firm_ids)
        except Exception as e:
            raise WRDSQueryError(f"Failed to fetch firm IDs in range: {e}")

    def close(self) -> None:
        """
        Close the WRDS connection if owned by this instance.

        Safe to call multiple times. If the connection was passed in
        via constructor, this is a no-op (caller owns the connection).
        """
        if self._conn is not None and self._owns_connection:
            try:
                self._conn.close()
                logger.debug("WRDS connection closed")
            except Exception as e:
                logger.warning(f"Error closing WRDS connection: {e}")
            finally:
                self._conn = None
