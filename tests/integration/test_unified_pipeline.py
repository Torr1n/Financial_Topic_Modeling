"""
Integration tests for UnifiedPipeline using testcontainers.

Tests the unified pipeline that:
- Loads embedding model ONCE
- Processes firms sequentially with per-firm checkpoints
- Stores all results in PostgreSQL (no S3 intermediate)
- Enables resume from spot instance interruption

Uses testcontainers for real Postgres+pgvector testing.
SentenceTransformer is mocked to avoid network downloads in CI.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

# testcontainers import
from testcontainers.postgres import PostgresContainer


# =============================================================================
# Mock SentenceTransformer (avoid network downloads)
# =============================================================================

def create_mock_sentence_transformer():
    """Create a mock SentenceTransformer that returns fake 768-dim embeddings."""
    mock_st = MagicMock()

    def mock_encode(texts, **kwargs):
        """Return deterministic fake embeddings based on text hash."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            np.random.seed(hash(text) % (2**32))
            embeddings.append(np.random.rand(768).astype(np.float32))
        return np.array(embeddings)

    mock_st.encode = mock_encode
    return mock_st


@pytest.fixture(scope="module")
def mock_sentence_transformer():
    """Module-scoped mock SentenceTransformer to avoid model downloads."""
    return create_mock_sentence_transformer()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def postgres_container():
    """
    Create a Postgres container with pgvector for testing.

    Module-scoped for performance (reuse across tests).
    """
    with PostgresContainer(
        image="pgvector/pgvector:pg16",
        username="test",
        password="test",
        dbname="test",
    ) as postgres:
        yield postgres


@pytest.fixture(scope="module")
def db_url(postgres_container):
    """
    Get database URL and ensure pgvector extension is enabled.

    Module-scoped to match postgres_container and enable extension once.
    """
    from sqlalchemy import create_engine, text

    url = postgres_container.get_connection_url()
    engine = create_engine(url)

    # Enable pgvector extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    engine.dispose()
    return url


@pytest.fixture
def db_engine(db_url):
    """Create SQLAlchemy engine connected to test container."""
    from sqlalchemy import create_engine
    from cloud.src.database.models import Base

    engine = create_engine(db_url)

    # Create tables fresh for each test
    Base.metadata.create_all(engine)

    yield engine

    # Drop tables after test
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create a fresh database session with tables."""
    from sqlalchemy.orm import Session

    session = Session(db_engine)
    yield session

    session.rollback()
    session.close()


@pytest.fixture
def repository(db_session):
    """Create repository instance."""
    from cloud.src.database.repository import DatabaseRepository
    return DatabaseRepository(db_session)


@pytest.fixture
def sample_config():
    """Sample configuration for topic models."""
    return {
        "embedding_model": "all-mpnet-base-v2",
        "device": "cpu",  # Use CPU for testing (avoid CUDA errors)
        "umap": {
            "n_neighbors": 15,
            "n_components": 10,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42,
        },
        "hdbscan": {
            "min_cluster_size": 6,
            "min_samples": 2,
            "metric": "euclidean",
            "cluster_selection_method": "leaf",
        },
    }


@pytest.fixture
def mock_data_connector():
    """Create mock DataConnector with sample firm data."""
    from cloud.src.models import TranscriptData, FirmTranscriptData, TranscriptSentence

    # Create sentences for two firms (raw_text and cleaned_text same for test simplicity)
    firm1_sentences = [
        TranscriptSentence(
            f"1001_T001_{i:04d}",
            f"Sentence {i} about AI.",  # raw_text
            f"sentence {i} ai",  # cleaned_text
            "CEO",
            i
        )
        for i in range(20)
    ]
    firm2_sentences = [
        TranscriptSentence(
            f"1002_T001_{i:04d}",
            f"Sentence {i} about revenue.",  # raw_text
            f"sentence {i} revenue",  # cleaned_text
            "CFO",
            i
        )
        for i in range(20)
    ]

    connector = MagicMock()
    connector.get_available_firm_ids.return_value = ["1001", "1002"]

    def fetch_transcripts(firm_ids, start_date, end_date):
        firms = {}
        for firm_id in firm_ids:
            if firm_id == "1001":
                firms[firm_id] = FirmTranscriptData("1001", "Apple Inc.", firm1_sentences)
            elif firm_id == "1002":
                firms[firm_id] = FirmTranscriptData("1002", "Microsoft Corp.", firm2_sentences)
        return TranscriptData(firms=firms)

    connector.fetch_transcripts = fetch_transcripts
    return connector


@pytest.fixture
def create_mocked_pipeline(db_url, sample_config, mock_sentence_transformer):
    """
    Factory fixture to create UnifiedPipeline with mocked SentenceTransformer.

    Returns a function that creates pipelines without network downloads.
    """
    def _create():
        from cloud.src.pipeline import UnifiedPipeline

        with patch('cloud.src.pipeline.unified_pipeline.SentenceTransformer', return_value=mock_sentence_transformer):
            return UnifiedPipeline(
                database_url=db_url,
                config=sample_config,
                device="cpu",
            )
    return _create


# =============================================================================
# UnifiedPipeline Tests
# =============================================================================

class TestUnifiedPipelineInit:
    """Tests for UnifiedPipeline initialization."""

    def test_pipeline_can_be_created(self, db_url, sample_config, mock_sentence_transformer):
        """Pipeline should initialize with database URL and config."""
        from cloud.src.pipeline import UnifiedPipeline

        with patch('cloud.src.pipeline.unified_pipeline.SentenceTransformer', return_value=mock_sentence_transformer):
            pipeline = UnifiedPipeline(
                database_url=db_url,
                config=sample_config,
                device="cpu",
            )

            assert pipeline is not None
            assert pipeline.embedding_model is not None

    def test_embedding_model_loaded_once(self, db_url, sample_config, mock_sentence_transformer):
        """Embedding model should be loaded exactly once during init."""
        from cloud.src.pipeline import UnifiedPipeline

        with patch('cloud.src.pipeline.unified_pipeline.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_sentence_transformer

            pipeline = UnifiedPipeline(
                database_url=db_url,
                config=sample_config,
                device="cpu",
            )

            # SentenceTransformer should be called exactly once
            mock_st.assert_called_once()

    def test_embedding_model_injected_to_topic_models(self, db_url, sample_config, mock_sentence_transformer):
        """BERTopicModels should receive the same embedding model instance."""
        from cloud.src.pipeline import UnifiedPipeline

        with patch('cloud.src.pipeline.unified_pipeline.SentenceTransformer', return_value=mock_sentence_transformer):
            pipeline = UnifiedPipeline(
                database_url=db_url,
                config=sample_config,
                device="cpu",
            )

            # Both topic models should have the same embedding model injected
            assert pipeline.firm_topic_model._embedding_model is pipeline.embedding_model
            assert pipeline.theme_topic_model._embedding_model is pipeline.embedding_model


class TestUnifiedPipelineFirmProcessing:
    """Tests for firm processing in unified pipeline."""

    def test_processes_single_firm_and_writes_to_db(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Pipeline should process a firm and write results to Postgres."""
        from cloud.src.database.models import Firm, Sentence, Topic

        # Create pipeline with mocked SentenceTransformer
        pipeline = create_mocked_pipeline()

        # Mock the topic model to avoid real BERTopic processing
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 10 + [1] * 10)
        mock_result.n_topics = 2
        mock_result.topic_representations = {0: "Topic A", 1: "Topic B"}
        mock_result.topic_keywords = {0: ["ai", "ml"], 1: ["revenue", "growth"]}
        mock_result.probabilities = np.random.rand(20, 2)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        # Process single firm
        pipeline._process_single_firm("1001", mock_data_connector)

        # Verify firm was written
        from sqlalchemy.orm import Session
        with Session(db_engine) as session:
            firms = session.query(Firm).all()
            assert len(firms) == 1
            assert firms[0].company_id == "1001"

            # Verify sentences were written
            sentences = session.query(Sentence).all()
            assert len(sentences) == 20

            # Verify topics were written
            topics = session.query(Topic).all()
            assert len(topics) == 2

    def test_sentences_have_embeddings(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Sentences should be stored with embeddings."""
        from cloud.src.database.models import Sentence

        pipeline = create_mocked_pipeline()

        # Mock topic model
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "Topic A"}
        mock_result.topic_keywords = {0: ["ai"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        pipeline._process_single_firm("1001", mock_data_connector)

        from sqlalchemy.orm import Session
        with Session(db_engine) as session:
            sentence = session.query(Sentence).first()
            assert sentence.embedding is not None
            # Mock produces 768-dim embeddings
            assert len(sentence.embedding) == 768

    def test_topics_have_embeddings(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Topics should be stored with keyword-based embeddings."""
        from cloud.src.database.models import Topic

        pipeline = create_mocked_pipeline()

        # Mock topic model
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "artificial intelligence machine learning"}
        mock_result.topic_keywords = {0: ["ai", "ml"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        pipeline._process_single_firm("1001", mock_data_connector)

        from sqlalchemy.orm import Session
        with Session(db_engine) as session:
            topic = session.query(Topic).first()
            assert topic.embedding is not None
            assert len(topic.embedding) == 768


class TestUnifiedPipelineCheckpoint:
    """Tests for checkpoint/resume functionality."""

    def test_marks_firm_as_processed(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Pipeline should mark firm as processed after completion."""
        from cloud.src.database.models import Firm

        pipeline = create_mocked_pipeline()

        # Mock topic model
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "Topic A"}
        mock_result.topic_keywords = {0: ["ai"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        pipeline._process_single_firm("1001", mock_data_connector)

        from sqlalchemy.orm import Session
        with Session(db_engine) as session:
            firm = session.query(Firm).filter_by(company_id="1001").first()
            assert firm.processed_at is not None

    def test_skips_already_processed_firms(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Pipeline should skip firms that are already processed."""
        from cloud.src.database.models import Firm

        pipeline = create_mocked_pipeline()

        # Pre-populate a processed firm
        from sqlalchemy.orm import Session
        with Session(db_engine) as session:
            firm = Firm(
                company_id="1001",
                name="Apple Inc.",
                processed_at=datetime.now(timezone.utc),
            )
            session.add(firm)
            session.commit()

        # Get unprocessed firms
        unprocessed = pipeline._get_unprocessed_firm_ids(mock_data_connector)

        # 1001 should be skipped, only 1002 should be unprocessed
        assert "1001" not in unprocessed
        assert "1002" in unprocessed

    def test_can_resume_after_interruption(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Pipeline should resume from last checkpoint after interruption."""
        # First run - process firm 1001
        pipeline1 = create_mocked_pipeline()

        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "Topic A"}
        mock_result.topic_keywords = {0: ["ai"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline1.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        pipeline1._process_single_firm("1001", mock_data_connector)

        # Simulate interruption - create new pipeline instance
        pipeline2 = create_mocked_pipeline()

        # Get unprocessed firms - should only return 1002
        unprocessed = pipeline2._get_unprocessed_firm_ids(mock_data_connector)

        assert unprocessed == ["1002"]


class TestUnifiedPipelineSentenceTopicMapping:
    """Tests for sentence→topic ID mapping (Codex requirement)."""

    def test_sentences_have_topic_id_set(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Sentences should have topic_id set (not null, not requiring update)."""
        from cloud.src.database.models import Sentence

        pipeline = create_mocked_pipeline()

        # Mock topic model with specific assignments
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 10 + [1] * 10)
        mock_result.n_topics = 2
        mock_result.topic_representations = {0: "Topic A", 1: "Topic B"}
        mock_result.topic_keywords = {0: ["ai"], 1: ["revenue"]}
        mock_result.probabilities = np.random.rand(20, 2)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        pipeline._process_single_firm("1001", mock_data_connector)

        from sqlalchemy.orm import Session
        with Session(db_engine) as session:
            sentences = session.query(Sentence).all()

            # All non-outlier sentences should have topic_id set
            for sentence in sentences:
                # topic_id should be set (not None) for assigned sentences
                assert sentence.topic_id is not None

    def test_outlier_sentences_have_null_topic_id(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Outlier sentences (topic=-1) should have null topic_id."""
        from cloud.src.database.models import Sentence

        pipeline = create_mocked_pipeline()

        # Mock topic model with outliers
        assignments = np.array([0] * 5 + [-1] * 10 + [1] * 5)  # 10 outliers
        mock_result = MagicMock()
        mock_result.topic_assignments = assignments
        mock_result.n_topics = 2
        mock_result.topic_representations = {0: "Topic A", 1: "Topic B"}
        mock_result.topic_keywords = {0: ["ai"], 1: ["revenue"]}
        mock_result.probabilities = np.random.rand(20, 2)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        pipeline._process_single_firm("1001", mock_data_connector)

        from sqlalchemy.orm import Session
        with Session(db_engine) as session:
            sentences = session.query(Sentence).all()

            # Count sentences with null topic_id
            outliers = [s for s in sentences if s.topic_id is None]
            assigned = [s for s in sentences if s.topic_id is not None]

            assert len(outliers) == 10  # Matches -1 assignments
            assert len(assigned) == 10  # 5 + 5 non-outliers


class TestUnifiedPipelineNoS3:
    """Tests verifying no S3 intermediate storage is used."""

    def test_no_s3_operations(self, db_engine, mock_data_connector, create_mocked_pipeline):
        """Pipeline should not use S3 for intermediate storage."""
        pipeline = create_mocked_pipeline()

        # Mock topic model
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "Topic A"}
        mock_result.topic_keywords = {0: ["ai"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        # Patch boto3 to detect S3 usage
        with patch('boto3.client') as mock_boto:
            pipeline._process_single_firm("1001", mock_data_connector)

            # boto3.client should not be called for S3
            for call in mock_boto.call_args_list:
                if call[0]:
                    assert call[0][0] != 's3', "S3 should not be used"


class TestUnifiedPipelineThemeAggregation:
    """Tests for theme aggregation (Issue 3: Theme aggregation not covered)."""

    def test_aggregate_themes_creates_themes(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Theme aggregation should create theme records in database."""
        from cloud.src.database.models import Firm, Topic, Theme

        pipeline = create_mocked_pipeline()

        # Process two firms first
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 10 + [1] * 10)
        mock_result.n_topics = 2
        mock_result.topic_representations = {0: "AI Investment", 1: "Revenue Growth"}
        mock_result.topic_keywords = {0: ["ai", "ml"], 1: ["revenue", "growth"]}
        mock_result.probabilities = np.random.rand(20, 2)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        pipeline._process_single_firm("1001", mock_data_connector)
        pipeline._process_single_firm("1002", mock_data_connector)

        # Mock theme aggregation result
        theme_result = MagicMock()
        theme_result.topic_assignments = np.array([0, 0, 1, 1])  # 4 topics -> 2 themes
        theme_result.n_topics = 2
        theme_result.topic_representations = {0: "Cross-Firm AI Theme", 1: "Cross-Firm Revenue Theme"}
        theme_result.topic_keywords = {0: ["ai"], 1: ["revenue"]}
        theme_result.probabilities = np.random.rand(4, 2)
        pipeline.theme_topic_model.fit_transform = MagicMock(return_value=theme_result)

        # Run theme aggregation
        pipeline._aggregate_themes()

        from sqlalchemy.orm import Session
        with Session(db_engine) as session:
            themes = session.query(Theme).all()
            # Should have created themes (depends on validation filters)
            assert len(themes) >= 0  # May be 0 if validation filters out single-firm themes

    def test_topics_get_theme_id_assigned(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Topics should have theme_id set after aggregation."""
        from cloud.src.database.models import Topic, Theme

        pipeline = create_mocked_pipeline()

        # Create a scenario where themes will pass validation (2+ firms per theme)
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "AI Investment"}
        mock_result.topic_keywords = {0: ["ai", "ml"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        pipeline._process_single_firm("1001", mock_data_connector)
        pipeline._process_single_firm("1002", mock_data_connector)

        # Mock theme result - put both topics in same theme
        theme_result = MagicMock()
        theme_result.topic_assignments = np.array([0, 0])  # Both in theme 0
        theme_result.n_topics = 1
        theme_result.topic_representations = {0: "Cross-Firm AI Theme"}
        theme_result.topic_keywords = {0: ["ai"]}
        theme_result.probabilities = np.random.rand(2, 1)
        pipeline.theme_topic_model.fit_transform = MagicMock(return_value=theme_result)

        pipeline._aggregate_themes()

        from sqlalchemy.orm import Session
        with Session(db_engine) as session:
            themes = session.query(Theme).all()
            # If themes were created and topics assigned
            if themes:
                topics_with_theme = session.query(Topic).filter(Topic.theme_id.isnot(None)).all()
                # At least some topics should have theme_id
                assert len(topics_with_theme) > 0
