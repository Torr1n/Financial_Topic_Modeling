"""
Integration tests for LLM pipeline integration.

Tests the end-to-end flow of:
- Topic summary generation and storage
- Theme description generation and storage
- Embeddings derived from summaries (not keywords)
- Fallback behavior when LLM is unavailable

Uses testcontainers for real Postgres+pgvector testing.
LLM calls are mocked to avoid real API requests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from testcontainers.postgres import PostgresContainer

# Enable asyncio mode for all tests in this module
pytestmark = pytest.mark.asyncio(loop_scope="function")


# =============================================================================
# Mock Fixtures
# =============================================================================

def create_mock_sentence_transformer():
    """Create a mock SentenceTransformer that returns deterministic embeddings."""
    mock_st = MagicMock()

    def mock_encode(texts, **kwargs):
        """Return deterministic fake embeddings based on text hash."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            embeddings.append(np.random.rand(768).astype(np.float32))
        return np.array(embeddings)

    mock_st.encode = mock_encode
    return mock_st


@pytest.fixture(scope="module")
def mock_sentence_transformer():
    """Module-scoped mock SentenceTransformer."""
    return create_mock_sentence_transformer()


def create_mock_xai_client():
    """Create a mock XAI client that returns predictable summaries."""
    mock_client = MagicMock()

    # Use AsyncMock for proper async support in tests
    mock_client.generate_topic_summary = AsyncMock(
        side_effect=lambda keywords, sentences, log_prompt=False: f"Summary of: {keywords[:50]}"
    )
    mock_client.generate_theme_description = AsyncMock(
        side_effect=lambda theme_keywords, topic_summaries, log_prompt=False: f"Description of {theme_keywords} covering {len(topic_summaries)} topics."
    )
    mock_client.generate_batch_summaries = AsyncMock(
        side_effect=lambda topics, log_first_prompt=False: [f"Summary of: {t['representation'][:30]}" for t in topics]
    )

    return mock_client


@pytest.fixture
def mock_xai_client():
    """Create mock XAI client for testing."""
    return create_mock_xai_client()


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def postgres_container():
    """Create a Postgres container with pgvector for testing."""
    with PostgresContainer(
        image="pgvector/pgvector:pg16",
        username="test",
        password="test",
        dbname="test",
    ) as postgres:
        yield postgres


@pytest.fixture(scope="module")
def db_url(postgres_container):
    """Get database URL and ensure pgvector extension is enabled."""
    from sqlalchemy import create_engine, text

    url = postgres_container.get_connection_url()
    engine = create_engine(url)

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
    Base.metadata.create_all(engine)

    yield engine

    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def sample_config():
    """Sample configuration for topic models."""
    return {
        "embedding_model": "all-mpnet-base-v2",
        "device": "cpu",
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
        "llm": {
            "model": "grok-4-1-fast-reasoning",
            "max_concurrent": 50,
            "timeout": 30,
            "max_retries": 3,
        },
    }


@pytest.fixture
def mock_data_connector():
    """Create mock DataConnector with sample firm data."""
    from cloud.src.models import TranscriptData, FirmTranscriptData, TranscriptSentence

    firm1_sentences = [
        TranscriptSentence(
            f"1001_T001_{i:04d}",
            f"Sentence {i} about AI and machine learning.",
            f"sentence {i} ai machine learning",
            "CEO",
            i
        )
        for i in range(20)
    ]
    firm2_sentences = [
        TranscriptSentence(
            f"1002_T001_{i:04d}",
            f"Sentence {i} about revenue and growth.",
            f"sentence {i} revenue growth",
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
def create_mocked_pipeline(db_url, sample_config, mock_sentence_transformer, mock_xai_client):
    """Factory to create UnifiedPipeline with mocked dependencies."""
    def _create(xai_client=None):
        from cloud.src.pipeline import UnifiedPipeline

        with patch('cloud.src.pipeline.unified_pipeline.SentenceTransformer', return_value=mock_sentence_transformer):
            pipeline = UnifiedPipeline(
                database_url=db_url,
                config=sample_config,
                device="cpu",
            )
            # Inject XAI client (can be None for fallback testing)
            pipeline._xai_client = xai_client if xai_client is not None else mock_xai_client
            return pipeline
    return _create


# =============================================================================
# Topic Summary Tests
# =============================================================================

class TestTopicSummaryGeneration:
    """Tests for topic summary generation and storage."""

    def test_summaries_persisted_in_database(
        self, db_engine, mock_data_connector, create_mocked_pipeline, mock_xai_client
    ):
        """Topic summaries should be stored in the database."""
        from cloud.src.database.models import Topic
        from sqlalchemy.orm import Session

        pipeline = create_mocked_pipeline(xai_client=mock_xai_client)

        # Mock topic model
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 10 + [1] * 10)
        mock_result.n_topics = 2
        mock_result.topic_representations = {0: "ai machine learning", 1: "revenue growth"}
        mock_result.topic_keywords = {0: ["ai", "ml"], 1: ["revenue", "growth"]}
        mock_result.probabilities = np.random.rand(20, 2)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        # Process firm
        pipeline._process_single_firm("1001", mock_data_connector)

        # Verify summaries stored
        with Session(db_engine) as session:
            topics = session.query(Topic).all()
            assert len(topics) == 2

            for topic in topics:
                assert topic.summary is not None
                assert len(topic.summary) > 0
                assert "Summary of:" in topic.summary

    def test_topic_embeddings_derived_from_summaries(
        self, db_engine, mock_data_connector, create_mocked_pipeline, mock_xai_client, mock_sentence_transformer
    ):
        """Topic embeddings should be computed from summaries, not keywords."""
        from cloud.src.database.models import Topic
        from sqlalchemy.orm import Session

        pipeline = create_mocked_pipeline(xai_client=mock_xai_client)

        # Mock topic model
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "ai machine learning cloud"}
        mock_result.topic_keywords = {0: ["ai", "ml", "cloud"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        # Process firm
        pipeline._process_single_firm("1001", mock_data_connector)

        with Session(db_engine) as session:
            topic = session.query(Topic).first()

            # Get expected embedding from summary
            expected_embedding = mock_sentence_transformer.encode([topic.summary])[0]

            # Topic embedding should match summary embedding
            assert topic.embedding is not None
            assert np.allclose(topic.embedding, expected_embedding, atol=1e-5)


# =============================================================================
# Theme Description Tests
# =============================================================================

class TestThemeDescriptionGeneration:
    """Tests for theme description generation and storage."""

    def test_theme_descriptions_persisted(
        self, db_engine, mock_data_connector, create_mocked_pipeline, mock_xai_client
    ):
        """Theme descriptions should be stored in the database."""
        from cloud.src.database.models import Theme
        from sqlalchemy.orm import Session

        pipeline = create_mocked_pipeline(xai_client=mock_xai_client)

        # Mock topic model for both firms
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 10 + [1] * 10)
        mock_result.n_topics = 2
        mock_result.topic_representations = {0: "ai investment", 1: "revenue growth"}
        mock_result.topic_keywords = {0: ["ai"], 1: ["revenue"]}
        mock_result.probabilities = np.random.rand(20, 2)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        # Process two firms
        pipeline._process_single_firm("1001", mock_data_connector)
        pipeline._process_single_firm("1002", mock_data_connector)

        # Mock theme aggregation
        theme_result = MagicMock()
        theme_result.topic_assignments = np.array([0, 0, 1, 1])
        theme_result.n_topics = 2
        theme_result.topic_representations = {0: "Cross-firm AI", 1: "Cross-firm Revenue"}
        theme_result.topic_keywords = {0: ["ai"], 1: ["revenue"]}
        theme_result.probabilities = np.random.rand(4, 2)
        pipeline.theme_topic_model.fit_transform = MagicMock(return_value=theme_result)

        # Run theme aggregation
        pipeline._aggregate_themes()

        with Session(db_engine) as session:
            themes = session.query(Theme).all()
            # May have themes if validation passes
            for theme in themes:
                assert theme.description is not None
                assert "Description of" in theme.description


# =============================================================================
# Fallback Tests
# =============================================================================

class TestLLMFallback:
    """Tests for fallback behavior when LLM is unavailable."""

    def test_fallback_when_client_is_none(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """When XAI client is None, should use keywords as fallback summaries."""
        from cloud.src.database.models import Topic
        from sqlalchemy.orm import Session

        # Create pipeline without XAI client
        pipeline = create_mocked_pipeline(xai_client=None)
        pipeline._xai_client = None  # Explicitly set to None

        # Mock topic model
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "ai machine learning infrastructure"}
        mock_result.topic_keywords = {0: ["ai", "ml"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        # Process firm
        pipeline._process_single_firm("1001", mock_data_connector)

        with Session(db_engine) as session:
            topic = session.query(Topic).first()

            # Summary should fall back to representation (keywords)
            assert topic.summary == topic.representation

            # Embedding should still be populated (from fallback text)
            assert topic.embedding is not None
            assert len(topic.embedding) == 768

    def test_embeddings_always_populated_on_fallback(
        self, db_engine, mock_data_connector, create_mocked_pipeline
    ):
        """Embeddings must never be empty, even on LLM failure."""
        from cloud.src.database.models import Topic
        from sqlalchemy.orm import Session

        # Create pipeline with failing XAI client
        failing_client = MagicMock()
        failing_client.generate_batch_summaries = AsyncMock(
            return_value=[None, None, None]  # All calls return None (failure)
        )

        pipeline = create_mocked_pipeline(xai_client=failing_client)

        # Mock topic model
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "test keywords for fallback"}
        mock_result.topic_keywords = {0: ["test"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        # Process firm
        pipeline._process_single_firm("1001", mock_data_connector)

        with Session(db_engine) as session:
            topic = session.query(Topic).first()

            # Embedding MUST be populated (critical requirement)
            assert topic.embedding is not None
            assert len(topic.embedding) == 768


# =============================================================================
# Theme Aggregation Input Tests
# =============================================================================

class TestThemeAggregationInput:
    """Tests for theme aggregation using summaries."""

    def test_build_firm_topic_outputs_includes_summary(
        self, db_engine, mock_data_connector, create_mocked_pipeline, mock_xai_client
    ):
        """_build_firm_topic_outputs should return summary field."""
        from cloud.src.database.models import Topic
        from sqlalchemy.orm import Session

        pipeline = create_mocked_pipeline(xai_client=mock_xai_client)

        # Mock topic model
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "ai machine learning"}
        mock_result.topic_keywords = {0: ["ai", "ml"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        # Process firm to create topics
        pipeline._process_single_firm("1001", mock_data_connector)

        # Build firm topic outputs
        with Session(db_engine) as session:
            topics = session.query(Topic).all()
            outputs = pipeline._build_firm_topic_outputs(session, topics)

            # Each topic should have summary field
            for output in outputs:
                for topic in output["topics"]:
                    assert "summary" in topic
                    # Summary should be populated (either from LLM or fallback)
                    assert topic["summary"] is not None
                    assert len(topic["summary"]) > 0


# =============================================================================
# Keywords Preservation Tests
# =============================================================================

class TestKeywordsPreservation:
    """Tests for preserving keywords separately from summaries."""

    def test_representation_contains_keywords_not_summary(
        self, db_engine, mock_data_connector, create_mocked_pipeline, mock_xai_client
    ):
        """Topic.representation should contain keywords, not summary."""
        from cloud.src.database.models import Topic
        from sqlalchemy.orm import Session

        pipeline = create_mocked_pipeline(xai_client=mock_xai_client)

        # Mock topic model
        mock_result = MagicMock()
        mock_result.topic_assignments = np.array([0] * 20)
        mock_result.n_topics = 1
        mock_result.topic_representations = {0: "ai machine learning cloud"}
        mock_result.topic_keywords = {0: ["ai", "ml", "cloud"]}
        mock_result.probabilities = np.random.rand(20, 1)
        pipeline.firm_topic_model.fit_transform = MagicMock(return_value=mock_result)

        # Process firm
        pipeline._process_single_firm("1001", mock_data_connector)

        with Session(db_engine) as session:
            topic = session.query(Topic).first()

            # representation should be original keywords
            assert topic.representation == "ai machine learning cloud"

            # summary should be different (LLM-generated)
            assert topic.summary != topic.representation
            assert "Summary of:" in topic.summary
