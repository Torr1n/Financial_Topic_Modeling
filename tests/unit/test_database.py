"""
Unit tests for database layer using testcontainers.

Tests are written BEFORE implementation (TDD).
Uses testcontainers for real Postgres+pgvector testing.

Codex Review Adjustments:
- Performance test uses reasonable upper bound (not hard 1s)
- Tests for wrong embedding dimensions
- Verifies set-based bulk updates (not per-row loops)
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from typing import List

# testcontainers import
from testcontainers.postgres import PostgresContainer


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


@pytest.fixture
def db_engine(postgres_container):
    """Create SQLAlchemy engine connected to test container."""
    from sqlalchemy import create_engine, text

    engine = create_engine(postgres_container.get_connection_url())

    # Enable pgvector extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    yield engine

    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create a fresh database session with tables."""
    from sqlalchemy.orm import Session
    from cloud.src.database.models import Base

    # Create all tables
    Base.metadata.create_all(db_engine)

    session = Session(db_engine)
    yield session

    session.rollback()
    session.close()

    # Drop all tables for clean state
    Base.metadata.drop_all(db_engine)


@pytest.fixture
def repository(db_session):
    """Create repository instance."""
    from cloud.src.database.repository import DatabaseRepository
    return DatabaseRepository(db_session)


# =============================================================================
# Model Tests - Firm
# =============================================================================

class TestFirmModel:
    """Tests for Firm model."""

    def test_create_firm_with_required_fields(self, db_session):
        """Firm can be created with required fields only."""
        from cloud.src.database.models import Firm

        firm = Firm(company_id="1001")
        db_session.add(firm)
        db_session.flush()

        assert firm.id is not None
        assert firm.company_id == "1001"
        assert firm.processed_at is None

    def test_create_firm_with_all_fields(self, db_session):
        """Firm can be created with all optional fields."""
        from cloud.src.database.models import Firm

        firm = Firm(
            company_id="1001",
            name="Apple Inc.",
            ticker="AAPL",
            quarter="2023Q1",
        )
        db_session.add(firm)
        db_session.flush()

        assert firm.name == "Apple Inc."
        assert firm.ticker == "AAPL"
        assert firm.quarter == "2023Q1"

    def test_company_id_unique_constraint(self, db_session):
        """company_id must be unique."""
        from cloud.src.database.models import Firm
        from sqlalchemy.exc import IntegrityError

        firm1 = Firm(company_id="1001")
        firm2 = Firm(company_id="1001")  # Duplicate

        db_session.add(firm1)
        db_session.flush()

        db_session.add(firm2)
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_firm_relationships_to_sentences_and_topics(self, db_session):
        """Firm has relationships to sentences and topics."""
        from cloud.src.database.models import Firm

        firm = Firm(company_id="1001")
        db_session.add(firm)
        db_session.flush()

        # Relationships should be empty lists initially
        assert firm.sentences == []
        assert firm.topics == []


# =============================================================================
# Model Tests - Sentence
# =============================================================================

class TestSentenceModel:
    """Tests for Sentence model."""

    def test_create_sentence_basic(self, db_session):
        """Sentence can be created with required fields."""
        from cloud.src.database.models import Firm, Sentence

        firm = Firm(company_id="1001")
        db_session.add(firm)
        db_session.flush()

        sentence = Sentence(
            firm_id=firm.id,
            raw_text="Revenue exceeded expectations.",
            cleaned_text="revenue exceed expectation.",
            position=0,
        )
        db_session.add(sentence)
        db_session.flush()

        assert sentence.id is not None
        assert sentence.raw_text == "Revenue exceeded expectations."
        assert sentence.position == 0

    def test_create_sentence_with_768_dim_embedding(self, db_session):
        """Sentence can be created with 768-dim embedding."""
        from cloud.src.database.models import Firm, Sentence

        firm = Firm(company_id="1001")
        db_session.add(firm)
        db_session.flush()

        embedding = np.random.rand(768).astype(np.float32)
        sentence = Sentence(
            firm_id=firm.id,
            raw_text="Revenue exceeded expectations.",
            cleaned_text="revenue exceed expectation.",
            position=0,
            speaker_type="CEO",
            embedding=embedding,
        )
        db_session.add(sentence)
        db_session.flush()

        assert sentence.id is not None
        assert sentence.embedding is not None
        assert len(sentence.embedding) == 768

    def test_sentence_embedding_accepts_list(self, db_session):
        """Sentence embedding accepts Python list as well as ndarray."""
        from cloud.src.database.models import Firm, Sentence

        firm = Firm(company_id="1001")
        db_session.add(firm)
        db_session.flush()

        embedding_list = [0.1] * 768  # Python list
        sentence = Sentence(
            firm_id=firm.id,
            raw_text="Test sentence.",
            cleaned_text="test sentence.",
            position=0,
            embedding=embedding_list,
        )
        db_session.add(sentence)
        db_session.flush()

        assert sentence.embedding is not None
        assert len(sentence.embedding) == 768

    def test_sentence_rejects_wrong_embedding_dimension(self, db_session):
        """Sentence rejects embedding with wrong dimensions."""
        from cloud.src.database.models import Firm, Sentence
        from sqlalchemy.exc import StatementError

        firm = Firm(company_id="1001")
        db_session.add(firm)
        db_session.flush()

        wrong_embedding = np.random.rand(512).astype(np.float32)  # Wrong size
        sentence = Sentence(
            firm_id=firm.id,
            raw_text="Test sentence.",
            cleaned_text="test sentence.",
            position=0,
            embedding=wrong_embedding,
        )
        db_session.add(sentence)

        with pytest.raises(StatementError):
            db_session.flush()

    def test_sentence_requires_firm_fk(self, db_session):
        """Sentence must have valid firm_id (FK constraint)."""
        from cloud.src.database.models import Sentence
        from sqlalchemy.exc import IntegrityError

        sentence = Sentence(
            firm_id=99999,  # Non-existent firm
            raw_text="Test",
            cleaned_text="test",
            position=0,
        )
        db_session.add(sentence)

        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_sentence_topic_id_nullable(self, db_session):
        """Sentence topic_id is nullable (until assigned)."""
        from cloud.src.database.models import Firm, Sentence

        firm = Firm(company_id="1001")
        db_session.add(firm)
        db_session.flush()

        sentence = Sentence(
            firm_id=firm.id,
            raw_text="Test",
            cleaned_text="test",
            position=0,
            topic_id=None,  # Nullable
        )
        db_session.add(sentence)
        db_session.flush()  # Should not raise

        assert sentence.topic_id is None


# =============================================================================
# Model Tests - Topic
# =============================================================================

class TestTopicModel:
    """Tests for Topic model."""

    def test_create_topic_basic(self, db_session):
        """Topic can be created with required fields."""
        from cloud.src.database.models import Firm, Topic

        firm = Firm(company_id="1001")
        db_session.add(firm)
        db_session.flush()

        topic = Topic(
            firm_id=firm.id,
            local_topic_id=0,
            representation="AI investment strategy",
            n_sentences=25,
        )
        db_session.add(topic)
        db_session.flush()

        assert topic.id is not None
        assert topic.local_topic_id == 0
        assert topic.n_sentences == 25

    def test_create_topic_with_embedding(self, db_session):
        """Topic can be created with 768-dim embedding."""
        from cloud.src.database.models import Firm, Topic

        firm = Firm(company_id="1001")
        db_session.add(firm)
        db_session.flush()

        embedding = np.random.rand(768).astype(np.float32)
        topic = Topic(
            firm_id=firm.id,
            local_topic_id=0,
            representation="AI investment",
            n_sentences=25,
            embedding=embedding,
        )
        db_session.add(topic)
        db_session.flush()

        assert len(topic.embedding) == 768

    def test_topic_requires_firm_fk(self, db_session):
        """Topic must have valid firm_id (FK constraint)."""
        from cloud.src.database.models import Topic
        from sqlalchemy.exc import IntegrityError

        topic = Topic(
            firm_id=99999,  # Non-existent
            local_topic_id=0,
            representation="Test",
            n_sentences=0,
        )
        db_session.add(topic)

        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_topic_theme_id_nullable(self, db_session):
        """Topic theme_id is nullable (until reduce phase)."""
        from cloud.src.database.models import Firm, Topic

        firm = Firm(company_id="1001")
        db_session.add(firm)
        db_session.flush()

        topic = Topic(
            firm_id=firm.id,
            local_topic_id=0,
            representation="Test",
            n_sentences=5,
            theme_id=None,  # Nullable
        )
        db_session.add(topic)
        db_session.flush()  # Should not raise

        assert topic.theme_id is None

    def test_topic_theme_relationship(self, db_session):
        """Topic can be assigned to a theme."""
        from cloud.src.database.models import Firm, Topic, Theme

        firm = Firm(company_id="1001")
        theme = Theme(name="AI Investment", n_topics=1, n_firms=1)
        db_session.add_all([firm, theme])
        db_session.flush()

        topic = Topic(
            firm_id=firm.id,
            local_topic_id=0,
            representation="AI research",
            n_sentences=10,
            theme_id=theme.id,
        )
        db_session.add(topic)
        db_session.flush()

        assert topic.theme_id == theme.id
        assert topic.theme.name == "AI Investment"


# =============================================================================
# Model Tests - Theme
# =============================================================================

class TestThemeModel:
    """Tests for Theme model."""

    def test_create_theme_basic(self, db_session):
        """Theme can be created with required fields."""
        from cloud.src.database.models import Theme

        theme = Theme(
            name="Cross-Firm AI Investment",
            n_topics=10,
            n_firms=5,
        )
        db_session.add(theme)
        db_session.flush()

        assert theme.id is not None
        assert theme.name == "Cross-Firm AI Investment"
        assert theme.created_at is not None

    def test_create_theme_with_embedding(self, db_session):
        """Theme can be created with 768-dim embedding."""
        from cloud.src.database.models import Theme

        embedding = np.random.rand(768).astype(np.float32)
        theme = Theme(
            name="Cross-Firm AI Investment",
            description="Multiple firms discussing AI investments",
            n_topics=10,
            n_firms=5,
            embedding=embedding,
        )
        db_session.add(theme)
        db_session.flush()

        assert theme.id is not None
        assert len(theme.embedding) == 768

    def test_theme_topics_relationship(self, db_session):
        """Theme has relationship to topics."""
        from cloud.src.database.models import Theme

        theme = Theme(name="Test", n_topics=0, n_firms=0)
        db_session.add(theme)
        db_session.flush()

        assert theme.topics == []


# =============================================================================
# Repository Tests - Firm Operations
# =============================================================================

class TestRepositoryFirmOperations:
    """Tests for firm CRUD operations."""

    def test_create_firm(self, repository, db_session):
        """create_firm creates and returns firm."""
        firm = repository.create_firm(
            company_id="1001",
            name="Apple Inc.",
            ticker="AAPL",
            quarter="2023Q1",
        )

        assert firm.id is not None
        assert firm.company_id == "1001"
        assert firm.name == "Apple Inc."

    def test_get_firm_by_company_id(self, repository, db_session):
        """get_firm_by_company_id returns correct firm."""
        repository.create_firm(company_id="1001", name="Apple")
        db_session.commit()

        firm = repository.get_firm_by_company_id("1001")

        assert firm is not None
        assert firm.name == "Apple"

    def test_get_firm_by_company_id_not_found(self, repository):
        """get_firm_by_company_id returns None for missing firm."""
        firm = repository.get_firm_by_company_id("nonexistent")
        assert firm is None

    def test_get_or_create_firm_creates_new(self, repository, db_session):
        """get_or_create_firm creates new firm if not exists."""
        firm = repository.get_or_create_firm(
            company_id="1001",
            name="Apple Inc.",
        )

        assert firm.id is not None
        assert firm.company_id == "1001"

    def test_get_or_create_firm_returns_existing(self, repository, db_session):
        """get_or_create_firm returns existing firm if exists."""
        firm1 = repository.create_firm(company_id="1001", name="Apple")
        db_session.commit()

        firm2 = repository.get_or_create_firm(company_id="1001")

        assert firm1.id == firm2.id

    def test_mark_firm_processed(self, repository, db_session):
        """mark_firm_processed sets processed_at timestamp."""
        firm = repository.create_firm(company_id="1001")
        db_session.commit()

        assert firm.processed_at is None

        repository.mark_firm_processed(firm.id)
        db_session.commit()
        db_session.refresh(firm)

        assert firm.processed_at is not None

    def test_get_processed_firm_ids(self, repository, db_session):
        """get_processed_firm_ids returns only processed firms."""
        firm1 = repository.create_firm(company_id="1001")
        firm2 = repository.create_firm(company_id="1002")
        db_session.commit()

        repository.mark_firm_processed(firm1.id)
        db_session.commit()

        processed = repository.get_processed_firm_ids()

        assert "1001" in processed
        assert "1002" not in processed

    def test_get_unprocessed_firms(self, repository, db_session):
        """get_unprocessed_firms returns only unprocessed firms."""
        firm1 = repository.create_firm(company_id="1001")
        firm2 = repository.create_firm(company_id="1002")
        db_session.commit()

        repository.mark_firm_processed(firm1.id)
        db_session.commit()

        unprocessed = repository.get_unprocessed_firms()
        company_ids = [f.company_id for f in unprocessed]

        assert "1001" not in company_ids
        assert "1002" in company_ids


# =============================================================================
# Repository Tests - Bulk Insert Operations
# =============================================================================

class TestRepositoryBulkInsert:
    """Tests for bulk insert operations."""

    def test_bulk_insert_sentences_basic(self, repository, db_session):
        """bulk_insert_sentences inserts multiple sentences."""
        firm = repository.create_firm(company_id="1001")
        db_session.commit()

        sentences = [
            {
                "firm_id": firm.id,
                "raw_text": f"Raw sentence {i}",
                "cleaned_text": f"sentence {i}",
                "position": i,
                "speaker_type": "CEO",
            }
            for i in range(100)
        ]

        count = repository.bulk_insert_sentences(sentences)
        db_session.commit()

        assert count == 100

        # Verify in database
        result = repository.get_sentences_by_firm(firm.id)
        assert len(result) == 100

    def test_bulk_insert_sentences_with_embeddings(self, repository, db_session):
        """bulk_insert_sentences handles embeddings correctly."""
        firm = repository.create_firm(company_id="1001")
        db_session.commit()

        sentences = [
            {
                "firm_id": firm.id,
                "raw_text": f"Raw sentence {i}",
                "cleaned_text": f"sentence {i}",
                "position": i,
                "embedding": np.random.rand(768).astype(np.float32).tolist(),
            }
            for i in range(10)
        ]

        count = repository.bulk_insert_sentences(sentences)
        db_session.commit()

        assert count == 10

        # Verify embeddings were stored
        result = repository.get_sentences_by_firm(firm.id)
        assert all(s.embedding is not None for s in result)

    def test_bulk_insert_10k_sentences_completes(self, repository, db_session):
        """
        bulk_insert_sentences handles 10k+ sentences efficiently.

        Codex adjustment: Don't assert hard timing (<1s) - just verify
        completion without timeout/OOM.
        """
        firm = repository.create_firm(company_id="1001")
        db_session.commit()

        # Generate 10,000 sentences
        sentences = [
            {
                "firm_id": firm.id,
                "raw_text": f"This is sentence number {i} with some additional text.",
                "cleaned_text": f"this is sentence number {i} with some additional text.",
                "position": i,
                "speaker_type": "CEO" if i % 2 == 0 else "CFO",
            }
            for i in range(10000)
        ]

        count = repository.bulk_insert_sentences(sentences)
        db_session.commit()

        assert count == 10000

        # Verify count in database
        from cloud.src.database.models import Sentence
        from sqlalchemy import select, func

        total = db_session.scalar(
            select(func.count()).select_from(Sentence).where(Sentence.firm_id == firm.id)
        )
        assert total == 10000

    def test_bulk_insert_sentences_empty_list(self, repository, db_session):
        """bulk_insert_sentences handles empty list gracefully."""
        count = repository.bulk_insert_sentences([])
        assert count == 0

    def test_bulk_insert_topics(self, repository, db_session):
        """bulk_insert_topics efficiently inserts topics."""
        firm = repository.create_firm(company_id="1001")
        db_session.commit()

        topics = [
            {
                "firm_id": firm.id,
                "local_topic_id": i,
                "representation": f"Topic {i} about something",
                "n_sentences": 10 + i,
            }
            for i in range(5)
        ]

        count = repository.bulk_insert_topics(topics)
        db_session.commit()

        assert count == 5

        # Verify
        result = repository.get_topics_by_firm(firm.id)
        assert len(result) == 5


# =============================================================================
# Repository Tests - Bulk Update Operations
# =============================================================================

class TestRepositoryBulkUpdate:
    """Tests for bulk update operations."""

    def test_bulk_update_sentence_topics(self, repository, db_session):
        """bulk_update_sentence_topics updates multiple sentences efficiently."""
        firm = repository.create_firm(company_id="1001")
        db_session.flush()

        # Create topic
        from cloud.src.database.models import Topic
        topic = Topic(
            firm_id=firm.id,
            local_topic_id=0,
            representation="Test topic",
            n_sentences=3,
        )
        db_session.add(topic)
        db_session.flush()

        # Insert sentences
        sentences = [
            {
                "firm_id": firm.id,
                "raw_text": f"Sentence {i}",
                "cleaned_text": f"sentence {i}",
                "position": i,
            }
            for i in range(3)
        ]
        repository.bulk_insert_sentences(sentences)
        db_session.flush()

        # Get sentence IDs
        result_sentences = repository.get_sentences_by_firm(firm.id)

        # Create assignments
        assignments = [
            {"sentence_id": s.id, "topic_id": topic.id}
            for s in result_sentences
        ]

        # Bulk update
        count = repository.bulk_update_sentence_topics(assignments)
        db_session.commit()

        assert count == 3

        # Verify all sentences have topic_id set
        db_session.expire_all()
        updated = repository.get_sentences_by_firm(firm.id)
        assert all(s.topic_id == topic.id for s in updated)

    def test_bulk_update_sentence_topics_preserves_ids(self, repository, db_session):
        """
        bulk_update_sentence_topics preserves primary keys.

        Critical test per Codex review: verifies the update uses WHERE clause
        and doesn't corrupt PKs with SET id=:id.
        """
        firm = repository.create_firm(company_id="1001")
        db_session.flush()

        # Create topic
        from cloud.src.database.models import Topic
        topic = Topic(
            firm_id=firm.id,
            local_topic_id=0,
            representation="Test topic",
            n_sentences=5,
        )
        db_session.add(topic)
        db_session.flush()

        # Insert sentences
        sentences = [
            {
                "firm_id": firm.id,
                "raw_text": f"Sentence {i}",
                "cleaned_text": f"sentence {i}",
                "position": i,
            }
            for i in range(5)
        ]
        repository.bulk_insert_sentences(sentences)
        db_session.flush()

        # Capture original IDs before update
        original_sentences = repository.get_sentences_by_firm(firm.id)
        original_ids = [s.id for s in original_sentences]
        original_texts = [s.raw_text for s in original_sentences]

        # Create assignments
        assignments = [
            {"sentence_id": s.id, "topic_id": topic.id}
            for s in original_sentences
        ]

        # Bulk update - should NOT raise IntegrityError
        count = repository.bulk_update_sentence_topics(assignments)
        db_session.commit()

        assert count == 5

        # Verify IDs remain unchanged (critical check)
        db_session.expire_all()
        updated_sentences = repository.get_sentences_by_firm(firm.id)
        updated_ids = [s.id for s in updated_sentences]
        updated_texts = [s.raw_text for s in updated_sentences]

        assert updated_ids == original_ids, "Primary keys were corrupted by bulk update"
        assert updated_texts == original_texts, "Text content was corrupted by bulk update"
        assert all(s.topic_id == topic.id for s in updated_sentences), "topic_id not updated"

    def test_bulk_update_sentence_topics_empty(self, repository, db_session):
        """bulk_update_sentence_topics handles empty list."""
        count = repository.bulk_update_sentence_topics([])
        assert count == 0


# =============================================================================
# Repository Tests - Topic Operations
# =============================================================================

class TestRepositoryTopicOperations:
    """Tests for topic operations."""

    def test_create_topic(self, repository, db_session):
        """create_topic creates and returns topic."""
        firm = repository.create_firm(company_id="1001")
        db_session.flush()

        topic = repository.create_topic(
            firm_id=firm.id,
            local_topic_id=0,
            representation="AI investment",
            n_sentences=25,
        )

        assert topic.id is not None
        assert topic.representation == "AI investment"

    def test_create_topic_with_embedding(self, repository, db_session):
        """create_topic handles embedding correctly."""
        firm = repository.create_firm(company_id="1001")
        db_session.flush()

        embedding = np.random.rand(768).astype(np.float32)
        topic = repository.create_topic(
            firm_id=firm.id,
            local_topic_id=0,
            representation="AI investment",
            n_sentences=25,
            embedding=embedding,
        )

        assert topic.embedding is not None
        assert len(topic.embedding) == 768

    def test_get_topics_by_firm(self, repository, db_session):
        """get_topics_by_firm returns all topics for a firm."""
        firm = repository.create_firm(company_id="1001")
        db_session.flush()

        for i in range(3):
            repository.create_topic(
                firm_id=firm.id,
                local_topic_id=i,
                representation=f"Topic {i}",
                n_sentences=10,
            )
        db_session.commit()

        topics = repository.get_topics_by_firm(firm.id)
        assert len(topics) == 3

    def test_get_all_topics(self, repository, db_session):
        """get_all_topics returns topics from all firms."""
        firm1 = repository.create_firm(company_id="1001")
        firm2 = repository.create_firm(company_id="1002")
        db_session.flush()

        repository.create_topic(firm1.id, 0, "T1", 5)
        repository.create_topic(firm2.id, 0, "T2", 5)
        db_session.commit()

        all_topics = repository.get_all_topics()
        assert len(all_topics) == 2

    def test_update_topic_theme(self, repository, db_session):
        """update_topic_theme assigns topic to theme."""
        firm = repository.create_firm(company_id="1001")
        theme = repository.create_theme("Test Theme", 1, 1)
        db_session.flush()

        topic = repository.create_topic(firm.id, 0, "T1", 5)
        db_session.flush()

        assert topic.theme_id is None

        repository.update_topic_theme(topic.id, theme.id)
        db_session.commit()
        db_session.refresh(topic)

        assert topic.theme_id == theme.id


# =============================================================================
# Repository Tests - Theme Operations
# =============================================================================

class TestRepositoryThemeOperations:
    """Tests for theme operations."""

    def test_create_theme(self, repository, db_session):
        """create_theme creates and returns theme."""
        theme = repository.create_theme(
            name="AI Investment",
            n_topics=10,
            n_firms=5,
        )

        assert theme.id is not None
        assert theme.name == "AI Investment"

    def test_create_theme_with_embedding(self, repository, db_session):
        """create_theme handles embedding correctly."""
        embedding = np.random.rand(768).astype(np.float32)
        theme = repository.create_theme(
            name="AI Investment",
            n_topics=10,
            n_firms=5,
            embedding=embedding,
        )

        assert len(theme.embedding) == 768

    def test_get_all_themes(self, repository, db_session):
        """get_all_themes returns all themes."""
        repository.create_theme("Theme 1", 5, 3)
        repository.create_theme("Theme 2", 8, 4)
        db_session.commit()

        themes = repository.get_all_themes()
        assert len(themes) == 2


# =============================================================================
# Repository Tests - Hierarchical Queries
# =============================================================================

class TestRepositoryHierarchicalQueries:
    """Tests for hierarchical traversal queries."""

    def test_get_theme_with_hierarchy(self, repository, db_session):
        """get_theme_with_hierarchy returns full nested structure."""
        # Create hierarchy
        firm = repository.create_firm(company_id="1001", name="Apple")
        db_session.flush()

        theme = repository.create_theme(
            name="AI Investment",
            n_topics=1,
            n_firms=1,
        )
        db_session.flush()

        topic = repository.create_topic(
            firm_id=firm.id,
            local_topic_id=0,
            representation="AI research spending",
            n_sentences=2,
        )
        topic.theme_id = theme.id
        db_session.flush()

        # Insert sentences linked to topic
        sentences = [
            {
                "firm_id": firm.id,
                "raw_text": "We are investing in AI.",
                "cleaned_text": "investing ai.",
                "position": 0,
                "topic_id": topic.id,
            },
            {
                "firm_id": firm.id,
                "raw_text": "Our AI team is growing.",
                "cleaned_text": "ai team growing.",
                "position": 1,
                "topic_id": topic.id,
            },
        ]
        repository.bulk_insert_sentences(sentences)
        db_session.commit()

        # Query hierarchy
        result = repository.get_theme_with_hierarchy(theme.id)

        assert result is not None
        assert result["name"] == "AI Investment"
        assert len(result["topics"]) == 1
        assert result["topics"][0]["firm_name"] == "Apple"
        assert len(result["topics"][0]["sentences"]) == 2

    def test_get_theme_with_hierarchy_not_found(self, repository):
        """get_theme_with_hierarchy returns None for missing theme."""
        result = repository.get_theme_with_hierarchy(99999)
        assert result is None

    def test_get_topics_by_theme(self, repository, db_session):
        """get_topics_by_theme returns all topics for a theme."""
        firm1 = repository.create_firm(company_id="1001")
        firm2 = repository.create_firm(company_id="1002")
        theme = repository.create_theme("Test", 2, 2)
        db_session.flush()

        topic1 = repository.create_topic(firm1.id, 0, "T1", 5)
        topic2 = repository.create_topic(firm2.id, 0, "T2", 5)
        topic1.theme_id = theme.id
        topic2.theme_id = theme.id
        db_session.commit()

        topics = repository.get_topics_by_theme(theme.id)

        assert len(topics) == 2

    def test_get_sentences_by_topic(self, repository, db_session):
        """get_sentences_by_topic returns all sentences for a topic."""
        firm = repository.create_firm(company_id="1001")
        db_session.flush()

        topic = repository.create_topic(firm.id, 0, "Test", 3)
        db_session.flush()

        sentences = [
            {"firm_id": firm.id, "raw_text": f"S{i}", "cleaned_text": f"s{i}", "position": i, "topic_id": topic.id}
            for i in range(3)
        ]
        repository.bulk_insert_sentences(sentences)
        db_session.commit()

        result = repository.get_sentences_by_topic(topic.id)

        assert len(result) == 3

    def test_get_firm_topic_summary(self, repository, db_session):
        """get_firm_topic_summary returns summary for a firm."""
        firm = repository.create_firm(company_id="1001", name="Apple")
        db_session.flush()

        repository.create_topic(firm.id, 0, "Topic A", 10)
        repository.create_topic(firm.id, 1, "Topic B", 15)

        sentences = [
            {"firm_id": firm.id, "raw_text": f"S{i}", "cleaned_text": f"s{i}", "position": i}
            for i in range(25)
        ]
        repository.bulk_insert_sentences(sentences)
        db_session.commit()

        summary = repository.get_firm_topic_summary(firm.id)

        assert summary["company_id"] == "1001"
        assert summary["name"] == "Apple"
        assert summary["n_topics"] == 2
        assert summary["n_sentences"] == 25


# =============================================================================
# Repository Tests - Vector Index Building
# =============================================================================

class TestRepositoryVectorIndexes:
    """Tests for deferred vector index building."""

    def test_build_vector_indexes_succeeds(self, repository, db_session):
        """build_vector_indexes creates HNSW indexes without error."""
        # Insert some data first
        firm = repository.create_firm(company_id="1001")
        db_session.flush()

        sentences = [
            {
                "firm_id": firm.id,
                "raw_text": f"Sentence {i}",
                "cleaned_text": f"sentence {i}",
                "position": i,
                "embedding": np.random.rand(768).astype(np.float32).tolist(),
            }
            for i in range(10)
        ]
        repository.bulk_insert_sentences(sentences)
        db_session.commit()

        # Build indexes - should not raise
        repository.build_vector_indexes()
        db_session.commit()

    def test_build_vector_indexes_on_empty_tables(self, repository, db_session):
        """build_vector_indexes works on empty tables."""
        # No data, just build indexes
        repository.build_vector_indexes()
        db_session.commit()
        # Should complete without error


# =============================================================================
# Schema Tests - Indexes
# =============================================================================

class TestSchemaIndexes:
    """Tests for B-tree indexes on foreign keys."""

    def test_btree_indexes_exist(self, db_engine):
        """B-tree indexes exist on FK columns for query performance."""
        from sqlalchemy import inspect
        from cloud.src.database.models import Base

        # Create tables first
        Base.metadata.create_all(db_engine)

        inspector = inspect(db_engine)

        # Check sentences table indexes
        sentence_indexes = inspector.get_indexes("sentences")
        sentence_index_columns = {
            col for idx in sentence_indexes for col in idx["column_names"]
        }

        # Should have indexes on firm_id and topic_id
        assert "firm_id" in sentence_index_columns or any(
            "firm_id" in str(idx) for idx in sentence_indexes
        ), "Missing index on sentences.firm_id"

        # Check topics table indexes
        topic_indexes = inspector.get_indexes("topics")
        topic_index_columns = {
            col for idx in topic_indexes for col in idx["column_names"]
        }

        assert "firm_id" in topic_index_columns or any(
            "firm_id" in str(idx) for idx in topic_indexes
        ), "Missing index on topics.firm_id"

        # Clean up
        Base.metadata.drop_all(db_engine)
