"""
Repository pattern for database operations.

Provides:
- CRUD operations for each entity
- Bulk insert methods (optimized for 10k+ sentences)
- Hierarchical traversal queries
- Checkpoint queries for resume support
- Deferred index building

Design Notes:
- Uses dependency injection for Session - caller manages transaction
- Bulk operations use set-based SQL (not per-row loops) per Codex review
- All write operations require explicit commit() from caller
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Sequence, Union

import numpy as np
from sqlalchemy import select, insert, update, text
from sqlalchemy.orm import Session

from cloud.src.database.models import (
    Base,
    Firm,
    Sentence,
    Topic,
    Theme,
    DEFERRED_INDEXES,
)


class DatabaseRepository:
    """
    Repository for all database operations.

    Uses dependency injection for Session - caller manages transaction.

    Usage:
        with Session(engine) as session:
            repo = DatabaseRepository(session)
            repo.create_firm(...)
            session.commit()
    """

    def __init__(self, session: Session):
        """
        Initialize repository with SQLAlchemy session.

        Args:
            session: SQLAlchemy Session (caller manages lifecycle)
        """
        self.session = session

    # =========================================================================
    # Schema Management
    # =========================================================================

    def create_tables(self, engine) -> None:
        """
        Create all tables using SQLAlchemy metadata.

        Args:
            engine: SQLAlchemy Engine instance
        """
        Base.metadata.create_all(engine)

    def build_vector_indexes(self) -> None:
        """
        Build HNSW vector indexes (call AFTER bulk inserts).

        This is intentionally deferred because HNSW index building
        is much faster when data is already in the table.

        Note:
            Caller must commit() after this method to persist the indexes.
            Example:
                repo.build_vector_indexes()
                session.commit()
        """
        for index_sql in DEFERRED_INDEXES:
            self.session.execute(text(index_sql))

    # =========================================================================
    # Firm Operations
    # =========================================================================

    def create_firm(
        self,
        company_id: str,
        name: Optional[str] = None,
        ticker: Optional[str] = None,
        quarter: Optional[str] = None,
    ) -> Firm:
        """
        Create a new firm record.

        Args:
            company_id: External company ID (required, unique)
            name: Company display name
            ticker: Stock ticker symbol
            quarter: Processing period (e.g., "2023Q1")

        Returns:
            Created Firm instance with id populated
        """
        firm = Firm(
            company_id=company_id,
            name=name,
            ticker=ticker,
            quarter=quarter,
        )
        self.session.add(firm)
        self.session.flush()  # Get ID without committing
        return firm

    def get_firm_by_company_id(self, company_id: str) -> Optional[Firm]:
        """
        Get firm by external company_id.

        Args:
            company_id: External company ID to search for

        Returns:
            Firm instance or None if not found
        """
        return self.session.scalar(
            select(Firm).where(Firm.company_id == company_id)
        )

    def get_or_create_firm(
        self,
        company_id: str,
        name: Optional[str] = None,
        ticker: Optional[str] = None,
        quarter: Optional[str] = None,
    ) -> Firm:
        """
        Get existing firm or create new one.

        Args:
            company_id: External company ID
            name: Company display name (used only if creating)
            ticker: Stock ticker (used only if creating)
            quarter: Processing period (used only if creating)

        Returns:
            Existing or newly created Firm instance
        """
        firm = self.get_firm_by_company_id(company_id)
        if firm is None:
            firm = self.create_firm(company_id, name, ticker, quarter)
        return firm

    def mark_firm_processed(self, firm_id: int) -> None:
        """
        Mark firm as processed (checkpoint for resume).

        Args:
            firm_id: Internal firm ID
        """
        self.session.execute(
            update(Firm)
            .where(Firm.id == firm_id)
            .values(processed_at=datetime.now(timezone.utc))
        )

    def get_processed_firm_ids(self) -> List[str]:
        """
        Get list of company_ids that have been processed.

        Used for resume logic - skip already-processed firms.

        Returns:
            List of company_id strings
        """
        result = self.session.scalars(
            select(Firm.company_id).where(Firm.processed_at.isnot(None))
        )
        return list(result)

    def get_unprocessed_firms(self) -> List[Firm]:
        """
        Get firms that have not been processed yet.

        Filters on processed_at IS NULL per Codex review.

        Returns:
            List of Firm instances with processed_at = None
        """
        return list(
            self.session.scalars(
                select(Firm).where(Firm.processed_at.is_(None))
            )
        )

    # =========================================================================
    # Sentence Operations
    # =========================================================================

    def bulk_insert_sentences(
        self,
        sentences: List[Dict[str, Any]],
    ) -> int:
        """
        Bulk insert sentences using efficient INSERT...VALUES.

        Args:
            sentences: List of dicts with keys:
                - firm_id: int (required)
                - raw_text: str (required)
                - cleaned_text: str (required)
                - position: int (required)
                - speaker_type: str (optional)
                - topic_id: int (optional)
                - embedding: np.ndarray or list (optional)

        Returns:
            Number of sentences inserted
        """
        if not sentences:
            return 0

        self.session.execute(insert(Sentence), sentences)
        return len(sentences)

    def get_sentences_by_firm(self, firm_id: int) -> List[Sentence]:
        """
        Get all sentences for a firm.

        Args:
            firm_id: Internal firm ID

        Returns:
            List of Sentence instances ordered by position
        """
        return list(
            self.session.scalars(
                select(Sentence)
                .where(Sentence.firm_id == firm_id)
                .order_by(Sentence.position)
            )
        )

    def get_sentences_by_topic(self, topic_id: int) -> List[Sentence]:
        """
        Get all sentences belonging to a topic.

        Args:
            topic_id: Internal topic ID

        Returns:
            List of Sentence instances
        """
        return list(
            self.session.scalars(
                select(Sentence).where(Sentence.topic_id == topic_id)
            )
        )

    def bulk_update_sentence_topics(
        self,
        assignments: List[Dict[str, int]],
    ) -> int:
        """
        Bulk update sentence topic assignments.

        Uses SQLAlchemy's bulk_update_mappings for efficient updates.
        Only updates topic_id, leaving primary keys unchanged.

        Args:
            assignments: List of dicts with:
                - sentence_id: int
                - topic_id: int

        Returns:
            Number of sentences updated
        """
        if not assignments:
            return 0

        # Use ORM bulk_update_mappings - cleanest approach for bulk updates by PK
        # Transform 'sentence_id' (public API) to 'id' (PK column name)
        rows = [{"id": a["sentence_id"], "topic_id": a["topic_id"]} for a in assignments]
        self.session.bulk_update_mappings(Sentence, rows)
        return len(assignments)

    # =========================================================================
    # Topic Operations
    # =========================================================================

    def create_topic(
        self,
        firm_id: int,
        local_topic_id: int,
        representation: str,
        n_sentences: int,
        summary: Optional[str] = None,
        embedding: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> Topic:
        """
        Create a new topic record.

        Args:
            firm_id: Internal firm ID
            local_topic_id: BERTopic's topic number (0, 1, 2, ...)
            representation: Keywords/phrases describing topic
            n_sentences: Count of sentences in this topic
            summary: LLM-generated summary (optional)
            embedding: Vector (optional, dimension from EMBEDDING_DIMENSION env var)

        Returns:
            Created Topic instance
        """
        topic = Topic(
            firm_id=firm_id,
            local_topic_id=local_topic_id,
            representation=representation,
            n_sentences=n_sentences,
            summary=summary,
            embedding=embedding,
        )
        self.session.add(topic)
        self.session.flush()
        return topic

    def bulk_insert_topics(self, topics: List[Dict[str, Any]]) -> int:
        """
        Bulk insert topics.

        Args:
            topics: List of dicts with topic attributes

        Returns:
            Number of topics inserted
        """
        if not topics:
            return 0

        self.session.execute(insert(Topic), topics)
        return len(topics)

    def get_topics_by_firm(self, firm_id: int) -> List[Topic]:
        """
        Get all topics for a firm.

        Args:
            firm_id: Internal firm ID

        Returns:
            List of Topic instances ordered by local_topic_id
        """
        return list(
            self.session.scalars(
                select(Topic)
                .where(Topic.firm_id == firm_id)
                .order_by(Topic.local_topic_id)
            )
        )

    def get_topics_by_theme(self, theme_id: int) -> List[Topic]:
        """
        Get all topics belonging to a theme.

        Args:
            theme_id: Internal theme ID

        Returns:
            List of Topic instances
        """
        return list(
            self.session.scalars(select(Topic).where(Topic.theme_id == theme_id))
        )

    def get_all_topics(self) -> List[Topic]:
        """
        Get all topics (for theme aggregation).

        Returns:
            List of all Topic instances
        """
        return list(self.session.scalars(select(Topic)))

    def update_topic_theme(self, topic_id: int, theme_id: int) -> None:
        """
        Assign a topic to a theme.

        Args:
            topic_id: Internal topic ID
            theme_id: Internal theme ID
        """
        self.session.execute(
            update(Topic).where(Topic.id == topic_id).values(theme_id=theme_id)
        )

    # =========================================================================
    # Theme Operations
    # =========================================================================

    def create_theme(
        self,
        name: str,
        n_topics: int,
        n_firms: int,
        description: Optional[str] = None,
        embedding: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> Theme:
        """
        Create a new theme record.

        Args:
            name: Theme name
            n_topics: Count of topics in this theme
            n_firms: Count of distinct firms
            description: LLM-generated description (optional)
            embedding: Vector (optional, dimension from EMBEDDING_DIMENSION env var)

        Returns:
            Created Theme instance
        """
        theme = Theme(
            name=name,
            description=description,
            n_topics=n_topics,
            n_firms=n_firms,
            embedding=embedding,
        )
        self.session.add(theme)
        self.session.flush()
        return theme

    def get_all_themes(self) -> List[Theme]:
        """
        Get all themes.

        Returns:
            List of all Theme instances
        """
        return list(self.session.scalars(select(Theme)))

    # =========================================================================
    # Hierarchical Traversal Queries
    # =========================================================================

    def get_theme_with_hierarchy(self, theme_id: int) -> Optional[Dict[str, Any]]:
        """
        Get theme with full hierarchy: theme -> topics -> sentences -> firms.

        Args:
            theme_id: Internal theme ID

        Returns:
            Dict with theme data and nested topics/sentences/firms,
            or None if theme not found
        """
        theme = self.session.get(Theme, theme_id)
        if theme is None:
            return None

        result = {
            "id": theme.id,
            "name": theme.name,
            "description": theme.description,
            "n_topics": theme.n_topics,
            "n_firms": theme.n_firms,
            "topics": [],
        }

        for topic in theme.topics:
            topic_data = {
                "id": topic.id,
                "firm_id": topic.firm_id,
                "firm_name": topic.firm.name,
                "local_topic_id": topic.local_topic_id,
                "representation": topic.representation,
                "n_sentences": topic.n_sentences,
                "sentences": [
                    {
                        "id": s.id,
                        "raw_text": s.raw_text,
                        "cleaned_text": s.cleaned_text,
                        "position": s.position,
                        "speaker_type": s.speaker_type,
                    }
                    for s in topic.sentences
                ],
            }
            result["topics"].append(topic_data)

        return result

    def get_firm_topic_summary(self, firm_id: int) -> Dict[str, Any]:
        """
        Get summary of topics for a firm.

        Args:
            firm_id: Internal firm ID

        Returns:
            Dict with firm info and topic summaries,
            or empty dict if firm not found
        """
        firm = self.session.get(Firm, firm_id)
        if firm is None:
            return {}

        return {
            "firm_id": firm.id,
            "company_id": firm.company_id,
            "name": firm.name,
            "n_topics": len(firm.topics),
            "n_sentences": len(firm.sentences),
            "topics": [
                {
                    "local_topic_id": t.local_topic_id,
                    "representation": t.representation,
                    "n_sentences": t.n_sentences,
                    "theme_id": t.theme_id,
                }
                for t in firm.topics
            ],
        }
