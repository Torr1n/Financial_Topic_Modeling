"""
SQLAlchemy ORM models for PostgreSQL + pgvector storage.

Hierarchy: Theme -> Topics -> Sentences -> Firms

Design Notes:
- Uses SQLAlchemy 2.0 style with Mapped types
- pgvector Vector dimension is configurable via EMBEDDING_DIMENSION env var
- Default: 768 (all-mpnet-base-v2 output)
- For larger models (e.g., Qwen3-Embedding-8B): set EMBEDDING_DIMENSION=4096
- Foreign keys enforce hierarchy constraints
- B-tree indexes on FK columns for query performance
- HNSW vector indexes built AFTER bulk insert (deferred)
"""

import os
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Union

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ForeignKey,
    String,
    Text,
    Integer,
    BigInteger,
    DateTime,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# =============================================================================
# CONFIGURABLE EMBEDDING DIMENSION
# =============================================================================
# Set via environment variable before importing this module.
# Default: 768 (all-mpnet-base-v2)
# For Qwen3-Embedding-8B: set EMBEDDING_DIMENSION=4096
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "768"))

# Type alias for embeddings: accepts ndarray or list
EmbeddingType = Union[np.ndarray, Sequence[float], None]


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Firm(Base):
    """
    Firm entity - represents a company in the dataset.

    Attributes:
        id: Internal primary key (auto-increment)
        company_id: External ID from CSV (unique)
        ticker: Stock ticker symbol (optional)
        name: Company display name
        quarter: Processing period (e.g., "2023Q1")
        earnings_call_date: Date of earnings call (for event study linkage)
        processed_at: Timestamp when firm was processed (null = pending)
    """

    __tablename__ = "firms"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    ticker: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    quarter: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    earnings_call_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    sentences: Mapped[List["Sentence"]] = relationship(
        back_populates="firm", lazy="selectin"
    )
    topics: Mapped[List["Topic"]] = relationship(
        back_populates="firm", lazy="selectin"
    )


class Sentence(Base):
    """
    Sentence entity - individual sentence from earnings transcript.

    Attributes:
        id: Primary key (bigserial for large datasets)
        firm_id: FK to firms table
        raw_text: Original unprocessed sentence (for observability)
        cleaned_text: Preprocessed text used for embeddings/topic modeling
        position: Order within transcript
        speaker_type: CEO, CFO, Analyst, etc.
        topic_id: FK to topics (nullable until assigned)
        embedding: Vector from sentence-transformers (dimension configurable via EMBEDDING_DIMENSION)
    """

    __tablename__ = "sentences"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    firm_id: Mapped[int] = mapped_column(
        ForeignKey("firms.id"), nullable=False, index=True
    )
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    speaker_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    topic_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("topics.id"), nullable=True, index=True
    )
    embedding: Mapped[Optional[np.ndarray]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=True)

    # Relationships
    firm: Mapped["Firm"] = relationship(back_populates="sentences")
    topic: Mapped[Optional["Topic"]] = relationship(back_populates="sentences")
    entities: Mapped[List["ExtractedEntity"]] = relationship(
        back_populates="sentence", lazy="selectin", cascade="all, delete-orphan"
    )

    # B-tree indexes for FK columns (defined via mapped_column index=True above)


class Topic(Base):
    """
    Topic entity - firm-level topic cluster.

    Attributes:
        id: Primary key
        firm_id: FK to firms
        local_topic_id: BERTopic's topic number (0, 1, 2, ...)
        representation: Keywords/phrases describing topic
        summary: LLM-generated summary (nullable)
        n_sentences: Count of sentences in topic
        theme_id: FK to themes (nullable until reduce phase)
        embedding: Vector from topic summary (dimension configurable via EMBEDDING_DIMENSION)
    """

    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    firm_id: Mapped[int] = mapped_column(
        ForeignKey("firms.id"), nullable=False, index=True
    )
    local_topic_id: Mapped[int] = mapped_column(Integer, nullable=False)
    representation: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    n_sentences: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    theme_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("themes.id"), nullable=True, index=True
    )
    embedding: Mapped[Optional[np.ndarray]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=True)

    # Relationships
    firm: Mapped["Firm"] = relationship(back_populates="topics")
    theme: Mapped[Optional["Theme"]] = relationship(back_populates="topics")
    sentences: Mapped[List["Sentence"]] = relationship(
        back_populates="topic", lazy="selectin"
    )


class Theme(Base):
    """
    Theme entity - cross-firm topic cluster.

    Attributes:
        id: Primary key
        name: Theme name (from BERTopic or LLM)
        description: LLM-generated description
        n_topics: Count of topics in theme
        n_firms: Count of distinct firms
        embedding: Vector from theme description (dimension configurable via EMBEDDING_DIMENSION)
        created_at: Timestamp of theme creation
    """

    __tablename__ = "themes"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    n_topics: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_firms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[Optional[np.ndarray]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    topics: Mapped[List["Topic"]] = relationship(back_populates="theme", lazy="selectin")


class ExtractedEntity(Base):
    """
    Structured data entity extracted from a sentence by an LLM.

    Attributes:
        id: Primary key
        sentence_id: FK to sentences table
        entity_type: Type of entity (e.g., "PERSON", "COMPANY", "METRIC")
        value: The extracted text value
        confidence: Confidence score from the extraction model (optional)
    """
    __tablename__ = "extracted_entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    sentence_id: Mapped[int] = mapped_column(
        ForeignKey("sentences.id"), nullable=False, index=True
    )
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(nullable=True)

    # Relationship
    sentence: Mapped["Sentence"] = relationship(back_populates="entities")


# =============================================================================
# Deferred HNSW Vector Indexes
# =============================================================================
# These indexes are built AFTER bulk insert for performance.
# Building HNSW during inserts is catastrophically slow.
# Call repository.build_vector_indexes() after all data is loaded.

DEFERRED_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_sentences_embedding ON sentences USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS ix_topics_embedding ON topics USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS ix_themes_embedding ON themes USING hnsw (embedding vector_cosine_ops)",
]
