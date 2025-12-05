"""
Data models for the Financial Topic Modeling pipeline.

These dataclasses define the contract between pipeline components.
All schemas are locked per the approved plan - do not modify without approval.

Design Philosophy:
    - Documents in -> Topic assignments + representations out
    - NO centroids or hardcoded embedding dimensions (MVP artifacts removed)
    - Clean, simple, model-agnostic interfaces
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class TranscriptSentence:
    """
    Single sentence from an earnings call transcript.

    Attributes:
        sentence_id: Unique identifier (format: {firm_id}_{transcript_id}_{position:04d})
        raw_text: Original unprocessed sentence text (for observability)
        cleaned_text: Preprocessed text (lowercase, lemmatized, stopwords removed)
                     Used for embeddings and topic modeling
        speaker_type: Speaker role (CEO, CFO, Analyst, etc.) - optional
        position: Order within the transcript (0-indexed)
    """

    sentence_id: str
    raw_text: str
    cleaned_text: str
    speaker_type: Optional[str]
    position: int


@dataclass
class FirmTranscriptData:
    """
    All transcript sentences for a single firm.

    Attributes:
        firm_id: Company ID from source data (e.g., CSV companyid column)
        firm_name: Human-readable company name for display
        sentences: List of TranscriptSentence objects
        metadata: Additional info (date_range, transcript_count, etc.)
    """

    firm_id: str
    firm_name: str
    sentences: List[TranscriptSentence]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptData:
    """
    Complete dataset from DataConnector.

    Attributes:
        firms: Mapping of firm_id to FirmTranscriptData
    """

    firms: Dict[str, FirmTranscriptData]

    def get_firm_sentences(self, firm_id: str) -> List[str]:
        """
        Get cleaned sentence texts for a firm (for topic modeling).

        Args:
            firm_id: The firm ID to retrieve sentences for

        Returns:
            List of cleaned sentence text strings
        """
        return [s.cleaned_text for s in self.firms[firm_id].sentences]

    def get_all_firm_ids(self) -> List[str]:
        """
        Get all firm IDs in the dataset.

        Returns:
            List of firm ID strings
        """
        return list(self.firms.keys())


@dataclass
class TopicModelResult:
    """
    Standardized output for ALL topic model implementations.

    Design: Documents in -> Topic assignments + representations out.
    NO centroids or hardcoded embedding dimensions (those were MVP artifacts
    from the old similarity-based approach - we now use Dual-BERTopic which
    re-embeds topic representations in the reduce phase).

    Required Attributes:
        topic_assignments: Array of topic IDs per document (-1 = outlier)
        n_topics: Number of topics discovered (excluding outliers)
        topic_representations: Human-readable description per topic
        topic_keywords: Top keywords per topic
        probabilities: Topic probability distribution per document (n_docs, n_topics)
                       Required for downstream processing (sentence ordering, etc.)

    Optional Attributes:
        topic_sizes: Document count per topic
        metadata: Model-specific debugging/audit info
    """

    # Required: Every implementation must provide these
    topic_assignments: np.ndarray  # (n_docs,) - topic ID per document
    n_topics: int  # Number of topics discovered (excluding outliers)
    topic_representations: Dict[int, str]  # topic_id -> human-readable name
    topic_keywords: Dict[int, List[str]]  # topic_id -> top keywords
    probabilities: np.ndarray  # (n_docs, n_topics) - full topic distribution

    # Optional: Model-specific, may not be available for all implementations
    topic_sizes: Optional[Dict[int, int]] = None  # topic_id -> count

    # Metadata for debugging/audit
    metadata: Dict[str, Any] = field(default_factory=dict)
