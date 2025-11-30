"""
Abstract interfaces for the Financial Topic Modeling pipeline.

These interfaces enable:
    - TopicModel: Swappable topic model implementations (BERTopic, LDA, neural)
    - DataConnector: Swappable data sources (CSV, S3, WRDS)

Design Philosophy:
    - Clean contracts that hide implementation complexity
    - Dependency injection for testing and flexibility
    - The reduce phase RE-EMBEDS topic representations - we do NOT carry
      embeddings forward (that was an MVP artifact from similarity-based approach)
"""

from abc import ABC, abstractmethod
from typing import List

from cloud.src.models import TopicModelResult, TranscriptData


class TopicModel(ABC):
    """
    Abstract interface for topic models.

    Contract: Takes documents, returns topic assignments and representations.
    The reduce phase will RE-EMBED topic representations as new documents -
    we do NOT carry embeddings forward (that was an MVP artifact from the
    old similarity-based approach).

    Implementations:
        - BERTopicModel: Primary implementation using BERTopic
        - LDATopicModel: Stub for future LDA comparison
        - NeuralTopicModel: Stub for future neural topic models
    """

    @abstractmethod
    def fit_transform(self, documents: List[str]) -> TopicModelResult:
        """
        Fit the topic model and transform documents to topics.

        Args:
            documents: List of document texts (sentences in our pipeline)

        Returns:
            TopicModelResult with:
                - topic_assignments: Topic ID per document (-1 = outlier)
                - n_topics: Number of discovered topics
                - topic_representations: Human-readable topic descriptions
                - topic_keywords: Top keywords per topic
        """
        pass


class DataConnector(ABC):
    """
    Abstract interface for transcript data sources.

    Enables swapping: CSV (testing) -> S3 (cloud) -> WRDS (future)

    Implementations:
        - LocalCSVConnector: For local testing and development
        - S3TranscriptConnector: For cloud deployment
        - WRDSConnector: Future, when WRDS access is available
    """

    @abstractmethod
    def fetch_transcripts(
        self,
        firms: List[str],
        start_date: str,
        end_date: str,
    ) -> TranscriptData:
        """
        Fetch transcript sentences for specified firms and date range.

        Args:
            firms: List of company names (case-insensitive matching)
            start_date: YYYY-MM-DD format (inclusive)
            end_date: YYYY-MM-DD format (inclusive)

        Returns:
            TranscriptData with firms mapped to their sentences
        """
        pass

    @abstractmethod
    def get_available_firms(self) -> List[str]:
        """
        List all firms available in the data source.

        Returns:
            List of firm names available for querying
        """
        pass

    def close(self) -> None:
        """
        Clean up resources.

        Default implementation is a no-op. Override if your connector
        needs cleanup (e.g., database connections, file handles).
        """
        pass
