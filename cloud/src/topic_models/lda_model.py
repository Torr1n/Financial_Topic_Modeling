"""
LDATopicModel - Stub for LDA topic model comparison.

This stub demonstrates interface extensibility and provides a placeholder
for future LDA implementation to enable model comparison experiments.
"""

from typing import List

from cloud.src.interfaces import TopicModel
from cloud.src.models import TopicModelResult


class LDATopicModel(TopicModel):
    """
    LDA placeholder - demonstrates interface extensibility.

    This stub allows the codebase to reference LDA as an alternative
    topic model without requiring the full implementation. Useful for:
        - Faculty review showing swappable model design
        - Future comparison experiments between BERTopic and LDA
    """

    def __init__(self, config: dict = None):
        """
        Initialize LDA model stub.

        Args:
            config: Configuration dict (ignored in stub)
        """
        self.config = config or {}

    def fit_transform(self, documents: List[str]) -> TopicModelResult:
        """
        LDA fit_transform is not yet implemented.

        Raises:
            NotImplementedError: Always raised (this is a stub)
        """
        raise NotImplementedError(
            "LDA topic model not yet implemented. "
            "This stub exists to demonstrate interface extensibility."
        )
