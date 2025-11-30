"""
NeuralTopicModel - Stub for neural topic model comparison.

This stub demonstrates interface extensibility and provides a placeholder
for future neural topic model implementation (e.g., Neural Topic Model,
Embedded Topic Model, or other deep learning approaches).
"""

from typing import List

from cloud.src.interfaces import TopicModel
from cloud.src.models import TopicModelResult


class NeuralTopicModel(TopicModel):
    """
    Neural topic model placeholder - demonstrates interface extensibility.

    This stub allows the codebase to reference neural topic models as an
    alternative without requiring the full implementation. Useful for:
        - Faculty review showing swappable model design
        - Future experiments with neural topic modeling approaches
    """

    def __init__(self, config: dict = None):
        """
        Initialize Neural model stub.

        Args:
            config: Configuration dict (ignored in stub)
        """
        self.config = config or {}

    def fit_transform(self, documents: List[str]) -> TopicModelResult:
        """
        Neural topic model fit_transform is not yet implemented.

        Raises:
            NotImplementedError: Always raised (this is a stub)
        """
        raise NotImplementedError(
            "Neural topic model not yet implemented. "
            "This stub exists to demonstrate interface extensibility."
        )
