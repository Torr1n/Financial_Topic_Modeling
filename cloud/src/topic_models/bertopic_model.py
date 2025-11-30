"""
BERTopicModel - Primary topic model implementation.

This is a CLEAN reimplementation. The MVP code at Local_BERTopic_MVP is
for intent reference only - it contains bloated logic, dead code paths,
and hardcoded magic numbers we explicitly want to avoid.

Design:
    - Documents in -> Topic assignments + representations out
    - NO centroids (MVP artifact from old similarity-based approach)
    - Configuration-driven UMAP/HDBSCAN parameters
"""

import logging
from typing import Dict, List, Any

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from cloud.src.interfaces import TopicModel
from cloud.src.models import TopicModelResult

logger = logging.getLogger(__name__)


class BERTopicModel(TopicModel):
    """
    BERTopic implementation of the TopicModel interface.

    This implementation:
        1. Embeds documents using sentence-transformers
        2. Reduces dimensions with UMAP
        3. Clusters with HDBSCAN
        4. Extracts topic representations and keywords

    Args:
        config: Configuration dict with embedding_model, umap, hdbscan settings
    """

    # Default configuration values (from MVP config that worked)
    DEFAULT_CONFIG = {
        "embedding_model": "all-mpnet-base-v2",
        "device": "cpu",  # "cpu", "cuda", or "auto" - default to cpu for compatibility
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

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BERTopicModel with configuration.

        Args:
            config: Configuration dict. Missing keys use defaults.
        """
        self.config = self._merge_config(config)
        self.embedding_model_name = self.config.get(
            "embedding_model", self.DEFAULT_CONFIG["embedding_model"]
        )

        # Initialize components lazily (on first fit_transform call)
        self._embedding_model = None
        self._bertopic_model = None

        logger.info(f"Initialized BERTopicModel with embedding: {self.embedding_model_name}")

    def _merge_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge provided config with defaults."""
        merged = self.DEFAULT_CONFIG.copy()

        # Update top-level keys
        for key in ["embedding_model", "device"]:
            if key in config:
                merged[key] = config[key]

        # Deep merge nested dicts
        for key in ["umap", "hdbscan"]:
            if key in config:
                merged[key] = {**self.DEFAULT_CONFIG[key], **config[key]}

        return merged

    def _init_models(self) -> None:
        """Initialize embedding model and BERTopic (lazy initialization)."""
        if self._embedding_model is None:
            device = self.config.get("device", "cpu")
            logger.info(f"Loading embedding model: {self.embedding_model_name} on {device}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device=device)

        if self._bertopic_model is None:
            # Configure UMAP
            umap_config = self.config["umap"]
            umap_model = UMAP(
                n_neighbors=umap_config["n_neighbors"],
                n_components=umap_config["n_components"],
                min_dist=umap_config["min_dist"],
                metric=umap_config["metric"],
                random_state=umap_config.get("random_state", 42),
            )

            # Configure HDBSCAN
            hdbscan_config = self.config["hdbscan"]
            hdbscan_model = HDBSCAN(
                min_cluster_size=hdbscan_config["min_cluster_size"],
                min_samples=hdbscan_config["min_samples"],
                metric=hdbscan_config["metric"],
                cluster_selection_method=hdbscan_config.get("cluster_selection_method", "leaf"),
            )

            # Create BERTopic with our configured components
            self._bertopic_model = BERTopic(
                embedding_model=self._embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=False,
            )

            logger.info("Initialized BERTopic model")

    def fit_transform(self, documents: List[str]) -> TopicModelResult:
        """
        Fit the topic model and transform documents to topics.

        Args:
            documents: List of document texts (sentences)

        Returns:
            TopicModelResult with topic assignments, representations, and keywords

        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Cannot fit topic model on empty document list")

        # Initialize models if needed
        self._init_models()

        logger.info(f"Fitting BERTopic on {len(documents)} documents")

        # Fit and transform
        topics, probs = self._bertopic_model.fit_transform(documents)

        # Extract topic info
        topic_info = self._bertopic_model.get_topic_info()

        # Build representations and keywords
        topic_representations = {}
        topic_keywords = {}
        topic_sizes = {}

        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                continue  # Skip outlier topic

            # Get representation (topic name or first keywords)
            if "Name" in row and row["Name"]:
                representation = row["Name"]
            else:
                # Fall back to joining top keywords
                topic_words = self._bertopic_model.get_topic(topic_id)
                representation = ", ".join([word for word, _ in topic_words[:5]])

            topic_representations[topic_id] = representation

            # Get keywords
            topic_words = self._bertopic_model.get_topic(topic_id)
            topic_keywords[topic_id] = [word for word, _ in topic_words[:10]]

            # Get size
            topic_sizes[topic_id] = int(row["Count"])

        n_topics = len(topic_representations)

        logger.info(f"Discovered {n_topics} topics")

        return TopicModelResult(
            topic_assignments=np.array(topics),
            n_topics=n_topics,
            topic_representations=topic_representations,
            topic_keywords=topic_keywords,
            probabilities=probs if probs is not None else None,
            topic_sizes=topic_sizes,
            metadata={
                "model": "bertopic",
                "embedding_model": self.embedding_model_name,
                "umap_config": self.config["umap"],
                "hdbscan_config": self.config["hdbscan"],
            },
        )
