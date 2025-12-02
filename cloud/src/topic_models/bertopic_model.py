"""
BERTopicModel - Primary topic model implementation.

This is a CLEAN reimplementation. The MVP code at Local_BERTopic_MVP is
for intent reference only - it contains bloated logic, dead code paths,
and hardcoded magic numbers we explicitly want to avoid.

Design:
    - Documents in -> Topic assignments + representations out
    - NO centroids (MVP artifact from old similarity-based approach)
    - Configuration-driven UMAP/HDBSCAN parameters

Best Practices Applied:
    - Pre-computed embeddings for speed
    - Multiple representation models (KeyBERT, MMR, POS) for better labels
    - Explicit CountVectorizer for n-gram support
    - Full topic distribution via approximate_distribution

GPU Note:
    For GPU acceleration, use cuML's HDBSCAN and UMAP:
        from cuml.cluster import HDBSCAN
        from cuml.manifold import UMAP
    This is a future enhancement - not implemented yet.
"""

import logging
from typing import Dict, List, Any

import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

from cloud.src.interfaces import TopicModel
from cloud.src.models import TopicModelResult

logger = logging.getLogger(__name__)


class BERTopicModel(TopicModel):
    """
    BERTopic implementation of the TopicModel interface.

    This implementation:
        1. Pre-computes embeddings using sentence-transformers
        2. Reduces dimensions with UMAP
        3. Clusters with HDBSCAN
        4. Uses multiple representation models for better topic labels
        5. Computes full topic distribution for downstream use

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
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 2,
        },
        "representation": {
            "mmr_diversity": 0.3,
            "pos_model": "en_core_web_sm",
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
        self._embeddings = None  # Store pre-computed embeddings

        logger.info(
            f"Initialized BERTopicModel with embedding: {self.embedding_model_name}"
        )

    def _merge_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge provided config with defaults."""
        merged = self.DEFAULT_CONFIG.copy()

        # Update top-level keys
        for key in ["embedding_model", "device"]:
            if key in config:
                merged[key] = config[key]

        # Deep merge nested dicts
        for key in ["umap", "hdbscan", "vectorizer", "representation"]:
            if key in config:
                merged[key] = {**self.DEFAULT_CONFIG.get(key, {}), **config[key]}

        return merged

    def _init_models(self) -> None:
        """Initialize embedding model and BERTopic (lazy initialization)."""
        if self._embedding_model is None:
            device = self.config.get("device", "cpu")
            logger.info(
                f"Loading embedding model: {self.embedding_model_name} on {device}"
            )
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name, device=device
            )

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
                cluster_selection_method=hdbscan_config.get(
                    "cluster_selection_method", "leaf"
                ),
            )

            # Configure CountVectorizer
            vectorizer_config = self.config.get("vectorizer", {})
            ngram_range = tuple(vectorizer_config.get("ngram_range", [1, 2]))
            min_df = vectorizer_config.get("min_df", 2)
            vectorizer_model = CountVectorizer(
                ngram_range=ngram_range,
                min_df=min_df,
            )

            # Configure representation models for better topic labels
            representation_models = self._create_representation_models()

            # Create BERTopic with our configured components
            # Note: We still pass embedding_model because representation models
            # (like KeyBERTInspired) need it to embed representative documents
            self._bertopic_model = BERTopic(
                embedding_model=self._embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_models,
                verbose=False,
            )

            logger.info("Initialized BERTopic model with enhanced representations")

    def _create_representation_models(self) -> Dict[str, Any]:
        """
        Create multiple representation models for better topic labels.

        Returns dict mapping representation names to models, allowing
        BERTopic to generate multiple representations per topic.
        """
        rep_config = self.config.get("representation", {})

        models = {}

        # KeyBERTInspired - extracts keywords using embedding similarity
        models["KeyBERT"] = KeyBERTInspired()

        # Maximal Marginal Relevance - diverse keyword selection
        mmr_diversity = rep_config.get("mmr_diversity", 0.3)
        models["MMR"] = MaximalMarginalRelevance(diversity=mmr_diversity)

        # Part of Speech - filter to nouns/noun phrases for cleaner labels
        # Note: Requires spacy model - uses sentencizer as fallback if not available
        try:
            pos_model = rep_config.get("pos_model", "en_core_web_sm")
            models["POS"] = PartOfSpeech(pos_model)
        except Exception as e:
            logger.warning(f"Failed to load POS model: {e}. Skipping POS representation.")

        return models

    def _build_custom_label(self, topic_id: int) -> str:
        """
        Build custom topic label from multiple representations.

        Concatenates KeyBERT, MMR, and POS representations for a more
        complete topic description.

        Args:
            topic_id: The topic ID to build label for

        Returns:
            Combined representation string
        """
        topic_info = self._bertopic_model.get_topic_info()
        row = topic_info[topic_info["Topic"] == topic_id]

        if row.empty:
            return f"Topic {topic_id}"

        parts = []

        # Get representations from each model
        for col_name in ["KeyBERT", "MMR", "POS"]:
            if col_name in row.columns:
                rep = row[col_name].values[0]
                if rep and isinstance(rep, list) and len(rep) > 0:
                    # Take top 3 keywords from each representation
                    if isinstance(rep[0], tuple):
                        keywords = [r[0] for r in rep[:3]]
                    else:
                        keywords = rep[:3]
                    parts.extend(keywords)

        # Deduplicate while preserving order
        seen = set()
        unique_parts = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                unique_parts.append(p)

        if unique_parts:
            return ", ".join(unique_parts[:8])  # Limit to 8 unique terms

        # Fallback to default representation
        return row["Name"].values[0] if "Name" in row.columns else f"Topic {topic_id}"

    def fit_transform(self, documents: List[str]) -> TopicModelResult:
        """
        Fit the topic model and transform documents to topics.

        Args:
            documents: List of document texts (sentences)

        Returns:
            TopicModelResult with topic assignments, representations, keywords,
            and full probability distribution

        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Cannot fit topic model on empty document list")

        # Initialize models if needed
        self._init_models()

        logger.info(f"Fitting BERTopic on {len(documents)} documents")

        # Pre-compute embeddings (best practice for speed)
        logger.info("Pre-computing document embeddings...")
        self._embeddings = self._embedding_model.encode(
            documents,
            show_progress_bar=False,
        )

        # Fit and transform with pre-computed embeddings
        topics, probs = self._bertopic_model.fit_transform(
            documents,
            embeddings=self._embeddings,
        )

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

            # Build custom representation from multiple models
            representation = self._build_custom_label(topic_id)
            topic_representations[topic_id] = representation

            # Get keywords from default BERTopic representation
            topic_words = self._bertopic_model.get_topic(topic_id)
            topic_keywords[topic_id] = [word for word, _ in topic_words[:10]]

            # Get size
            topic_sizes[topic_id] = int(row["Count"])

        n_topics = len(topic_representations)

        logger.info(f"Discovered {n_topics} topics")

        # Compute full topic distribution using approximate_distribution
        # This gives us (n_docs, n_topics) matrix needed for sentence ordering
        logger.info("Computing full topic distribution...")
        topic_distr, _ = self._bertopic_model.approximate_distribution(
            documents,
            use_embedding_model=True,
        )

        # Normalize to get probabilities (rows sum to 1)
        row_sums = topic_distr.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        probabilities = topic_distr / row_sums

        logger.info(f"Topic distribution shape: {probabilities.shape}")

        return TopicModelResult(
            topic_assignments=np.array(topics),
            n_topics=n_topics,
            topic_representations=topic_representations,
            topic_keywords=topic_keywords,
            probabilities=probabilities,
            topic_sizes=topic_sizes,
            metadata={
                "model": "bertopic",
                "embedding_model": self.embedding_model_name,
                "umap_config": self.config["umap"],
                "hdbscan_config": self.config["hdbscan"],
                "vectorizer_config": self.config.get("vectorizer", {}),
                "embeddings_shape": self._embeddings.shape,
            },
        )
