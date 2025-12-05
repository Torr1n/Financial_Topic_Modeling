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

GPU Acceleration:
    When device="cuda", uses cuML's GPU-accelerated HDBSCAN and UMAP.
    This provides 10-100x speedup for dimensionality reduction and clustering.
    Requires: pip install cuml-cu11 (or cuml-cu12 depending on CUDA version)
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    PartOfSpeech,
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from cloud.src.interfaces import TopicModel
from cloud.src.models import TopicModelResult

logger = logging.getLogger(__name__)

# GPU acceleration: Import cuML if available
_CUML_AVAILABLE = False
try:
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.manifold import UMAP as cuUMAP
    from cuml.preprocessing import normalize as cu_normalize
    _CUML_AVAILABLE = True
    logger.info("cuML available - GPU acceleration enabled for UMAP/HDBSCAN")
except ImportError:
    logger.debug("cuML not available - using CPU implementations")

# CPU fallback imports
from umap import UMAP
from hdbscan import HDBSCAN


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
        # ⚠️ CLOUD DEPLOYMENT: Change to "cuda" for GPU acceleration (10x faster)
        # CPU default is for local testing compatibility only
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
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 2,
        },
        "representation": {
            "mmr_diversity": 0.3,
            "pos_model": "en_core_web_sm",
        },
    }

    def __init__(
        self,
        config: Dict[str, Any],
        embedding_model: Optional[SentenceTransformer] = None,
    ):
        """
        Initialize BERTopicModel with configuration.

        Args:
            config: Configuration dict. Missing keys use defaults.
            embedding_model: Optional pre-loaded SentenceTransformer. If provided,
                            this model is reused instead of loading a new one.
                            Use this when the unified pipeline loads the model once.
        """
        self.config = self._merge_config(config)
        self.embedding_model_name = self.config.get(
            "embedding_model", self.DEFAULT_CONFIG["embedding_model"]
        )

        # Use injected embedding model or initialize lazily
        self._embedding_model = embedding_model
        self._bertopic_model = None
        self._embeddings = None  # Store pre-computed embeddings

        if embedding_model:
            logger.info(f"Initialized BERTopicModel with injected embedding model")
        else:
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

            if device == "cpu":
                logger.warning(
                    "⚠️  BERTopicModel using CPU - set config['device']='cuda' for production"
                )

            self._embedding_model = SentenceTransformer(
                self.embedding_model_name, device=device
            )

        if self._bertopic_model is None:
            device = self.config.get("device", "cpu")
            use_gpu = device == "cuda" and _CUML_AVAILABLE

            if use_gpu:
                logger.info("Using cuML GPU-accelerated UMAP and HDBSCAN")
                umap_model, hdbscan_model = self._create_gpu_models()
            else:
                if device == "cuda" and not _CUML_AVAILABLE:
                    logger.warning(
                        "GPU requested but cuML not available - falling back to CPU. "
                        "Install cuML: pip install cuml-cu11 (or cuml-cu12)"
                    )
                umap_model, hdbscan_model = self._create_cpu_models()

            # Configure CountVectorizer (same for CPU/GPU)
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
            self._bertopic_model = BERTopic(
                embedding_model=self._embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_models,
                verbose=False,
            )

            logger.info(f"Initialized BERTopic model ({'GPU' if use_gpu else 'CPU'} acceleration)")

    def _create_cpu_models(self):
        """Create CPU-based UMAP and HDBSCAN models."""
        umap_config = self.config["umap"]
        umap_model = UMAP(
            n_neighbors=umap_config["n_neighbors"],
            n_components=umap_config["n_components"],
            min_dist=umap_config["min_dist"],
            metric=umap_config["metric"],
            random_state=umap_config.get("random_state", 42),
        )

        hdbscan_config = self.config["hdbscan"]
        hdbscan_model = HDBSCAN(
            min_cluster_size=hdbscan_config["min_cluster_size"],
            min_samples=hdbscan_config["min_samples"],
            metric=hdbscan_config["metric"],
            cluster_selection_method=hdbscan_config.get("cluster_selection_method", "leaf"),
        )

        return umap_model, hdbscan_model

    def _create_gpu_models(self):
        """Create cuML GPU-accelerated UMAP and HDBSCAN models."""
        umap_config = self.config["umap"]

        # cuML UMAP has slightly different parameter names
        # Note: cuML UMAP uses 'metric' but only supports 'euclidean' and 'cosine'
        umap_model = cuUMAP(
            n_neighbors=umap_config["n_neighbors"],
            n_components=umap_config["n_components"],
            min_dist=umap_config["min_dist"],
            metric=umap_config.get("metric", "cosine"),
            random_state=umap_config.get("random_state", 42),
        )

        hdbscan_config = self.config["hdbscan"]

        # cuML HDBSCAN parameters
        # Note: cuML HDBSCAN requires gen_min_span_tree=True for prediction_data
        hdbscan_model = cuHDBSCAN(
            min_cluster_size=hdbscan_config["min_cluster_size"],
            min_samples=hdbscan_config["min_samples"],
            metric=hdbscan_config.get("metric", "euclidean"),
            cluster_selection_method=hdbscan_config.get("cluster_selection_method", "leaf"),
            gen_min_span_tree=True,
            prediction_data=True,
        )

        return umap_model, hdbscan_model

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
            logger.warning(
                f"Failed to load POS model: {e}. Skipping POS representation."
            )

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

    def fit_transform(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> TopicModelResult:
        """
        Fit the topic model and transform documents to topics.

        Args:
            documents: List of document texts (sentences)
            embeddings: Optional pre-computed embeddings. If provided, skips
                       internal SentenceTransformer encoding. Shape must be
                       (len(documents), embedding_dim). Use this when the
                       unified pipeline computes embeddings externally for
                       efficiency (model loaded once, reused for all firms).

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

        # Use provided embeddings or compute them internally
        if embeddings is not None:
            logger.info("Using pre-computed embeddings")
            self._embeddings = embeddings
        else:
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

        # Reduce outliers with pre-computed embeddings
        topics = self._bertopic_model.reduce_outliers(
            documents, topics, strategy="embeddings", embeddings=self._embeddings
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

        # Guard: approximate_distribution can return all-zero rows when
        # clustering produces no assignment; ensure valid distributions
        if n_topics > 0:
            zero_rows = np.where(probabilities.sum(axis=1) == 0)[0]
            if len(zero_rows) > 0:
                logger.debug(
                    "Replacing %d zero-probability rows with uniform distribution",
                    len(zero_rows),
                )
                probabilities[zero_rows] = 1.0 / n_topics

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
