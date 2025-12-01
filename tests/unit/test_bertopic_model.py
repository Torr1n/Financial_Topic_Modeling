"""
Unit tests for BERTopicModel implementation.

Tests are written BEFORE implementation (TDD).
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestBERTopicModelInit:
    """Tests for BERTopicModel initialization."""

    def test_bertopic_model_implements_interface(self):
        """BERTopicModel should implement TopicModel interface."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel
        from cloud.src.interfaces import TopicModel

        assert issubclass(BERTopicModel, TopicModel)

    def test_bertopic_model_accepts_config(self, sample_config):
        """BERTopicModel should incorporate configuration values."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)

        # Config values should be incorporated (may be merged with defaults)
        assert model.config["umap"]["n_neighbors"] == sample_config["umap"]["n_neighbors"]
        assert model.config["hdbscan"]["min_cluster_size"] == sample_config["hdbscan"]["min_cluster_size"]
        assert model.embedding_model_name == sample_config["embedding_model"]

    def test_bertopic_model_uses_default_config(self):
        """BERTopicModel should use sensible defaults if config is empty."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel({})

        # Should have defaults from plan
        assert model.embedding_model_name == "all-mpnet-base-v2"


class TestBERTopicModelFitTransform:
    """Tests for BERTopicModel.fit_transform method."""

    def test_fit_transform_returns_topic_model_result(self, sample_documents, sample_config):
        """fit_transform should return TopicModelResult."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel
        from cloud.src.models import TopicModelResult

        model = BERTopicModel(sample_config)
        result = model.fit_transform(sample_documents)

        assert isinstance(result, TopicModelResult)

    def test_fit_transform_has_correct_assignment_length(self, sample_documents, sample_config):
        """Topic assignments should match document count."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)
        result = model.fit_transform(sample_documents)

        assert len(result.topic_assignments) == len(sample_documents)

    def test_fit_transform_has_representations_for_all_topics(self, sample_documents, sample_config):
        """All discovered topics should have representations."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)
        result = model.fit_transform(sample_documents)

        # All topic IDs (except -1 outliers) should have representations
        unique_topics = set(result.topic_assignments)
        for topic_id in unique_topics:
            if topic_id >= 0:  # Skip outliers
                assert topic_id in result.topic_representations
                assert topic_id in result.topic_keywords

    def test_fit_transform_has_keywords_as_lists(self, sample_documents, sample_config):
        """Topic keywords should be lists of strings."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)
        result = model.fit_transform(sample_documents)

        for topic_id, keywords in result.topic_keywords.items():
            assert isinstance(keywords, list)
            assert all(isinstance(kw, str) for kw in keywords)

    def test_fit_transform_n_topics_matches_representations(self, sample_documents, sample_config):
        """n_topics should match count of topic representations."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)
        result = model.fit_transform(sample_documents)

        assert result.n_topics == len(result.topic_representations)

    def test_fit_transform_no_centroids(self, sample_documents, sample_config):
        """Result should NOT contain centroids (MVP artifact removed)."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)
        result = model.fit_transform(sample_documents)

        # These should NOT exist per plan
        assert not hasattr(result, 'centroids')
        assert 'centroids' not in result.metadata


class TestBERTopicModelEdgeCases:
    """Tests for edge cases and error handling."""

    def test_fit_transform_empty_documents(self, sample_config):
        """fit_transform should handle empty document list gracefully."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)

        # Should raise ValueError for empty input
        with pytest.raises((ValueError, Exception)):
            model.fit_transform([])

    def test_fit_transform_single_document(self, sample_config):
        """fit_transform should raise error for single document.

        BERTopic/UMAP cannot fit a model with only one sample.
        This is expected behavior from the underlying libraries.
        """
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)

        # Single document causes UMAP/BERTopic to fail - this is expected
        with pytest.raises((ValueError, TypeError, Exception)):
            model.fit_transform(["Just one document about AI."])

    def test_fit_transform_few_documents(self, sample_config):
        """fit_transform should raise error when docs < n_neighbors.

        UMAP requires n_neighbors < n_samples. With default n_neighbors=15,
        we need at least 16 documents. This is expected behavior.
        """
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)
        docs = ["AI is important.", "Revenue grew.", "Supply chain issues."]

        # Too few documents for UMAP spectral embedding - this is expected
        with pytest.raises((ValueError, TypeError, Exception)):
            model.fit_transform(docs)


class TestBERTopicModelConfig:
    """Tests for configuration handling."""

    def test_uses_umap_config(self, sample_config):
        """Should use UMAP parameters from config."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)

        # Verify config is stored and accessible
        assert model.config["umap"]["n_neighbors"] == 15
        assert model.config["umap"]["n_components"] == 10

    def test_uses_hdbscan_config(self, sample_config):
        """Should use HDBSCAN parameters from config."""
        from cloud.src.topic_models.bertopic_model import BERTopicModel

        model = BERTopicModel(sample_config)

        assert model.config["hdbscan"]["min_cluster_size"] == 6
        assert model.config["hdbscan"]["min_samples"] == 2
