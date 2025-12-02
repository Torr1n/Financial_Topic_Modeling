"""
Unit tests for data models.

Tests are written BEFORE implementation (TDD).
These tests define the expected behavior of our data models.
"""

import pytest
import numpy as np
from typing import Dict, List


class TestTranscriptSentence:
    """Tests for TranscriptSentence dataclass."""

    def test_create_sentence_with_required_fields(self):
        """TranscriptSentence should be created with required fields."""
        from cloud.src.models import TranscriptSentence

        sentence = TranscriptSentence(
            sentence_id="AAPL_T001_0001",
            text="We are investing in AI.",
            speaker_type="CEO",
            position=0,
        )

        assert sentence.sentence_id == "AAPL_T001_0001"
        assert sentence.text == "We are investing in AI."
        assert sentence.speaker_type == "CEO"
        assert sentence.position == 0

    def test_speaker_type_is_optional(self):
        """TranscriptSentence should allow None speaker_type."""
        from cloud.src.models import TranscriptSentence

        sentence = TranscriptSentence(
            sentence_id="AAPL_T001_0001",
            text="Some text.",
            speaker_type=None,
            position=0,
        )

        assert sentence.speaker_type is None


class TestFirmTranscriptData:
    """Tests for FirmTranscriptData dataclass."""

    def test_create_firm_data(self):
        """FirmTranscriptData should be created with firm info and sentences."""
        from cloud.src.models import TranscriptSentence, FirmTranscriptData

        sentences = [
            TranscriptSentence("AAPL_T001_0001", "Sentence one.", "CEO", 0),
            TranscriptSentence("AAPL_T001_0002", "Sentence two.", "CFO", 1),
        ]

        firm_data = FirmTranscriptData(
            firm_id="1001",
            firm_name="Apple Inc.",
            sentences=sentences,
            metadata={"date_range": "2023-Q1"},
        )

        assert firm_data.firm_id == "1001"
        assert firm_data.firm_name == "Apple Inc."
        assert len(firm_data.sentences) == 2
        assert firm_data.metadata["date_range"] == "2023-Q1"

    def test_firm_data_default_metadata(self):
        """FirmTranscriptData should default metadata to empty dict."""
        from cloud.src.models import TranscriptSentence, FirmTranscriptData

        firm_data = FirmTranscriptData(
            firm_id="1001",
            firm_name="Apple Inc.",
            sentences=[],
        )

        assert firm_data.metadata == {}


class TestTranscriptData:
    """Tests for TranscriptData dataclass."""

    def test_create_transcript_data(self):
        """TranscriptData should store firms dict."""
        from cloud.src.models import TranscriptSentence, FirmTranscriptData, TranscriptData

        sentences = [TranscriptSentence("AAPL_T001_0001", "Test.", "CEO", 0)]
        firm = FirmTranscriptData("1001", "Apple Inc.", sentences)

        data = TranscriptData(firms={"1001": firm})

        assert "1001" in data.firms
        assert data.firms["1001"].firm_name == "Apple Inc."

    def test_get_firm_sentences(self):
        """TranscriptData.get_firm_sentences should return list of texts."""
        from cloud.src.models import TranscriptSentence, FirmTranscriptData, TranscriptData

        sentences = [
            TranscriptSentence("AAPL_T001_0001", "First sentence.", "CEO", 0),
            TranscriptSentence("AAPL_T001_0002", "Second sentence.", "CFO", 1),
        ]
        firm = FirmTranscriptData("1001", "Apple Inc.", sentences)
        data = TranscriptData(firms={"1001": firm})

        texts = data.get_firm_sentences("1001")

        assert texts == ["First sentence.", "Second sentence."]

    def test_get_all_firm_ids(self):
        """TranscriptData.get_all_firm_ids should return list of firm IDs."""
        from cloud.src.models import FirmTranscriptData, TranscriptData

        firm1 = FirmTranscriptData("1001", "Apple Inc.", [])
        firm2 = FirmTranscriptData("1002", "Microsoft Corp.", [])
        data = TranscriptData(firms={"1001": firm1, "1002": firm2})

        firm_ids = data.get_all_firm_ids()

        assert set(firm_ids) == {"1001", "1002"}


class TestTopicModelResult:
    """Tests for TopicModelResult dataclass."""

    def test_create_topic_model_result_with_required_fields(self):
        """TopicModelResult should be created with required fields including probabilities."""
        from cloud.src.models import TopicModelResult

        # probabilities is now required - (n_docs, n_topics) matrix
        probabilities = np.array([
            [0.8, 0.2],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],  # Outlier still has distribution
        ])

        result = TopicModelResult(
            topic_assignments=np.array([0, 0, 1, 1, -1]),
            n_topics=2,
            topic_representations={0: "AI Investment", 1: "Revenue Growth"},
            topic_keywords={0: ["ai", "ml"], 1: ["revenue", "growth"]},
            probabilities=probabilities,
        )

        assert len(result.topic_assignments) == 5
        assert result.n_topics == 2
        assert result.topic_representations[0] == "AI Investment"
        assert result.topic_keywords[1] == ["revenue", "growth"]
        assert result.probabilities is not None
        assert result.probabilities.shape == (5, 2)

    def test_topic_model_result_optional_fields(self):
        """TopicModelResult should have topic_sizes and metadata optional."""
        from cloud.src.models import TopicModelResult

        # probabilities is required, but topic_sizes/metadata are optional
        result = TopicModelResult(
            topic_assignments=np.array([0, 1]),
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["a"], 1: ["b"]},
            probabilities=np.array([[0.8, 0.2], [0.2, 0.8]]),
        )

        # topic_sizes and metadata are still optional
        assert result.topic_sizes is None
        assert result.metadata == {}

    def test_topic_model_result_with_all_fields(self):
        """TopicModelResult should accept all fields."""
        from cloud.src.models import TopicModelResult

        result = TopicModelResult(
            topic_assignments=np.array([0, 1, 0]),
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["a"], 1: ["b"]},
            probabilities=np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]),
            topic_sizes={0: 2, 1: 1},
            metadata={"model": "bertopic"},
        )

        assert result.probabilities is not None
        assert result.probabilities.shape == (3, 2)
        assert result.topic_sizes[0] == 2
        assert result.metadata["model"] == "bertopic"

    def test_probabilities_shape_correct(self):
        """Probabilities should be (n_docs, n_topics) matrix."""
        from cloud.src.models import TopicModelResult

        n_docs = 5
        n_topics = 3
        probabilities = np.random.rand(n_docs, n_topics)
        # Normalize rows to sum to 1
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        result = TopicModelResult(
            topic_assignments=np.array([0, 1, 2, 0, 1]),
            n_topics=n_topics,
            topic_representations={0: "A", 1: "B", 2: "C"},
            topic_keywords={0: ["a"], 1: ["b"], 2: ["c"]},
            probabilities=probabilities,
        )

        assert result.probabilities.shape == (n_docs, n_topics)
        # Each row should sum to ~1
        row_sums = result.probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_no_centroids_or_embeddings(self):
        """TopicModelResult should NOT have centroids or embeddings (MVP artifact removed)."""
        from cloud.src.models import TopicModelResult

        result = TopicModelResult(
            topic_assignments=np.array([0]),
            n_topics=1,
            topic_representations={0: "Test"},
            topic_keywords={0: ["test"]},
            probabilities=np.array([[1.0]]),
        )

        # These should NOT exist per plan - centroids were an MVP artifact
        assert not hasattr(result, 'centroids')
        assert not hasattr(result, 'embeddings')
        assert not hasattr(result, 'topic_embeddings')
