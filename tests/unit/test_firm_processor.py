"""
Unit tests for FirmProcessor.

Tests are written BEFORE implementation (TDD).
FirmProcessor converts firm transcript data into FirmTopicOutput format.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from datetime import datetime


class TestFirmProcessorInit:
    """Tests for FirmProcessor initialization."""

    def test_accepts_topic_model_and_config(self, mock_topic_model, sample_config):
        """FirmProcessor should accept topic model and config."""
        from cloud.src.firm_processor import FirmProcessor

        processor = FirmProcessor(mock_topic_model, sample_config)

        assert processor.model == mock_topic_model
        assert processor.config == sample_config

    def test_dependency_injection(self, sample_config):
        """Should use injected topic model (not create its own)."""
        from cloud.src.firm_processor import FirmProcessor

        mock_model = MagicMock()
        processor = FirmProcessor(mock_model, sample_config)

        assert processor.model is mock_model


class TestFirmProcessorProcess:
    """Tests for FirmProcessor.process method."""

    def test_process_returns_tuple(self, sample_config):
        """process() should return a tuple of (dict, topic_assignments)."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        # Create mock with 2 docs to match input
        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 1]),
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["ai"], 1: ["revenue"]},
            probabilities=np.array([[0.8, 0.2], [0.3, 0.7]]),
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "AI investment.", "AI investment.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Revenue grew.", "Revenue grew.", "CFO", 1),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        assert isinstance(result, tuple)
        output, topic_assignments = result
        assert isinstance(output, dict)
        assert isinstance(topic_assignments, np.ndarray)

    def test_process_calls_model_fit_transform(self, sample_config):
        """process() should call model.fit_transform with sentence texts."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        # Create mock with 2 docs to match input
        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 1]),
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["ai"], 1: ["revenue"]},
            probabilities=np.array([[0.8, 0.2], [0.3, 0.7]]),
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "AI investment.", "AI investment.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Revenue grew.", "Revenue grew.", "CFO", 1),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        processor.process(firm_data)

        mock_model.fit_transform.assert_called_once()
        call_args = mock_model.fit_transform.call_args[0][0]
        assert call_args == ["AI investment.", "Revenue grew."]


class TestFirmTopicOutputSchema:
    """Tests for FirmTopicOutput schema compliance."""

    def test_output_has_required_fields(self, mock_topic_model, sample_config):
        """Output should have all required FirmTopicOutput fields."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        # Required fields per plan
        assert "firm_id" in output
        assert "firm_name" in output
        assert "n_topics" in output
        assert "topics" in output
        assert "outlier_sentence_ids" in output
        assert "metadata" in output

    def test_output_firm_id_and_name(self, mock_topic_model, sample_config):
        """Output should preserve firm_id and firm_name."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        assert output["firm_id"] == "1001"
        assert output["firm_name"] == "Apple Inc."

    def test_topic_structure(self, mock_topic_model, sample_config):
        """Each topic should have required fields."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        for topic in output["topics"]:
            assert "topic_id" in topic
            assert "representation" in topic
            assert "keywords" in topic
            assert "size" in topic
            assert "sentence_ids" in topic
            assert isinstance(topic["keywords"], list)
            assert isinstance(topic["sentence_ids"], list)

    def test_n_topics_matches_topic_count(self, mock_topic_model, sample_config):
        """n_topics should match number of topics in list."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        assert output["n_topics"] == len(output["topics"])

    def test_metadata_has_required_fields(self, mock_topic_model, sample_config):
        """Metadata should have processing_timestamp, model_config, n_sentences_processed."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        assert "processing_timestamp" in output["metadata"]
        assert "model_config" in output["metadata"]
        assert "n_sentences_processed" in output["metadata"]
        assert output["metadata"]["n_sentences_processed"] == 10


class TestFirmProcessorSentenceIdMapping:
    """Tests for correct sentence_id assignment to topics."""

    def test_sentence_ids_mapped_to_correct_topics(self, sample_config):
        """Sentence IDs should be correctly mapped to their topics."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        # Create a mock model with known assignments and probabilities
        probabilities = np.array([
            [0.8, 0.2],  # doc 0: topic 0
            [0.7, 0.3],  # doc 1: topic 0
            [0.2, 0.8],  # doc 2: topic 1
            [0.3, 0.7],  # doc 3: topic 1
            [0.5, 0.5],  # doc 4: outlier
        ])

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 1, 1, -1]),  # 2 in topic 0, 2 in topic 1, 1 outlier
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["a", "b"], 1: ["c", "d"]},
            probabilities=probabilities,
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "Sentence 0.", "Sentence 0.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Sentence 1.", "Sentence 1.", "CEO", 1),
            TranscriptSentence("1001_T001_0002", "Sentence 2.", "Sentence 2.", "CFO", 2),
            TranscriptSentence("1001_T001_0003", "Sentence 3.", "Sentence 3.", "CFO", 3),
            TranscriptSentence("1001_T001_0004", "Sentence 4.", "Sentence 4.", "COO", 4),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        # Topic 0 should have sentences 0 and 1 (ordered by probability: 0.8 > 0.7)
        topic_0 = next(t for t in output["topics"] if t["topic_id"] == 0)
        assert topic_0["sentence_ids"] == ["1001_T001_0000", "1001_T001_0001"]
        assert topic_0["size"] == 2

        # Topic 1 should have sentences 2 and 3 (ordered by probability: 0.8 > 0.7)
        topic_1 = next(t for t in output["topics"] if t["topic_id"] == 1)
        assert topic_1["sentence_ids"] == ["1001_T001_0002", "1001_T001_0003"]
        assert topic_1["size"] == 2

        # Outlier should be sentence 4
        assert output["outlier_sentence_ids"] == ["1001_T001_0004"]


class TestFirmProcessorOutlierHandling:
    """Tests for outlier (-1 topic) handling."""

    def test_outliers_tracked_separately(self, sample_config):
        """Outliers (topic_id=-1) should be in outlier_sentence_ids."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        # Probabilities for 4 docs, 1 topic
        probabilities = np.array([
            [0.3],  # doc 0: outlier
            [0.4],  # doc 1: outlier
            [0.9],  # doc 2: topic 0
            [0.2],  # doc 3: outlier
        ])

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([-1, -1, 0, -1]),  # 3 outliers, 1 in topic 0
            n_topics=1,
            topic_representations={0: "Single Topic"},
            topic_keywords={0: ["keyword"]},
            probabilities=probabilities,
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", f"Sentence {i}.", "CEO", i) for i in range(4)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        assert len(output["outlier_sentence_ids"]) == 3
        assert "1001_T001_0000" in output["outlier_sentence_ids"]
        assert "1001_T001_0001" in output["outlier_sentence_ids"]
        assert "1001_T001_0003" in output["outlier_sentence_ids"]

    def test_all_outliers_gives_empty_topics(self, sample_config):
        """If all sentences are outliers, topics list should be empty."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        # All outliers - still need probabilities (empty n_topics dimension)
        # When n_topics=0, probabilities can be (n_docs, 0) shaped
        probabilities = np.zeros((3, 0))

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([-1, -1, -1]),  # All outliers
            n_topics=0,
            topic_representations={},
            topic_keywords={},
            probabilities=probabilities,
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", f"Sentence {i}.", "CEO", i) for i in range(3)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        assert output["n_topics"] == 0
        assert output["topics"] == []
        assert len(output["outlier_sentence_ids"]) == 3


class TestFirmProcessorJsonSerializable:
    """Tests for JSON serializability of output."""

    def test_output_is_json_serializable(self, mock_topic_model, sample_config):
        """Output should be JSON serializable."""
        import json
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        # Should not raise
        json_str = json.dumps(output, default=str)
        assert isinstance(json_str, str)


class TestFirmProcessorSentenceOrdering:
    """Tests for sentence ordering by probability."""

    def test_sentence_ids_ordered_by_probability(self, sample_config):
        """Sentence IDs within each topic should be ordered by probability (highest first)."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        # Create a mock model with known assignments and probabilities
        # 5 docs, 2 topics
        probabilities = np.array([
            [0.8, 0.2],  # doc 0: topic 0, prob 0.8
            [0.6, 0.4],  # doc 1: topic 0, prob 0.6 (lower than doc 0)
            [0.9, 0.1],  # doc 2: topic 0, prob 0.9 (highest)
            [0.3, 0.7],  # doc 3: topic 1, prob 0.7
            [0.2, 0.8],  # doc 4: topic 1, prob 0.8 (higher than doc 3)
        ])

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0, 1, 1]),
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["a", "b"], 1: ["c", "d"]},
            probabilities=probabilities,
            topic_sizes={0: 3, 1: 2},
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "Sentence 0.", "Sentence 0.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Sentence 1.", "Sentence 1.", "CEO", 1),
            TranscriptSentence("1001_T001_0002", "Sentence 2.", "Sentence 2.", "CFO", 2),
            TranscriptSentence("1001_T001_0003", "Sentence 3.", "Sentence 3.", "CFO", 3),
            TranscriptSentence("1001_T001_0004", "Sentence 4.", "Sentence 4.", "COO", 4),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        # Topic 0: doc 2 (0.9) > doc 0 (0.8) > doc 1 (0.6)
        topic_0 = next(t for t in output["topics"] if t["topic_id"] == 0)
        assert topic_0["sentence_ids"] == ["1001_T001_0002", "1001_T001_0000", "1001_T001_0001"]

        # Topic 1: doc 4 (0.8) > doc 3 (0.7)
        topic_1 = next(t for t in output["topics"] if t["topic_id"] == 1)
        assert topic_1["sentence_ids"] == ["1001_T001_0004", "1001_T001_0003"]

    def test_sentence_ordering_handles_equal_probabilities(self, sample_config):
        """When probabilities are equal, maintain stable ordering."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        # Equal probabilities
        probabilities = np.array([
            [0.8, 0.2],
            [0.8, 0.2],
            [0.8, 0.2],
        ])

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0]),
            n_topics=1,
            topic_representations={0: "Topic A"},
            topic_keywords={0: ["a"]},
            probabilities=probabilities,
            topic_sizes={0: 3},
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "Sentence 0.", "Sentence 0.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Sentence 1.", "Sentence 1.", "CEO", 1),
            TranscriptSentence("1001_T001_0002", "Sentence 2.", "Sentence 2.", "CEO", 2),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        # With equal probabilities, should have some consistent ordering
        topic_0 = output["topics"][0]
        assert len(topic_0["sentence_ids"]) == 3
        # All three sentences should be present
        assert set(topic_0["sentence_ids"]) == {"1001_T001_0000", "1001_T001_0001", "1001_T001_0002"}


class TestFirmProcessorPrecomputedEmbeddings:
    """Tests for pre-computed embeddings support (Phase 2 - Pipeline Unification).

    FirmProcessor should accept optional pre-computed embeddings and pass
    them to the topic model. This enables the unified pipeline to compute
    embeddings once and reuse them for storage in PostgreSQL.
    """

    def test_process_accepts_embeddings_parameter(self, sample_config):
        """process() should accept optional embeddings parameter."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 1]),
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["ai"], 1: ["revenue"]},
            probabilities=np.array([[0.8, 0.2], [0.3, 0.7]]),
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "AI investment.", "AI investment.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Revenue grew.", "Revenue grew.", "CFO", 1),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        # Pre-computed embeddings
        embeddings = np.random.rand(2, 768)

        # Should accept embeddings without error and return tuple
        output, topic_assignments = processor.process(firm_data, embeddings=embeddings)

        assert isinstance(output, dict)
        assert "firm_id" in output
        assert isinstance(topic_assignments, np.ndarray)

    def test_process_passes_embeddings_to_model(self, sample_config):
        """process() should pass embeddings to model.fit_transform()."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 1]),
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["ai"], 1: ["revenue"]},
            probabilities=np.array([[0.8, 0.2], [0.3, 0.7]]),
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "AI investment.", "AI investment.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Revenue grew.", "Revenue grew.", "CFO", 1),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        # Pre-computed embeddings
        embeddings = np.random.rand(2, 768)

        processor.process(firm_data, embeddings=embeddings)

        # Verify fit_transform was called with embeddings
        mock_model.fit_transform.assert_called_once()
        call_kwargs = mock_model.fit_transform.call_args.kwargs
        assert "embeddings" in call_kwargs
        assert np.array_equal(call_kwargs["embeddings"], embeddings)

    def test_process_without_embeddings_calls_model_normally(self, sample_config):
        """process() without embeddings should call model without embeddings param."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 1]),
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["ai"], 1: ["revenue"]},
            probabilities=np.array([[0.8, 0.2], [0.3, 0.7]]),
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "AI investment.", "AI investment.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Revenue grew.", "Revenue grew.", "CFO", 1),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        # Call without embeddings
        processor.process(firm_data)

        # Verify fit_transform called without embeddings (or embeddings=None)
        mock_model.fit_transform.assert_called_once()
        call_kwargs = mock_model.fit_transform.call_args.kwargs
        # Either no embeddings key, or embeddings=None
        assert call_kwargs.get("embeddings") is None


class TestFirmProcessorTopicAssignments:
    """Tests for topic_assignments access (needed for Postgres sentence→topic mapping).

    The unified pipeline needs access to topic_assignments array to set
    sentence.topic_id before bulk insert (avoiding insert-then-update pattern).
    """

    def test_process_returns_topic_assignments(self, sample_config):
        """process() should return topic_assignments for sentence→topic mapping."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        expected_assignments = np.array([0, 0, 1, 1, -1])

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=expected_assignments,
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["ai"], 1: ["revenue"]},
            probabilities=np.array([
                [0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8], [0.5, 0.5]
            ]),
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", f"Sentence {i}.", "CEO", i)
            for i in range(5)
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        # process() returns (output_dict, topic_assignments)
        output, topic_assignments = processor.process(firm_data)

        assert isinstance(output, dict)
        assert np.array_equal(topic_assignments, expected_assignments)

    def test_topic_assignments_length_matches_sentences(self, sample_config):
        """topic_assignments should have one entry per sentence."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 1, 0, -1, 1]),
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["ai"], 1: ["revenue"]},
            probabilities=np.array([
                [0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.5, 0.5], [0.2, 0.8]
            ]),
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", f"Sentence {i}.", "CEO", i)
            for i in range(5)
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        output, topic_assignments = processor.process(firm_data)

        assert len(topic_assignments) == len(sentences)
