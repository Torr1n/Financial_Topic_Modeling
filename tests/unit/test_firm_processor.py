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

    def test_process_returns_dict(self, mock_topic_model, sample_config):
        """process() should return a dict (FirmTopicOutput schema)."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "AI investment.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Revenue grew.", "CFO", 1),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        assert isinstance(result, dict)

    def test_process_calls_model_fit_transform(self, mock_topic_model, sample_config):
        """process() should call model.fit_transform with sentence texts."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "AI investment.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Revenue grew.", "CFO", 1),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        processor.process(firm_data)

        mock_topic_model.fit_transform.assert_called_once()
        call_args = mock_topic_model.fit_transform.call_args[0][0]
        assert call_args == ["AI investment.", "Revenue grew."]


class TestFirmTopicOutputSchema:
    """Tests for FirmTopicOutput schema compliance."""

    def test_output_has_required_fields(self, mock_topic_model, sample_config):
        """Output should have all required FirmTopicOutput fields."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        # Required fields per plan
        assert "firm_id" in result
        assert "firm_name" in result
        assert "n_topics" in result
        assert "topics" in result
        assert "outlier_sentence_ids" in result
        assert "metadata" in result

    def test_output_firm_id_and_name(self, mock_topic_model, sample_config):
        """Output should preserve firm_id and firm_name."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        assert result["firm_id"] == "1001"
        assert result["firm_name"] == "Apple Inc."

    def test_topic_structure(self, mock_topic_model, sample_config):
        """Each topic should have required fields."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        for topic in result["topics"]:
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

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        assert result["n_topics"] == len(result["topics"])

    def test_metadata_has_required_fields(self, mock_topic_model, sample_config):
        """Metadata should have processing_timestamp, model_config, n_sentences_processed."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        assert "processing_timestamp" in result["metadata"]
        assert "model_config" in result["metadata"]
        assert "n_sentences_processed" in result["metadata"]
        assert result["metadata"]["n_sentences_processed"] == 10


class TestFirmProcessorSentenceIdMapping:
    """Tests for correct sentence_id assignment to topics."""

    def test_sentence_ids_mapped_to_correct_topics(self, sample_config):
        """Sentence IDs should be correctly mapped to their topics."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        # Create a mock model with known assignments
        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 1, 1, -1]),  # 2 in topic 0, 2 in topic 1, 1 outlier
            n_topics=2,
            topic_representations={0: "Topic A", 1: "Topic B"},
            topic_keywords={0: ["a", "b"], 1: ["c", "d"]},
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [
            TranscriptSentence("1001_T001_0000", "Sentence 0.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "Sentence 1.", "CEO", 1),
            TranscriptSentence("1001_T001_0002", "Sentence 2.", "CFO", 2),
            TranscriptSentence("1001_T001_0003", "Sentence 3.", "CFO", 3),
            TranscriptSentence("1001_T001_0004", "Sentence 4.", "COO", 4),
        ]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        # Topic 0 should have sentences 0 and 1
        topic_0 = next(t for t in result["topics"] if t["topic_id"] == 0)
        assert set(topic_0["sentence_ids"]) == {"1001_T001_0000", "1001_T001_0001"}
        assert topic_0["size"] == 2

        # Topic 1 should have sentences 2 and 3
        topic_1 = next(t for t in result["topics"] if t["topic_id"] == 1)
        assert set(topic_1["sentence_ids"]) == {"1001_T001_0002", "1001_T001_0003"}
        assert topic_1["size"] == 2

        # Outlier should be sentence 4
        assert result["outlier_sentence_ids"] == ["1001_T001_0004"]


class TestFirmProcessorOutlierHandling:
    """Tests for outlier (-1 topic) handling."""

    def test_outliers_tracked_separately(self, sample_config):
        """Outliers (topic_id=-1) should be in outlier_sentence_ids."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([-1, -1, 0, -1]),  # 3 outliers, 1 in topic 0
            n_topics=1,
            topic_representations={0: "Single Topic"},
            topic_keywords={0: ["keyword"]},
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", "CEO", i) for i in range(4)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        assert len(result["outlier_sentence_ids"]) == 3
        assert "1001_T001_0000" in result["outlier_sentence_ids"]
        assert "1001_T001_0001" in result["outlier_sentence_ids"]
        assert "1001_T001_0003" in result["outlier_sentence_ids"]

    def test_all_outliers_gives_empty_topics(self, sample_config):
        """If all sentences are outliers, topics list should be empty."""
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence, TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([-1, -1, -1]),  # All outliers
            n_topics=0,
            topic_representations={},
            topic_keywords={},
        )

        processor = FirmProcessor(mock_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", "CEO", i) for i in range(3)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        assert result["n_topics"] == 0
        assert result["topics"] == []
        assert len(result["outlier_sentence_ids"]) == 3


class TestFirmProcessorJsonSerializable:
    """Tests for JSON serializability of output."""

    def test_output_is_json_serializable(self, mock_topic_model, sample_config):
        """Output should be JSON serializable."""
        import json
        from cloud.src.firm_processor import FirmProcessor
        from cloud.src.models import FirmTranscriptData, TranscriptSentence

        processor = FirmProcessor(mock_topic_model, sample_config)

        sentences = [TranscriptSentence(f"1001_T001_{i:04d}", f"Sentence {i}.", "CEO", i) for i in range(10)]
        firm_data = FirmTranscriptData("1001", "Apple Inc.", sentences)

        result = processor.process(firm_data)

        # Should not raise
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)
