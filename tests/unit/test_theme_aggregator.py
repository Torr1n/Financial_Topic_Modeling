"""
Unit tests for ThemeAggregator.

Tests are written BEFORE implementation (TDD).
ThemeAggregator aggregates firm-level topics into cross-firm themes using Dual-BERTopic.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from collections import Counter


class TestThemeAggregatorInit:
    """Tests for ThemeAggregator initialization."""

    def test_accepts_topic_model_and_config(self, mock_topic_model, sample_config):
        """ThemeAggregator should accept topic model and config."""
        from cloud.src.theme_aggregator import ThemeAggregator

        aggregator = ThemeAggregator(mock_topic_model, sample_config)

        assert aggregator.model == mock_topic_model
        assert aggregator.config == sample_config

    def test_dependency_injection(self, sample_config):
        """Should use injected topic model (not create its own)."""
        from cloud.src.theme_aggregator import ThemeAggregator

        mock_model = MagicMock()
        aggregator = ThemeAggregator(mock_model, sample_config)

        assert aggregator.model is mock_model

    def test_extracts_validation_config(self, mock_topic_model, sample_config):
        """Should extract min_firms and max_dominance from config."""
        from cloud.src.theme_aggregator import ThemeAggregator

        aggregator = ThemeAggregator(mock_topic_model, sample_config)

        assert aggregator.min_firms == 2
        assert aggregator.max_dominance == 0.4

    def test_uses_defaults_when_config_missing(self, mock_topic_model):
        """Should use defaults when validation config is missing."""
        from cloud.src.theme_aggregator import ThemeAggregator

        aggregator = ThemeAggregator(mock_topic_model, {})

        assert aggregator.min_firms == 2
        assert aggregator.max_dominance == 0.4


class TestThemeAggregatorAggregate:
    """Tests for ThemeAggregator.aggregate method."""

    def test_aggregate_returns_list(self, sample_config, sample_firm_topic_outputs):
        """aggregate() should return a list of ThemeOutput dicts."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        # Create mock with 6 topics -> 2 themes
        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0, 1, 1, 1]),  # 3 topics per theme
            n_topics=2,
            topic_representations={0: "AI Strategy Theme", 1: "Revenue Growth Theme"},
            topic_keywords={0: ["ai", "strategy"], 1: ["revenue", "growth"]},
            probabilities=np.array([
                [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
                [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]
            ]),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(sample_firm_topic_outputs)

        assert isinstance(result, list)

    def test_aggregate_calls_model_with_representations(self, sample_config, sample_firm_topic_outputs):
        """aggregate() should call model.fit_transform with topic representations."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0, 1, 1, 1]),
            n_topics=2,
            topic_representations={0: "Theme A", 1: "Theme B"},
            topic_keywords={0: ["a"], 1: ["b"]},
            probabilities=np.array([[0.8, 0.2]] * 6),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        aggregator.aggregate(sample_firm_topic_outputs)

        mock_model.fit_transform.assert_called_once()
        call_args = mock_model.fit_transform.call_args[0][0]

        # Should be a list of representation strings
        assert isinstance(call_args, list)
        assert len(call_args) == 6  # 2 topics per firm * 3 firms

    def test_aggregate_extracts_correct_representations(self, sample_config):
        """Should extract topic representations as documents for re-embedding."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        firm_outputs = [
            {
                "firm_id": "1001",
                "firm_name": "Firm A",
                "n_topics": 1,
                "topics": [{"topic_id": 0, "representation": "ai investment", "keywords": ["ai"], "size": 10, "sentence_ids": []}],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1002",
                "firm_name": "Firm B",
                "n_topics": 1,
                "topics": [{"topic_id": 0, "representation": "revenue growth", "keywords": ["revenue"], "size": 15, "sentence_ids": []}],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
        ]

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0]),
            n_topics=1,
            topic_representations={0: "Combined Theme"},
            topic_keywords={0: ["combined"]},
            probabilities=np.array([[1.0], [1.0]]),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        aggregator.aggregate(firm_outputs)

        call_args = mock_model.fit_transform.call_args[0][0]
        assert "ai investment" in call_args
        assert "revenue growth" in call_args


class TestThemeAggregatorValidation:
    """Tests for theme validation filters."""

    def test_min_firms_filter(self, sample_config):
        """Themes with fewer than min_firms should be filtered out."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        # Three firms - theme 0 has topics from all 3 (valid), theme 1 has topic from only 1 (invalid)
        # Using 3 firms ensures max_dominance (33%) < 40% threshold
        firm_outputs = [
            {
                "firm_id": "1001",
                "firm_name": "Firm A",
                "n_topics": 2,
                "topics": [
                    {"topic_id": 0, "representation": "ai", "keywords": ["ai"], "size": 10, "sentence_ids": []},
                    {"topic_id": 1, "representation": "cloud", "keywords": ["cloud"], "size": 8, "sentence_ids": []},
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1002",
                "firm_name": "Firm B",
                "n_topics": 1,
                "topics": [
                    {"topic_id": 0, "representation": "ai tech", "keywords": ["ai"], "size": 12, "sentence_ids": []},
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1003",
                "firm_name": "Firm C",
                "n_topics": 1,
                "topics": [
                    {"topic_id": 0, "representation": "ai innovation", "keywords": ["ai"], "size": 11, "sentence_ids": []},
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
        ]

        mock_model = MagicMock()
        # Theme 0: topics from all 3 firms (valid - 3 firms, 33% each)
        # Theme 1: topic only from firm 1001 (invalid - single firm)
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 1, 0, 0]),  # A0->0, A1->1, B0->0, C0->0
            n_topics=2,
            topic_representations={0: "AI Theme", 1: "Cloud Theme"},
            topic_keywords={0: ["ai"], 1: ["cloud"]},
            probabilities=np.array([[0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.85, 0.15]]),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(firm_outputs)

        # Only theme 0 should remain (has 3 firms, each with 33% dominance)
        assert len(result) == 1
        assert result[0]["n_firms"] == 3

    def test_max_dominance_filter(self, sample_config):
        """Themes where one firm has >40% of topics should be filtered out."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        # Create scenario where one firm dominates a theme
        firm_outputs = [
            {
                "firm_id": "1001",
                "firm_name": "Firm A",
                "n_topics": 3,
                "topics": [
                    {"topic_id": i, "representation": f"topic {i}", "keywords": ["a"], "size": 10, "sentence_ids": []}
                    for i in range(3)
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1002",
                "firm_name": "Firm B",
                "n_topics": 1,
                "topics": [
                    {"topic_id": 0, "representation": "topic b", "keywords": ["b"], "size": 5, "sentence_ids": []},
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
        ]

        mock_model = MagicMock()
        # All 4 topics go to theme 0 -> firm A has 3/4 = 75% dominance (>40%)
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0, 0]),
            n_topics=1,
            topic_representations={0: "Dominated Theme"},
            topic_keywords={0: ["dominated"]},
            probabilities=np.array([[1.0]] * 4),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(firm_outputs)

        # Theme should be filtered out (75% > 40%)
        assert len(result) == 0

    def test_valid_theme_passes_both_filters(self, sample_config):
        """Themes meeting both min_firms and max_dominance should pass."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        # 3 firms with equal distribution
        firm_outputs = [
            {
                "firm_id": f"100{i}",
                "firm_name": f"Firm {i}",
                "n_topics": 1,
                "topics": [
                    {"topic_id": 0, "representation": f"topic {i}", "keywords": ["x"], "size": 10, "sentence_ids": []},
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            }
            for i in range(3)
        ]

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0]),  # All to same theme
            n_topics=1,
            topic_representations={0: "Valid Theme"},
            topic_keywords={0: ["valid"]},
            probabilities=np.array([[1.0]] * 3),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(firm_outputs)

        # Theme should pass (3 firms, each has 33% = valid)
        assert len(result) == 1
        assert result[0]["n_firms"] == 3

    def test_outlier_topics_dropped(self, sample_config):
        """Topics assigned to theme -1 should be dropped."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        # Use 3 firms so that theme passes max_dominance filter (33% each < 40%)
        firm_outputs = [
            {
                "firm_id": "1001",
                "firm_name": "Firm A",
                "n_topics": 2,
                "topics": [
                    {"topic_id": 0, "representation": "valid topic", "keywords": ["valid"], "size": 10, "sentence_ids": []},
                    {"topic_id": 1, "representation": "outlier topic", "keywords": ["outlier"], "size": 5, "sentence_ids": []},
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1002",
                "firm_name": "Firm B",
                "n_topics": 1,
                "topics": [
                    {"topic_id": 0, "representation": "another valid", "keywords": ["valid"], "size": 12, "sentence_ids": []},
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1003",
                "firm_name": "Firm C",
                "n_topics": 1,
                "topics": [
                    {"topic_id": 0, "representation": "third valid", "keywords": ["valid"], "size": 11, "sentence_ids": []},
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
        ]

        mock_model = MagicMock()
        # Topic 1 from firm A is outlier (-1), other 3 topics go to theme 0
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, -1, 0, 0]),  # A1 is outlier
            n_topics=1,
            topic_representations={0: "Valid Theme"},
            topic_keywords={0: ["valid"]},
            probabilities=np.array([[1.0], [0.5], [1.0], [1.0]]),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(firm_outputs)

        # Should have 1 theme with 3 topics (outlier dropped)
        assert len(result) == 1
        assert result[0]["n_topics"] == 3


class TestThemeAggregatorEdgeCases:
    """Tests for edge cases."""

    def test_empty_firm_results(self, mock_topic_model, sample_config):
        """Empty firm_results should return empty list."""
        from cloud.src.theme_aggregator import ThemeAggregator

        aggregator = ThemeAggregator(mock_topic_model, sample_config)
        result = aggregator.aggregate([])

        assert result == []

    def test_all_topics_become_outliers(self, sample_config):
        """If all topics become outliers, return empty list."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        firm_outputs = [
            {
                "firm_id": "1001",
                "firm_name": "Firm A",
                "n_topics": 1,
                "topics": [{"topic_id": 0, "representation": "topic", "keywords": ["x"], "size": 10, "sentence_ids": []}],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
        ]

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([-1]),  # All outliers
            n_topics=0,
            topic_representations={},
            topic_keywords={},
            probabilities=np.zeros((1, 0)),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(firm_outputs)

        assert result == []

    def test_single_firm_returns_no_themes(self, sample_config):
        """Single firm input should return no themes (min_firms=2)."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        firm_outputs = [
            {
                "firm_id": "1001",
                "firm_name": "Firm A",
                "n_topics": 3,
                "topics": [
                    {"topic_id": i, "representation": f"topic {i}", "keywords": ["x"], "size": 10, "sentence_ids": []}
                    for i in range(3)
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
        ]

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0]),
            n_topics=1,
            topic_representations={0: "Single Firm Theme"},
            topic_keywords={0: ["single"]},
            probabilities=np.array([[1.0]] * 3),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(firm_outputs)

        # All themes filtered (single firm)
        assert result == []

    def test_firms_with_no_topics_skipped(self, sample_config):
        """Firms with n_topics=0 should be skipped."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        # Use 4 firms (1 with no topics) so 3 valid firms remain (33% each < 40%)
        firm_outputs = [
            {
                "firm_id": "1001",
                "firm_name": "Firm A",
                "n_topics": 1,
                "topics": [{"topic_id": 0, "representation": "topic a", "keywords": ["a"], "size": 10, "sentence_ids": []}],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1002",
                "firm_name": "Firm B",
                "n_topics": 0,  # No topics - should be skipped
                "topics": [],
                "outlier_sentence_ids": ["s1", "s2"],
                "metadata": {},
            },
            {
                "firm_id": "1003",
                "firm_name": "Firm C",
                "n_topics": 1,
                "topics": [{"topic_id": 0, "representation": "topic c", "keywords": ["c"], "size": 8, "sentence_ids": []}],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1004",
                "firm_name": "Firm D",
                "n_topics": 1,
                "topics": [{"topic_id": 0, "representation": "topic d", "keywords": ["d"], "size": 9, "sentence_ids": []}],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
        ]

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0]),  # Only 3 topics (firm B skipped)
            n_topics=1,
            topic_representations={0: "Valid Theme"},
            topic_keywords={0: ["valid"]},
            probabilities=np.array([[1.0], [1.0], [1.0]]),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(firm_outputs)

        # Should have 1 theme with 3 firms (firm B skipped)
        assert len(result) == 1
        assert result[0]["n_firms"] == 3


class TestThemeOutputSchema:
    """Tests for ThemeOutput schema compliance."""

    def test_output_has_required_fields(self, sample_config, sample_firm_topic_outputs):
        """Output should have all required ThemeOutput fields."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0, 0, 0, 0]),
            n_topics=1,
            topic_representations={0: "Theme"},
            topic_keywords={0: ["keyword"]},
            probabilities=np.array([[1.0]] * 6),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(sample_firm_topic_outputs)

        assert len(result) == 1
        theme = result[0]

        # Required fields per plan
        assert "theme_id" in theme
        assert "name" in theme
        assert "keywords" in theme
        assert "n_firms" in theme
        assert "n_topics" in theme
        assert "topics" in theme
        assert "metadata" in theme

    def test_topic_structure_in_theme(self, sample_config, sample_firm_topic_outputs):
        """Each topic in theme should have required fields."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0, 0, 0, 0]),
            n_topics=1,
            topic_representations={0: "Theme"},
            topic_keywords={0: ["keyword"]},
            probabilities=np.array([[1.0]] * 6),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(sample_firm_topic_outputs)

        for topic in result[0]["topics"]:
            assert "firm_id" in topic
            assert "topic_id" in topic
            assert "representation" in topic
            assert "size" in topic

    def test_n_firms_matches_distinct_firms(self, sample_config, sample_firm_topic_outputs):
        """n_firms should match number of distinct firms in theme."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0, 0, 0, 0]),
            n_topics=1,
            topic_representations={0: "Theme"},
            topic_keywords={0: ["keyword"]},
            probabilities=np.array([[1.0]] * 6),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(sample_firm_topic_outputs)

        theme = result[0]
        distinct_firms = len(set(t["firm_id"] for t in theme["topics"]))
        assert theme["n_firms"] == distinct_firms

    def test_n_topics_matches_topic_count(self, sample_config, sample_firm_topic_outputs):
        """n_topics should match number of topics in list."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0, 0, 0, 0]),
            n_topics=1,
            topic_representations={0: "Theme"},
            topic_keywords={0: ["keyword"]},
            probabilities=np.array([[1.0]] * 6),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(sample_firm_topic_outputs)

        theme = result[0]
        assert theme["n_topics"] == len(theme["topics"])

    def test_metadata_has_required_fields(self, sample_config, sample_firm_topic_outputs):
        """Metadata should have processing_timestamp, model_config, validation."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0, 0, 0, 0]),
            n_topics=1,
            topic_representations={0: "Theme"},
            topic_keywords={0: ["keyword"]},
            probabilities=np.array([[1.0]] * 6),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(sample_firm_topic_outputs)

        metadata = result[0]["metadata"]
        assert "processing_timestamp" in metadata
        assert "model_config" in metadata
        assert "validation" in metadata


class TestThemeOutputJsonSerializable:
    """Tests for JSON serializability of output."""

    def test_output_is_json_serializable(self, sample_config, sample_firm_topic_outputs):
        """Output should be JSON serializable."""
        import json
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0, 0, 0, 0]),
            n_topics=1,
            topic_representations={0: "Theme"},
            topic_keywords={0: ["keyword"]},
            probabilities=np.array([[1.0]] * 6),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(sample_firm_topic_outputs)

        # Should not raise
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)


class TestPhase3SummaryBasedClustering:
    """Tests for Phase 3: Summary-based theme clustering."""

    def test_aggregate_uses_summaries_when_available(self, sample_config):
        """aggregate() should use summaries (not keywords) for clustering when present."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        # Topics with both representation (keywords) and summary (LLM-generated)
        firm_outputs = [
            {
                "firm_id": "1001",
                "firm_name": "Firm A",
                "n_topics": 1,
                "topics": [{
                    "topic_id": 0,
                    "representation": "ai ml cloud",  # Keywords
                    "summary": "Discussion of AI and machine learning investments in cloud infrastructure.",  # LLM summary
                    "keywords": ["ai", "ml"],
                    "size": 10,
                    "sentence_ids": [],
                }],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1002",
                "firm_name": "Firm B",
                "n_topics": 1,
                "topics": [{
                    "topic_id": 0,
                    "representation": "revenue earnings growth",  # Keywords
                    "summary": "Quarterly revenue performance exceeded analyst expectations.",  # LLM summary
                    "keywords": ["revenue"],
                    "size": 15,
                    "sentence_ids": [],
                }],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
        ]

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0]),
            n_topics=1,
            topic_representations={0: "Combined Theme"},
            topic_keywords={0: ["combined"]},
            probabilities=np.array([[1.0], [1.0]]),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        aggregator.aggregate(firm_outputs)

        # Check what was passed to fit_transform
        call_args = mock_model.fit_transform.call_args[0][0]

        # Should contain LLM summaries, NOT keywords
        assert "Discussion of AI and machine learning investments" in call_args[0]
        assert "Quarterly revenue performance exceeded" in call_args[1]
        # Should NOT contain raw keywords
        assert "ai ml cloud" not in call_args
        assert "revenue earnings growth" not in call_args

    def test_aggregate_falls_back_to_representation_when_no_summary(self, sample_config):
        """aggregate() should fall back to representation when summary is missing."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        # Mix of topics with and without summaries
        firm_outputs = [
            {
                "firm_id": "1001",
                "firm_name": "Firm A",
                "n_topics": 1,
                "topics": [{
                    "topic_id": 0,
                    "representation": "ai ml cloud",
                    "summary": "AI and ML discussion.",  # Has summary
                    "keywords": ["ai"],
                    "size": 10,
                    "sentence_ids": [],
                }],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1002",
                "firm_name": "Firm B",
                "n_topics": 1,
                "topics": [{
                    "topic_id": 0,
                    "representation": "revenue growth performance",  # Keywords only
                    # No summary field - should fall back to representation
                    "keywords": ["revenue"],
                    "size": 15,
                    "sentence_ids": [],
                }],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
        ]

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0]),
            n_topics=1,
            topic_representations={0: "Combined Theme"},
            topic_keywords={0: ["combined"]},
            probabilities=np.array([[1.0], [1.0]]),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        aggregator.aggregate(firm_outputs)

        call_args = mock_model.fit_transform.call_args[0][0]

        # First topic uses summary
        assert "AI and ML discussion." in call_args[0]
        # Second topic falls back to representation (no summary)
        assert "revenue growth performance" in call_args[1]

    def test_topic_metadata_includes_summary(self, sample_config):
        """Topic metadata should include summary field for downstream use."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        # Need 3 firms so each has 33% dominance (< 40% threshold)
        firm_outputs = [
            {
                "firm_id": "1001",
                "firm_name": "Firm A",
                "n_topics": 1,
                "topics": [{
                    "topic_id": 0,
                    "representation": "ai ml",
                    "summary": "AI discussion summary.",
                    "keywords": ["ai"],
                    "size": 10,
                    "sentence_ids": [],
                }],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1002",
                "firm_name": "Firm B",
                "n_topics": 1,
                "topics": [{
                    "topic_id": 0,
                    "representation": "revenue",
                    "summary": "Revenue summary.",
                    "keywords": ["revenue"],
                    "size": 15,
                    "sentence_ids": [],
                }],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
            {
                "firm_id": "1003",
                "firm_name": "Firm C",
                "n_topics": 1,
                "topics": [{
                    "topic_id": 0,
                    "representation": "growth",
                    "summary": "Growth summary.",
                    "keywords": ["growth"],
                    "size": 12,
                    "sentence_ids": [],
                }],
                "outlier_sentence_ids": [],
                "metadata": {},
            },
        ]

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 0]),  # 3 topics
            n_topics=1,
            topic_representations={0: "Theme"},
            topic_keywords={0: ["theme"]},
            probabilities=np.array([[1.0], [1.0], [1.0]]),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(firm_outputs)

        # Each topic in the theme should have summary field
        assert len(result) == 1, f"Expected 1 theme, got {len(result)}"
        for topic in result[0]["topics"]:
            assert "summary" in topic
            assert topic["summary"] is not None


class TestThemeIdGeneration:
    """Tests for theme ID generation."""

    def test_theme_ids_follow_format(self, sample_config, sample_firm_topic_outputs):
        """Theme IDs should follow theme_YYYYMMDD_NNN format."""
        import re
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 1, 1, 0, 1]),  # 2 themes
            n_topics=2,
            topic_representations={0: "Theme A", 1: "Theme B"},
            topic_keywords={0: ["a"], 1: ["b"]},
            probabilities=np.array([[0.8, 0.2]] * 6),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(sample_firm_topic_outputs)

        pattern = r"theme_\d{8}_\d{3}"
        for theme in result:
            assert re.match(pattern, theme["theme_id"]), f"Invalid theme_id: {theme['theme_id']}"

    def test_themes_sorted_by_n_topics(self, sample_config, sample_firm_topic_outputs):
        """Themes should be sorted by n_topics descending."""
        from cloud.src.theme_aggregator import ThemeAggregator
        from cloud.src.models import TopicModelResult

        mock_model = MagicMock()
        # Theme 0: 2 topics, Theme 1: 4 topics
        mock_model.fit_transform.return_value = TopicModelResult(
            topic_assignments=np.array([0, 0, 1, 1, 1, 1]),
            n_topics=2,
            topic_representations={0: "Small Theme", 1: "Large Theme"},
            topic_keywords={0: ["small"], 1: ["large"]},
            probabilities=np.array([[0.8, 0.2]] * 6),
        )

        aggregator = ThemeAggregator(mock_model, sample_config)
        result = aggregator.aggregate(sample_firm_topic_outputs)

        # Themes should be sorted largest first
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i]["n_topics"] >= result[i + 1]["n_topics"]
