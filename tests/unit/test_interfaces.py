"""
Unit tests for abstract interfaces.

Tests are written BEFORE implementation (TDD).
These tests define the contract that implementations must follow.
"""

import pytest
from abc import ABC
from typing import List


class TestTopicModelInterface:
    """Tests for TopicModel abstract base class."""

    def test_topic_model_is_abstract(self):
        """TopicModel should be an abstract base class."""
        from cloud.src.interfaces import TopicModel

        assert issubclass(TopicModel, ABC)

    def test_topic_model_requires_fit_transform(self):
        """TopicModel should require fit_transform method."""
        from cloud.src.interfaces import TopicModel

        # Try to create a concrete class without fit_transform
        class BadModel(TopicModel):
            pass

        with pytest.raises(TypeError):
            BadModel()

    def test_topic_model_implementation(self):
        """A valid TopicModel implementation should work."""
        from cloud.src.interfaces import TopicModel
        from cloud.src.models import TopicModelResult
        import numpy as np

        class GoodModel(TopicModel):
            def fit_transform(self, documents: List[str]) -> TopicModelResult:
                return TopicModelResult(
                    topic_assignments=np.array([0] * len(documents)),
                    n_topics=1,
                    topic_representations={0: "Test Topic"},
                    topic_keywords={0: ["test"]},
                )

        model = GoodModel()
        result = model.fit_transform(["doc1", "doc2"])

        assert isinstance(result, TopicModelResult)
        assert result.n_topics == 1


class TestDataConnectorInterface:
    """Tests for DataConnector abstract base class."""

    def test_data_connector_is_abstract(self):
        """DataConnector should be an abstract base class."""
        from cloud.src.interfaces import DataConnector

        assert issubclass(DataConnector, ABC)

    def test_data_connector_requires_fetch_transcripts(self):
        """DataConnector should require fetch_transcripts method."""
        from cloud.src.interfaces import DataConnector

        class MissingFetch(DataConnector):
            def get_available_firms(self) -> List[str]:
                return []

        with pytest.raises(TypeError):
            MissingFetch()

    def test_data_connector_requires_get_available_firms(self):
        """DataConnector should require get_available_firms method."""
        from cloud.src.interfaces import DataConnector
        from cloud.src.models import TranscriptData

        class MissingGetFirms(DataConnector):
            def fetch_transcripts(
                self, firms: List[str], start_date: str, end_date: str
            ) -> TranscriptData:
                return TranscriptData(firms={})

        with pytest.raises(TypeError):
            MissingGetFirms()

    def test_data_connector_close_has_default(self):
        """DataConnector.close should have a default no-op implementation."""
        from cloud.src.interfaces import DataConnector
        from cloud.src.models import TranscriptData

        class MinimalConnector(DataConnector):
            def fetch_transcripts(
                self, firms: List[str], start_date: str, end_date: str
            ) -> TranscriptData:
                return TranscriptData(firms={})

            def get_available_firms(self) -> List[str]:
                return []

        connector = MinimalConnector()
        # close() should work without raising an exception
        connector.close()

    def test_data_connector_implementation(self):
        """A valid DataConnector implementation should work."""
        from cloud.src.interfaces import DataConnector
        from cloud.src.models import TranscriptData, FirmTranscriptData, TranscriptSentence

        class TestConnector(DataConnector):
            def fetch_transcripts(
                self, firms: List[str], start_date: str, end_date: str
            ) -> TranscriptData:
                # Create mock data
                sentences = [TranscriptSentence("TEST_001", "Test sentence.", "CEO", 0)]
                firm = FirmTranscriptData("1001", "Test Corp", sentences)
                return TranscriptData(firms={"1001": firm})

            def get_available_firms(self) -> List[str]:
                return ["Test Corp"]

        connector = TestConnector()
        result = connector.fetch_transcripts(["Test Corp"], "2023-01-01", "2023-03-31")

        assert isinstance(result, TranscriptData)
        assert "1001" in result.firms
        assert connector.get_available_firms() == ["Test Corp"]


class TestInterfaceMethodSignatures:
    """Tests to verify interface method signatures match the plan."""

    def test_topic_model_fit_transform_signature(self):
        """TopicModel.fit_transform should accept List[str] and return TopicModelResult."""
        from cloud.src.interfaces import TopicModel
        from cloud.src.models import TopicModelResult
        import inspect

        sig = inspect.signature(TopicModel.fit_transform)
        params = list(sig.parameters.keys())

        # Should have self and documents parameters
        assert "self" in params
        assert "documents" in params

        # Return annotation should be TopicModelResult
        assert sig.return_annotation == TopicModelResult

    def test_data_connector_fetch_transcripts_signature(self):
        """DataConnector.fetch_transcripts should have correct signature."""
        from cloud.src.interfaces import DataConnector
        from cloud.src.models import TranscriptData
        import inspect

        sig = inspect.signature(DataConnector.fetch_transcripts)
        params = list(sig.parameters.keys())

        # Should have self, firms, start_date, end_date parameters
        assert "self" in params
        assert "firms" in params
        assert "start_date" in params
        assert "end_date" in params

        # Return annotation should be TranscriptData
        assert sig.return_annotation == TranscriptData
