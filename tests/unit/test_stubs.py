"""
Unit tests for stub implementations.

These tests verify that stubs properly raise NotImplementedError
and implement the correct interfaces.
"""

import pytest


class TestLDATopicModelStub:
    """Tests for LDATopicModel stub."""

    def test_implements_topic_model_interface(self):
        """LDATopicModel should implement TopicModel interface."""
        from cloud.src.topic_models.lda_model import LDATopicModel
        from cloud.src.interfaces import TopicModel

        assert issubclass(LDATopicModel, TopicModel)

    def test_can_instantiate(self):
        """LDATopicModel should be instantiable."""
        from cloud.src.topic_models.lda_model import LDATopicModel

        model = LDATopicModel()
        assert model is not None

    def test_fit_transform_raises_not_implemented(self):
        """fit_transform should raise NotImplementedError."""
        from cloud.src.topic_models.lda_model import LDATopicModel

        model = LDATopicModel()

        with pytest.raises(NotImplementedError):
            model.fit_transform(["doc1", "doc2"])


class TestNeuralTopicModelStub:
    """Tests for NeuralTopicModel stub."""

    def test_implements_topic_model_interface(self):
        """NeuralTopicModel should implement TopicModel interface."""
        from cloud.src.topic_models.neural_model import NeuralTopicModel
        from cloud.src.interfaces import TopicModel

        assert issubclass(NeuralTopicModel, TopicModel)

    def test_can_instantiate(self):
        """NeuralTopicModel should be instantiable."""
        from cloud.src.topic_models.neural_model import NeuralTopicModel

        model = NeuralTopicModel()
        assert model is not None

    def test_fit_transform_raises_not_implemented(self):
        """fit_transform should raise NotImplementedError."""
        from cloud.src.topic_models.neural_model import NeuralTopicModel

        model = NeuralTopicModel()

        with pytest.raises(NotImplementedError):
            model.fit_transform(["doc1", "doc2"])


class TestS3TranscriptConnectorStub:
    """Tests for S3TranscriptConnector stub."""

    def test_implements_data_connector_interface(self):
        """S3TranscriptConnector should implement DataConnector interface."""
        from cloud.src.connectors.s3_connector import S3TranscriptConnector
        from cloud.src.interfaces import DataConnector

        assert issubclass(S3TranscriptConnector, DataConnector)

    def test_can_instantiate(self):
        """S3TranscriptConnector should be instantiable."""
        from cloud.src.connectors.s3_connector import S3TranscriptConnector

        connector = S3TranscriptConnector(bucket="test", key="test.csv")
        assert connector is not None

    def test_fetch_transcripts_raises_not_implemented(self):
        """fetch_transcripts should raise NotImplementedError."""
        from cloud.src.connectors.s3_connector import S3TranscriptConnector

        connector = S3TranscriptConnector(bucket="test", key="test.csv")

        with pytest.raises(NotImplementedError):
            connector.fetch_transcripts(["1001"], "2023-01-01", "2023-03-31")

    def test_get_available_firm_ids_raises_not_implemented(self):
        """get_available_firm_ids should raise NotImplementedError."""
        from cloud.src.connectors.s3_connector import S3TranscriptConnector

        connector = S3TranscriptConnector(bucket="test", key="test.csv")

        with pytest.raises(NotImplementedError):
            connector.get_available_firm_ids()
