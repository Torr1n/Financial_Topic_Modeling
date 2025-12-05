"""
Pytest configuration and shared fixtures for cloud migration tests.

This module provides reusable fixtures for testing the topic modeling pipeline.
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock
import tempfile
import os

# Configure pytest-asyncio to auto-detect async tests
pytest_plugins = ('pytest_asyncio',)


# =============================================================================
# Test Helpers
# =============================================================================

def make_sentence(
    sentence_id: str,
    text: str,
    speaker_type: Optional[str] = "CEO",
    position: int = 0,
    cleaned_text: Optional[str] = None,
):
    """
    Helper to create TranscriptSentence for tests.

    For tests, cleaned_text defaults to the same as raw_text (text param).
    This simplifies test creation while supporting the new dual-text model.

    Args:
        sentence_id: Unique sentence identifier
        text: The sentence text (used as raw_text, and cleaned_text if not specified)
        speaker_type: Speaker role (default: "CEO")
        position: Position in transcript (default: 0)
        cleaned_text: Optional explicit cleaned text (defaults to text)

    Returns:
        TranscriptSentence instance
    """
    from cloud.src.models import TranscriptSentence
    return TranscriptSentence(
        sentence_id=sentence_id,
        raw_text=text,
        cleaned_text=cleaned_text if cleaned_text is not None else text,
        speaker_type=speaker_type,
        position=position,
    )


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_documents() -> List[str]:
    """Sample documents for topic modeling tests.

    Note: Must have more documents than UMAP n_neighbors (15) to avoid
    spectral embedding errors.
    """
    return [
        # AI/ML cluster
        "We are investing heavily in artificial intelligence and machine learning capabilities.",
        "Our AI initiatives have shown strong results in customer engagement.",
        "The company is expanding its machine learning infrastructure.",
        "Deep learning models are driving product improvements.",
        "Neural networks power our recommendation systems.",
        "AI-driven automation has improved operational efficiency.",
        # Revenue/Growth cluster
        "Revenue growth exceeded expectations this quarter.",
        "We saw strong performance in our cloud computing segment.",
        "Profit margins have expanded due to cost optimization.",
        "Sales increased by fifteen percent year over year.",
        "Our subscription revenue continues to grow rapidly.",
        "Customer lifetime value has improved significantly.",
        # Supply Chain cluster
        "Supply chain disruptions continue to impact our operations.",
        "We are diversifying our supply chain to reduce risk.",
        "Logistics costs have increased due to global shipping constraints.",
        "Inventory management improvements are underway.",
        "We have secured alternative suppliers in key regions.",
        "Manufacturing capacity is being expanded globally.",
        # Digital/Customer cluster
        "Our digital transformation strategy is progressing well.",
        "Customer acquisition costs have decreased significantly.",
        "User engagement metrics show positive trends.",
        "Mobile app downloads reached record levels.",
    ]


@pytest.fixture
def sample_firm_sentences():
    """Sample sentences with metadata for testing FirmTranscriptData."""
    return [
        {"text": "AI investment is our top priority.", "speaker_type": "CEO", "position": 0},
        {"text": "Machine learning drives efficiency.", "speaker_type": "CTO", "position": 1},
        {"text": "Revenue exceeded expectations.", "speaker_type": "CFO", "position": 2},
        {"text": "Supply chain remains strong.", "speaker_type": "COO", "position": 3},
        {"text": "Customer growth continues.", "speaker_type": "CEO", "position": 4},
    ]


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for topic models."""
    return {
        "embedding_model": "all-mpnet-base-v2",
        "device": "cpu",  # Use CPU for testing (avoid CUDA errors on machines without GPU)
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
        "validation": {
            "min_firms": 2,
            "max_firm_dominance": 0.4,
        },
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


class MockSentenceTransformer:
    """
    Mock SentenceTransformer for tests.

    Returns deterministic embeddings without downloading models.
    All tests should use this to avoid network dependencies.
    """

    def __init__(self, model_name: str = "mock-model", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._embedding_dim = 768  # all-mpnet-base-v2 dimension

    def encode(
        self,
        sentences,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Return deterministic embeddings based on text hash."""
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []
        for i, text in enumerate(sentences):
            # Create deterministic but unique embedding per text
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self._embedding_dim).astype(np.float32)
            # Normalize to unit length (like real embeddings)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings)


@pytest.fixture
def mock_sentence_transformer():
    """Return a MockSentenceTransformer instance."""
    return MockSentenceTransformer()


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    """
    Patch SentenceTransformer globally to avoid network downloads.

    Applied autouse so ALL tests use the mock by default, preventing
    external model downloads in network-restricted environments.
    """
    mock_cls = MockSentenceTransformer

    # Replace the class entirely so instances are our mock
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        mock_cls,
    )

    return mock_cls


@pytest.fixture
def mock_topic_model_result():
    """Create a mock TopicModelResult for testing processors.

    Note: probabilities is now required - full (n_docs x n_topics) distribution matrix.
    """
    from cloud.src.models import TopicModelResult

    # 10 docs, 3 topics - probabilities matrix
    # Each row sums to ~1 (topic distribution per document)
    probabilities = np.array([
        [0.8, 0.1, 0.1],  # doc 0: topic 0 (high prob)
        [0.7, 0.2, 0.1],  # doc 1: topic 0
        [0.6, 0.3, 0.1],  # doc 2: topic 0 (lower prob)
        [0.1, 0.85, 0.05],  # doc 3: topic 1 (high prob)
        [0.2, 0.6, 0.2],  # doc 4: topic 1 (lower prob)
        [0.1, 0.1, 0.8],  # doc 5: topic 2 (high prob)
        [0.15, 0.15, 0.7],  # doc 6: topic 2
        [0.2, 0.2, 0.6],  # doc 7: topic 2 (lower prob)
        [0.33, 0.33, 0.34],  # doc 8: outlier (no clear topic)
        [0.34, 0.33, 0.33],  # doc 9: outlier
    ])

    return TopicModelResult(
        topic_assignments=np.array([0, 0, 0, 1, 1, 2, 2, 2, -1, -1]),
        n_topics=3,
        topic_representations={
            0: "Artificial Intelligence Investment",
            1: "Revenue Growth",
            2: "Supply Chain Management",
        },
        topic_keywords={
            0: ["ai", "machine learning", "investment", "technology", "capabilities"],
            1: ["revenue", "growth", "performance", "quarter", "expectations"],
            2: ["supply", "chain", "logistics", "operations", "disruption"],
        },
        probabilities=probabilities,
        topic_sizes={0: 3, 1: 2, 2: 3},
        metadata={"model": "test"},
    )


@pytest.fixture
def mock_topic_model(mock_topic_model_result):
    """Create a mock TopicModel for dependency injection tests."""
    mock = MagicMock()
    mock.fit_transform.return_value = mock_topic_model_result
    return mock


# =============================================================================
# CSV Fixtures
# =============================================================================

@pytest.fixture
def sample_csv_content() -> str:
    """Sample CSV content matching the expected schema.

    Note: Each componenttext contains MULTIPLE sentences to test sentence splitting.
    This mimics the real CSV where components are transcript chunks, not sentences.
    """
    return """companyid,companyname,transcriptid,componenttext,componentorder,mostimportantdateutc,speakertypename
1001,Apple Inc.,T001,"We are investing heavily in AI and machine learning. Our research team has made significant breakthroughs. This will drive future growth.",1,2023-01-15,CEO
1001,Apple Inc.,T001,"Revenue growth has been exceptional. We exceeded analyst expectations. Margins expanded significantly.",2,2023-01-15,CFO
1001,Apple Inc.,T001,"Supply chain remains stable. We have diversified our suppliers. Logistics costs are under control.",3,2023-01-15,COO
1002,Microsoft Corp.,T002,"Cloud computing continues to drive growth. Azure adoption is accelerating. Enterprise customers are expanding deployments.",1,2023-01-20,CEO
1002,Microsoft Corp.,T002,"We see strong demand for AI services. Copilot adoption exceeded expectations. Developer productivity has improved.",2,2023-01-20,CTO
1003,Tesla Inc.,T003,"Vehicle production exceeded targets. Manufacturing efficiency has improved. We opened new facilities.",1,2023-02-01,CEO
1003,Tesla Inc.,T003,"Battery technology advances continue. Energy density has increased. Cost per kilowatt-hour has decreased.",2,2023-02-01,CTO
"""


@pytest.fixture
def temp_csv_file(sample_csv_content) -> str:
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# =============================================================================
# Firm Topic Output Fixtures (for ThemeAggregator tests)
# =============================================================================

@pytest.fixture
def sample_firm_topic_outputs() -> List[Dict[str, Any]]:
    """Sample FirmTopicOutput dicts for testing ThemeAggregator.

    Creates 3 firms with 2 topics each (6 total topics).
    This enables testing cross-firm theme aggregation.
    """
    return [
        {
            "firm_id": "1001",
            "firm_name": "Apple Inc.",
            "n_topics": 2,
            "topics": [
                {
                    "topic_id": 0,
                    "representation": "ai investment strategy",
                    "keywords": ["ai", "investment", "strategy"],
                    "size": 25,
                    "sentence_ids": ["1001_T001_0001", "1001_T001_0002"],
                },
                {
                    "topic_id": 1,
                    "representation": "revenue growth quarterly",
                    "keywords": ["revenue", "growth", "quarterly"],
                    "size": 18,
                    "sentence_ids": ["1001_T001_0003", "1001_T001_0004"],
                },
            ],
            "outlier_sentence_ids": ["1001_T001_0005"],
            "metadata": {"processing_timestamp": "2024-12-01T12:00:00Z"},
        },
        {
            "firm_id": "1002",
            "firm_name": "Microsoft Corp.",
            "n_topics": 2,
            "topics": [
                {
                    "topic_id": 0,
                    "representation": "cloud computing azure",
                    "keywords": ["cloud", "computing", "azure"],
                    "size": 30,
                    "sentence_ids": ["1002_T001_0001", "1002_T001_0002"],
                },
                {
                    "topic_id": 1,
                    "representation": "ai machine learning",
                    "keywords": ["ai", "machine", "learning"],
                    "size": 22,
                    "sentence_ids": ["1002_T001_0003", "1002_T001_0004"],
                },
            ],
            "outlier_sentence_ids": [],
            "metadata": {"processing_timestamp": "2024-12-01T12:00:00Z"},
        },
        {
            "firm_id": "1003",
            "firm_name": "Tesla Inc.",
            "n_topics": 2,
            "topics": [
                {
                    "topic_id": 0,
                    "representation": "electric vehicle production",
                    "keywords": ["electric", "vehicle", "production"],
                    "size": 28,
                    "sentence_ids": ["1003_T001_0001", "1003_T001_0002"],
                },
                {
                    "topic_id": 1,
                    "representation": "battery technology energy",
                    "keywords": ["battery", "technology", "energy"],
                    "size": 20,
                    "sentence_ids": ["1003_T001_0003", "1003_T001_0004"],
                },
            ],
            "outlier_sentence_ids": ["1003_T001_0005"],
            "metadata": {"processing_timestamp": "2024-12-01T12:00:00Z"},
        },
    ]


# =============================================================================
# Validation Helpers
# =============================================================================

@pytest.fixture
def validate_firm_topic_output():
    """Validator for FirmTopicOutput schema."""
    def _validate(output: dict) -> bool:
        required_keys = ["firm_id", "firm_name", "n_topics", "topics", "outlier_sentence_ids", "metadata"]
        if not all(k in output for k in required_keys):
            return False

        for topic in output["topics"]:
            topic_keys = ["topic_id", "representation", "keywords", "size", "sentence_ids"]
            if not all(k in topic for k in topic_keys):
                return False

        return True

    return _validate


@pytest.fixture
def validate_theme_output():
    """Validator for ThemeOutput schema."""
    def _validate(output: dict) -> bool:
        required_keys = ["theme_id", "name", "keywords", "n_firms", "n_topics", "topics", "metadata"]
        if not all(k in output for k in required_keys):
            return False

        for topic in output["topics"]:
            topic_keys = ["firm_id", "topic_id", "representation", "size"]
            if not all(k in topic for k in topic_keys):
                return False

        return True

    return _validate
