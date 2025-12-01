"""
Pytest configuration and shared fixtures for cloud migration tests.

This module provides reusable fixtures for testing the topic modeling pipeline.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import MagicMock
import tempfile
import os


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

@pytest.fixture
def mock_topic_model_result():
    """Create a mock TopicModelResult for testing processors."""
    from cloud.src.models import TopicModelResult

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
        probabilities=None,
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
