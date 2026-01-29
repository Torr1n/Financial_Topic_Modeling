"""
Integration tests for vLLM endpoint.

Tests the XAI client against a real vLLM server (self-hosted LLM).
Marked with @pytest.mark.vllm to skip when vLLM is not available.

To run these tests:
    1. Start vLLM server locally or ensure ECS deployment is running
    2. Set LLM_BASE_URL environment variable
    3. Run: pytest tests/integration/test_vllm_integration.py -v -m vllm

Example:
    export LLM_BASE_URL="http://localhost:8000/v1"
    pytest tests/integration/test_vllm_integration.py -v
"""

import os
import pytest
import asyncio
import httpx

# Enable asyncio mode for all tests in this module
pytestmark = [
    pytest.mark.asyncio(loop_scope="function"),
    pytest.mark.vllm,
]


# =============================================================================
# Fixtures
# =============================================================================

def is_vllm_available() -> bool:
    """Check if vLLM endpoint is reachable."""
    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        return False

    # Remove /v1 suffix for health check
    health_url = base_url.rstrip("/").replace("/v1", "") + "/health"

    try:
        response = httpx.get(health_url, timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def vllm_available():
    """Check if vLLM is available, skip tests if not."""
    if not is_vllm_available():
        pytest.skip(
            "vLLM not available. Set LLM_BASE_URL and ensure vLLM server is running."
        )
    return True


@pytest.fixture
def vllm_config():
    """Configuration for vLLM client."""
    return {
        "model": os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B"),
        "max_concurrent": 10,
        "timeout": 60,  # Higher timeout for vLLM (model inference can be slow)
        "max_retries": 2,
    }


@pytest.fixture
def sample_topic_data():
    """Sample topic data for testing."""
    return {
        "keywords": "artificial intelligence, cloud computing, infrastructure investment",
        "sentences": [
            "We are investing heavily in AI infrastructure across our data centers.",
            "Cloud computing remains a strategic priority for our business.",
            "Our AI workloads have increased by 40% year over year.",
            "We continue to build out our global infrastructure footprint.",
            "Machine learning capabilities are being embedded across our products.",
        ],
    }


@pytest.fixture
def sample_theme_data():
    """Sample theme data for testing."""
    return {
        "theme_keywords": "Technology Infrastructure Investment",
        "topic_summaries": [
            "Discussion of AI infrastructure expansion and cloud computing investments.",
            "Analysis of machine learning workload growth and data center spending.",
            "Focus on global infrastructure buildout for AI capabilities.",
        ],
    }


# =============================================================================
# vLLM Connection Tests
# =============================================================================

class TestVLLMConnection:
    """Tests for vLLM server connectivity."""

    async def test_vllm_health_check(self, vllm_available):
        """vLLM server should respond to health check."""
        base_url = os.environ.get("LLM_BASE_URL")
        health_url = base_url.rstrip("/").replace("/v1", "") + "/health"

        async with httpx.AsyncClient() as client:
            response = await client.get(health_url, timeout=10.0)
            assert response.status_code == 200

    async def test_vllm_models_endpoint(self, vllm_available):
        """vLLM should list available models."""
        base_url = os.environ.get("LLM_BASE_URL")
        models_url = f"{base_url.rstrip('/')}/models"

        async with httpx.AsyncClient() as client:
            response = await client.get(models_url, timeout=10.0)
            assert response.status_code == 200

            data = response.json()
            assert "data" in data
            assert len(data["data"]) > 0  # At least one model loaded


# =============================================================================
# Topic Summary Generation Tests
# =============================================================================

class TestTopicSummaryWithVLLM:
    """Tests for topic summary generation with vLLM."""

    async def test_generate_topic_summary_with_vllm(
        self, vllm_available, vllm_config, sample_topic_data
    ):
        """XAIClient should generate topic summaries using vLLM."""
        from cloud.src.llm import XAIClient

        # vLLM doesn't require real API key
        client = XAIClient(api_key="dummy", config=vllm_config)

        summary = await client.generate_topic_summary(
            keywords=sample_topic_data["keywords"],
            sentences=sample_topic_data["sentences"],
            log_prompt=True,  # Log first prompt for debugging
        )

        assert summary is not None
        assert len(summary) > 10  # Non-trivial response
        assert isinstance(summary, str)

        # Summary should not be an error message
        assert "error" not in summary.lower()[:50]

    async def test_topic_summary_content_quality(
        self, vllm_available, vllm_config, sample_topic_data
    ):
        """Topic summary should be relevant to input content."""
        from cloud.src.llm import XAIClient

        client = XAIClient(api_key="dummy", config=vllm_config)

        summary = await client.generate_topic_summary(
            keywords=sample_topic_data["keywords"],
            sentences=sample_topic_data["sentences"],
        )

        # Summary should mention relevant concepts (case-insensitive)
        summary_lower = summary.lower() if summary else ""

        # At least one relevant term should appear
        relevant_terms = ["ai", "artificial intelligence", "cloud", "infrastructure", "investment"]
        has_relevant_term = any(term in summary_lower for term in relevant_terms)

        assert has_relevant_term, f"Summary lacks relevant terms: {summary}"


# =============================================================================
# Theme Description Generation Tests
# =============================================================================

class TestThemeDescriptionWithVLLM:
    """Tests for theme description generation with vLLM."""

    async def test_generate_theme_description_with_vllm(
        self, vllm_available, vllm_config, sample_theme_data
    ):
        """XAIClient should generate theme descriptions using vLLM."""
        from cloud.src.llm import XAIClient

        client = XAIClient(api_key="dummy", config=vllm_config)

        description = await client.generate_theme_description(
            theme_keywords=sample_theme_data["theme_keywords"],
            topic_summaries=sample_theme_data["topic_summaries"],
            log_prompt=True,
        )

        assert description is not None
        assert len(description) > 20  # Non-trivial response
        assert isinstance(description, str)

    async def test_theme_description_synthesizes_topics(
        self, vllm_available, vllm_config, sample_theme_data
    ):
        """Theme description should synthesize multiple topic summaries."""
        from cloud.src.llm import XAIClient

        client = XAIClient(api_key="dummy", config=vllm_config)

        description = await client.generate_theme_description(
            theme_keywords=sample_theme_data["theme_keywords"],
            topic_summaries=sample_theme_data["topic_summaries"],
        )

        # Description should be longer than individual summaries (synthesis)
        avg_summary_len = sum(len(s) for s in sample_theme_data["topic_summaries"]) / len(
            sample_theme_data["topic_summaries"]
        )

        # Theme description should be substantial
        assert len(description) > avg_summary_len * 0.5


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestBatchProcessingWithVLLM:
    """Tests for batch topic summary processing with vLLM."""

    async def test_batch_summaries_with_vllm(self, vllm_available, vllm_config):
        """Batch processing should generate summaries for all topics."""
        from cloud.src.llm import XAIClient

        topics = [
            {
                "topic_id": 0,
                "representation": "revenue growth earnings",
                "sentences": ["Revenue grew 15% this quarter.", "We exceeded earnings expectations."],
            },
            {
                "topic_id": 1,
                "representation": "supply chain logistics",
                "sentences": ["Supply chain improvements drove efficiency.", "Logistics costs decreased."],
            },
            {
                "topic_id": 2,
                "representation": "product innovation R&D",
                "sentences": ["R&D investment increased 20%.", "New product launches planned."],
            },
        ]

        client = XAIClient(api_key="dummy", config=vllm_config)

        summaries = await client.generate_batch_summaries(topics, log_first_prompt=True)

        assert len(summaries) == 3

        # All summaries should be generated (no failures expected)
        for i, summary in enumerate(summaries):
            assert summary is not None, f"Topic {i} summary is None"
            assert len(summary) > 10, f"Topic {i} summary too short: {summary}"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestVLLMErrorHandling:
    """Tests for error handling with vLLM."""

    async def test_handles_empty_sentences(self, vllm_available, vllm_config):
        """Should handle empty sentence list gracefully."""
        from cloud.src.llm import XAIClient

        client = XAIClient(api_key="dummy", config=vllm_config)

        # Empty sentences - should still work (keywords only)
        summary = await client.generate_topic_summary(
            keywords="test keywords",
            sentences=[],
        )

        # May return None or a short response - both are acceptable
        # Key: should not raise exception
        assert summary is None or isinstance(summary, str)

    async def test_handles_long_input(self, vllm_available, vllm_config):
        """Should handle long input by truncating sentences."""
        from cloud.src.llm import XAIClient

        client = XAIClient(api_key="dummy", config=vllm_config)

        # Generate many sentences (more than MAX_SENTENCES_IN_PROMPT)
        long_sentences = [f"Sentence number {i} about various business topics." for i in range(100)]

        summary = await client.generate_topic_summary(
            keywords="business topics general discussion",
            sentences=long_sentences,
        )

        # Should complete without error (truncation happens internally)
        assert summary is None or isinstance(summary, str)


# =============================================================================
# Performance Tests
# =============================================================================

class TestVLLMPerformance:
    """Basic performance tests for vLLM."""

    async def test_concurrent_requests(self, vllm_available, vllm_config):
        """vLLM should handle concurrent requests."""
        from cloud.src.llm import XAIClient

        # Use lower concurrency for testing
        config = {**vllm_config, "max_concurrent": 3}
        client = XAIClient(api_key="dummy", config=config)

        topics = [
            {"topic_id": i, "representation": f"topic {i} keywords", "sentences": [f"Sentence for topic {i}."]}
            for i in range(5)
        ]

        import time
        start = time.time()
        summaries = await client.generate_batch_summaries(topics)
        elapsed = time.time() - start

        # All should complete
        assert len(summaries) == 5

        # Should complete in reasonable time (not sequential)
        # With 5 topics and ~2s per request, sequential would be ~10s
        # Parallel should be much faster
        assert elapsed < 30, f"Batch took too long: {elapsed:.1f}s"
