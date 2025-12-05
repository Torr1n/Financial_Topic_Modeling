"""
Unit tests for XAI LLM Client.

Tests are written BEFORE implementation (TDD).
All tests use mocked API calls - no real network requests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

# Enable asyncio mode for all tests in this module
pytestmark = pytest.mark.asyncio(loop_scope="function")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample LLM configuration."""
    return {
        "model": "grok-4-1-fast-reasoning",
        "max_concurrent": 50,
        "timeout": 30,
        "max_retries": 3,
    }


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a test summary."
    return mock_response


@pytest.fixture
def mock_async_openai_client(mock_openai_response):
    """Create a mock AsyncOpenAI client."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    return mock_client


# =============================================================================
# Client Initialization Tests
# =============================================================================

class TestXAIClientInit:
    """Tests for XAIClient initialization."""

    def test_client_initializes_with_api_key(self, sample_config):
        """Client should initialize with API key."""
        from cloud.src.llm import XAIClient

        client = XAIClient(api_key="test_key", config=sample_config)

        assert client is not None
        assert client._api_key == "test_key"

    def test_client_uses_default_config(self):
        """Client should use sensible defaults if config is empty."""
        from cloud.src.llm import XAIClient

        client = XAIClient(api_key="test_key", config={})

        assert client._model == "grok-4-1-fast-reasoning"
        assert client._max_concurrent == 50
        assert client._timeout == 30
        assert client._max_retries == 3

    def test_client_uses_provided_config(self, sample_config):
        """Client should use provided configuration values."""
        from cloud.src.llm import XAIClient

        custom_config = {
            "model": "custom-model",
            "max_concurrent": 25,
            "timeout": 60,
            "max_retries": 5,
        }
        client = XAIClient(api_key="test_key", config=custom_config)

        assert client._model == "custom-model"
        assert client._max_concurrent == 25
        assert client._timeout == 60
        assert client._max_retries == 5

    def test_client_creates_semaphore(self, sample_config):
        """Client should create semaphore for rate limiting."""
        from cloud.src.llm import XAIClient

        client = XAIClient(api_key="test_key", config=sample_config)

        assert client._semaphore is not None
        # Semaphore should match max_concurrent config
        assert client._semaphore._value == 50


# =============================================================================
# Topic Summary Generation Tests
# =============================================================================

class TestGenerateTopicSummary:
    """Tests for generate_topic_summary method."""

    async def test_generate_summary_calls_api(self, sample_config, mock_async_openai_client):
        """generate_topic_summary should call the OpenAI API."""
        from cloud.src.llm import XAIClient

        sentences = ["We are investing heavily in AI.", "Machine learning is our focus."]

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=sample_config)
            result = await client.generate_topic_summary(
                keywords="ai, machine learning, cloud",
                sentences=sentences,
            )

        mock_async_openai_client.chat.completions.create.assert_called_once()
        assert result == "This is a test summary."

    async def test_generate_summary_includes_keywords_in_prompt(self, sample_config, mock_async_openai_client):
        """Prompt should include the provided keywords."""
        from cloud.src.llm import XAIClient

        sentences = ["Revenue grew 15% this quarter.", "We exceeded earnings expectations."]

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=sample_config)
            await client.generate_topic_summary(
                keywords="revenue growth, quarterly earnings",
                sentences=sentences,
            )

        # Check the prompt content
        call_args = mock_async_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        prompt_content = messages[0]["content"] if messages else ""

        assert "revenue growth" in prompt_content
        assert "quarterly earnings" in prompt_content

    async def test_generate_summary_includes_sentences_in_prompt(self, sample_config, mock_async_openai_client):
        """Prompt should include the raw sentences for context."""
        from cloud.src.llm import XAIClient

        sentences = [
            "Our AI workloads have doubled this year.",
            "We're seeing strong demand for cloud services.",
        ]

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=sample_config)
            await client.generate_topic_summary(
                keywords="ai, cloud",
                sentences=sentences,
            )

        call_args = mock_async_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        prompt_content = messages[0]["content"] if messages else ""

        assert "Our AI workloads have doubled this year" in prompt_content
        assert "We're seeing strong demand for cloud services" in prompt_content

    async def test_generate_summary_uses_configured_model(self, sample_config, mock_async_openai_client):
        """API call should use the configured model."""
        from cloud.src.llm import XAIClient

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=sample_config)
            await client.generate_topic_summary(
                keywords="test keywords",
                sentences=["Test sentence one.", "Test sentence two."],
            )

        call_args = mock_async_openai_client.chat.completions.create.call_args
        model = call_args.kwargs.get("model", call_args[1].get("model", ""))

        assert model == "grok-4-1-fast-reasoning"


# =============================================================================
# Theme Description Generation Tests
# =============================================================================

class TestGenerateThemeDescription:
    """Tests for generate_theme_description method."""

    async def test_generate_description_calls_api(self, sample_config, mock_async_openai_client):
        """generate_theme_description should call the OpenAI API."""
        from cloud.src.llm import XAIClient

        mock_async_openai_client.chat.completions.create.return_value.choices[0].message.content = "Theme description."

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=sample_config)
            result = await client.generate_theme_description(
                theme_keywords="AI Investment Theme",
                topic_summaries=["Topic 1 summary", "Topic 2 summary"]
            )

        mock_async_openai_client.chat.completions.create.assert_called_once()
        assert result == "Theme description."

    async def test_generate_description_includes_topic_summaries(self, sample_config, mock_async_openai_client):
        """Prompt should include topic summaries (not keywords)."""
        from cloud.src.llm import XAIClient

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=sample_config)
            await client.generate_theme_description(
                theme_keywords="Revenue Theme",
                topic_summaries=["Quarterly revenue discussion", "Earnings growth analysis"]
            )

        call_args = mock_async_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        prompt_content = messages[0]["content"] if messages else ""

        assert "Quarterly revenue discussion" in prompt_content
        assert "Earnings growth analysis" in prompt_content
        assert "Revenue Theme" in prompt_content


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiting:
    """Tests for semaphore-based rate limiting."""

    async def test_semaphore_limits_concurrency(self, sample_config, mock_async_openai_client):
        """Semaphore should limit concurrent requests."""
        from cloud.src.llm import XAIClient

        # Track concurrent calls
        concurrent_count = 0
        max_concurrent_observed = 0
        lock = asyncio.Lock()

        async def mock_create(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent_observed
            async with lock:
                concurrent_count += 1
                max_concurrent_observed = max(max_concurrent_observed, concurrent_count)

            await asyncio.sleep(0.01)  # Simulate API latency

            async with lock:
                concurrent_count -= 1

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Summary"
            return mock_response

        mock_async_openai_client.chat.completions.create = mock_create

        # Use low concurrency for testing
        config = {**sample_config, "max_concurrent": 5}

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=config)

            # Launch more tasks than the semaphore allows
            tasks = [
                client.generate_topic_summary(
                    keywords=f"topic {i}",
                    sentences=[f"Sentence about topic {i}."],
                )
                for i in range(20)
            ]
            await asyncio.gather(*tasks)

        # Should never exceed max_concurrent
        assert max_concurrent_observed <= 5


# =============================================================================
# Retry Logic Tests
# =============================================================================

class TestRetryLogic:
    """Tests for retry on transient errors."""

    async def test_retry_on_rate_limit_error(self, sample_config, mock_openai_response):
        """Client should retry on 429 rate limit errors."""
        from cloud.src.llm import XAIClient
        from openai import RateLimitError

        mock_client = AsyncMock()
        # First two calls raise 429, third succeeds
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                RateLimitError("Rate limit exceeded", response=MagicMock(status_code=429), body={}),
                RateLimitError("Rate limit exceeded", response=MagicMock(status_code=429), body={}),
                mock_openai_response,
            ]
        )

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_client):
            with patch('asyncio.sleep', new_callable=AsyncMock):  # Skip actual sleep
                client = XAIClient(api_key="test_key", config=sample_config)
                result = await client.generate_topic_summary(
                    keywords="test keywords",
                    sentences=["Test sentence."],
                )

        assert result == "This is a test summary."
        assert mock_client.chat.completions.create.call_count == 3

    async def test_retry_on_server_error(self, sample_config, mock_openai_response):
        """Client should retry on 503 server errors."""
        from cloud.src.llm import XAIClient
        from openai import APIStatusError

        mock_client = AsyncMock()
        # First call raises 503, second succeeds
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                APIStatusError("Service unavailable", response=mock_response, body={}),
                mock_openai_response,
            ]
        )

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_client):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                client = XAIClient(api_key="test_key", config=sample_config)
                result = await client.generate_topic_summary(
                    keywords="test keywords",
                    sentences=["Test sentence."],
                )

        assert result == "This is a test summary."
        assert mock_client.chat.completions.create.call_count == 2


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and fallbacks."""

    async def test_returns_none_on_persistent_failure(self, sample_config):
        """Client should return None after max retries exhausted."""
        from cloud.src.llm import XAIClient
        from openai import RateLimitError

        mock_client = AsyncMock()
        # All calls fail
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError("Rate limit exceeded", response=MagicMock(status_code=429), body={})
        )

        config = {**sample_config, "max_retries": 2}

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_client):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                client = XAIClient(api_key="test_key", config=config)
                result = await client.generate_topic_summary(
                    keywords="test keywords",
                    sentences=["Test sentence."],
                )

        assert result is None
        # Should have tried max_retries times
        assert mock_client.chat.completions.create.call_count == 2

    async def test_handles_timeout_error(self, sample_config):
        """Client should handle timeout errors gracefully."""
        from cloud.src.llm import XAIClient
        from openai import APITimeoutError

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock())
        )

        config = {**sample_config, "max_retries": 1}

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_client):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                client = XAIClient(api_key="test_key", config=config)
                result = await client.generate_topic_summary(
                    keywords="test keywords",
                    sentences=["Test sentence."],
                )

        assert result is None


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestBatchProcessing:
    """Tests for batch summary generation."""

    async def test_batch_summaries_processes_all_topics(self, sample_config, mock_async_openai_client):
        """generate_batch_summaries should process all topics."""
        from cloud.src.llm import XAIClient

        topics = [
            {"topic_id": 0, "representation": "ai, cloud, infrastructure", "sentences": ["AI is growing.", "Cloud is key."]},
            {"topic_id": 1, "representation": "revenue, earnings, growth", "sentences": ["Revenue up 10%."]},
            {"topic_id": 2, "representation": "supply chain, logistics", "sentences": ["Supply chain improved."]},
        ]

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=sample_config)
            results = await client.generate_batch_summaries(topics)

        assert len(results) == 3
        assert all(r == "This is a test summary." for r in results)

    async def test_batch_summaries_runs_in_parallel(self, sample_config, mock_async_openai_client):
        """Batch processing should run requests in parallel."""
        from cloud.src.llm import XAIClient

        call_times = []

        async def mock_create(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Summary"
            return mock_response

        mock_async_openai_client.chat.completions.create = mock_create

        topics = [
            {"topic_id": i, "representation": f"topic {i}", "sentences": [f"Sentence for topic {i}."]}
            for i in range(10)
        ]

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=sample_config)
            await client.generate_batch_summaries(topics)

        # If parallel, all calls should start at roughly the same time
        # (within 0.05s of each other)
        time_spread = max(call_times) - min(call_times)
        assert time_spread < 0.05  # All started nearly simultaneously

    async def test_batch_summaries_handles_partial_failure(self, sample_config):
        """Batch should return None for failed items, not fail entirely."""
        from cloud.src.llm import XAIClient
        from openai import APIError

        mock_client = AsyncMock()
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second call fails
                raise APIError("API error", request=MagicMock(), body={})
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = f"Summary {call_count}"
            return mock_response

        mock_client.chat.completions.create = mock_create

        topics = [
            {"topic_id": 0, "representation": "topic 0", "sentences": ["Sentence 0."]},
            {"topic_id": 1, "representation": "topic 1", "sentences": ["Sentence 1."]},
            {"topic_id": 2, "representation": "topic 2", "sentences": ["Sentence 2."]},
        ]

        config = {**sample_config, "max_retries": 1}

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_client):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                client = XAIClient(api_key="test_key", config=config)
                results = await client.generate_batch_summaries(topics)

        # Should have 3 results, one is None (failed)
        assert len(results) == 3
        assert results[1] is None  # Second topic failed
        assert results[0] is not None
        assert results[2] is not None


# =============================================================================
# Prompt Content Tests
# =============================================================================

class TestPromptContent:
    """Tests for prompt construction."""

    async def test_topic_prompt_has_correct_structure(self, sample_config, mock_async_openai_client):
        """Topic summary prompt should have expected structure."""
        from cloud.src.llm import XAIClient

        sentences = [
            "We are investing in AI workloads.",
            "Cloud infrastructure is our priority.",
        ]

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=sample_config)
            await client.generate_topic_summary(
                keywords="ai workloads, cloud infrastructure",
                sentences=sentences,
            )

        call_args = mock_async_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        prompt = messages[0]["content"]

        # Should mention earnings calls and summary task
        assert "earnings call" in prompt.lower() or "topic" in prompt.lower()
        assert "ai workloads" in prompt
        assert "cloud infrastructure" in prompt
        # Should include the raw sentences
        assert "We are investing in AI workloads" in prompt
        assert "Cloud infrastructure is our priority" in prompt

    async def test_theme_prompt_uses_summaries_not_keywords(self, sample_config, mock_async_openai_client):
        """Theme description prompt should use topic summaries, not keywords."""
        from cloud.src.llm import XAIClient

        topic_summaries = [
            "Discussion of AI investments in cloud infrastructure",
            "Analysis of machine learning workload growth",
        ]

        with patch('cloud.src.llm.xai_client.AsyncOpenAI', return_value=mock_async_openai_client):
            client = XAIClient(api_key="test_key", config=sample_config)
            await client.generate_theme_description(
                theme_keywords="AI Infrastructure",
                topic_summaries=topic_summaries
            )

        call_args = mock_async_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        prompt = messages[0]["content"]

        # Should contain full summaries, not just keywords
        assert "Discussion of AI investments in cloud infrastructure" in prompt
        assert "Analysis of machine learning workload growth" in prompt
        assert "AI Infrastructure" in prompt
