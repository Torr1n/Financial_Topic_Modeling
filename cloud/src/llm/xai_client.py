"""
Async xAI LLM Client for topic summarization and theme description.

Uses OpenAI-compatible API with:
- Semaphore-based rate limiting
- Exponential backoff retry logic
- Graceful error handling with fallbacks
- Configurable base URL (supports vLLM, xAI, or other OpenAI-compatible endpoints)

Usage:
    client = XAIClient(api_key="your_key", config={"max_concurrent": 50})
    summary = await client.generate_topic_summary("ai, cloud, infrastructure")

    # With vLLM:
    import os
    os.environ["LLM_BASE_URL"] = "http://vllm-alb.internal/v1"
    client = XAIClient(api_key="dummy", config={})  # vLLM doesn't require API key
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI, RateLimitError, APIStatusError, APITimeoutError, APIError

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MODEL = "grok-4-1-fast-reasoning"
DEFAULT_MAX_CONCURRENT = 50
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_URL = "https://api.x.ai/v1"

# Maximum number of sentences to include in topic prompt (to avoid token limits)
MAX_SENTENCES_IN_PROMPT = 50

# Prompt templates
TOPIC_SUMMARY_PROMPT = """## ROLE ##
You are a experienced financial analyst specializing in identifying themes in earnings call transcripts. Your task is to analyze and synthesize topic keywords and sentences to summarize a topic cluster discussed in this call.
## CONTEXT ##
I have a topic from an earnings call transcript derived from the collection of these sentences found to be in a cluster of similar sentence embeddings. This topic contains the following sentences:
{sentences}
Through a thorough preprocessing and analysis of the topic, it is described by these core keywords, ordered by importance:
{keywords}
## OBJECTIVE ##
Based on the above, provide a concise financial topic human-readable summary (max 1–2 sentences) that captures the key shared underlying theme and is the "why" behind the reason for these sentences to fall in a similar cluster. 
## GUIDELINES ##
In this summary, you MUST **avoid** company-specific procedural terms.
This is because of where these summaries fit in our cross-firm earnings call theme identification pipeline, the quality of our themes relies on our firm-level topics being **GENERALIZABLE**.
As such we want to summarize our topics without focusing on any company-specific brand names or specific services to ensure the topic-describing summaries that arise are suitable for highlighting the underlying cross-firm themes rather than company-specific themes.
## OUTPUT: RAW, UNFORMATTED TEXT ONLY ##
financial topic summary"""

THEME_DESCRIPTION_PROMPT = """## ROLE ##
You are a experienced financial analyst specializing in identifying themes across earnings call transcripts. Your task is to analyze and synthesize topic keywords and sentences to describe a thematic cluster found to be discussed across multiple firms' earnings calls.
## TOPIC CONTEXT ##
I have a cross-firm theme derived from the collection of these firm-topic-descriptions found to be in a cluster of similar textual embeddings. This theme contains the following topics:
{topic_summaries}
## REPRESENTATIVE KEYWORDS ##
The theme is described by these additional cross-firm keywords, ordered by importance: 
{theme_keywords}
## OBJECTIVE ##
Based on both the firm-level topic summaries and the theme-level keywords above, provide a concise cross-industry theme human-readable description (2–3 sentences) that captures the shared underlying concept and is the "why" behind the reason for this clustering.
## OUTPUT: RAW, UNFORMATTED TEXT ONLY ##
cross-firm theme description"""


class XAIClient:
    """
    Async xAI client for LLM-based topic summarization.

    Features:
    - OpenAI-compatible API (xAI uses same protocol)
    - Semaphore rate limiting (configurable concurrency)
    - Exponential backoff retry on transient errors
    - Graceful fallback (returns None on persistent failure)

    Attributes:
        _api_key: xAI API key
        _model: Model name (default: grok-beta)
        _max_concurrent: Max concurrent requests (default: 50)
        _timeout: Request timeout in seconds (default: 30)
        _max_retries: Max retry attempts (default: 3)
    """

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the xAI client.

        Args:
            api_key: xAI API key (can be "dummy" for vLLM which doesn't require auth)
            config: Optional configuration dict with keys:
                - model: Model name (default: grok-4-1-fast-reasoning)
                - max_concurrent: Max concurrent requests (default: 50)
                - timeout: Request timeout in seconds (default: 30)
                - max_retries: Max retry attempts (default: 3)
                - base_url: Override API base URL (default: xAI API)

        Environment Variables:
            LLM_BASE_URL: If set, overrides the default base URL (xAI API).
                          Config base_url takes priority over this env var.
        """
        config = config or {}

        self._api_key = api_key
        self._model = config.get("model", DEFAULT_MODEL)
        self._max_concurrent = config.get("max_concurrent", DEFAULT_MAX_CONCURRENT)
        self._timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self._max_retries = config.get("max_retries", DEFAULT_MAX_RETRIES)

        # Base URL priority: config > env var > default (xAI)
        self._base_url = config.get(
            "base_url",
            os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
        )

        # Rate limiting semaphore
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        # OpenAI-compatible client (works with xAI, vLLM, or any compatible endpoint)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

        logger.info(
            f"XAIClient initialized: model={self._model}, "
            f"base_url={self._base_url}, "
            f"max_concurrent={self._max_concurrent}, timeout={self._timeout}s"
        )

    async def generate_topic_summary(
        self,
        keywords: str,
        sentences: List[str],
        log_prompt: bool = False,
    ) -> Optional[str]:
        """
        Generate a human-readable summary for a topic.

        Args:
            keywords: Comma-separated keywords representing the topic
            sentences: Raw sentences from the transcript that belong to this topic
            log_prompt: If True, log the full prompt for debugging

        Returns:
            1-2 sentence summary, or None if generation fails
        """
        # Limit sentences to avoid token limits
        limited_sentences = sentences[:MAX_SENTENCES_IN_PROMPT]
        sentences_text = "\n".join(f"- {s}" for s in limited_sentences)

        prompt = TOPIC_SUMMARY_PROMPT.format(
            keywords=keywords,
            sentences=sentences_text,
        )

        if log_prompt:
            logger.info(f"[PROMPT EXAMPLE - Topic Summary]\n{prompt}\n{'='*60}")

        return await self._call_llm(prompt)

    async def generate_theme_description(
        self,
        theme_keywords: str,
        topic_summaries: List[str],
        log_prompt: bool = False,
    ) -> Optional[str]:
        """
        Generate a description for a cross-firm theme.

        Args:
            theme_keywords: Theme-level keywords or name (comma-separated string)
            topic_summaries: List of topic summaries (not keywords!)
            log_prompt: If True, log the full prompt for debugging

        Returns:
            2-3 sentence description, or None if generation fails
        """
        summaries_text = "\n".join(f"- {s}" for s in topic_summaries)
        prompt = THEME_DESCRIPTION_PROMPT.format(
            theme_keywords=theme_keywords,
            topic_summaries=summaries_text,
        )

        if log_prompt:
            logger.info(f"[PROMPT EXAMPLE - Theme Description]\n{prompt}\n{'='*60}")

        return await self._call_llm(prompt)

    async def generate_batch_summaries(
        self,
        topics: List[Dict[str, Any]],
        log_first_prompt: bool = False,
    ) -> List[Optional[str]]:
        """
        Generate summaries for multiple topics in parallel.

        Args:
            topics: List of topic dicts with "representation" and "sentences" keys
            log_first_prompt: If True, log the first prompt for debugging

        Returns:
            List of summaries (None for failed items)
        """
        tasks = [
            self.generate_topic_summary(
                keywords=topic["representation"],
                sentences=topic.get("sentences", []),
                log_prompt=(log_first_prompt and i == 0),
            )
            for i, topic in enumerate(topics)
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Make an LLM API call with rate limiting and retry logic.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response text, or None if all retries fail
        """
        async with self._semaphore:
            for attempt in range(self._max_retries):
                try:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content

                except (RateLimitError, APIStatusError) as e:
                    # Retry on rate limit (429) or server errors (5xx)
                    if isinstance(e, APIStatusError) and e.status_code < 500:
                        # Client error (4xx except 429) - don't retry
                        if not isinstance(e, RateLimitError):
                            logger.warning(f"Client error, not retrying: {e}")
                            return None

                    if attempt < self._max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(
                            f"API error (attempt {attempt + 1}/{self._max_retries}), "
                            f"retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exhausted: {e}")
                        return None

                except APITimeoutError as e:
                    if attempt < self._max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(
                            f"Timeout (attempt {attempt + 1}/{self._max_retries}), "
                            f"retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exhausted after timeout: {e}")
                        return None

                except APIError as e:
                    # General API error - log and return None
                    logger.error(f"API error: {e}")
                    return None

                except Exception as e:
                    # Unexpected error - log and return None
                    logger.error(f"Unexpected error calling LLM: {e}")
                    return None

        return None
