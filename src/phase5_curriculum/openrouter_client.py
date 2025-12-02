"""
OpenRouter API Client for Phase 5 Curriculum Learning

Production-grade async client with FREE models for testing.
Uses httpx for async HTTP requests.

Features:
- 4 frontier models (FREE tier for testing)
- Async request handling with retry logic
- Cost tracking and rate limiting
- Exponential backoff on failures

Usage:
    async with OpenRouterClient() as client:
        response = await client.complete(
            prompt="Explain recursion",
            model=ModelProvider.QWEN_FREE
        )
        print(response.content)
        print(f"Cost: ${response.cost_usd:.6f}")
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """
    Supported models via OpenRouter.

    FREE TIER (for testing - $0 cost):
    - QWEN_FREE: qwen/qwen-2-7b-instruct:free
    - GEMMA_FREE: google/gemma-7b-it:free
    - MISTRAL_FREE: mistralai/mistral-7b-instruct:free
    - LLAMA_FREE: meta-llama/llama-3-8b-instruct:free

    PAID TIER (for production - latest frontier models):
    - GEMINI_3: google/gemini-3
    - GPT_51_DEEP_THINKING: openai/gpt-5.1-deep-thinking
    - CLAUDE_45_OPUS: anthropic/claude-4.5-opus
    """

    # FREE models for testing (no cost)
    QWEN_FREE = "qwen/qwen-2-7b-instruct:free"
    GEMMA_FREE = "google/gemma-7b-it:free"
    MISTRAL_FREE = "mistralai/mistral-7b-instruct:free"
    LLAMA_FREE = "meta-llama/llama-3-8b-instruct:free"

    # Paid models for production (latest frontier models)
    GEMINI_3 = "google/gemini-3"
    GPT_51_DEEP_THINKING = "openai/gpt-5.1-deep-thinking"
    CLAUDE_45_OPUS = "anthropic/claude-4.5-opus"


@dataclass
class ModelConfig:
    """Configuration for a frontier model."""

    name: str
    model_id: str
    cost_per_1m_input_tokens: float
    cost_per_1m_output_tokens: float
    max_tokens: int = 4096
    context_window: int = 8192
    is_free: bool = False


# Model configurations with pricing
MODEL_CONFIGS: Dict[ModelProvider, ModelConfig] = {
    # FREE models (for testing)
    ModelProvider.QWEN_FREE: ModelConfig(
        name="Qwen 2 7B Instruct (Free)",
        model_id="qwen/qwen-2-7b-instruct:free",
        cost_per_1m_input_tokens=0.0,
        cost_per_1m_output_tokens=0.0,
        max_tokens=4096,
        context_window=32768,
        is_free=True,
    ),
    ModelProvider.GEMMA_FREE: ModelConfig(
        name="Gemma 7B IT (Free)",
        model_id="google/gemma-7b-it:free",
        cost_per_1m_input_tokens=0.0,
        cost_per_1m_output_tokens=0.0,
        max_tokens=4096,
        context_window=8192,
        is_free=True,
    ),
    ModelProvider.MISTRAL_FREE: ModelConfig(
        name="Mistral 7B Instruct (Free)",
        model_id="mistralai/mistral-7b-instruct:free",
        cost_per_1m_input_tokens=0.0,
        cost_per_1m_output_tokens=0.0,
        max_tokens=4096,
        context_window=32768,
        is_free=True,
    ),
    ModelProvider.LLAMA_FREE: ModelConfig(
        name="Llama 3 8B Instruct (Free)",
        model_id="meta-llama/llama-3-8b-instruct:free",
        cost_per_1m_input_tokens=0.0,
        cost_per_1m_output_tokens=0.0,
        max_tokens=4096,
        context_window=8192,
        is_free=True,
    ),
    # Paid models (for production - latest frontier models)
    ModelProvider.GEMINI_3: ModelConfig(
        name="Gemini 3",
        model_id="google/gemini-3",
        cost_per_1m_input_tokens=5.00,
        cost_per_1m_output_tokens=15.00,
        max_tokens=32768,
        context_window=2000000,  # 2M context
    ),
    ModelProvider.GPT_51_DEEP_THINKING: ModelConfig(
        name="GPT-5.1 Deep Thinking",
        model_id="openai/gpt-5.1-deep-thinking",
        cost_per_1m_input_tokens=10.00,
        cost_per_1m_output_tokens=30.00,
        max_tokens=32768,
        context_window=256000,
    ),
    ModelProvider.CLAUDE_45_OPUS: ModelConfig(
        name="Claude 4.5 Opus",
        model_id="anthropic/claude-4.5-opus",
        cost_per_1m_input_tokens=15.00,
        cost_per_1m_output_tokens=75.00,
        max_tokens=32768,
        context_window=500000,
    ),
}


@dataclass
class CompletionResponse:
    """Structured response from model completion."""

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class OpenRouterClient:
    """
    Async client for OpenRouter API with FREE model support.

    Usage:
        async with OpenRouterClient() as client:
            # Use free model for testing (no cost)
            response = await client.complete(
                prompt="Explain quantum computing",
                model=ModelProvider.QWEN_FREE
            )
            print(response.content)

            # Check total cost
            print(f"Total cost: ${client.total_cost:.4f}")
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: ModelProvider = ModelProvider.QWEN_FREE,
        timeout: float = 120.0,
        max_retries: int = 3,
        rate_limit_rpm: int = 60,
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (or OPENROUTER_API_KEY env var)
            default_model: Default model to use (FREE by default)
            timeout: Request timeout in seconds
            max_retries: Max retry attempts on failure
            rate_limit_rpm: Rate limit in requests per minute
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY required. Set env var or pass api_key parameter.")

        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_rpm = rate_limit_rpm

        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 60.0 / rate_limit_rpm

        # Cost tracking
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._request_count = 0

        # HTTP client (created on __aenter__)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "OpenRouterClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def _estimate_cost(self, model: ModelProvider, usage: Dict[str, int]) -> float:
        """Estimate cost based on token usage."""
        config = MODEL_CONFIGS.get(model)
        if not config:
            return 0.0

        if config.is_free:
            return 0.0

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        input_cost = (input_tokens / 1_000_000) * config.cost_per_1m_input_tokens
        output_cost = (output_tokens / 1_000_000) * config.cost_per_1m_output_tokens

        return input_cost + output_cost

    async def complete(
        self,
        prompt: str,
        model: Optional[ModelProvider] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> CompletionResponse:
        """
        Generate a completion from the specified model.

        Args:
            prompt: User prompt
            model: Model to use (defaults to free model)
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            CompletionResponse with content and metadata
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        start_time = time.perf_counter()
        model = model or self.default_model
        config = MODEL_CONFIGS.get(model)

        if not config:
            return CompletionResponse(
                content="", model=model.value, success=False, error=f"Unknown model: {model}"
            )

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": config.model_id,
            "messages": messages,
            "max_tokens": min(max_tokens, config.max_tokens),
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://agent-maker.local",
            "X-Title": "Agent Maker Phase 5",
        }

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self._apply_rate_limit()

                response = await self._client.post(self.BASE_URL, json=payload, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    usage = data.get("usage", {})
                    cost = self._estimate_cost(model, usage)

                    # Update tracking
                    self._total_cost += cost
                    self._total_input_tokens += usage.get("prompt_tokens", 0)
                    self._total_output_tokens += usage.get("completion_tokens", 0)
                    self._request_count += 1

                    content = ""
                    if data.get("choices"):
                        content = data["choices"][0].get("message", {}).get("content", "")

                    return CompletionResponse(
                        content=content,
                        model=config.model_id,
                        usage=usage,
                        cost_usd=cost,
                        latency_ms=latency_ms,
                        success=True,
                    )

                elif response.status_code == 429:
                    # Rate limited - exponential backoff
                    delay = 2**attempt
                    logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    last_error = f"Rate limited (429)"

                elif response.status_code >= 500:
                    # Server error - retry
                    delay = 2**attempt
                    logger.warning(f"Server error {response.status_code}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    last_error = f"Server error ({response.status_code})"

                else:
                    # Client error - don't retry
                    error_text = response.text
                    return CompletionResponse(
                        content="",
                        model=config.model_id,
                        success=False,
                        error=f"API error {response.status_code}: {error_text[:200]}",
                    )

            except httpx.TimeoutException:
                delay = 2**attempt
                logger.warning(f"Timeout, retrying in {delay}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
                last_error = "Request timeout"

            except Exception as e:
                last_error = str(e)
                logger.error(f"Request failed: {e}")
                break

        # All retries exhausted
        return CompletionResponse(
            content="",
            model=model.value,
            success=False,
            error=f"Max retries exceeded: {last_error}",
        )

    async def complete_batch(
        self,
        prompts: List[str],
        model: Optional[ModelProvider] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        concurrency: int = 5,
    ) -> List[CompletionResponse]:
        """
        Generate completions for multiple prompts with controlled concurrency.

        Args:
            prompts: List of prompts
            model: Model to use
            system_prompt: Optional system prompt
            max_tokens: Max tokens per response
            temperature: Sampling temperature
            concurrency: Max concurrent requests

        Returns:
            List of CompletionResponse objects
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_complete(prompt: str) -> CompletionResponse:
            async with semaphore:
                return await self.complete(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        tasks = [limited_complete(p) for p in prompts]
        return await asyncio.gather(*tasks)

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens used."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens used."""
        return self._total_output_tokens

    @property
    def request_count(self) -> int:
        """Total number of requests made."""
        return self._request_count

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._request_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": self._total_cost,
            "avg_cost_per_request": (
                self._total_cost / self._request_count if self._request_count > 0 else 0.0
            ),
        }


# Convenience function for synchronous usage
def get_free_models() -> List[ModelProvider]:
    """Get list of FREE models for testing."""
    return [
        ModelProvider.QWEN_FREE,
        ModelProvider.GEMMA_FREE,
        ModelProvider.MISTRAL_FREE,
        ModelProvider.LLAMA_FREE,
    ]


def get_production_models() -> List[ModelProvider]:
    """Get list of production (paid) models - latest frontier models."""
    return [
        ModelProvider.GEMINI_3,
        ModelProvider.GPT_51_DEEP_THINKING,
        ModelProvider.CLAUDE_45_OPUS,
    ]


__all__ = [
    "OpenRouterClient",
    "ModelProvider",
    "ModelConfig",
    "CompletionResponse",
    "MODEL_CONFIGS",
    "get_free_models",
    "get_production_models",
]
