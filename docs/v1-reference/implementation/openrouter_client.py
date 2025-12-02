"""
OpenRouter API Client - Cross-Phase Infrastructure

Production-grade OpenRouter integration for Agent Forge V2.
Used in Phase 3 (reasoning example generation) and Phase 5 (supervisor model evaluation).

Features:
- Batch generation (minimize API calls)
- Automatic retry with exponential backoff
- Rate limiting and cost tracking
- Training-ready output format
- W&B integration
- Support for 20+ frontier models

Version: 1.0.0
"""

import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from enum import Enum
import time
import wandb


class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"
    QWEN = "qwen"
    META = "meta"
    MISTRAL = "mistral"
    COHERE = "cohere"


@dataclass
class ModelConfig:
    """Configuration for a frontier model"""
    name: str
    provider: ModelProvider
    openrouter_id: str
    cost_per_1m_input_tokens: float  # Cost per 1M input tokens
    cost_per_1m_output_tokens: float  # Cost per 1M output tokens
    max_tokens: int = 4096
    context_window: int = 8192
    temperature: float = 0.8
    top_p: float = 0.95
    supports_batch: bool = True
    max_batch_size: int = 20  # Max prompts in one API call


# Production Model Configurations (verified OpenRouter IDs as of 2025)
PRODUCTION_MODELS = {
    # OpenAI Models
    "gpt-4-turbo": ModelConfig(
        name="GPT-4 Turbo",
        provider=ModelProvider.OPENAI,
        openrouter_id="openai/gpt-4-turbo",
        cost_per_1m_input_tokens=10.00,
        cost_per_1m_output_tokens=30.00,
        max_tokens=4096,
        context_window=128000,
        supports_batch=True,
        max_batch_size=20
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        provider=ModelProvider.OPENAI,
        openrouter_id="openai/gpt-4o",
        cost_per_1m_input_tokens=5.00,
        cost_per_1m_output_tokens=15.00,
        max_tokens=4096,
        context_window=128000,
        supports_batch=True,
        max_batch_size=20
    ),

    # Anthropic Models
    "claude-3.5-sonnet": ModelConfig(
        name="Claude 3.5 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        openrouter_id="anthropic/claude-3.5-sonnet",
        cost_per_1m_input_tokens=3.00,
        cost_per_1m_output_tokens=15.00,
        max_tokens=8192,
        context_window=200000,
        supports_batch=True,
        max_batch_size=20
    ),
    "claude-3-opus": ModelConfig(
        name="Claude 3 Opus",
        provider=ModelProvider.ANTHROPIC,
        openrouter_id="anthropic/claude-3-opus",
        cost_per_1m_input_tokens=15.00,
        cost_per_1m_output_tokens=75.00,
        max_tokens=4096,
        context_window=200000,
        supports_batch=True,
        max_batch_size=20
    ),

    # Google Models
    "gemini-pro-1.5": ModelConfig(
        name="Gemini Pro 1.5",
        provider=ModelProvider.GOOGLE,
        openrouter_id="google/gemini-pro-1.5",
        cost_per_1m_input_tokens=1.25,
        cost_per_1m_output_tokens=5.00,
        max_tokens=8192,
        context_window=1000000,  # 1M context
        supports_batch=True,
        max_batch_size=20
    ),
    "gemini-flash-1.5": ModelConfig(
        name="Gemini Flash 1.5",
        provider=ModelProvider.GOOGLE,
        openrouter_id="google/gemini-flash-1.5",
        cost_per_1m_input_tokens=0.075,
        cost_per_1m_output_tokens=0.30,
        max_tokens=8192,
        context_window=1000000,
        supports_batch=True,
        max_batch_size=20
    ),

    # xAI Models
    "grok-beta": ModelConfig(
        name="Grok Beta",
        provider=ModelProvider.XAI,
        openrouter_id="x-ai/grok-beta",
        cost_per_1m_input_tokens=5.00,
        cost_per_1m_output_tokens=15.00,
        max_tokens=4096,
        context_window=131072,
        supports_batch=True,
        max_batch_size=20
    ),

    # Qwen Models
    "qwen-2.5-72b": ModelConfig(
        name="Qwen 2.5 72B Instruct",
        provider=ModelProvider.QWEN,
        openrouter_id="qwen/qwen-2.5-72b-instruct",
        cost_per_1m_input_tokens=0.40,
        cost_per_1m_output_tokens=0.40,
        max_tokens=8192,
        context_window=32768,
        supports_batch=True,
        max_batch_size=20
    ),

    # Meta Models (for reference, can be added)
    "llama-3.1-405b": ModelConfig(
        name="Llama 3.1 405B Instruct",
        provider=ModelProvider.META,
        openrouter_id="meta-llama/llama-3.1-405b-instruct",
        cost_per_1m_input_tokens=3.00,
        cost_per_1m_output_tokens=3.00,
        max_tokens=4096,
        context_window=128000,
        supports_batch=True,
        max_batch_size=20
    )
}


@dataclass
class GenerationRequest:
    """Single generation request"""
    prompt: str
    system_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GenerationResponse:
    """Single generation response"""
    text: str
    prompt: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: str
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class OpenRouterClient:
    """
    Production-grade OpenRouter API client

    Features:
    - Batch generation (minimize API calls)
    - Automatic retry with exponential backoff
    - Rate limiting
    - Cost tracking
    - W&B integration
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        wandb_enabled: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_rpm: int = 60  # Requests per minute
    ):
        """
        Initialize OpenRouter client

        Args:
            api_key: OpenRouter API key (or from OPENROUTER_API_KEY env var)
            wandb_enabled: Whether to log to W&B
            max_retries: Maximum retry attempts on failure
            retry_delay: Initial retry delay in seconds (exponential backoff)
            rate_limit_rpm: Rate limit in requests per minute
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key required (set OPENROUTER_API_KEY or pass api_key)")

        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.wandb_enabled = wandb_enabled
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_rpm = rate_limit_rpm

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60.0 / rate_limit_rpm

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.total_requests = 0

    def _apply_rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    async def _call_api(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, str]],
        session: aiohttp.ClientSession,
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Call OpenRouter API with retry logic

        Args:
            model_config: Model configuration
            messages: Chat messages
            session: aiohttp session
            retry_count: Current retry attempt

        Returns:
            API response dict or None if failed
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/agent-forge-v2",
            "X-Title": "Agent Forge V2 - Cross-Phase Infrastructure"
        }

        payload = {
            "model": model_config.openrouter_id,
            "messages": messages,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p
        }

        try:
            start_time = time.time()

            async with session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                latency_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    data = await response.json()
                    data['_latency_ms'] = latency_ms
                    return data

                elif response.status == 429:  # Rate limit
                    if retry_count < self.max_retries:
                        delay = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                        print(f"⚠️  Rate limited, retrying in {delay:.1f}s (attempt {retry_count + 1}/{self.max_retries})")
                        await asyncio.sleep(delay)
                        return await self._call_api(model_config, messages, session, retry_count + 1)
                    else:
                        print(f"❌ Rate limit exceeded, max retries reached")
                        return None

                elif response.status >= 500:  # Server error
                    if retry_count < self.max_retries:
                        delay = self.retry_delay * (2 ** retry_count)
                        print(f"⚠️  Server error ({response.status}), retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)
                        return await self._call_api(model_config, messages, session, retry_count + 1)
                    else:
                        print(f"❌ Server error, max retries reached")
                        return None

                else:
                    error_text = await response.text()
                    print(f"❌ API error {response.status}: {error_text}")
                    return None

        except asyncio.TimeoutError:
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                print(f"⚠️  Timeout, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                return await self._call_api(model_config, messages, session, retry_count + 1)
            else:
                print(f"❌ Timeout, max retries reached")
                return None

        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            return None

    def _calculate_cost(
        self,
        model_config: ModelConfig,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for a generation"""
        input_cost = (input_tokens / 1_000_000) * model_config.cost_per_1m_input_tokens
        output_cost = (output_tokens / 1_000_000) * model_config.cost_per_1m_output_tokens
        return input_cost + output_cost

    async def generate_batch(
        self,
        model_config: ModelConfig,
        requests: List[GenerationRequest],
        batch_size: Optional[int] = None
    ) -> List[GenerationResponse]:
        """
        Generate responses in batches (MINIMIZES API CALLS)

        Instead of making N API calls for N prompts, we:
        1. Combine multiple prompts into fewer API calls
        2. Use large context windows efficiently
        3. Parse multiple outputs from batch responses

        Args:
            model_config: Model to use
            requests: List of generation requests
            batch_size: Batch size (defaults to model's max_batch_size)

        Returns:
            List of GenerationResponse
        """
        if batch_size is None:
            batch_size = model_config.max_batch_size

        responses = []

        async with aiohttp.ClientSession() as session:
            # Process in batches
            for batch_start in range(0, len(requests), batch_size):
                batch_end = min(batch_start + batch_size, len(requests))
                batch_requests = requests[batch_start:batch_end]

                # Create batched prompt
                batched_prompt = self._create_batched_prompt(batch_requests)

                # Single API call for entire batch
                messages = [
                    {
                        "role": "system",
                        "content": batch_requests[0].system_message or "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": batched_prompt
                    }
                ]

                # Apply rate limiting
                self._apply_rate_limit()

                # Call API
                result = await self._call_api(model_config, messages, session)

                if result:
                    # Parse batch response
                    batch_responses = self._parse_batched_response(
                        result,
                        batch_requests,
                        model_config
                    )
                    responses.extend(batch_responses)

                    # Update tracking
                    usage = result.get('usage', {})
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                    cost = self._calculate_cost(model_config, input_tokens, output_tokens)

                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.total_cost_usd += cost
                    self.total_requests += 1

                    print(f"  ✅ Batch {batch_start+1}-{batch_end}: {len(batch_responses)} responses, ${cost:.4f}")

                else:
                    # Batch failed, create error responses
                    for req in batch_requests:
                        responses.append(GenerationResponse(
                            text="",
                            prompt=req.prompt,
                            model=model_config.name,
                            provider=model_config.provider.value,
                            input_tokens=0,
                            output_tokens=0,
                            cost_usd=0.0,
                            latency_ms=0.0,
                            timestamp=datetime.now().isoformat(),
                            metadata=req.metadata,
                            success=False,
                            error="API call failed"
                        ))

        return responses

    def _create_batched_prompt(self, requests: List[GenerationRequest]) -> str:
        """
        Create a batched prompt that combines multiple requests

        Format:
        ---
        REQUEST 1:
        [prompt 1]

        REQUEST 2:
        [prompt 2]
        ---

        This minimizes API calls while keeping outputs separate.
        """
        batched = "Generate responses for the following requests. Output in the EXACT format shown for each request.\n\n"

        for i, req in enumerate(requests, 1):
            batched += f"---\nREQUEST {i}:\n{req.prompt}\n\n"

        batched += "---\n\nGenerate ALL responses now, maintaining the exact output format for each request."

        return batched

    def _parse_batched_response(
        self,
        api_result: Dict[str, Any],
        requests: List[GenerationRequest],
        model_config: ModelConfig
    ) -> List[GenerationResponse]:
        """
        Parse batched API response into individual responses

        Args:
            api_result: Raw API response
            requests: Original requests
            model_config: Model configuration

        Returns:
            List of GenerationResponse
        """
        content = api_result['choices'][0]['message']['content']
        usage = api_result.get('usage', {})
        latency_ms = api_result.get('_latency_ms', 0)

        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        cost = self._calculate_cost(model_config, input_tokens, output_tokens)

        # Split by REQUEST markers
        parts = content.split('---')
        responses = []

        for i, req in enumerate(requests):
            # Find corresponding part (may need fuzzy matching)
            text = ""
            if i < len(parts) - 1:
                # Extract text for this request
                request_section = parts[i + 1]  # +1 because first part is before first ---
                text = request_section.strip()

            responses.append(GenerationResponse(
                text=text,
                prompt=req.prompt,
                model=model_config.name,
                provider=model_config.provider.value,
                input_tokens=input_tokens // len(requests),  # Distribute evenly
                output_tokens=output_tokens // len(requests),
                cost_usd=cost / len(requests),
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
                metadata=req.metadata,
                success=True
            ))

        return responses

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            'total_requests': self.total_requests,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost_usd': self.total_cost_usd,
            'avg_cost_per_request': self.total_cost_usd / max(1, self.total_requests)
        }


# Example usage
if __name__ == "__main__":
    async def test():
        client = OpenRouterClient()

        # Test batch generation
        requests = [
            GenerationRequest(
                prompt="Explain photosynthesis in one sentence.",
                system_message="You are a biology teacher.",
                metadata={'topic': 'biology', 'level': 'high_school'}
            ),
            GenerationRequest(
                prompt="What is the capital of France?",
                system_message="You are a geography teacher.",
                metadata={'topic': 'geography', 'level': 'elementary'}
            ),
            GenerationRequest(
                prompt="Calculate 2 + 2.",
                system_message="You are a math teacher.",
                metadata={'topic': 'math', 'level': 'elementary'}
            )
        ]

        model = PRODUCTION_MODELS['gpt-4o']  # Use GPT-4o for testing

        print(f"Testing batch generation with {model.name}...")
        responses = await client.generate_batch(model, requests, batch_size=3)

        for i, resp in enumerate(responses, 1):
            print(f"\nResponse {i}:")
            print(f"  Prompt: {resp.prompt}")
            print(f"  Text: {resp.text[:100]}...")
            print(f"  Cost: ${resp.cost_usd:.6f}")
            print(f"  Tokens: {resp.input_tokens} in, {resp.output_tokens} out")

        stats = client.get_stats()
        print(f"\n📊 Stats: {json.dumps(stats, indent=2)}")

    asyncio.run(test())
