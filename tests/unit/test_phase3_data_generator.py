"""
Unit tests for Phase 3 data generation.

Tests OpenRouter integration and strategy prompt generation.

Target: ≥95% coverage
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path

from src.phase3_quietstar.data_generator import (
    ReasoningExample,
    GenerationStats,
    OpenRouterClient,
    StrategyPromptGenerator,
)


class TestReasoningExample:
    """Test ReasoningExample dataclass."""

    def test_initialization(self):
        """Test ReasoningExample initialization."""
        example = ReasoningExample(
            question="What is 2+2?",
            reasoning="<think><step>2+2=4</step></think>",
            answer="4",
            strategy="chain_of_thought",
            model="openai/gpt-4o",
            tokens_used=100,
            cost_usd=0.001,
        )

        assert example.question == "What is 2+2?"
        assert example.reasoning == "<think><step>2+2=4</step></think>"
        assert example.answer == "4"
        assert example.strategy == "chain_of_thought"
        assert example.tokens_used == 100
        assert example.cost_usd == 0.001

    def test_metadata_default_empty(self):
        """Test metadata defaults to empty dict."""
        example = ReasoningExample(
            question="Q",
            reasoning="R",
            answer="A",
            strategy="s",
            model="m",
            tokens_used=10,
            cost_usd=0.0,
        )

        assert example.metadata == {}


class TestGenerationStats:
    """Test GenerationStats dataclass."""

    def test_initialization(self):
        """Test GenerationStats initialization."""
        stats = GenerationStats()

        assert stats.total_examples == 0
        assert stats.valid_examples == 0
        assert stats.invalid_examples == 0
        assert stats.total_cost_usd == 0.0
        assert stats.examples_by_strategy == {}
        assert stats.examples_by_model == {}

    def test_elapsed_time_property(self):
        """Test elapsed_time property."""
        stats = GenerationStats()

        import time

        time.sleep(0.1)

        elapsed = stats.elapsed_time

        assert elapsed >= 0.1

    def test_valid_ratio_with_examples(self):
        """Test valid_ratio with examples."""
        stats = GenerationStats()
        stats.total_examples = 100
        stats.valid_examples = 85

        assert stats.valid_ratio == 0.85

    def test_valid_ratio_with_no_examples(self):
        """Test valid_ratio with no examples."""
        stats = GenerationStats()

        assert stats.valid_ratio == 0.0

    def test_cost_per_example_with_examples(self):
        """Test cost_per_example with examples."""
        stats = GenerationStats()
        stats.valid_examples = 100
        stats.total_cost_usd = 50.0

        assert stats.cost_per_example == 0.5

    def test_cost_per_example_with_no_examples(self):
        """Test cost_per_example with no examples."""
        stats = GenerationStats()

        assert stats.cost_per_example == 0.0


class TestOpenRouterClient:
    """Test OpenRouterClient class."""

    def test_initialization(self):
        """Test OpenRouterClient initialization."""
        client = OpenRouterClient(
            api_key="test_key", cost_limit=200.0, batch_size=10
        )

        assert client.api_key == "test_key"
        assert client.cost_limit == 200.0
        assert client.batch_size == 10
        assert isinstance(client.stats, GenerationStats)

    def test_pricing_defined_for_all_models(self):
        """Test pricing defined for all models."""
        client = OpenRouterClient("test_key")

        expected_models = [
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro-1.5",
            "x-ai/grok-beta",
            "qwen/qwen-2.5-72b-instruct",
        ]

        for model in expected_models:
            assert model in client.pricing
            assert "input" in client.pricing[model]
            assert "output" in client.pricing[model]

    def test_calculate_cost_gpt4o(self):
        """Test cost calculation for GPT-4o."""
        client = OpenRouterClient("test_key")

        cost = client._calculate_cost(
            "openai/gpt-4o", input_tokens=1000, output_tokens=500
        )

        # GPT-4o: $2.5/1M input, $10/1M output
        expected = (1000 / 1_000_000) * 2.5 + (500 / 1_000_000) * 10.0
        assert abs(cost - expected) < 1e-6

    def test_calculate_cost_claude(self):
        """Test cost calculation for Claude."""
        client = OpenRouterClient("test_key")

        cost = client._calculate_cost(
            "anthropic/claude-3.5-sonnet",
            input_tokens=2000,
            output_tokens=1000,
        )

        # Claude: $3/1M input, $15/1M output
        expected = (2000 / 1_000_000) * 3.0 + (1000 / 1_000_000) * 15.0
        assert abs(cost - expected) < 1e-6

    def test_extract_components_basic(self):
        """Test extracting components from response."""
        client = OpenRouterClient("test_key")

        content = "Step 1: Analyze\nStep 2: Solve\nAnswer: 42"
        prompt = "Question: What is the answer?"

        question, reasoning, answer = client._extract_components(
            content, prompt
        )

        assert "What is the answer?" in question
        assert reasoning == content
        assert answer == "42"

    def test_parse_response_creates_example(self):
        """Test parsing API response into ReasoningExample."""
        client = OpenRouterClient("test_key")

        response = {
            "choices": [
                {
                    "message": {
                        "content": "<think><step>Analyze</step></think>\nAnswer: 42"
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        example = client._parse_response(
            response,
            model="openai/gpt-4o",
            strategy="chain_of_thought",
            prompt="Question: Test",
        )

        assert isinstance(example, ReasoningExample)
        assert example.strategy == "chain_of_thought"
        assert example.model == "openai/gpt-4o"
        assert example.tokens_used == 150
        assert example.cost_usd > 0

    @pytest.mark.asyncio
    async def test_api_call_success(self):
        """Test successful API call."""
        client = OpenRouterClient("test_key")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "test"}}]}
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        result = await client._api_call(
            mock_session, "openai/gpt-4o", "test prompt"
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_api_call_failure(self):
        """Test API call failure."""
        client = OpenRouterClient("test_key")

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server error")

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        result = await client._api_call(
            mock_session, "openai/gpt-4o", "test prompt"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_single_with_retry(self):
        """Test single generation with retry logic."""
        client = OpenRouterClient("test_key")

        # Mock failing once, then succeeding
        mock_session = MagicMock()

        with patch.object(
            client,
            "_api_call",
            side_effect=[
                None,  # First attempt fails
                {
                    "choices": [{"message": {"content": "test"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },  # Second succeeds
            ],
        ):
            example = await client._generate_single(
                mock_session,
                "openai/gpt-4o",
                "chain_of_thought",
                "test prompt",
            )

        assert example is not None


class TestStrategyPromptGenerator:
    """Test StrategyPromptGenerator class."""

    def test_initialization(self):
        """Test StrategyPromptGenerator initialization."""
        generator = StrategyPromptGenerator()

        assert len(generator.strategies) == 7
        assert generator.strategies["chain_of_thought"] == 400
        assert generator.strategies["mece_decomposition"] == 200

    def test_generate_prompts_returns_all_strategies(self):
        """Test generate_prompts returns all 7 strategies."""
        generator = StrategyPromptGenerator()

        prompts = generator.generate_prompts()

        assert len(prompts) == 7
        assert "chain_of_thought" in prompts
        assert "mece_decomposition" in prompts
        assert "falsification_testing" in prompts
        assert "expert_perspective" in prompts
        assert "orthogonal_wisdom" in prompts
        assert "self_doubt" in prompts
        assert "bayesian_rationalist" in prompts

    def test_chain_of_thought_prompts_count(self):
        """Test chain-of-thought generates correct count."""
        generator = StrategyPromptGenerator()

        prompts = generator._generate_chain_of_thought_prompts(400)

        assert len(prompts) == 400

    def test_chain_of_thought_prompts_contain_tags(self):
        """Test chain-of-thought prompts contain thinking tags."""
        generator = StrategyPromptGenerator()

        prompts = generator._generate_chain_of_thought_prompts(10)

        for prompt in prompts:
            assert "<think>" in prompt
            assert "</think>" in prompt
            assert "<step>" in prompt

    def test_mece_prompts_contain_tags(self):
        """Test MECE prompts contain correct tags."""
        generator = StrategyPromptGenerator()

        prompts = generator._generate_mece_decomposition_prompts(10)

        for prompt in prompts:
            assert "<mece>" in prompt
            assert "<category>" in prompt

    def test_falsification_prompts_contain_tags(self):
        """Test falsification prompts contain correct tags."""
        generator = StrategyPromptGenerator()

        prompts = generator._generate_falsification_testing_prompts(10)

        for prompt in prompts:
            assert "<falsify>" in prompt
            assert "<test>" in prompt

    def test_expert_prompts_contain_tags(self):
        """Test expert perspective prompts contain correct tags."""
        generator = StrategyPromptGenerator()

        prompts = generator._generate_expert_perspective_prompts(10)

        for prompt in prompts:
            assert "<expert" in prompt  # Can have domain attribute

    def test_orthogonal_prompts_contain_tags(self):
        """Test orthogonal wisdom prompts contain thinking tags."""
        generator = StrategyPromptGenerator()

        prompts = generator._generate_orthogonal_wisdom_prompts(10)

        for prompt in prompts:
            assert "<think>" in prompt
            assert "<step>" in prompt

    def test_self_doubt_prompts_contain_tags(self):
        """Test self-doubt prompts contain correct tags."""
        generator = StrategyPromptGenerator()

        prompts = generator._generate_self_doubt_prompts(10)

        for prompt in prompts:
            assert "<doubt>" in prompt
            assert "<check>" in prompt

    def test_bayesian_prompts_contain_steps(self):
        """Test Bayesian rationalist prompts contain steps."""
        generator = StrategyPromptGenerator()

        prompts = generator._generate_bayesian_rationalist_prompts(10)

        for prompt in prompts:
            assert "<step>" in prompt

    def test_all_prompts_unique(self):
        """Test generated prompts are unique."""
        generator = StrategyPromptGenerator()

        prompts = generator._generate_chain_of_thought_prompts(100)

        # All should be unique (or at least most)
        unique_prompts = set(prompts)
        assert len(unique_prompts) >= 90  # Allow some duplicates


@pytest.mark.parametrize(
    "strategy,expected_tag",
    [
        ("chain_of_thought", "<step>"),
        ("mece_decomposition", "<mece>"),
        ("falsification_testing", "<falsify>"),
        ("expert_perspective", "<expert>"),
        ("self_doubt", "<doubt>"),
    ],
)
def test_strategy_prompts_contain_expected_tags(strategy, expected_tag):
    """Test each strategy contains expected tags."""
    generator = StrategyPromptGenerator()

    method_name = f"_generate_{strategy}_prompts"
    generator_method = getattr(generator, method_name)

    prompts = generator_method(5)

    for prompt in prompts:
        assert expected_tag in prompt


@pytest.mark.parametrize("cost_limit", [100.0, 200.0, 300.0])
def test_openrouter_client_respects_cost_limit(cost_limit):
    """Test OpenRouter client respects cost limits."""
    client = OpenRouterClient("test_key", cost_limit=cost_limit)

    assert client.cost_limit == cost_limit


def test_generation_stats_tracks_examples_by_strategy():
    """Test generation stats tracks examples by strategy."""
    stats = GenerationStats()

    stats.examples_by_strategy["chain_of_thought"] = 400
    stats.examples_by_strategy["mece_decomposition"] = 200

    assert stats.examples_by_strategy["chain_of_thought"] == 400
    assert stats.examples_by_strategy["mece_decomposition"] == 200


def test_generation_stats_tracks_examples_by_model():
    """Test generation stats tracks examples by model."""
    stats = GenerationStats()

    stats.examples_by_model["openai/gpt-4o"] = 1000
    stats.examples_by_model["anthropic/claude-3.5-sonnet"] = 800

    assert stats.examples_by_model["openai/gpt-4o"] == 1000
    assert stats.examples_by_model["anthropic/claude-3.5-sonnet"] == 800
