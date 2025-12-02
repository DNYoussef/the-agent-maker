"""
Unit tests for Phase 3 configuration.

Tests all configuration dataclasses:
- ThinkingTokens
- ReasoningStrategies
- PromptBakingConfig
- QuietSTaRRLConfig
- AntiTheaterConfig
- QuietSTaRConfig

Target: ≥95% coverage
"""

import pytest
from src.phase3_quietstar.config import (
    ThinkingTokens,
    ReasoningStrategies,
    PromptBakingConfig,
    QuietSTaRRLConfig,
    AntiTheaterConfig,
    QuietSTaRConfig,
)


class TestThinkingTokens:
    """Test ThinkingTokens dataclass."""

    def test_default_values(self):
        """Test default token values."""
        tokens = ThinkingTokens()

        assert tokens.start_think == "<think>"
        assert tokens.end_think == "</think>"
        assert tokens.step == "<step>"
        assert tokens.reason == "<reason>"
        assert tokens.mece == "<mece>"
        assert tokens.falsify == "<falsify>"
        assert tokens.expert == "<expert>"
        assert tokens.doubt == "<doubt>"

    def test_to_list_returns_all_tokens(self):
        """Test to_list returns all 8 tokens."""
        tokens = ThinkingTokens()

        token_list = tokens.to_list()

        assert len(token_list) == 8
        assert "<think>" in token_list
        assert "</think>" in token_list
        assert "<step>" in token_list

    def test_to_list_preserves_order(self):
        """Test to_list preserves token order."""
        tokens = ThinkingTokens()

        token_list = tokens.to_list()

        expected_order = [
            "<think>",
            "</think>",
            "<step>",
            "<reason>",
            "<mece>",
            "<falsify>",
            "<expert>",
            "<doubt>",
        ]

        assert token_list == expected_order


class TestReasoningStrategies:
    """Test ReasoningStrategies dataclass."""

    def test_default_counts(self):
        """Test default example counts."""
        strategies = ReasoningStrategies()

        assert strategies.chain_of_thought == 400
        assert strategies.mece_decomposition == 200
        assert strategies.falsification_testing == 200
        assert strategies.expert_perspective == 200
        assert strategies.orthogonal_wisdom == 200
        assert strategies.self_doubt == 200
        assert strategies.bayesian_rationalist == 200

    def test_total_examples_correct(self):
        """Test total_examples property."""
        strategies = ReasoningStrategies()

        total = strategies.total_examples

        assert total == 1600  # 400 + 6*200

    def test_custom_counts(self):
        """Test custom example counts."""
        strategies = ReasoningStrategies(
            chain_of_thought=100,
            mece_decomposition=50,
            falsification_testing=50,
            expert_perspective=50,
            orthogonal_wisdom=50,
            self_doubt=50,
            bayesian_rationalist=50,
        )

        assert strategies.total_examples == 400


class TestPromptBakingConfig:
    """Test PromptBakingConfig dataclass."""

    def test_default_mugrokfast_settings(self):
        """Test default MuGrokfast optimizer settings."""
        config = PromptBakingConfig()

        assert config.muon_lr == 1e-4
        assert config.grokfast_lambda == 0.2
        assert config.qk_clip_threshold == 30.0
        assert config.kl_coefficient == 0.0  # No KL reg

    def test_default_training_params(self):
        """Test default training parameters."""
        config = PromptBakingConfig()

        assert config.num_epochs == 5
        assert config.batch_size == 4
        assert config.convergence_threshold == 0.85
        assert config.weight_decay == 0.01

    def test_default_lora_params(self):
        """Test default LoRA adapter parameters."""
        config = PromptBakingConfig()

        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = PromptBakingConfig(
            muon_lr=5e-4,
            num_epochs=10,
            convergence_threshold=0.9,
        )

        assert config.muon_lr == 5e-4
        assert config.num_epochs == 10
        assert config.convergence_threshold == 0.9


class TestQuietSTaRRLConfig:
    """Test QuietSTaRRLConfig dataclass."""

    def test_default_mugrokfast_settings(self):
        """Test default MuGrokfast optimizer settings for RL."""
        config = QuietSTaRRLConfig()

        assert config.muon_lr == 5e-4  # Higher than baking
        assert config.grokfast_lambda == 0.1  # Lower than baking
        assert config.qk_clip_threshold == 25.0  # Tighter than baking
        assert config.kl_coefficient == 0.1  # Prevent drift

    def test_default_training_params(self):
        """Test default RL training parameters."""
        config = QuietSTaRRLConfig()

        assert config.num_episodes == 10000
        assert config.reward_type == "reinforce"
        assert config.gradient_clip == 1.0

    def test_default_thought_generation_params(self):
        """Test default thought generation parameters."""
        config = QuietSTaRRLConfig()

        assert config.num_thoughts == 4
        assert config.max_thought_length == 20
        assert config.min_thought_length == 10
        assert config.temperature == 1.0
        assert config.top_p == 0.9

    def test_default_coherence_weights(self):
        """Test default coherence scoring weights."""
        config = QuietSTaRRLConfig()

        weights = config.coherence_weights

        assert weights["semantic"] == 0.4
        assert weights["syntactic"] == 0.3
        assert weights["predictive"] == 0.3
        assert sum(weights.values()) == 1.0  # Should sum to 1

    def test_default_thought_injection_params(self):
        """Test default thought injection parameters."""
        config = QuietSTaRRLConfig()

        assert config.injection_threshold == 0.6
        assert config.min_injection_interval == 3


class TestAntiTheaterConfig:
    """Test AntiTheaterConfig dataclass."""

    def test_default_thresholds(self):
        """Test default validation thresholds."""
        config = AntiTheaterConfig()

        assert config.divergence_threshold == 0.30
        assert config.ablation_threshold == 0.02
        assert config.correlation_threshold == 0.5

    def test_default_test_interval(self):
        """Test default testing interval."""
        config = AntiTheaterConfig()

        assert config.test_interval_steps == 1000

    def test_custom_thresholds(self):
        """Test custom thresholds."""
        config = AntiTheaterConfig(
            divergence_threshold=0.4,
            ablation_threshold=0.03,
            correlation_threshold=0.6,
        )

        assert config.divergence_threshold == 0.4
        assert config.ablation_threshold == 0.03
        assert config.correlation_threshold == 0.6


class TestQuietSTaRConfig:
    """Test QuietSTaRConfig master configuration."""

    def test_default_initialization(self):
        """Test default QuietSTaRConfig initialization."""
        config = QuietSTaRConfig()

        assert isinstance(config.thinking_tokens, ThinkingTokens)
        assert isinstance(config.strategies, ReasoningStrategies)
        assert isinstance(config.baking, PromptBakingConfig)
        assert isinstance(config.rl, QuietSTaRRLConfig)
        assert isinstance(config.anti_theater, AntiTheaterConfig)

    def test_default_openrouter_models(self):
        """Test default OpenRouter models."""
        config = QuietSTaRConfig()

        assert len(config.openrouter_models) == 5
        assert "openai/gpt-4o" in config.openrouter_models
        assert "anthropic/claude-3.5-sonnet" in config.openrouter_models
        assert "google/gemini-pro-1.5" in config.openrouter_models
        assert "x-ai/grok-beta" in config.openrouter_models
        assert "qwen/qwen-2.5-72b-instruct" in config.openrouter_models

    def test_default_data_generation_params(self):
        """Test default data generation parameters."""
        config = QuietSTaRConfig()

        assert config.examples_per_model == 4000
        assert config.cost_limit == 200.0

    def test_default_performance_targets(self):
        """Test default performance targets."""
        config = QuietSTaRConfig()

        assert config.target_accuracy_improvement == 0.07  # 7%
        assert config.target_inference_latency_ms == 200.0

    def test_to_dict_conversion(self):
        """Test to_dict conversion for W&B logging."""
        config = QuietSTaRConfig()

        config_dict = config.to_dict()

        # Check key fields present
        assert "thinking_tokens" in config_dict
        assert "num_strategies" in config_dict
        assert "total_examples" in config_dict
        assert "baking_muon_lr" in config_dict
        assert "rl_muon_lr" in config_dict
        assert "coherence_weights" in config_dict

    def test_to_dict_values_correct(self):
        """Test to_dict values are correct."""
        config = QuietSTaRConfig()

        config_dict = config.to_dict()

        assert config_dict["num_strategies"] == 7
        assert config_dict["total_examples"] == 1600
        assert config_dict["baking_muon_lr"] == 1e-4
        assert config_dict["rl_muon_lr"] == 5e-4
        assert config_dict["baking_grokfast_lambda"] == 0.2
        assert config_dict["rl_grokfast_lambda"] == 0.1

    def test_custom_nested_config(self):
        """Test custom nested configuration."""
        custom_baking = PromptBakingConfig(num_epochs=10)
        custom_rl = QuietSTaRRLConfig(num_episodes=20000)

        config = QuietSTaRConfig(
            baking=custom_baking,
            rl=custom_rl,
        )

        assert config.baking.num_epochs == 10
        assert config.rl.num_episodes == 20000

    def test_thinking_tokens_accessible(self):
        """Test thinking tokens accessible via config."""
        config = QuietSTaRConfig()

        tokens = config.thinking_tokens.to_list()

        assert len(tokens) == 8
        assert "<think>" in tokens

    def test_strategies_total_accessible(self):
        """Test strategies total accessible via config."""
        config = QuietSTaRConfig()

        total = config.strategies.total_examples

        assert total == 1600


class TestConfigConsistency:
    """Test configuration consistency and relationships."""

    def test_baking_vs_rl_optimizer_differences(self):
        """Test baking and RL configs have different optimizer settings."""
        config = QuietSTaRConfig()

        # RL should have higher LR for exploration
        assert config.rl.muon_lr > config.baking.muon_lr

        # RL should have lower lambda for more filtering
        assert config.rl.grokfast_lambda < config.baking.grokfast_lambda

        # RL should have tighter clipping
        assert config.rl.qk_clip_threshold < config.baking.qk_clip_threshold

        # Only RL should have KL coefficient
        assert config.baking.kl_coefficient == 0.0
        assert config.rl.kl_coefficient > 0.0

    def test_coherence_weights_sum_to_one(self):
        """Test coherence weights sum to 1.0."""
        config = QuietSTaRConfig()

        weights = config.rl.coherence_weights
        total_weight = sum(weights.values())

        assert abs(total_weight - 1.0) < 1e-6

    def test_cost_limit_reasonable(self):
        """Test cost limit is reasonable."""
        config = QuietSTaRConfig()

        # Should be between $100-$300
        assert 100.0 <= config.cost_limit <= 300.0

    def test_examples_per_model_sufficient(self):
        """Test examples per model is sufficient."""
        config = QuietSTaRConfig()

        # Should generate at least 1600 examples per model
        assert config.examples_per_model >= 1600

    def test_performance_targets_reasonable(self):
        """Test performance targets are reasonable."""
        config = QuietSTaRConfig()

        # Accuracy improvement should be 5-10%
        assert 0.05 <= config.target_accuracy_improvement <= 0.10

        # Latency should be <300ms
        assert config.target_inference_latency_ms <= 300.0


@pytest.mark.parametrize(
    "muon_lr,grokfast_lambda",
    [(1e-4, 0.2), (5e-4, 0.1), (1e-3, 0.05)],
)
def test_custom_optimizer_configs(muon_lr, grokfast_lambda):
    """Test custom optimizer configurations."""
    config = PromptBakingConfig(
        muon_lr=muon_lr, grokfast_lambda=grokfast_lambda
    )

    assert config.muon_lr == muon_lr
    assert config.grokfast_lambda == grokfast_lambda


@pytest.mark.parametrize("num_thoughts", [2, 4, 8])
def test_custom_num_thoughts(num_thoughts):
    """Test custom number of thoughts."""
    config = QuietSTaRRLConfig(num_thoughts=num_thoughts)

    assert config.num_thoughts == num_thoughts
