"""
Phase 3 Quiet-STaR Configuration

Defines configuration for the two-step Quiet-STaR process:
- Step 1 (Prompt Baking): Supervised learning configuration
- Step 2 (Quiet-STaR RL): REINFORCE RL configuration
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ThinkingTokens:
    """Special thinking tokens to add to vocabulary."""

    start_think: str = "<think>"
    end_think: str = "</think>"
    step: str = "<step>"
    reason: str = "<reason>"
    mece: str = "<mece>"
    falsify: str = "<falsify>"
    expert: str = "<expert>"
    doubt: str = "<doubt>"

    def to_list(self) -> List[str]:
        """Convert to list of tokens."""
        return [
            self.start_think,
            self.end_think,
            self.step,
            self.reason,
            self.mece,
            self.falsify,
            self.expert,
            self.doubt,
        ]


@dataclass
class ReasoningStrategies:
    """7 advanced reasoning strategies with example counts."""

    chain_of_thought: int = 400
    mece_decomposition: int = 200
    falsification_testing: int = 200
    expert_perspective: int = 200
    orthogonal_wisdom: int = 200
    self_doubt: int = 200
    bayesian_rationalist: int = 200

    @property
    def total_examples(self) -> int:
        """Total training examples."""
        return sum([
            self.chain_of_thought,
            self.mece_decomposition,
            self.falsification_testing,
            self.expert_perspective,
            self.orthogonal_wisdom,
            self.self_doubt,
            self.bayesian_rationalist,
        ])


@dataclass
class PromptBakingConfig:
    """Step 1: Prompt Baking configuration (Supervised Learning)."""

    # MuGrokfast optimizer (baking-specific)
    muon_lr: float = 1e-4
    grokfast_lambda: float = 0.2
    qk_clip_threshold: float = 30.0
    kl_coefficient: float = 0.0  # No KL reg in baking
    weight_decay: float = 0.01

    # Training parameters
    num_epochs: int = 5
    batch_size: int = 4
    convergence_threshold: float = 0.85  # ≥85% accuracy

    # LoRA adapter
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


@dataclass
class QuietSTaRRLConfig:
    """Step 2: Quiet-STaR RL configuration (REINFORCE)."""

    # MuGrokfast optimizer (RL-specific)
    muon_lr: float = 5e-4  # Higher for exploration
    grokfast_lambda: float = 0.1  # More filtering for RL
    qk_clip_threshold: float = 25.0  # Tighter clipping
    kl_coefficient: float = 0.1  # Prevent drift

    # Training parameters
    num_episodes: int = 10000
    reward_type: str = "reinforce"
    gradient_clip: float = 1.0

    # ISS-007: Full RL training parameters
    # Learning rate scheduling
    lr_schedule: str = "cosine"  # "constant", "cosine", "linear"
    warmup_episodes: int = 500  # Warmup before LR decay

    # Exploration (entropy bonus)
    entropy_coefficient: float = 0.01  # Weight for entropy bonus
    entropy_decay: float = 0.9995  # Decay entropy over training
    min_entropy_coefficient: float = 0.001  # Floor for entropy

    # Advantage estimation
    use_gae: bool = True  # Use Generalized Advantage Estimation
    gae_lambda: float = 0.95  # GAE lambda parameter
    gamma: float = 0.99  # Discount factor

    # Value function (baseline)
    value_loss_coefficient: float = 0.5  # Weight for value loss
    baseline_hidden_size: int = 256  # Baseline network hidden size

    # Early stopping
    patience: int = 10  # Stop if no improvement for N validations
    validation_frequency: int = 500  # Validate every N episodes
    min_improvement: float = 0.001  # Minimum reward improvement

    # Thought generation
    num_thoughts: int = 4
    max_thought_length: int = 20
    min_thought_length: int = 10
    temperature: float = 3.0  # M7: High temp for exploration
    top_p: float = 0.9

    # Coherence scoring weights
    coherence_weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic": 0.4,
        "syntactic": 0.3,
        "predictive": 0.3,
    })

    # Thought injection
    injection_threshold: float = 0.6
    min_injection_interval: int = 3


@dataclass
class AntiTheaterConfig:
    """Anti-theater validation thresholds."""

    divergence_threshold: float = 0.30
    ablation_threshold: float = 0.02  # 2% improvement
    correlation_threshold: float = 0.5

    # Testing frequency
    test_interval_steps: int = 1000


@dataclass
class QuietSTaRConfig:
    """Complete Phase 3 Quiet-STaR configuration."""

    # Thinking tokens
    thinking_tokens: ThinkingTokens = field(
        default_factory=ThinkingTokens
    )

    # Reasoning strategies
    strategies: ReasoningStrategies = field(
        default_factory=ReasoningStrategies
    )

    # Step 1: Prompt Baking
    baking: PromptBakingConfig = field(
        default_factory=PromptBakingConfig
    )

    # Step 2: Quiet-STaR RL
    rl: QuietSTaRRLConfig = field(default_factory=QuietSTaRRLConfig)

    # Anti-theater validation
    anti_theater: AntiTheaterConfig = field(
        default_factory=AntiTheaterConfig
    )

    # Data generation (OpenRouter)
    openrouter_models: List[str] = field(default_factory=lambda: [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-pro-1.5",
        "x-ai/grok-beta",
        "qwen/qwen-2.5-72b-instruct",
    ])
    examples_per_model: int = 4000
    cost_limit: float = 200.0  # USD

    # Performance targets
    target_accuracy_improvement: float = 0.07  # +7% (5-10%)
    target_inference_latency_ms: float = 200.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for W&B logging."""
        return {
            "thinking_tokens": self.thinking_tokens.to_list(),
            "num_strategies": 7,
            "total_examples": self.strategies.total_examples,
            "baking_muon_lr": self.baking.muon_lr,
            "baking_grokfast_lambda": self.baking.grokfast_lambda,
            "baking_convergence_threshold": self.baking.convergence_threshold,
            "rl_muon_lr": self.rl.muon_lr,
            "rl_grokfast_lambda": self.rl.grokfast_lambda,
            "rl_kl_coefficient": self.rl.kl_coefficient,
            "num_thoughts": self.rl.num_thoughts,
            "coherence_weights": self.rl.coherence_weights,
        }
