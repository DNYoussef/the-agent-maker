"""
Phase 1 Model Configuration Dataclasses
TRM × Titans-MAG Architecture Configuration

This module provides dataclass-based configuration for the complete Phase 1
architecture including:
- TitansMAG backbone (transformer with SW-Attn, LMM memory)
- MAG gate (memory-augmented gating)
- TRM wrapper (recursive refinement)
- ACT head (adaptive computation time)

Author: Agent Forge V2 Team
Version: 2.0
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple
import yaml
from pathlib import Path


@dataclass
class AttentionConfig:
    """Attention mechanism configuration."""

    type: Literal["sliding_window", "sliding_window_with_global"] = "sliding_window"
    """Attention type: sliding_window or hybrid with global tokens."""

    sw_window: int = 1024
    """Sliding window size (tokens can attend to ±window/2)."""

    global_tokens: Optional[int] = None
    """Number of global tokens (for hybrid mode). Every Nth token attends globally."""

    dropout: float = 0.1
    """Attention dropout probability."""


@dataclass
class MLPConfig:
    """MLP configuration."""

    type: Literal["swiglu", "gelu"] = "swiglu"
    """MLP activation type."""

    expansion_factor: float = 4.0
    """Hidden size = expansion_factor × d_model."""

    dropout: float = 0.1
    """MLP dropout probability."""


@dataclass
class MemoryConfig:
    """LMM (Long-Range Memory Module) configuration."""

    type: Literal["factorized", "full"] = "factorized"
    """Memory type: factorized (compressed) or full."""

    d_mem: int = 256
    """Memory dimension (typically d_model / 2 for factorized)."""

    decay: float = 0.99
    """Exponential decay for temporal weighting: m_t = decay*m_{t-1} + (1-decay)*x_t."""

    init: Literal["zeros", "learned"] = "zeros"
    """Memory initialization strategy."""


@dataclass
class NormConfig:
    """Normalization configuration."""

    type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    """Normalization type."""

    eps: float = 1e-6
    """Epsilon for numerical stability."""


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""

    tie_weights: bool = True
    """Tie input and output embedding weights."""

    dropout: float = 0.1
    """Embedding dropout probability."""


@dataclass
class TitansMAGConfig:
    """TitansMAG backbone configuration (8-layer transformer with memory)."""

    # Core transformer settings
    d_model: int = 512
    """Hidden dimension."""

    n_layers: int = 8
    """Number of transformer layers."""

    n_heads: int = 8
    """Number of attention heads."""

    head_dim: int = 64
    """Dimension per head (should be d_model / n_heads)."""

    vocab_size: int = 32768
    """Vocabulary size."""

    max_seq_len: int = 2048
    """Maximum sequence length."""

    # Component configs
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

        assert self.head_dim == self.d_model // self.n_heads, \
            f"head_dim ({self.head_dim}) must equal d_model / n_heads ({self.d_model // self.n_heads})"

        # Warn if parameter budget likely exceeded
        estimated_params = self._estimate_params()
        if estimated_params > 26e6:
            import warnings
            warnings.warn(
                f"Estimated params: {estimated_params/1e6:.2f}M (target: 25±1M). "
                "Consider reducing d_model, n_layers, or d_mem."
            )

    def _estimate_params(self) -> int:
        """Estimate total parameter count."""
        # Embeddings (tied with output)
        embed_params = self.vocab_size * self.d_model

        # Attention per layer
        attn_params = 4 * self.d_model * self.d_model  # Q, K, V, O projections

        # MLP per layer
        mlp_hidden = int(self.d_model * self.mlp.expansion_factor)
        mlp_params = (
            self.d_model * mlp_hidden +  # w_gate
            self.d_model * mlp_hidden +  # w_up
            mlp_hidden * self.d_model    # w_down
        )

        # Memory per layer (factorized)
        mem_params = (
            self.d_model * self.memory.d_mem +  # down projection
            self.memory.d_mem * self.d_model    # up projection
        )

        # Total per layer
        layer_params = attn_params + mlp_params + mem_params

        # Total model
        total = embed_params + (layer_params * self.n_layers)

        return total


@dataclass
class MAGGateConfig:
    """MAG (Memory-Augmented Gate) configuration."""

    hidden: int = 256
    """Hidden layer size for gating network."""

    entropy_reg: float = 0.001
    """Entropy regularization coefficient to prevent saturation."""

    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden > 0, "hidden must be positive"
        assert 0 <= self.entropy_reg <= 1.0, "entropy_reg must be in [0, 1]"


@dataclass
class TRMConfig:
    """TRM (Transformer Recursive Memory) wrapper configuration."""

    T_max: int = 3
    """Maximum recursion depth (starting value)."""

    micro_steps: int = 2
    """Number of micro-refinement steps in g_φ."""

    deep_supervision: bool = True
    """Compute loss at each recursion step."""

    detach_between_steps: bool = True
    """Detach y, z between recursion steps for memory efficiency."""

    step_weights: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    """Weight for each recursion step's loss (earlier steps weighted lower)."""

    # Curriculum settings
    curriculum_enabled: bool = True
    """Enable T_max curriculum (increase depth after plateau)."""

    plateau_steps: int = 20000
    """Increase T_max after this many steps."""

    T_max_final: int = 6
    """Final recursion depth after curriculum."""

    # Phase 2 compatibility
    stateless_mode: bool = False
    """Stateless mode for Phase 2 mergeability (no z/y carry-over)."""

    def __post_init__(self):
        """Validate configuration."""
        assert self.T_max > 0, "T_max must be positive"
        assert self.micro_steps > 0, "micro_steps must be positive"
        assert len(self.step_weights) >= self.T_max, \
            f"step_weights length ({len(self.step_weights)}) must be >= T_max ({self.T_max})"


@dataclass
class ACTConfig:
    """ACT (Adaptive Computation Time) head configuration."""

    halt_thresh: float = 0.5
    """Halt probability threshold (halt if q > halt_thresh)."""

    ema_teacher: float = 0.98
    """EMA decay for tracking step accuracies."""

    entropy_reg: float = 0.001
    """Entropy regularization to prevent q → 0.5 saturation."""

    warmup_steps: int = 5000
    """Don't enforce EMA targets during warmup."""

    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.halt_thresh < 1, "halt_thresh must be in (0, 1)"
        assert 0 < self.ema_teacher < 1, "ema_teacher must be in (0, 1)"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"


@dataclass
class Phase1ModelConfig:
    """
    Complete Phase 1 model configuration.

    Combines TitansMAG backbone, MAG gate, TRM wrapper, and ACT head.
    """

    titans_mag: TitansMAGConfig = field(default_factory=TitansMAGConfig)
    mag_gate: MAGGateConfig = field(default_factory=MAGGateConfig)
    trm: TRMConfig = field(default_factory=TRMConfig)
    act: ACTConfig = field(default_factory=ACTConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Phase1ModelConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Phase1ModelConfig instance

        Example:
            >>> config = Phase1ModelConfig.from_yaml("configs/phase1_config.yaml")
        """
        with open(yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)

        # Extract model config section
        model_dict = yaml_dict.get('model', {})
        gate_dict = yaml_dict.get('gate', {})
        trm_dict = yaml_dict.get('trm', {})
        act_dict = yaml_dict.get('act', {})

        # Build nested configs
        titans_config = TitansMAGConfig(
            d_model=model_dict.get('d_model', 512),
            n_layers=model_dict.get('n_layers', 8),
            n_heads=model_dict.get('n_heads', 8),
            head_dim=model_dict.get('head_dim', 64),
            vocab_size=model_dict.get('vocab_size', 32768),
            max_seq_len=model_dict.get('max_seq_len', 2048),
            attention=AttentionConfig(**model_dict.get('attention', {})),
            mlp=MLPConfig(**model_dict.get('mlp', {})),
            memory=MemoryConfig(**model_dict.get('memory', {})),
            norm=NormConfig(**model_dict.get('norm', {})),
            embedding=EmbeddingConfig(**model_dict.get('embedding', {}))
        )

        gate_config = MAGGateConfig(**gate_dict)

        trm_config = TRMConfig(
            T_max=trm_dict.get('T_max', 3),
            micro_steps=trm_dict.get('micro_steps', 2),
            deep_supervision=trm_dict.get('deep_supervision', True),
            detach_between_steps=trm_dict.get('detach_between_steps', True),
            step_weights=trm_dict.get('step_weights', [0.5, 0.75, 1.0]),
            curriculum_enabled=trm_dict.get('curriculum', {}).get('enabled', True),
            plateau_steps=trm_dict.get('curriculum', {}).get('plateau_steps', 20000),
            T_max_final=trm_dict.get('curriculum', {}).get('T_max_final', 6),
            stateless_mode=trm_dict.get('stateless_mode', False)
        )

        act_config = ACTConfig(**act_dict)

        return cls(
            titans_mag=titans_config,
            mag_gate=gate_config,
            trm=trm_config,
            act=act_config
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary (for serialization)."""
        from dataclasses import asdict
        return asdict(self)

    def estimate_params(self) -> int:
        """
        Estimate total parameter count.

        Returns:
            Total parameters (target: 25±1M)
        """
        # TitansMAG backbone
        backbone_params = self.titans_mag._estimate_params()

        # MAG gate: (2*d_model × hidden) + (hidden × d_model)
        d_model = self.titans_mag.d_model
        gate_params = (
            2 * d_model * self.mag_gate.hidden +  # w_concat
            self.mag_gate.hidden * d_model        # w_gate
        )

        # TRM wrapper: g_φ cross-attention + h_ψ update
        # Rough estimate: 1.5M params
        trm_params = 1_500_000

        # ACT head: d_model × 1
        act_params = d_model

        total = backbone_params + gate_params + trm_params + act_params

        return total

    def check_budget(self, target: float = 25e6, tolerance: float = 1e6):
        """
        Check if configuration fits within parameter budget.

        Args:
            target: Target parameter count (default: 25M)
            tolerance: Allowed deviation (default: ±1M)

        Raises:
            AssertionError: If parameter count exceeds budget
        """
        total = self.estimate_params()
        assert abs(total - target) <= tolerance, \
            f"Parameter budget exceeded: {total/1e6:.2f}M (target: {target/1e6:.2f}M ± {tolerance/1e6:.2f}M)"

    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = [
            "Phase1ModelConfig(",
            "  TitansMAG:",
            f"    d_model={self.titans_mag.d_model}, n_layers={self.titans_mag.n_layers}",
            f"    attention={self.titans_mag.attention.type}, sw_window={self.titans_mag.attention.sw_window}",
            f"    memory: d_mem={self.titans_mag.memory.d_mem}, decay={self.titans_mag.memory.decay}",
            "  MAG Gate:",
            f"    hidden={self.mag_gate.hidden}, entropy_reg={self.mag_gate.entropy_reg}",
            "  TRM:",
            f"    T_max={self.trm.T_max}, micro_steps={self.trm.micro_steps}",
            f"    deep_supervision={self.trm.deep_supervision}, detach={self.trm.detach_between_steps}",
            "  ACT:",
            f"    halt_thresh={self.act.halt_thresh}, ema_teacher={self.act.ema_teacher}",
            f"  Estimated params: {self.estimate_params()/1e6:.2f}M",
            ")"
        ]
        return "\n".join(lines)


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_default_config() -> Phase1ModelConfig:
    """
    Get default Phase 1 configuration (25M params, GTX 1660 optimized).

    Returns:
        Phase1ModelConfig with default settings
    """
    return Phase1ModelConfig()


def get_reasoning_config() -> Phase1ModelConfig:
    """
    Get reasoning-optimized configuration (more recursion, deeper supervision).

    Returns:
        Phase1ModelConfig optimized for reasoning tasks
    """
    config = Phase1ModelConfig()
    config.trm.T_max = 6  # Start with deeper recursion
    config.trm.micro_steps = 3  # More refinement steps
    config.act.halt_thresh = 0.6  # Higher bar for halting
    return config


def get_memory_config() -> Phase1ModelConfig:
    """
    Get memory-optimized configuration (larger LMM, stronger gate).

    Returns:
        Phase1ModelConfig optimized for long-range memory
    """
    config = Phase1ModelConfig()
    config.titans_mag.memory.d_mem = 384  # Larger memory (3/4 of d_model)
    config.titans_mag.memory.decay = 0.95  # Faster decay (more recent bias)
    config.mag_gate.hidden = 384  # Larger gate network
    config.mag_gate.entropy_reg = 0.01  # Stronger regularization
    return config


def get_speed_config() -> Phase1ModelConfig:
    """
    Get speed-optimized configuration (fewer layers, smaller memory).

    Returns:
        Phase1ModelConfig optimized for inference speed
    """
    config = Phase1ModelConfig()
    config.titans_mag.n_layers = 6  # Fewer layers (saves ~6M params)
    config.titans_mag.memory.d_mem = 192  # Smaller memory
    config.trm.T_max = 2  # Fewer recursion steps
    config.trm.micro_steps = 1  # Single refinement step
    return config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_config(config: Phase1ModelConfig) -> List[str]:
    """
    Validate configuration and return list of warnings.

    Args:
        config: Configuration to validate

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    # Check parameter budget
    params = config.estimate_params()
    if params < 24e6:
        warnings.append(f"Parameter count ({params/1e6:.2f}M) below target (25M). Consider increasing d_model or n_layers.")
    elif params > 26e6:
        warnings.append(f"Parameter count ({params/1e6:.2f}M) above target (25M). Consider reducing d_model, n_layers, or d_mem.")

    # Check head dimension
    if config.titans_mag.head_dim not in [64, 128]:
        warnings.append(f"Unusual head_dim ({config.titans_mag.head_dim}). Standard values: 64, 128.")

    # Check memory dimension
    if config.titans_mag.memory.d_mem > config.titans_mag.d_model:
        warnings.append(f"Memory dimension ({config.titans_mag.memory.d_mem}) > d_model ({config.titans_mag.d_model}). Usually d_mem ≤ d_model.")

    # Check step weights
    if len(config.trm.step_weights) < config.trm.T_max_final:
        warnings.append(f"step_weights length ({len(config.trm.step_weights)}) < T_max_final ({config.trm.T_max_final}). Add more weights.")

    # Check curriculum
    if config.trm.curriculum_enabled and config.trm.T_max >= config.trm.T_max_final:
        warnings.append(f"T_max ({config.trm.T_max}) >= T_max_final ({config.trm.T_max_final}). Curriculum won't increase depth.")

    return warnings


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test configuration loading
    print("Testing Phase 1 Model Configuration")
    print("=" * 60)

    # Default config
    print("\n1. Default Configuration:")
    config = get_default_config()
    print(config)

    # Check budget
    print(f"\nParameter budget check:")
    try:
        config.check_budget()
        print("✅ Budget OK")
    except AssertionError as e:
        print(f"❌ Budget exceeded: {e}")

    # Validation
    print(f"\nValidation:")
    warnings_list = validate_config(config)
    if warnings_list:
        for w in warnings_list:
            print(f"⚠️  {w}")
    else:
        print("✅ No warnings")

    # Test presets
    print("\n2. Reasoning-Optimized Configuration:")
    reasoning_config = get_reasoning_config()
    print(f"  T_max: {reasoning_config.trm.T_max}")
    print(f"  micro_steps: {reasoning_config.trm.micro_steps}")
    print(f"  Estimated params: {reasoning_config.estimate_params()/1e6:.2f}M")

    print("\n3. Memory-Optimized Configuration:")
    memory_config = get_memory_config()
    print(f"  d_mem: {memory_config.titans_mag.memory.d_mem}")
    print(f"  gate.hidden: {memory_config.mag_gate.hidden}")
    print(f"  Estimated params: {memory_config.estimate_params()/1e6:.2f}M")

    print("\n4. Speed-Optimized Configuration:")
    speed_config = get_speed_config()
    print(f"  n_layers: {speed_config.titans_mag.n_layers}")
    print(f"  T_max: {speed_config.trm.T_max}")
    print(f"  Estimated params: {speed_config.estimate_params()/1e6:.2f}M")

    # Test YAML loading (if file exists)
    yaml_path = Path(__file__).parent.parent / "configs" / "phase1_config.yaml"
    if yaml_path.exists():
        print(f"\n5. Loading from YAML ({yaml_path}):")
        yaml_config = Phase1ModelConfig.from_yaml(str(yaml_path))
        print(f"  d_model: {yaml_config.titans_mag.d_model}")
        print(f"  n_layers: {yaml_config.titans_mag.n_layers}")
        print(f"  Estimated params: {yaml_config.estimate_params()/1e6:.2f}M")
    else:
        print(f"\n5. YAML config not found at {yaml_path}")

    print("\n" + "=" * 60)
    print("Configuration testing complete!")
