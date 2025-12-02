"""
MuonGrokfast: Cross-Phase Optimizer System for Agent Forge V2

This package provides the unified Muon × GrokFast optimizer system that
combines three complementary optimization techniques:

1. **Grokfast** (time-spectrum): Temporal gradient filtering via EMA
2. **Muon** (space-geometry): Spatial gradient orthogonalization via Newton-Schulz
3. **QK-Clip** (safety): Attention logit bounding for RL stability

The optimizer automatically routes parameters based on dimensionality:
- 2-D parameters (weight matrices) → Muon path
- 1-D parameters (bias, LayerNorm) → Fallback path (AdamW/Lion)

All parameters receive Grokfast prefiltering and QK-clip safety rails.

Synergy: Temporal stability + spatial diversity = faster generalization,
reduced low-rank collapse, smoother training trajectories.

Usage:
    >>> from cross_phase import MuonGrokfast, MuGrokConfig
    >>> config = MuGrokConfig.from_phase(1)  # Phase 1: Cognate
    >>> optimizer = MuonGrokfast(model.parameters(), config=config)
    >>> # Training loop
    >>> for batch in dataloader:
    ...     loss = model(batch)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()

Phase-Specific Presets:
    - Phase 1 (Cognate): Pretraining, gentle Grokfast (λ=0.05)
    - Phase 3 (Quiet-STaR): RL training, tighter QK-clip (τ=25.0)
    - Phase 5 (Forge): STE mode for quantized training
    - Phase 6 (Baking): Two-stage (prompt tuning → weight merging)
    - Phase 7 (Transformer²): Fallback-only (z vectors are 1-D)

Author: Agent Forge V2 Team
Version: 1.0.0
License: Internal Research Project
"""

__version__ = "1.0.0"
__author__ = "Agent Forge V2 Team"

# Import main classes
from .mugrokfast_config import MuGrokConfig
from .mugrokfast_optimizer import MuonGrokfast

# Define public API
__all__ = [
    "MuonGrokfast",
    "MuGrokConfig",
]

# Convenience: Phase-specific config shortcuts
def get_phase_config(phase: int, **overrides):
    """
    Convenience function to get phase-specific configuration.

    Args:
        phase: Phase number (1, 3, 5, 6, 7)
        **overrides: Override specific config values

    Returns:
        MuGrokConfig instance

    Example:
        >>> config = get_phase_config(1, muon_lr=0.005)
        >>> optimizer = MuonGrokfast(model.parameters(), config=config)
    """
    return MuGrokConfig.from_phase(phase, **overrides)


# Version check helper
def check_version(required: str) -> bool:
    """
    Check if current version meets requirement.

    Args:
        required: Required version string (e.g., "1.0.0")

    Returns:
        True if current version >= required version
    """
    from packaging import version
    return version.parse(__version__) >= version.parse(required)


# Display banner on import
def _print_banner():
    """Print welcome banner (only in interactive mode)."""
    import sys
    if hasattr(sys, 'ps1'):  # Interactive mode
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  MuonGrokfast Optimizer v{__version__}                          ║
║  Cross-Phase Optimizer System for Agent Forge V2            ║
║                                                              ║
║  Synergy: Time-Spectrum + Space-Geometry                    ║
║  - Grokfast: Temporal filtering (EMA)                       ║
║  - Muon: Spatial orthogonalization (Newton-Schulz)          ║
║  - QK-Clip: Attention safety rails                          ║
║                                                              ║
║  Usage: from cross_phase import MuonGrokfast, MuGrokConfig  ║
╚══════════════════════════════════════════════════════════════╝
        """)

# Conditionally print banner
# _print_banner()  # Uncomment for interactive sessions
