"""
Phase Integration Module for Meta-Calculus

This module provides automatic integration of meta-calculus enhancements
into each phase of Agent Forge V2. Import this module at the start of
any phase to get automatic optimizer upgrades and monitoring.

Usage:
    from cross_phase.meta_calculus.phase_integration import (
        get_phase_optimizer,
        get_phase_monitors,
        PhaseIntegration
    )

    # Get optimizer for Phase 1
    optimizer = get_phase_optimizer("phase1", model)

    # Get monitors for any phase
    monitors = get_phase_monitors("phase3")

    # Or use the integration class for full setup
    integration = PhaseIntegration("phase1")
    optimizer = integration.create_optimizer(model)
    integration.log_step(loss, step)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .bigeometric import BigeometricTransform

# Meta-calculus imports
from .meta_grokfast import PHASE_CONFIGS, MetaGrokfast
from .spectral_gap import (
    SpectralGapMonitor,
    compute_expert_diversity,
    compute_merge_diversity_change,
    compute_thought_diversity,
)

# Optional MOO imports
try:
    from .moo_bridge import EvoMergeProblem, ExpertDiscoveryProblem, MOOConfig, MOORunner

    MOO_AVAILABLE = True
except ImportError:
    MOO_AVAILABLE = False


@dataclass
class PhaseIntegrationConfig:
    """Configuration for phase integration."""

    # General
    phase: str
    enable_meta_calculus: bool = True
    enable_spectral_monitoring: bool = True
    enable_moo: bool = True

    # Logging
    log_frequency: int = 100  # Log every N steps
    log_dir: Optional[Path] = None

    # Thresholds
    spectral_gap_warning: float = 0.05
    gradient_explosion_threshold: float = 100.0


# Phase-specific defaults
PHASE_INTEGRATION_DEFAULTS = {
    "phase1": PhaseIntegrationConfig(
        phase="phase1_cognate",
        enable_spectral_monitoring=True,
        enable_moo=False,  # No MOO for Phase 1
    ),
    "phase2": PhaseIntegrationConfig(
        phase="phase2_evomerge",
        enable_spectral_monitoring=True,
        enable_moo=True,  # MOO for merge optimization
    ),
    "phase3": PhaseIntegrationConfig(
        phase="phase3_quietstar",
        enable_spectral_monitoring=True,  # Thought diversity
        enable_moo=False,
    ),
    "phase4": PhaseIntegrationConfig(
        phase="phase4_bitnet",
        enable_spectral_monitoring=True,
        enable_moo=False,
    ),
    "phase5": PhaseIntegrationConfig(
        phase="phase5_curriculum",
        enable_spectral_monitoring=True,
        enable_moo=False,
    ),
    "phase6": PhaseIntegrationConfig(
        phase="phase6_baking",
        enable_spectral_monitoring=True,
        enable_moo=False,
    ),
    "phase7": PhaseIntegrationConfig(
        phase="phase7_experts",
        enable_spectral_monitoring=True,  # Expert diversity
        enable_moo=True,  # MOO for expert discovery
    ),
    "phase8": PhaseIntegrationConfig(
        phase="phase8_compression",
        enable_spectral_monitoring=True,  # Compression quality
        enable_moo=False,
    ),
}


class PhaseIntegration:
    """
    Main integration class for meta-calculus in Agent Forge phases.

    Provides:
    - Automatic optimizer creation with phase-specific config
    - Spectral gap monitoring
    - MOO integration (for Phases 2, 7)
    - Logging and metrics

    Example:
        integration = PhaseIntegration("phase1")
        optimizer = integration.create_optimizer(model)

        for step, batch in enumerate(dataloader):
            loss = train_step(model, batch, optimizer)
            integration.log_step(loss, step)

        # Get final metrics
        metrics = integration.get_metrics()
    """

    def __init__(self, phase: str, config: Optional[PhaseIntegrationConfig] = None):
        """
        Initialize phase integration.

        Args:
            phase: Phase name (e.g., "phase1", "phase2", etc.)
            config: Optional custom configuration
        """
        # Normalize phase name
        phase_key = phase.lower().replace("_", "").replace("-", "")
        if not phase_key.startswith("phase"):
            phase_key = f"phase{phase_key}"

        # Extract phase number
        phase_num = "".join(filter(str.isdigit, phase_key))
        self.phase_key = f"phase{phase_num}"

        # Get default config or use provided
        if config is None:
            config = PHASE_INTEGRATION_DEFAULTS.get(
                self.phase_key, PhaseIntegrationConfig(phase=phase)
            )
        self.config = config

        # Initialize components
        self.spectral_monitor = SpectralGapMonitor() if config.enable_spectral_monitoring else None
        self.optimizer = None
        self.model = None

        # Metrics tracking
        self.step_count = 0
        self.metrics_history: List[Dict[str, Any]] = []
        self.warnings: List[str] = []

    def create_optimizer(self, model: nn.Module, **kwargs) -> MetaGrokfast:
        """
        Create phase-optimized MetaGrokfast optimizer.

        Args:
            model: PyTorch model
            **kwargs: Override any config values

        Returns:
            MetaGrokfast optimizer configured for this phase
        """
        self.model = model

        # Map phase key to config name
        phase_config_map = {
            "phase1": "phase1_cognate",
            "phase2": "phase1_cognate",  # Phase 2 doesn't train, use Phase 1 config
            "phase3": "phase3_quietstar",
            "phase4": "phase5_curriculum",  # Phase 4 uses Phase 5 (BitNet) config
            "phase5": "phase5_curriculum",
            "phase6": "phase6_baking",
            "phase7": "phase7_experts",
            "phase8": "phase8_compression",
        }

        config_name = phase_config_map.get(self.phase_key, "phase1_cognate")

        self.optimizer = MetaGrokfast.for_phase(config_name, model.parameters(), **kwargs)

        return self.optimizer

    def log_step(
        self, loss: float, step: int, extra_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a training step with meta-calculus metrics.

        Args:
            loss: Current loss value
            step: Current step number
            extra_metrics: Additional metrics to log

        Returns:
            Dictionary of all logged metrics
        """
        self.step_count = step
        metrics = {"loss": loss, "step": step}

        # Add optimizer stats
        if self.optimizer is not None:
            opt_stats = self.optimizer.get_stats()
            metrics.update({f"opt_{k}": v for k, v in opt_stats.items()})

        # Add spectral gap metrics (periodically)
        if (
            self.spectral_monitor is not None
            and self.model is not None
            and step % self.config.log_frequency == 0
        ):
            gap_metrics = self._compute_spectral_metrics()
            metrics.update(gap_metrics)

            # Check for warnings
            if gap_metrics.get("avg_gap", 1.0) < self.config.spectral_gap_warning:
                warning = (
                    f"Step {step}: Spectral gap below threshold ({gap_metrics['avg_gap']:.4f})"
                )
                self.warnings.append(warning)

        # Add extra metrics
        if extra_metrics:
            metrics.update(extra_metrics)

        self.metrics_history.append(metrics)
        return metrics

    def _compute_spectral_metrics(self) -> Dict[str, float]:
        """Compute spectral gap metrics for current model state."""
        if self.model is None or self.spectral_monitor is None:
            return {}

        gaps = self.spectral_monitor.compute_model_gaps(self.model)
        if not gaps:
            return {}

        gap_values = [g["gap"] for g in gaps.values()]
        return {
            "avg_gap": sum(gap_values) / len(gap_values),
            "min_gap": min(gap_values),
            "max_gap": max(gap_values),
            "n_healthy": sum(1 for g in gaps.values() if g["is_healthy"]),
            "n_collapsed": sum(1 for g in gaps.values() if g["is_collapsed"]),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get summary metrics for the phase.

        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}

        # Compute summary
        losses = [m["loss"] for m in self.metrics_history if "loss" in m]
        gaps = [m["avg_gap"] for m in self.metrics_history if "avg_gap" in m]

        summary = {
            "phase": self.phase_key,
            "total_steps": self.step_count,
            "final_loss": losses[-1] if losses else None,
            "avg_loss": sum(losses) / len(losses) if losses else None,
            "min_loss": min(losses) if losses else None,
            "final_gap": gaps[-1] if gaps else None,
            "avg_gap": sum(gaps) / len(gaps) if gaps else None,
            "min_gap": min(gaps) if gaps else None,
            "n_warnings": len(self.warnings),
            "warnings": self.warnings[-5:],  # Last 5 warnings
        }

        # Add optimizer final stats
        if self.optimizer is not None:
            summary["optimizer_stats"] = self.optimizer.get_stats()

        return summary

    # Phase-specific methods

    def monitor_thoughts(self, thoughts: torch.Tensor) -> Dict[str, float]:
        """
        Monitor thought diversity (Phase 3 specific).

        Args:
            thoughts: [N, D] thought embeddings

        Returns:
            Thought diversity metrics
        """
        return compute_thought_diversity(thoughts)

    def monitor_experts(self, expert_weights: List[torch.Tensor]) -> Dict[str, float]:
        """
        Monitor expert diversity (Phase 7 specific).

        Args:
            expert_weights: List of expert weight tensors

        Returns:
            Expert diversity metrics
        """
        return compute_expert_diversity(expert_weights)

    def evaluate_merge(
        self, models_before: List[nn.Module], model_after: nn.Module
    ) -> Dict[str, float]:
        """
        Evaluate merge quality (Phase 2 specific).

        Args:
            models_before: Models before merge
            model_after: Merged model

        Returns:
            Merge quality metrics
        """
        return compute_merge_diversity_change(models_before, model_after)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_phase_optimizer(phase: str, model: nn.Module, **kwargs) -> MetaGrokfast:
    """
    Quick function to get optimizer for a phase.

    Args:
        phase: Phase name (e.g., "phase1", "1", "phase1_cognate")
        model: PyTorch model
        **kwargs: Override config values

    Returns:
        Configured MetaGrokfast optimizer
    """
    integration = PhaseIntegration(phase)
    return integration.create_optimizer(model, **kwargs)


def get_phase_monitors(phase: str) -> Dict[str, Any]:
    """
    Get monitoring functions for a phase.

    Args:
        phase: Phase name

    Returns:
        Dictionary of monitoring functions
    """
    monitors = {
        "spectral_gap": SpectralGapMonitor(),
        "bigeometric": BigeometricTransform(),
    }

    # Add phase-specific monitors
    phase_key = f"phase{phase[-1]}" if phase[-1].isdigit() else phase

    if phase_key == "phase3":
        monitors["thought_diversity"] = compute_thought_diversity
    elif phase_key == "phase7":
        monitors["expert_diversity"] = compute_expert_diversity
    elif phase_key == "phase2":
        monitors["merge_quality"] = compute_merge_diversity_change

    return monitors


def create_moo_problem(phase: str, evaluator: Optional[Callable] = None):
    """
    Create MOO problem for a phase.

    Args:
        phase: Phase name (phase2 or phase7)
        evaluator: Optional custom evaluation function

    Returns:
        MOO problem instance
    """
    if not MOO_AVAILABLE:
        raise ImportError("pymoo required for MOO. Install with: pip install pymoo")

    phase_key = f"phase{phase[-1]}" if phase[-1].isdigit() else phase

    if phase_key == "phase2":
        # EvoMerge problem
        return EvoMergeProblem(
            n_layers=8, n_techniques=6, model_evaluator=evaluator  # Default, can be overridden
        )
    elif phase_key == "phase7":
        # Expert discovery problem
        return ExpertDiscoveryProblem(expert_evaluator=evaluator)
    else:
        raise ValueError(f"MOO not supported for {phase}. Use phase2 or phase7.")


def run_moo_optimization(
    phase: str, evaluator: Callable, config: Optional[MOOConfig] = None
) -> Dict[str, Any]:
    """
    Run MOO optimization for a phase.

    Args:
        phase: Phase name (phase2 or phase7)
        evaluator: Evaluation function
        config: Optional MOO configuration

    Returns:
        MOO results with Pareto front
    """
    if not MOO_AVAILABLE:
        raise ImportError("pymoo required for MOO. Install with: pip install pymoo")

    problem = create_moo_problem(phase, evaluator)
    runner = MOORunner(config or MOOConfig())
    return runner.optimize(problem)


# =============================================================================
# Auto-Integration Hooks
# =============================================================================


def auto_integrate_phase(
    phase: str, model: nn.Module, enable_logging: bool = True
) -> Dict[str, Any]:
    """
    Automatically integrate meta-calculus into a phase.

    This is the main entry point for automatic integration.

    Args:
        phase: Phase name
        model: PyTorch model
        enable_logging: Whether to enable metrics logging

    Returns:
        Dictionary with optimizer and integration objects

    Example:
        components = auto_integrate_phase("phase1", model)
        optimizer = components["optimizer"]
        integration = components["integration"]

        for batch in dataloader:
            loss = train_step(model, batch, optimizer)
            integration.log_step(loss.item(), step)
    """
    integration = PhaseIntegration(phase)
    optimizer = integration.create_optimizer(model)

    return {
        "optimizer": optimizer,
        "integration": integration,
        "monitors": get_phase_monitors(phase),
        "config": integration.config,
    }


# =============================================================================
# Print Integration Status
# =============================================================================


def print_integration_status():
    """Print meta-calculus integration status."""
    print("\n" + "=" * 60)
    print("META-CALCULUS PHASE INTEGRATION STATUS")
    print("=" * 60)

    print("\nAvailable Phase Configs:")
    for phase, config in PHASE_INTEGRATION_DEFAULTS.items():
        moo_status = "MOO" if config.enable_moo else "---"
        spectral_status = "SPECTRAL" if config.enable_spectral_monitoring else "-------"
        print(f"  {phase}: {spectral_status} | {moo_status}")

    print("\nOptimizer Presets:")
    for name, cfg in PHASE_CONFIGS.items():
        print(
            f"  {name}: lr={cfg.lr}, lambda={cfg.grokfast_lambda}, bigeometric={cfg.use_bigeometric}"
        )

    print("\nMOO Available:", "Yes" if MOO_AVAILABLE else "No (install pymoo)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_integration_status()
