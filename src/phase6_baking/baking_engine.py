"""
Phase 6: Baking Engine - A/B Cycle Orchestrator

Implements iterative A/B optimization loops for tool and persona baking.
A-Cycle focuses on tool use optimization (SWE-Bench).
B-Cycle focuses on persona/behavior optimization (self-guided).

Research: "Prompt Baking" (arXiv:2409.13697v1)

M4 TIER 1: W&B logging for A/B cycle metrics.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# W&B integration (optional - graceful fallback)
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]


class BakingCycleType(Enum):
    """Type of baking cycle."""

    A_CYCLE = "tool"  # Tool use optimization
    B_CYCLE = "persona"  # Persona optimization


@dataclass
class BakingConfig:
    """Configuration for Phase 6 baking."""

    # A-Cycle (Tool) settings
    a_cycle_iterations: int = 5
    tool_prompts: List[str] = field(
        default_factory=lambda: [
            "You are an expert at using tools systematically.",
            "Always verify tool outputs before proceeding.",
            "Break complex tasks into tool-executable steps.",
        ]
    )

    # B-Cycle (Persona) settings
    b_cycle_iterations: int = 5
    persona_prompts: List[str] = field(
        default_factory=lambda: [
            "You are a careful and thorough assistant.",
            "Think step by step before responding.",
            "Verify your answers before providing them.",
        ]
    )

    # Half-baking settings
    half_bake_strength: float = 0.5
    baking_epochs: int = 3
    learning_rate: float = 5e-5  # Fixed: was 1e-4, now 5e-5 per M4 spec

    # Convergence settings
    plateau_window: int = 3
    plateau_threshold: float = 0.01
    max_total_iterations: int = 20

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32


@dataclass
class BakingResult:
    """Result from Phase 6 baking."""

    success: bool
    model: nn.Module
    total_iterations: int
    a_cycle_count: int
    b_cycle_count: int
    final_tool_score: float
    final_persona_score: float
    metrics: Dict
    artifacts: Dict
    error: Optional[str] = None


class BakingEngine:
    """
    Phase 6: Tool & Persona Baking Engine.

    Implements A/B cycle optimization:
    - A-Cycle: Optimize tool use via SWE-Bench style tasks
    - B-Cycle: Optimize persona via self-guided discovery
    - Half-baking for iterative improvement
    - Plateau detection for automatic cycle switching

    M4 TIER 1: W&B logging for all A/B cycle metrics.
    """

    def __init__(
        self,
        config: Optional[BakingConfig] = None,
        use_wandb: bool = True,
        wandb_project: str = "agent-forge-phase6",
        wandb_run_name: Optional[str] = None,
    ):
        """
        Initialize baking engine.

        Args:
            config: Baking configuration
            use_wandb: Enable W&B logging (default True)
            wandb_project: W&B project name
            wandb_run_name: Optional run name (auto-generated if None)
        """
        self.config = config or BakingConfig()
        self.metrics: Dict[str, List[Any]] = {
            "a_cycle_scores": [],
            "b_cycle_scores": [],
            "iteration_times": [],
            "plateau_detections": [],
        }

        # W&B integration
        self.use_wandb: bool = use_wandb and WANDB_AVAILABLE
        self.wandb_project: str = wandb_project
        self.wandb_run_name: Optional[str] = wandb_run_name
        self._wandb_run: Optional[Any] = None

    def _init_wandb(self) -> None:
        """Initialize W&B run for Phase 6 logging."""
        if not self.use_wandb:
            return

        try:
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name or f"phase6-baking-{int(time.time())}",
                config={
                    "phase": 6,
                    "a_cycle_iterations": self.config.a_cycle_iterations,
                    "b_cycle_iterations": self.config.b_cycle_iterations,
                    "half_bake_strength": self.config.half_bake_strength,
                    "baking_epochs": self.config.baking_epochs,
                    "learning_rate": self.config.learning_rate,
                    "plateau_window": self.config.plateau_window,
                    "plateau_threshold": self.config.plateau_threshold,
                    "max_total_iterations": self.config.max_total_iterations,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                },
                reinit=True,
            )
            print(f"  W&B run initialized: {wandb.run.name if wandb.run else 'unknown'}")
        except Exception as e:
            print(f"  W&B init failed: {e}. Continuing without logging.")
            self.use_wandb = False

    def _log_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        if not self.use_wandb or not self._wandb_run:
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"  W&B log failed: {e}")

    def _finish_wandb(self, result: "BakingResult") -> None:
        """Finish W&B run with summary."""
        if not self.use_wandb or not self._wandb_run:
            return

        try:
            # Log final summary
            wandb.summary.update(
                {
                    "final_tool_score": result.final_tool_score,
                    "final_persona_score": result.final_persona_score,
                    "total_iterations": result.total_iterations,
                    "a_cycle_count": result.a_cycle_count,
                    "b_cycle_count": result.b_cycle_count,
                    "success": result.success,
                    "plateau_count": len(result.metrics.get("plateau_detections", [])),
                }
            )

            # Log score histories as artifacts
            if result.metrics.get("a_cycle_scores"):
                wandb.log({"a_cycle_final_scores": result.metrics["a_cycle_scores"]})
            if result.metrics.get("b_cycle_scores"):
                wandb.log({"b_cycle_final_scores": result.metrics["b_cycle_scores"]})

            wandb.finish()
            print(f"  W&B run finished successfully")
        except Exception as e:
            print(f"  W&B finish failed: {e}")

    def run(
        self,
        model: nn.Module,
        tokenizer: Any,
        tool_evaluator: Any = None,
        persona_evaluator: Any = None,
    ) -> BakingResult:
        """
        Execute Phase 6 baking pipeline.

        Args:
            model: Model from Phase 5
            tokenizer: Tokenizer for encoding
            tool_evaluator: Evaluator for A-cycle (SWE-Bench style)
            persona_evaluator: Evaluator for B-cycle

        Returns:
            BakingResult with optimized model
        """
        print("Phase 6: Starting A/B Baking Cycles")
        print("=" * 50)

        # Initialize W&B
        self._init_wandb()

        start_time = time.time()
        current_model = model

        a_cycle_count = 0
        b_cycle_count = 0
        total_iterations = 0

        # Import components
        from .a_cycle_tool import ACycleOptimizer
        from .b_cycle_persona import BCycleOptimizer
        from .half_baking import HalfBaker
        from .plateau_detector import PlateauDetector

        # Initialize components
        a_optimizer = ACycleOptimizer(
            tool_prompts=self.config.tool_prompts,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            num_epochs=self.config.baking_epochs,
            learning_rate=self.config.learning_rate,
        )

        b_optimizer = BCycleOptimizer(
            persona_prompts=self.config.persona_prompts,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            num_epochs=self.config.baking_epochs,
            learning_rate=self.config.learning_rate,
        )

        plateau_detector = PlateauDetector(
            window_size=self.config.plateau_window, threshold=self.config.plateau_threshold
        )

        half_baker = HalfBaker(strength=self.config.half_bake_strength)

        # Current cycle (start with A)
        current_cycle = BakingCycleType.A_CYCLE

        try:
            while total_iterations < self.config.max_total_iterations:
                iter_start = time.time()
                total_iterations += 1

                print(f"\n--- Iteration {total_iterations} ({current_cycle.value} cycle) ---")

                if current_cycle == BakingCycleType.A_CYCLE:
                    # A-Cycle: Tool optimization
                    a_cycle_count += 1

                    # Run A-cycle optimization
                    baked_model, score = a_optimizer.optimize(
                        model=current_model, tokenizer=tokenizer, evaluator=tool_evaluator
                    )

                    # Half-bake the result
                    current_model = half_baker.half_bake(
                        original_model=current_model, baked_model=baked_model
                    )

                    self.metrics["a_cycle_scores"].append(score)
                    print(f"  A-cycle score: {score:.3f}")

                    # W&B logging for A-cycle
                    self._log_wandb(
                        {
                            "a_cycle/score": score,
                            "a_cycle/iteration": a_cycle_count,
                            "a_cycle/best_score": max(self.metrics["a_cycle_scores"]),
                            "cycle_type": "A",
                        },
                        step=total_iterations,
                    )

                    # Check for plateau
                    if plateau_detector.check(score, "a_cycle"):
                        print(f"  Plateau detected in A-cycle, switching to B-cycle")
                        self.metrics["plateau_detections"].append(("a_cycle", total_iterations))
                        current_cycle = BakingCycleType.B_CYCLE

                else:
                    # B-Cycle: Persona optimization
                    b_cycle_count += 1

                    # Run B-cycle optimization
                    baked_model, score = b_optimizer.optimize(
                        model=current_model, tokenizer=tokenizer, evaluator=persona_evaluator
                    )

                    # Half-bake the result
                    current_model = half_baker.half_bake(
                        original_model=current_model, baked_model=baked_model
                    )

                    self.metrics["b_cycle_scores"].append(score)
                    print(f"  B-cycle score: {score:.3f}")

                    # W&B logging for B-cycle
                    self._log_wandb(
                        {
                            "b_cycle/score": score,
                            "b_cycle/iteration": b_cycle_count,
                            "b_cycle/best_score": max(self.metrics["b_cycle_scores"]),
                            "cycle_type": "B",
                        },
                        step=total_iterations,
                    )

                    # Check for plateau
                    if plateau_detector.check(score, "b_cycle"):
                        print(f"  Plateau detected in B-cycle, switching to A-cycle")
                        self.metrics["plateau_detections"].append(("b_cycle", total_iterations))
                        current_cycle = BakingCycleType.A_CYCLE

                iter_time = time.time() - iter_start
                self.metrics["iteration_times"].append(iter_time)

                # Log iteration metrics
                self._log_wandb(
                    {
                        "iteration/time_seconds": iter_time,
                        "iteration/total": total_iterations,
                        "iteration/a_count": a_cycle_count,
                        "iteration/b_count": b_cycle_count,
                    },
                    step=total_iterations,
                )

                # Check if both cycles have plateaued
                if plateau_detector.both_plateaued():
                    print(f"\n  Both cycles plateaued. Stopping at iteration {total_iterations}")
                    break

            # Get final scores
            final_tool_score = (
                self.metrics["a_cycle_scores"][-1] if self.metrics["a_cycle_scores"] else 0.0
            )
            final_persona_score = (
                self.metrics["b_cycle_scores"][-1] if self.metrics["b_cycle_scores"] else 0.0
            )

            total_time = time.time() - start_time

            print(f"\nPhase 6 Complete:")
            print(f"  Total iterations: {total_iterations}")
            print(f"  A-cycle iterations: {a_cycle_count}")
            print(f"  B-cycle iterations: {b_cycle_count}")
            print(f"  Final tool score: {final_tool_score:.3f}")
            print(f"  Final persona score: {final_persona_score:.3f}")
            print(f"  Total time: {total_time:.1f}s")

            result = BakingResult(
                success=True,
                model=current_model,
                total_iterations=total_iterations,
                a_cycle_count=a_cycle_count,
                b_cycle_count=b_cycle_count,
                final_tool_score=final_tool_score,
                final_persona_score=final_persona_score,
                metrics=self.metrics,
                artifacts={
                    "a_optimizer_state": a_optimizer.get_state(),
                    "b_optimizer_state": b_optimizer.get_state(),
                    "plateau_history": plateau_detector.get_history(),
                },
            )

            # Finish W&B run with success
            self._finish_wandb(result)
            return result

        except Exception as e:
            result = BakingResult(
                success=False,
                model=current_model,
                total_iterations=total_iterations,
                a_cycle_count=a_cycle_count,
                b_cycle_count=b_cycle_count,
                final_tool_score=0.0,
                final_persona_score=0.0,
                metrics=self.metrics,
                artifacts={},
                error=str(e),
            )

            # Finish W&B run with failure
            self._finish_wandb(result)
            return result


__all__ = ["BakingEngine", "BakingConfig", "BakingResult", "BakingCycleType"]
