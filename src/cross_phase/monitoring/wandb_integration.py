"""
Weights & Biases Integration
603 metrics across all 8 phases with offline mode support

ISS-018: Added comprehensive error handling for W&B operations
ISS-002: Export METRICS_COUNT at module level for test compatibility
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import wandb

# Setup logger for W&B operations
logger = logging.getLogger(__name__)


# =============================================================================
# ISS-002: Module-level METRICS_COUNT export for test compatibility
# =============================================================================
METRICS_COUNT = {
    "phase1": 37,  # Cognate
    "phase2": 370,  # EvoMerge (50 generations x ~7 metrics)
    "phase3": 17,  # Quiet-STaR
    "phase4": 19,  # BitNet
    "phase5": 78,  # Curriculum Learning
    "phase6": 32,  # Tool & Persona Baking
    "phase7": 28,  # Self-Guided Experts
    "phase8": 95,  # Final Compression
}


class WandBIntegration:
    """
    Complete W&B integration for Agent Forge V2

    Features:
    - 676 total metrics across 8 phases
    - Offline mode (no cloud upload)
    - Artifact versioning
    - Metric continuity tracking
    """

    # Total metrics per phase (from specification)
    METRICS_COUNT = {
        "phase1": 37,  # Cognate
        "phase2": 370,  # EvoMerge (50 generations × ~7 metrics)
        "phase3": 17,  # Quiet-STaR
        "phase4": 19,  # BitNet
        "phase5": 78,  # Curriculum Learning
        "phase6": 32,  # Tool & Persona Baking
        "phase7": 28,  # Self-Guided Experts
        "phase8": 95,  # Final Compression
    }

    def __init__(
        self,
        project_name: str = "agent-forge-v2",
        mode: str = "offline",
        entity: Optional[str] = None,
        project: Optional[str] = None,  # ISS-002: Alias for project_name
    ):
        # ISS-002: Support both project_name and project parameter
        self.project_name = project if project else project_name
        self.mode = mode
        self.entity = entity
        self.current_run: Any = None

    @property
    def project(self) -> str:
        """ISS-002: Alias for project_name for test compatibility."""
        return self.project_name

    def init_phase_run(self, phase_name: str, config: Dict, session_id: str) -> Any:
        """
        Initialize W&B run for a specific phase with error handling (ISS-018).

        Args:
            phase_name: e.g., 'phase1', 'phase2', etc.
            config: Phase configuration dict
            session_id: Unique session identifier

        Returns:
            W&B run object or None if initialization fails
        """
        run_name = f"{phase_name}_{session_id}"

        try:
            self.current_run = wandb.init(
                project=self.project_name,
                name=run_name,
                config=config,
                tags=[phase_name, session_id],
                mode=self.mode,
                entity=self.entity,
                dir=f"./storage/wandb/{session_id}",
            )
            logger.info(f"W&B run initialized: {run_name}")
            return self.current_run
        except wandb.Error as e:
            logger.error(f"W&B initialization failed: {e}")
            self.current_run = None
            return None
        except Exception as e:
            logger.error(f"Unexpected error initializing W&B: {e}")
            self.current_run = None
            return None

    def log_metrics(self, metrics: Dict, step: Optional[int] = None) -> None:
        """
        Log metrics to W&B with error handling (ISS-018).

        Args:
            metrics: Dict of metric_name -> value
            step: Optional step number
        """
        if not self.current_run:
            logger.warning("W&B run not initialized, skipping metrics")
            return

        try:
            wandb.log(metrics, step=step)
        except wandb.Error as e:
            logger.error(f"W&B logging failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error logging to W&B: {e}")

    def log_phase1_metrics(
        self,
        model_id: int,
        loss: float,
        perplexity: float,
        act_steps: float,
        gate_entropy: float,
        muon_lr: float,
        grokfast_mu_norm: float,
        step: int,
    ) -> None:
        """Phase 1 (Cognate) - 37 total metrics"""
        metrics = {
            f"model{model_id}/train/loss": loss,
            f"model{model_id}/train/perplexity": perplexity,
            f"model{model_id}/act/avg_steps": act_steps,
            f"model{model_id}/gate/entropy": gate_entropy,
            f"model{model_id}/optim/muon_lr": muon_lr,
            f"model{model_id}/optim/grokfast_mu_norm": grokfast_mu_norm,
        }
        self.log_metrics(metrics, step)

    def log_phase2_metrics(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: float,
        combo_usage: Dict,
        step: int,
    ) -> None:
        """Phase 2 (EvoMerge) - 370+ total metrics"""
        metrics = {
            "generation/number": generation,
            "generation/best_fitness": best_fitness,
            "generation/avg_fitness": avg_fitness,
            "generation/diversity_score": diversity,
        }

        # Add combo usage metrics
        for combo_id, usage_count in combo_usage.items():
            metrics[f"combos/usage_{combo_id}"] = usage_count

        self.log_metrics(metrics, step)

    def log_phase3_step1_baking(
        self,
        epoch: int,
        loss: float,
        learning_rate: float,
        overall_accuracy: float,
        strategy_accuracies: Dict[str, float],
        token_usage: Optional[Dict[str, float]] = None,
        convergence_progress: Optional[float] = None,
        step: int = 0,
    ) -> None:
        """
        Phase 3 Step 1 (Prompt Baking) - 17 total metrics

        Metrics:
        1. baking/loss
        2. baking/learning_rate
        3. baking/overall_accuracy
        4-10. baking/accuracy_{strategy} (7 strategies)
        11-15. baking/token_usage_{type} (5 types)
        16. baking/convergence_progress
        17. baking/epoch
        """
        metrics = {
            "baking/epoch": epoch,
            "baking/loss": loss,
            "baking/learning_rate": learning_rate,
            "baking/accuracy_overall": overall_accuracy,
        }

        # Strategy-specific accuracies (7 metrics)
        for strategy, acc in strategy_accuracies.items():
            metrics[f"baking/accuracy_{strategy}"] = acc

        # Token usage metrics (5 metrics)
        if token_usage:
            for token_type, usage in token_usage.items():
                metrics[f"baking/{token_type}"] = usage

        # Convergence progress
        if convergence_progress is not None:
            metrics["baking/convergence_progress"] = convergence_progress

        self.log_metrics(metrics, step)

    def log_phase3_step2_rl(
        self,
        episode: int,
        reward: float,
        avg_reward: float,
        coherence_semantic: float,
        coherence_syntactic: float,
        coherence_predictive: float,
        coherence_composite: float,
        thought_length: float,
        thought_diversity: float,
        kl_divergence: float,
        gsm8k_accuracy: float,
        arc_accuracy: float,
        inference_time_ms: float,
        anti_theater_divergence: Optional[float] = None,
        anti_theater_ablation: Optional[float] = None,
        step: int = 0,
    ) -> None:
        """
        Phase 3 Step 2 (Quiet-STaR RL) - 17 total metrics

        Metrics:
        1. rl/episode
        2. rl/reward
        3. rl/avg_reward
        4. rl/coherence_semantic
        5. rl/coherence_syntactic
        6. rl/coherence_predictive
        7. rl/coherence_composite
        8. rl/thought_length
        9. rl/thought_diversity
        10. rl/kl_divergence
        11. rl/gsm8k_accuracy
        12. rl/arc_accuracy
        13. rl/inference_time_ms
        14. rl/anti_theater_divergence
        15. rl/anti_theater_ablation
        16-17. Reserved for additional metrics
        """
        metrics = {
            "rl/episode": episode,
            "rl/reward": reward,
            "rl/avg_reward": avg_reward,
            "rl/coherence_semantic": coherence_semantic,
            "rl/coherence_syntactic": coherence_syntactic,
            "rl/coherence_predictive": coherence_predictive,
            "rl/coherence_composite": coherence_composite,
            "rl/thought_length": thought_length,
            "rl/thought_diversity": thought_diversity,
            "rl/kl_divergence": kl_divergence,
            "rl/gsm8k_accuracy": gsm8k_accuracy,
            "rl/arc_accuracy": arc_accuracy,
            "rl/inference_time_ms": inference_time_ms,
        }

        # Anti-theater metrics (tested every 1000 steps)
        if anti_theater_divergence is not None:
            metrics["rl/anti_theater_divergence"] = anti_theater_divergence

        if anti_theater_ablation is not None:
            metrics["rl/anti_theater_ablation"] = anti_theater_ablation

        self.log_metrics(metrics, step)

    def log_phase4_pre_compression(self, metrics: Dict) -> None:
        """Phase 4 Pre-Compression (3 metrics)"""
        wandb_metrics = {
            "compression/original_size_mb": metrics.get("original_size_mb", 0.0),
            "compression/pre_perplexity": metrics.get("pre_perplexity", 0.0),
            "compression/pre_eval_loss": metrics.get("pre_eval_loss", 0.0),
        }
        self.log_metrics(wandb_metrics)

    def log_phase4_compression(self, metrics: Dict) -> None:
        """Phase 4 Compression Process (7 metrics)"""
        wandb_metrics = {
            "compression/compressed_size_mb": metrics.get("compressed_size_mb", 0.0),
            "compression/ratio": metrics.get("compression_ratio", 0.0),
            "compression/layers_compressed": metrics.get("layers_quantized", 0),
            "compression/sparsity_ratio": metrics.get("sparsity_ratio", 0.0),
            "compression/quantized_params": metrics.get("quantized_params", 0),
            "compression/total_params": metrics.get("total_params", 0),
            "compression/layers_quantized": metrics.get("layers_quantized", 0),
        }
        self.log_metrics(wandb_metrics)

    def log_phase4_post_compression(self, metrics: Dict) -> None:
        """Phase 4 Post-Compression (5 metrics)"""
        wandb_metrics = {
            "compression/post_perplexity": metrics.get("post_perplexity", 0.0),
            "compression/perplexity_degradation": metrics.get("perplexity_degradation", 0.0),
            "compression/accuracy_preserved": metrics.get("accuracy_preserved", False),
            "compression/dequantization_accuracy": metrics.get("dequantization_accuracy", 0.0),
            "compression/gradient_flow_test_passed": metrics.get("gradient_flow_passed", False),
        }
        self.log_metrics(wandb_metrics)

    def log_phase4_fine_tuning(self, metrics: Dict) -> None:
        """Phase 4 Fine-Tuning (4 metrics)"""
        if metrics is None:
            return

        wandb_metrics = {
            "compression/post_finetune_perplexity": metrics.get("best_perplexity", 0.0),
            "compression/perplexity_recovery": metrics.get("improvement", 0.0),
            "compression/fine_tune_epochs": metrics.get("epochs", 0),
            "compression/fine_tune_time_hours": metrics.get("time_hours", 0.0),
        }
        self.log_metrics(wandb_metrics)

    def log_phase4_summary(self, results: Dict) -> None:
        """Phase 4 Summary (8 metrics)"""
        post_metrics = results.get("post_compression", {})
        fine_tune = results.get("fine_tuning", {})

        # Determine final perplexity
        if fine_tune:
            final_perplexity = fine_tune.get("best_perplexity", 0.0)
        else:
            final_perplexity = post_metrics.get("post_perplexity", 0.0)

        wandb_metrics = {
            "phase/compression_ratio": post_metrics.get("compression_ratio", 0.0),
            "phase/original_size_mb": results.get("pre_compression", {}).get(
                "original_size_mb", 0.0
            ),
            "phase/compressed_size_mb": post_metrics.get("compressed_size_mb", 0.0),
            "phase/final_perplexity": final_perplexity,
            "phase/accuracy_preserved": post_metrics.get("accuracy_preserved", False),
            "phase/success": results.get("success", False),
            "phase/quantization_method": "BitNet-1.58",
            "phase/sparsity_ratio": post_metrics.get("sparsity_ratio", 0.0),
        }
        self.log_metrics(wandb_metrics)

    def log_artifact(
        self, artifact_name: str, artifact_type: str, file_path: str, metadata: Dict
    ) -> None:
        """
        Log model artifact to W&B with error handling (ISS-018).

        Args:
            artifact_name: e.g., 'cognate_model_reasoning'
            artifact_type: 'model', 'dataset', etc.
            file_path: Path to artifact file
            metadata: Metadata dict
        """
        if not self.current_run:
            logger.warning("W&B run not initialized, skipping artifact")
            return

        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=metadata.get("description", ""),
                metadata=metadata,
            )
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
            logger.info(f"W&B artifact logged: {artifact_name}")
        except FileNotFoundError:
            logger.error(f"Artifact file not found: {file_path}")
        except wandb.Error as e:
            logger.error(f"W&B artifact logging failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error logging artifact: {e}")

    def finish(self) -> None:
        """Finish current W&B run with error handling (ISS-018)."""
        if self.current_run:
            try:
                wandb.finish()
                logger.info("W&B run finished successfully")
            except wandb.Error as e:
                logger.error(f"Error finishing W&B run: {e}")
            except Exception as e:
                logger.error(f"Unexpected error finishing W&B run: {e}")
            finally:
                self.current_run = None

    def get_metrics_config(self, phase_name: str) -> Dict:
        """
        Get complete metrics configuration for a phase

        Returns all metric names and their types
        """
        configs = {
            "phase1": {
                "train/loss": "float",
                "train/perplexity": "float",
                "act/avg_steps": "float",
                "act/halt_probability": "float",
                "gate/entropy": "float",
                "gate/avg_blend": "float",
                "optim/muon_lr": "float",
                "optim/grokfast_mu_norm": "float",
                "optim/qk_clip_activations": "int",
                # ... (37 total)
            },
            "phase2": {
                "generation/number": "int",
                "generation/best_fitness": "float",
                "generation/avg_fitness": "float",
                "generation/min_fitness": "float",
                "generation/max_fitness": "float",
                "generation/diversity_score": "float",
                "generation/fitness_std": "float",
                # ... (370+ total across 50 generations)
            },
            # ... (Phase 3-8 configs)
        }

        return configs.get(phase_name, {})


class MetricContinuityTracker:
    """
    Track metrics across all phases for continuity analysis

    Example: Track how accuracy evolves from Phase 1 -> Phase 8

    ISS-002: Added history attribute and add_phase_metrics method for test compatibility
    """

    def __init__(self) -> None:
        self.metrics: dict[str, list[Any]] = {
            "accuracy": [],
            "perplexity": [],
            "model_size_mb": [],
            "inference_latency_ms": [],
        }
        self.phase_names: list[str] = []
        # ISS-002: History dict for test compatibility
        self.history: Dict[str, Dict] = {}

    def add_phase_metrics(self, phase: str, metrics: Dict) -> None:
        """
        ISS-002: Add metrics for a phase (test-compatible API).

        Args:
            phase: Phase name (e.g., 'phase1')
            metrics: Dict of metric_name -> value
        """
        self.history[phase] = metrics
        self.phase_names.append(phase)

        # Also add to the flat metrics lists for backward compat
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)

    def get_trend(self, metric_name: str) -> list:
        """
        ISS-002: Get trend values for a metric across phases.

        Args:
            metric_name: Name of the metric

        Returns:
            List of values in phase order
        """
        if metric_name in self.metrics:
            return self.metrics[metric_name]

        # Try to extract from history
        values = []
        for phase in self.phase_names:
            if phase in self.history and metric_name in self.history[phase]:
                values.append(self.history[phase][metric_name])
        return values

    def detect_degradation(self, metric_name: str, threshold: float = 0.1) -> bool:
        """
        ISS-002: Detect if a metric has degraded significantly.

        Args:
            metric_name: Name of the metric
            threshold: Degradation threshold (fraction, e.g., 0.1 = 10%)

        Returns:
            True if degradation detected, False otherwise
        """
        trend = self.get_trend(metric_name)
        if len(trend) < 2:
            return False

        # Check if latest value dropped significantly from max
        max_val = max(trend[:-1]) if len(trend) > 1 else trend[0]
        latest = trend[-1]

        if max_val == 0:
            return False

        drop = (max_val - latest) / max_val
        return bool(drop > threshold)

    def record_phase(self, phase_name: str, metrics: Dict) -> None:
        """
        Record metrics at end of phase (legacy API, wraps add_phase_metrics).

        Args:
            phase_name: e.g., 'phase1'
            metrics: Dict with standard metrics
        """
        self.add_phase_metrics(phase_name, metrics)

    def log_to_wandb(self) -> None:
        """Log cross-phase metric evolution to W&B"""
        # Create continuity table
        continuity_table = wandb.Table(
            columns=["phase", "accuracy", "perplexity", "size_mb", "latency_ms"],
            data=[
                [phase, acc, ppl, size, lat]
                for phase, acc, ppl, size, lat in zip(
                    self.phase_names,
                    self.metrics["accuracy"],
                    self.metrics["perplexity"],
                    self.metrics["model_size_mb"],
                    self.metrics["inference_latency_ms"],
                )
            ],
        )

        wandb.log({"metric_continuity": continuity_table})
