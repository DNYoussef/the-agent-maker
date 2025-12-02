"""
Phase 3 W&B Logger Wrapper

Simplifies W&B logging for Phase 3 (Step 1 & Step 2).
"""

from typing import Dict, Optional
from pathlib import Path
import wandb

from ..cross_phase.monitoring.wandb_integration import WandBIntegration


class WandBLogger:
    """
    Simplified W&B logger for Phase 3.

    Wraps WandBIntegration with Phase 3-specific convenience methods.
    """

    def __init__(
        self,
        project: str = "agent-forge-v2",
        name: str = "phase3-training",
        config: Dict = None,
        tags: list = None,
        mode: str = "offline",
    ):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            name: Run name
            config: Configuration dict
            tags: Tags for run
            mode: 'offline' or 'online'
        """
        self.integration = WandBIntegration(
            project_name=project, mode=mode
        )

        # Initialize run
        self.run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            tags=tags or [],
            mode=mode,
        )

    def log(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to W&B."""
        self.integration.log_metrics(metrics, step)

    def log_baking_epoch(
        self,
        epoch: int,
        loss: float,
        learning_rate: float,
        overall_accuracy: float,
        strategy_accuracies: Dict[str, float],
        token_usage: Optional[Dict[str, float]] = None,
        step: int = 0,
    ):
        """
        Log Step 1 (Prompt Baking) epoch metrics.

        Automatically computes convergence progress.
        """
        # Compute convergence progress (target: 0.85)
        convergence_progress = overall_accuracy / 0.85

        self.integration.log_phase3_step1_baking(
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            overall_accuracy=overall_accuracy,
            strategy_accuracies=strategy_accuracies,
            token_usage=token_usage,
            convergence_progress=convergence_progress,
            step=step,
        )

    def log_rl_episode(
        self,
        episode: int,
        reward: float,
        avg_reward: float,
        coherence_scores: Dict[str, float],
        thought_metrics: Dict[str, float],
        accuracy_metrics: Dict[str, float],
        anti_theater_metrics: Optional[Dict[str, float]] = None,
        step: int = 0,
    ):
        """
        Log Step 2 (Quiet-STaR RL) episode metrics.

        Args:
            episode: Episode number
            reward: Episode reward
            avg_reward: Average reward (last 100 episodes)
            coherence_scores: {
                'semantic': float,
                'syntactic': float,
                'predictive': float,
                'composite': float
            }
            thought_metrics: {
                'length': float,
                'diversity': float,
                'kl_divergence': float
            }
            accuracy_metrics: {
                'gsm8k': float,
                'arc': float,
                'inference_time_ms': float
            }
            anti_theater_metrics: {
                'divergence': float,
                'ablation': float
            } (optional, only every 1000 steps)
        """
        self.integration.log_phase3_step2_rl(
            episode=episode,
            reward=reward,
            avg_reward=avg_reward,
            coherence_semantic=coherence_scores.get("semantic", 0.0),
            coherence_syntactic=coherence_scores.get("syntactic", 0.0),
            coherence_predictive=coherence_scores.get("predictive", 0.0),
            coherence_composite=coherence_scores.get("composite", 0.0),
            thought_length=thought_metrics.get("length", 0.0),
            thought_diversity=thought_metrics.get("diversity", 0.0),
            kl_divergence=thought_metrics.get("kl_divergence", 0.0),
            gsm8k_accuracy=accuracy_metrics.get("gsm8k", 0.0),
            arc_accuracy=accuracy_metrics.get("arc", 0.0),
            inference_time_ms=accuracy_metrics.get("inference_time_ms", 0.0),
            anti_theater_divergence=(
                anti_theater_metrics.get("divergence")
                if anti_theater_metrics
                else None
            ),
            anti_theater_ablation=(
                anti_theater_metrics.get("ablation")
                if anti_theater_metrics
                else None
            ),
            step=step,
        )

    def save_artifact(
        self,
        file_path: Path,
        name: str,
        type: str = "model",
        metadata: Optional[Dict] = None,
    ):
        """
        Save artifact to W&B.

        Args:
            file_path: Path to artifact file
            name: Artifact name
            type: Artifact type ('model', 'dataset', etc.)
            metadata: Metadata dict
        """
        self.integration.log_artifact(
            artifact_name=name,
            artifact_type=type,
            file_path=str(file_path),
            metadata=metadata or {},
        )

    def finish(self):
        """Finish W&B run."""
        self.integration.finish()
