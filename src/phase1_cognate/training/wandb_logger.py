"""
W&B Logger for Phase 1

Logs all 37 metrics defined for Phase 1 Cognate.

ISS-017: Consolidated to use central WandBIntegration
ISS-018: Uses error-handled logging from central module
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import wandb

# Import central W&B integration
try:
    from ..cross_phase.monitoring.wandb_integration import WandBIntegration
except ImportError:
    # Fallback for direct execution
    from cross_phase.monitoring.wandb_integration import WandBIntegration

logger = logging.getLogger(__name__)


class Phase1WandBLogger:
    """
    Weights & Biases logger for Phase 1

    ISS-017: Now wraps WandBIntegration for consolidated logging.

    Tracks 37 metrics:
    - Per step: loss, perplexity, lr, grad_norm, ACT, LTM, GPU
    - Per epoch: val_loss, val_accuracy, curriculum_stage
    - Final: total_params, training_time, diversity_score
    """

    def __init__(self, config: Dict[str, Any], model_name: str, mode: str = "offline"):
        """
        Initialize W&B using central integration (ISS-017).

        Args:
            config: Training configuration
            model_name: Model specialization (reasoning/memory/speed)
            mode: W&B mode (online/offline)
        """
        self.model_name = model_name
        self.config = config

        # Use central integration
        self.integration = WandBIntegration(
            project_name=config.get("wandb_project", "agent-forge-v2"), mode=mode
        )

        # Initialize run
        try:
            wandb.init(
                project=config.get("wandb_project", "agent-forge-v2"),
                name=f"phase1-cognate-{model_name}",
                config=config,
                mode=mode,
                tags=["phase1", "cognate", model_name, "pretraining"],
            )
            if wandb.run is not None:
                logger.info(f"W&B initialized: {wandb.run.name}")
            self.run = wandb.run
        except Exception as e:
            logger.error(f"W&B initialization failed: {e}")
            self.run = None

    def watch_model(self, model: nn.Module, log_freq: int = 100) -> None:
        """
        Track model gradients and parameters with W&B

        Args:
            model: PyTorch model to track
            log_freq: How often to log gradients (every N steps)
        """
        if self.run is not None:
            self.run.watch(model, log="all", log_freq=log_freq)  # Log gradients and parameters
            print(f"W&B watching model (log_freq={log_freq})")

    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        halting_steps: torch.Tensor,
        ltm_usage: float,
        gpu_memory_gb: float,
        gpu_util: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
    ) -> None:
        """
        Log metrics at each training step

        Args:
            step: Global step
            epoch: Current epoch
            loss: Training loss
            learning_rate: Current LR
            grad_norm: Gradient norm
            halting_steps: ACT halting steps [batch]
            ltm_usage: LTM memory usage (0-1)
            gpu_memory_gb: GPU memory in GB
            gpu_util: GPU utilization % (optional)
            tokens_per_sec: Throughput (optional)
        """
        metrics = {
            # Basic training
            "train/step": step,
            "train/epoch": epoch,
            "train/loss": loss,
            "train/perplexity": torch.exp(torch.tensor(loss)).item(),
            "train/learning_rate": learning_rate,
            "train/gradient_norm": grad_norm,
            # ACT metrics
            "act/avg_halting_steps": halting_steps.float().mean().item(),
            "act/max_halting_steps": halting_steps.max().item(),
            "act/min_halting_steps": halting_steps.min().item(),
            "act/halting_variance": halting_steps.float().var().item(),
            # LTM metrics
            "ltm/memory_usage": ltm_usage,
            # System metrics
            "system/gpu_memory_used_gb": gpu_memory_gb,
        }

        if gpu_util is not None:
            metrics["system/gpu_utilization"] = gpu_util

        if tokens_per_sec is not None:
            metrics["system/tokens_per_second"] = tokens_per_sec

        wandb.log(metrics)

    def log_epoch(
        self,
        epoch: int,
        val_loss: float,
        val_perplexity: float,
        val_accuracies: Dict[str, float],
        curriculum_stage: int,
        epoch_time_minutes: float,
    ) -> None:
        """
        Log metrics at end of epoch

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            val_perplexity: Validation perplexity
            val_accuracies: Dict of {dataset: accuracy}
            curriculum_stage: Current stage (1/2/3)
            epoch_time_minutes: Epoch duration
        """
        metrics = {
            "val/loss": val_loss,
            "val/perplexity": val_perplexity,
            "curriculum/stage": curriculum_stage,
            "time/epoch_duration_minutes": epoch_time_minutes,
        }

        # Add validation accuracies
        for dataset, acc in val_accuracies.items():
            metrics[f"val/{dataset}_accuracy"] = acc

        wandb.log(metrics)

    def log_final(
        self,
        total_params: int,
        training_time_hours: float,
        final_loss: float,
        final_perplexity: float,
        model_size_mb: float,
        diversity_metrics: Dict[str, float],
    ) -> None:
        """
        Log final metrics at end of training

        Args:
            total_params: Total model parameters
            training_time_hours: Total training time
            final_loss: Final training loss
            final_perplexity: Final perplexity
            model_size_mb: Model size in MB
            diversity_metrics: Diversity scores
        """
        metrics = {
            "final/total_params": total_params,
            "final/training_time_hours": training_time_hours,
            "final/train_loss": final_loss,
            "final/perplexity": final_perplexity,
            "final/model_size_mb": model_size_mb,
        }

        # Add diversity metrics
        for key, value in diversity_metrics.items():
            metrics[f"diversity/{key}"] = value

        wandb.log(metrics)

        # Create diversity comparison table
        diversity_table = wandb.Table(
            columns=["Metric", "Value"], data=[[k, v] for k, v in diversity_metrics.items()]
        )
        wandb.log({"diversity/metrics_table": diversity_table})

    def log_model_artifact(self, model_path: str, metadata: Dict) -> None:
        """
        Log model as W&B artifact

        Args:
            model_path: Path to saved model
            metadata: Model metadata
        """
        artifact = wandb.Artifact(
            name=f"phase1-{self.model_name}-checkpoint",
            type="model",
            description=f"Phase 1 {self.model_name} model checkpoint",
            metadata=metadata,
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Finish W&B run"""
        wandb.finish()
