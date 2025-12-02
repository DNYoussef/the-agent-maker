"""
Weights & Biases Integration for Phase 5 Curriculum Learning

Comprehensive logging for 7,200+ metrics across 10 levels.

Metric Categories:
- Assessment metrics (edge-of-chaos detection)
- Curriculum metrics (questions, variants, hints)
- Training metrics (loss, accuracy, convergence)
- Self-modeling metrics (temperature prediction)
- Dream consolidation metrics (memory retention)
- Eudaimonia metrics (moral alignment scores)
- Level progression metrics (completion, time)

Usage:
    logger = Phase5WandBLogger(project="agent-forge-v2")
    logger.start_run("curriculum_training_run_001")

    # Log assessment
    logger.log_assessment(level=1, accuracy=0.75, threshold=0.75)

    # Log training step
    logger.log_training_step(
        level=1, step=100, loss=0.5, accuracy=0.78
    )

    # Finish run
    logger.finish()
"""
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import wandb, but don't fail if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Logging to console only.")


@dataclass
class MetricsConfig:
    """Configuration for metrics logging."""
    project: str = "agent-forge-v2"
    entity: Optional[str] = None
    enabled: bool = True
    log_to_console: bool = True
    log_frequency: int = 10  # Log every N steps


@dataclass
class AssessmentMetrics:
    """Metrics from edge-of-chaos assessment."""
    level: int
    accuracy: float
    threshold: float
    questions_tested: int
    baseline_difficulty: float
    time_seconds: float


@dataclass
class TrainingStepMetrics:
    """Metrics from a single training step."""
    level: int
    step: int
    loss: float
    accuracy: float
    learning_rate: float
    remaining_questions: int
    variants_generated: int
    hints_given: int


@dataclass
class SelfModelingMetrics:
    """Metrics from self-modeling training."""
    level: int
    temperature_ranges: int
    prediction_accuracy: float
    loss: float
    epochs: int


@dataclass
class DreamConsolidationMetrics:
    """Metrics from dream consolidation."""
    level: int
    dream_samples: int
    reconstruction_loss: float
    memory_retention: float
    consolidation_epochs: int


@dataclass
class EudaimoniaMetrics:
    """Metrics from Eudaimonia alignment."""
    level: int
    eudaimonia_score: float
    rule_1_score: float  # Prime directive
    rule_2_score: float  # Curiosity
    rule_3_score: float  # Esprit de corps
    rule_4_score: float  # Self-preservation
    archetype_alignment: Dict[str, float] = field(default_factory=dict)
    ooda_interventions: int = 0


@dataclass
class LevelCompletionMetrics:
    """Metrics for level completion."""
    level: int
    total_questions: int
    mastered_questions: int
    final_accuracy: float
    training_time_hours: float
    baking_time_minutes: float


class Phase5WandBLogger:
    """
    Comprehensive W&B logger for Phase 5 curriculum learning.

    Logs 7,200+ metrics across 10 levels:
    - ~720 metrics per level
    - 7 metric categories
    - Automatic summarization
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize W&B logger.

        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        self._run = None
        self._step_counter = 0
        self._level_metrics: Dict[int, Dict[str, Any]] = {}

        # Check if W&B is available and enabled
        self._enabled = WANDB_AVAILABLE and self.config.enabled

        if not WANDB_AVAILABLE and self.config.enabled:
            logger.warning(
                "W&B logging requested but wandb not installed. "
                "Install with: pip install wandb"
            )

    def start_run(
        self,
        run_name: str,
        run_config: Optional[Dict[str, Any]] = None,
        resume: Optional[str] = None
    ) -> bool:
        """
        Start a new W&B run.

        Args:
            run_name: Name for this run
            run_config: Configuration to log
            resume: Resume mode ('allow', 'must', 'never', or run ID)

        Returns:
            True if run started successfully
        """
        if not self._enabled:
            logger.info(f"[Console] Starting run: {run_name}")
            return False

        try:
            self._run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=run_name,
                config=run_config or {},
                resume=resume
            )
            logger.info(f"W&B run started: {run_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start W&B run: {e}")
            self._enabled = False
            return False

    def log_assessment(self, metrics: AssessmentMetrics):
        """Log edge-of-chaos assessment metrics."""
        data = {
            f"assessment/level_{metrics.level}/accuracy": metrics.accuracy,
            f"assessment/level_{metrics.level}/threshold": metrics.threshold,
            f"assessment/level_{metrics.level}/questions_tested": metrics.questions_tested,
            f"assessment/level_{metrics.level}/baseline_difficulty": metrics.baseline_difficulty,
            f"assessment/level_{metrics.level}/time_seconds": metrics.time_seconds,
        }
        self._log(data)

    def log_training_step(self, metrics: TrainingStepMetrics):
        """Log training step metrics."""
        data = {
            f"training/level_{metrics.level}/step": metrics.step,
            f"training/level_{metrics.level}/loss": metrics.loss,
            f"training/level_{metrics.level}/accuracy": metrics.accuracy,
            f"training/level_{metrics.level}/learning_rate": metrics.learning_rate,
            f"training/level_{metrics.level}/remaining_questions": metrics.remaining_questions,
            f"training/level_{metrics.level}/variants_generated": metrics.variants_generated,
            f"training/level_{metrics.level}/hints_given": metrics.hints_given,
        }

        # Only log at configured frequency
        if metrics.step % self.config.log_frequency == 0:
            self._log(data)
            self._step_counter += 1

    def log_self_modeling(self, metrics: SelfModelingMetrics):
        """Log self-modeling training metrics."""
        data = {
            f"self_modeling/level_{metrics.level}/temperature_ranges": metrics.temperature_ranges,
            f"self_modeling/level_{metrics.level}/prediction_accuracy": metrics.prediction_accuracy,
            f"self_modeling/level_{metrics.level}/loss": metrics.loss,
            f"self_modeling/level_{metrics.level}/epochs": metrics.epochs,
        }
        self._log(data)

    def log_dream_consolidation(self, metrics: DreamConsolidationMetrics):
        """Log dream consolidation metrics."""
        data = {
            f"dream/level_{metrics.level}/samples": metrics.dream_samples,
            f"dream/level_{metrics.level}/reconstruction_loss": metrics.reconstruction_loss,
            f"dream/level_{metrics.level}/memory_retention": metrics.memory_retention,
            f"dream/level_{metrics.level}/epochs": metrics.consolidation_epochs,
        }
        self._log(data)

    def log_eudaimonia(self, metrics: EudaimoniaMetrics):
        """Log Eudaimonia alignment metrics."""
        data = {
            f"eudaimonia/level_{metrics.level}/score": metrics.eudaimonia_score,
            f"eudaimonia/level_{metrics.level}/rule_1_prime_directive": metrics.rule_1_score,
            f"eudaimonia/level_{metrics.level}/rule_2_curiosity": metrics.rule_2_score,
            f"eudaimonia/level_{metrics.level}/rule_3_esprit_de_corps": metrics.rule_3_score,
            f"eudaimonia/level_{metrics.level}/rule_4_self_preservation": metrics.rule_4_score,
            f"eudaimonia/level_{metrics.level}/ooda_interventions": metrics.ooda_interventions,
        }

        # Log archetype alignment
        for archetype, score in metrics.archetype_alignment.items():
            data[f"eudaimonia/level_{metrics.level}/archetype_{archetype}"] = score

        self._log(data)

    def log_level_completion(self, metrics: LevelCompletionMetrics):
        """Log level completion summary."""
        data = {
            f"level_{metrics.level}/total_questions": metrics.total_questions,
            f"level_{metrics.level}/mastered_questions": metrics.mastered_questions,
            f"level_{metrics.level}/final_accuracy": metrics.final_accuracy,
            f"level_{metrics.level}/training_time_hours": metrics.training_time_hours,
            f"level_{metrics.level}/baking_time_minutes": metrics.baking_time_minutes,
            f"level_{metrics.level}/mastery_rate": (
                metrics.mastered_questions / metrics.total_questions
                if metrics.total_questions > 0 else 0.0
            ),
        }
        self._log(data)

        # Store for final summary
        self._level_metrics[metrics.level] = data

    def log_prompt_baking(
        self,
        level: int,
        prompt_type: str,
        loss: float,
        time_minutes: float
    ):
        """Log prompt baking metrics."""
        data = {
            f"baking/level_{level}/{prompt_type}_loss": loss,
            f"baking/level_{level}/{prompt_type}_time_minutes": time_minutes,
        }
        self._log(data)

    def log_curriculum_stats(
        self,
        level: int,
        questions_generated: int,
        model_distribution: Dict[str, int],
        difficulty_distribution: Dict[str, int]
    ):
        """Log curriculum generation statistics."""
        data = {
            f"curriculum/level_{level}/questions_generated": questions_generated,
        }

        for model, count in model_distribution.items():
            data[f"curriculum/level_{level}/model_{model}"] = count

        for difficulty, count in difficulty_distribution.items():
            data[f"curriculum/level_{level}/difficulty_{difficulty}"] = count

        self._log(data)

    def log_api_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float
    ):
        """Log OpenRouter API usage."""
        data = {
            "api/input_tokens": input_tokens,
            "api/output_tokens": output_tokens,
            "api/cost_usd": cost_usd,
            "api/latency_ms": latency_ms,
            f"api/model_{model}/calls": 1,
        }
        self._log(data)

    def log_custom(self, metrics: Dict[str, Any]):
        """Log custom metrics."""
        self._log(metrics)

    def _log(self, data: Dict[str, Any]):
        """Internal logging method."""
        if self._enabled and self._run:
            wandb.log(data, step=self._step_counter)

        if self.config.log_to_console:
            for key, value in data.items():
                if isinstance(value, float):
                    logger.debug(f"  {key}: {value:.4f}")
                else:
                    logger.debug(f"  {key}: {value}")

    def log_summary(self):
        """Log final run summary."""
        if not self._level_metrics:
            return

        summary = {
            "summary/total_levels": len(self._level_metrics),
            "summary/total_questions_mastered": sum(
                m.get(f"level_{l}/mastered_questions", 0)
                for l, m in self._level_metrics.items()
            ),
            "summary/total_training_hours": sum(
                m.get(f"level_{l}/training_time_hours", 0)
                for l, m in self._level_metrics.items()
            ),
            "summary/average_accuracy": (
                sum(
                    m.get(f"level_{l}/final_accuracy", 0)
                    for l, m in self._level_metrics.items()
                ) / len(self._level_metrics)
            ),
        }

        if self._enabled and self._run:
            for key, value in summary.items():
                wandb.run.summary[key] = value

        if self.config.log_to_console:
            logger.info("Run Summary:")
            for key, value in summary.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")

    def finish(self):
        """Finish the W&B run."""
        self.log_summary()

        if self._enabled and self._run:
            wandb.finish()
            logger.info("W&B run finished")

        self._run = None

    def is_enabled(self) -> bool:
        """Check if W&B logging is enabled."""
        return self._enabled


# Convenience function for creating logger
def create_phase5_logger(
    project: str = "agent-forge-v2",
    enabled: bool = True
) -> Phase5WandBLogger:
    """
    Create a Phase 5 W&B logger.

    Args:
        project: W&B project name
        enabled: Whether to enable logging

    Returns:
        Configured Phase5WandBLogger
    """
    config = MetricsConfig(project=project, enabled=enabled)
    return Phase5WandBLogger(config)


__all__ = [
    "Phase5WandBLogger",
    "MetricsConfig",
    "AssessmentMetrics",
    "TrainingStepMetrics",
    "SelfModelingMetrics",
    "DreamConsolidationMetrics",
    "EudaimoniaMetrics",
    "LevelCompletionMetrics",
    "create_phase5_logger",
    "WANDB_AVAILABLE"
]
