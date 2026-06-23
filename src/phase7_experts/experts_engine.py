"""
Phase 7: Self-Guided Experts Engine

Main orchestrator for the 3-stage expert training pipeline:
1. Expert Discovery - Model self-analyzes to determine experts
2. SVF Training - Train experts via singular value fine-tuning
3. ADAS Optimization - Architecture search for optimal routing

Research: Transformer^2 SVF, NSGA-II ADAS
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from src.cross_phase.monitoring.wandb_integration import WandBIntegration
from .adas_optimizer import ADASConfig, ADASOptimizer
from .expert_discovery import DiscoveryConfig, ExpertDiscovery
from .svf_trainer import SVFConfig, SVFTrainer


@dataclass
class ExpertsConfig:
    """Configuration for Phase 7 expert system."""

    # Discovery settings
    min_experts: int = 3
    max_experts: int = 10
    discovery_samples: int = 100

    # SVF settings
    svf_epochs: int = 5
    svf_learning_rate: float = 1e-4
    num_singular_values: int = 32

    # ADAS settings
    adas_population: int = 50
    adas_generations: int = 100
    mutation_rate: float = 0.1


@dataclass
class Phase7Result:
    """Result from Phase 7 expert training."""

    success: bool
    model: nn.Module
    num_experts: int
    expert_profiles: List[Any]
    routing_config: Dict
    metrics: Dict
    artifacts: Dict
    duration: float
    error: Optional[str] = None


class ExpertsEngine:
    """
    Phase 7: Self-Guided Experts Engine.

    Three-stage pipeline:
    1. Discovery: Model self-analyzes to find natural expert groupings
    2. SVF Training: Train each expert via Transformer^2 SVF
    3. ADAS: Optimize routing architecture via NSGA-II

    Key V2 innovation: Model-driven expert count (N=3-10) vs manual design.
    """

    def __init__(
        self,
        config: ExpertsConfig = None,
        wandb_integration: Optional[WandBIntegration] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize experts engine.

        Args:
            config: Expert configuration
        """
        self.config = config or ExpertsConfig()
        self.metrics = {
            "discovery_time": 0.0,
            "svf_time": 0.0,
            "adas_time": 0.0,
            "expert_metrics": [],
        }
        self.wandb = wandb_integration
        self.session_id = session_id

    def run(self, model: nn.Module, tokenizer: Any) -> Phase7Result:
        """
        Execute Phase 7 expert training pipeline.

        Args:
            model: Model from Phase 6
            tokenizer: Tokenizer

        Returns:
            Phase7Result with expert-enhanced model
        """
        print("\n" + "=" * 60)
        print("PHASE 7: SELF-GUIDED EXPERTS")
        print("=" * 60 + "\n")

        start_time = time.time()

        try:
            if self.wandb:
                self.wandb.init_phase_run(
                    phase_name="phase7",
                    config={
                        "min_experts": self.config.min_experts,
                        "max_experts": self.config.max_experts,
                        "svf_epochs": self.config.svf_epochs,
                        "adas_generations": self.config.adas_generations,
                    },
                    session_id=self.session_id or "phase7",
                )

            # Stage 1: Expert Discovery
            stage1_start = time.time()
            discovery_config = DiscoveryConfig(
                min_experts=self.config.min_experts,
                max_experts=self.config.max_experts,
                discovery_samples=self.config.discovery_samples,
            )
            discovery = ExpertDiscovery(config=discovery_config)

            num_experts, expert_profiles = discovery.discover(model, tokenizer)
            self.metrics["discovery_time"] = time.time() - stage1_start
            if self.wandb:
                self.wandb.log_metrics(
                    {
                        "phase7/discovery_time": self.metrics["discovery_time"],
                        "phase7/num_experts": num_experts,
                    }
                )

            # Stage 2: SVF Training
            stage2_start = time.time()
            svf_config = SVFConfig(
                num_singular_values=self.config.num_singular_values,
                num_epochs=self.config.svf_epochs,
                learning_rate=self.config.svf_learning_rate,
            )

            print("\nStage 2: SVF Expert Training")
            print("-" * 40)

            current_model = model
            svf_results = []

            for expert in expert_profiles:
                trainer = SVFTrainer(config=svf_config)
                trained_model, result = trainer.train_expert(
                    model=current_model,
                    expert_id=expert.id,
                    expert_capabilities=expert.capabilities,
                    tokenizer=tokenizer,
                )

                if result.success:
                    current_model = trained_model
                    svf_results.append(result)
                    self.metrics["expert_metrics"].append(
                        {
                            "expert_id": expert.id,
                            "final_loss": result.final_loss,
                            "sv_changes": result.sv_changes,
                        }
                    )

            self.metrics["svf_time"] = time.time() - stage2_start
            if self.wandb:
                self.wandb.log_metrics({"phase7/svf_time": self.metrics["svf_time"]})

            # Stage 3: ADAS Optimization
            stage3_start = time.time()
            adas_config = ADASConfig(
                population_size=self.config.adas_population,
                num_generations=self.config.adas_generations,
                mutation_rate=self.config.mutation_rate,
            )

            adas = ADASOptimizer(config=adas_config)
            try:
                optimized_model, adas_result = adas.optimize(
                    model=current_model, experts=expert_profiles, tokenizer=tokenizer
                )
            except RuntimeError as exc:
                # Wave-0: ADAS fitness needs a real evaluator ("measurement truth",
                # Wave 1). evaluation.py correctly refuses synthetic scores and
                # raises. Until an evaluator is wired, skip ADAS HONESTLY (keep the
                # SVF-optimized model) instead of letting the broad except below
                # swallow it into a silent success=False.
                logger.warning("Phase 7 ADAS skipped (no fitness evaluator): %s", exc)
                optimized_model = current_model
                adas_result = {"adas_skipped": True, "reason": str(exc)}
            self.metrics["adas_time"] = time.time() - stage3_start
            if self.wandb:
                self.wandb.log_metrics({"phase7/adas_time": self.metrics["adas_time"]})

            # Extract routing configuration
            routing_config = {}
            if hasattr(optimized_model, "_expert_routing"):
                routing_config = optimized_model._expert_routing

            total_duration = time.time() - start_time

            print(f"\nPhase 7 Complete:")
            print(f"  Discovered experts: {num_experts}")
            print(f"  SVF training: {len(svf_results)} experts trained")
            print(f"  ADAS generations: {self.config.adas_generations}")
            print(f"  Total time: {total_duration:.1f}s")

            return Phase7Result(
                success=True,
                model=optimized_model,
                num_experts=num_experts,
                expert_profiles=expert_profiles,
                routing_config=routing_config,
                metrics=self.metrics,
                artifacts={
                    "svf_results": svf_results,
                    "adas_result": adas_result,
                    "discovery": discovery,
                },
                duration=total_duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return Phase7Result(
                success=False,
                model=model,
                num_experts=0,
                expert_profiles=[],
                routing_config={},
                metrics=self.metrics,
                artifacts={},
                duration=duration,
                error=str(e),
            )
        finally:
            if self.wandb:
                self.wandb.finish()


__all__ = [
    "ADASConfig",
    "ADASOptimizer",
    "DiscoveryConfig",
    "ExpertDiscovery",
    "ExpertsEngine",
    "ExpertsConfig",
    "Phase7Result",
    "SVFConfig",
    "SVFTrainer",
]
