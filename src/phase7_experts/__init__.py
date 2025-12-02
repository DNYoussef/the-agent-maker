# Phase 7: Self-Guided Experts
"""
Phase 7: Self-Guided Experts

3-Stage Pipeline:
1. Expert Discovery - Model self-analyzes to determine expert count (N=3-10)
2. SVF Training - Transformer^2 Singular Value Fine-tuning
3. ADAS Optimization - NSGA-II architecture search for routing

Research:
- "Transformer^2: Self-adaptive LLMs"
- "NSGA-II: Non-dominated Sorting Genetic Algorithm"
- "Automated Design of Agentic Systems"

Key insight: Model-driven expert discovery, not manual design.
"""

from .experts_engine import (
    ExpertsEngine,
    ExpertsConfig,
    Phase7Result
)
from .expert_discovery import (
    ExpertDiscovery,
    ExpertProfile,
    DiscoveryConfig
)
from .svf_trainer import (
    SVFTrainer,
    SVFConfig,
    SVFResult,
    REINFORCEConfig,
    REINFORCETrainer,
    SVFPolicy
)
from .transformer2 import (
    Transformer2,
    Transformer2Config,
    Transformer2Result,
    ExpertAdapter,
    Router,
    ZVectorSparseRouter
)
from .adas_optimizer import (
    ADASOptimizer,
    ADASConfig,
    ADASResult,
    Individual
)

__all__ = [
    # Main engine
    'ExpertsEngine',
    'ExpertsConfig',
    'Phase7Result',
    # Expert discovery
    'ExpertDiscovery',
    'ExpertProfile',
    'DiscoveryConfig',
    # SVF training
    'SVFTrainer',
    'SVFConfig',
    'SVFResult',
    # REINFORCE training
    'REINFORCEConfig',
    'REINFORCETrainer',
    'SVFPolicy',
    # Transformer2 two-pass inference
    'Transformer2',
    'Transformer2Config',
    'Transformer2Result',
    'ExpertAdapter',
    'Router',
    'ZVectorSparseRouter',
    # ADAS optimization
    'ADASOptimizer',
    'ADASConfig',
    'ADASResult',
    'Individual',
]
