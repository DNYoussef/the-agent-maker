# Phase 5: Curriculum Learning
"""
Phase 5: Specialized Curriculum Training

7-Stage Pipeline for AI Agent Specialization:
1. Assessment - Edge-of-chaos detection (75% accuracy threshold)
2. Curriculum Generation - 20,000 questions via frontier models
3. Training Loop - Recursive thinking + tool use + variants/hints
4. Prompt Baking - Eudaimonia + OODA + Identity
5. Self-Modeling - Temperature range prediction
6. Dream Consolidation - Memory preservation
7. Level Progression - 10 progressively harder levels

Research Foundation:
- "Intelligence at the Edge of Chaos"
- "Unexpected Benefits of Self-Modeling in Neural Systems"
- "Dreaming Is All You Need"
"""

from .assessment import AssessmentResult, EdgeOfChaosAssessment
from .curriculum_engine import (
    CurriculumConfig,
    CurriculumEngine,
    LevelProgress,
    Phase5Result,
    SpecializationType,
)
from .curriculum_generator import AdaptiveCurriculumGenerator, Question
from .docker_sandbox import DockerSandbox, ExecutionResult, Language, SandboxConfig
from .dream_consolidation import DreamConfig, DreamConsolidator
from .eudaimonia import (
    ArchetypeCouncil,
    ArchetypeType,
    EudaimoniaRuleSystem,
    EudaimoniaScore,
    OODADecision,
    OODALoop,
    RuleType,
)

# New M5 implementations
from .openrouter_client import (
    CompletionResponse,
    ModelProvider,
    OpenRouterClient,
    get_free_models,
    get_production_models,
)
from .self_modeling import SelfModelingTrainer, TemperatureRange
from .training_loop import CurriculumTrainingLoop, TrainingMetrics
from .wandb_logger import EudaimoniaMetrics, MetricsConfig, Phase5WandBLogger, create_phase5_logger

__all__ = [
    # Main engine
    "CurriculumEngine",
    "CurriculumConfig",
    "Phase5Result",
    "LevelProgress",
    "SpecializationType",
    # Assessment
    "EdgeOfChaosAssessment",
    "AssessmentResult",
    # Curriculum
    "AdaptiveCurriculumGenerator",
    "Question",
    # Training
    "CurriculumTrainingLoop",
    "TrainingMetrics",
    # Self-modeling
    "SelfModelingTrainer",
    "TemperatureRange",
    # Dream consolidation
    "DreamConsolidator",
    "DreamConfig",
    # OpenRouter API
    "OpenRouterClient",
    "ModelProvider",
    "CompletionResponse",
    "get_free_models",
    "get_production_models",
    # Docker Sandbox
    "DockerSandbox",
    "SandboxConfig",
    "ExecutionResult",
    "Language",
    # W&B Logger
    "Phase5WandBLogger",
    "MetricsConfig",
    "EudaimoniaMetrics",
    "create_phase5_logger",
    # Eudaimonia System
    "EudaimoniaRuleSystem",
    "RuleType",
    "EudaimoniaScore",
    "ArchetypeCouncil",
    "ArchetypeType",
    "OODALoop",
    "OODADecision",
]
