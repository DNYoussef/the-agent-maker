"""
Configuration classes for Phase 5 curriculum learning.

Defines specialization types and curriculum configuration parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class SpecializationType(Enum):
    """Types of agent specialization."""
    CODING = "coding"
    RESEARCH = "research"
    WRITING = "writing"
    REASONING = "reasoning"
    GENERAL = "general"


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    # Assessment
    edge_of_chaos_threshold: float = 0.75  # 75% accuracy target
    assessment_questions: int = 2000

    # Curriculum
    num_levels: int = 10
    questions_per_level: int = 2000
    frontier_models: List[str] = field(default_factory=lambda: [
        "gpt-4", "claude-3.5", "gemini", "llama-3"
    ])

    # Training
    consecutive_successes_for_mastery: int = 3
    max_hints_per_question: int = 5
    variant_generation_enabled: bool = True

    # Self-modeling
    base_temperature_width: float = 0.2
    temperature_width_growth: float = 0.1
    base_num_ranges: int = 10

    # Dream consolidation
    dream_temperature: float = 1.5
    dream_training_temperature: float = 0.8
    dream_samples: int = 1000

    # Prompt baking
    baking_time_minutes: float = 5.0
    bake_after_each_level: bool = True

    # Specialization
    specialization: SpecializationType = SpecializationType.CODING
