"""
Phase 5 Curriculum Learning Engine

Re-exports all classes for backward compatibility.
"""

from .config import CurriculumConfig, SpecializationType
from .progress import LevelProgress
from .result import Phase5Result
from .curriculum_engine import CurriculumEngine

__all__ = [
    'CurriculumEngine',
    'CurriculumConfig',
    'Phase5Result',
    'LevelProgress',
    'SpecializationType'
]
