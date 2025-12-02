"""
Progress tracking for curriculum levels.

Tracks question progress, mastery, and accuracy through curriculum levels.
"""

from dataclasses import dataclass


@dataclass
class LevelProgress:
    """Track progress through a curriculum level."""

    level: int
    initial_questions: int
    current_questions: int
    mastered_questions: int
    variants_generated: int
    hints_given: int
    accuracy: float
    completed: bool = False
