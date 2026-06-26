"""Branch-deletion audit capture: MOOCurriculumGenerator._generate_placeholder_questions
called self._get_question_templates(difficulty), a method that does not exist
(the real one is _get_templates_for_difficulty) -> AttributeError on every call
to the placeholder generation path. This was still present on main.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.phase5_curriculum.curriculum_generator import (
    MOOCurriculumGenerator,
    Question,
)
from src.phase5_curriculum.curriculum_engine import SpecializationType


def test_generate_placeholder_questions_does_not_raise_attribute_error():
    gen = MOOCurriculumGenerator.__new__(MOOCurriculumGenerator)
    gen.specialization = SpecializationType.CODING  # used by _get_templates_for_difficulty
    questions = gen._generate_placeholder_questions(difficulty=3, level=1, count=2)
    assert len(questions) == 2
    assert all(isinstance(q, Question) for q in questions)
