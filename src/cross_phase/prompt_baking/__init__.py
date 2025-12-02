"""
Prompt Baking Package

Converts prompts into weight updates.
"""

from .baker import PromptBaker
from .prompts import PhasePrompts, PromptManager

__all__ = ["PromptBaker", "PhasePrompts", "PromptManager"]
