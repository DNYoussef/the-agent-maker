"""
Eudaimonia Alignment System for Phase 5

A two-part ethical framework baked into model weights:
1. Four-Rule System - Hierarchical ethical decision framework
2. Three-Part Moral Compass - Philosophical archetype averaging

Based on the 589-line specification in PHASE5_EUDAIMONIA_SYSTEM.md.

Research Foundation:
- Beauchamp & Childress biomedical ethics (adapted for AI)
- Aristotelian virtue ethics (eudaimonia concept)
- OODA loop decision-making (military strategy)
"""
from .archetypes import (
    ArchetypeCouncil,
    ArchetypeGuidance,
    ArchetypeType,
    ChristArchetype,
    HarmonyArchetype,
    StoicArchetype,
)
from .ooda_loop import OODADecision, OODALoop, OODAState, SmallestMeasurableAction
from .rules import (
    EudaimoniaRuleSystem,
    EudaimoniaScore,
    RuleAssessment,
    RuleType,
    calculate_eudaimonia_score,
)

__all__ = [
    # Rules
    "EudaimoniaRuleSystem",
    "RuleType",
    "RuleAssessment",
    "EudaimoniaScore",
    "calculate_eudaimonia_score",
    # Archetypes
    "ArchetypeCouncil",
    "ArchetypeType",
    "ArchetypeGuidance",
    "ChristArchetype",
    "HarmonyArchetype",
    "StoicArchetype",
    # OODA
    "OODALoop",
    "OODAState",
    "OODADecision",
    "SmallestMeasurableAction",
]
