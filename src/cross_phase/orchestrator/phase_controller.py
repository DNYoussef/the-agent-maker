"""
PhaseController - Re-exports from canonical sources.

AGM-003: Consolidated phase controllers to eliminate duplication.
All phase controllers are defined in their individual files:
- phase1_controller.py through phase8_controller.py

This file provides backward compatibility for imports like:
    from cross_phase.orchestrator.phase_controller import Phase1Controller

Canonical imports should use:
    from cross_phase.orchestrator import Phase1Controller
"""

# Re-export base classes
from .base_controller import PhaseController, PhaseResult

# Re-export all phase controllers from their canonical locations
from .phase1_controller import Phase1Controller
from .phase2_controller import Phase2Controller
from .phase3_controller import Phase3Controller
from .phase4_controller import Phase4Controller
from .phase5_controller import Phase5Controller
from .phase6_controller import Phase6Controller
from .phase7_controller import Phase7Controller
from .phase8_controller import Phase8Controller

# Re-export constants for backward compatibility
# ISS-015/ISS-022: Import constants from cross_phase.constants
from src.cross_phase.constants import (
    CURRICULUM_LEVELS,
    EVOMERGE_GENERATIONS,
    MIN_EXPERTS,
    ValidationThresholds,
)

# ISS-016: Import unified tokenizer utility
from src.cross_phase.utils import MockTokenizer, get_tokenizer

__all__ = [
    # Base classes
    "PhaseResult",
    "PhaseController",
    # Phase controllers
    "Phase1Controller",
    "Phase2Controller",
    "Phase3Controller",
    "Phase4Controller",
    "Phase5Controller",
    "Phase6Controller",
    "Phase7Controller",
    "Phase8Controller",
    # Constants
    "CURRICULUM_LEVELS",
    "EVOMERGE_GENERATIONS",
    "MIN_EXPERTS",
    "ValidationThresholds",
    # Utilities
    "MockTokenizer",
    "get_tokenizer",
]
