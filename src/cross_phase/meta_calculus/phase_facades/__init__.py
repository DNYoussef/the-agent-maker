"""
Phase Facades (Layer 3)

Simple per-phase imports that provide exactly what each phase needs.
This is the public API for phase developers.

Usage:
    from src.cross_phase.meta_calculus.phase_facades import phase1
    optimizer = phase1.create_optimizer(model)
    monitor = phase1.create_gap_monitor()

    from src.cross_phase.meta_calculus.phase_facades import phase2
    ratios = phase2.get_merge_ratios(num_layers=8)
    result = phase2.run_moo(evaluator)

Design:
    - Each phase module re-exports ONLY what that phase needs
    - No phase needs to understand the full meta_calculus API
    - Facade pattern: simple interface hiding complex implementation
"""

from . import phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8

__all__ = [
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
]
