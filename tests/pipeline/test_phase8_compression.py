"""E9 gate - CRUCIBLE: Phase 8 acceptance does not green a no-op / failed compression.

Synthesis: validate_output passed on total_compression>=1.0 (a 1.0x no-op compression shipped
green) and ignored result.success (a rolled-back/corrupted run still passed). E9 requires a
REAL ratio (>1.0) and result.success. (Making the engine actually carry compressed_state /
real SeedLM is the deeper follow-on noted in the commit.)
"""

from cross_phase.orchestrator.base_controller import PhaseResult
from cross_phase.orchestrator.phase8_controller import Phase8Controller


def _result(success=True, compression=2.0, retention=0.9):
    return PhaseResult(
        success,
        "phase8",
        object(),
        {"total_compression": compression, "retention_score": retention},
        0.0,
        {},
        {},
    )


def _ctrl():
    return Phase8Controller(config={}, session_id="t")


def test_accepts_real_compression():
    assert _ctrl().validate_output(_result(compression=2.0)) is True


def test_rejects_noop_compression():
    assert _ctrl().validate_output(_result(compression=1.0)) is False, "1.0x is no compression"


def test_rejects_failed_run():
    assert _ctrl().validate_output(_result(success=False)) is False


def test_rejects_empty_metrics():
    r = PhaseResult(True, "phase8", object(), {}, 0.0, {}, {})
    assert _ctrl().validate_output(r) is False, "no metrics -> cannot claim compression"
