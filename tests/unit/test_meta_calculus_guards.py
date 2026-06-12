"""Guard tests for meta-calculus landmines (P1 of META_CALCULUS_VALIDATION_PLAN).

Every test documents the defect it catches: see the "Fails if:" line in each
docstring. Deliberate-break protocol: disabling the guard under test must make
the test fail.
"""

import logging
import math

import numpy as np
import pytest
import torch

from src.cross_phase.meta_calculus import bigeometric as bg_module
from src.cross_phase.meta_calculus.bigeometric import (
    BigeometricConfig,
    BigeometricTransform,
)
from src.cross_phase.meta_calculus import moo_bridge
from src.cross_phase.meta_calculus.moo_bridge import select_from_pareto


@pytest.fixture(autouse=True)
def _reset_warn_once():
    """Each test sees the warn-once latch in its initial state."""
    bg_module._K_CLAMP_WARNED = False
    yield
    bg_module._K_CLAMP_WARNED = False


# ---------------------------------------------------------------------------
# 1.1 pymoo fail-fast
# ---------------------------------------------------------------------------


def test_moorunner_fails_fast_without_pymoo(monkeypatch):
    """MOORunner must raise at construction, not deep inside pymoo.

    Fails if: someone reorders __init__ so pymoo machinery is touched before
    the availability guard, or removes the guard - Phases 2/7 would then
    crash with an opaque AttributeError mid-optimization.
    """
    monkeypatch.setattr(moo_bridge, "PYMOO_AVAILABLE", False)
    with pytest.raises(ImportError, match="pip install pymoo"):
        moo_bridge.MOORunner()


def test_globalmoo_adapter_reports_unavailable():
    """The cloud adapter must self-report unavailable before submit_problem.

    Fails if: is_available() starts returning True while submit_problem is
    still a NotImplementedError stub - callers checking the gate would then
    walk into the stub.
    """
    adapter = moo_bridge.GlobalMOOAdapter()
    assert adapter.is_available() is False
    # Without a key it must fail fast (RuntimeError); with a key the client
    # is still unimplemented (NotImplementedError). Either way: never silent.
    with pytest.raises((RuntimeError, NotImplementedError)):
        adapter.submit_problem(problem=None, config=None)


# ---------------------------------------------------------------------------
# 1.2 k-range guardrail
# ---------------------------------------------------------------------------


def test_out_of_range_k_is_clamped_and_warns():
    """k <= 0 is the noise-floor-amplification regime; it must be clamped.

    Unclamped, k=-0.1 maps g=1e-6 to ~1.0 (a zero-ish gradient becomes an
    order-1 update via the max_magnitude cap). Clamped to k_min=0.05 the
    output is |g|^(2*0.05) ~ 0.25.

    Fails if: the clamp in BigeometricTransform.transform is removed - the
    output magnitude jumps from ~0.25 to ~1.0 and the warning disappears.
    """
    t = BigeometricTransform()
    g = torch.tensor([1e-6])

    with pytest.warns(UserWarning, match="safe window"):
        out = t.transform(g, k=-0.1)

    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 0.5, (
        f"clamped output should be ~0.25, got {out.abs().max().item():.4g}"
    )
    # Sign must survive the clamp path too.
    assert torch.sign(out).item() == 1.0


def test_in_range_k_is_untouched_and_silent(recwarn):
    """Clamping must not perturb in-range behavior (R3 regression guard).

    Fails if: the guardrail alters k inside [k_min, k_max], or warns on
    healthy values - production k(L) lives at 0.07-0.16 and must be
    bit-identical to the raw formula.
    """
    g = torch.tensor([-4.0, -0.5, 0.5, 4.0])
    k = 0.3

    out_guarded = BigeometricTransform().transform(g, k=k)
    out_raw = BigeometricTransform(BigeometricConfig(clamp_k=False)).transform(g, k=k)

    # Identical code path for in-range k: bitwise equality, not allclose.
    assert torch.equal(out_guarded, out_raw)
    assert len([w for w in recwarn if "safe window" in str(w.message)]) == 0


def test_clamp_optout_reproduces_legacy_behavior(recwarn):
    """clamp_k=False must reproduce the raw transform exactly (research door).

    Fails if: the opt-out flag is wired through incompletely, silently
    changing exploratory sweeps that scan k outside (0, 0.5].
    """
    t = BigeometricTransform(BigeometricConfig(clamp_k=False))
    g = torch.tensor([1e-6, 2.0])
    k = -0.1

    out = t.transform(g, k=k)

    eps = t.config.eps
    scale = torch.abs(g).clamp(min=eps) ** torch.tensor(2 * k - 1)
    scale = scale.clamp(max=t.config.max_magnitude)
    expected = g * scale
    assert torch.allclose(out, expected)
    assert len([w for w in recwarn if "safe window" in str(w.message)]) == 0


def test_clamp_warns_only_once_per_process():
    """The warning is a once-per-process latch, not a per-step log flood.

    Fails if: the latch is dropped - a training run with a bad k would emit
    one warning per optimizer step.
    """
    import warnings as _warnings

    t = BigeometricTransform()
    g = torch.tensor([1.0])

    with pytest.warns(UserWarning, match="safe window"):
        t.transform(g, k=0.9)

    with _warnings.catch_warnings(record=True) as record:
        _warnings.simplefilter("always")
        t.transform(g, k=0.9)
    assert len([w for w in record if "safe window" in str(w.message)]) == 0


def test_large_gradient_contracts_in_safe_window():
    """The anti-explosion property itself: |out| < |g| for |g| > 1, 0<k<0.5.

    This is the Lean theorem bgTransform_contracts checked numerically on
    the production code path.

    Fails if: the exponent wiring changes (e.g. 2k-1 -> 2k+1) so the
    transform amplifies instead of contracting - the no-explosion claim
    would be silently false while every happy-path test still passes.
    """
    t = BigeometricTransform()
    g = torch.tensor([3.0, -7.0, 100.0])
    out = t.transform(g, k=0.16)
    assert (out.abs() < g.abs()).all()
    assert torch.equal(torch.sign(out), torch.sign(g))


# ---------------------------------------------------------------------------
# 1.4 governance logging in select_from_pareto
# ---------------------------------------------------------------------------


def test_select_from_pareto_logs_choice(caplog):
    """Frontier collapse (weights -> single point) must leave an audit line.

    Fails if: someone removes the log call or changes selection without
    logging - the regime choice becomes unauditable, which is the exact
    failure the GCC Pareto theorems exist to prevent.
    """
    result = {
        "pareto_front": np.array([[1.0, 0.9], [1.0, 0.1]]),
        "pareto_solutions": np.array([[10.0], [20.0]]),
        "objective_names": ["loss", "spectral_gap"],
        "objective_minimize": [True, False],
    }
    preference = {"loss": 1.0, "spectral_gap": 2.0}

    with caplog.at_level(logging.INFO, logger="src.cross_phase.meta_calculus.moo_bridge"):
        x, f = select_from_pareto(result, preference)

    audit_lines = [r for r in caplog.records if "pareto-select" in r.getMessage()]
    assert len(audit_lines) == 1
    msg = audit_lines[0].getMessage()
    assert "front_size=2" in msg
    assert "spectral_gap" in msg
    # Selection itself still respects the maximize flag (regression from the
    # existing core test, re-checked on the logged path).
    assert x.tolist() == [10.0]
    assert f.tolist() == [1.0, 0.9]
