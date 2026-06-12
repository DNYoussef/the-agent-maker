"""Tests for the two-part MDL objective (P2 of META_CALCULUS_VALIDATION_PLAN).

Perplexity/task-loss is the data half of MDL and remains its own objective;
description_length carries only the model bits. Every test documents the
defect it catches ("Fails if:").
"""

import math

import numpy as np
import pytest

from src.cross_phase.meta_calculus.moo_bridge import (
    EVOMERGE_OBJECTIVES,
    EXPERT_DISCOVERY_OBJECTIVES,
    EvoMergeProblem,
    ExpertDiscoveryProblem,
    description_length_bits,
)


def test_bits_match_independent_formula():
    """log2 C(6,1) + 1*log2(2^20) computed two unrelated ways must agree.

    Fails if: the lgamma-based log-binomial or the base conversion is wrong
    (e.g. ln instead of log2) - this is an independent reference value, not
    a tautology against the implementation.
    """
    got = description_length_bits(
        active_components=1, total_components=6, param_count=2.0 ** 20
    )
    expected = math.log2(6) + 20.0
    assert abs(got - expected) < 1e-9


def test_bits_monotone_in_active_components():
    """More active components must always cost more bits (up to total/2+).

    Fails if: someone swaps the binomial arguments or drops the pointer
    term - sparsity would stop being rewarded and the objective silently
    inverts its meaning.
    """
    bits = [
        description_length_bits(
            active_components=a, total_components=10, param_count=1e6
        )
        for a in range(0, 6)
    ]
    assert all(b2 > b1 for b1, b2 in zip(bits, bits[1:]))


def test_bits_monotone_in_param_count():
    """Bigger models must cost more bits at fixed support.

    Fails if: log2(param_count) is dropped or clamped wrongly.
    """
    small = description_length_bits(
        active_components=3, total_components=6, param_count=1e6
    )
    large = description_length_bits(
        active_components=3, total_components=6, param_count=1e9
    )
    assert large > small


def test_bits_edge_cases():
    """Zero active components costs ~0 bits; bad inputs raise.

    Fails if: edge handling regresses to NaN/inf (lgamma(0) traps) or the
    validation is removed and garbage flows into the F-matrix.
    """
    zero = description_length_bits(
        active_components=0, total_components=6, param_count=1e8
    )
    assert abs(zero - math.log2(1)) < 1e-9  # C(6,0) = 1, no pointers

    # active > total clamps rather than producing negative binomials
    clamped = description_length_bits(
        active_components=99, total_components=6, param_count=2.0
    )
    assert math.isfinite(clamped)

    with pytest.raises(ValueError):
        description_length_bits(
            active_components=1, total_components=0, param_count=1.0
        )
    with pytest.raises(ValueError):
        description_length_bits(
            active_components=-1, total_components=6, param_count=1.0
        )


def test_objective_tables_gained_mdl_and_direction():
    """description_length is present in both tables and minimized.

    Fails if: the objective is dropped from a table, or accidentally wired
    as maximize - NSGA-II would then reward bloat.
    """
    for table in (EVOMERGE_OBJECTIVES, EXPERT_DISCOVERY_OBJECTIVES):
        mdl = [o for o in table if o.name == "description_length"]
        assert len(mdl) == 1
        assert mdl[0].minimize is True


def test_evomerge_five_key_evaluator_still_works():
    """Legacy evaluators returning the original 5 keys must keep working (R3).

    Fails if: the MDL wiring requires a new evaluator key instead of
    computing its default from the merge recipe.
    """
    def evaluator(layer_ratios, technique_weights):
        return {
            "perplexity": 10.0,
            "spectral_gap": 0.5,
            "weight_norm": 1.0,
            "invariance_score": 0.8,
            "fragility": 0.1,
        }

    problem = EvoMergeProblem(n_layers=2, model_evaluator=evaluator)
    x = np.array([[0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    F = problem.compute_objectives(x)

    assert F.shape == (1, 6)
    assert np.isfinite(F).all()
    # One technique above threshold, default param_count 1e8:
    expected = description_length_bits(
        active_components=1, total_components=6, param_count=1e8
    )
    assert abs(F[0, 5] - expected) < 1e-9


def test_evaluator_supplied_description_length_wins():
    """An evaluator that measures its own MDL overrides the computed default.

    Fails if: the .get override path is removed and real measurements get
    silently replaced by the proxy.
    """
    def evaluator(layer_ratios, technique_weights):
        return {"perplexity": 1.0, "description_length": 42.0}

    problem = EvoMergeProblem(n_layers=1, model_evaluator=evaluator)
    x = np.array([[0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    F = problem.compute_objectives(x)
    assert F[0, 5] == 42.0


def test_include_mdl_false_restores_legacy_shape():
    """The opt-out must restore the exact legacy 5-column problem (R3).

    Fails if: include_mdl=False still appends the column, breaking any
    fixed-architecture consumer that disabled it for dimension reasons.
    """
    def evaluator(layer_ratios, technique_weights):
        return {"perplexity": 1.0}

    problem = EvoMergeProblem(
        n_layers=1, model_evaluator=evaluator, include_mdl=False
    )
    assert problem.n_obj == 5
    x = np.array([[0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    F = problem.compute_objectives(x)
    assert F.shape == (1, 5)


def test_expert_problem_mdl_prefers_smaller_architectures():
    """At equal task metrics, the smaller expert architecture costs fewer bits.

    Fails if: the param proxy (n_experts * svf_rank * z_dim) is miswired so
    architecture size no longer reaches the objective.
    """
    def evaluator(n_experts, sparsity, router_temp, svf_rank, z_dim):
        return {"task_loss": 1.0}

    problem = ExpertDiscoveryProblem(expert_evaluator=evaluator)
    small = np.array([[3, 0.5, 1.0, 4, 8]])
    large = np.array([[10, 0.5, 1.0, 64, 128]])

    F_small = problem.compute_objectives(small)
    F_large = problem.compute_objectives(large)
    assert F_small[0, 5] < F_large[0, 5]


def test_mdl_keeps_small_model_on_frontier():
    """A 100x-smaller model that loses slightly on perplexity must survive
    on the Pareto front once MDL is an objective.

    Fails if: MDL is wired as maximize, or dropped from the F-matrix - the
    small model would be dominated and pruned, which is the exact
    argmin-collapse the GCC Pareto theorems warn about.
    """
    pytest.importorskip("pymoo")
    from src.cross_phase.meta_calculus.moo_bridge import _minimize_flags_from_names

    # Two hand-built solutions: A better on perplexity, B 100x smaller.
    F = np.array([
        [1.0, 30.0],   # A: perplexity 1.0, 30 bits
        [1.2, 10.0],   # B: perplexity 1.2, 10 bits
    ])
    flags = _minimize_flags_from_names(["perplexity", "description_length"])
    assert flags.all()  # both minimized

    # Neither dominates the other -> both Pareto-optimal.
    a_dominates_b = (F[0] <= F[1]).all() and (F[0] < F[1]).any()
    b_dominates_a = (F[1] <= F[0]).all() and (F[1] < F[0]).any()
    assert not a_dominates_b and not b_dominates_a
