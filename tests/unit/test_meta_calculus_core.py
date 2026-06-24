"""Regression tests for meta-calculus integration bugs."""

import math

import numpy as np
import torch


def test_meta_calculus_public_imports():
    import src.cross_phase.meta_calculus as meta
    from src.cross_phase.meta_calculus.phase_facades import phase1, phase2, phase8

    assert meta.__version__
    assert phase1.PHASE1_CONFIG.grokfast_alpha > 0
    assert callable(phase2.get_merge_ratios)
    assert callable(phase8.fit_weights_log_space)


def test_k_from_parameter_variance_accepts_scalars_and_singletons():
    from src.cross_phase.meta_calculus.k_formula import k_from_parameter_variance

    scalar_k = k_from_parameter_variance(0.1)
    singleton_k = k_from_parameter_variance(torch.tensor([1.0]))

    assert math.isfinite(float(scalar_k))
    assert torch.isfinite(singleton_k)


def test_thought_count_and_single_thought_diversity_are_defined():
    from src.cross_phase.meta_calculus.k_utils.adaptive import get_thought_count
    from src.cross_phase.meta_calculus.spectral_gap import (
        compute_thought_diversity,
        thought_diversity_loss,
    )

    assert 1 <= get_thought_count(5.0) <= 8

    thought = torch.randn(1, 4)
    diversity = compute_thought_diversity(thought)
    loss = thought_diversity_loss(thought)

    assert diversity["diversity_score"] == 0.0
    assert diversity["is_collapsed"] is True
    assert torch.isfinite(loss)


def test_bigeometric_merge_preserves_common_negative_sign():
    from src.cross_phase.meta_calculus.transform_utils.weights import bigeometric_merge_tensors

    w1 = torch.tensor([-4.0])
    w2 = torch.tensor([-9.0])
    merged = bigeometric_merge_tensors(w1, w2, alpha=0.5)

    assert merged.item() < 0
    assert torch.isclose(merged.abs(), torch.tensor([6.0]), atol=1e-5).all()


def test_select_from_pareto_respects_maximize_objectives():
    from src.cross_phase.meta_calculus.moo_bridge import select_from_pareto

    result = {
        "pareto_front": np.array([[1.0, 0.9], [1.0, 0.1]]),
        "pareto_solutions": np.array([[10.0], [20.0]]),
        "objective_names": ["loss", "spectral_gap"],
        "objective_minimize": [True, False],
    }

    x, f = select_from_pareto(result, {"loss": 1.0, "spectral_gap": 2.0})

    assert x.tolist() == [10.0]
    assert f.tolist() == [1.0, 0.9]


def test_phase8_log_space_roundtrip_preserves_signs():
    from src.cross_phase.meta_calculus.phase_facades import phase8

    weights = torch.tensor([-2.0, -0.5, 0.0, 3.0])
    encoded = phase8.apply_log_space_transform(weights)
    decoded = phase8.reverse_log_space_transform(encoded)

    assert torch.allclose(decoded, weights, atol=1e-6)
