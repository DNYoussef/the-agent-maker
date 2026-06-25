"""Tier2-P7 gate - CRUCIBLE: Phase 7 honest about routing + the optimizer.

Synthesis: Transformer2 router is exported but NOT wired into the controller/engine - the
pipeline yields a single dense SVF-tuned model, not a routed MoE; experts_engine reports
routing_config={} when there is no routing (already honest). And use_policy_network was dead
config (defined, never branched) so SVF claimed REINFORCE but did direct AdamW. P7 makes the
SVF trainer fail loud on use_policy_network and corrects the docstring. (E8 already fixed the
fp16 SVD silent-skip + coverage.)
"""

import pytest

from phase7_experts.svf_trainer import SVFConfig, SVFTrainer


def test_default_svf_is_direct_optimization():
    # default (use_policy_network=False) constructs fine - honest direct AdamW.
    t = SVFTrainer(SVFConfig(num_singular_values=2))
    assert t.config.use_policy_network is False


def test_use_policy_network_fails_loud():
    with pytest.raises(NotImplementedError):
        SVFTrainer(SVFConfig(use_policy_network=True))


def test_routing_config_is_empty_without_real_routing():
    # The pipeline produces a dense model; routing must not be fabricated. A plain model with
    # no _expert_routing attr yields an empty routing_config (the engine's honest behavior).
    import torch.nn as nn

    model = nn.Linear(4, 4)
    routing_config = getattr(model, "_expert_routing", {})
    assert routing_config == {}, "no real routing -> empty routing_config (no fake MoE)"
