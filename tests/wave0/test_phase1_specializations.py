"""Phase 1 cognate specializations honesty: reasoning/memory/speed are the SAME
architecture differing only by halt threshold and seed. ltm_capacities and
surprise_thresholds are config that is never applied (no memory slots, no
surprise signal in the EMA memory). Lock that so the config can't masquerade as
three distinct designs."""

from phase1_cognate.model.model_config import Phase1Config


def test_specializations_differ_only_by_halt_threshold_and_seed():
    r = Phase1Config(specialization="reasoning")
    m = Phase1Config(specialization="memory")
    s = Phase1Config(specialization="speed")

    # The two wired knobs genuinely differ.
    halts = {r.act_config.halt_threshold, m.act_config.halt_threshold, s.act_config.halt_threshold}
    assert len(halts) == 3, "halt thresholds should differ per specialization"
    assert len({r.get_seed(), m.get_seed(), s.get_seed()}) == 3, "seeds should differ"

    # The architecture itself is identical across specializations.
    for cfg in (m, s):
        assert cfg.titans_config.d_model == r.titans_config.d_model
        assert cfg.titans_config.d_mem == r.titans_config.d_mem
        assert cfg.titans_config.n_layers == r.titans_config.n_layers
        assert cfg.trm_config.T_max == r.trm_config.T_max


def test_capacity_and_surprise_config_is_not_silently_applied():
    # These dicts exist but are NOT wired into the model. If someone later wires
    # them, they must also implement the slotted/surprise memory they presuppose
    # (and update this test), rather than letting dead config imply real behavior.
    r = Phase1Config(specialization="reasoning")
    s = Phase1Config(specialization="speed")
    # d_mem is the single factorized dim and does NOT track ltm_capacities.
    assert r.titans_config.d_mem == s.titans_config.d_mem
    assert r.ltm_capacities["reasoning"] != r.ltm_capacities["speed"]  # config differs...
    assert r.titans_config.d_mem == 160  # ...but the model dim is fixed regardless
