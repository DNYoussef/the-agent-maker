"""
Wave 4 deep-ML-correctness test: Phase 3 Quiet-STaR REINFORCE must put a
genuine policy-gradient term on the SAMPLED thought tokens.

Defect (pre-fix):
    ThoughtGenerator computes per-token log-probs of the SAMPLED thought
    tokens (attached to the autograd graph) and returns them in
    ThoughtOutput.log_probs. But QuietSTaRModel.forward used only
    `.thoughts` / `.thought_ids` and *discarded* `.log_probs` -- they never
    left forward(). REINFORCETrainer.train_episode then did:

        log_prob   = -outputs_with["loss"]          # the CE loss
        policy_loss = -(log_prob * advantages.detach().mean())

    i.e. it merely RESCALED the cross-entropy gradient by a scalar. The
    thought-sampling policy received NO score-function gradient, so REINFORCE
    never optimized which thoughts get sampled. "REINFORCE on reasoning
    tokens" was degenerate.

Fix:
    QuietSTaRModel.forward now aggregates the sampled-thought log-probs and
    returns them as `thought_log_probs` (attached to the graph).
    train_episode builds a real REINFORCE term

        policy_loss = -(advantage.detach() * thought_log_probs)

    with a baseline (reward - learned value) for variance reduction, added to
    the (still present) supervised CE loss.

These tests prove the score-function gradient genuinely reaches the
thought-sampling policy (base_model), and that it scales with the reward /
advantage -- the defining property the CE-rescaling lacked.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.phase3_quietstar.architecture import QuietSTaRModel
from src.phase3_quietstar.config import QuietSTaRConfig


class TinyLM(nn.Module):
    """Minimal HF-style causal LM: real, differentiable, deterministic.

    Provides exactly the surface QuietSTaRModel / REINFORCETrainer touch:
      - __call__(input_ids, attention_mask=None) -> .logits, .last_hidden_state
      - .lm_head (hidden -> vocab)
      - .get_input_embeddings()
      - .config.hidden_size
    """

    def __init__(self, vocab: int = 8, hidden: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = nn.Embedding(vocab, hidden)
        self.trunk = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(self, input_ids, attention_mask=None, **kw):
        h = torch.tanh(self.trunk(self.embed(input_ids)))
        logits = self.lm_head(h)
        return SimpleNamespace(logits=logits, last_hidden_state=h)


def _model(num_thoughts: int = 2) -> QuietSTaRModel:
    torch.manual_seed(0)
    base = TinyLM()
    # injection_threshold=0.0 forces thought injection (default 0.6 is never
    # reached by the difficulty heuristic, but that is a separate concern);
    # short thoughts keep the test fast.
    return QuietSTaRModel(
        base_model=base,
        hidden_size=base.config.hidden_size,
        num_thoughts=num_thoughts,
        max_thought_length=11,
        injection_threshold=0.0,
    )


def _batch(seq_len: int = 8, vocab: int = 8):
    torch.manual_seed(1)
    input_ids = torch.randint(0, vocab, (2, seq_len))
    labels = torch.randint(0, vocab, (2, seq_len))
    return input_ids, labels


def test_thought_log_probs_exposed_and_flow_to_policy():
    """Core defect: sampled-thought log-probs must be exposed by forward() and
    a REINFORCE term built from them must deliver gradient to the thought
    sampling policy (base_model)."""
    model = _model()
    input_ids, labels = _batch()

    torch.manual_seed(7)
    out = model(input_ids=input_ids, labels=labels, use_thoughts=True)

    assert out["num_thoughts_used"] > 0, "test setup: no thoughts were injected"

    # (1) forward must expose the sampled-thought log-probs.
    assert "thought_log_probs" in out, (
        "QuietSTaRModel.forward discards thought log-probs; "
        "REINFORCE cannot form a policy-gradient term"
    )
    tlp = out["thought_log_probs"]
    assert isinstance(tlp, torch.Tensor)

    # (2) they must be attached to the autograd graph.
    assert tlp.requires_grad, "thought_log_probs detached from graph -> no policy gradient"

    # (3) the pure REINFORCE term (advantage * sum log-prob) must put gradient
    #     onto the thought-sampling policy = base_model parameters.
    # thought_log_probs is per-sample (batch,) -- the correct REINFORCE shape so
    # each sample is reinforced by its OWN advantage. Sum to a scalar to form the
    # backward root (with a scalar advantage this is the pure score-function term).
    advantage = 2.0
    model.zero_grad(set_to_none=True)
    (-(advantage * tlp)).sum().backward()

    g = model.base_model.trunk.weight.grad
    assert g is not None, "policy-gradient term delivered NO gradient to base_model"
    assert torch.isfinite(g).all()
    assert g.abs().sum().item() > 0.0, "policy gradient is all-zero"


def test_reward_enters_as_multiplicative_coefficient():
    """REINFORCE property: the reward/advantage is a multiplicative score-
    function coefficient, so the policy gradient scales linearly with it.

    The old code only rescaled the CE gradient -- it had this property by
    accident for the CE term but never moved the *sampling* policy. Here we
    prove the gradient on the sampling policy itself scales with advantage."""
    model = _model()
    input_ids, labels = _batch()

    torch.manual_seed(7)
    out = model(input_ids=input_ids, labels=labels, use_thoughts=True)
    tlp = out["thought_log_probs"]
    assert tlp.requires_grad

    param = model.base_model.trunk.weight

    # Sum the per-sample (batch,) log-probs to a scalar backward root; the scalar
    # advantage stays a pure multiplicative coefficient.
    model.zero_grad(set_to_none=True)
    (-(1.0 * tlp)).sum().backward(retain_graph=True)
    g1 = param.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    (-(3.0 * tlp)).sum().backward()
    g3 = param.grad.detach().clone()

    assert g1.abs().sum().item() > 0.0
    # grad(advantage=3) == 3 * grad(advantage=1): pure score-function scaling.
    assert torch.allclose(g3, 3.0 * g1, rtol=1e-4, atol=1e-6), (
        "policy gradient does not scale with reward/advantage -> not REINFORCE"
    )


def test_train_episode_wires_real_reinforce(monkeypatch):
    """Trainer-level wiring: train_episode must (a) report the thought
    log-prob term, (b) deliver gradient to the thought-sampling policy, and
    (c) keep the supervised CE loss (so the mixing head still learns)."""
    import src.phase3_quietstar.step2_rl as step2

    # No-op W&B logger (offline, no files/network).
    class _NoOpLogger:
        def __init__(self, *a, **k):
            pass

        def log(self, *a, **k):
            pass

    monkeypatch.setattr(step2, "WandBLogger", _NoOpLogger)

    cfg = QuietSTaRConfig()
    cfg.rl.num_thoughts = 2
    cfg.rl.max_thought_length = 11
    cfg.rl.injection_threshold = 0.0
    cfg.rl.use_gae = False
    cfg.rl.enable_teacher_forcing = False
    cfg.rl.lr_schedule = "constant"
    # Square shapes only: avoids an unrelated tall-matrix bug in the
    # MuonGrokfast Newton-Schulz step (out-of-scope optimizer file).
    cfg.rl.baseline_hidden_size = 8

    torch.manual_seed(0)
    base = TinyLM()

    trainer = step2.REINFORCETrainer(
        model=base,
        baked_model=TinyLM(),
        tokenizer=None,
        config=cfg,
        device="cpu",
    )

    # Isolate the test from MuonGrokfast: its Newton-Schulz step is broken for
    # non-square 2D params (a pre-existing, out-of-scope optimizer bug). We are
    # testing train_episode's loss/gradient WIRING, not the optimizer, so swap
    # in a no-op SGD (lr=0): params stay fixed but .grad is still populated.
    import itertools

    trainer.optimizer = torch.optim.SGD(
        itertools.chain(trainer.model.parameters(), trainer.baseline_network.parameters()),
        lr=0.0,
    )

    input_ids, labels = _batch()

    torch.manual_seed(7)
    metrics = trainer.train_episode(input_ids, labels)

    # (a) the REINFORCE log-prob term is reported.
    assert "thought_log_prob" in metrics, "train_episode does not report the policy log-prob term"
    assert "ce_loss" in metrics, "supervised CE loss term not reported"
    for k in ("policy_loss", "total_loss", "thought_log_prob", "ce_loss"):
        assert metrics[k] == metrics[k], f"{k} is NaN"  # NaN != NaN

    # (b) the thought-sampling policy (base_model) received gradient.
    g = trainer.model.base_model.trunk.weight.grad
    assert g is not None and g.abs().sum().item() > 0.0, "no gradient reached base_model"

    # (c) the supervised CE loss is non-trivial (mixing head still learns).
    assert abs(metrics["ce_loss"]) > 0.0
