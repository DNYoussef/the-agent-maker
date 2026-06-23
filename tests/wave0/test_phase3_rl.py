"""Phase 3 RL: REINFORCE must reinforce the sampled THOUGHT log-probs, not a
cross-entropy proxy. The thought log-probs have to leave the model forward and
be wired into the policy loss."""

import torch
import torch.nn as nn


class _StubOut:
    def __init__(self, logits, hidden):
        self.logits = logits
        self.last_hidden_state = hidden


class _StubBase(nn.Module):
    """Minimal base model exposing the interface QuietSTaRModel needs."""

    def __init__(self, vocab=16, h=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, h)
        self.lm_head = nn.Linear(h, vocab, bias=False)

    def forward(self, input_ids):
        hidden = self.emb(input_ids)
        return _StubOut(self.lm_head(hidden), hidden)

    def get_input_embeddings(self):
        return self.emb


def _model():
    from phase3_quietstar.architecture import QuietSTaRModel

    base = _StubBase()
    model = QuietSTaRModel(
        base_model=base,
        hidden_size=8,
        num_thoughts=2,
        max_thought_length=12,
        injection_threshold=-1.0,  # force a thought injection
    )
    return base, model


def test_forward_returns_graph_connected_thought_log_probs():
    # Bug: the forward discarded thought_output.log_probs entirely, so the
    # REINFORCE trainer had no action log-probs and fell back to -CE_loss.
    torch.manual_seed(0)
    base, model = _model()
    ids = torch.randint(0, 16, (2, 6))
    labels = torch.randint(0, 16, (2, 6))

    out = model(input_ids=ids, labels=labels, use_thoughts=True)
    assert out.get("num_thoughts_used", 0) > 0, "test needs at least one thought injected"

    tlp = out.get("thought_log_probs", None)
    assert tlp is not None, "forward must expose thought_log_probs"
    assert tlp.requires_grad, "thought_log_probs must stay graph-connected (no detach)"
    assert tlp.shape == (2,), "thought_log_probs must be per-sample (batch,) for credit assignment"

    # The policy gradient must reach the thought generator (base model).
    base.lm_head.weight.grad = None
    tlp.mean().backward()
    grad = base.lm_head.weight.grad
    assert grad is not None and grad.abs().sum().item() > 0, (
        "thought log-probs must carry gradient to the policy"
    )


def test_train_episode_uses_thought_log_probs_for_policy_loss():
    # Lock the wiring: the policy loss must consume thought_log_probs, not only
    # the -CE proxy. step2_rl imports MuGrokfast/wandb and uses `..cross_phase`
    # relative imports that only resolve under the `src.` package path, so we
    # read the source file directly instead of importing the module.
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "phase3_quietstar"
        / "step2_rl.py"
    )
    src = path.read_text(encoding="utf-8")
    body = src.split("def train_episode", 1)[1].split("\n    def ", 1)[0]
    assert "thought_log_probs" in body, "policy loss must use the thought log-probs"
    # Per-sample REINFORCE: each sample's log-prob times its own advantage.
    assert "-(thought_log_probs * advantages.detach()).mean()" in body, (
        "policy loss must do per-sample credit assignment"
    )
    # No silent CE proxy fallback when there are no thoughts.
    assert "torch.zeros" in body, "no-thought case must be zero policy loss, not a CE proxy"


def test_run_step2_rl_does_not_freeze_the_policy():
    # Bug: run_step2_rl passed the trainable model as BOTH model and baked_model.
    # The trainer freezes baked_model's params, and that same object is the base
    # model inside QuietSTaRModel, so the policy was frozen and trained nothing.
    # The baked baseline must be a separate frozen snapshot (deepcopy).
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "phase3_quietstar"
        / "step2_rl.py"
    )
    src = path.read_text(encoding="utf-8")
    body = src.split("def run_step2_rl", 1)[1]
    assert "baked_model=model," not in body, "must not alias the trainable model as the frozen baseline"
    assert "copy.deepcopy(model)" in body, "baked baseline must be a frozen snapshot"
