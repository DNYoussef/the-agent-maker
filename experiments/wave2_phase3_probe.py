"""Wave-2 Layer 3 probe: what does the REAL phase3 Quiet-STaR contribute in the loop?

Honest on-device finding (RTX 2060 SUPER, 8 GB) from running the merged phase3
`QuietSTaRModel` - NOT a success demo. The autoresearch loop needs the proposer's
held-out score to reflect the phase being improved; this probe checks whether
phase3's thoughts actually move that score on the arithmetic benchmark. They do not,
for three independent, code-level reasons:

  A. INJECTOR IS SILENT on easy text. ThoughtInjector fires on entropy
     (difficulty = 0.4*entropy + 0.15 > threshold=0.6). Once the model learns the
     repetitive template, entropy collapses and `num_thoughts_used` is 0 every step
     - so Quiet-STaR is never exercised during training.

  B. THE GENERATION FRONTIER IS NEVER ENHANCED. `QuietSTaRModel.forward` loops
     `pos in range(seq_len-1)` and overwrites `enhanced_hidden[pos]`; the last index
     (the one autoregressive generation reads via logits[-1]) is never injected.
     With forced injection (threshold=-1) the final-position logits are byte-identical
     with and without thoughts - so thoughts cannot change a generated token.

  C. MIXING IS UNTRAINED. Because of (A), the coherence/mixing heads get no gradient,
     so even an inference-time frontier predictor that calls them barely changes the
     prediction.

IMPLICATION (the design fork for a real phase3 proposer):
  - phase3's value is a TEACHER-FORCED, with-vs-without-thoughts accuracy gain at
    injected positions (its own compute_reward), NOT greedy generation accuracy. A
    faithful phase3-in-the-loop needs that metric, plus a task whose key positions are
    high-entropy enough to fire the injector (a reasoning task, not a memorised
    template), and/or injection forced at the answer positions during training so the
    mixing heads actually learn. Wiring phase3 to the generation-based gate as-is would
    score the BASE model and credit it to "phase3" - theater.

This file is a DIAGNOSTIC, not a packaged module; run it to reproduce the finding:
    python experiments/wave2_phase3_probe.py
"""

import os
import sys
import time
import types

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src")]

from phase3_quietstar.architecture import QuietSTaRModel  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK, H, NL = 40, 128, 3


class CharBase(nn.Module):
    """Small base satisfying the QuietSTaRModel contract (logits + last_hidden_state,
    lm_head, get_input_embeddings) - this same shape learns arithmetic in the GPU demo."""

    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, H)
        self.pos = nn.Embedding(BLOCK, H)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(H),
                        "attn": nn.MultiheadAttention(H, 4, batch_first=True),
                        "ln2": nn.LayerNorm(H),
                        "mlp": nn.Sequential(nn.Linear(H, 4 * H), nn.GELU(), nn.Linear(4 * H, H)),
                    }
                )
                for _ in range(NL)
            ]
        )
        self.lnf = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, vocab)

    def get_input_embeddings(self):
        return self.tok

    def forward(self, input_ids):
        t = input_ids.size(1)
        m = torch.triu(torch.full((t, t), float("-inf"), device=input_ids.device), 1)
        x = self.tok(input_ids) + self.pos(torch.arange(t, device=input_ids.device))
        for ly in self.layers:
            q = ly["ln1"](x)
            a, _ = ly["attn"](q, q, q, attn_mask=m, need_weights=False)
            x = x + a
            x = x + ly["mlp"](ly["ln2"](x))
        h = self.lnf(x)
        return types.SimpleNamespace(logits=self.lm_head(h), last_hidden_state=h)


def _corpus(n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        r = torch.rand(2, generator=g)
        a, b = 1 + int(r[0] * 8), 1 + int(r[1] * 8)
        out.append(f"What is {a} + {b}? Answer: {a + b}\n")
    return "".join(out)


def main():
    text = _corpus(6000, 0)
    chars = sorted(set(text + "0123456789"))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    base = CharBase(len(chars)).to(DEV)
    qs = QuietSTaRModel(base, hidden_size=H, num_thoughts=2, max_thought_length=3).to(DEV)
    qs.thought_generator.min_length = 2
    opt = torch.optim.AdamW(qs.parameters(), lr=3e-3)
    print(f"device={DEV} params={sum(p.numel() for p in qs.parameters())/1e6:.2f}M")

    # (A) train; track how often the injector actually fires.
    t0, inj_total = time.time(), 0
    qs.train()
    for step in range(250):
        ix = torch.randint(0, data.size(0) - BLOCK - 1, (16,))
        xb = torch.stack([data[i : i + BLOCK] for i in ix]).to(DEV)
        yb = torch.stack([data[i + 1 : i + 1 + BLOCK] for i in ix]).to(DEV)
        out = qs(input_ids=xb, labels=yb, use_thoughts=True)
        inj_total += out["num_thoughts_used"]
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    print(f"(A) trained 250 steps in {time.time()-t0:.1f}s, final loss={out['loss'].item():.3f}")
    print(f"(A) injector fired {inj_total} times across 250 steps (entropy < threshold -> silent)")

    # (B) with FORCED injection, is the generation frontier (logits[-1]) changed?
    qs.eval()
    qs.thought_injector.threshold = -1.0  # force injection everywhere it can
    ids = torch.tensor([[stoi[c] for c in "What is 2 + 2? Answer: "]], device=DEV)
    with torch.no_grad():
        base_last = qs.base_model(ids).logits[:, -1, :]
        enh_last = qs(input_ids=ids, use_thoughts=True)["logits"][:, -1, :]
    same = torch.allclose(base_last, enh_last)
    print(f"(B) forced injection: frontier logits[-1] identical with/without thoughts? {same}")
    print("    -> thoughts provably cannot change an autoregressively generated token")


if __name__ == "__main__":
    main()
