"""Wave-2 autoresearch loop driven by REAL GPU training - on-device proof.

Demonstrates that the Wave-2 self-improvement loop runs end-to-end with a real
model trained on a local consumer GPU (no stubs): a small char-level transformer
is trained on the arithmetic distribution, and the REAL run_loop + SealedScorer +
BenchmarkSuite drive a greedy ratchet whose accept/reject comes from genuinely
measured VAL/LOCKED scores.

Verified 2026-06-24 on an NVIDIA RTX 2060 SUPER (8 GB, CUDA 12.9, torch 2.4.1):

    candidate  VAL     LOCKED   accepted  reason
    r0         0.775   0.731    True      end-to-end 0.7308 > 0.0769 (baseline)
    r1         0.813   0.692    False     0.6923 <= 0.7308; per-phase gains do not vote
    r2         0.963   1.000    True      end-to-end 1.0000 > 0.7308
    r3         1.000   1.000    False     tie
    r4         1.000   1.000    False     tie
    final incumbent: r2  VAL=0.963  LOCKED=1.000

The load-bearing moment is r1: VAL improved (0.775 -> 0.813) but the END-TO-END
LOCKED score regressed (0.731 -> 0.692), so the ratchet REJECTED it. That is the
autoresearch single-metric failure mode being caught on real training data, not a
unit test. The sealed scorer ran the trained model OUT OF PROCESS for every LOCKED
number, so the loop never held the held-out answers.

Exact numbers vary per run (training is stochastic - no fixed seed on init/batch
sampling); the behaviour is invariant. Other runs show the CHEAP TIER instead: once
VAL plateaus at 1.0, later candidates are rejected at the VAL gate with NO LOCKED
spend (empty LOCKED column), conserving the budget - both are the gates working.

Scope: this proves the LOOP MECHANISM on commercial hardware with a small model.
Scaling the model (e.g. a 3B in 4-bit / QLoRA) is a VRAM/technique change to the
`proposer` only - the gates are unchanged. This is a demo, not a packaged module;
it is not under the CI lint/test gates.

Run:
    python experiments/wave2_gpu_demo.py             # run the loop, print the journal
    python experiments/wave2_gpu_demo.py generate CKPT   # sealed-scorer subprocess mode
"""

import json
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src")]

BLOCK = 48
N_EMBD = 128
N_HEAD = 4
N_LAYER = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# model: tiny char-level GPT
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, batch_first=True)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD)
        )

    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab)

    def forward(self, idx):
        t = idx.size(1)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), diagonal=1)
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _example(a, op, b):
    res = {"+": a + b, "-": a - b, "*": a * b}[op]
    return f"What is {a} {op} {b}? Answer: {res}\n"


def _corpus(seed):
    """A large arithmetic corpus over the SAME ranges as benchmark_suite."""
    g = torch.Generator().manual_seed(seed)
    lines = []
    for _ in range(6000):
        r = torch.rand(3, generator=g)
        fam = int(r[0] * 3)
        if fam == 0:
            a, b = 1 + int(r[1] * 8), 1 + int(r[2] * 8)
            lines.append(_example(a, "+", b))
        elif fam == 1:
            a = 2 + int(r[1] * 8)
            b = 1 + int(r[2] * (a - 1))
            lines.append(_example(a, "-", b))
        else:
            a, b = 2 + int(r[1] * 6), 2 + int(r[2] * 6)
            lines.append(_example(a, "*", b))
    return "".join(lines)


def _vocab(text):
    chars = sorted(set(text + "0123456789"))
    stoi = {c: i for i, c in enumerate(chars)}
    return stoi, {i: c for c, i in stoi.items()}


# --------------------------------------------------------------------------- #
# train / generate
# --------------------------------------------------------------------------- #
def train_more(model, opt, data, steps, bs=64):
    model.train()
    loss = torch.tensor(0.0)
    for _ in range(steps):
        ix = torch.randint(0, data.size(0) - BLOCK - 1, (bs,))
        xb = torch.stack([data[i : i + BLOCK] for i in ix]).to(DEVICE)
        yb = torch.stack([data[i + 1 : i + 1 + BLOCK] for i in ix]).to(DEVICE)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


@torch.no_grad()
def generate(model, stoi, itos, prompt, max_new=4):
    model.eval()
    idx = torch.tensor([[stoi[c] for c in prompt if c in stoi]], device=DEVICE)
    out = []
    for _ in range(max_new):
        logits = model(idx[:, -BLOCK:])
        nxt = int(logits[0, -1].argmax())
        ch = itos[nxt]
        if ch == "\n":
            break
        out.append(ch)
        idx = torch.cat([idx, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return "".join(out)


def save_ckpt(path, model, stoi, itos):
    torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos, "vocab": len(stoi)}, path)


def load_ckpt(path):
    ck = torch.load(path, map_location=DEVICE)
    model = CharGPT(ck["vocab"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    return model, ck["stoi"], {int(k): v for k, v in ck["itos"].items()}


def _serve(ckpt_path):
    """Sealed-scorer subprocess mode: JSON prompts in -> JSON completions out."""
    model, stoi, itos = load_ckpt(ckpt_path)
    prompts = json.load(sys.stdin)
    json.dump([generate(model, stoi, itos, p) for p in prompts], sys.stdout)


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def main():
    from cross_phase.evaluation.autoresearch_loop import Candidate, journal_tsv, run_loop
    from cross_phase.evaluation.benchmark_suite import BenchmarkSuite
    from cross_phase.evaluation.sealed_scorer import SealedScorer, seal

    work = tempfile.mkdtemp(prefix="wave2_gpu_demo_")  # artifacts out of the repo tree
    sealed_dir = os.path.join(work, "_sealed")
    seal(sealed_dir, locked_budget=12, blind_budget=1, force=True)
    sealed = SealedScorer(sealed_dir=sealed_dir)
    suite = BenchmarkSuite()

    text = _corpus(0)
    stoi, itos = _vocab(text)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    model = CharGPT(len(stoi)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    nparams = sum(p.numel() for p in model.parameters())
    print(
        f"device={DEVICE} model_params={nparams/1e6:.2f}M vocab={len(stoi)} corpus={len(text)} chars"
    )

    def mk_candidate(cid):
        ckpt = os.path.join(work, f"_ck_{cid}.pt")
        save_ckpt(ckpt, model, stoi, itos)
        gen = lambda p: generate(model, stoi, itos, p)  # noqa: E731 - in-process VAL scorer
        cmd = (sys.executable, os.path.abspath(__file__), "generate", ckpt)
        return Candidate(cid, gen, cmd)

    loss = train_more(model, opt, data, steps=40)  # baseline: barely trained, room to climb
    incumbent = mk_candidate("base")
    print(f"baseline trained (loss={loss:.3f})")

    def proposer(i, _inc):
        last = train_more(model, opt, data, steps=400)
        print(f"  round {i}: trained +400 steps (loss={last:.3f})")
        return mk_candidate(f"r{i}")

    result = run_loop(suite=suite, sealed=sealed, incumbent=incumbent, proposer=proposer, rounds=5)

    print("\n=== JOURNAL (real GPU training drove every number) ===")
    print(journal_tsv(result))
    print(
        f"\nfinal incumbent: {result.incumbent_id}  "
        f"VAL={result.incumbent_val:.3f}  LOCKED={result.incumbent_locked:.3f}"
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "generate":
        _serve(sys.argv[2])
    else:
        main()
