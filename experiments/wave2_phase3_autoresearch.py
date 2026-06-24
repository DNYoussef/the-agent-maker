"""Wave-2 Layer 3: autoresearch loop that EDITS phase3 - MACHINERY + a methodology finding.

WHAT IS REAL HERE: the loop proposes real behavioral EDITS to phase3, trains each variant
on the GPU, scores them on the sealed held-out split out-of-process, and ratchets - all
through the real gates (run_loop + SealedScorer + BenchmarkSuite). No mutation of the
merged src/phase3_quietstar module; edits are selectable variants:
  stock          - generation reads the BASE frontier; thoughts never reach it.
  frontier_base  - control: extra answer-position CE supervision, NO thoughts.
  frontier       - THE EDIT: the same supervision routed through the REAL phase3 thought
                   path (thought_generator -> coherence -> mixing) at the answer position.

WHAT IS NOT SOUND (Codex-audited, do NOT read the numbers as a verdict): on the arithmetic
benchmark this experiment CANNOT decide whether the edit helps, because:
  1. CONTAMINATION (critical). The training corpus samples the SAME finite arithmetic grid
     that BenchmarkSuite partitions into VAL/LOCKED/BLIND, so every "held-out" item is in
     training (LOCKED 26/26 overlap, repeated 16-100x). Stock's LOCKED=1.000 is memorisation,
     not generalisation. A computable/finite task makes the sealed split meaningless once you
     train on its distribution - the exact limit documented in sealed_scorer.
  2. The variants don't start from identical weights, the edits add an objective (not just
     thoughts), thought sampling is single-seed, and the observed frontier-vs-control gap
     (-0.038) is exactly 1 item of 26 - i.e. noise.

THE FINDING: arithmetic-on-a-small-grid is memorisable, so the held-out gate measures
memorisation. A SOUND phase3 verdict needs a NON-MEMORISABLE task (a disjoint train split
with asserted zero overlap, ideally a reasoning task whose key positions are high-entropy
enough to fire the injector), identical init/seeds, a fair objective matrix, and multi-seed
paired evaluation. This file demonstrates the edit-loop machinery and records why a real
task is required; it is a diagnostic, not a result.

Run on GPU:
    python experiments/wave2_phase3_autoresearch.py
    python experiments/wave2_phase3_autoresearch.py generate CKPT   # sealed subprocess
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

from phase3_quietstar.architecture import QuietSTaRModel  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK, H, NL = 48, 128, 3


class CharBase(nn.Module):
    """Small base satisfying the QuietSTaRModel contract."""

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
        import types

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


# --------------------------------------------------------------------------- #
# vocab / data
# --------------------------------------------------------------------------- #
def _line(a, op, b):
    res = {"+": a + b, "-": a - b, "*": a * b}[op]
    return f"What is {a} {op} {b}? Answer: {res}\n"


def _examples(seed, n):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        r = torch.rand(3, generator=g)
        fam = int(r[0] * 3)
        if fam == 0:
            out.append(_line(1 + int(r[1] * 8), "+", 1 + int(r[2] * 8)))
        elif fam == 1:
            a = 2 + int(r[1] * 8)
            out.append(_line(a, "-", 1 + int(r[2] * (a - 1))))
        else:
            out.append(_line(2 + int(r[1] * 6), "*", 2 + int(r[2] * 6)))
    return out


_CHARS = sorted(set("".join(_examples(0, 200)) + "0123456789"))
STOI = {c: i for i, c in enumerate(_CHARS)}
ITOS = {i: c for c, i in STOI.items()}


def _enc(s):
    return [STOI[c] for c in s if c in STOI]


# --------------------------------------------------------------------------- #
# the EDIT: thought-enhanced frontier logit (uses the REAL phase3 components)
# --------------------------------------------------------------------------- #
def frontier_logit(qs, ids, pos, thoughts):
    """Logit predicting token pos+1 from position `pos`. thoughts=False is the stock
    base frontier; thoughts=True routes the REAL phase3 thought path to the frontier."""
    b = qs.base_model(ids[:, : pos + 1])
    h = b.last_hidden_state[:, pos, :]
    if not thoughts:
        return b.logits[:, pos, :]
    to = qs.thought_generator(ids[:, : pos + 1], pos, h)
    coh = qs.coherence_scorer(h, to.thoughts, b.logits[:, pos, :])
    mixed = qs.mixing_head(h, to.thoughts.mean(dim=2), coh.composite)
    return qs.base_model.lm_head(mixed)


def _ans_pos(s):
    """Index whose next token is the first answer char (the hard prediction)."""
    return s.index("Answer: ") + len("Answer: ") - 1


def train(qs, variant, examples, steps, bs=24, lr=3e-3):
    opt = torch.optim.AdamW(qs.parameters(), lr=lr)
    qs.train()
    g = torch.Generator().manual_seed(0)
    last = 0.0
    for _ in range(steps):
        idx = torch.randint(0, len(examples), (bs,), generator=g)
        batch = [examples[i] for i in idx]
        # base LM loss (both variants) over padded sequences
        maxlen = min(BLOCK, max(len(s) for s in batch))
        x = torch.zeros(bs, maxlen, dtype=torch.long)
        for j, s in enumerate(batch):
            e = _enc(s)[:maxlen]
            x[j, : len(e)] = torch.tensor(e)
        x = x.to(DEV)
        logits = qs.base_model(x).logits
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        loss = lm
        # answer-position CE. frontier_base trains it WITHOUT thoughts (control for
        # "extra supervision"); frontier trains it THROUGH the thought path (the edit).
        if variant in ("frontier", "frontier_base"):
            use_thoughts = variant == "frontier"
            ace = []
            for s in batch:
                p = _ans_pos(s)
                if p + 1 >= len(s):
                    continue
                ids = torch.tensor([_enc(s)], device=DEV)
                logit = frontier_logit(qs, ids, p, thoughts=use_thoughts)
                tgt = torch.tensor([STOI[s[p + 1]]], device=DEV)
                ace.append(F.cross_entropy(logit, tgt))
            if ace:
                loss = lm + torch.stack(ace).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss)
    return last


@torch.no_grad()
def generate(qs, variant, prompt, max_new=3):
    qs.eval()
    ids = torch.tensor([_enc(prompt)], device=DEV)
    out = []
    for k in range(max_new):
        pos = ids.size(1) - 1
        use = variant == "frontier" and k == 0  # thoughts on the hard first answer token
        logit = frontier_logit(qs, ids, pos, thoughts=use)
        nx = int(logit[0].argmax())
        if ITOS[nx] == "\n":
            break
        out.append(ITOS[nx])
        ids = torch.cat([ids, torch.tensor([[nx]], device=DEV)], 1)
    return "".join(out)


# --------------------------------------------------------------------------- #
# checkpoint + sealed-scorer subprocess
# --------------------------------------------------------------------------- #
def _build_qs():
    base = CharBase(len(_CHARS)).to(DEV)
    qs = QuietSTaRModel(base, hidden_size=H, num_thoughts=2, max_thought_length=3).to(DEV)
    qs.thought_generator.min_length = 2
    return qs


def save_ckpt(path, qs, variant):
    torch.save({"model": qs.state_dict(), "variant": variant}, path)


def _serve(ckpt_path):
    torch.manual_seed(0)  # stable thought sampling for a reproducible LOCKED score
    ck = torch.load(ckpt_path, map_location=DEV)
    qs = _build_qs()
    qs.load_state_dict(ck["model"])
    prompts = json.load(sys.stdin)
    json.dump([generate(qs, ck["variant"], p) for p in prompts], sys.stdout)


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def main():
    from cross_phase.evaluation.autoresearch_loop import Candidate, journal_tsv, run_loop
    from cross_phase.evaluation.benchmark_suite import BenchmarkSuite, Split
    from cross_phase.evaluation.sealed_scorer import SealedScorer, seal

    work = tempfile.mkdtemp(prefix="wave2_phase3_")
    sealed_dir = os.path.join(work, "_sealed")
    seal(sealed_dir, locked_budget=12, blind_budget=1, force=True)
    sealed = SealedScorer(sealed_dir=sealed_dir)
    suite = BenchmarkSuite()
    examples = _examples(0, 5000)
    print(f"device={DEV} variants=[stock, frontier_base(control), frontier(edit)]")

    def make(variant, steps=600):
        qs = _build_qs()
        loss = train(qs, variant, examples, steps)
        ck = os.path.join(work, f"_ck_{variant}.pt")
        save_ckpt(ck, qs, variant)
        gen = lambda p: generate(qs, variant, p)  # noqa: E731 - in-process VAL scorer
        cmd = (sys.executable, os.path.abspath(__file__), "generate", ck)
        print(f"  trained {variant} ({steps} steps, loss={loss:.3f})")
        return Candidate(variant, gen, cmd)

    cands = {v: make(v) for v in ("stock", "frontier_base", "frontier")}

    # Direct 3-way held-out comparison (isolates the thought contribution).
    print("\n=== held-out comparison (VAL public, LOCKED sealed/out-of-process) ===")
    print("variant         VAL    LOCKED   note")
    scores = {}
    notes = {
        "stock": "stock phase3 (thoughts never reach the frontier)",
        "frontier_base": "control: extra answer supervision, NO thoughts",
        "frontier": "EDIT: same supervision routed THROUGH the thought path",
    }
    for v, c in cands.items():
        val = suite.score(c.val_generate, Split.VAL)
        locked = sealed.score(c.locked_cmd, "locked")
        scores[v] = (val, locked)
        print(f"{v:15} {val:.3f}  {locked:.3f}   {notes[v]}")
    thought_gain = scores["frontier"][1] - scores["frontier_base"][1]
    print(f"\nthought contribution on LOCKED (frontier - frontier_base) = {thought_gain:+.3f}")
    print(
        "WARNING: NOT a sound verdict. The training corpus reproduces the benchmark grid, so\n"
        "LOCKED is contaminated (memorised, not held-out); init/seed/objective are not matched;\n"
        "and a 1/26-item gap is within noise. See the module docstring for the fixes a sound\n"
        "phase3 verdict requires (non-memorisable task + disjoint split + identical init + multi-seed)."
    )

    # The autoresearch narrative: the loop keeps an edit only if held-out improves.
    edits = ["frontier_base", "frontier"]
    result = run_loop(
        suite=suite,
        sealed=sealed,
        incumbent=cands["stock"],
        proposer=lambda i, _inc: cands[edits[i]],
        rounds=2,
    )
    print("\n=== autoresearch journal (MACHINERY demo - the loop edits phase3 + ratchets) ===")
    print(journal_tsv(result))
    print(f"final incumbent={result.incumbent_id} LOCKED={result.incumbent_locked:.3f}")
    print("(machinery works; the LOCKED numbers are contaminated - not a phase3 verdict)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "generate":
        _serve(sys.argv[2])
    else:
        main()
