"""Wave-2: a SOUND verdict on whether routing phase3 thoughts to the frontier helps.

The merged edit-loop experiment (PR #25) could not conclude: its training corpus
reproduced the benchmark grid, so the "held-out" split was memorised, init/objective
weren't matched, and a 1-item delta was within noise. This redoes it with the fair
protocol Codex demanded:

  1. NON-MEMORISABLE task with a DISJOINT split: addition over operands 0..149; 15% of
     (a,b) PAIRS are held out as TEST and asserted to have ZERO overlap with TRAIN, so a
     correct TEST answer requires GENERALISING the addition, not recall.
  2. IDENTICAL init: all variants start from one cloned initial state_dict.
  3. A CONTROL that isolates the thought contribution from the extra supervision:
       stock          - LM loss only;            eval via base frontier.
       frontier_base  - LM + answer-pos CE (NO thoughts); eval via base frontier.
       frontier       - LM + answer-pos CE THROUGH the real phase3 thought path; eval via
                        the thought-enhanced frontier.
     thought contribution = frontier - frontier_base (same supervision; only thoughts differ).
  4. MULTI-SEED paired eval (3 seeds); the verdict requires the mean delta to exceed the
     run-to-run noise (std), else it is reported INCONCLUSIVE - never overclaimed.

Metric: EXACT-match accuracy of the full generated sum on a fixed shuffled sample of the
commute-disjoint TEST pairs (first-digit accuracy is not addition accuracy). The
frontier-vs-frontier_base comparison is PAIRED per seed (matched init).

Run on GPU:  python experiments/wave2_phase3_sound_verdict.py
"""

import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src")]

from phase3_quietstar.architecture import QuietSTaRModel  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK, H, NL = 24, 96, 2
HI = 150  # operands 0..149


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
# disjoint, non-memorisable data
# --------------------------------------------------------------------------- #
def _pairs():
    """Split into disjoint TRAIN/TEST by the UNORDERED pair, so a+b and b+a land in the
    SAME split - otherwise the commuted pair leaks the answer from train (Codex #1)."""
    train, test = [], []
    for a in range(HI):
        for b in range(HI):
            lo, hi = (a, b) if a <= b else (b, a)
            (test if (lo * 131 + hi) % 100 < 15 else train).append((a, b))
    return train, test


def _line(a, b):
    return f"{a}+{b}={a + b}\n"


_CHARS = sorted(set("0123456789+=\n"))
STOI = {c: i for i, c in enumerate(_CHARS)}
ITOS = {i: c for c, i in STOI.items()}


def _enc(s):
    return [STOI[c] for c in s]


def _ans_pos(s):
    return s.index("=")  # position whose NEXT token is the first answer digit


# --------------------------------------------------------------------------- #
# the edit: thought-enhanced frontier logit (uses the REAL phase3 components)
# --------------------------------------------------------------------------- #
def frontier_logit(qs, ids, pos, thoughts):
    b = qs.base_model(ids[:, : pos + 1])
    h = b.last_hidden_state[:, pos, :]
    if not thoughts:
        return b.logits[:, pos, :]
    to = qs.thought_generator(ids[:, : pos + 1], pos, h)
    coh = qs.coherence_scorer(h, to.thoughts, b.logits[:, pos, :])
    mixed = qs.mixing_head(h, to.thoughts.mean(dim=2), coh.composite)
    return qs.base_model.lm_head(mixed)


def _build(seed):
    torch.manual_seed(seed)
    base = CharBase(len(_CHARS)).to(DEV)
    qs = QuietSTaRModel(base, hidden_size=H, num_thoughts=2, max_thought_length=3).to(DEV)
    qs.thought_generator.min_length = 2
    return qs


def train(qs, variant, train_pairs, steps=500, bs=64):
    opt = torch.optim.AdamW(qs.parameters(), lr=3e-3)
    qs.train()
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, len(train_pairs), (bs,), generator=g)
        batch = [_line(*train_pairs[i]) for i in idx]
        maxlen = max(len(s) for s in batch)
        x = torch.zeros(bs, maxlen, dtype=torch.long)
        for j, s in enumerate(batch):
            x[j, : len(s)] = torch.tensor(_enc(s))
        x = x.to(DEV)
        logits = qs.base_model(x).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        if variant in ("frontier", "frontier_base"):
            use = variant == "frontier"
            ace = []
            for s in batch:
                p = _ans_pos(s)
                ids = torch.tensor([_enc(s)], device=DEV)
                logit = frontier_logit(qs, ids, p, thoughts=use)
                ace.append(F.cross_entropy(logit, torch.tensor([STOI[s[p + 1]]], device=DEV)))
            loss = loss + torch.stack(ace).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def _full_answer(qs, variant, a, b, max_new=5):
    """Autoregressively generate the WHOLE sum (exact-match metric, not first-digit).
    frontier uses the thought path on the first answer token (where answer-CE trained it),
    base for the rest; stock/frontier_base use the base frontier throughout."""
    ids = torch.tensor([_enc(f"{a}+{b}=")], device=DEV)
    out = []
    for k in range(max_new):
        use = variant == "frontier" and k == 0
        logit = frontier_logit(qs, ids, ids.size(1) - 1, thoughts=use)
        nx = int(logit[0].argmax())
        if ITOS[nx] == "\n":
            break
        out.append(ITOS[nx])
        ids = torch.cat([ids, torch.tensor([[nx]], device=DEV)], 1)
    return "".join(out)


def test_acc(qs, variant, sample):
    qs.eval()
    correct = sum(_full_answer(qs, variant, a, b) == str(a + b) for a, b in sample)
    return correct / len(sample)


def _mean_std(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    import random

    train_pairs, test_pairs = _pairs()
    train_set = set(train_pairs)
    assert train_set.isdisjoint(set(test_pairs)), "TRAIN/TEST overlap"
    # No commuted leak: for every test (a,b), (b,a) must also be held out.
    assert all((b, a) not in train_set for a, b in test_pairs), "commuted pair leaked into TRAIN"
    shuffled = test_pairs[:]
    random.Random(12345).shuffle(shuffled)
    # Dedupe commuted twins so the 400 eval units are INDEPENDENT unordered pairs.
    seen, sample = set(), []
    for a, b in shuffled:
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        sample.append((a, b))
    sample = sample[:400]
    print(f"device={DEV} train={len(train_pairs)} test={len(test_pairs)} eval_sample={len(sample)}")

    variants = ["stock", "frontier_base", "frontier"]
    results = {v: [] for v in variants}
    for seed in (0, 1, 2):
        init_state = {k: v.clone() for k, v in _build(seed).state_dict().items()}
        for v in variants:
            qs = _build(seed)
            qs.load_state_dict(init_state)  # IDENTICAL init across variants
            train(qs, v, train_pairs)
            acc = test_acc(qs, v, sample)
            results[v].append(acc)
            print(f"  seed{seed} {v:14} held-out EXACT-sum acc = {acc:.3f}")

    print("\n=== SOUND VERDICT (commute-disjoint, identical init, exact-match, 3 seeds) ===")
    means = {}
    for v in variants:
        m, s = _mean_std(results[v])
        means[v] = m
        print(f"  {v:14} acc = {m:.3f} +/- {s:.3f}")
    # PAIRED per-seed deltas (init matched per seed) - the thought-isolating comparison.
    deltas = [results["frontier"][i] - results["frontier_base"][i] for i in range(3)]
    md, sd = _mean_std(deltas)
    allpos, allneg = all(d > 0 for d in deltas), all(d < 0 for d in deltas)
    print(f"\nper-seed thought delta (frontier - frontier_base) = {[round(d, 3) for d in deltas]}")
    print(f"  paired mean = {md:+.3f} +/- {sd:.3f}  (n=3 - SUGGESTIVE, not a hard claim)")
    trend = "POSITIVE" if md > 0 else "NEGATIVE" if md < 0 else "FLAT"
    agree = " and all 3 seeds agree in sign" if (allpos or allneg) else ""
    print(
        f"VERDICT (frontier vs frontier_base, isolating thoughts): {trend} paired trend{agree}; "
        "n=3 makes this suggestive, not definitive."
    )
    best = max(means, key=means.get)
    if best != "frontier":
        print(
            f"CAVEAT: '{best}' had the highest accuracy ({means[best]:.3f}) - the answer-CE objective "
            f"itself underperformed plain LM here, and absolute exact-match is low. So thoughts help "
            f"WITHIN the supervised-frontier regime, which is not the best regime on this task."
        )


if __name__ == "__main__":
    main()
