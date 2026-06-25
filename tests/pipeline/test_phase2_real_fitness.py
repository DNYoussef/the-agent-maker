"""Tier2-P2 gate - CRUCIBLE: Phase 2 real-fitness primitives work on the real model.

The synthesis 1->2 break meant the real-fitness path (perplexity/accuracy) did
`logits = model(input_ids)` and treated it as a tensor - but the Phase 1 model returns a
ModelOutput (dict). F0 added .logits; P2 makes the fitness primitives extract it, so
use_real_fitness can actually run instead of silently scoring 0.
"""

import math

import torch

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from phase2_evomerge.fitness.accuracy import calculate_accuracy
from phase2_evomerge.fitness.perplexity import calculate_perplexity


def _small_model():
    cfg = Phase1Config(specialization="reasoning")
    cfg.titans_config.d_model, cfg.titans_config.n_heads = 128, 4
    cfg.titans_config.d_ff, cfg.titans_config.d_mem, cfg.titans_config.mag_hidden = 256, 64, 64
    cfg.titans_config.n_layers = 2
    cfg.trm_config.T_max, cfg.trm_config.micro_steps = 2, 1
    cfg.trm_config.step_weights = [1.0, 1.0, 1.0]
    return TRMTitansMAGModel(cfg).eval(), cfg


def _data(vocab, n=2):
    return [(torch.randint(0, vocab, (2, 8)), torch.randint(0, vocab, (2, 8))) for _ in range(n)]


def test_perplexity_runs_on_real_model():
    m, cfg = _small_model()
    ppl = calculate_perplexity(
        m, _data(cfg.titans_config.vocab_size), device="cpu", mixed_precision=False
    )
    assert math.isfinite(ppl) and ppl > 0, f"perplexity must be finite/positive, got {ppl}"


def test_accuracy_runs_on_real_model():
    m, cfg = _small_model()
    acc = calculate_accuracy(m, _data(cfg.titans_config.vocab_size), device="cpu")
    assert 0.0 <= acc <= 1.0, f"accuracy must be a fraction, got {acc}"
