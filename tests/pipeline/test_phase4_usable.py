"""Usability gate - CRUCIBLE: the Phase-4 CompressedModel is actually usable downstream.

The end-to-end verification found the final compressed model could not be used:
'CompressedModel' object has no attribute 'generate'. The wrapper now delegates
generate/get_input_embeddings/config to the wrapped base model.
"""

import torch

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config


def _compressed():
    cfg = Phase1Config(specialization="reasoning")
    cfg.titans_config.d_model, cfg.titans_config.n_heads = 64, 4
    cfg.titans_config.d_ff, cfg.titans_config.d_mem, cfg.titans_config.mag_hidden = 128, 32, 32
    cfg.titans_config.n_layers = 1
    cfg.trm_config.T_max, cfg.trm_config.micro_steps = 1, 1
    cfg.trm_config.step_weights = [1.0, 1.0]
    base = TRMTitansMAGModel(cfg).eval()
    from phase4_bitnet.compressed_model import CompressedModel
    from phase4_bitnet.config import Phase4Config
    from phase4_bitnet.quantizer import BitNetQuantizer

    p4cfg = Phase4Config()
    return CompressedModel(base, BitNetQuantizer(p4cfg), p4cfg, use_bitlinear=True), cfg


def test_compressed_model_is_usable_for_generation():
    model, cfg = _compressed()
    ids = torch.randint(0, cfg.titans_config.vocab_size, (1, 6))
    gen = model.generate(ids, max_new_tokens=4)
    assert gen.shape == (1, 10), f"compressed model must generate; got {tuple(gen.shape)}"


def test_compressed_model_delegates_embeddings():
    model, _ = _compressed()
    assert isinstance(model.get_input_embeddings(), torch.nn.Module)
