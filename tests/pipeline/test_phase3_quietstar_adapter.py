"""G1 gate - CRUCIBLE: the TRM model exposes the QuietSTaR adapter surface, so full RL
constructs on the REAL model instead of falling back.

Synthesis/Codex: REINFORCETrainer wraps the base model in QuietSTaRModel, which reads
model.config.hidden_size + outputs.last_hidden_state + base_model.get_input_embeddings(); the
TRM model lacked all three, so enable_full_rl silently fell back. G1 exposes them.
"""

import torch

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config


def _tiny():
    cfg = Phase1Config(specialization="reasoning")
    cfg.titans_config.d_model, cfg.titans_config.n_heads = 128, 4
    cfg.titans_config.d_ff, cfg.titans_config.d_mem, cfg.titans_config.mag_hidden = 256, 64, 64
    cfg.titans_config.n_layers = 2
    cfg.trm_config.T_max, cfg.trm_config.micro_steps = 2, 1
    cfg.trm_config.step_weights = [1.0, 1.0, 1.0]
    return TRMTitansMAGModel(cfg).eval(), cfg


def test_trm_exposes_quietstar_adapter_surface():
    model, cfg = _tiny()
    # 1. config.hidden_size == d_model
    assert model.config.hidden_size == cfg.titans_config.d_model
    # 2. get_input_embeddings() returns the token embedding
    emb = model.get_input_embeddings()
    assert isinstance(emb, torch.nn.Module)
    # 3. forward output exposes last_hidden_state [b, s, hidden_size]
    with torch.no_grad():
        out = model(torch.randint(0, cfg.titans_config.vocab_size, (1, 6)))
    assert out.last_hidden_state.shape == (1, 6, cfg.titans_config.d_model)


def test_quietstar_model_wraps_and_forwards_on_real_trm():
    from phase3_quietstar.architecture.quiet_star_model import QuietSTaRModel

    model, cfg = _tiny()
    qs = QuietSTaRModel(base_model=model, hidden_size=model.config.hidden_size, num_thoughts=2)
    with torch.no_grad():
        out = qs(torch.randint(0, cfg.titans_config.vocab_size, (1, 6)))
    logits = out["logits"] if isinstance(out, dict) else getattr(out, "logits", out)
    assert logits.shape[-1] == cfg.titans_config.vocab_size


def test_reinforce_trainer_constructs_on_real_trm():
    # The headline: the trainer used to raise on model.config.hidden_size -> silent fallback.
    from phase3_quietstar.config import QuietSTaRConfig
    from phase3_quietstar.step2_rl import REINFORCETrainer

    model, _ = _tiny()
    baked, _ = _tiny()
    trainer = REINFORCETrainer(
        model=model, baked_model=baked, tokenizer=None, config=QuietSTaRConfig(), device="cpu"
    )
    assert trainer.hidden_size == model.config.hidden_size
