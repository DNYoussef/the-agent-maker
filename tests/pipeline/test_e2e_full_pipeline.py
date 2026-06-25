"""Full-pipeline usability E2E (@slow) - CRUCIBLE.

Proves a real Cognate model threads Phase 1 -> 4 (BitNet) -> 5 (curriculum train step on the
compressed model) -> 8 (SeedLM+VPTQ+Hyper) and the final artifact is USABLE (generates).
Phases needing network/data (full curriculum data-gen via OpenRouter) use the hermetic
training-step path. ~5 min, so marked slow (CI fast path skips it).

HONEST: on an UNTRAINED random model the aggressive stages drop retention (hyper ~0.03) - the
high ratio is real but quality needs a TRAINED model + the default retention gates (which would
roll back the lossy stages). This test asserts the pipeline RUNS end-to-end and stays usable.
"""

import tempfile

import pytest
import torch
import torch.optim as optim

from cross_phase.orchestrator.phase4_controller import Phase4Controller
from cross_phase.utils import get_tokenizer
from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from phase5_curriculum.curriculum_generator import Question
from phase5_curriculum.training_loop import CurriculumTrainingLoop
from phase8_compression.compression_engine import CompressionConfig, CompressionEngine


def _tiny_cognate():
    cfg = Phase1Config(specialization="reasoning")
    cfg.titans_config.d_model, cfg.titans_config.n_heads = 128, 4
    cfg.titans_config.d_ff, cfg.titans_config.d_mem, cfg.titans_config.mag_hidden = 256, 64, 64
    cfg.titans_config.n_layers = 2
    cfg.trm_config.T_max, cfg.trm_config.micro_steps = 2, 1
    cfg.trm_config.step_weights = [1.0, 1.0, 1.0]
    return TRMTitansMAGModel(cfg).eval(), cfg


@pytest.mark.slow
def test_full_pipeline_1_4_5_8_and_inference():
    torch.manual_seed(0)
    model, cfg = _tiny_cognate()
    tok = get_tokenizer("gpt2")
    ids = torch.randint(0, cfg.titans_config.vocab_size, (1, 16))

    # Phase 1: real Cognate forward + generate
    assert model(ids).logits.shape == (1, 16, cfg.titans_config.vocab_size)
    assert model.generate(ids, max_new_tokens=8).shape == (1, 24)

    # Phase 4: BitNet quantization
    p4 = Phase4Controller(config={}, session_id="e2e")
    p4.input_tokenizer = tok
    r4 = p4.execute([model])
    assert r4.success and r4.model is not None

    # Phase 5: curriculum training step on the COMPRESSED model
    qm = r4.model
    opt = optim.SGD([p for p in qm.parameters() if p.requires_grad], lr=1e-4)
    q = Question(
        id="q",
        level=1,
        original_difficulty=10,
        question="What is 2+2?",
        source="t",
        test_cases=[],
        hints=[],
    )
    res = {"success": True, "response": "The answer is 4.", "error": None, "prompt": "What is 2+2?"}
    loss = object.__new__(CurriculumTrainingLoop)._train_step(qm, opt, q, res, tok)
    assert isinstance(loss, float)

    # Phase 8: SeedLM + VPTQ + Hyper (retention floored so stages apply on the untrained model)
    ccfg = CompressionConfig(
        seedlm_enabled=True,
        vptq_enabled=True,
        hyper_enabled=True,
        run_benchmarks=False,
        min_retention_seedlm=0.0,
        min_retention_vptq=0.0,
        min_retention_final=0.0,
        artifacts_dir=tempfile.mkdtemp(),
    )
    r8 = CompressionEngine(ccfg).run(qm, tokenizer=tok)
    assert r8.success
    assert r8.total_compression > 1.0  # real compression
    assert r8.artifact_path  # a real on-disk artifact

    # Usability: the final model still generates
    gen = r8.model.generate(ids, max_new_tokens=8)
    assert gen.shape == (1, 24), "final compressed model must be usable for inference"
