"""E6 gate - CRUCIBLE: Phase 5 training signal is correct.

Synthesis: _train_step trained on question.question ONLY (ignored result['response'], so
the model never saw the correct answer) and used ignore_index=0 in the CE loss, masking
token id 0 ('!' in gpt2) instead of padding. E6 trains on prompt+response and masks via
ignore_index=-100 (padding masked through attention_mask).
"""

import torch
import torch.nn as nn

from phase5_curriculum.curriculum_generator import Question
from phase5_curriculum.training_loop import CurriculumTrainingLoop


class _TinyLM(nn.Module):
    def __init__(self, vocab=10):
        super().__init__()
        self.emb = nn.Embedding(vocab, vocab)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        out = type("O", (), {})()
        out.logits = self.emb(input_ids)  # [B, T, vocab]
        return out


def _question():
    return Question(
        id="q",
        level=1,
        original_difficulty=10,
        question="solve x",
        source="t",
        test_cases=[],
        hints=[],
    )


def test_lm_loss_counts_token_id_zero_and_ignores_minus_100():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 5, requires_grad=True)
    # targets include id 0 (token '!') - the OLD ignore_index=0 wrongly dropped these.
    labels = torch.tensor([[1, 0, 2, 3]])
    loss = CurriculumTrainingLoop._lm_loss(logits, labels)
    assert torch.isfinite(loss) and loss.item() > 0, "token id 0 must contribute to the loss"
    # everything masked -> nothing to learn (proves -100 is the ignore value).
    all_masked = torch.full((1, 4), -100)
    assert torch.isnan(CurriculumTrainingLoop._lm_loss(logits, all_masked))


def test_train_step_trains_on_the_response():
    seen = []

    def spy_tok(text, **kw):
        seen.append(text)
        return {
            "input_ids": torch.tensor([[1, 2, 3, 0, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

    model = _TinyLM(vocab=10)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    result = {"success": True, "response": "the answer is 42", "error": None, "prompt": "solve x"}

    tl = object.__new__(CurriculumTrainingLoop)
    loss = tl._train_step(model, opt, _question(), result, spy_tok)

    assert any("the answer is 42" in t for t in seen), "training must include result['response']"
    assert isinstance(loss, float)
