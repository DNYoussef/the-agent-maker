"""
Integration tests for Phase 3 (Quiet-STaR).

Tests complete pipeline from Phase 2 input to Phase 4 output.

Target: ≥85% coverage
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import json

from src.phase3_quietstar.config import QuietSTaRConfig
from src.phase3_quietstar.vocabulary import prepare_model_for_phase3
from src.phase3_quietstar.step1_baking import (
    PromptBakingTrainer,
    ReasoningDataset,
)
from src.phase3_quietstar.step2_rl import REINFORCETrainer
from src.phase3_quietstar.anti_theater import validate_anti_theater
from src.phase3_quietstar.phase_handoff import (
    validate_full_phase3_pipeline,
)


@pytest.fixture
def mock_model():
    """Mock language model."""
    model = Mock(spec=nn.Module)
    model.config = Mock()
    model.config.hidden_size = 512
    model.config.vocab_size = 50257

    # Mock forward
    mock_output = Mock()
    mock_output.logits = torch.randn(2, 10, 50257)
    mock_output.loss = torch.tensor(0.5)
    mock_output.last_hidden_state = torch.randn(2, 10, 512)
    model.return_value = mock_output

    # Mock parameters
    model.parameters = Mock(
        return_value=[torch.randn(100, 100) for _ in range(10)]
    )

    # Mock state dict
    model.state_dict = Mock(
        return_value={
            f"layer{i}.weight": torch.randn(100, 100) for i in range(10)
        }
    )

    # Mock load_state_dict
    model.load_state_dict = Mock()

    # Mock embeddings
    input_emb = nn.Embedding(50257, 512)
    output_emb = nn.Linear(512, 50257)
    model.get_input_embeddings = Mock(return_value=input_emb)
    model.get_output_embeddings = Mock(return_value=output_emb)
    model.set_input_embeddings = Mock()
    model.set_output_embeddings = Mock()

    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    tokenizer = Mock()
    tokenizer.__len__ = Mock(return_value=50257)
    tokenizer.add_special_tokens = Mock(return_value=8)
    tokenizer.convert_tokens_to_ids = Mock(
        side_effect=lambda t: hash(t) % 100000
    )
    tokenizer.get_vocab = Mock(
        return_value={
            "<think>": 50257,
            "</think>": 50258,
            "<step>": 50259,
            "<reason>": 50260,
            "<mece>": 50261,
            "<falsify>": 50262,
            "<expert>": 50263,
            "<doubt>": 50264,
        }
    )

    # Mock encode
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 50257, (1, 20)),
        "attention_mask": torch.ones(1, 20),
    }

    return tokenizer


@pytest.fixture
def sample_reasoning_data(tmp_path):
    """Create sample reasoning dataset."""
    data = [
        {
            "question": f"Question {i}",
            "reasoning": f"<think><step>Step 1</step></think>",
            "answer": f"Answer {i}",
            "strategy": "chain_of_thought",
            "model": "test",
            "tokens_used": 100,
            "cost_usd": 0.001,
        }
        for i in range(100)
    ]

    data_path = tmp_path / "reasoning_data.json"
    with open(data_path, "w") as f:
        json.dump(data, f)

    return data_path


@pytest.fixture
def config():
    """Phase 3 configuration."""
    return QuietSTaRConfig()


class TestVocabularyIntegration:
    """Test vocabulary integration."""

    def test_prepare_model_adds_tokens(self, mock_model, mock_tokenizer):
        """Test prepare_model_for_phase3 adds tokens."""
        model, tokenizer, vocab = prepare_model_for_phase3(
            mock_model, mock_tokenizer
        )

        # Check tokens were added
        mock_tokenizer.add_special_tokens.assert_called_once()

        # Check embeddings were set
        mock_model.set_input_embeddings.assert_called_once()
        mock_model.set_output_embeddings.assert_called_once()

    def test_thinking_tokens_count(self, mock_model, mock_tokenizer):
        """Test correct number of thinking tokens."""
        model, tokenizer, vocab = prepare_model_for_phase3(
            mock_model, mock_tokenizer
        )

        assert len(vocab.thinking_tokens) == 8


class TestStep1Integration:
    """Test Step 1 (Prompt Baking) integration."""

    def test_trainer_initialization(
        self, mock_model, mock_tokenizer, config
    ):
        """Test PromptBakingTrainer initialization."""
        with patch(
            "src.phase3_quietstar.step1_baking.prepare_model_for_phase3"
        ) as mock_prepare:
            mock_prepare.return_value = (
                mock_model,
                mock_tokenizer,
                Mock(),
            )

            trainer = PromptBakingTrainer(
                mock_model, mock_tokenizer, config, device="cpu"
            )

            assert trainer.model is not None
            assert trainer.tokenizer is not None
            assert trainer.optimizer is not None

    def test_dataset_creation(
        self, sample_reasoning_data, mock_tokenizer
    ):
        """Test ReasoningDataset creation."""
        with open(sample_reasoning_data) as f:
            examples = json.load(f)

        dataset = ReasoningDataset(examples, mock_tokenizer, max_length=512)

        assert len(dataset) == 100

        # Get sample
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample
        assert "strategy" in sample


class TestStep2Integration:
    """Test Step 2 (Quiet-STaR RL) integration."""

    def test_reinforce_trainer_initialization(
        self, mock_model, mock_tokenizer, config
    ):
        """Test REINFORCETrainer initialization."""
        with patch(
            "src.phase3_quietstar.step2_rl.QuietSTaRModel"
        ) as mock_quietstar:
            mock_quietstar.return_value = Mock(spec=nn.Module)

            trainer = REINFORCETrainer(
                mock_model,
                mock_model,  # baked model
                mock_tokenizer,
                config,
                device="cpu",
            )

            assert trainer.model is not None
            assert trainer.baked_model is not None
            assert trainer.optimizer is not None

    def test_compute_reward(self, mock_model, mock_tokenizer, config):
        """Test reward computation."""
        with patch(
            "src.phase3_quietstar.step2_rl.QuietSTaRModel"
        ) as mock_quietstar:
            mock_quietstar.return_value = Mock(spec=nn.Module)

            trainer = REINFORCETrainer(
                mock_model,
                mock_model,
                mock_tokenizer,
                config,
                device="cpu",
            )

            logits_with = torch.randn(2, 10, 50257)
            logits_without = torch.randn(2, 10, 50257)
            labels = torch.randint(0, 50257, (2, 10))

            reward = trainer.compute_reward(
                logits_with, logits_without, labels
            )

            assert reward.shape == (2,)
            assert torch.all((reward == 0) | (reward == 1))

    def test_compute_kl_divergence(
        self, mock_model, mock_tokenizer, config
    ):
        """Test KL divergence computation."""
        with patch(
            "src.phase3_quietstar.step2_rl.QuietSTaRModel"
        ) as mock_quietstar:
            mock_quietstar.return_value = Mock(spec=nn.Module)

            trainer = REINFORCETrainer(
                mock_model,
                mock_model,
                mock_tokenizer,
                config,
                device="cpu",
            )

            logits = torch.randn(2, 10, 50257)
            baked_logits = torch.randn(2, 10, 50257)

            kl_div = trainer.compute_kl_divergence(logits, baked_logits)

            assert isinstance(kl_div, torch.Tensor)
            assert kl_div.dim() == 0  # Scalar


class TestAntiTheaterIntegration:
    """Test anti-theater validation integration."""

    def test_divergence_test(self, mock_model, mock_tokenizer, config):
        """Test divergence test."""
        from src.phase3_quietstar.anti_theater import (
            AntiTheaterValidator,
        )

        with patch.object(
            mock_model, "generate", return_value=torch.randint(0, 50257, (1, 30))
        ):
            validator = AntiTheaterValidator(
                mock_model, mock_tokenizer, config.anti_theater, device="cpu"
            )

            input_ids = torch.randint(0, 50257, (2, 10))

            # Should not raise
            divergence = validator.divergence_test(input_ids, num_samples=2)

            assert isinstance(divergence, float)
            assert 0.0 <= divergence <= 1.0


class TestPhaseHandoffIntegration:
    """Test phase handoff validation."""

    def test_validate_phase2_input(self, tmp_path):
        """Test Phase 2 input validation."""
        from src.phase3_quietstar.phase_handoff import (
            Phase3HandoffValidator,
        )

        # Create mock Phase 2 checkpoint
        phase2_path = tmp_path / "phase2_model.pt"
        torch.save(
            {
                "model_state_dict": {
                    f"layer{i}.weight": torch.randn(100, 100)
                    for i in range(10)
                },
                "config": {"phase": 2},
                "metadata": {
                    "phase": 2,
                    "champion_selected": True,
                    "fitness_improvement": 0.235,
                },
            },
            phase2_path,
        )

        registry_path = tmp_path / "registry.db"
        validator = Phase3HandoffValidator(registry_path)

        valid, metadata = validator.validate_phase2_input(phase2_path)

        assert valid is True
        assert metadata["fitness_gain"] == 0.235
        assert metadata["phase"] == 2

    def test_validate_phase3_output(self, tmp_path):
        """Test Phase 3 output validation."""
        from src.phase3_quietstar.phase_handoff import (
            Phase3HandoffValidator,
        )

        # Create mock checkpoints
        final_path = tmp_path / "phase3_final.pt"
        baked_path = tmp_path / "phase3_baked.pt"
        rl_path = tmp_path / "phase3_rl.pt"

        torch.save(
            {
                "model_state_dict": {
                    f"layer{i}.weight": torch.randn(100, 100)
                    for i in range(10)
                },
                "config": {
                    "thinking_tokens": [
                        "<think>",
                        "</think>",
                        "<step>",
                        "<reason>",
                        "<mece>",
                        "<falsify>",
                        "<expert>",
                        "<doubt>",
                    ]
                },
                "anti_theater_results": {
                    "divergence": 0.35,
                    "ablation": 0.05,
                    "correlation": 0.62,
                    "all_passed": True,
                },
            },
            final_path,
        )

        torch.save(
            {"final_accuracy": 0.87, "strategy_accuracies": {}}, baked_path
        )

        torch.save({"reward_history": [0.5] * 1000, "episode": 1000}, rl_path)

        registry_path = tmp_path / "registry.db"
        validator = Phase3HandoffValidator(registry_path)

        valid, metadata = validator.validate_phase3_output(
            final_path, baked_path, rl_path
        )

        assert valid is True
        assert metadata["num_thinking_tokens"] == 8
        assert metadata["baking_accuracy"] == 0.87
        assert metadata["anti_theater_passed"] is True


@pytest.mark.integration
class TestFullPipeline:
    """Test complete Phase 3 pipeline."""

    def test_end_to_end_pipeline(self, tmp_path):
        """Test end-to-end Phase 2→3→4 pipeline."""
        # Create all required checkpoints
        phase2_path = tmp_path / "phase2_model.pt"
        phase3_baked_path = tmp_path / "phase3_baked.pt"
        phase3_rl_path = tmp_path / "phase3_rl.pt"
        phase3_final_path = tmp_path / "phase3_final.pt"
        registry_path = tmp_path / "registry.db"

        # Phase 2 checkpoint
        torch.save(
            {
                "model_state_dict": {
                    f"layer{i}.weight": torch.randn(100, 100)
                    for i in range(10)
                },
                "config": {},
                "metadata": {
                    "phase": 2,
                    "champion_selected": True,
                    "fitness_improvement": 0.25,
                },
            },
            phase2_path,
        )

        # Phase 3 Step 1 (baked)
        torch.save(
            {"final_accuracy": 0.86, "strategy_accuracies": {}},
            phase3_baked_path,
        )

        # Phase 3 Step 2 (RL)
        torch.save(
            {"reward_history": [0.6] * 1000, "episode": 1000}, phase3_rl_path
        )

        # Phase 3 final
        torch.save(
            {
                "model_state_dict": {
                    f"layer{i}.weight": torch.randn(100, 100)
                    for i in range(10)
                },
                "config": {"thinking_tokens": ["<think>"] * 8},
                "anti_theater_results": {"all_passed": True},
            },
            phase3_final_path,
        )

        # Validate pipeline
        valid = validate_full_phase3_pipeline(
            phase2_path,
            phase3_baked_path,
            phase3_rl_path,
            phase3_final_path,
            registry_path,
            session_id="test_session",
        )

        assert valid is True


@pytest.mark.parametrize("num_thoughts", [2, 4, 8])
def test_different_thought_counts(mock_model, mock_tokenizer, config, num_thoughts):
    """Test with different numbers of thoughts."""
    config.rl.num_thoughts = num_thoughts

    with patch("src.phase3_quietstar.step2_rl.QuietSTaRModel") as mock_quietstar:
        mock_quietstar.return_value = Mock(spec=nn.Module)

        trainer = REINFORCETrainer(
            mock_model, mock_model, mock_tokenizer, config, device="cpu"
        )

        # Should initialize without error
        assert trainer.config.rl.num_thoughts == num_thoughts
