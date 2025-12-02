"""
Unit tests for Phase 3 vocabulary extension.

Tests thinking token management and model preparation.

Target: ≥95% coverage
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch

from src.phase3_quietstar.vocabulary import (
    ThinkingVocabulary,
    prepare_model_for_phase3,
    compute_thinking_token_usage,
)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    tokenizer = Mock()
    tokenizer.__len__ = Mock(return_value=50257)  # GPT-2 vocab size

    # Mock methods
    tokenizer.add_special_tokens = Mock(return_value=8)
    tokenizer.convert_tokens_to_ids = Mock(
        side_effect=lambda token: hash(token) % 100000
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

    return tokenizer


@pytest.fixture
def mock_model():
    """Mock model with embeddings."""
    model = Mock(spec=nn.Module)

    # Input embeddings
    input_embeddings = nn.Embedding(50257, 512)
    model.get_input_embeddings = Mock(return_value=input_embeddings)

    # Output embeddings (LM head)
    lm_head = nn.Linear(512, 50257)
    model.get_output_embeddings = Mock(return_value=lm_head)

    # Set methods
    model.set_input_embeddings = Mock()
    model.set_output_embeddings = Mock()

    return model


class TestThinkingVocabulary:
    """Test ThinkingVocabulary class."""

    def test_initialization_core_tokens(self, mock_tokenizer):
        """Test initialization with core tokens only."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        assert vocab.tokenizer == mock_tokenizer
        assert not vocab.use_extended
        assert len(vocab.thinking_tokens) == 8  # Core tokens
        assert vocab.original_vocab_size == 50257

    def test_initialization_with_extended_tokens(self, mock_tokenizer):
        """Test initialization with extended tokens."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=True)

        assert vocab.use_extended
        assert len(vocab.thinking_tokens) == 12  # 8 core + 4 extended

    def test_core_tokens_defined(self):
        """Test core tokens are properly defined."""
        expected_core = [
            "<think>",
            "</think>",
            "<step>",
            "<reason>",
            "<mece>",
            "<falsify>",
            "<expert>",
            "<doubt>",
        ]

        assert ThinkingVocabulary.CORE_TOKENS == expected_core

    def test_extended_tokens_defined(self):
        """Test extended tokens are properly defined."""
        expected_extended = [
            "<bayesian>",
            "<multidomain>",
            "<correct>",
            "<uncertain>",
        ]

        assert ThinkingVocabulary.EXTENDED_TOKENS == expected_extended

    def test_add_tokens_calls_tokenizer(self, mock_tokenizer):
        """Test add_tokens calls tokenizer correctly."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        num_added = vocab.add_tokens()

        assert num_added == 8
        mock_tokenizer.add_special_tokens.assert_called_once()

    def test_add_tokens_builds_mappings(self, mock_tokenizer):
        """Test add_tokens builds token mappings."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        vocab.add_tokens()

        # Check mappings are built
        assert len(vocab.token_to_id) == 8
        assert len(vocab.id_to_token) == 8

        # Check bidirectional mapping
        for token in vocab.thinking_tokens:
            token_id = vocab.token_to_id[token]
            assert vocab.id_to_token[token_id] == token

    def test_resize_embeddings_preserves_old_weights(
        self, mock_tokenizer, mock_model
    ):
        """Test resizing preserves old embedding weights."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        # Mock new vocab size
        mock_tokenizer.__len__ = Mock(return_value=50265)  # +8 tokens

        old_embeddings = mock_model.get_input_embeddings()
        old_weights = old_embeddings.weight.data.clone()

        new_embeddings, new_lm_head = vocab.resize_embeddings(mock_model)

        # Check old weights preserved
        assert torch.allclose(
            new_embeddings.weight[:50257], old_weights
        )

    def test_resize_embeddings_initializes_new_tokens(
        self, mock_tokenizer, mock_model
    ):
        """Test new token embeddings are initialized."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        mock_tokenizer.__len__ = Mock(return_value=50265)

        new_embeddings, new_lm_head = vocab.resize_embeddings(mock_model)

        # Check new embeddings initialized
        assert new_embeddings.weight.shape[0] == 50265
        assert new_lm_head.weight.shape[0] == 50265

        # Check new embeddings are non-zero
        new_token_embeddings = new_embeddings.weight[50257:]
        assert not torch.all(new_token_embeddings == 0)

    def test_get_token_id_returns_correct_id(self, mock_tokenizer):
        """Test get_token_id returns correct ID."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        token_id = vocab.get_token_id("<think>")

        assert token_id is not None
        assert isinstance(token_id, int)

    def test_get_token_id_returns_none_for_unknown(self, mock_tokenizer):
        """Test get_token_id returns None for unknown token."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        token_id = vocab.get_token_id("<unknown>")

        assert token_id is None

    def test_get_token_returns_correct_string(self, mock_tokenizer):
        """Test get_token returns correct string."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        # Get a token ID
        token_id = vocab.token_to_id["<think>"]

        token = vocab.get_token(token_id)

        assert token == "<think>"

    def test_is_thinking_token_correct(self, mock_tokenizer):
        """Test is_thinking_token identifies correctly."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        thinking_id = vocab.token_to_id["<think>"]
        regular_id = 100  # Not a thinking token

        assert vocab.is_thinking_token(thinking_id) is True
        assert vocab.is_thinking_token(regular_id) is False

    def test_validate_tokens_success(self, mock_tokenizer):
        """Test validate_tokens with all tokens present."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        is_valid = vocab.validate_tokens()

        assert is_valid is True

    def test_validate_tokens_failure(self, mock_tokenizer):
        """Test validate_tokens with missing tokens."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        # Mock missing token
        mock_tokenizer.get_vocab = Mock(return_value={})

        is_valid = vocab.validate_tokens()

        assert is_valid is False

    def test_get_stats_returns_correct_counts(self, mock_tokenizer):
        """Test get_stats returns correct statistics."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        mock_tokenizer.__len__ = Mock(return_value=50265)

        stats = vocab.get_stats()

        assert stats["original_vocab_size"] == 50257
        assert stats["new_vocab_size"] == 50265
        assert stats["thinking_tokens_added"] == 8
        assert stats["core_tokens"] == 8
        assert stats["extended_tokens"] == 0

    def test_format_with_thinking_chain_of_thought(self, mock_tokenizer):
        """Test format_with_thinking for chain-of-thought."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        formatted = vocab.format_with_thinking(
            "Step 1: Analyze", strategy="chain_of_thought"
        )

        assert "<think>" in formatted
        assert "</think>" in formatted
        assert "<step>" in formatted
        assert "Step 1: Analyze" in formatted

    def test_format_with_thinking_mece(self, mock_tokenizer):
        """Test format_with_thinking for MECE."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        formatted = vocab.format_with_thinking(
            "Category 1", strategy="mece_decomposition"
        )

        assert "<mece>" in formatted

    def test_format_with_thinking_falsification(self, mock_tokenizer):
        """Test format_with_thinking for falsification."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        formatted = vocab.format_with_thinking(
            "Test hypothesis", strategy="falsification_testing"
        )

        assert "<falsify>" in formatted

    def test_format_with_thinking_expert(self, mock_tokenizer):
        """Test format_with_thinking for expert perspective."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        formatted = vocab.format_with_thinking(
            "Expert view", strategy="expert_perspective"
        )

        assert "<expert>" in formatted

    def test_format_with_thinking_doubt(self, mock_tokenizer):
        """Test format_with_thinking for self-doubt."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        formatted = vocab.format_with_thinking(
            "Check answer", strategy="self_doubt"
        )

        assert "<doubt>" in formatted

    def test_format_with_thinking_bayesian_extended(self, mock_tokenizer):
        """Test format_with_thinking for Bayesian (extended)."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=True)

        formatted = vocab.format_with_thinking(
            "Update belief", strategy="bayesian_rationalist"
        )

        assert "<bayesian>" in formatted

    def test_extract_thinking_content_single_block(self, mock_tokenizer):
        """Test extract_thinking_content with single block."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        text = "<think>Step 1: First step\nStep 2: Second step</think>"

        extracted = vocab.extract_thinking_content(text)

        assert len(extracted) == 1
        assert "Step 1" in extracted[0]
        assert "Step 2" in extracted[0]

    def test_extract_thinking_content_multiple_blocks(self, mock_tokenizer):
        """Test extract_thinking_content with multiple blocks."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        text = (
            "<think>Block 1</think> Some text <think>Block 2</think>"
        )

        extracted = vocab.extract_thinking_content(text)

        assert len(extracted) == 2
        assert "Block 1" in extracted[0]
        assert "Block 2" in extracted[1]

    def test_count_thinking_tokens_1d(self, mock_tokenizer):
        """Test count_thinking_tokens with 1D tensor."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        # Create tensor with some thinking tokens
        thinking_ids = [
            vocab.token_to_id["<think>"],
            vocab.token_to_id["</think>"],
        ]
        regular_ids = [100, 200, 300]
        token_ids = torch.tensor(thinking_ids + regular_ids)

        count = vocab.count_thinking_tokens(token_ids)

        assert count == 2

    def test_count_thinking_tokens_2d(self, mock_tokenizer):
        """Test count_thinking_tokens with 2D tensor."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)
        vocab.add_tokens()

        thinking_id = vocab.token_to_id["<think>"]
        token_ids = torch.tensor(
            [[thinking_id, 100, 200], [300, thinking_id, 400]]
        )

        count = vocab.count_thinking_tokens(token_ids)

        assert count == 2


class TestPrepareModelForPhase3:
    """Test prepare_model_for_phase3 function."""

    def test_prepare_model_adds_tokens(
        self, mock_model, mock_tokenizer
    ):
        """Test prepare_model_for_phase3 adds tokens."""
        model, tokenizer, vocab = prepare_model_for_phase3(
            mock_model, mock_tokenizer, use_extended_tokens=False
        )

        mock_tokenizer.add_special_tokens.assert_called_once()

    def test_prepare_model_resizes_embeddings(
        self, mock_model, mock_tokenizer
    ):
        """Test prepare_model_for_phase3 resizes embeddings."""
        mock_tokenizer.__len__ = Mock(
            side_effect=[50257, 50265]
        )  # Before and after

        model, tokenizer, vocab = prepare_model_for_phase3(
            mock_model, mock_tokenizer, use_extended_tokens=False
        )

        mock_model.set_input_embeddings.assert_called_once()
        mock_model.set_output_embeddings.assert_called_once()

    def test_prepare_model_validates_tokens(
        self, mock_model, mock_tokenizer
    ):
        """Test prepare_model_for_phase3 validates tokens."""
        with patch.object(
            ThinkingVocabulary, "validate_tokens", return_value=True
        ):
            model, tokenizer, vocab = prepare_model_for_phase3(
                mock_model, mock_tokenizer
            )

        # Should not raise ValueError

    def test_prepare_model_raises_on_validation_failure(
        self, mock_model, mock_tokenizer
    ):
        """Test prepare_model_for_phase3 raises on validation failure."""
        with patch.object(
            ThinkingVocabulary, "validate_tokens", return_value=False
        ):
            with pytest.raises(ValueError, match="Failed to add"):
                prepare_model_for_phase3(mock_model, mock_tokenizer)

    def test_prepare_model_returns_vocabulary(
        self, mock_model, mock_tokenizer
    ):
        """Test prepare_model_for_phase3 returns vocabulary."""
        model, tokenizer, vocab = prepare_model_for_phase3(
            mock_model, mock_tokenizer
        )

        assert isinstance(vocab, ThinkingVocabulary)


class TestComputeThinkingTokenUsage:
    """Test compute_thinking_token_usage function."""

    def test_compute_usage_all_tags_present(self, mock_tokenizer):
        """Test usage computation with all tags."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        outputs = [
            "<think><step>Step 1</step></think>",
            "<think><mece>Category 1</mece></think>",
            "<think><falsify>Test 1</falsify></think>",
            "<think><doubt>Check</doubt></think>",
        ]

        usage = compute_thinking_token_usage(outputs, vocab)

        assert usage["thinking_tag_usage"] == 1.0  # All have <think>
        assert usage["step_tag_usage"] == 0.25  # 1/4
        assert usage["mece_tag_usage"] == 0.25  # 1/4
        assert usage["falsify_tag_usage"] == 0.25  # 1/4
        assert usage["doubt_tag_usage"] == 0.25  # 1/4

    def test_compute_usage_no_tags(self, mock_tokenizer):
        """Test usage computation with no tags."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        outputs = [
            "Regular output 1",
            "Regular output 2",
        ]

        usage = compute_thinking_token_usage(outputs, vocab)

        assert usage["thinking_tag_usage"] == 0.0
        assert usage["step_tag_usage"] == 0.0
        assert usage["mece_tag_usage"] == 0.0
        assert usage["falsify_tag_usage"] == 0.0
        assert usage["doubt_tag_usage"] == 0.0
        assert usage["overall_usage"] == 0.0

    def test_compute_usage_partial_tags(self, mock_tokenizer):
        """Test usage computation with partial tags."""
        vocab = ThinkingVocabulary(mock_tokenizer, use_extended=False)

        outputs = [
            "<think>Thinking...</think>",
            "No tags",
            "<think><step>Step 1</step></think>",
            "No tags",
        ]

        usage = compute_thinking_token_usage(outputs, vocab)

        assert usage["thinking_tag_usage"] == 0.5  # 2/4
        assert usage["step_tag_usage"] == 0.25  # 1/4


@pytest.mark.parametrize("use_extended", [False, True])
def test_vocabulary_with_extended_parameter(mock_tokenizer, use_extended):
    """Test vocabulary with and without extended tokens."""
    vocab = ThinkingVocabulary(mock_tokenizer, use_extended=use_extended)

    expected_count = 12 if use_extended else 8
    assert len(vocab.thinking_tokens) == expected_count
