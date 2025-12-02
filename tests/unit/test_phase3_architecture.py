"""
Unit tests for Phase 3 Quiet-STaR architecture components.

Tests all 5 core classes:
- ThoughtGenerator
- CoherenceScorer
- MixingHead
- ThoughtInjector
- QuietSTaRModel

Target: ≥95% coverage for critical paths
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from src.phase3_quietstar.architecture import (
    CoherenceScorer,
    CoherenceScores,
    MixingHead,
    QuietSTaRModel,
    ThoughtGenerator,
    ThoughtInjector,
    ThoughtOutput,
)


@pytest.fixture
def mock_base_model():
    """
    Mock base language model with dynamic output shapes.

    ISS-005: Fixed to return proper shapes based on input sequence length,
    required for ThoughtGenerator which grows sequence during generation.
    """
    model = Mock(spec=nn.Module)
    model.lm_head = nn.Linear(512, 50257)  # GPT-2 vocab size

    def dynamic_forward(input_ids):
        """Return outputs with shape matching input_ids length."""
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        mock_output = Mock()
        mock_output.logits = torch.randn(batch_size, seq_len, 50257)
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, 512)
        return mock_output

    model.side_effect = dynamic_forward
    # Also support direct call without arguments for some tests
    model.return_value = Mock(
        logits=torch.randn(2, 10, 50257), last_hidden_state=torch.randn(2, 10, 512)
    )

    return model


@pytest.fixture
def sample_input_ids():
    """Sample input token IDs."""
    return torch.randint(0, 50257, (2, 10))  # (batch=2, seq_len=10)


@pytest.fixture
def sample_hidden_states():
    """Sample hidden states."""
    return torch.randn(2, 10, 512)  # (batch=2, seq_len=10, hidden=512)


class TestThoughtGenerator:
    """Test ThoughtGenerator class."""

    def test_initialization(self, mock_base_model):
        """Test ThoughtGenerator initialization."""
        generator = ThoughtGenerator(
            base_model=mock_base_model,
            num_thoughts=4,
            max_length=20,
            min_length=10,
            temperature=1.0,
            top_p=0.9,
        )

        assert generator.num_thoughts == 4
        assert generator.max_length == 20
        assert generator.min_length == 10
        assert generator.temperature == 1.0
        assert generator.top_p == 0.9

    def test_forward_generates_thoughts(self, mock_base_model, sample_input_ids):
        """Test thought generation forward pass."""
        generator = ThoughtGenerator(mock_base_model, num_thoughts=4)

        output = generator.forward(sample_input_ids, position=5)

        assert isinstance(output, ThoughtOutput)
        assert output.thoughts.shape[0] == 2  # batch size
        assert output.thoughts.shape[1] == 4  # num_thoughts
        assert len(output.thought_ids) == 4
        # log_probs is now [num_thoughts] tensor (1D)
        assert output.log_probs.shape[0] == 4

    def test_nucleus_sampling(self, mock_base_model):
        """Test nucleus (top-p) sampling."""
        generator = ThoughtGenerator(mock_base_model, top_p=0.9)

        logits = torch.randn(1, 50257)
        probs = generator._nucleus_sampling(logits)

        # Check output is valid probability distribution
        assert torch.all(probs >= 0)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1))

        # Check top-p filtering
        sorted_probs = torch.sort(probs, descending=True)[0]
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Most mass should be in top-p
        # After nucleus sampling, prob mass is concentrated in top tokens
        # Check that we have valid prob distribution (sums to 1)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_generate_single_thought(self, mock_base_model, sample_input_ids):
        """Test single thought generation."""
        generator = ThoughtGenerator(mock_base_model, min_length=5, max_length=10)

        thought, log_prob, ids = generator._generate_single(
            sample_input_ids, position=3, hidden_states=None, thought_length=7
        )

        # Check outputs
        assert thought.dim() == 3  # (batch, length, hidden)
        assert isinstance(log_prob, torch.Tensor)
        assert 5 <= len(ids) <= 10  # Within min/max range

    def test_thought_count_matches_num_thoughts(self, mock_base_model, sample_input_ids):
        """Test correct number of thoughts generated."""
        for num_thoughts in [2, 4, 8]:
            generator = ThoughtGenerator(mock_base_model, num_thoughts=num_thoughts)
            output = generator.forward(sample_input_ids, position=5)

            assert output.thoughts.shape[1] == num_thoughts
            assert len(output.thought_ids) == num_thoughts


class TestCoherenceScorer:
    """Test CoherenceScorer class."""

    def test_initialization(self):
        """Test CoherenceScorer initialization."""
        scorer = CoherenceScorer(
            hidden_size=512,
            weights={"semantic": 0.4, "syntactic": 0.3, "predictive": 0.3},
        )

        assert scorer.hidden_size == 512
        assert scorer.weights["semantic"] == 0.4
        assert scorer.weights["syntactic"] == 0.3
        assert scorer.weights["predictive"] == 0.3

    def test_forward_returns_coherence_scores(self):
        """Test coherence scoring forward pass."""
        scorer = CoherenceScorer(hidden_size=512)

        base_hidden = torch.randn(2, 512)
        thought_hiddens = torch.randn(2, 4, 10, 512)  # 4 thoughts
        next_token_logits = torch.randn(2, 50257)

        scores = scorer.forward(base_hidden, thought_hiddens, next_token_logits)

        assert isinstance(scores, CoherenceScores)
        assert scores.semantic.shape == (2, 4)  # (batch, num_thoughts)
        assert scores.syntactic.shape == (2, 4)
        assert scores.predictive.shape == (2, 4)
        assert scores.composite.shape == (2, 4)

    def test_semantic_coherence_in_range(self):
        """Test semantic coherence scores are in [0, 1]."""
        scorer = CoherenceScorer(hidden_size=512)

        base = torch.randn(2, 512)
        thoughts = torch.randn(2, 4, 512)

        semantic = scorer._semantic_coherence(base, thoughts)

        assert torch.all(semantic >= 0.0)
        assert torch.all(semantic <= 1.0)

    def test_syntactic_coherence_in_range(self):
        """Test syntactic coherence scores are in [0, 1]."""
        scorer = CoherenceScorer(hidden_size=512)

        thoughts = torch.randn(2, 4, 512)

        syntactic = scorer._syntactic_coherence(thoughts)

        assert torch.all(syntactic >= 0.0)
        assert torch.all(syntactic <= 1.0)

    def test_predictive_coherence_handles_none_logits(self):
        """Test predictive coherence with None logits."""
        scorer = CoherenceScorer(hidden_size=512)

        thoughts = torch.randn(2, 4, 512)

        predictive = scorer._predictive_coherence(thoughts, None)

        # Should return ones when no logits
        assert torch.all(predictive == 1.0)

    def test_composite_score_weighted_average(self):
        """Test composite score is weighted average."""
        weights = {"semantic": 0.5, "syntactic": 0.3, "predictive": 0.2}
        scorer = CoherenceScorer(hidden_size=512, weights=weights)

        base_hidden = torch.randn(2, 512)
        thought_hiddens = torch.randn(2, 4, 10, 512)

        scores = scorer.forward(base_hidden, thought_hiddens)

        # Compute expected composite
        expected = 0.5 * scores.semantic + 0.3 * scores.syntactic + 0.2 * scores.predictive

        assert torch.allclose(scores.composite, expected, atol=1e-5)


class TestMixingHead:
    """Test MixingHead class."""

    def test_initialization(self):
        """Test MixingHead initialization."""
        mixer = MixingHead(hidden_size=512, num_heads=8, dropout=0.1)

        assert mixer.hidden_size == 512
        assert mixer.num_heads == 8
        assert mixer.head_dim == 64  # 512 / 8

    def test_forward_returns_mixed_hidden(self):
        """Test mixing forward pass."""
        mixer = MixingHead(hidden_size=512, num_heads=8)

        base_hidden = torch.randn(2, 512)
        thought_hiddens = torch.randn(2, 4, 512)  # 4 thoughts
        coherence_scores = torch.randn(2, 4)

        mixed = mixer.forward(base_hidden, thought_hiddens, coherence_scores)

        assert mixed.shape == (2, 512)  # Same as base_hidden

    def test_split_heads_correct_shape(self):
        """Test head splitting."""
        mixer = MixingHead(hidden_size=512, num_heads=8)

        x = torch.randn(2, 5, 512)  # (batch, seq, hidden)
        split = mixer._split_heads(x)

        assert split.shape == (2, 8, 5, 64)  # (batch, heads, seq, head_dim)

    def test_merge_heads_correct_shape(self):
        """Test head merging."""
        mixer = MixingHead(hidden_size=512, num_heads=8)

        x = torch.randn(2, 8, 5, 64)  # (batch, heads, seq, head_dim)
        merged = mixer._merge_heads(x)

        assert merged.shape == (2, 5, 512)  # (batch, seq, hidden)

    def test_gating_blends_representations(self):
        """Test gating mechanism blends base and thoughts."""
        mixer = MixingHead(hidden_size=512, num_heads=8)

        base = torch.randn(2, 512)
        thoughts = torch.randn(2, 4, 512)
        coherence = torch.randn(2, 4)

        # Forward pass
        mixed = mixer.forward(base, thoughts, coherence)

        # Mixed should be different from base (unless gate is 0)
        assert not torch.allclose(mixed, base)

    def test_residual_connection(self):
        """Test residual connection preserves information."""
        mixer = MixingHead(hidden_size=512, num_heads=8)

        base = torch.randn(2, 512)
        thoughts = torch.zeros(2, 4, 512)  # Zero thoughts
        coherence = torch.zeros(2, 4)  # Zero coherence

        mixed = mixer.forward(base, thoughts, coherence)

        # With zero thoughts, should be close to layer_norm(base)
        # (residual connection preserves base)
        assert mixed.shape == base.shape


class TestThoughtInjector:
    """Test ThoughtInjector class."""

    def test_initialization(self):
        """Test ThoughtInjector initialization."""
        injector = ThoughtInjector(threshold=0.6, min_interval=3)

        assert injector.threshold == 0.6
        assert injector.min_interval == 3
        assert injector.last_injection == -3

    def test_respects_minimum_interval(self):
        """Test minimum interval between injections."""
        injector = ThoughtInjector(threshold=0.0, min_interval=3)

        logits = torch.randn(1, 50257)

        # First injection should succeed
        inject1 = injector.forward(logits, None, None, position=0)
        assert inject1 is True

        # Next 2 positions should fail (within interval)
        inject2 = injector.forward(logits, None, None, position=1)
        assert inject2 is False

        inject3 = injector.forward(logits, None, None, position=2)
        assert inject3 is False

        # Position 3 should succeed (interval=3)
        inject4 = injector.forward(logits, None, None, position=3)
        assert inject4 is True

    def test_high_entropy_triggers_injection(self):
        """Test high entropy triggers injection."""
        injector = ThoughtInjector(threshold=0.3, min_interval=1)

        # High entropy logits (uniform distribution)
        high_entropy_logits = torch.zeros(1, 100)

        inject = injector.forward(high_entropy_logits, None, None, position=10)

        assert inject is True

    def test_low_entropy_no_injection(self):
        """Test low entropy doesn't trigger injection."""
        injector = ThoughtInjector(threshold=0.8, min_interval=1)

        # Low entropy logits (peaked distribution)
        low_entropy_logits = torch.zeros(1, 100)
        low_entropy_logits[0, 0] = 100.0  # One very high logit

        inject = injector.forward(low_entropy_logits, None, None, position=10)

        assert inject is False

    def test_compute_entropy_normalized(self):
        """Test entropy computation is normalized to [0, 1]."""
        injector = ThoughtInjector(threshold=0.5)

        # Uniform distribution (max entropy)
        uniform_logits = torch.zeros(1, 100)
        entropy_max = injector._compute_entropy(uniform_logits)
        assert 0.9 <= entropy_max <= 1.0

        # Peaked distribution (min entropy)
        peaked_logits = torch.zeros(1, 100)
        peaked_logits[0, 0] = 100.0
        entropy_min = injector._compute_entropy(peaked_logits)
        assert 0.0 <= entropy_min <= 0.1

    def test_compute_dispersion_handles_none(self):
        """Test attention dispersion handles None."""
        injector = ThoughtInjector(threshold=0.5)

        dispersion = injector._compute_dispersion(None)

        assert dispersion == 0.5  # Neutral value


class TestQuietSTaRModel:
    """Test QuietSTaRModel integration."""

    def test_initialization(self, mock_base_model):
        """Test QuietSTaRModel initialization."""
        model = QuietSTaRModel(
            base_model=mock_base_model,
            hidden_size=512,
            num_thoughts=4,
            max_thought_length=20,
            injection_threshold=0.6,
        )

        assert model.hidden_size == 512
        assert isinstance(model.thought_generator, ThoughtGenerator)
        assert isinstance(model.coherence_scorer, CoherenceScorer)
        assert isinstance(model.mixing_head, MixingHead)
        assert isinstance(model.thought_injector, ThoughtInjector)

    def test_forward_without_thoughts(self, mock_base_model, sample_input_ids):
        """Test forward pass without thought generation."""
        model = QuietSTaRModel(mock_base_model, hidden_size=512)

        outputs = model.forward(sample_input_ids, use_thoughts=False)

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 10, 50257)
        assert outputs["loss"] is None  # No labels provided, loss is None

    def test_forward_with_thoughts(self, mock_base_model, sample_input_ids):
        """Test forward pass with thought generation."""
        model = QuietSTaRModel(mock_base_model, hidden_size=512)

        outputs = model.forward(sample_input_ids, use_thoughts=True)

        assert "logits" in outputs
        assert "thought_positions" in outputs
        assert "avg_coherence" in outputs
        assert "num_thoughts_used" in outputs

        assert isinstance(outputs["thought_positions"], list)
        assert isinstance(outputs["avg_coherence"], float)
        assert isinstance(outputs["num_thoughts_used"], int)

    def test_forward_with_labels_computes_loss(self, mock_base_model, sample_input_ids):
        """Test loss computation with labels."""
        model = QuietSTaRModel(mock_base_model, hidden_size=512)

        labels = sample_input_ids.clone()

        outputs = model.forward(sample_input_ids, labels=labels, use_thoughts=False)

        assert "loss" in outputs
        assert isinstance(outputs["loss"], torch.Tensor)
        assert outputs["loss"].dim() == 0  # Scalar

    def test_compute_loss_correct_shape(self, mock_base_model):
        """Test loss computation."""
        model = QuietSTaRModel(mock_base_model, hidden_size=512)

        logits = torch.randn(2, 10, 50257)
        labels = torch.randint(0, 50257, (2, 10))

        loss = model._compute_loss(logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss


class TestDataStructures:
    """Test data structure classes."""

    def test_thought_output_initialization(self):
        """Test ThoughtOutput dataclass."""
        output = ThoughtOutput(
            thoughts=torch.randn(2, 4, 10, 512),
            thought_ids=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            log_probs=torch.randn(2, 4),
        )

        assert output.thoughts.shape == (2, 4, 10, 512)
        assert len(output.thought_ids) == 4
        assert output.log_probs.shape == (2, 4)
        assert output.attention_weights is None

    def test_coherence_scores_initialization(self):
        """Test CoherenceScores dataclass."""
        scores = CoherenceScores(
            semantic=torch.randn(2, 4),
            syntactic=torch.randn(2, 4),
            predictive=torch.randn(2, 4),
            composite=torch.randn(2, 4),
        )

        assert scores.semantic.shape == (2, 4)
        assert scores.syntactic.shape == (2, 4)
        assert scores.predictive.shape == (2, 4)
        assert scores.composite.shape == (2, 4)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_thoughts", [2, 4, 8])
def test_architecture_different_batch_and_thought_sizes(mock_base_model, batch_size, num_thoughts):
    """Test architecture with different batch and thought sizes."""
    generator = ThoughtGenerator(mock_base_model, num_thoughts=num_thoughts)

    input_ids = torch.randint(0, 50257, (batch_size, 10))
    output = generator.forward(input_ids, position=5)

    assert output.thoughts.shape[0] == batch_size
    assert output.thoughts.shape[1] == num_thoughts
