"""
Comprehensive unit tests for Phase 2 fitness evaluation system.

Tests cover:
- Perplexity calculation
- Accuracy measurement
- Speed benchmarking
- Memory measurement
- Composite fitness scoring
- Fitness caching
- FitnessEvaluator API
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.phase2_evomerge.fitness import (
    DEFAULT_EXPECTED,
    DEFAULT_WEIGHTS,
    FitnessCache,
    FitnessEvaluator,
    benchmark_speed,
    calculate_accuracy,
    calculate_perplexity,
    compute_composite_fitness,
    measure_memory_usage,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.linear = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids):
            # input_ids: (batch_size, seq_len)
            embeds = self.embedding(input_ids)  # (batch_size, seq_len, hidden)
            logits = self.linear(embeds)  # (batch_size, seq_len, vocab_size)
            return logits

    return SimpleModel()


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset for testing."""
    # Create synthetic data
    vocab_size = 1000
    batch_size = 32
    seq_len = 64
    num_batches = 10

    # Generate random input_ids and labels
    input_ids = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))

    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


@pytest.fixture
def cuda_available():
    """Check if CUDA is available (skip memory/speed tests if not)."""
    return torch.cuda.is_available()


# ============================================================================
# Perplexity Tests
# ============================================================================


@pytest.mark.phase2
@pytest.mark.fitness
class TestPerplexityCalculation:
    """Tests for perplexity calculation."""

    def test_perplexity_returns_positive_float(self, simple_model, dummy_dataset):
        """Perplexity should return a positive float."""
        ppl = calculate_perplexity(simple_model, dummy_dataset, device="cpu", mixed_precision=False)

        assert isinstance(ppl, float)
        assert ppl > 0
        assert not math.isnan(ppl)
        assert not math.isinf(ppl)

    def test_perplexity_max_batches_limit(self, simple_model, dummy_dataset):
        """Test max_batches parameter limits evaluation."""
        ppl_full = calculate_perplexity(
            simple_model, dummy_dataset, device="cpu", mixed_precision=False, max_batches=None
        )

        ppl_limited = calculate_perplexity(
            simple_model, dummy_dataset, device="cpu", mixed_precision=False, max_batches=2
        )

        # Both should be valid, may differ slightly
        assert isinstance(ppl_full, float)
        assert isinstance(ppl_limited, float)
        assert ppl_full > 0
        assert ppl_limited > 0

    def test_perplexity_empty_dataset_raises(self, simple_model):
        """Empty dataset should raise ValueError."""
        empty_loader = DataLoader(TensorDataset(torch.tensor([]), torch.tensor([])))

        with pytest.raises(ValueError, match="No batches processed"):
            calculate_perplexity(simple_model, empty_loader, device="cpu")


# ============================================================================
# Accuracy Tests
# ============================================================================


@pytest.mark.phase2
@pytest.mark.fitness
class TestAccuracyCalculation:
    """Tests for accuracy calculation."""

    def test_accuracy_returns_value_in_range(self, simple_model, dummy_dataset):
        """Accuracy should be between 0.0 and 1.0."""
        acc = calculate_accuracy(simple_model, dummy_dataset, device="cpu")

        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_accuracy_max_batches_limit(self, simple_model, dummy_dataset):
        """Test max_batches parameter."""
        acc_full = calculate_accuracy(simple_model, dummy_dataset, device="cpu", max_batches=None)

        acc_limited = calculate_accuracy(simple_model, dummy_dataset, device="cpu", max_batches=2)

        # Both should be valid
        assert 0.0 <= acc_full <= 1.0
        assert 0.0 <= acc_limited <= 1.0

    def test_accuracy_empty_dataset_returns_zero(self, simple_model):
        """Empty dataset should return 0.0."""
        empty_loader = DataLoader(TensorDataset(torch.tensor([[]]), torch.tensor([[]])))

        acc = calculate_accuracy(simple_model, empty_loader, device="cpu")
        assert acc == 0.0


# ============================================================================
# Speed Benchmark Tests
# ============================================================================


@pytest.mark.phase2
@pytest.mark.fitness
class TestSpeedBenchmark:
    """Tests for speed benchmarking."""

    def test_speed_returns_positive_float(self, simple_model):
        """Speed should return positive tokens/second."""
        batch = torch.randint(0, 1000, (32, 64))

        tokens_per_sec = benchmark_speed(
            simple_model, batch, device="cpu", num_warmup=2, num_iterations=10
        )

        assert isinstance(tokens_per_sec, float)
        assert tokens_per_sec > 0

    def test_speed_warmup_affects_timing(self, simple_model):
        """Warmup should stabilize timing."""
        batch = torch.randint(0, 1000, (32, 64))

        # With warmup
        speed_warm = benchmark_speed(
            simple_model, batch, device="cpu", num_warmup=10, num_iterations=20
        )

        # Without warmup
        speed_cold = benchmark_speed(
            simple_model, batch, device="cpu", num_warmup=0, num_iterations=20
        )

        # Both should be positive
        assert speed_warm > 0
        assert speed_cold > 0


# ============================================================================
# Memory Measurement Tests
# ============================================================================


@pytest.mark.phase2
@pytest.mark.fitness
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMemoryMeasurement:
    """Tests for memory measurement (requires CUDA)."""

    def test_memory_returns_positive_mb(self, simple_model):
        """Memory should return positive MB value."""
        model_cuda = simple_model.cuda()
        batch = torch.randint(0, 1000, (32, 64)).cuda()

        memory_mb = measure_memory_usage(model_cuda, batch, device="cuda")

        assert isinstance(memory_mb, float)
        assert memory_mb > 0

    def test_memory_cpu_device_raises(self, simple_model):
        """Memory measurement on CPU should raise RuntimeError."""
        batch = torch.randint(0, 1000, (32, 64))

        with pytest.raises(RuntimeError, match="requires CUDA"):
            measure_memory_usage(simple_model, batch, device="cpu")


# ============================================================================
# Composite Fitness Tests
# ============================================================================


@pytest.mark.phase2
@pytest.mark.fitness
class TestCompositeFitness:
    """Tests for composite fitness scoring."""

    def test_default_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        weight_sum = sum(DEFAULT_WEIGHTS.values())
        assert abs(weight_sum - 1.0) < 1e-6

    def test_composite_fitness_calculation(self):
        """Test basic composite fitness calculation."""
        result = compute_composite_fitness(
            perplexity=15.0, accuracy=0.5, speed=1200.0, memory=500.0
        )

        assert "composite" in result
        assert "components" in result
        assert isinstance(result["composite"], float)
        assert result["composite"] > 0

    def test_custom_weights(self):
        """Test custom fitness weights."""
        custom_weights = {"perplexity": 0.3, "accuracy": 0.2, "speed": 0.4, "memory": 0.1}

        result = compute_composite_fitness(
            perplexity=15.0, accuracy=0.5, speed=1200.0, memory=500.0, weights=custom_weights
        )

        assert result["composite"] > 0

    def test_weights_must_sum_to_one(self):
        """Weights not summing to 1.0 should raise ValueError."""
        bad_weights = {
            "perplexity": 0.5,
            "accuracy": 0.3,
            "speed": 0.2,
            "memory": 0.05,  # Sum = 1.05
        }

        with pytest.raises(ValueError, match="must sum to 1.0"):
            compute_composite_fitness(
                perplexity=15.0, accuracy=0.5, speed=1200.0, memory=500.0, weights=bad_weights
            )

    def test_negative_values_raise_error(self):
        """Negative metric values should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            compute_composite_fitness(
                perplexity=-5.0, accuracy=0.5, speed=1200.0, memory=500.0  # Invalid
            )

    def test_zero_perplexity_raises_error(self):
        """Zero perplexity should raise ValueError."""
        with pytest.raises(ValueError, match="Perplexity cannot be zero"):
            compute_composite_fitness(
                perplexity=0.0, accuracy=0.5, speed=1200.0, memory=500.0  # Invalid
            )


# ============================================================================
# Fitness Cache Tests
# ============================================================================


@pytest.mark.phase2
@pytest.mark.fitness
class TestFitnessCache:
    """Tests for fitness caching system."""

    def test_cache_miss(self, simple_model):
        """Initial lookup should be a cache miss."""
        cache = FitnessCache()
        result = cache.get(simple_model)
        assert result is None

    def test_cache_hit(self, simple_model):
        """After storing, lookup should be a cache hit."""
        cache = FitnessCache()
        fitness = {"composite": 0.185, "components": {}}

        cache.put(simple_model, fitness)
        result = cache.get(simple_model)

        assert result is not None
        assert result["composite"] == 0.185

    def test_cache_eviction(self):
        """Cache should evict LRU entries when full."""
        cache = FitnessCache(max_size=3)

        # Create 4 different models
        models = [create_simple_model() for _ in range(4)]
        fitness_values = [{"composite": 0.1 * i, "components": {}} for i in range(4)]

        # Fill cache (models 0, 1, 2)
        for model, fitness in zip(models[:3], fitness_values[:3]):
            cache.put(model, fitness)

        assert cache.size() == 3

        # Add 4th model (should evict model 0 - LRU)
        cache.put(models[3], fitness_values[3])

        assert cache.size() == 3
        assert cache.get(models[0]) is None  # Evicted
        assert cache.get(models[1]) is not None
        assert cache.get(models[2]) is not None
        assert cache.get(models[3]) is not None

    def test_cache_clear(self, simple_model):
        """Clear should remove all entries."""
        cache = FitnessCache()
        cache.put(simple_model, {"composite": 0.185, "components": {}})

        assert cache.size() == 1
        cache.clear()
        assert cache.size() == 0

    def test_hash_model_deterministic(self, simple_model):
        """Same model should produce same hash."""
        cache = FitnessCache()

        hash1 = cache.hash_model(simple_model)
        hash2 = cache.hash_model(simple_model)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex length


# ============================================================================
# FitnessEvaluator API Tests
# ============================================================================


@pytest.mark.phase2
@pytest.mark.fitness
class TestFitnessEvaluator:
    """Tests for FitnessEvaluator main API."""

    def test_evaluator_initialization(self, dummy_dataset):
        """Test basic evaluator initialization."""
        evaluator = FitnessEvaluator(
            validation_dataset=dummy_dataset, device="cpu", cache_enabled=False
        )

        assert evaluator.validation_dataset is dummy_dataset
        assert evaluator.device == "cpu"
        assert evaluator.cache is None  # Cache disabled

    def test_evaluate_single_model(self, simple_model, dummy_dataset):
        """Test evaluating a single model."""
        evaluator = FitnessEvaluator(
            validation_dataset=dummy_dataset,
            device="cpu",
            cache_enabled=False,
            mixed_precision=False,
        )

        fitness = evaluator.evaluate(simple_model)

        assert "composite" in fitness
        assert "components" in fitness
        assert isinstance(fitness["composite"], float)
        assert fitness["composite"] > 0

    def test_evaluate_batch(self, dummy_dataset):
        """Test evaluating a batch of models."""
        # Create 3 models
        models = [create_simple_model() for _ in range(3)]

        evaluator = FitnessEvaluator(
            validation_dataset=dummy_dataset,
            device="cpu",
            cache_enabled=False,
            mixed_precision=False,
        )

        scores = evaluator.evaluate_batch(models)

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(s > 0 for s in scores)

    def test_cache_integration(self, simple_model, dummy_dataset):
        """Test fitness caching in evaluator."""
        evaluator = FitnessEvaluator(
            validation_dataset=dummy_dataset,
            device="cpu",
            cache_enabled=True,
            mixed_precision=False,
        )

        # First evaluation (cache miss)
        fitness1 = evaluator.evaluate(simple_model)

        # Second evaluation (cache hit)
        fitness2 = evaluator.evaluate(simple_model)

        # Should return same result
        assert fitness1["composite"] == fitness2["composite"]

        # Cache should have 1 entry
        assert evaluator.cache.size() == 1

    def test_clear_cache(self, simple_model, dummy_dataset):
        """Test cache clearing."""
        evaluator = FitnessEvaluator(
            validation_dataset=dummy_dataset, device="cpu", cache_enabled=True
        )

        evaluator.evaluate(simple_model)
        assert evaluator.cache.size() == 1

        evaluator.clear_cache()
        assert evaluator.cache.size() == 0


def create_simple_model():
    """Helper to create SimpleModel instances."""

    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.linear = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids):
            embeds = self.embedding(input_ids)
            logits = self.linear(embeds)
            return logits

    return SimpleModel()
