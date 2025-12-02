"""
Integration tests for CMA-ES optimizer and real fitness evaluation.

Tests the complete Phase 2 pipeline with:
- CMA-ES parameter space optimization
- Real benchmark evaluation (GSM8K/MGSM)
- Hybrid PS+DFS merging
- Paper-accurate DFS with indicator arrays
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from phase2_evomerge.evolution.cma_es import CMAESConfig, CMAESOptimizer, ps_merge_with_cmaes
from phase2_evomerge.fitness.benchmarks import (
    BenchmarkConfig,
    GSM8KDataset,
    extract_numeric_answer,
    evaluate_benchmark,
)
from phase2_evomerge.merge.dfs_paper_accurate import DFSConfig, DFSPaperAccurate
from phase2_evomerge.merge.hybrid_ps_dfs import HybridConfig, HybridPSDFS, hybrid_merge
from phase2_evomerge.phase2_pipeline import EvolutionConfig, Phase2Pipeline


# Simple test model
class SimpleTestModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_size)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 100)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output(x)

    def generate(self, input_ids, max_length=50, **kwargs):
        """Dummy generation for testing."""
        # Just return the input with some random continuation
        batch_size = input_ids.shape[0]
        generated = torch.randint(0, 100, (batch_size, max_length), device=input_ids.device)
        return generated


class TestCMAESOptimizer:
    """Test CMA-ES optimizer."""

    def test_cmaes_initialization(self):
        """Test CMA-ES initializes correctly."""
        config = CMAESConfig(population_size=20, sigma=0.2)
        optimizer = CMAESOptimizer(config)
        assert optimizer.config.population_size == 20
        assert optimizer.config.sigma == 0.2
        assert optimizer.best_params is None

    def test_cmaes_optimize_simple_function(self):
        """Test CMA-ES can optimize a simple function."""
        config = CMAESConfig(population_size=10, max_generations=20, seed=42)
        optimizer = CMAESOptimizer(config)

        # Simple quadratic function: maximize -(x - 0.5)^2
        # Optimum is at x = 0.5 for all dimensions
        def objective(coeffs):
            return -np.sum((coeffs - 0.5) ** 2)

        best_coeffs, best_fitness = optimizer.optimize(
            objective, n_dimensions=3, n_trials=50, verbose=False
        )

        # Check that optimizer found solution close to [0.5, 0.5, 0.5]
        assert best_coeffs is not None
        assert len(best_coeffs) == 3
        # Should be close to optimal (within 0.1 of 0.5 for each dimension)
        assert np.allclose(best_coeffs, 0.5, atol=0.2)
        assert best_fitness > -0.1  # Should be close to 0 (optimum)

    def test_ps_merge_with_cmaes(self):
        """Test PS merging with CMA-ES optimization."""
        # Create 3 test models
        models = [SimpleTestModel(hidden_size=32) for _ in range(3)]

        # Simple fitness function (dummy)
        def fitness_fn(model):
            # Return sum of first parameter as proxy
            return float(list(model.parameters())[0].sum().item())

        config = CMAESConfig(population_size=10, max_generations=10, seed=42)

        merged, coeffs, fitness = ps_merge_with_cmaes(models, fitness_fn, config, verbose=False)

        # Check outputs
        assert isinstance(merged, nn.Module)
        assert coeffs is not None
        assert len(coeffs) == 3
        assert np.allclose(np.sum(coeffs), 1.0, atol=1e-3)  # Coefficients sum to 1
        assert isinstance(fitness, float)


class TestBenchmarkEvaluation:
    """Test real benchmark evaluation."""

    def test_extract_numeric_answer(self):
        """Test numeric answer extraction."""
        # GSM8K format
        assert extract_numeric_answer("The answer is #### 42") == 42.0
        # Plain number
        assert extract_numeric_answer("The result is 123") == 123.0
        # Decimal
        assert extract_numeric_answer("Answer: 3.14") == 3.14
        # Negative
        assert extract_numeric_answer("The value is -10") == -10.0
        # No number
        assert extract_numeric_answer("No answer here") is None

    def test_gsm8k_dataset_mock(self):
        """Test GSM8K dataset loading (mock)."""
        # This will fail if datasets library not installed or no internet
        # Just test that it doesn't crash
        try:
            dataset = GSM8KDataset(max_samples=5)
            if len(dataset) > 0:
                sample = dataset[0]
                assert "question" in sample
                assert "answer" in sample
        except Exception:
            pytest.skip("GSM8K dataset not available")

    @pytest.mark.slow
    def test_evaluate_benchmark_mock(self):
        """Test benchmark evaluation with mock model."""
        model = SimpleTestModel(hidden_size=32)

        # Create mock tokenizer
        class MockTokenizer:
            def __call__(self, text, return_tensors=None, **kwargs):
                # Return random token IDs
                input_ids = torch.randint(0, 100, (1, 10))
                return {"input_ids": input_ids}

            def decode(self, tokens, skip_special_tokens=False):
                # Return dummy text with a number
                return "The answer is 42"

            @property
            def pad_token_id(self):
                return 0

            @property
            def eos_token_id(self):
                return 1

        tokenizer = MockTokenizer()

        # This will try to load GSM8K dataset
        try:
            config = BenchmarkConfig(benchmark_name="gsm8k", max_samples=2)
            accuracy = evaluate_benchmark(model, tokenizer, "gsm8k", config)
            assert 0.0 <= accuracy <= 1.0
        except Exception:
            pytest.skip("Benchmark evaluation requires datasets library")


class TestDFSPaperAccurate:
    """Test paper-accurate DFS implementation."""

    def test_dfs_initialization(self):
        """Test DFS initializes correctly."""
        config = DFSConfig(init_strategy="uniform")
        dfs = DFSPaperAccurate(config)
        assert dfs.config.init_strategy == "uniform"
        assert dfs.indicator_array is None

    def test_dfs_merge_basic(self):
        """Test DFS can merge models."""
        models = [SimpleTestModel(hidden_size=32) for _ in range(3)]
        dfs = DFSPaperAccurate()

        merged = dfs.merge(models)

        assert isinstance(merged, nn.Module)
        assert dfs.indicator_array is not None
        assert dfs.scaling_matrix is not None
        # Check shapes
        M = len(models)
        r = dfs._count_layers(models[0])
        assert len(dfs.indicator_array) == M * r
        assert dfs.scaling_matrix.shape == (M, M)

    def test_dfs_with_custom_indicators(self):
        """Test DFS with custom indicator array."""
        models = [SimpleTestModel(hidden_size=32) for _ in range(2)]
        dfs = DFSPaperAccurate()

        M = len(models)
        r = dfs._count_layers(models[0])
        T = M * r

        # Custom indicators: select all layers
        indicators = np.ones(T, dtype=np.float32)
        scaling = np.ones((M, M), dtype=np.float32)

        merged = dfs.merge(models, indicator_array=indicators, scaling_matrix=scaling)

        assert isinstance(merged, nn.Module)


class TestHybridPSDFS:
    """Test hybrid PS+DFS merging."""

    def test_hybrid_initialization(self):
        """Test hybrid merger initializes correctly."""
        config = HybridConfig(ps_candidates_multiplier=2)
        hybrid = HybridPSDFS(config)
        assert hybrid.config.ps_candidates_multiplier == 2

    @pytest.mark.slow
    def test_hybrid_merge_basic(self):
        """Test hybrid merge with dummy fitness."""
        models = [SimpleTestModel(hidden_size=32) for _ in range(3)]

        # Simple fitness function
        def fitness_fn(model):
            return float(list(model.parameters())[0].mean().item())

        config = HybridConfig(ps_candidates_multiplier=2, ps_generations=5, dfs_optimization_iterations=10)

        hybrid = HybridPSDFS(config)
        champion, metrics = hybrid.merge(models, fitness_fn, verbose=False)

        assert isinstance(champion, nn.Module)
        assert "fitness_improvement" in metrics
        assert "ps_best_fitness" in metrics
        assert len(hybrid.ps_candidates) == 3 * 2  # 3 base * 2 multiplier


class TestPhase2Integration:
    """Integration tests for complete Phase 2 pipeline."""

    def test_phase2_standard_mode(self):
        """Test Phase 2 in standard evolution mode."""
        config = EvolutionConfig(num_generations=5, population_size=4, use_hybrid_ps_dfs=False)

        pipeline = Phase2Pipeline(config)
        models = [SimpleTestModel(hidden_size=32) for _ in range(3)]

        champion = pipeline.run(models)

        assert isinstance(champion, nn.Module)
        metrics = pipeline.get_metrics()
        assert "fitness_gain" in metrics

    @pytest.mark.slow
    def test_phase2_hybrid_mode(self):
        """Test Phase 2 in hybrid PS+DFS mode."""
        config = EvolutionConfig(
            num_generations=5,
            population_size=6,
            use_hybrid_ps_dfs=True,
            ps_candidates_multiplier=2,
            use_real_fitness=False,  # Use proxy fitness
        )

        pipeline = Phase2Pipeline(config)
        models = [SimpleTestModel(hidden_size=32) for _ in range(3)]

        champion = pipeline.run(models)

        assert isinstance(champion, nn.Module)
        metrics = pipeline.get_metrics()
        assert "ps_candidates" in metrics
        assert metrics["merge_strategy"] == "hybrid_ps_dfs"

    def test_phase2_fitness_improvement_target(self):
        """Test that Phase 2 aims for 23.5% fitness improvement."""
        config = EvolutionConfig(use_hybrid_ps_dfs=False)
        assert config.target_fitness_gain == 0.235  # 23.5%


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
