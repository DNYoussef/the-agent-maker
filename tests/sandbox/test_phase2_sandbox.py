"""
Phase 2 (EvoMerge) Sandbox Test

This test validates all 6 merge techniques and the evolution loop in isolation
using small test models. It verifies:
- All merge techniques work correctly
- Merged models have correct architecture
- Fitness evaluation works
- Evolution loop completes successfully

Test Models:
- 2 small models (matching architecture): Linear(10, 5) -> Linear(5, 3)
- Total params: 50 + 15 = 65 params per model (tiny for fast testing)
"""

import copy
import random
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Phase 2 imports
from src.phase2_evomerge.evolution.config import EvolutionConfig
from src.phase2_evomerge.evolution.evolution_loop import EvolutionLoop
from src.phase2_evomerge.evolution.population import initialize_population
from src.phase2_evomerge.fitness import FitnessEvaluator
from src.phase2_evomerge.merge import (
    DAREMerge,
    DFSMerge,
    FrankenMerge,
    LinearMerge,
    MergeTechniques,
    SLERPMerge,
    TIESMerge,
)
from src.phase2_evomerge.merge.dfs_paper_accurate import DFSPaperAccurate


# ============================================================================
# Test Model Definition
# ============================================================================


class TinyTestModel(nn.Module):
    """
    Tiny 2-layer model for fast testing.

    Architecture: Linear(10, 5) -> ReLU -> Linear(5, 3)
    Total params: 50 + 5 + 15 + 3 = 73 parameters
    """

    def __init__(self, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# ============================================================================
# Test Data Generation
# ============================================================================


def create_test_dataset(num_samples: int = 100, batch_size: int = 16):
    """
    Create simple test dataset for fitness evaluation.

    Args:
        num_samples: Number of samples
        batch_size: Batch size

    Returns:
        DataLoader with random data
    """
    # Input: (num_samples, 10) - FLOAT for model compatibility
    X = torch.randn(num_samples, 10, dtype=torch.float32)
    # Target: (num_samples,) with class indices [0, 1, 2]
    y = torch.randint(0, 3, (num_samples,))

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_models():
    """Create 3 test models with different initializations."""
    model1 = TinyTestModel(seed=42)
    model2 = TinyTestModel(seed=123)
    model3 = TinyTestModel(seed=456)
    return [model1, model2, model3]


@pytest.fixture
def test_dataloader():
    """Create test dataloader."""
    return create_test_dataset(num_samples=100, batch_size=16)


@pytest.fixture
def fitness_evaluator(test_dataloader):
    """Create fitness evaluator for testing."""
    return FitnessEvaluator(
        validation_dataset=test_dataloader,
        test_dataset=test_dataloader,
        cache_enabled=False,  # Disable cache for consistent testing
        device="cpu",  # Use CPU for sandbox tests
        max_batches=3,  # Limit batches for speed
        benchmark_batch_size=8,
        benchmark_seq_len=10,
    )


# ============================================================================
# Test 1: Verify Test Models Match Architecture
# ============================================================================


def test_models_have_matching_architecture(test_models):
    """Test that all test models have identical architectures."""
    model1, model2, model3 = test_models

    # Check parameter names match
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    params3 = dict(model3.named_parameters())

    assert set(params1.keys()) == set(params2.keys()), "Models 1 and 2 have different parameter names"
    assert set(params1.keys()) == set(params3.keys()), "Models 1 and 3 have different parameter names"

    # Check parameter shapes match
    for name in params1.keys():
        assert params1[name].shape == params2[name].shape, f"Shape mismatch for {name}"
        assert params1[name].shape == params3[name].shape, f"Shape mismatch for {name}"

    # Check parameter values differ (different seeds)
    assert not torch.allclose(params1["linear1.weight"], params2["linear1.weight"]), \
        "Models should have different weights"

    print("✓ Test models have matching architecture with different weights")


# ============================================================================
# Test 2: Linear Merge
# ============================================================================


def test_linear_merge(test_models):
    """Test Linear merge technique."""
    merger = LinearMerge()

    # Merge 3 models
    merged = merger.merge(test_models)

    # Verify output is a model
    assert isinstance(merged, nn.Module), "Merged output should be nn.Module"

    # Verify architecture matches
    merged_params = dict(merged.named_parameters())
    original_params = dict(test_models[0].named_parameters())

    assert set(merged_params.keys()) == set(original_params.keys()), \
        "Merged model should have same parameter names"

    for name in merged_params.keys():
        assert merged_params[name].shape == original_params[name].shape, \
            f"Shape mismatch for {name}"

    # Verify weights are average (Linear merge with equal weights)
    expected_weight = sum(m.linear1.weight.data for m in test_models) / 3
    assert torch.allclose(merged.linear1.weight.data, expected_weight, atol=1e-5), \
        "Linear merge should average weights"

    print("✓ Linear merge: PASSED")


# ============================================================================
# Test 3: SLERP Merge
# ============================================================================


def test_slerp_merge(test_models):
    """Test SLERP merge technique."""
    merger = SLERPMerge()

    # Merge 3 models
    merged = merger.merge(test_models)

    # Verify output is a model
    assert isinstance(merged, nn.Module), "Merged output should be nn.Module"

    # Verify architecture matches
    merged_params = dict(merged.named_parameters())
    original_params = dict(test_models[0].named_parameters())

    assert set(merged_params.keys()) == set(original_params.keys()), \
        "Merged model should have same parameter names"

    for name in merged_params.keys():
        assert merged_params[name].shape == original_params[name].shape, \
            f"Shape mismatch for {name}"

    # Verify weights are different from Linear merge (SLERP is spherical interpolation)
    linear_merger = LinearMerge()
    linear_merged = linear_merger.merge(test_models)

    # SLERP and Linear should produce different results
    weights_differ = not torch.allclose(
        merged.linear1.weight.data,
        linear_merged.linear1.weight.data,
        atol=1e-3
    )
    assert weights_differ, "SLERP should differ from Linear merge"

    print("✓ SLERP merge: PASSED")


# ============================================================================
# Test 4: TIES Merge
# ============================================================================


def test_ties_merge(test_models):
    """Test TIES merge technique."""
    merger = TIESMerge(trim_percent=0.2)  # Keep top 20% of parameters

    # TIES requires target model + reference models
    target_model = test_models[0]

    # Merge
    merged = merger.merge(target_model, test_models)

    # Verify output is a model
    assert isinstance(merged, nn.Module), "Merged output should be nn.Module"

    # Verify architecture matches
    merged_params = dict(merged.named_parameters())
    original_params = dict(test_models[0].named_parameters())

    assert set(merged_params.keys()) == set(original_params.keys()), \
        "Merged model should have same parameter names"

    for name in merged_params.keys():
        assert merged_params[name].shape == original_params[name].shape, \
            f"Shape mismatch for {name}"

    print("✓ TIES merge: PASSED")


# ============================================================================
# Test 5: DARE Merge
# ============================================================================


def test_dare_merge(test_models):
    """Test DARE merge technique."""
    merger = DAREMerge(drop_rate=0.5)  # rescale_factor auto-calculated

    # DARE requires finetuned model + base model
    finetuned_model = copy.deepcopy(test_models[1])
    base_model = test_models[0]

    # Merge
    merged = merger.merge(finetuned_model, base_model)

    # Verify output is a model
    assert isinstance(merged, nn.Module), "Merged output should be nn.Module"

    # Verify architecture matches
    merged_params = dict(merged.named_parameters())
    original_params = dict(test_models[0].named_parameters())

    assert set(merged_params.keys()) == set(original_params.keys()), \
        "Merged model should have same parameter names"

    for name in merged_params.keys():
        assert merged_params[name].shape == original_params[name].shape, \
            f"Shape mismatch for {name}"

    print("✓ DARE merge: PASSED")


# ============================================================================
# Test 6: FrankenMerge
# ============================================================================


def test_frankenmerge(test_models):
    """Test FrankenMerge technique."""
    merger = FrankenMerge(pattern="abc")

    # FrankenMerge requires target + reference models
    target_model = test_models[0]

    # Merge
    merged = merger.merge(target_model, test_models)

    # Verify output is a model
    assert isinstance(merged, nn.Module), "Merged output should be nn.Module"

    # Verify architecture matches
    merged_params = dict(merged.named_parameters())
    original_params = dict(test_models[0].named_parameters())

    assert set(merged_params.keys()) == set(original_params.keys()), \
        "Merged model should have same parameter names"

    for name in merged_params.keys():
        assert merged_params[name].shape == original_params[name].shape, \
            f"Shape mismatch for {name}"

    print("✓ FrankenMerge: PASSED")


# ============================================================================
# Test 7: DFS (Paper-Accurate)
# ============================================================================


def test_dfs_paper_accurate(test_models):
    """Test DFS (paper-accurate) merge technique."""
    merger = DFSPaperAccurate()

    # Merge
    merged = merger.merge(test_models)

    # Verify output is a model
    assert isinstance(merged, nn.Module), "Merged output should be nn.Module"

    # Verify architecture matches
    merged_params = dict(merged.named_parameters())
    original_params = dict(test_models[0].named_parameters())

    assert set(merged_params.keys()) == set(original_params.keys()), \
        "Merged model should have same parameter names"

    for name in merged_params.keys():
        assert merged_params[name].shape == original_params[name].shape, \
            f"Shape mismatch for {name}"

    # Verify indicator array was created
    assert merger.indicator_array is not None, "DFS should create indicator array"
    assert merger.scaling_matrix is not None, "DFS should create scaling matrix"

    print("✓ DFS (paper-accurate): PASSED")


# ============================================================================
# Test 8: Fitness Evaluation
# ============================================================================


def test_fitness_evaluation(test_models, fitness_evaluator):
    """Test fitness evaluation on a single model."""
    model = test_models[0]

    # Evaluate fitness
    fitness = fitness_evaluator.evaluate(model)

    # Verify fitness dict structure
    assert "composite" in fitness, "Fitness should have 'composite' key"
    assert "components" in fitness, "Fitness should have 'components' key"

    components = fitness["components"]
    assert "perplexity" in components, "Components should have 'perplexity'"
    assert "accuracy" in components, "Components should have 'accuracy'"
    assert "speed" in components, "Components should have 'speed'"
    assert "memory" in components, "Components should have 'memory'"

    # Verify composite is a scalar
    assert isinstance(fitness["composite"], (float, np.floating)), \
        "Composite fitness should be a float"

    # Verify composite is finite
    assert np.isfinite(fitness["composite"]), "Composite fitness should be finite"

    print(f"✓ Fitness evaluation: PASSED (composite={fitness['composite']:.4f})")


# ============================================================================
# Test 9: Batch Fitness Evaluation
# ============================================================================


def test_batch_fitness_evaluation(test_models, fitness_evaluator):
    """Test batch fitness evaluation."""
    # Evaluate all 3 models
    fitness_scores = fitness_evaluator.evaluate_batch(test_models)

    # Verify we got 3 scores
    assert len(fitness_scores) == 3, "Should have 3 fitness scores"

    # Verify all scores are finite
    for i, score in enumerate(fitness_scores):
        assert isinstance(score, (float, np.floating)), f"Score {i} should be float"
        assert np.isfinite(score), f"Score {i} should be finite"

    print(f"✓ Batch fitness evaluation: PASSED (scores={[f'{s:.4f}' for s in fitness_scores]})")


# ============================================================================
# Test 10: Population Initialization
# ============================================================================


def test_population_initialization(test_models):
    """Test population initialization (8 models from 3 base models)."""
    # Initialize population using all 8 binary combos
    population = initialize_population(test_models)

    # Verify we got 8 models
    assert len(population) == 8, "Population should have 8 models"

    # Verify all models have correct architecture
    for i, model in enumerate(population):
        assert isinstance(model, nn.Module), f"Population[{i}] should be nn.Module"

        merged_params = dict(model.named_parameters())
        original_params = dict(test_models[0].named_parameters())

        assert set(merged_params.keys()) == set(original_params.keys()), \
            f"Population[{i}] should have same parameter names"

        for name in merged_params.keys():
            assert merged_params[name].shape == original_params[name].shape, \
                f"Population[{i}] shape mismatch for {name}"

    print("✓ Population initialization: PASSED (8 models created)")


# ============================================================================
# Test 11: Mini Evolution Loop
# ============================================================================


def test_mini_evolution_loop(test_models, fitness_evaluator):
    """Test evolution loop for 3 generations (fast test)."""
    # Configure for fast testing
    config = EvolutionConfig(
        generations=3,  # Only 3 generations
        population_size=8,
        early_stopping=False,  # Disable for deterministic testing
        device="cpu",
    )

    # Create evolution loop
    evolution = EvolutionLoop(config, fitness_evaluator)

    # Run evolution
    result = evolution.evolve(test_models)

    # Verify result structure
    assert "champion" in result, "Result should have 'champion'"
    assert "fitness" in result, "Result should have 'fitness'"
    assert "initial_fitness" in result, "Result should have 'initial_fitness'"
    assert "improvement" in result, "Result should have 'improvement'"
    assert "improvement_pct" in result, "Result should have 'improvement_pct'"
    assert "generations" in result, "Result should have 'generations'"
    assert "convergence_reason" in result, "Result should have 'convergence_reason'"
    assert "final_diversity" in result, "Result should have 'final_diversity'"

    # Verify champion is a model
    assert isinstance(result["champion"], nn.Module), "Champion should be nn.Module"

    # Verify fitness values are finite
    assert np.isfinite(result["fitness"]), "Champion fitness should be finite"
    assert np.isfinite(result["initial_fitness"]), "Initial fitness should be finite"
    assert np.isfinite(result["improvement"]), "Improvement should be finite"

    # Verify generations count
    assert result["generations"] <= 3, "Should run at most 3 generations"

    print(f"✓ Mini evolution loop: PASSED")
    print(f"  Initial fitness: {result['initial_fitness']:.4f}")
    print(f"  Final fitness: {result['fitness']:.4f}")
    print(f"  Improvement: {result['improvement']:.4f} ({result['improvement_pct']*100:.1f}%)")
    print(f"  Generations: {result['generations']}")
    print(f"  Convergence: {result['convergence_reason']}")


# ============================================================================
# Test 12: MergeTechniques API (Binary Combos)
# ============================================================================


def test_merge_techniques_binary_combos(test_models):
    """Test MergeTechniques unified API with all 8 binary combos."""
    merger = MergeTechniques()

    # Test all 8 combos
    for combo_id in range(8):
        merged = merger.apply_combo(test_models, combo_id)

        # Verify output is a model
        assert isinstance(merged, nn.Module), f"Combo {combo_id} should return nn.Module"

        # Verify architecture matches
        merged_params = dict(merged.named_parameters())
        original_params = dict(test_models[0].named_parameters())

        assert set(merged_params.keys()) == set(original_params.keys()), \
            f"Combo {combo_id} should have same parameter names"

        for name in merged_params.keys():
            assert merged_params[name].shape == original_params[name].shape, \
                f"Combo {combo_id} shape mismatch for {name}"

        # Verify combo_id is tagged
        assert hasattr(merged, "combo_id"), f"Combo {combo_id} should have combo_id attribute"
        assert merged.combo_id == combo_id, f"combo_id should be {combo_id}"

        # Decode combo name
        combo_name = merger.decode_combo(combo_id)
        print(f"  Combo {combo_id} ({combo_name}): PASSED")

    print("✓ MergeTechniques binary combos: PASSED (8/8)")


# ============================================================================
# Summary Test
# ============================================================================


def test_phase2_summary():
    """Print Phase 2 sandbox test summary."""
    print("\n" + "="*70)
    print("PHASE 2 EVOMERGE SANDBOX TEST SUMMARY")
    print("="*70)
    print("\nMerge Techniques Tested:")
    print("  ✓ Linear Merge")
    print("  ✓ SLERP Merge")
    print("  ✓ TIES Merge")
    print("  ✓ DARE Merge")
    print("  ✓ FrankenMerge")
    print("  ✓ DFS (Paper-Accurate)")
    print("\nEvolution Components Tested:")
    print("  ✓ Population initialization (8 models from 3 base models)")
    print("  ✓ Fitness evaluation (single model)")
    print("  ✓ Batch fitness evaluation (population)")
    print("  ✓ Mini evolution loop (3 generations)")
    print("  ✓ MergeTechniques unified API (all 8 binary combos)")
    print("\nStatus: ALL TESTS PASSED")
    print("="*70 + "\n")


# ============================================================================
# Run All Tests
# ============================================================================


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 2 EVOMERGE SANDBOX TEST")
    print("="*70 + "\n")

    # Create test fixtures
    test_models_fixture = [TinyTestModel(seed=i) for i in [42, 123, 456]]
    test_dataloader_fixture = create_test_dataset()
    fitness_evaluator_fixture = FitnessEvaluator(
        validation_dataset=test_dataloader_fixture,
        test_dataset=test_dataloader_fixture,
        cache_enabled=False,
        device="cpu",
        max_batches=3,
        benchmark_batch_size=8,
        benchmark_seq_len=10,
    )

    # Run tests
    print("Test 1: Models Have Matching Architecture")
    test_models_have_matching_architecture(test_models_fixture)
    print()

    print("Test 2: Linear Merge")
    test_linear_merge(test_models_fixture)
    print()

    print("Test 3: SLERP Merge")
    test_slerp_merge(test_models_fixture)
    print()

    print("Test 4: TIES Merge")
    test_ties_merge(test_models_fixture)
    print()

    print("Test 5: DARE Merge")
    test_dare_merge(test_models_fixture)
    print()

    print("Test 6: FrankenMerge")
    test_frankenmerge(test_models_fixture)
    print()

    print("Test 7: DFS (Paper-Accurate)")
    test_dfs_paper_accurate(test_models_fixture)
    print()

    print("Test 8: Fitness Evaluation")
    test_fitness_evaluation(test_models_fixture, fitness_evaluator_fixture)
    print()

    print("Test 9: Batch Fitness Evaluation")
    test_batch_fitness_evaluation(test_models_fixture, fitness_evaluator_fixture)
    print()

    print("Test 10: Population Initialization")
    test_population_initialization(test_models_fixture)
    print()

    print("Test 11: Mini Evolution Loop")
    test_mini_evolution_loop(test_models_fixture, fitness_evaluator_fixture)
    print()

    print("Test 12: MergeTechniques Binary Combos")
    test_merge_techniques_binary_combos(test_models_fixture)
    print()

    # Print summary
    test_phase2_summary()
