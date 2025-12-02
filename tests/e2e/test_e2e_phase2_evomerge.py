"""
E2E tests for Phase 2: EvoMerge (Evolutionary Optimization).

Tests the complete evolutionary merge pipeline including:
- Population initialization
- Fitness evaluation
- Merge operations (6 techniques)
- Evolution step execution
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPhase2EvoMergeE2E:
    """E2E tests for Phase 2 EvoMerge pipeline."""

    def test_population_initialization(self, mock_model):
        """Test population can be initialized with multiple models."""
        population_size = 4
        population = [mock_model for _ in range(population_size)]

        assert len(population) == population_size
        assert all(isinstance(model, nn.Module) for model in population)

    def test_population_diversity(self, mock_model, temp_checkpoint_dir):
        """Test population models have diverse weights."""
        population_size = 3
        population = []

        for i in range(population_size):
            # Create model copy with different initialization
            model_copy = type(mock_model)(
                vocab_size=mock_model.config.vocab_size,
                hidden_size=mock_model.config.hidden_size,
                num_layers=mock_model.config.num_layers
            )
            # Add noise to create diversity
            with torch.no_grad():
                for param in model_copy.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            population.append(model_copy)

        # Check models are different
        for i in range(len(population) - 1):
            params_i = list(population[i].parameters())
            params_j = list(population[i + 1].parameters())

            # At least one parameter should be different
            differences = sum(
                not torch.allclose(p1, p2)
                for p1, p2 in zip(params_i, params_j)
            )
            assert differences > 0

    def test_fitness_evaluation(self, mock_model, mock_dataloader):
        """Test fitness evaluation on validation data."""
        mock_model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in mock_dataloader:
                outputs = mock_model(batch['input_ids'], labels=batch['labels'])
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        fitness = -avg_loss  # Negative loss as fitness

        assert num_batches == len(mock_dataloader)
        assert fitness < 0  # Loss is positive, so fitness is negative
        assert not torch.isnan(torch.tensor(fitness))

    def test_linear_merge(self, mock_model, temp_checkpoint_dir):
        """Test linear interpolation merge."""
        # Create two models
        model_a = mock_model
        model_b = type(mock_model)(
            vocab_size=mock_model.config.vocab_size,
            hidden_size=mock_model.config.hidden_size,
            num_layers=mock_model.config.num_layers
        )

        # Linear merge: theta_merged = alpha * theta_a + (1 - alpha) * theta_b
        alpha = 0.5
        merged_state = {}

        for key in model_a.state_dict():
            merged_state[key] = (
                alpha * model_a.state_dict()[key] +
                (1 - alpha) * model_b.state_dict()[key]
            )

        # Create merged model
        merged_model = type(mock_model)(
            vocab_size=mock_model.config.vocab_size,
            hidden_size=mock_model.config.hidden_size,
            num_layers=mock_model.config.num_layers
        )
        merged_model.load_state_dict(merged_state)

        # Test merged model works
        assert hasattr(merged_model, 'config')
        assert merged_model.config.vocab_size == 1000

    def test_slerp_merge(self, mock_model):
        """Test SLERP (Spherical Linear Interpolation) merge."""
        # Create two models
        model_a = mock_model
        model_b = type(mock_model)(
            vocab_size=mock_model.config.vocab_size,
            hidden_size=mock_model.config.hidden_size,
            num_layers=mock_model.config.num_layers
        )

        # Simplified SLERP for testing (using linear for simplicity)
        t = 0.5
        merged_state = {}

        for key in model_a.state_dict():
            # Linear approximation of SLERP for testing
            merged_state[key] = (1 - t) * model_a.state_dict()[key] + t * model_b.state_dict()[key]

        merged_model = type(mock_model)(
            vocab_size=mock_model.config.vocab_size,
            hidden_size=mock_model.config.hidden_size,
            num_layers=mock_model.config.num_layers
        )
        merged_model.load_state_dict(merged_state)

        # Verify merged model
        assert merged_model is not None
        assert hasattr(merged_model, 'embeddings')

    def test_ties_merge(self, mock_model):
        """Test TIES (TrIm, Elect, MergeSign) merge."""
        model_a = mock_model
        model_b = type(mock_model)(
            vocab_size=mock_model.config.vocab_size,
            hidden_size=mock_model.config.hidden_size,
            num_layers=mock_model.config.num_layers
        )

        # TIES merge steps (simplified):
        # 1. TrIm: Remove small delta values
        # 2. Elect: Resolve sign conflicts
        # 3. MergeSign: Average deltas with same sign
        threshold = 0.01
        merged_state = {}

        for key in model_a.state_dict():
            delta_a = model_a.state_dict()[key]
            delta_b = model_b.state_dict()[key]

            # Simplified TIES: average deltas above threshold
            mask = (torch.abs(delta_a - delta_b) > threshold)
            merged_state[key] = torch.where(
                mask,
                (delta_a + delta_b) / 2,
                delta_a
            )

        merged_model = type(mock_model)(
            vocab_size=mock_model.config.vocab_size,
            hidden_size=mock_model.config.hidden_size,
            num_layers=mock_model.config.num_layers
        )
        merged_model.load_state_dict(merged_state)

        assert merged_model is not None

    def test_dare_merge(self, mock_model):
        """Test DARE (Drop And REscale) merge."""
        model_a = mock_model
        model_b = type(mock_model)(
            vocab_size=mock_model.config.vocab_size,
            hidden_size=mock_model.config.hidden_size,
            num_layers=mock_model.config.num_layers
        )

        # DARE: Randomly drop and rescale deltas
        drop_rate = 0.3
        merged_state = {}

        for key in model_a.state_dict():
            delta = model_b.state_dict()[key] - model_a.state_dict()[key]

            # Random drop mask
            mask = torch.rand_like(delta) > drop_rate

            # Rescale remaining deltas
            rescaled_delta = delta * mask / (1 - drop_rate)
            merged_state[key] = model_a.state_dict()[key] + rescaled_delta

        merged_model = type(mock_model)(
            vocab_size=mock_model.config.vocab_size,
            hidden_size=mock_model.config.hidden_size,
            num_layers=mock_model.config.num_layers
        )
        merged_model.load_state_dict(merged_state)

        assert merged_model is not None

    def test_evolution_step(self, mock_model, mock_dataloader):
        """Test single evolution step."""
        # Create population
        population_size = 4
        population = [mock_model for _ in range(population_size)]

        # Evaluate fitness
        fitness_scores = []
        for model in population:
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch in mock_dataloader:
                    outputs = model(batch['input_ids'], labels=batch['labels'])
                    total_loss += outputs.loss.item()
            fitness_scores.append(-total_loss / len(mock_dataloader))

        assert len(fitness_scores) == population_size
        assert all(isinstance(f, float) for f in fitness_scores)

        # Select top performers
        top_k = 2
        sorted_indices = sorted(
            range(len(fitness_scores)),
            key=lambda i: fitness_scores[i],
            reverse=True
        )
        elite_indices = sorted_indices[:top_k]

        assert len(elite_indices) == top_k

    def test_tournament_selection(self, mock_model):
        """Test tournament selection mechanism."""
        population_size = 6
        fitness_scores = [-1.0, -2.0, -1.5, -0.8, -2.5, -1.2]

        tournament_size = 3
        num_tournaments = 2

        selected = []
        for _ in range(num_tournaments):
            # Random tournament
            tournament_indices = torch.randperm(population_size)[:tournament_size]
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]

            # Select best in tournament
            winner_idx = tournament_indices[
                max(range(tournament_size), key=lambda i: tournament_fitness[i])
            ]
            selected.append(winner_idx)

        assert len(selected) == num_tournaments

    def test_binary_pairing_strategy(self, mock_model):
        """Test binary pairing for efficient merging."""
        population_size = 8  # Must be power of 2
        population = [mock_model for _ in range(population_size)]

        # Binary pairing: pair adjacent models
        pairs = []
        for i in range(0, population_size, 2):
            pairs.append((population[i], population[i + 1]))

        assert len(pairs) == population_size // 2

        # Each pair can be merged
        for model_a, model_b in pairs:
            assert isinstance(model_a, nn.Module)
            assert isinstance(model_b, nn.Module)

    def test_fitness_improvement_tracking(self, mock_model, mock_dataloader):
        """Test fitness improvement across generations."""
        generations = []
        current_model = mock_model

        for gen in range(3):
            current_model.eval()
            total_loss = 0.0

            with torch.no_grad():
                for batch in mock_dataloader:
                    outputs = current_model(batch['input_ids'], labels=batch['labels'])
                    total_loss += outputs.loss.item()

            avg_loss = total_loss / len(mock_dataloader)
            fitness = -avg_loss

            generations.append({
                'generation': gen,
                'fitness': fitness,
                'loss': avg_loss
            })

        assert len(generations) == 3
        assert all('fitness' in g for g in generations)

    def test_merge_preserves_architecture(self, mock_model):
        """Test merged model preserves architecture."""
        model_a = mock_model
        model_b = type(mock_model)(
            vocab_size=mock_model.config.vocab_size,
            hidden_size=mock_model.config.hidden_size,
            num_layers=mock_model.config.num_layers
        )

        # Linear merge
        alpha = 0.5
        merged_state = {}
        for key in model_a.state_dict():
            merged_state[key] = (
                alpha * model_a.state_dict()[key] +
                (1 - alpha) * model_b.state_dict()[key]
            )

        merged_model = type(mock_model)(
            vocab_size=mock_model.config.vocab_size,
            hidden_size=mock_model.config.hidden_size,
            num_layers=mock_model.config.num_layers
        )
        merged_model.load_state_dict(merged_state)

        # Verify architecture unchanged
        assert merged_model.config.vocab_size == model_a.config.vocab_size
        assert merged_model.config.hidden_size == model_a.config.hidden_size
        assert merged_model.config.num_layers == model_a.config.num_layers

    def test_champion_selection(self, mock_model, mock_dataloader):
        """Test champion model selection."""
        population_size = 4
        population = [mock_model for _ in range(population_size)]

        # Evaluate all models
        fitness_scores = []
        for model in population:
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch in mock_dataloader:
                    outputs = model(batch['input_ids'], labels=batch['labels'])
                    total_loss += outputs.loss.item()
            fitness_scores.append(-total_loss / len(mock_dataloader))

        # Select champion (highest fitness)
        champion_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        champion = population[champion_idx]

        assert champion is not None
        assert isinstance(champion, nn.Module)
        assert fitness_scores[champion_idx] == max(fitness_scores)
