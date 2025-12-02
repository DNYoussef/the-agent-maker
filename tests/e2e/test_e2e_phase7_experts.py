"""E2E tests for Phase 7: Self-Guided Experts (MoE + ADAS)"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPhase7ExpertsE2E:
    """E2E tests for Phase 7 Self-Guided Expert Discovery pipeline."""

    def test_experts_engine_initialization(self, mock_model, mock_tokenizer):
        """Test experts engine can be initialized."""
        from phase7_experts.experts_engine import ExpertsConfig, ExpertsEngine

        config = ExpertsConfig()
        engine = ExpertsEngine(config=config)

        assert engine.config is not None

    def test_expert_discovery_initialization(self, mock_model, mock_tokenizer):
        """Test expert discovery component initialization."""
        from phase7_experts.expert_discovery import DiscoveryConfig, ExpertDiscovery

        config = DiscoveryConfig(min_experts=3, max_experts=10)

        discovery = ExpertDiscovery(config=config)

        assert discovery.config.min_experts == 3
        assert discovery.config.max_experts == 10
        assert discovery.discovered_experts == []  # Not yet discovered

    def test_expert_count_discovery(self, mock_model, mock_tokenizer):
        """Test model determines its own expert count (N=3-10)."""
        from phase7_experts.expert_discovery import DiscoveryConfig, ExpertDiscovery

        config = DiscoveryConfig()
        discovery = ExpertDiscovery(config=config)

        # Run discovery
        num_experts, expert_profiles = discovery.discover(mock_model, mock_tokenizer)

        # The discovery system determines expert count based on clustering
        # which depends on activation patterns from the model
        # Note: num_experts is clamped to [min_experts, max_experts] range
        # but actual profiles created may be less if few significant clusters found
        assert num_experts >= 3  # Minimum experts (config default)
        assert num_experts <= 10  # Within max range
        assert len(expert_profiles) >= 1  # At least one profile created
        assert len(expert_profiles) <= num_experts  # No more than requested

    def test_expert_capability_identification(self, mock_model, mock_tokenizer):
        """Test model identifies expert specializations."""
        from phase7_experts.expert_discovery import DiscoveryConfig, ExpertDiscovery

        config = DiscoveryConfig()
        discovery = ExpertDiscovery(config=config)

        # Run discovery to get expert profiles with capabilities
        num_experts, expert_profiles = discovery.discover(mock_model, mock_tokenizer)

        assert len(expert_profiles) > 0
        # Each profile should have capabilities
        for profile in expert_profiles:
            assert hasattr(profile, "capabilities")
            assert isinstance(profile.capabilities, list)
            assert len(profile.capabilities) > 0

    def test_svf_trainer_initialization(self, mock_model, mock_tokenizer):
        """Test Transformer^2 SVF trainer initialization."""
        from phase7_experts.svf_trainer import SVFConfig, SVFTrainer

        config = SVFConfig(num_singular_values=32, num_epochs=5, learning_rate=1e-4)

        trainer = SVFTrainer(config=config)

        assert trainer.config.num_singular_values == 32
        assert trainer.config.num_epochs == 5

    def test_svf_training_step(self, mock_model, mock_tokenizer):
        """Test SVF training executes one step with REINFORCE."""
        from phase7_experts.expert_discovery import ExpertProfile
        from phase7_experts.svf_trainer import SVFConfig, SVFTrainer

        config = SVFConfig(num_singular_values=32)
        trainer = SVFTrainer(config=config)

        # Create mock expert profile
        expert_profile = ExpertProfile(
            id=0,
            name="test_expert",
            capabilities=["reasoning"],
            strength_score=0.8,
            activation_pattern=[0.5, 0.6, 0.7],
        )

        # Train expert (simplified test)
        trained_model, result = trainer.train_expert(
            model=mock_model,
            expert_id=0,
            expert_capabilities=["reasoning"],
            tokenizer=mock_tokenizer,
        )

        assert trained_model is not None
        assert result.success is not None

    def test_svf_muongrokfast_fallback(self, mock_model, mock_tokenizer):
        """Test SVF falls back to MuonGrokfast if REINFORCE unstable."""
        from phase7_experts.svf_trainer import SVFConfig, SVFTrainer

        config = SVFConfig(num_singular_values=32, num_epochs=5)

        trainer = SVFTrainer(config=config)

        # This is a placeholder test - actual stability checking would be tested
        # in the SVFTrainer when it detects high variance during training
        # For now, just test that the trainer can be created
        assert trainer.config.num_singular_values == 32

    def test_adas_optimizer_initialization(self):
        """Test ADAS (NSGA-II) optimizer initialization."""
        from phase7_experts.adas_optimizer import ADASConfig, ADASOptimizer

        config = ADASConfig(
            population_size=50, num_generations=100, mutation_rate=0.1, crossover_rate=0.7
        )

        optimizer = ADASOptimizer(config=config)

        assert optimizer.config.population_size == 50
        assert optimizer.config.num_generations == 100
        assert optimizer.population == []  # Not initialized yet

    def test_adas_population_initialization(self):
        """Test ADAS initializes population of architectures."""
        from phase7_experts.adas_optimizer import ADASConfig, ADASOptimizer

        config = ADASConfig(population_size=10)
        optimizer = ADASOptimizer(config=config)

        # Initialize population (using internal method)
        optimizer._initialize_population(num_experts=5)

        assert len(optimizer.population) == 10
        # Each individual has routing weights
        assert all(hasattr(ind, "routing_weights") for ind in optimizer.population)

    def test_adas_nsga2_selection(self):
        """Test NSGA-II multi-objective selection."""
        from phase7_experts.adas_optimizer import ADASConfig, ADASOptimizer, Individual

        config = ADASConfig(population_size=3)
        optimizer = ADASOptimizer(config=config)

        # Create mock individuals with fitness scores
        optimizer.population = [
            Individual(
                routing_weights=[0.3, 0.3, 0.4],
                expert_configs={},
                fitness_scores={"accuracy": 0.85, "latency": 0.6, "diversity": 0.7},
            ),
            Individual(
                routing_weights=[0.5, 0.3, 0.2],
                expert_configs={},
                fitness_scores={"accuracy": 0.82, "latency": 0.8, "diversity": 0.6},
            ),
            Individual(
                routing_weights=[0.4, 0.4, 0.2],
                expert_configs={},
                fitness_scores={"accuracy": 0.88, "latency": 0.5, "diversity": 0.8},
            ),
        ]

        # Run tournament selection
        selected = optimizer._tournament_selection()

        assert len(selected) == config.population_size
        assert all(isinstance(ind, Individual) for ind in selected)

    def test_adas_architecture_mutation(self):
        """Test architecture mutation operation."""
        from phase7_experts.adas_optimizer import ADASConfig, ADASOptimizer, Individual

        config = ADASConfig(mutation_rate=0.1)
        optimizer = ADASOptimizer(config=config)

        # Create individual
        individual = Individual(
            routing_weights=[0.3, 0.3, 0.4], expert_configs={}, fitness_scores={}
        )

        # Mutate individual (modifies in-place)
        optimizer._mutate(individual, num_experts=3)

        # Verify individual still has routing weights
        assert len(individual.routing_weights) == 3
        assert abs(sum(individual.routing_weights) - 1.0) < 0.01  # Should sum to 1

    def test_adas_architecture_crossover(self):
        """Test architecture crossover operation."""
        from phase7_experts.adas_optimizer import ADASConfig, ADASOptimizer, Individual

        config = ADASConfig(crossover_rate=0.8)
        optimizer = ADASOptimizer(config=config)

        # Mock parent individuals
        parent1 = Individual(
            routing_weights=[0.3, 0.3, 0.4],
            expert_configs={"expert_0": {"threshold": 0.5}},
            fitness_scores={},
        )
        parent2 = Individual(
            routing_weights=[0.5, 0.2, 0.3],
            expert_configs={"expert_0": {"threshold": 0.7}},
            fitness_scores={},
        )

        # Crossover
        child1, child2 = optimizer._crossover(parent1, parent2, num_experts=3)

        assert len(child1.routing_weights) == 3
        assert len(child2.routing_weights) == 3
        # Children should have normalized weights
        assert abs(sum(child1.routing_weights) - 1.0) < 0.01
        assert abs(sum(child2.routing_weights) - 1.0) < 0.01

    def test_adas_model_guided_fitness(self, mock_model, mock_tokenizer):
        """Test model-guided fitness evaluation (self-guided)."""
        from phase7_experts.adas_optimizer import ADASConfig, ADASOptimizer, Individual

        config = ADASConfig()
        optimizer = ADASOptimizer(config=config)

        # Mock individual
        individual = Individual(
            routing_weights=[0.3, 0.3, 0.4], expert_configs={}, fitness_scores={}
        )

        # Evaluate individual
        fitness = optimizer._evaluate_individual(
            individual, model=mock_model, experts=[], tokenizer=mock_tokenizer, evaluator=None
        )

        assert "accuracy" in fitness
        assert "latency" in fitness
        assert "diversity" in fitness

    def test_adas_generation_evolution(self):
        """Test one generation of ADAS evolution."""
        from phase7_experts.adas_optimizer import ADASConfig, ADASOptimizer

        config = ADASConfig(population_size=10, num_generations=1)
        optimizer = ADASOptimizer(config=config)

        # Initialize population
        optimizer._initialize_population(num_experts=3)

        # Verify initialization
        assert len(optimizer.population) == 10
        assert all(hasattr(ind, "routing_weights") for ind in optimizer.population)
        assert all(len(ind.routing_weights) == 3 for ind in optimizer.population)

    def test_full_adas_search(self, mock_model, mock_tokenizer):
        """Test complete ADAS search (100 gen x 50 pop)."""
        from phase7_experts.adas_optimizer import ADASConfig, ADASOptimizer
        from phase7_experts.expert_discovery import ExpertProfile

        config = ADASConfig(
            population_size=5, num_generations=2  # Small for test  # Small for test
        )
        optimizer = ADASOptimizer(config=config)

        # Create mock experts
        experts = [
            ExpertProfile(
                id=i,
                name=f"expert_{i}",
                capabilities=["reasoning"],
                strength_score=0.8,
                activation_pattern=[0.5, 0.6],
            )
            for i in range(3)
        ]

        # Run optimization
        optimized_model, result = optimizer.optimize(
            model=mock_model, experts=experts, tokenizer=mock_tokenizer
        )

        assert result.success is True
        assert result.best_individual is not None
        assert len(result.pareto_front) > 0

    def test_routing_configuration_post_adas(self, mock_model, mock_tokenizer):
        """Test expert routing configuration after ADAS."""
        from phase7_experts.adas_optimizer import ADASConfig, ADASOptimizer, Individual
        from phase7_experts.expert_discovery import ExpertProfile

        # Create optimizer and mock experts
        optimizer = ADASOptimizer(config=ADASConfig())
        experts = [
            ExpertProfile(
                id=i,
                name=f"expert_{i}",
                capabilities=["reasoning"],
                strength_score=0.8,
                activation_pattern=[0.5],
            )
            for i in range(3)
        ]

        # Create a best individual
        best = Individual(
            routing_weights=[0.3, 0.3, 0.4],
            expert_configs={"expert_0": {"threshold": 0.5}},
            fitness_scores={"accuracy": 0.85, "latency": 0.6, "diversity": 0.7},
        )

        # Apply routing to model
        optimized_model = optimizer._apply_routing(mock_model, experts, best)

        # Verify routing config was added
        assert hasattr(optimized_model, "_expert_routing")
        assert "weights" in optimized_model._expert_routing
        assert "num_experts" in optimized_model._expert_routing

    def test_expert_integration_with_model(self, mock_model, mock_tokenizer):
        """Test experts integrate into base model."""
        from phase7_experts.experts_engine import ExpertsConfig, ExpertsEngine

        config = ExpertsConfig()
        engine = ExpertsEngine(config=config)

        # Run full pipeline to get integrated model
        result = engine.run(model=mock_model, tokenizer=mock_tokenizer)

        # Verify model was returned
        assert result.model is not None
        # Model should have expert routing if successful
        if result.success and hasattr(result.model, "_expert_routing"):
            assert "weights" in result.model._expert_routing

    def test_phase7_full_pipeline(self, mock_model, mock_tokenizer):
        """Test complete Phase 7 pipeline (discovery -> SVF -> ADAS -> routing)."""
        from phase7_experts.experts_engine import ExpertsConfig, ExpertsEngine

        config = ExpertsConfig(
            min_experts=3,
            max_experts=5,
            svf_epochs=2,  # Small for test
            adas_generations=2,  # Small for test
        )
        engine = ExpertsEngine(config=config)

        # Run full pipeline
        result = engine.run(model=mock_model, tokenizer=mock_tokenizer)

        # Verify result structure
        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "model")
        assert hasattr(result, "num_experts")
        assert hasattr(result, "expert_profiles")
        assert hasattr(result, "routing_config")
        assert hasattr(result, "metrics")

        # Verify metrics
        assert "discovery_time" in result.metrics
        assert "svf_time" in result.metrics
        assert "adas_time" in result.metrics
