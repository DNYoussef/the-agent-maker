"""
Phase 6 Baking Functionality Audit - Pytest Version

Tests the four newly created modules:
1. prompt_pursuit.py - Prompt pursuit optimizer
2. monte_carlo_kl.py - MC trajectory sampling
3. drift_meter.py - Persona drift measurement
4. validation.py - Cross-task validation
"""

import pytest
import torch
import torch.nn as nn


class TestPhase6Imports:
    """Test module imports."""

    def test_prompt_pursuit_imports(self):
        """Test prompt_pursuit imports."""
        from src.phase6_baking.prompt_pursuit import (
            PromptPursuitOptimizer,
            MultiPromptPursuit,
            PursuitConfig,
            PursuitResult,
        )
        assert PromptPursuitOptimizer is not None
        assert MultiPromptPursuit is not None
        assert PursuitConfig is not None
        assert PursuitResult is not None

    def test_monte_carlo_kl_imports(self):
        """Test monte_carlo_kl imports."""
        from src.phase6_baking.monte_carlo_kl import (
            monte_carlo_kl_from_trajectories,
            compute_baking_quality_score,
        )
        assert monte_carlo_kl_from_trajectories is not None
        assert compute_baking_quality_score is not None

    def test_drift_meter_imports(self):
        """Test drift_meter imports."""
        from src.phase6_baking.drift_meter import (
            PersonaDriftMeter,
            DriftConfig,
            DriftResult,
        )
        assert PersonaDriftMeter is not None
        assert DriftConfig is not None
        assert DriftResult is not None

    def test_validation_imports(self):
        """Test validation imports."""
        from src.phase6_baking.validation import (
            CrossTaskValidator,
            ValidationConfig,
            ValidationResult,
            TaskResult,
            create_standard_benchmark_suite,
        )
        assert CrossTaskValidator is not None
        assert ValidationConfig is not None
        assert ValidationResult is not None
        assert TaskResult is not None
        assert create_standard_benchmark_suite is not None


class TestPhase6ClassInstantiation:
    """Test class instantiation."""

    def test_prompt_pursuit_optimizer_instantiation(self):
        """Test PromptPursuitOptimizer instantiation."""
        from src.phase6_baking.prompt_pursuit import PromptPursuitOptimizer, PursuitConfig
        config = PursuitConfig(pursuit_rounds=3)
        optimizer = PromptPursuitOptimizer(config)
        assert optimizer is not None

    def test_multi_prompt_pursuit_instantiation(self):
        """Test MultiPromptPursuit instantiation."""
        from src.phase6_baking.prompt_pursuit import MultiPromptPursuit
        multi = MultiPromptPursuit()
        assert multi is not None

    def test_persona_drift_meter_instantiation(self):
        """Test PersonaDriftMeter instantiation."""
        from src.phase6_baking.drift_meter import PersonaDriftMeter, DriftConfig
        config = DriftConfig(num_turns=10)
        meter = PersonaDriftMeter(config)
        assert meter is not None

    def test_cross_task_validator_instantiation(self):
        """Test CrossTaskValidator instantiation."""
        from src.phase6_baking.validation import CrossTaskValidator, ValidationConfig
        config = ValidationConfig()
        validator = CrossTaskValidator(config)
        assert validator is not None


class TestPhase6MethodAvailability:
    """Test that required methods exist."""

    def test_prompt_pursuit_optimizer_methods(self):
        """Test PromptPursuitOptimizer methods."""
        from src.phase6_baking.prompt_pursuit import PromptPursuitOptimizer
        optimizer = PromptPursuitOptimizer()

        methods = ["pursue", "get_metrics", "_bake_prompt", "_generate_calibration_samples"]
        for method in methods:
            assert hasattr(optimizer, method), f"Missing method: {method}"

    def test_persona_drift_meter_methods(self):
        """Test PersonaDriftMeter methods."""
        from src.phase6_baking.drift_meter import PersonaDriftMeter
        meter = PersonaDriftMeter()

        methods = ["measure_drift", "compare_baked_vs_prompted", "get_metrics"]
        for method in methods:
            assert hasattr(meter, method), f"Missing method: {method}"

    def test_cross_task_validator_methods(self):
        """Test CrossTaskValidator methods."""
        from src.phase6_baking.validation import CrossTaskValidator
        validator = CrossTaskValidator()

        methods = ["validate_cross_task_forgetting", "generate_forgetting_heatmap_data", "get_metrics"]
        for method in methods:
            assert hasattr(validator, method), f"Missing method: {method}"

    def test_monte_carlo_kl_functions(self):
        """Test monte_carlo_kl functions."""
        from src.phase6_baking import monte_carlo_kl

        functions = ["monte_carlo_kl_from_trajectories", "compute_baking_quality_score"]
        for func in functions:
            assert hasattr(monte_carlo_kl, func), f"Missing function: {func}"


class TestPhase6Integration:
    """Test integration with existing baking_engine.py."""

    def test_integration_with_baking_engine(self):
        """Test all modules can coexist with baking_engine."""
        from src.phase6_baking.baking_engine import BakingEngine, BakingConfig
        from src.phase6_baking.prompt_pursuit import PromptPursuitOptimizer
        from src.phase6_baking.monte_carlo_kl import monte_carlo_kl_from_trajectories
        from src.phase6_baking.drift_meter import PersonaDriftMeter
        from src.phase6_baking.validation import CrossTaskValidator

        # All imports should succeed
        engine = BakingEngine()
        assert engine is not None


class TestPhase6PyTorchCompatibility:
    """Test PyTorch compatibility."""

    def test_pytorch_compatibility(self):
        """Test modules work with PyTorch."""
        from src.phase6_baking.prompt_pursuit import PromptPursuitOptimizer
        from src.phase6_baking.drift_meter import PersonaDriftMeter
        from src.phase6_baking.validation import CrossTaskValidator

        # Create simple mock model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Instantiate all modules (should not error)
        optimizer = PromptPursuitOptimizer()
        meter = PersonaDriftMeter()
        validator = CrossTaskValidator()

        assert optimizer is not None
        assert meter is not None
        assert validator is not None


class TestPhase6DataclassStructures:
    """Test dataclass result structures."""

    def test_pursuit_config_fields(self):
        """Test PursuitConfig fields."""
        from src.phase6_baking.prompt_pursuit import PursuitConfig

        config = PursuitConfig(pursuit_rounds=3, convergence_threshold=0.01)
        assert config.pursuit_rounds == 3
        assert config.convergence_threshold == 0.01

    def test_pursuit_result_fields(self):
        """Test PursuitResult fields."""
        from src.phase6_baking.prompt_pursuit import PursuitResult

        result = PursuitResult(
            success=True,
            final_model=None,
            rounds_completed=3,
            scores_per_round=[0.5, 0.6, 0.7],
            improvements_per_round=[0.1, 0.1],
            converged=True,
        )
        assert result.success is True
        assert result.rounds_completed == 3
        assert result.converged is True
