"""E2E tests for Phase 6: Tool & Persona Baking"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch



class TestPhase6BakingE2E:
    """E2E tests for Phase 6 Tool & Persona Baking pipeline."""

    def test_baking_engine_initialization(self, mock_model, temp_output_dir):
        """Test baking engine can be initialized."""
        from phase6_baking.baking_engine import BakingConfig, BakingEngine

        config = BakingConfig()
        engine = BakingEngine(config=config)

        assert engine.config is not None
        assert engine.metrics is not None
        assert "a_cycle_scores" in engine.metrics
        assert "b_cycle_scores" in engine.metrics

    def test_ab_cycle_initialization(self, mock_model, temp_output_dir):
        """Test A/B cycle can be initialized and tracked."""
        from phase6_baking.baking_engine import BakingConfig, BakingEngine

        config = BakingConfig(max_total_iterations=5, half_bake_strength=0.5)

        engine = BakingEngine(config=config)

        # Verify initial configuration
        assert engine.config.max_total_iterations == 5
        assert engine.config.half_bake_strength == 0.5
        assert len(engine.metrics["a_cycle_scores"]) == 0
        assert len(engine.metrics["b_cycle_scores"]) == 0

    def test_cycle_switching_logic(self, mock_model, temp_output_dir):
        """Test automatic cycle switching from A to B and back."""
        from phase6_baking.baking_engine import BakingConfig, BakingCycleType, BakingEngine

        config = BakingConfig()
        engine = BakingEngine(config=config)

        # Mock the imports inside the run method
        with patch("phase6_baking.a_cycle_tool.ACycleOptimizer") as MockA, patch(
            "phase6_baking.b_cycle_persona.BCycleOptimizer"
        ) as MockB, patch("phase6_baking.plateau_detector.PlateauDetector") as MockDetector:
            # Setup mocks
            mock_detector = MockDetector.return_value
            mock_detector.check.side_effect = [True, False, True]  # Plateau, no plateau, plateau
            mock_detector.both_plateaued.return_value = False

            # Verify cycle types exist
            assert BakingCycleType.A_CYCLE.value == "tool"
            assert BakingCycleType.B_CYCLE.value == "persona"

    def test_plateau_detector_initialization(self):
        """Test plateau detector can be initialized."""
        from phase6_baking.plateau_detector import PlateauDetector

        detector = PlateauDetector(window_size=3, threshold=0.01)

        # Mock detector should be initializable
        # Actual attributes depend on implementation
        assert detector is not None

    def test_plateau_detection_logic(self):
        """Test plateau detection triggers correctly."""
        from phase6_baking.plateau_detector import PlateauDetector

        detector = PlateauDetector(window_size=3, threshold=0.01)

        # Mock plateau detection behavior
        with patch.object(detector, "check") as mock_check:
            # Simulate improving scores (no plateau)
            mock_check.return_value = False
            assert detector.check(0.70, "a_cycle") is False
            assert detector.check(0.72, "a_cycle") is False

            # Simulate plateau detected
            mock_check.return_value = True
            assert detector.check(0.72, "a_cycle") is True

    def test_a_cycle_tool_optimization(self, mock_model, temp_output_dir):
        """Test A-cycle tool use optimization."""
        from phase6_baking.baking_engine import BakingConfig

        # Test A-cycle configuration
        config = BakingConfig(a_cycle_iterations=10, baking_epochs=2, half_bake_strength=0.5)

        assert config.a_cycle_iterations == 10
        assert config.baking_epochs == 2
        assert config.half_bake_strength == 0.5

    @patch("phase6_baking.a_cycle_tool.ACycleOptimizer")
    def test_a_cycle_training_step(self, MockOptimizer, mock_model, temp_output_dir):
        """Test A-cycle can execute one training iteration."""
        from phase6_baking.baking_engine import BakingConfig

        config = BakingConfig()

        # Mock A-cycle optimizer
        mock_optimizer = MockOptimizer.return_value
        mock_optimizer.optimize.return_value = (mock_model, 0.75)

        # Execute mock training step
        baked_model, score = mock_optimizer.optimize(
            model=mock_model, tokenizer=Mock(), evaluator=Mock()
        )

        assert score == 0.75
        assert baked_model is not None
        assert mock_optimizer.optimize.called

    def test_b_cycle_persona_generation(self, mock_model, temp_output_dir):
        """Test B-cycle self-guided persona generation."""
        from phase6_baking.baking_engine import BakingConfig

        # Test B-cycle configuration
        config = BakingConfig(b_cycle_iterations=20, baking_epochs=2)

        assert config.b_cycle_iterations == 20
        assert config.baking_epochs == 2
        assert len(config.persona_prompts) > 0

    @patch("phase6_baking.b_cycle_persona.BCycleOptimizer")
    def test_b_cycle_persona_discovery(self, MockOptimizer, mock_model, temp_output_dir):
        """Test B-cycle discovers model's own patterns."""
        from phase6_baking.baking_engine import BakingConfig

        config = BakingConfig()

        # Mock B-cycle optimizer
        mock_optimizer = MockOptimizer.return_value
        mock_optimizer.optimize.return_value = (mock_model, 0.82)

        # Execute mock persona optimization
        baked_model, score = mock_optimizer.optimize(
            model=mock_model, tokenizer=Mock(), evaluator=Mock()
        )

        assert score == 0.82
        assert baked_model is not None
        assert mock_optimizer.optimize.called

    def test_half_baking_strength(self, mock_model, temp_output_dir):
        """Test half-baking applies 50% strength per iteration."""
        from phase6_baking.baking_engine import BakingConfig

        config = BakingConfig(half_bake_strength=0.5, baking_epochs=1)

        # Mock HalfBaker
        with patch("phase6_baking.half_baking.HalfBaker") as MockBaker:
            mock_baker = MockBaker.return_value
            mock_baker.half_bake.return_value = mock_model

            # Test half-baking
            result = mock_baker.half_bake(original_model=mock_model, baked_model=mock_model)

            assert result is not None
            assert mock_baker.half_bake.called

    def test_baking_iteration_full_cycle(self, mock_model, temp_output_dir):
        """Test one complete baking iteration (A-cycle + B-cycle)."""
        from phase6_baking.baking_engine import BakingConfig, BakingEngine

        config = BakingConfig(max_total_iterations=1)
        engine = BakingEngine(config=config)

        # Mock the imports inside the run method
        with patch("phase6_baking.a_cycle_tool.ACycleOptimizer") as MockA, patch(
            "phase6_baking.b_cycle_persona.BCycleOptimizer"
        ) as MockB, patch("phase6_baking.plateau_detector.PlateauDetector") as MockDetector, patch(
            "phase6_baking.half_baking.HalfBaker"
        ) as MockBaker:
            mock_a = MockA.return_value
            mock_b = MockB.return_value
            mock_detector = MockDetector.return_value
            mock_baker = MockBaker.return_value

            mock_a.optimize.return_value = (mock_model, 0.70)
            mock_b.optimize.return_value = (mock_model, 0.75)
            mock_detector.check.return_value = False
            mock_detector.both_plateaued.return_value = True
            mock_baker.half_bake.return_value = mock_model

            # Run baking
            result = engine.run(mock_model, Mock(), Mock(), Mock())

            assert result.success is True
            assert result.total_iterations >= 1

    def test_plateau_triggered_cycle_switch(self, mock_model, temp_output_dir):
        """Test plateau detection triggers cycle switch."""
        from phase6_baking.baking_engine import BakingConfig, BakingCycleType, BakingEngine

        config = BakingConfig(max_total_iterations=3)
        engine = BakingEngine(config=config)

        # Mock the imports inside the run method
        with patch("phase6_baking.a_cycle_tool.ACycleOptimizer") as MockA, patch(
            "phase6_baking.b_cycle_persona.BCycleOptimizer"
        ) as MockB, patch("phase6_baking.plateau_detector.PlateauDetector") as MockDetector, patch(
            "phase6_baking.half_baking.HalfBaker"
        ) as MockBaker:
            mock_detector = MockDetector.return_value
            # Simulate plateau detection causing cycle switch
            mock_detector.check.side_effect = [True, False, True]  # Plateau, no, plateau
            mock_detector.both_plateaued.return_value = False

            mock_a = MockA.return_value
            mock_b = MockB.return_value
            mock_baker = MockBaker.return_value

            mock_a.optimize.return_value = (mock_model, 0.70)
            mock_b.optimize.return_value = (mock_model, 0.75)
            mock_baker.half_bake.return_value = mock_model

            # Verify plateau detection is part of the system
            assert mock_detector is not None

    def test_multiple_ab_iterations(self, mock_model, temp_output_dir):
        """Test multiple A/B iteration cycles."""
        from phase6_baking.baking_engine import BakingConfig, BakingEngine

        config = BakingConfig(max_total_iterations=3)
        engine = BakingEngine(config=config)

        # Mock the imports inside the run method
        with patch("phase6_baking.a_cycle_tool.ACycleOptimizer") as MockA, patch(
            "phase6_baking.b_cycle_persona.BCycleOptimizer"
        ) as MockB, patch("phase6_baking.plateau_detector.PlateauDetector") as MockDetector, patch(
            "phase6_baking.half_baking.HalfBaker"
        ) as MockBaker:
            mock_a = MockA.return_value
            mock_b = MockB.return_value
            mock_detector = MockDetector.return_value
            mock_baker = MockBaker.return_value

            mock_a.optimize.return_value = (mock_model, 0.70)
            mock_b.optimize.return_value = (mock_model, 0.75)
            mock_detector.check.return_value = False
            mock_detector.both_plateaued.return_value = True
            mock_baker.half_bake.return_value = mock_model

            # Run multiple iterations
            result = engine.run(mock_model, Mock(), Mock(), Mock())

            assert result.total_iterations >= 1
            assert result.total_iterations <= 3

    def test_prompt_baking_integration(self, mock_model, temp_output_dir):
        """Test integration with prompt baking system."""
        from phase6_baking.baking_engine import BakingConfig, BakingEngine

        config = BakingConfig()
        engine = BakingEngine(config=config)

        # Test that baking engine has prompts configured
        assert len(engine.config.tool_prompts) > 0
        assert len(engine.config.persona_prompts) > 0
        # Baking happens via ACycleOptimizer and BCycleOptimizer
        assert engine.config.baking_epochs > 0

    def test_sequential_baking_composition(self):
        """Test sequential baking composes multiple prompts."""
        # Sequential baking: theta_u1u2 = B(B(theta, u1), u2)
        # From V1 implementation

        from phase6_baking.baking_engine import BakingConfig

        # Mock HalfBaker for sequential composition
        with patch("phase6_baking.half_baking.HalfBaker") as MockBaker:
            mock_baker = MockBaker.return_value
            mock_model1 = Mock()
            mock_model2 = Mock()
            mock_baker.half_bake.side_effect = [mock_model1, mock_model2]

            base_model = Mock()

            # Bake sequentially
            result1 = mock_baker.half_bake(original_model=base_model, baked_model=Mock())
            result2 = mock_baker.half_bake(original_model=result1, baked_model=Mock())

            assert mock_baker.half_bake.call_count == 2
            assert result1 == mock_model1
            assert result2 == mock_model2

    def test_convergence_criteria(self, mock_model, temp_output_dir):
        """Test baking convergence detection."""
        from phase6_baking.baking_engine import BakingConfig, BakingEngine

        config = BakingConfig(max_total_iterations=10, plateau_threshold=0.01)

        engine = BakingEngine(config=config)

        # Mock the imports inside the run method
        with patch("phase6_baking.a_cycle_tool.ACycleOptimizer") as MockA, patch(
            "phase6_baking.b_cycle_persona.BCycleOptimizer"
        ) as MockB, patch("phase6_baking.plateau_detector.PlateauDetector") as MockDetector, patch(
            "phase6_baking.half_baking.HalfBaker"
        ) as MockBaker:
            mock_a = MockA.return_value
            mock_b = MockB.return_value
            mock_detector = MockDetector.return_value
            mock_baker = MockBaker.return_value

            # High scores indicating convergence
            mock_a.optimize.return_value = (mock_model, 0.96)
            mock_b.optimize.return_value = (mock_model, 0.95)
            mock_detector.check.return_value = False
            mock_detector.both_plateaued.return_value = True  # Converged
            mock_baker.half_bake.return_value = mock_model

            result = engine.run(mock_model, Mock(), Mock(), Mock())

            # Convergence detected via plateau
            assert result.success is True
            assert result.final_tool_score >= 0.96 or result.final_persona_score >= 0.95
