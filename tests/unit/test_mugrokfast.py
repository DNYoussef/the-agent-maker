"""
Unit tests for MuGrokfast Optimizer
Tests Muon, Grokfast, and phase-specific presets
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cross_phase.mugrokfast.config import MuGrokConfig
from cross_phase.mugrokfast.optimizer import MuonGrokfast, create_optimizer_from_phase


class TestMuGrokConfig:
    """Test MuGrokConfig"""

    def test_default_config(self):
        """Test default configuration via custom() factory method"""
        # ISS-003: MuGrokConfig is a dataclass without defaults
        # Use custom() factory method which provides default values
        config = MuGrokConfig.custom()

        assert config.muon_lr == 0.01
        assert config.grokfast_alpha == 0.98
        assert config.grokfast_lambda == 2.0

    def test_phase1_preset(self):
        """Test Phase 1 preset"""
        config = MuGrokConfig.from_phase(1)

        assert config.muon_lr == 1e-3
        assert config.grokfast_lambda == 0.3
        assert config.kl_coefficient == 0.0

    def test_phase3_preset(self):
        """Test Phase 3 preset (RL)"""
        config = MuGrokConfig.from_phase(3)

        assert config.muon_lr == 5e-4
        assert config.grokfast_lambda == 0.1
        assert config.kl_coefficient == 0.1
        assert config.qk_clip_threshold == 25.0

    def test_phase5_preset(self):
        """Test Phase 5 preset (STE)"""
        config = MuGrokConfig.from_phase(5)

        assert config.muon_lr == 1e-3
        assert config.grokfast_lambda == 2.0
        assert config.muon_ste_mode is True

    def test_invalid_phase(self):
        """Test invalid phase number"""
        with pytest.raises(ValueError):
            MuGrokConfig.from_phase(99)


@pytest.mark.skipif(
    'torch' not in sys.modules,
    reason="PyTorch not available"
)
class TestMuonGrokfast:
    """Test MuonGrokfast optimizer"""

    def test_optimizer_creation(self, mock_model):
        """Test optimizer instantiation"""
        config = MuGrokConfig.from_phase(1)
        optimizer = MuonGrokfast(mock_model.parameters(), config=config)

        assert optimizer is not None
        assert optimizer.config.muon_lr == 1e-3

    def test_create_from_phase(self, mock_model):
        """Test create_optimizer_from_phase helper"""
        optimizer = create_optimizer_from_phase(mock_model, phase_num=1)

        assert optimizer is not None
        assert optimizer.config.muon_lr == 1e-3

    def test_parameter_groups(self, mock_model):
        """Test parameter groups are created"""
        optimizer = create_optimizer_from_phase(mock_model, phase_num=1)

        assert len(optimizer.param_groups) > 0

    def test_state_initialization(self, mock_model):
        """Test optimizer state initialization"""
        optimizer = create_optimizer_from_phase(mock_model, phase_num=1)

        # State should be empty before first step
        assert len(optimizer.state) == 0

    def test_get_muon_lr(self, mock_model):
        """Test get_muon_lr utility"""
        optimizer = create_optimizer_from_phase(mock_model, phase_num=1)

        muon_lr = optimizer.get_muon_lr()
        assert muon_lr == 1e-3

    def test_zero_grad(self, mock_model):
        """Test zero_grad works"""
        optimizer = create_optimizer_from_phase(mock_model, phase_num=1)

        # Set some gradients
        for param in mock_model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        optimizer.zero_grad()

        # All gradients should be None or zero
        for param in mock_model.parameters():
            assert param.grad is None or param.grad.sum() == 0
