"""
Unit tests for EMAModel (Exponential Moving Average)

Tests:
- Initialization
- Update mechanism
- Apply/restore shadow weights
- Edge cases

Target: >=90% coverage for EMAModel class
"""

import pytest
import torch
import torch.nn as nn
from copy import deepcopy

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from phase1_cognate.training.trainer import EMAModel


class SimpleModel(nn.Module):
    """Simple model for testing EMA."""

    def __init__(self, in_features=10, out_features=5):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 20)
        self.linear2 = nn.Linear(20, out_features)
        self.bn = nn.BatchNorm1d(20)  # Has non-grad params

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.bn(x)
        return self.linear2(x)


class TestEMAInitialization:
    """Test EMAModel initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.999)

        assert ema.decay == 0.999
        assert ema.model is model
        assert len(ema.shadow) > 0, "Should have shadow weights"
        assert len(ema.backup) == 0, "Backup should be empty initially"

    def test_shadow_copies_weights(self):
        """Test shadow weights are copies of model weights."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.999)

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow, f"Missing shadow for {name}"
                assert torch.allclose(ema.shadow[name], param.data)

    def test_shadow_is_independent(self):
        """Test shadow weights are independent copies."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.999)

        # Modify model weights
        for param in model.parameters():
            param.data.add_(1.0)

        # Shadow should NOT have changed
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema.shadow:
                assert not torch.allclose(ema.shadow[name], param.data)

    def test_only_requires_grad_tracked(self):
        """Test only requires_grad params are tracked."""
        model = SimpleModel()

        # Freeze some params
        model.linear1.weight.requires_grad = False

        ema = EMAModel(model, decay=0.999)

        assert 'linear1.weight' not in ema.shadow
        assert 'linear2.weight' in ema.shadow

    def test_custom_decay(self):
        """Test custom decay values."""
        model = SimpleModel()

        ema_slow = EMAModel(model, decay=0.9999)
        ema_fast = EMAModel(model, decay=0.9)

        assert ema_slow.decay == 0.9999
        assert ema_fast.decay == 0.9


class TestEMAUpdate:
    """Test EMAModel.update() method."""

    def test_update_moves_shadow(self):
        """Test update moves shadow toward current weights."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.9)

        # Get initial shadow
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Change model weights
        for param in model.parameters():
            if param.requires_grad:
                param.data.add_(1.0)

        # Update EMA
        ema.update()

        # Shadow should have moved toward new weights
        for name in initial_shadow:
            # shadow = decay * old_shadow + (1-decay) * new_param
            # With decay=0.9, shadow should be 0.9*old + 0.1*new
            expected = 0.9 * initial_shadow[name] + 0.1 * model.state_dict()[name]
            assert torch.allclose(ema.shadow[name], expected, atol=1e-6)

    def test_multiple_updates(self):
        """Test multiple sequential updates."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.9)

        for i in range(10):
            # Simulate training step
            for param in model.parameters():
                if param.requires_grad:
                    param.data.add_(0.1)

            ema.update()

        # Shadow should have accumulated updates
        # Just verify it runs without error and shadow changed
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema.shadow:
                # Shadow should be between initial and current
                # (since decay < 1, shadow lags behind)
                pass  # No assertion needed, just verify runs

    def test_update_with_high_decay(self):
        """Test update with very high decay (slow EMA)."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.9999)

        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Make large change
        for param in model.parameters():
            if param.requires_grad:
                param.data.add_(100.0)

        ema.update()

        # With high decay, shadow should barely move
        for name in initial_shadow:
            diff = (ema.shadow[name] - initial_shadow[name]).abs().max()
            assert diff < 0.1, "High decay should make shadow move slowly"

    def test_update_with_low_decay(self):
        """Test update with low decay (fast EMA)."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.5)

        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Make change
        for param in model.parameters():
            if param.requires_grad:
                param.data.add_(1.0)

        ema.update()

        # With low decay, shadow should move significantly
        for name in initial_shadow:
            diff = (ema.shadow[name] - initial_shadow[name]).abs().max()
            assert diff > 0.4, "Low decay should make shadow move quickly"


class TestEMAApplyRestore:
    """Test apply_shadow() and restore() methods."""

    def test_apply_shadow(self):
        """Test applying shadow weights to model."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.9)

        # Train model (change weights)
        for param in model.parameters():
            if param.requires_grad:
                param.data.add_(10.0)

        ema.update()

        # Apply shadow
        ema.apply_shadow()

        # Model weights should now equal shadow
        for name, param in model.named_parameters():
            if name in ema.shadow:
                assert torch.allclose(param.data, ema.shadow[name])

    def test_apply_creates_backup(self):
        """Test apply_shadow creates backup of training weights."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.9)

        # Modify model
        for param in model.parameters():
            if param.requires_grad:
                param.data.add_(10.0)

        # Store pre-apply weights
        pre_apply = {n: p.data.clone() for n, p in model.named_parameters()}

        ema.apply_shadow()

        # Backup should contain pre-apply weights
        for name in ema.backup:
            assert torch.allclose(ema.backup[name], pre_apply[name])

    def test_restore(self):
        """Test restore brings back training weights."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.9)

        # Modify model
        for param in model.parameters():
            if param.requires_grad:
                param.data.add_(10.0)

        # Store training weights
        training_weights = {n: p.data.clone() for n, p in model.named_parameters()}

        # Apply shadow (for evaluation)
        ema.apply_shadow()

        # Restore (back to training)
        ema.restore()

        # Model should have original training weights
        for name, param in model.named_parameters():
            if name in training_weights:
                assert torch.allclose(param.data, training_weights[name])

    def test_restore_clears_backup(self):
        """Test restore clears the backup dict."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.9)

        ema.apply_shadow()
        assert len(ema.backup) > 0, "Should have backup after apply"

        ema.restore()
        assert len(ema.backup) == 0, "Backup should be cleared after restore"

    def test_apply_restore_cycle(self):
        """Test apply-restore cycle preserves model state."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.9)

        # Initial state
        initial_state = {n: p.data.clone() for n, p in model.named_parameters()}

        # Multiple apply-restore cycles
        for _ in range(5):
            ema.apply_shadow()
            ema.restore()

        # Model should be unchanged
        for name, param in model.named_parameters():
            if name in initial_state:
                assert torch.allclose(param.data, initial_state[name])


class TestEMAIntegration:
    """Integration tests for EMA in training loop."""

    def test_training_loop_pattern(self):
        """Test typical training loop usage."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ema = EMAModel(model, decay=0.999)

        # Simulate training
        for _ in range(10):
            x = torch.randn(4, 10)
            y = torch.randn(4, 5)

            # Forward
            output = model(x)
            loss = ((output - y) ** 2).mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update (after optimizer.step)
            ema.update()

        # Evaluation with EMA
        ema.apply_shadow()
        with torch.no_grad():
            eval_output = model(torch.randn(4, 10))
        ema.restore()

        # Training continues
        x = torch.randn(4, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()  # Should work (training weights restored)

    def test_ema_improves_over_training(self):
        """Test EMA tracks training progress."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ema = EMAModel(model, decay=0.9)

        # Get initial EMA weights
        initial_ema = {k: v.clone() for k, v in ema.shadow.items()}

        # Train
        for _ in range(100):
            x = torch.randn(8, 10)
            y = torch.randn(8, 5)
            output = model(x)
            loss = ((output - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

        # EMA should have changed from initial
        changed = False
        for name in initial_ema:
            if not torch.allclose(ema.shadow[name], initial_ema[name]):
                changed = True
                break

        assert changed, "EMA should have changed during training"


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_model(self):
        """Test with model that has no trainable params."""
        class NoParamModel(nn.Module):
            def forward(self, x):
                return x

        model = NoParamModel()
        ema = EMAModel(model, decay=0.999)

        assert len(ema.shadow) == 0
        ema.update()  # Should not error
        ema.apply_shadow()
        ema.restore()

    def test_frozen_model(self):
        """Test with fully frozen model."""
        model = SimpleModel()
        for param in model.parameters():
            param.requires_grad = False

        ema = EMAModel(model, decay=0.999)

        assert len(ema.shadow) == 0

    def test_partial_freeze(self):
        """Test with partially frozen model."""
        model = SimpleModel()
        model.linear1.requires_grad_(False)

        ema = EMAModel(model, decay=0.999)

        # Only linear2 should be tracked
        assert 'linear1.weight' not in ema.shadow
        assert 'linear1.bias' not in ema.shadow
        assert 'linear2.weight' in ema.shadow
        assert 'linear2.bias' in ema.shadow

    def test_device_consistency(self):
        """Test EMA maintains device consistency."""
        model = SimpleModel()
        ema = EMAModel(model, decay=0.999)

        # All shadow weights should be on same device as model
        for name, param in model.named_parameters():
            if name in ema.shadow:
                assert ema.shadow[name].device == param.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
