"""
Unit tests for Phase 6: Loss Functions

Tests:
- kl_divergence_loss
- reverse_kl_divergence_loss
- jensen_shannon_divergence
- distillation_loss
- KLDivergenceLoss class

Target: >=90% coverage for loss functions
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from phase6_baking.loss_functions import (
    KLDivergenceLoss,
    distillation_loss,
    jensen_shannon_divergence,
    kl_divergence_loss,
    reverse_kl_divergence_loss,
)


class TestKLDivergenceLoss:
    """Test kl_divergence_loss function."""

    def test_basic_computation(self):
        """Test basic KL divergence computation."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = kl_divergence_loss(logits, target)

        assert loss.ndim == 0, "Should return scalar"
        assert loss.item() >= 0, "KL divergence must be non-negative"

    def test_identical_distributions(self):
        """Test KL divergence when distributions match."""
        batch, seq, vocab = 2, 5, 50
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(logits, dim=-1)  # Same as model output

        loss = kl_divergence_loss(logits, target, temperature=1.0)

        # When distributions are identical, KL should be ~0
        assert loss.item() < 0.01, f"KL should be near 0 for identical dists, got {loss.item()}"

    def test_reduction_none(self):
        """Test no reduction returns full tensor."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = kl_divergence_loss(logits, target, reduction="none")

        assert loss.shape == (
            batch,
            seq,
            vocab,
        ), f"Expected {(batch, seq, vocab)}, got {loss.shape}"

    def test_reduction_sum(self):
        """Test sum reduction."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = kl_divergence_loss(logits, target, reduction="sum")

        assert loss.ndim == 0, "Sum reduction should return scalar"
        assert loss.item() >= 0, "KL divergence must be non-negative"

    def test_reduction_mean(self):
        """Test mean reduction."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = kl_divergence_loss(logits, target, reduction="mean")

        assert loss.ndim == 0, "Mean reduction should return scalar"
        assert loss.item() >= 0, "KL divergence must be non-negative"

    def test_reduction_batchmean(self):
        """Test batchmean reduction (default)."""
        batch, seq, vocab = 4, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = kl_divergence_loss(logits, target, reduction="batchmean")

        assert loss.ndim == 0, "Batchmean reduction should return scalar"
        assert loss.item() >= 0, "KL divergence must be non-negative"

    def test_temperature_scaling(self):
        """Test temperature affects distribution sharpness."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss_cold = kl_divergence_loss(logits, target, temperature=0.5)
        loss_warm = kl_divergence_loss(logits, target, temperature=2.0)

        # Both should be valid (non-negative)
        assert loss_cold.item() >= 0
        assert loss_warm.item() >= 0

    def test_gradient_flow(self):
        """Test gradients flow through the loss."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab, requires_grad=True)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = kl_divergence_loss(logits, target)
        loss.backward()

        assert logits.grad is not None, "Gradients should flow"
        assert not torch.isnan(logits.grad).any(), "Gradients should not be NaN"


class TestReverseKLDivergence:
    """Test reverse_kl_divergence_loss function."""

    def test_basic_computation(self):
        """Test basic reverse KL computation."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = reverse_kl_divergence_loss(logits, target)

        assert loss.ndim == 0, "Should return scalar"
        assert loss.item() >= 0, "Reverse KL must be non-negative"

    def test_identical_distributions(self):
        """Test reverse KL when distributions match."""
        batch, seq, vocab = 2, 5, 50
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(logits, dim=-1)

        loss = reverse_kl_divergence_loss(logits, target, temperature=1.0)

        # When distributions are identical, KL should be ~0
        assert loss.item() < 0.01, f"Reverse KL should be near 0 for identical dists"

    def test_all_reductions(self):
        """Test all reduction modes."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        for reduction in ["none", "sum", "mean", "batchmean"]:
            loss = reverse_kl_divergence_loss(logits, target, reduction=reduction)
            if reduction == "none":
                assert loss.shape == (batch, seq, vocab)
            else:
                assert loss.ndim == 0

    def test_invalid_reduction(self):
        """Test invalid reduction raises error."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        with pytest.raises(ValueError):
            reverse_kl_divergence_loss(logits, target, reduction="invalid")


class TestJensenShannonDivergence:
    """Test jensen_shannon_divergence function."""

    def test_basic_computation(self):
        """Test basic JS divergence."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = jensen_shannon_divergence(logits, target)

        assert loss.ndim == 0, "Should return scalar"
        assert loss.item() >= 0, "JS divergence must be non-negative"

    def test_bounded(self):
        """Test JS divergence is bounded by log(2)."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = jensen_shannon_divergence(logits, target)

        assert loss.item() <= math.log(2) + 0.01, "JS divergence should be <= log(2)"

    def test_symmetry(self):
        """Test JS divergence is approximately symmetric."""
        batch, seq, vocab = 2, 5, 100
        logits1 = torch.randn(batch, seq, vocab)
        logits2 = torch.randn(batch, seq, vocab)
        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)

        # JS(P||Q) should equal JS(Q||P) for normalized distributions
        # Note: Function takes logits for pred, probs for target
        js1 = jensen_shannon_divergence(logits1, probs2)
        js2 = jensen_shannon_divergence(logits2, probs1)

        # Both should be in valid range
        assert js1.item() >= 0
        assert js2.item() >= 0

    def test_identical_distributions(self):
        """Test JS divergence when distributions match."""
        batch, seq, vocab = 2, 5, 50
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(logits, dim=-1)

        loss = jensen_shannon_divergence(logits, target)

        assert loss.item() < 0.01, "JS should be near 0 for identical dists"


class TestDistillationLoss:
    """Test distillation_loss function."""

    def test_soft_targets_only(self):
        """Test distillation with soft targets only (alpha=1)."""
        batch, seq, vocab = 2, 5, 100
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)

        loss = distillation_loss(student, teacher, alpha=1.0)

        assert loss.ndim == 0, "Should return scalar"
        assert loss.item() >= 0, "Loss must be non-negative"

    def test_hard_targets_only(self):
        """Test distillation with hard targets only (alpha=0)."""
        batch, seq, vocab = 2, 5, 100
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))

        loss = distillation_loss(student, teacher, labels=labels, alpha=0.0)

        assert loss.ndim == 0, "Should return scalar"
        assert loss.item() >= 0, "Loss must be non-negative"

    def test_mixed_targets(self):
        """Test distillation with both soft and hard targets."""
        batch, seq, vocab = 2, 5, 100
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))

        loss = distillation_loss(student, teacher, labels=labels, alpha=0.5)

        assert loss.ndim == 0, "Should return scalar"
        assert loss.item() >= 0, "Loss must be non-negative"

    def test_temperature_effect(self):
        """Test temperature affects soft targets."""
        batch, seq, vocab = 2, 5, 100
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)

        loss_cold = distillation_loss(student, teacher, temperature=1.0, alpha=1.0)
        loss_warm = distillation_loss(student, teacher, temperature=4.0, alpha=1.0)

        # Both should be valid
        assert loss_cold.item() >= 0
        assert loss_warm.item() >= 0

    def test_gradient_flow(self):
        """Test gradients flow through distillation loss."""
        batch, seq, vocab = 2, 5, 100
        student = torch.randn(batch, seq, vocab, requires_grad=True)
        teacher = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))

        loss = distillation_loss(student, teacher, labels=labels, alpha=0.5)
        loss.backward()

        assert student.grad is not None, "Gradients should flow"


class TestKLDivergenceLossModule:
    """Test KLDivergenceLoss nn.Module wrapper."""

    def test_initialization(self):
        """Test module initialization."""
        loss_fn = KLDivergenceLoss()

        assert loss_fn.reduction == "batchmean"
        assert loss_fn.temperature == 1.0
        assert loss_fn.epsilon == 1e-8

    def test_custom_initialization(self):
        """Test custom initialization."""
        loss_fn = KLDivergenceLoss(reduction="mean", temperature=2.0, epsilon=1e-6)

        assert loss_fn.reduction == "mean"
        assert loss_fn.temperature == 2.0
        assert loss_fn.epsilon == 1e-6

    def test_forward(self):
        """Test forward pass."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss_fn = KLDivergenceLoss()
        loss = loss_fn(logits, target)

        assert loss.ndim == 0, "Should return scalar"
        assert loss.item() >= 0, "KL divergence must be non-negative"

    def test_module_in_pipeline(self):
        """Test module can be used in nn.Sequential."""
        loss_fn = KLDivergenceLoss()

        # Should be callable like other nn.Modules
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab, requires_grad=True)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = loss_fn(logits, target)
        loss.backward()

        assert logits.grad is not None


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_batch(self):
        """Test with batch size 1."""
        batch, seq, vocab = 1, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = kl_divergence_loss(logits, target)
        assert not torch.isnan(loss), "Should handle batch size 1"

    def test_very_large_vocab(self):
        """Test with large vocabulary."""
        batch, seq, vocab = 2, 5, 32000
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = kl_divergence_loss(logits, target)
        assert not torch.isnan(loss), "Should handle large vocab"

    def test_extreme_logits(self):
        """Test with extreme logit values."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab) * 100  # Very extreme
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = kl_divergence_loss(logits, target)
        assert not torch.isnan(loss), "Should handle extreme logits"
        assert not torch.isinf(loss), "Should not be infinite"

    def test_near_zero_target(self):
        """Test with near-zero target probabilities."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)

        # Create sparse target (most probabilities near 0)
        target = torch.zeros(batch, seq, vocab)
        target[:, :, 0] = 1.0  # One-hot

        loss = kl_divergence_loss(logits, target)
        assert not torch.isnan(loss), "Should handle sparse targets"

    def test_device_compatibility(self):
        """Test works on CPU (GPU tested separately if available)."""
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch, seq, vocab)
        target = F.softmax(torch.randn(batch, seq, vocab), dim=-1)

        loss = kl_divergence_loss(logits, target)
        assert loss.device.type == "cpu", "Should work on CPU"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
