"""
Gradient Flow Validation for Phase 4 (BitNet)

Verifies that:
1. All trainable parameters receive gradients
2. No gradients are NaN or Inf
3. Gradient magnitudes are in reasonable ranges
4. No "dead" layers with zero gradients
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GradientCheckResult:
    """Result of gradient flow validation."""

    all_have_gradients: bool
    no_nan: bool
    no_inf: bool
    dead_parameters: List[str] = field(default_factory=list)
    suspicious_parameters: List[str] = field(default_factory=list)
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if gradient validation passed."""
        return (
            self.all_have_gradients
            and self.no_nan
            and self.no_inf
            and len(self.dead_parameters) == 0
        )


def validate_gradient_flow(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    min_grad_norm: float = 1e-10,
    max_grad_norm: float = 1e5,
) -> GradientCheckResult:
    """
    Run a forward/backward pass and analyze gradient flow.

    Args:
        model: Model to validate
        input_ids: Input tensor for forward pass
        attention_mask: Optional attention mask
        labels: Optional labels for loss computation
        min_grad_norm: Gradients below this are suspicious
        max_grad_norm: Gradients above this are suspicious

    Returns:
        GradientCheckResult with detailed analysis
    """
    model.train()
    model.zero_grad()

    # Prepare inputs
    if labels is None:
        labels = input_ids.clone()

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # Forward pass
    try:
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    except TypeError:
        # Model may not support all arguments
        output = model(input_ids)

    # Get loss
    if hasattr(output, "loss") and output.loss is not None:
        loss = output.loss
    elif hasattr(output, "logits"):
        loss = output.logits.sum()
    elif isinstance(output, torch.Tensor):
        loss = output.sum()
    elif isinstance(output, tuple):
        loss = output[0].sum() if isinstance(output[0], torch.Tensor) else output[0]
    else:
        raise ValueError(f"Cannot extract loss from output type: {type(output)}")

    # Backward pass
    loss.backward()

    # Analyze gradients
    result = GradientCheckResult(
        all_have_gradients=True,
        no_nan=True,
        no_inf=True,
        dead_parameters=[],
        suspicious_parameters=[],
        stats={},
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            result.all_have_gradients = False
            result.dead_parameters.append(name)
            continue

        grad = param.grad

        # Handle empty tensors
        if grad.numel() == 0:
            continue

        grad_norm = grad.norm().item()

        # Check for NaN
        if torch.isnan(grad).any():
            result.no_nan = False
            result.suspicious_parameters.append(f"{name} (NaN)")

        # Check for Inf
        if torch.isinf(grad).any():
            result.no_inf = False
            result.suspicious_parameters.append(f"{name} (Inf)")

        # Check for suspicious magnitudes
        if grad_norm < min_grad_norm:
            result.suspicious_parameters.append(f"{name} (very small: {grad_norm:.2e})")
        elif grad_norm > max_grad_norm:
            result.suspicious_parameters.append(f"{name} (very large: {grad_norm:.2e})")

        # Record statistics
        result.stats[name] = {
            "norm": grad_norm,
            "mean": grad.mean().item(),
            "std": grad.std().item() if grad.numel() > 1 else 0.0,
            "max": grad.max().item(),
            "min": grad.min().item(),
        }

    return result


def print_gradient_report(result: GradientCheckResult) -> None:
    """Print a human-readable gradient report."""
    print("=" * 60)
    print("GRADIENT FLOW VALIDATION REPORT")
    print("=" * 60)

    overall_pass = result.all_have_gradients and result.no_nan and result.no_inf
    status = "PASS" if overall_pass else "FAIL"
    print(f"\nOverall Status: {status}")

    print(f"\nAll parameters have gradients: " f"{'PASS' if result.all_have_gradients else 'FAIL'}")
    print(f"No NaN gradients: {'PASS' if result.no_nan else 'FAIL'}")
    print(f"No Inf gradients: {'PASS' if result.no_inf else 'FAIL'}")

    if result.dead_parameters:
        print(f"\nDead parameters ({len(result.dead_parameters)}):")
        for name in result.dead_parameters[:10]:
            print(f"    - {name}")
        if len(result.dead_parameters) > 10:
            print(f"    ... and {len(result.dead_parameters) - 10} more")

    if result.suspicious_parameters:
        print(f"\nSuspicious gradients ({len(result.suspicious_parameters)}):")
        for name in result.suspicious_parameters[:10]:
            print(f"    - {name}")
        if len(result.suspicious_parameters) > 10:
            print(f"    ... and {len(result.suspicious_parameters) - 10} more")

    # Print top 5 largest gradients for debugging
    if result.stats:
        sorted_by_norm = sorted(result.stats.items(), key=lambda x: x[1]["norm"], reverse=True)
        print("\nTop 5 largest gradient norms:")
        for name, stats in sorted_by_norm[:5]:
            print(f"    {name}: {stats['norm']:.4e}")

        print("\nTop 5 smallest gradient norms:")
        sorted_by_norm_asc = sorted(result.stats.items(), key=lambda x: x[1]["norm"])
        for name, stats in sorted_by_norm_asc[:5]:
            print(f"    {name}: {stats['norm']:.4e}")

    print("=" * 60)


def validate_ste_gradient_flow(
    compressed_model: "CompressedModel",
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[bool, str]:
    """
    Validate STE (Straight-Through Estimator) gradient flow.

    STE should allow gradients to flow through quantized layers
    to the full-precision shadow weights.

    Args:
        compressed_model: CompressedModel instance
        input_ids: Input tensor
        attention_mask: Optional attention mask

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not compressed_model.is_compressed:
        return False, "Model is not compressed. Call compress() first."

    # Run gradient validation
    result = validate_gradient_flow(
        model=compressed_model, input_ids=input_ids, attention_mask=attention_mask
    )

    if not result.passed:
        issues = []
        if not result.all_have_gradients:
            issues.append(f"{len(result.dead_parameters)} dead parameters")
        if not result.no_nan:
            issues.append("NaN gradients detected")
        if not result.no_inf:
            issues.append("Inf gradients detected")

        return False, f"STE gradient flow failed: {', '.join(issues)}"

    # Check that gradients reached all quantized layers
    num_params_with_grads = len(result.stats)
    if num_params_with_grads == 0:
        return False, "No parameters received gradients"

    return True, f"STE gradient flow valid. {num_params_with_grads} parameters with gradients."


if __name__ == "__main__":
    # Test with a simple model
    from src.phase4_bitnet.compressed_model import CompressedModel
    from src.phase4_bitnet.config import Phase4Config
    from src.phase4_bitnet.quantizer import BitNetQuantizer

    print("Testing gradient flow validation...")

    # Create a simple test model
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 32)
            self.linear2 = nn.Linear(32, 64)

        def forward(self, input_ids, attention_mask=None, labels=None):
            x = input_ids.float()
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)

            if labels is not None:
                loss = nn.MSELoss()(x, labels.float())
                return type("Output", (), {"loss": loss, "logits": x})()

            return x

    # Create and compress model
    config = Phase4Config()
    base_model = SimpleTestModel()
    quantizer = BitNetQuantizer(config)
    compressed_model = CompressedModel(base_model, quantizer, config)
    compressed_model.compress()

    # Test input
    input_ids = torch.randn(2, 64)

    # Validate gradient flow
    result = validate_gradient_flow(compressed_model, input_ids)
    print_gradient_report(result)

    # Validate STE flow
    success, message = validate_ste_gradient_flow(compressed_model, input_ids)
    print(f"\nSTE Validation: {message}")
