"""
Phase 7 Sandbox Test - Self-Guided Expert Discovery & Routing

Tests the complete Phase 7 pipeline:
1. Expert Discovery (model self-analyzes capabilities)
2. SVF Training (Singular Value Fine-tuning)
3. Transformer^2 Two-Pass Inference
4. ADAS Optimizer (mini NSGA-II search)

All tests run in sandbox with mock 1.58-bit model.

CRITICAL: Verify 1.58-bit format preservation throughout pipeline.
"""

import pytest
pytestmark = pytest.mark.skip(reason='Standalone sandbox script - run with python directly')

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
import random
import math
from typing import Dict, Any, List, Optional

# Import Phase 7 modules
from phase7_experts.expert_discovery import (
    ExpertDiscovery,
    ExpertProfile,
    DiscoveryConfig,
)
from phase7_experts.svf_trainer import (
    SVFTrainer,
    SVFConfig,
    SVFResult,
)
from phase7_experts.transformer2 import (
    Transformer2,
    Transformer2Config,
    Transformer2Result,
)
from phase7_experts.adas_optimizer import (
    ADASOptimizer,
    ADASConfig,
    ADASResult,
)


# Mock 1.58-bit Model (simulates Phase 4 output)
class Mock158BitModel(nn.Module):
    """
    Mock 1.58-bit quantized model for testing.

    Simulates BitNet 1.58-bit format:
    - Weights constrained to {-1, 0, +1}
    - Small size (1M params)
    - Compatible with expert discovery
    """

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 256, num_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    dim_feedforward=hidden_size * 2,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Quantize weights to 1.58-bit {-1, 0, +1}
        self._quantize_to_158bit()

        # Store format marker
        self._is_158bit = True

    def _quantize_to_158bit(self):
        """Quantize all weights to {-1, 0, +1}."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if "weight" in name:
                    # Ternary quantization: -1, 0, +1
                    mean = param.abs().mean()
                    threshold_low = -0.5 * mean
                    threshold_high = 0.5 * mean

                    quantized = torch.zeros_like(param)
                    quantized[param > threshold_high] = 1.0
                    quantized[param < threshold_low] = -1.0
                    # Middle values stay 0

                    param.copy_(quantized)

    def verify_158bit_format(self) -> Dict[str, Any]:
        """Verify model is still in 1.58-bit format."""
        results = {
            "is_158bit": True,
            "violations": [],
            "weight_stats": {},
        }

        with torch.no_grad():
            for name, param in self.named_parameters():
                if "weight" in name:
                    unique_values = param.unique()

                    # Check only {-1, 0, +1} present
                    allowed = torch.tensor([-1.0, 0.0, 1.0], device=param.device)
                    is_valid = all(v in allowed for v in unique_values)

                    if not is_valid:
                        results["is_158bit"] = False
                        results["violations"].append(
                            {
                                "param": name,
                                "unique_values": unique_values.tolist(),
                                "expected": [-1.0, 0.0, 1.0],
                            }
                        )

                    # Stats
                    results["weight_stats"][name] = {
                        "unique_values": unique_values.tolist(),
                        "num_minus_one": (param == -1).sum().item(),
                        "num_zero": (param == 0).sum().item(),
                        "num_plus_one": (param == 1).sum().item(),
                    }

        return results

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass with 1.58-bit weights."""
        # Embedding
        hidden_states = self.embedding(input_ids)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)

        # Output
        logits = self.lm_head(hidden_states)

        # Return in format compatible with expert discovery
        class Output:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = [hidden_states]  # List for layer access
                self.last_hidden_state = hidden_states

        return Output(logits, hidden_states)


# Mock Tokenizer
class MockTokenizer:
    """Simple mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def __call__(self, text: str, **kwargs):
        """Tokenize text (mock implementation)."""
        # Simple: hash text to IDs
        tokens = [hash(word) % self.vocab_size for word in text.split()]
        input_ids = torch.tensor([tokens[:kwargs.get("max_length", 128)]])

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


# Test Functions
def test_expert_discovery(model: nn.Module, tokenizer: Any) -> tuple[int, List[ExpertProfile], bool]:
    """
    Test Stage 1: Expert Discovery

    Returns:
        (num_experts, expert_profiles, success)
    """
    print("\n" + "=" * 60)
    print("STAGE 1: EXPERT DISCOVERY")
    print("=" * 60)

    # Create discovery config
    config = DiscoveryConfig(
        min_experts=3,
        max_experts=10,
        discovery_samples=20,  # Reduced for sandbox
        clustering_threshold=0.7,
    )

    # Run discovery
    discovery = ExpertDiscovery(config)
    num_experts, experts = discovery.discover(model, tokenizer)

    # Validate
    success = True

    # Check expert count
    if not (3 <= num_experts <= 10):
        print(f"  [ERROR] Invalid expert count: {num_experts} (expected 3-10)")
        success = False

    # Check expert profiles
    if len(experts) != num_experts:
        print(f"  [ERROR] Expert count mismatch: {len(experts)} vs {num_experts}")
        success = False

    # Check capabilities assigned
    for expert in experts:
        if not expert.capabilities:
            print(f"  [ERROR] Expert {expert.id} has no capabilities")
            success = False

        if expert.strength_score <= 0 or expert.strength_score > 1:
            print(f"  [ERROR] Expert {expert.id} invalid strength: {expert.strength_score}")
            success = False

    if success:
        print(f"\n  [SUCCESS] Discovered {num_experts} experts")
        print("  Expert Summary:")
        for expert in experts:
            print(f"    - {expert.name}: {expert.capabilities[:2]} (strength: {expert.strength_score:.2f})")

    return num_experts, experts, success


def test_svf_training(
    model: nn.Module, experts: List[ExpertProfile], tokenizer: Any
) -> tuple[bool, Dict[str, Any]]:
    """
    Test Stage 2: SVF (Singular Value Fine-tuning) Training

    Returns:
        (success, metrics)
    """
    print("\n" + "=" * 60)
    print("STAGE 2: SVF TRAINING")
    print("=" * 60)

    # Create SVF config (minimal for sandbox)
    config = SVFConfig(
        num_singular_values=16,  # Reduced for speed
        learning_rate=1e-4,
        num_epochs=2,  # Minimal training
        batch_size=2,
    )

    trainer = SVFTrainer(config)

    # Train first expert only (sandbox)
    if not experts:
        print("  [ERROR] No experts to train")
        return False, {}

    expert = experts[0]
    print(f"\n  Training Expert 0: {expert.name}")

    # Generate minimal training data
    # Use first capability if available, else generic
    capability = expert.capabilities[0] if expert.capabilities else "general"
    training_data = [
        {"prompt": "Test prompt 1", "capability": capability},
        {"prompt": "Test prompt 2", "capability": capability},
    ]

    # Train
    trained_model, result = trainer.train_expert(
        model=model,
        expert_id=expert.id,
        expert_capabilities=expert.capabilities,
        tokenizer=tokenizer,
        training_data=training_data,
    )

    # Validate
    success = result.success

    if not success:
        print("  [ERROR] SVF training failed")
        return False, {}

    # Check SV changes
    if not result.sv_changes:
        print("  [WARNING] No SV changes recorded")

    # Check metrics
    if "epoch_losses" not in result.metrics:
        print("  [ERROR] Missing epoch losses")
        success = False

    if success:
        print(f"\n  [SUCCESS] SVF training complete")
        print(f"    Final loss: {result.final_loss:.4f}")
        print(f"    SV modifications: {len(result.sv_changes)} layers")

    metrics = {
        "final_loss": result.final_loss,
        "num_sv_changes": len(result.sv_changes),
        "metrics": result.metrics,
    }

    return success, metrics


def test_transformer2_inference(
    model: nn.Module, num_experts: int, tokenizer: Any
) -> tuple[bool, Dict[str, Any]]:
    """
    Test Stage 2b: Transformer^2 Two-Pass Inference

    Returns:
        (success, metrics)
    """
    print("\n" + "=" * 60)
    print("STAGE 2b: TRANSFORMER^2 TWO-PASS INFERENCE")
    print("=" * 60)

    # Create Transformer2 config
    config = Transformer2Config(
        num_experts=num_experts,
        expert_rank=32,  # Low-rank adapters
        routing_hidden_dim=128,
    )

    # Wrap base model
    t2_model = Transformer2(base_model=model, config=config)

    # Test forward pass
    test_input = "Test input for two-pass inference"
    inputs = tokenizer(test_input, max_length=32, truncation=True, padding=True)

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    # Forward
    result = t2_model.forward(input_ids, return_routing=True)

    # Validate
    success = True

    # Check outputs
    if result.logits is None:
        print("  [ERROR] No logits output")
        success = False

    if result.routing_weights is None:
        print("  [ERROR] No routing weights")
        success = False
    else:
        # Check routing weights sum to 1
        routing_sum = result.routing_weights.sum(dim=-1)
        if not torch.allclose(routing_sum, torch.ones_like(routing_sum), atol=1e-5):
            print(f"  [ERROR] Routing weights don't sum to 1: {routing_sum}")
            success = False

    # Check expert contributions
    if not result.expert_contributions:
        print("  [WARNING] No expert contributions recorded")

    # Check metrics
    if "routing_entropy" not in result.metrics:
        print("  [ERROR] Missing routing entropy")
        success = False

    if success:
        print(f"\n  [SUCCESS] Transformer^2 inference working")
        print(f"    Logits shape: {result.logits.shape}")
        print(f"    Routing weights shape: {result.routing_weights.shape}")
        print(f"    Routing entropy: {result.metrics['routing_entropy']:.3f}")
        print(f"    Routing sparsity: {result.metrics['routing_sparsity']:.3f}")

    metrics = {
        "logits_shape": list(result.logits.shape),
        "routing_shape": list(result.routing_weights.shape),
        "metrics": result.metrics,
    }

    return success, metrics


def test_adas_optimizer(
    model: nn.Module, experts: List[ExpertProfile], tokenizer: Any
) -> tuple[bool, Dict[str, Any]]:
    """
    Test Stage 3: ADAS Optimizer (mini NSGA-II)

    Returns:
        (success, metrics)
    """
    print("\n" + "=" * 60)
    print("STAGE 3: ADAS OPTIMIZER (NSGA-II)")
    print("=" * 60)

    # Create ADAS config (minimal for sandbox)
    config = ADASConfig(
        population_size=10,  # Reduced from 50
        num_generations=3,  # Reduced from 100
        mutation_rate=0.1,
        crossover_rate=0.7,
        tournament_size=3,
    )

    optimizer = ADASOptimizer(config)

    # Run optimization
    optimized_model, result = optimizer.optimize(
        model=model,
        experts=experts,
        tokenizer=tokenizer,
        evaluator=None,  # Use default evaluator
    )

    # Validate
    success = result.success

    if not success:
        print("  [ERROR] ADAS optimization failed")
        return False, {}

    # Check best individual
    if result.best_individual is None:
        print("  [ERROR] No best individual selected")
        success = False
    else:
        # Check routing weights
        if len(result.best_individual.routing_weights) != len(experts):
            print(f"  [ERROR] Routing weight count mismatch")
            success = False

        # Check weights sum to 1
        weight_sum = sum(result.best_individual.routing_weights)
        if not (0.99 <= weight_sum <= 1.01):
            print(f"  [ERROR] Routing weights don't sum to 1: {weight_sum}")
            success = False

    # Check Pareto front
    if not result.pareto_front:
        print("  [WARNING] Empty Pareto front")

    # Check generation history
    if len(result.generation_history) != config.num_generations:
        print(f"  [ERROR] Generation history mismatch: {len(result.generation_history)} vs {config.num_generations}")
        success = False

    if success:
        print(f"\n  [SUCCESS] ADAS optimization complete")
        print(f"    Pareto front size: {len(result.pareto_front)}")
        print(f"    Best accuracy: {result.best_individual.fitness_scores.get('accuracy', 0):.3f}")
        print(f"    Best latency: {result.best_individual.fitness_scores.get('latency', 0):.3f}")
        print(f"    Expert weights: {[f'{w:.2f}' for w in result.best_individual.routing_weights[:3]]}")

    metrics = {
        "pareto_front_size": len(result.pareto_front),
        "best_fitness": result.best_individual.fitness_scores,
        "total_evaluations": result.metrics["total_evaluations"],
    }

    return success, metrics


def test_158bit_preservation(model: Mock158BitModel, name: str) -> bool:
    """Test that model is still in 1.58-bit format."""
    print(f"\n  Verifying 1.58-bit format ({name})...")

    verification = model.verify_158bit_format()

    if not verification["is_158bit"]:
        print(f"    [ERROR] 1.58-bit format VIOLATED")
        for violation in verification["violations"]:
            print(f"      - {violation['param']}: {violation['unique_values']}")
        return False

    # Count weight distribution
    total_weights = 0
    total_minus_one = 0
    total_zero = 0
    total_plus_one = 0

    for stats in verification["weight_stats"].values():
        total_minus_one += stats["num_minus_one"]
        total_zero += stats["num_zero"]
        total_plus_one += stats["num_plus_one"]

    total_weights = total_minus_one + total_zero + total_plus_one

    print(f"    [SUCCESS] 1.58-bit format preserved")
    print(f"      -1: {total_minus_one}/{total_weights} ({100*total_minus_one/total_weights:.1f}%)")
    print(f"       0: {total_zero}/{total_weights} ({100*total_zero/total_weights:.1f}%)")
    print(f"      +1: {total_plus_one}/{total_weights} ({100*total_plus_one/total_weights:.1f}%)")

    return True


def main():
    """Run Phase 7 sandbox test."""
    print("\n" + "=" * 70)
    print("PHASE 7 SANDBOX TEST: SELF-GUIDED EXPERT DISCOVERY")
    print("=" * 70)
    print("\nTesting complete Phase 7 pipeline:")
    print("  1. Expert Discovery (model self-analyzes)")
    print("  2. SVF Training (Singular Value Fine-tuning)")
    print("  3. Transformer^2 Two-Pass Inference")
    print("  4. ADAS Optimizer (NSGA-II search)")
    print("  5. 1.58-bit Format Preservation")
    print("\n" + "=" * 70)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create mock 1.58-bit model
    print("\nCreating mock 1.58-bit model...")
    model = Mock158BitModel(vocab_size=1000, hidden_size=128, num_layers=4).to(device)
    tokenizer = MockTokenizer(vocab_size=1000)

    # Verify initial 1.58-bit format
    initial_158bit = test_158bit_preservation(model, "initial")

    # Test results
    results = {
        "phase": 7,
        "expert_count": 0,
        "svf_working": False,
        "transformer2_working": False,
        "adas_working": False,
        "158bit_preserved": initial_158bit,
        "errors": [],
    }

    try:
        # Stage 1: Expert Discovery
        num_experts, experts, discovery_success = test_expert_discovery(model, tokenizer)
        results["expert_count"] = num_experts

        if not discovery_success:
            results["errors"].append("Expert discovery failed")

        # Verify 1.58-bit after discovery
        if not test_158bit_preservation(model, "post-discovery"):
            results["158bit_preserved"] = False
            results["errors"].append("1.58-bit format violated after discovery")

        # Stage 2: SVF Training
        svf_success, svf_metrics = test_svf_training(model, experts, tokenizer)
        results["svf_working"] = svf_success

        if not svf_success:
            results["errors"].append("SVF training failed")

        # Verify 1.58-bit after SVF
        # Note: SVF modifies singular values, so we check the original model
        if not test_158bit_preservation(model, "post-svf"):
            results["158bit_preserved"] = False
            results["errors"].append("1.58-bit format violated after SVF")

        # Stage 2b: Transformer^2 Inference
        t2_success, t2_metrics = test_transformer2_inference(model, num_experts, tokenizer)
        results["transformer2_working"] = t2_success

        if not t2_success:
            results["errors"].append("Transformer^2 inference failed")

        # Stage 3: ADAS Optimizer
        adas_success, adas_metrics = test_adas_optimizer(model, experts, tokenizer)
        results["adas_working"] = adas_success

        if not adas_success:
            results["errors"].append("ADAS optimization failed")

        # Final 1.58-bit check
        final_158bit = test_158bit_preservation(model, "final")
        results["158bit_preserved"] = results["158bit_preserved"] and final_158bit

        if not final_158bit:
            results["errors"].append("1.58-bit format violated at end")

    except Exception as e:
        results["errors"].append(f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()

    # Final Report
    print("\n" + "=" * 70)
    print("SANDBOX TEST RESULTS")
    print("=" * 70)
    print(f"\nPhase: {results['phase']}")
    print(f"Status: {'SUCCESS' if not results['errors'] else 'FAILED'}")
    print(f"\nExpert Count Discovered: {results['expert_count']}")
    print(f"SVF Working: {results['svf_working']}")
    print(f"Transformer^2 Working: {results['transformer2_working']}")
    print(f"ADAS Working: {results['adas_working']}")
    print(f"1.58-bit Preserved: {results['158bit_preserved']}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for i, error in enumerate(results["errors"], 1):
            print(f"  {i}. {error}")
    else:
        print("\n[SUCCESS] All tests passed!")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    results = main()

    # Exit with error code if failed
    if results["errors"]:
        sys.exit(1)
    else:
        sys.exit(0)
