import pytest
pytestmark = pytest.mark.skip(reason="Standalone sandbox script - run with python directly")

"""
Phase 7 Sandbox Test V2 - Self-Guided Expert Discovery & Routing

Simplified sandbox test that works around activation collection issues.
Tests the complete Phase 7 pipeline with mock data.

CRITICAL: Verify 1.58-bit format preservation throughout pipeline.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
from typing import Dict, Any, List

# Import Phase 7 modules
from phase7_experts.expert_discovery import ExpertProfile
from phase7_experts.svf_trainer import SVFTrainer, SVFConfig
from phase7_experts.transformer2 import Transformer2, Transformer2Config
from phase7_experts.adas_optimizer import ADASOptimizer, ADASConfig, Individual


# Mock 1.58-bit Model (simulates Phase 4 output)
class Mock158BitModel(nn.Module):
    """Mock 1.58-bit quantized model for testing."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 256, num_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Simple architecture
        self.embedding = nn.Embedding(vocab_size, hidden_size)
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
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Quantize to 1.58-bit
        self._quantize_to_158bit()
        self._is_158bit = True

    def _quantize_to_158bit(self):
        """Quantize all weights to {-1, 0, +1}."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if "weight" in name:
                    mean = param.abs().mean()
                    threshold_low = -0.5 * mean
                    threshold_high = 0.5 * mean

                    quantized = torch.zeros_like(param)
                    quantized[param > threshold_high] = 1.0
                    quantized[param < threshold_low] = -1.0
                    param.copy_(quantized)

    def verify_158bit_format(self) -> Dict[str, Any]:
        """Verify model is still in 1.58-bit format."""
        results = {"is_158bit": True, "violations": [], "stats": {}}

        with torch.no_grad():
            total_minus = total_zero = total_plus = 0

            for name, param in self.named_parameters():
                if "weight" in name:
                    unique_values = param.unique()
                    allowed = torch.tensor([-1.0, 0.0, 1.0], device=param.device)
                    is_valid = all(v in allowed for v in unique_values)

                    if not is_valid:
                        results["is_158bit"] = False
                        results["violations"].append(name)

                    total_minus += (param == -1).sum().item()
                    total_zero += (param == 0).sum().item()
                    total_plus += (param == 1).sum().item()

            total = total_minus + total_zero + total_plus
            results["stats"] = {
                "minus_one": f"{total_minus}/{total} ({100*total_minus/total:.1f}%)" if total > 0 else "0/0",
                "zero": f"{total_zero}/{total} ({100*total_zero/total:.1f}%)" if total > 0 else "0/0",
                "plus_one": f"{total_plus}/{total} ({100*total_plus/total:.1f}%)" if total > 0 else "0/0",
            }

        return results

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass."""
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(hidden_states)

        class Output:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = [hidden_states]
                self.last_hidden_state = hidden_states

        return Output(logits, hidden_states)


class MockTokenizer:
    """Simple mock tokenizer."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def __call__(self, text: str, **kwargs):
        tokens = [hash(word) % self.vocab_size for word in text.split()]
        input_ids = torch.tensor([tokens[:kwargs.get("max_length", 128)]])
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def create_mock_experts(num_experts: int = 5) -> List[ExpertProfile]:
    """Create mock expert profiles (bypass discovery for sandbox)."""
    capabilities_pool = [
        ["reasoning", "logic"],
        ["coding", "programming"],
        ["math", "calculation"],
        ["writing", "creativity"],
        ["analysis", "data"],
    ]

    experts = []
    for i in range(num_experts):
        experts.append(
            ExpertProfile(
                id=i,
                name=f"expert_{i}",
                capabilities=capabilities_pool[i % len(capabilities_pool)],
                strength_score=0.5 + i * 0.1,
                activation_pattern=[0.5] * 10,
            )
        )

    return experts


def test_expert_discovery_mock():
    """Test Stage 1: Expert Discovery (mocked)."""
    print("\n" + "=" * 70)
    print("STAGE 1: EXPERT DISCOVERY (MOCKED)")
    print("=" * 70)

    num_experts = 5
    experts = create_mock_experts(num_experts)

    print(f"\n  Created {num_experts} mock experts:")
    for expert in experts:
        print(f"    - {expert.name}: {expert.capabilities} (strength: {expert.strength_score:.2f})")

    print(f"\n  [SUCCESS] Expert discovery (mock) complete")

    return num_experts, experts, True


def test_svf_training(model, experts, tokenizer):
    """Test Stage 2: SVF Training."""
    print("\n" + "=" * 70)
    print("STAGE 2: SVF TRAINING")
    print("=" * 70)

    config = SVFConfig(
        num_singular_values=16,
        learning_rate=1e-4,
        num_epochs=2,
        batch_size=2,
    )

    trainer = SVFTrainer(config)
    expert = experts[0]

    print(f"\n  Training Expert 0: {expert.name}")

    training_data = [
        {"prompt": "Test reasoning prompt", "capability": "reasoning"},
        {"prompt": "Another test prompt", "capability": "reasoning"},
    ]

    try:
        trained_model, result = trainer.train_expert(
            model=model,
            expert_id=expert.id,
            expert_capabilities=expert.capabilities,
            tokenizer=tokenizer,
            training_data=training_data,
        )

        success = result.success
        if success:
            print(f"\n  [SUCCESS] SVF training complete")
            print(f"    Final loss: {result.final_loss:.4f}")
            print(f"    SV modifications: {len(result.sv_changes)} layers")

        return success, {
            "final_loss": result.final_loss,
            "num_sv_changes": len(result.sv_changes),
        }

    except Exception as e:
        print(f"\n  [ERROR] SVF training failed: {e}")
        return False, {}


def test_transformer2_inference(model, num_experts, tokenizer):
    """Test Stage 2b: Transformer^2 Inference."""
    print("\n" + "=" * 70)
    print("STAGE 2b: TRANSFORMER^2 TWO-PASS INFERENCE")
    print("=" * 70)

    try:
        config = Transformer2Config(
            num_experts=num_experts,
            expert_rank=32,
            routing_hidden_dim=128,
        )

        t2_model = Transformer2(base_model=model, config=config)

        # Move Transformer2 components to same device as base model
        device = next(model.parameters()).device
        t2_model = t2_model.to(device)

        test_input = "Test input for routing"
        inputs = tokenizer(test_input, max_length=32, truncation=True, padding=True)

        input_ids = inputs["input_ids"].to(device)

        result = t2_model.forward(input_ids, return_routing=True)

        # Validate
        success = True
        if result.logits is None:
            print("  [ERROR] No logits output")
            success = False

        if result.routing_weights is None:
            print("  [ERROR] No routing weights")
            success = False
        else:
            routing_sum = result.routing_weights.sum(dim=-1)
            if not torch.allclose(routing_sum, torch.ones_like(routing_sum), atol=1e-5):
                print(f"  [WARNING] Routing weights sum: {routing_sum.item():.4f}")

        if success:
            print(f"\n  [SUCCESS] Transformer^2 inference working")
            print(f"    Logits shape: {list(result.logits.shape)}")
            print(f"    Routing shape: {list(result.routing_weights.shape)}")
            print(f"    Routing entropy: {result.metrics['routing_entropy']:.3f}")

        return success, {"metrics": result.metrics}

    except Exception as e:
        print(f"\n  [ERROR] Transformer^2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_adas_optimizer(model, experts, tokenizer):
    """Test Stage 3: ADAS Optimizer."""
    print("\n" + "=" * 70)
    print("STAGE 3: ADAS OPTIMIZER (MINI NSGA-II)")
    print("=" * 70)

    try:
        config = ADASConfig(
            population_size=10,
            num_generations=3,
            mutation_rate=0.1,
            crossover_rate=0.7,
        )

        optimizer = ADASOptimizer(config)

        optimized_model, result = optimizer.optimize(
            model=model, experts=experts, tokenizer=tokenizer
        )

        success = result.success
        if success:
            print(f"\n  [SUCCESS] ADAS optimization complete")
            print(f"    Pareto front size: {len(result.pareto_front)}")
            print(f"    Best accuracy: {result.best_individual.fitness_scores.get('accuracy', 0):.3f}")
            print(f"    Best latency: {result.best_individual.fitness_scores.get('latency', 0):.3f}")

        return success, {
            "pareto_front_size": len(result.pareto_front),
            "best_fitness": result.best_individual.fitness_scores,
        }

    except Exception as e:
        print(f"\n  [ERROR] ADAS failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def verify_158bit(model, stage_name):
    """Verify 1.58-bit format."""
    print(f"\n  Verifying 1.58-bit format ({stage_name})...")
    verification = model.verify_158bit_format()

    if not verification["is_158bit"]:
        print(f"    [ERROR] 1.58-bit format VIOLATED")
        for v in verification["violations"]:
            print(f"      - {v}")
        return False

    stats = verification["stats"]
    print(f"    [SUCCESS] 1.58-bit preserved")
    print(f"      -1: {stats['minus_one']}")
    print(f"       0: {stats['zero']}")
    print(f"      +1: {stats['plus_one']}")
    return True


def main():
    """Run Phase 7 sandbox test."""
    print("\n" + "=" * 70)
    print("PHASE 7 SANDBOX TEST V2")
    print("=" * 70)
    print("\nTesting complete Phase 7 pipeline (sandbox):")
    print("  1. Expert Discovery (mocked for stability)")
    print("  2. SVF Training")
    print("  3. Transformer^2 Inference")
    print("  4. ADAS Optimizer (3 generations)")
    print("  5. 1.58-bit Format Preservation")
    print("\n" + "=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create model
    print("\nCreating mock 1.58-bit model...")
    model = Mock158BitModel(vocab_size=1000, hidden_size=128, num_layers=4).to(device)
    tokenizer = MockTokenizer()

    results = {
        "phase": 7,
        "status": "PENDING",
        "expert_count_discovered": 0,
        "svf_working": False,
        "transformer2_working": False,
        "adas_working": False,
        "158bit_preserved": True,
        "errors": [],
    }

    try:
        # Initial check
        if not verify_158bit(model, "initial"):
            results["158bit_preserved"] = False
            results["errors"].append("Initial 1.58-bit verification failed")

        # Stage 1: Expert Discovery (mocked)
        num_experts, experts, discovery_success = test_expert_discovery_mock()
        results["expert_count_discovered"] = num_experts

        if not discovery_success:
            results["errors"].append("Expert discovery failed")

        # Stage 2: SVF Training
        svf_success, svf_metrics = test_svf_training(model, experts, tokenizer)
        results["svf_working"] = svf_success

        if not svf_success:
            results["errors"].append("SVF training failed")

        if not verify_158bit(model, "post-SVF"):
            results["158bit_preserved"] = False
            results["errors"].append("1.58-bit violated after SVF")

        # Stage 2b: Transformer^2
        t2_success, t2_metrics = test_transformer2_inference(model, num_experts, tokenizer)
        results["transformer2_working"] = t2_success

        if not t2_success:
            results["errors"].append("Transformer^2 failed")

        # Stage 3: ADAS
        adas_success, adas_metrics = test_adas_optimizer(model, experts, tokenizer)
        results["adas_working"] = adas_success

        if not adas_success:
            results["errors"].append("ADAS failed")

        # Final check
        if not verify_158bit(model, "final"):
            results["158bit_preserved"] = False
            results["errors"].append("1.58-bit violated at end")

        # Determine status
        if not results["errors"]:
            results["status"] = "SUCCESS"
        else:
            results["status"] = "FAILED"

    except Exception as e:
        results["errors"].append(f"Exception: {str(e)}")
        results["status"] = "FAILED"
        import traceback
        traceback.print_exc()

    # Report
    print("\n" + "=" * 70)
    print("SANDBOX TEST RESULTS")
    print("=" * 70)
    print(f"\nPhase: {results['phase']}")
    print(f"Status: {results['status']}")
    print(f"\nExpert Count Discovered: {results['expert_count_discovered']}")
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
    sys.exit(0 if results["status"] == "SUCCESS" else 1)
