import pytest
pytestmark = pytest.mark.skip(reason="Standalone sandbox script - run with python directly")

"""
Phase 6 Sandbox Test: Tool & Persona Baking

Tests Phase 6 implementation with mock 1.58-bit models in isolated environment.

Components Tested:
- BakingEngine initialization
- A-Cycle tool optimization (mock)
- B-Cycle persona discovery (mock)
- Half-baking gradual integration
- PromptPursuit iterative re-baking
- DriftMeter persona consistency
- CrossTaskValidator forgetting detection
- 1.58-bit format preservation

Critical: Model must remain in 1.58-bit format throughout!
"""

import sys
import torch
import torch.nn as nn
from typing import Dict, List, Any
from dataclasses import dataclass


# ============================================================================
# MOCK 1.58-BIT MODEL
# ============================================================================

class Mock158BitLinear(nn.Module):
    """Mock 1.58-bit quantized linear layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weights are quantized to {-1, 0, +1}
        self.weight = nn.Parameter(
            torch.randint(-1, 2, (out_features, in_features), dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Metadata to verify 1.58-bit format
        self.quantization_format = "1.58-bit"
        self.quantization_levels = 3  # {-1, 0, +1}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

    def verify_quantization(self) -> bool:
        """Verify weights are still in {-1, 0, +1}."""
        unique_values = torch.unique(self.weight)
        return torch.all(torch.isin(unique_values, torch.tensor([-1.0, 0.0, 1.0])))


class Mock158BitModel(nn.Module):
    """Mock 1.58-bit quantized model for testing."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 256, num_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding (not quantized)
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

        # 1.58-bit quantized layers
        self.layers = nn.ModuleList([
            Mock158BitLinear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        # Output projection (1.58-bit)
        self.lm_head = Mock158BitLinear(hidden_size, vocab_size)

        # Metadata
        self.quantization_format = "1.58-bit"
        self.from_phase = 4

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Any:
        """Forward pass with mock outputs."""
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = torch.relu(layer(x))

        logits = self.lm_head(x)

        # Mock outputs structure
        @dataclass
        class MockOutput:
            logits: torch.Tensor
            loss: torch.Tensor = None
            hidden_states: List[torch.Tensor] = None
            last_hidden_state: torch.Tensor = None

        return MockOutput(
            logits=logits,
            loss=torch.tensor(0.5),
            hidden_states=[x],
            last_hidden_state=x
        )

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, **kwargs) -> torch.Tensor:
        """Mock generation method."""
        batch_size = input_ids.shape[0]
        generated = torch.randint(0, self.vocab_size, (batch_size, max_new_tokens))
        return torch.cat([input_ids, generated], dim=1)

    def verify_quantization(self) -> Dict[str, bool]:
        """Verify all layers remain in 1.58-bit format."""
        results = {}
        for i, layer in enumerate(self.layers):
            results[f'layer_{i}'] = layer.verify_quantization()
        results['lm_head'] = self.lm_head.verify_quantization()
        return results


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize text (mock)."""
        # Generate random token IDs
        length = min(kwargs.get('max_length', 256), 128)
        input_ids = torch.randint(2, self.vocab_size, (1, length))
        return {'input_ids': input_ids}

    def decode(self, token_ids: torch.Tensor, **kwargs) -> str:
        """Decode tokens (mock)."""
        return f"Generated text with {len(token_ids)} tokens"


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_baking_engine_initialization():
    """Test BakingEngine initialization."""
    print("\n" + "="*70)
    print("TEST 1: BakingEngine Initialization")
    print("="*70)

    try:
        # Import Phase 6 components
        sys.path.insert(0, 'C:/Users/17175/Desktop/_ACTIVE_PROJECTS/the-agent-maker/src')
        from phase6_baking.baking_engine import BakingEngine, BakingConfig

        # Create config
        config = BakingConfig(
            a_cycle_iterations=3,
            b_cycle_iterations=3,
            max_total_iterations=10,
            baking_epochs=2,
            learning_rate=5e-5
        )

        # Initialize engine (no W&B)
        engine = BakingEngine(config=config, use_wandb=False)

        print("  [OK] BakingEngine initialized successfully")
        print(f"  - A-cycle iterations: {config.a_cycle_iterations}")
        print(f"  - B-cycle iterations: {config.b_cycle_iterations}")
        print(f"  - Max total iterations: {config.max_total_iterations}")
        print(f"  - Learning rate: {config.learning_rate}")

        return True, engine, config

    except Exception as e:
        print(f"  [X] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_a_cycle_tool_optimization(model: nn.Module, tokenizer: Any):
    """Test A-Cycle tool optimization."""
    print("\n" + "="*70)
    print("TEST 2: A-Cycle Tool Optimization")
    print("="*70)

    try:
        from phase6_baking.a_cycle_tool import ACycleOptimizer

        # Create A-cycle optimizer
        a_optimizer = ACycleOptimizer(
            tool_prompts=["You are expert at using tools systematically."],
            lora_r=16,
            lora_alpha=32,
            num_epochs=2,
            learning_rate=5e-5
        )

        # Verify quantization before
        before_quant = model.verify_quantization()
        print(f"  Before optimization: {sum(before_quant.values())}/{len(before_quant)} layers quantized")

        # Run optimization
        baked_model, score = a_optimizer.optimize(model, tokenizer, evaluator=None)

        # Verify quantization after
        after_quant = baked_model.verify_quantization()
        print(f"  After optimization: {sum(after_quant.values())}/{len(after_quant)} layers quantized")

        print(f"  [OK] A-Cycle completed successfully")
        print(f"  - Tool score: {score:.3f}")
        print(f"  - 1.58-bit preserved: {all(after_quant.values())}")

        return True, baked_model, score

    except Exception as e:
        print(f"  [X] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, model, 0.0


def test_b_cycle_persona_discovery(model: nn.Module, tokenizer: Any):
    """Test B-Cycle persona discovery."""
    print("\n" + "="*70)
    print("TEST 3: B-Cycle Persona Discovery")
    print("="*70)

    try:
        from phase6_baking.b_cycle_persona import BCycleOptimizer

        # Create B-cycle optimizer
        b_optimizer = BCycleOptimizer(
            persona_prompts=["You are careful and thorough."],
            lora_r=16,
            lora_alpha=32,
            num_epochs=2,
            learning_rate=5e-5
        )

        # Verify quantization before
        before_quant = model.verify_quantization()
        print(f"  Before optimization: {sum(before_quant.values())}/{len(before_quant)} layers quantized")

        # Run optimization
        baked_model, score = b_optimizer.optimize(model, tokenizer, evaluator=None)

        # Verify quantization after
        after_quant = baked_model.verify_quantization()
        print(f"  After optimization: {sum(after_quant.values())}/{len(after_quant)} layers quantized")

        # Get state
        state = b_optimizer.get_state()
        discovered = state.get('discovered_traits', [])

        print(f"  [OK] B-Cycle completed successfully")
        print(f"  - Persona score: {score:.3f}")
        print(f"  - Discovered traits: {discovered[:3]}")
        print(f"  - 1.58-bit preserved: {all(after_quant.values())}")

        return True, baked_model, score

    except Exception as e:
        print(f"  [X] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, model, 0.0


def test_half_baking(original_model: nn.Module, baked_model: nn.Module):
    """Test half-baking gradual integration."""
    print("\n" + "="*70)
    print("TEST 4: Half-Baking Gradual Integration")
    print("="*70)

    try:
        from phase6_baking.half_baking import HalfBaker

        # Create half-baker
        half_baker = HalfBaker(strength=0.5)

        # Verify quantization before
        before_quant = original_model.verify_quantization()
        print(f"  Original model: {sum(before_quant.values())}/{len(before_quant)} layers quantized")

        # Half-bake
        half_baked = half_baker.half_bake(original_model, baked_model)

        # Verify quantization after
        after_quant = half_baked.verify_quantization()
        print(f"  Half-baked model: {sum(after_quant.values())}/{len(after_quant)} layers quantized")

        # Get metrics
        metrics = half_baker.get_metrics()

        print(f"  [OK] Half-baking completed successfully")
        print(f"  - Strength: 0.5")
        print(f"  - Layers interpolated: {metrics['layers_interpolated']}")
        print(f"  - Layers preserved: {metrics['layers_preserved']}")
        print(f"  - 1.58-bit preserved: {all(after_quant.values())}")

        return True, half_baked

    except Exception as e:
        print(f"  [X] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, original_model


def test_prompt_pursuit(model: nn.Module, tokenizer: Any):
    """Test PromptPursuit iterative re-baking."""
    print("\n" + "="*70)
    print("TEST 5: PromptPursuit Iterative Re-Baking")
    print("="*70)

    try:
        from phase6_baking.prompt_pursuit import PromptPursuitOptimizer, PursuitConfig

        # Create pursuit optimizer
        config = PursuitConfig(
            pursuit_rounds=3,
            min_rounds=2,
            max_rounds=3,
            num_epochs=2,
            learning_rate=5e-5
        )
        optimizer = PromptPursuitOptimizer(config)

        # Mock evaluator
        def mock_evaluator(m):
            import random
            return 0.6 + random.random() * 0.2

        # Verify quantization before
        before_quant = model.verify_quantization()
        print(f"  Before pursuit: {sum(before_quant.values())}/{len(before_quant)} layers quantized")

        # Run pursuit
        result = optimizer.pursue(
            model=model,
            prompt="You are expert at systematic problem-solving.",
            tokenizer=tokenizer,
            evaluator=mock_evaluator
        )

        # Verify quantization after
        after_quant = result.final_model.verify_quantization()
        print(f"  After pursuit: {sum(after_quant.values())}/{len(after_quant)} layers quantized")

        print(f"  [OK] Prompt pursuit completed successfully")
        print(f"  - Rounds completed: {result.rounds_completed}")
        print(f"  - Converged: {result.converged}")
        print(f"  - Scores: {[f'{s:.3f}' for s in result.scores_per_round]}")
        print(f"  - 1.58-bit preserved: {all(after_quant.values())}")

        return True, result

    except Exception as e:
        print(f"  [X] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_drift_meter(model: nn.Module, tokenizer: Any):
    """Test DriftMeter persona consistency."""
    print("\n" + "="*70)
    print("TEST 6: DriftMeter Persona Consistency")
    print("="*70)

    try:
        from phase6_baking.drift_meter import PersonaDriftMeter, DriftConfig

        # Create drift meter
        config = DriftConfig(
            num_turns=10,  # Reduced for sandbox
            persona_keywords=["careful", "thorough", "verify"]
        )
        meter = PersonaDriftMeter(config)

        # Verify quantization before
        before_quant = model.verify_quantization()
        print(f"  Before measurement: {sum(before_quant.values())}/{len(before_quant)} layers quantized")

        # Measure drift
        result = meter.measure_drift(
            model=model,
            persona_description="You are a careful and thorough assistant.",
            tokenizer=tokenizer,
            num_turns=10
        )

        # Verify quantization after (shouldn't change - read-only)
        after_quant = model.verify_quantization()
        print(f"  After measurement: {sum(after_quant.values())}/{len(after_quant)} layers quantized")

        print(f"  [OK] Drift measurement completed successfully")
        print(f"  - Turns completed: {result.turns_completed}")
        print(f"  - Avg drift: {result.avg_drift:.4f}")
        print(f"  - Max drift: {result.max_drift:.4f}")
        print(f"  - 1.58-bit preserved: {all(after_quant.values())}")

        return True, result

    except Exception as e:
        print(f"  [X] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_cross_task_validator(base_model: nn.Module, baked_model: nn.Module):
    """Test CrossTaskValidator forgetting detection."""
    print("\n" + "="*70)
    print("TEST 7: CrossTaskValidator Forgetting Detection")
    print("="*70)

    try:
        from phase6_baking.validation import CrossTaskValidator, ValidationConfig

        # Create validator
        config = ValidationConfig(
            max_acceptable_degradation=0.034,  # 3.4%
            min_task_score=0.5
        )
        validator = CrossTaskValidator(config)

        # Mock task evaluators
        def mock_task_a(m): return 0.75
        def mock_task_b(m): return 0.68
        def mock_task_c(m): return 0.82

        tasks = {
            'task_a': mock_task_a,
            'task_b': mock_task_b,
            'task_c': mock_task_c
        }

        # Verify quantization before
        before_base = base_model.verify_quantization()
        before_baked = baked_model.verify_quantization()
        print(f"  Base model: {sum(before_base.values())}/{len(before_base)} layers quantized")
        print(f"  Baked model: {sum(before_baked.values())}/{len(before_baked)} layers quantized")

        # Validate cross-task forgetting
        result = validator.validate_cross_task_forgetting(
            base_model=base_model,
            baked_model=baked_model,
            baked_task='task_a',
            all_tasks=tasks
        )

        # Verify quantization after (shouldn't change - read-only)
        after_base = base_model.verify_quantization()
        after_baked = baked_model.verify_quantization()
        print(f"  After validation base: {sum(after_base.values())}/{len(after_base)} layers quantized")
        print(f"  After validation baked: {sum(after_baked.values())}/{len(after_baked)} layers quantized")

        print(f"  [OK] Cross-task validation completed successfully")
        print(f"  - Tasks passed: {result.tasks_passed}/{len(tasks)}")
        print(f"  - Max degradation: {result.max_degradation*100:.2f}%")
        print(f"  - Avg degradation: {result.avg_degradation*100:.2f}%")
        print(f"  - 1.58-bit preserved: {all(after_base.values()) and all(after_baked.values())}")

        return True, result

    except Exception as e:
        print(f"  [X] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_end_to_end_ab_cycles(model: nn.Module, tokenizer: Any):
    """Test end-to-end A/B cycles."""
    print("\n" + "="*70)
    print("TEST 8: End-to-End A/B Cycles")
    print("="*70)

    try:
        from phase6_baking.baking_engine import BakingEngine, BakingConfig

        # Create config
        config = BakingConfig(
            a_cycle_iterations=2,
            b_cycle_iterations=2,
            max_total_iterations=6,
            baking_epochs=2,
            learning_rate=5e-5,
            plateau_window=2,
            plateau_threshold=0.01
        )

        # Initialize engine (no W&B)
        engine = BakingEngine(config=config, use_wandb=False)

        # Verify quantization before
        before_quant = model.verify_quantization()
        print(f"  Before A/B cycles: {sum(before_quant.values())}/{len(before_quant)} layers quantized")

        # Run A/B cycles
        result = engine.run(
            model=model,
            tokenizer=tokenizer,
            tool_evaluator=None,
            persona_evaluator=None
        )

        # Verify quantization after
        after_quant = result.model.verify_quantization()
        print(f"  After A/B cycles: {sum(after_quant.values())}/{len(after_quant)} layers quantized")

        print(f"  [OK] End-to-end A/B cycles completed successfully")
        print(f"  - Total iterations: {result.total_iterations}")
        print(f"  - A-cycle count: {result.a_cycle_count}")
        print(f"  - B-cycle count: {result.b_cycle_count}")
        print(f"  - Final tool score: {result.final_tool_score:.3f}")
        print(f"  - Final persona score: {result.final_persona_score:.3f}")
        print(f"  - 1.58-bit preserved: {all(after_quant.values())}")

        return True, result

    except Exception as e:
        print(f"  [X] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all Phase 6 sandbox tests."""
    print("\n" + "="*70)
    print("PHASE 6 SANDBOX TEST SUITE")
    print("Tool & Persona Baking with 1.58-bit Model")
    print("="*70)

    # Create mock model and tokenizer
    print("\nCreating mock 1.58-bit model...")
    model = Mock158BitModel(vocab_size=1000, hidden_size=256, num_layers=4)
    tokenizer = MockTokenizer(vocab_size=1000)

    # Verify initial quantization
    initial_quant = model.verify_quantization()
    print(f"  Initial quantization: {sum(initial_quant.values())}/{len(initial_quant)} layers in 1.58-bit")
    print(f"  Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Test results
    results = {
        'baking_engine_init': False,
        'a_cycle_tool': False,
        'b_cycle_persona': False,
        'half_baking': False,
        'prompt_pursuit': False,
        'drift_meter': False,
        'cross_task_validator': False,
        'end_to_end_ab': False
    }

    components_tested = []
    ab_cycles_working = False
    bit_preserved = False
    errors = []

    # Run tests
    try:
        # Test 1: BakingEngine initialization
        success, engine, config = test_baking_engine_initialization()
        results['baking_engine_init'] = success
        if success:
            components_tested.append('BakingEngine')
        else:
            errors.append('BakingEngine initialization failed')

        # Test 2: A-Cycle tool optimization
        success, a_model, a_score = test_a_cycle_tool_optimization(model, tokenizer)
        results['a_cycle_tool'] = success
        if success:
            components_tested.append('ACycleOptimizer')
        else:
            errors.append('A-Cycle optimization failed')

        # Test 3: B-Cycle persona discovery
        success, b_model, b_score = test_b_cycle_persona_discovery(model, tokenizer)
        results['b_cycle_persona'] = success
        if success:
            components_tested.append('BCycleOptimizer')
        else:
            errors.append('B-Cycle optimization failed')

        # Test 4: Half-baking
        if results['a_cycle_tool']:
            success, half_model = test_half_baking(model, a_model)
            results['half_baking'] = success
            if success:
                components_tested.append('HalfBaker')
            else:
                errors.append('Half-baking failed')

        # Test 5: Prompt pursuit
        success, pursuit_result = test_prompt_pursuit(model, tokenizer)
        results['prompt_pursuit'] = success
        if success:
            components_tested.append('PromptPursuit')
        else:
            errors.append('Prompt pursuit failed')

        # Test 6: Drift meter
        success, drift_result = test_drift_meter(model, tokenizer)
        results['drift_meter'] = success
        if success:
            components_tested.append('DriftMeter')
        else:
            errors.append('Drift meter failed')

        # Test 7: Cross-task validator
        if results['a_cycle_tool']:
            success, validation_result = test_cross_task_validator(model, a_model)
            results['cross_task_validator'] = success
            if success:
                components_tested.append('CrossTaskValidator')
            else:
                errors.append('Cross-task validation failed')

        # Test 8: End-to-end A/B cycles
        success, ab_result = test_end_to_end_ab_cycles(model, tokenizer)
        results['end_to_end_ab'] = success
        if success:
            ab_cycles_working = True
            components_tested.append('End-to-End A/B')
        else:
            errors.append('End-to-end A/B cycles failed')

        # Final quantization check
        final_quant = model.verify_quantization()
        bit_preserved = all(final_quant.values())

    except Exception as e:
        errors.append(f'Critical error: {e}')
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SANDBOX TEST SUMMARY")
    print("="*70)

    print(f"\nPhase: 6 (Tool & Persona Baking)")
    print(f"Status: {'PASSED' if all(results.values()) else 'PARTIAL'}")
    print(f"Components Tested: {', '.join(components_tested)}")
    print(f"A/B Cycles Working: {ab_cycles_working}")
    print(f"1.58-bit Preserved: {bit_preserved}")

    print(f"\nTest Results:")
    for test_name, passed in results.items():
        status = '[OK]' if passed else '[X]'
        print(f"  {status} {test_name.replace('_', ' ').title()}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n[OK] All tests passed! Phase 6 sandbox validated.")

    # Return code
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    exit(main())
