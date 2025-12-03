"""
Phase 3 Quiet-STaR Sandbox Test

Comprehensive sandbox testing for Phase 3 components:
- ThoughtGenerator: Generate parallel thoughts at token positions
- CoherenceScorer: Score thought quality (semantic, syntactic, predictive)
- MixingHead: Attention-based integration of thoughts
- ThoughtInjector: Identify difficult positions for thought injection
- Anti-Theater Detection: Validate genuine reasoning vs memorization

This test uses mock models to validate architecture without training.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional


# Mock base model for testing
class MockBaseModel(nn.Module):
    """Mock language model for sandbox testing."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Simple embedding + linear output
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> Any:
        """Forward pass returning logits and hidden states."""
        # Embed tokens
        hidden = self.embedding(input_ids)

        # Generate logits
        logits = self.output(hidden)

        # Mock outputs object
        @dataclass
        class MockOutput:
            logits: torch.Tensor
            last_hidden_state: torch.Tensor

        return MockOutput(logits=logits, last_hidden_state=hidden)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20,
                 do_sample: bool = False, num_return_sequences: int = 1) -> torch.Tensor:
        """Mock generation for anti-theater testing."""
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Simple greedy generation
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            next_token_logits = outputs.logits[:, -1, :]

            if do_sample:
                next_token = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1),
                    num_samples=1
                )
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated


def test_thought_generator():
    """Test ThoughtGenerator component."""
    print("\n" + "="*80)
    print("TEST 1: ThoughtGenerator - Parallel thought generation")
    print("="*80)

    from phase3_quietstar.architecture.thought_generator import ThoughtGenerator

    # Setup
    base_model = MockBaseModel(vocab_size=1000, hidden_size=256)
    generator = ThoughtGenerator(
        base_model=base_model,
        num_thoughts=4,
        max_length=20,
        min_length=10,
        temperature=1.0,
        top_p=0.9
    )

    # Create input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    position = 5

    print(f"Input shape: {input_ids.shape}")
    print(f"Generating thoughts at position: {position}")
    print(f"Num thoughts: {generator.num_thoughts}")

    # Generate thoughts
    try:
        output = generator(input_ids, position)

        print(f"\n[PASS] ThoughtGenerator Output:")
        print(f"  - Thoughts shape: {output.thoughts.shape}")
        print(f"  - Expected: (batch={batch_size}, num_thoughts={generator.num_thoughts}, thought_len=10-20, hidden={base_model.hidden_size})")
        print(f"  - Log probs shape: {output.log_probs.shape}")
        print(f"  - Num thought sequences: {len(output.thought_ids)}")
        print(f"  - First thought length: {len(output.thought_ids[0])}")

        # Validate shapes
        assert output.thoughts.size(0) == batch_size, f"Wrong batch size: {output.thoughts.size(0)} != {batch_size}"
        assert output.thoughts.size(1) == generator.num_thoughts, f"Wrong num_thoughts: {output.thoughts.size(1)} != {generator.num_thoughts}"
        assert output.thoughts.size(-1) == base_model.hidden_size, f"Wrong hidden_size: {output.thoughts.size(-1)} != {base_model.hidden_size}"
        assert len(output.thought_ids) == generator.num_thoughts, f"Wrong num thought IDs: {len(output.thought_ids)} != {generator.num_thoughts}"

        print(f"\n[PASS] All shape validations passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] ThoughtGenerator FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coherence_scorer():
    """Test CoherenceScorer component."""
    print("\n" + "="*80)
    print("TEST 2: CoherenceScorer - 3-dimensional scoring")
    print("="*80)

    from phase3_quietstar.architecture.coherence_scorer import CoherenceScorer

    # Setup
    hidden_size = 256
    scorer = CoherenceScorer(
        hidden_size=hidden_size,
        weights={"semantic": 0.4, "syntactic": 0.3, "predictive": 0.3}
    )

    # Create inputs
    batch_size = 2
    num_thoughts = 4
    thought_len = 15
    vocab_size = 1000

    base_hidden = torch.randn(batch_size, hidden_size)
    thought_hiddens = torch.randn(batch_size, num_thoughts, thought_len, hidden_size)
    next_token_logits = torch.randn(batch_size, vocab_size)

    print(f"Base hidden shape: {base_hidden.shape}")
    print(f"Thought hiddens shape: {thought_hiddens.shape}")
    print(f"Next token logits shape: {next_token_logits.shape}")
    print(f"Scoring weights: {scorer.weights}")

    # Score thoughts
    try:
        scores = scorer(base_hidden, thought_hiddens, next_token_logits)

        print(f"\n[PASS] CoherenceScorer Output:")
        print(f"  - Semantic scores: {scores.semantic.shape}")
        print(f"  - Syntactic scores: {scores.syntactic.shape}")
        print(f"  - Predictive scores: {scores.predictive.shape}")
        print(f"  - Composite scores: {scores.composite.shape}")
        print(f"  - Expected shape: (batch={batch_size}, num_thoughts={num_thoughts})")

        # Validate shapes
        assert scores.semantic.shape == (batch_size, num_thoughts), f"Wrong semantic shape: {scores.semantic.shape}"
        assert scores.syntactic.shape == (batch_size, num_thoughts), f"Wrong syntactic shape: {scores.syntactic.shape}"
        assert scores.predictive.shape == (batch_size, num_thoughts), f"Wrong predictive shape: {scores.predictive.shape}"
        assert scores.composite.shape == (batch_size, num_thoughts), f"Wrong composite shape: {scores.composite.shape}"

        # Validate ranges [0, 1]
        assert (scores.semantic >= 0).all() and (scores.semantic <= 1).all(), "Semantic scores out of range"
        assert (scores.syntactic >= 0).all() and (scores.syntactic <= 1).all(), "Syntactic scores out of range"
        assert (scores.predictive >= 0).all() and (scores.predictive <= 1).all(), "Predictive scores out of range"
        assert (scores.composite >= 0).all() and (scores.composite <= 1).all(), "Composite scores out of range"

        print(f"\n[PASS] Sample scores (first batch, all thoughts):")
        print(f"  - Semantic: {scores.semantic[0].tolist()}")
        print(f"  - Syntactic: {scores.syntactic[0].tolist()}")
        print(f"  - Predictive: {scores.predictive[0].tolist()}")
        print(f"  - Composite: {scores.composite[0].tolist()}")

        print(f"\n[PASS] All validations passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] CoherenceScorer FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixing_head():
    """Test MixingHead component."""
    print("\n" + "="*80)
    print("TEST 3: MixingHead - Attention-based thought integration")
    print("="*80)

    from phase3_quietstar.architecture.mixing_head import MixingHead

    # Setup
    hidden_size = 256
    num_heads = 8
    mixer = MixingHead(hidden_size=hidden_size, num_heads=num_heads, dropout=0.1)

    # Create inputs
    batch_size = 2
    num_thoughts = 4

    base_hidden = torch.randn(batch_size, hidden_size)
    thought_hiddens = torch.randn(batch_size, num_thoughts, hidden_size)
    coherence_scores = torch.rand(batch_size, num_thoughts)  # [0, 1]

    print(f"Base hidden shape: {base_hidden.shape}")
    print(f"Thought hiddens shape: {thought_hiddens.shape}")
    print(f"Coherence scores shape: {coherence_scores.shape}")
    print(f"Num attention heads: {num_heads}")

    # Mix thoughts
    try:
        mixer.eval()  # Disable dropout for deterministic test
        mixed_output = mixer(base_hidden, thought_hiddens, coherence_scores)

        print(f"\n[PASS] MixingHead Output:")
        print(f"  - Mixed hidden shape: {mixed_output.shape}")
        print(f"  - Expected: (batch={batch_size}, hidden_size={hidden_size})")

        # Validate shape
        assert mixed_output.shape == (batch_size, hidden_size), f"Wrong output shape: {mixed_output.shape}"

        # Validate not NaN/Inf
        assert not torch.isnan(mixed_output).any(), "Output contains NaN"
        assert not torch.isinf(mixed_output).any(), "Output contains Inf"

        # Check gating mechanism (output should be blend of base + thoughts)
        # Not exactly equal to base or thoughts
        base_diff = (mixed_output - base_hidden).abs().mean().item()
        print(f"\n[PASS] Average difference from base: {base_diff:.4f}")
        print(f"   (Should be >0, indicating thoughts were integrated)")

        assert base_diff > 0, "Mixed output identical to base (no integration)"

        print(f"\n[PASS] All validations passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] MixingHead FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thought_injector():
    """Test ThoughtInjector component."""
    print("\n" + "="*80)
    print("TEST 4: ThoughtInjector - Difficulty-based injection")
    print("="*80)

    from phase3_quietstar.architecture.thought_injector import ThoughtInjector

    # Setup
    injector = ThoughtInjector(threshold=0.6, min_interval=3)

    vocab_size = 1000
    batch_size = 2

    print(f"Injection threshold: {injector.threshold}")
    print(f"Min interval: {injector.min_interval}")

    # Test 1: High difficulty (should inject)
    print(f"\n--- Test Case 1: High Difficulty ---")
    high_difficulty_logits = torch.randn(batch_size, vocab_size) * 0.1  # Low confidence (high entropy)
    attention = torch.rand(batch_size, 10) / 10  # Dispersed attention
    loss = torch.tensor(5.0)  # High loss

    should_inject = injector(high_difficulty_logits, attention, loss, position=5)
    print(f"  High difficulty -> Should inject: {should_inject}")
    assert should_inject == True, "Failed to inject on high difficulty"

    # Test 2: Low difficulty (should NOT inject)
    print(f"\n--- Test Case 2: Low Difficulty ---")
    low_difficulty_logits = torch.zeros(batch_size, vocab_size)
    low_difficulty_logits[:, 0] = 10.0  # Very confident (low entropy)
    attention = torch.zeros(batch_size, 10)
    attention[:, 0] = 1.0  # Focused attention
    loss = torch.tensor(0.1)  # Low loss

    should_inject = injector(low_difficulty_logits, attention, loss, position=10)
    print(f"  Low difficulty -> Should inject: {should_inject}")
    assert should_inject == False, "Incorrectly injected on low difficulty"

    # Test 3: Interval enforcement (should NOT inject)
    print(f"\n--- Test Case 3: Interval Enforcement ---")
    injector.last_injection = 9  # Recent injection at position 9
    should_inject = injector(high_difficulty_logits, attention, loss, position=10)
    print(f"  High difficulty but recent injection -> Should inject: {should_inject}")
    assert should_inject == False, "Failed to enforce min interval"

    print(f"\n[PASS] All injection decisions correct!")
    return True


def test_anti_theater_detection():
    """Test Anti-Theater Detection."""
    print("\n" + "="*80)
    print("TEST 5: Anti-Theater Detection - Validate genuine reasoning")
    print("="*80)

    from phase3_quietstar.anti_theater import AntiTheaterValidator
    from phase3_quietstar.config import AntiTheaterConfig

    print("Testing anti-theater detection mechanisms...")
    print("(Using mock model - full validation requires trained model)")

    # Mock QuietSTaR model for testing
    class MockQuietSTaRModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids, labels=None, use_thoughts=True):
            outputs = self.base_model(input_ids)
            result = {
                "logits": outputs.logits,
                "avg_coherence": 0.7 if use_thoughts else 0.0
            }
            if labels is not None:
                result["loss"] = torch.tensor(1.0)
            return result

        def eval(self):
            self.base_model.eval()
            return self

    # Mock tokenizer
    class MockTokenizer:
        pass

    # Mock dataloader
    class MockDataLoader:
        def __init__(self, num_batches=3):
            self.num_batches = num_batches

        def __iter__(self):
            for _ in range(self.num_batches):
                yield {
                    "input_ids": torch.randint(0, 1000, (2, 20)),
                    "labels": torch.randint(0, 1000, (2, 20))
                }

    # Setup
    base_model = MockBaseModel(vocab_size=1000, hidden_size=256)
    model = MockQuietSTaRModel(base_model)
    tokenizer = MockTokenizer()
    dataloader = MockDataLoader(num_batches=3)
    config = AntiTheaterConfig()

    print(f"\nAnti-Theater Thresholds:")
    print(f"  - Divergence: >{config.divergence_threshold}")
    print(f"  - Ablation: >{config.ablation_threshold}")
    print(f"  - Correlation: >{config.correlation_threshold}")

    try:
        validator = AntiTheaterValidator(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device="cpu"
        )

        print(f"\n[PASS] AntiTheaterValidator initialized successfully")

        # Test individual methods (with reduced samples for speed)
        print(f"\n--- Testing Divergence Detection ---")
        input_ids = torch.randint(0, 1000, (5, 20))
        divergence = validator.divergence_test(input_ids, num_samples=5)
        print(f"  Divergence score: {divergence:.3f} (threshold: {config.divergence_threshold})")

        print(f"\n--- Testing Ablation Study ---")
        ablation = validator.ablation_test(dataloader, max_batches=3)
        print(f"  Ablation improvement: {ablation:.4f} (threshold: {config.ablation_threshold})")

        print(f"\n--- Testing Correlation ---")
        correlation = validator.correlation_test(dataloader, max_batches=3)
        print(f"  Correlation: {correlation:.3f} (threshold: {config.correlation_threshold})")

        print(f"\n[PASS] Anti-theater detection methods executed successfully!")
        print(f"\nNOTE: With mock model, scores are not meaningful.")
        print(f"      Real validation requires trained Quiet-STaR model.")

        return True

    except Exception as e:
        print(f"\n[FAIL] Anti-Theater Detection FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test full Quiet-STaR forward pass integration."""
    print("\n" + "="*80)
    print("TEST 6: Full Integration - Complete Quiet-STaR forward pass")
    print("="*80)

    from phase3_quietstar.architecture.thought_generator import ThoughtGenerator
    from phase3_quietstar.architecture.coherence_scorer import CoherenceScorer
    from phase3_quietstar.architecture.mixing_head import MixingHead
    from phase3_quietstar.architecture.thought_injector import ThoughtInjector

    # Setup all components
    hidden_size = 256
    vocab_size = 1000

    base_model = MockBaseModel(vocab_size=vocab_size, hidden_size=hidden_size)
    generator = ThoughtGenerator(base_model=base_model, num_thoughts=4)
    scorer = CoherenceScorer(hidden_size=hidden_size)
    mixer = MixingHead(hidden_size=hidden_size, num_heads=8)
    injector = ThoughtInjector(threshold=0.6)

    # Create input
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"Input shape: {input_ids.shape}")
    print(f"Testing full forward pass at multiple positions...")

    try:
        mixer.eval()
        base_model.eval()

        num_injections = 0

        # Simulate forward pass through sequence
        for position in range(5, seq_len - 5):
            # Get base output
            base_output = base_model(input_ids[:, :position+1])
            base_hidden = base_output.last_hidden_state[:, -1, :]  # Last position
            base_logits = base_output.logits[:, -1, :]

            # Check if injection needed
            should_inject = injector(base_logits, None, torch.tensor(1.0), position)

            if should_inject:
                num_injections += 1

                # Generate thoughts
                thought_output = generator(input_ids, position)

                # Average thought representations (batch, num_thoughts, thought_len, hidden) -> (batch, num_thoughts, hidden)
                thought_avg = thought_output.thoughts.mean(dim=2)

                # Score thoughts
                scores = scorer(base_hidden, thought_output.thoughts, base_logits)

                # Mix thoughts
                mixed_hidden = mixer(base_hidden, thought_avg, scores.composite)

                # Validate shapes
                assert mixed_hidden.shape == (batch_size, hidden_size), f"Wrong mixed shape at pos {position}"

        print(f"\n[PASS] Full Integration Results:")
        print(f"  - Sequence length: {seq_len}")
        print(f"  - Positions checked: {seq_len - 10}")
        print(f"  - Thoughts injected: {num_injections}")
        print(f"  - Injection rate: {num_injections / (seq_len - 10) * 100:.1f}%")
        print(f"\n[PASS] All components integrated successfully!")

        return True

    except Exception as e:
        print(f"\n[FAIL] Full Integration FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 3 sandbox tests."""
    print("\n" + "="*80)
    print("PHASE 3 QUIET-STAR SANDBOX TEST SUITE")
    print("="*80)
    print("\nTesting Phase 3 Quiet-STaR components in isolated sandbox environment.")
    print("Using mock models to validate architecture without training.\n")

    results = {
        "ThoughtGenerator": False,
        "CoherenceScorer": False,
        "MixingHead": False,
        "ThoughtInjector": False,
        "Anti-Theater Detection": False,
        "Full Integration": False
    }

    # Run tests
    results["ThoughtGenerator"] = test_thought_generator()
    results["CoherenceScorer"] = test_coherence_scorer()
    results["MixingHead"] = test_mixing_head()
    results["ThoughtInjector"] = test_thought_injector()
    results["Anti-Theater Detection"] = test_anti_theater_detection()
    results["Full Integration"] = test_full_integration()

    # Summary
    print("\n" + "="*80)
    print("PHASE 3 SANDBOX TEST SUMMARY")
    print("="*80)

    for component, passed in results.items():
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"{component:30s} : {status}")

    total = len(results)
    passed = sum(results.values())

    print(f"\n{'='*80}")
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*80}\n")

    # Return values for automated testing
    return {
        "phase": "Phase 3 (Quiet-STaR)",
        "status": "PASS" if all(results.values()) else "PARTIAL" if any(results.values()) else "FAIL",
        "components_tested": list(results.keys()),
        "components_passed": [k for k, v in results.items() if v],
        "components_failed": [k for k, v in results.items() if not v],
        "anti_theater_status": "TESTED (mock model)" if results["Anti-Theater Detection"] else "FAILED",
        "errors": [k for k, v in results.items() if not v]
    }


if __name__ == "__main__":
    result = main()

    # Print structured output
    print("\n" + "="*80)
    print("STRUCTURED OUTPUT FOR AUTOMATION")
    print("="*80)
    print(f"Phase: {result['phase']}")
    print(f"Status: {result['status']}")
    print(f"Components Tested: {', '.join(result['components_tested'])}")
    print(f"Anti-Theater Status: {result['anti_theater_status']}")
    if result['errors']:
        print(f"Errors: {', '.join(result['errors'])}")
    else:
        print("Errors: None")
    print("="*80 + "\n")

    # Exit with appropriate code
    import sys
    sys.exit(0 if result['status'] == 'PASS' else 1)
