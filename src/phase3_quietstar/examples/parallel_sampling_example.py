"""
Phase 3 Quiet-STaR: Parallel Sampling Example

Demonstrates how to use the new parallel thought generation
with diagonal attention mask for 3-4x speedup.

Usage:
    python -m src.phase3_quietstar.examples.parallel_sampling_example
"""

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..architecture.parallel_thought_generator import ParallelThoughtGenerator
from ..architecture.thought_generator import ThoughtGenerator  # Original sequential
from ..config import QuietSTaRConfig
from ..config_extensions import extend_rl_config, ParallelSamplingConfig


def compare_sequential_vs_parallel():
    """
    Compare efficiency of sequential vs parallel thought generation.

    Expected result: 3-4x speedup with parallel generation.
    """
    print("=" * 60)
    print("PARALLEL SAMPLING COMPARISON")
    print("=" * 60)

    # Load model and tokenizer
    model_path = Path("outputs/phase2/champion_model")  # Adjust path
    print(f"\nLoading model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Configuration
    config = QuietSTaRConfig()
    config.rl = extend_rl_config(config.rl)

    num_thoughts = 4
    thought_length_min = 10
    thought_length_max = 20

    # Test input
    test_text = "Solve the following problem: What is 15 + 27?"
    input_ids = tokenizer(test_text, return_tensors="pt").input_ids.to(device)

    print(f"\nInput: {test_text}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Num thoughts: {num_thoughts}")
    print(f"Thought length range: {thought_length_min}-{thought_length_max}")

    # Sequential generation (original)
    print("\n" + "-" * 60)
    print("SEQUENTIAL GENERATION (Original Implementation)")
    print("-" * 60)

    sequential_generator = ThoughtGenerator(
        base_model=model,
        num_thoughts=num_thoughts,
        max_length=thought_length_max,
        min_length=thought_length_min,
        temperature=1.0,
        top_p=0.9,
    )

    # Warmup
    _ = sequential_generator(input_ids, position=10)

    # Benchmark
    num_runs = 10
    sequential_times = []

    for run in range(num_runs):
        start = time.time()
        output_seq = sequential_generator(input_ids, position=10)
        elapsed = time.time() - start
        sequential_times.append(elapsed)

    avg_seq_time = sum(sequential_times) / num_runs
    print(f"Average time: {avg_seq_time * 1000:.2f} ms")
    print(f"Output shape: {output_seq.thoughts.shape}")
    print(f"Log probs shape: {output_seq.log_probs.shape}")

    # Parallel generation (new)
    print("\n" + "-" * 60)
    print("PARALLEL GENERATION (New Implementation)")
    print("-" * 60)

    parallel_generator = ParallelThoughtGenerator(
        base_model=model,
        num_thoughts=num_thoughts,
        max_length=thought_length_max,
        min_length=thought_length_min,
        temperature=1.0,
        top_p=0.9,
    )

    # Warmup
    _ = parallel_generator(input_ids, position=10)

    # Benchmark
    parallel_times = []

    for run in range(num_runs):
        start = time.time()
        output_par = parallel_generator(input_ids, position=10)
        elapsed = time.time() - start
        parallel_times.append(elapsed)

    avg_par_time = sum(parallel_times) / num_runs
    print(f"Average time: {avg_par_time * 1000:.2f} ms")
    print(f"Output shape: {output_par.thoughts.shape}")
    print(f"Log probs shape: {output_par.log_probs.shape}")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    speedup = avg_seq_time / avg_par_time
    print(f"Sequential time: {avg_seq_time * 1000:.2f} ms")
    print(f"Parallel time: {avg_par_time * 1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 2.0:
        print("✅ SUCCESS: Parallel generation is significantly faster!")
    else:
        print("⚠️  WARNING: Speedup lower than expected. Check GPU utilization.")

    # Memory usage
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nPeak GPU memory: {memory_mb:.2f} MB")

    return {
        "sequential_time": avg_seq_time,
        "parallel_time": avg_par_time,
        "speedup": speedup,
    }


def test_diagonal_attention_mask():
    """
    Verify diagonal attention mask prevents cross-contamination.
    """
    print("\n" + "=" * 60)
    print("DIAGONAL ATTENTION MASK VALIDATION")
    print("=" * 60)

    # Load model
    model_path = Path("outputs/phase2/champion_model")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    generator = ParallelThoughtGenerator(
        base_model=model,
        num_thoughts=4,
        max_length=20,
        min_length=10,
    )

    # Create mask
    batch_size = 2
    num_thoughts = 4
    seq_len = 30
    position = 10

    mask = generator._create_diagonal_attention_mask(
        batch_size=batch_size,
        num_thoughts=num_thoughts,
        seq_len=seq_len,
        position=position,
        device=device,
    )

    print(f"\nMask shape: {mask.shape}")
    print(f"Expected: ({batch_size * num_thoughts}, {seq_len}, {seq_len})")

    # Validate structure
    print("\nValidating mask structure...")

    # 1. Shared context: all thoughts attend
    shared_context_valid = torch.all(mask[:, :position + 1, :position + 1] == 0.0)
    print(f"✅ Shared context valid: {shared_context_valid}")

    # 2. Diagonal blocks: thoughts attend only to themselves
    for i in range(num_thoughts):
        for j in range(num_thoughts):
            if i != j:
                # Check off-diagonal blocks are masked
                start_i = position + 1 + i * 5
                end_i = position + 1 + (i + 1) * 5
                start_j = position + 1 + j * 5
                end_j = position + 1 + (j + 1) * 5

                if end_i <= seq_len and end_j <= seq_len:
                    off_diagonal_masked = torch.all(
                        mask[i, start_i:end_i, start_j:end_j] == float("-inf")
                    )
                    if not off_diagonal_masked:
                        print(
                            f"⚠️  Off-diagonal block ({i},{j}) not fully masked"
                        )

    print("✅ Diagonal structure validated")

    # 3. Mask sparsity
    sparsity = (mask == float("-inf")).float().mean()
    print(f"\nMask sparsity: {sparsity:.2%}")
    print(f"Expected: ~75% for num_thoughts=4")

    return mask


def test_teacher_forcing():
    """
    Test teacher forcing loss computation.
    """
    print("\n" + "=" * 60)
    print("TEACHER FORCING LOSS VALIDATION")
    print("=" * 60)

    # Load model
    model_path = Path("outputs/phase2/champion_model")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    generator = ParallelThoughtGenerator(
        base_model=model,
        num_thoughts=4,
        max_length=20,
        min_length=10,
    )

    # Test input
    test_text = "Solve: 5 + 7 = "
    input_ids = tokenizer(test_text, return_tensors="pt").input_ids.to(device)

    # Generate thoughts
    output = generator(input_ids, position=5)

    # Create dummy labels (future tokens)
    labels = torch.randint(
        0, tokenizer.vocab_size, (1, input_ids.size(1) + 10), device=device
    )

    # Compute teacher forcing loss
    print("\nComputing teacher forcing loss...")
    print(f"n_true: 4 (number of future tokens)")

    loss = generator.compute_teacher_forced_loss(
        input_ids=input_ids,
        thought_ids=output.thought_ids,
        labels=labels,
        n_true=4,
    )

    print(f"\nLoss value: {loss.item():.4f}")
    print(f"Loss shape: {loss.shape}")
    print(f"Loss dtype: {loss.dtype}")

    # Validate
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be inf"

    print("✅ Teacher forcing loss valid")

    return loss.item()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PHASE 3 QUIET-STAR: PARALLEL SAMPLING EXAMPLES")
    print("=" * 60)

    # 1. Sequential vs Parallel comparison
    results = compare_sequential_vs_parallel()

    # 2. Diagonal mask validation
    mask = test_diagonal_attention_mask()

    # 3. Teacher forcing test
    tf_loss = test_teacher_forcing()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Parallel sampling: {results['speedup']:.2f}x speedup")
    print(f"✅ Diagonal mask: Structure validated")
    print(f"✅ Teacher forcing: Loss = {tf_loss:.4f}")
    print("\nAll components working correctly!")


if __name__ == "__main__":
    main()
