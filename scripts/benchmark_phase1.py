#!/usr/bin/env python3
"""
Benchmark Phase 1 Reasoning Model

Evaluates the trained model on:
- GSM8K (Grade School Math) - Primary benchmark for reasoning
- MMLU (Multiple choice knowledge) - Secondary benchmark
- Perplexity on validation set

Usage:
    python scripts/benchmark_phase1.py --checkpoint checkpoints/phase1/reasoning/epoch_10.safetensors
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from cross_phase.utils.checkpoint_utils import load_checkpoint


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Create model with reasoning config
    config = Phase1Config(specialization="reasoning")
    model = TRMTitansMAGModel(config)

    # Load checkpoint
    load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")

    return model, config


def get_tokenizer():
    """Get GPT-2 tokenizer."""
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def benchmark_gsm8k(model, tokenizer, device="cuda", num_samples=100):
    """Evaluate on GSM8K benchmark."""
    print(f"\n{'='*60}")
    print("GSM8K BENCHMARK (Grade School Math)")
    print(f"{'='*60}")

    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main", split="test")
        samples = list(dataset)[:num_samples]
    except Exception as e:
        print(f"Could not load GSM8K: {e}")
        return {"accuracy": 0, "correct": 0, "total": 0}

    import re

    def extract_answer(text):
        # Try #### format
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
        if match:
            return float(match.group(1))
        # Try last number
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return float(numbers[-1])
        return None

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            question = sample["question"]
            gold_text = sample["answer"]
            gold_answer = extract_answer(gold_text)

            if gold_answer is None:
                continue

            # Format prompt
            prompt = f"Question: {question}\nAnswer: Let me solve this step by step.\n"

            # Tokenize - only use input_ids
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)

            # Generate (simple forward pass for speed)
            try:
                outputs = model(input_ids=input_ids)
                logits = outputs["logits"]

                # Sample next few tokens (greedy)
                generated_ids = input_ids.clone()
                for _ in range(50):
                    next_logits = logits[0, -1, :]
                    next_token = next_logits.argmax()
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

                    # Get next logits
                    outputs = model(input_ids=generated_ids)
                    logits = outputs["logits"]

                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                pred_answer = extract_answer(generated_text)

                if pred_answer is not None and abs(pred_answer - gold_answer) < 0.01:
                    correct += 1

            except Exception as e:
                print(f"  Error: {e}")

            total += 1

            if (i + 1) % 10 == 0:
                acc = correct / total if total > 0 else 0
                print(f"  Progress: {i+1}/{num_samples}, Accuracy: {acc:.1%}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nGSM8K Results: {correct}/{total} = {accuracy:.1%}")

    return {"accuracy": accuracy, "correct": correct, "total": total}


def benchmark_perplexity(model, tokenizer, device="cuda", num_samples=100):
    """Calculate perplexity on validation samples."""
    print(f"\n{'='*60}")
    print("PERPLEXITY BENCHMARK")
    print(f"{'='*60}")

    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t for t in dataset["text"] if len(t.strip()) > 100][:num_samples]
    except Exception as e:
        print(f"Could not load wikitext: {e}")
        return {"perplexity": float("inf")}

    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)

            try:
                # Model expects input_ids and labels
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs["loss"]

                num_tokens = input_ids.numel()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

            except Exception as e:
                print(f"  Error: {e}")

            if (i + 1) % 20 == 0:
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                ppl = torch.exp(torch.tensor(avg_loss)).item()
                print(f"  Progress: {i+1}/{num_samples}, Perplexity: {ppl:.2f}")

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"\nPerplexity: {perplexity:.2f}")

    return {"perplexity": perplexity, "avg_loss": avg_loss}


def benchmark_inference_speed(model, tokenizer, device="cuda", num_runs=50):
    """Measure inference speed."""
    print(f"\n{'='*60}")
    print("INFERENCE SPEED BENCHMARK")
    print(f"{'='*60}")

    # Warmup
    dummy_input = tokenizer("Hello world", return_tensors="pt")
    input_ids = dummy_input["input_ids"].to(device)

    for _ in range(5):
        with torch.no_grad():
            model(input_ids=input_ids)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for benchmarking."
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            model(input_ids=input_ids)

        if device == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times) * 1000  # ms
    tokens_per_sec = input_ids.numel() / (avg_time / 1000)

    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Tokens per second: {tokens_per_sec:.0f}")

    return {"avg_time_ms": avg_time, "tokens_per_sec": tokens_per_sec}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Phase 1 Model")
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/phase1/reasoning/epoch_10.safetensors",
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gsm8k-samples", type=int, default=100, help="Number of GSM8K samples")
    parser.add_argument("--perplexity-samples", type=int, default=100, help="Number of perplexity samples")

    args = parser.parse_args()

    print("="*60)
    print("PHASE 1 MODEL BENCHMARKS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # Load model and tokenizer
    model, config = load_model(args.checkpoint, args.device)
    tokenizer = get_tokenizer()

    results = {}

    # Run benchmarks
    results["gsm8k"] = benchmark_gsm8k(model, tokenizer, args.device, args.gsm8k_samples)
    results["perplexity"] = benchmark_perplexity(model, tokenizer, args.device, args.perplexity_samples)
    results["speed"] = benchmark_inference_speed(model, tokenizer, args.device)

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"GSM8K Accuracy:     {results['gsm8k']['accuracy']:.1%}")
    print(f"Perplexity:         {results['perplexity']['perplexity']:.2f}")
    print(f"Inference Speed:    {results['speed']['tokens_per_sec']:.0f} tokens/sec")
    print(f"Inference Time:     {results['speed']['avg_time_ms']:.2f} ms")

    return results


if __name__ == "__main__":
    main()
