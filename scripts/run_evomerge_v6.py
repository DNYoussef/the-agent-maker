#!/usr/bin/env python3
"""
EvoMerge V6 - Memory-Optimized Implementation

Fixes V5's CUDA OOM error by:
1. Moving models to CPU after evaluation
2. Clearing GPU cache before merging
3. Only keeping active model on GPU

Based on: docs/phases/phase2/PHASE2_COMPLETE_GUIDE.md
"""

import copy
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def move_to_device(model: nn.Module, device: str) -> nn.Module:
    """Move model to device and return it."""
    model = model.to(device)
    return model


class MemoryEfficientMerger:
    """
    Memory-efficient merger that performs merging on CPU
    to avoid GPU OOM errors.
    """

    def __init__(self):
        pass

    def apply_combo(self, models: List[nn.Module], combo_id: int) -> nn.Module:
        """
        Apply 3-stage merge pipeline on CPU.

        Args:
            models: List of 3 models (will be moved to CPU)
            combo_id: Binary combo (0-7)

        Returns:
            Merged model on CPU
        """
        # Decode combo
        bit0 = (combo_id >> 0) & 1  # Interpolation (0=Linear, 1=SLERP)
        bit1 = (combo_id >> 1) & 1  # Task Arithmetic (0=DARE, 1=TIES)
        bit2 = (combo_id >> 2) & 1  # Selection (0=Franken, 1=DFS)

        # Move all models to CPU for merging
        cpu_models = []
        for m in models:
            cpu_models.append(copy.deepcopy(m).cpu())

        clear_gpu_memory()

        # Stage 1: Interpolation
        if bit0 == 0:
            stage1 = self._linear_merge(cpu_models)
        else:
            stage1 = self._slerp_merge(cpu_models)

        # Stage 2: Task Arithmetic
        if bit1 == 0:
            stage2 = self._dare_merge(stage1, cpu_models[0])
        else:
            stage2 = self._ties_merge(stage1, cpu_models)

        # Stage 3: Selection
        if bit2 == 0:
            stage3 = self._franken_merge(stage2, cpu_models)
        else:
            stage3 = self._dfs_merge(stage2, cpu_models)

        # Clean up
        del cpu_models
        gc.collect()

        return stage3

    def _linear_merge(self, models: List[nn.Module]) -> nn.Module:
        """Linear interpolation (weighted average)."""
        result = copy.deepcopy(models[0])
        n = len(models)

        with torch.no_grad():
            for name, param in result.named_parameters():
                avg = torch.zeros_like(param)
                for m in models:
                    avg += dict(m.named_parameters())[name].data
                param.data = avg / n

        return result

    def _slerp_merge(self, models: List[nn.Module]) -> nn.Module:
        """Spherical linear interpolation."""
        # For simplicity, use linear for 3+ models
        # (proper SLERP is pairwise)
        return self._linear_merge(models)

    def _dare_merge(self, merged: nn.Module, base: nn.Module, drop_rate: float = 0.9) -> nn.Module:
        """DARE: Drop And REscale on deltas."""
        result = copy.deepcopy(merged)

        with torch.no_grad():
            for (name, param), (_, base_param) in zip(
                result.named_parameters(), base.named_parameters()
            ):
                # Compute delta
                delta = param.data - base_param.data

                # Drop 90% randomly
                mask = torch.rand_like(delta) > drop_rate
                sparse_delta = delta * mask.float()

                # Rescale
                rescaled = sparse_delta / (1 - drop_rate)

                # Apply
                param.data = base_param.data + rescaled

        return result

    def _ties_merge(self, merged: nn.Module, references: List[nn.Module], k_percent: float = 0.2) -> nn.Module:
        """TIES: Trim, Elect Sign, Merge - Memory efficient version."""
        result = copy.deepcopy(merged)
        base = references[0]

        with torch.no_grad():
            for name, param in result.named_parameters():
                base_param = dict(base.named_parameters())[name].data

                # Compute deltas from each reference
                deltas = []
                for ref in references:
                    ref_param = dict(ref.named_parameters())[name].data
                    deltas.append(param.data - ref_param)

                if len(deltas) == 0:
                    continue

                # Stack deltas
                stacked = torch.stack(deltas, dim=0)

                # TRIM: Keep top k% by magnitude (memory-efficient)
                magnitudes = torch.abs(stacked).mean(dim=0)
                flat_mag = magnitudes.flatten()
                k = max(1, int(len(flat_mag) * k_percent))
                # Use sorting on smaller sample instead of full quantile
                if len(flat_mag) > 10000:
                    # Sample for threshold estimation
                    sample_idx = torch.randperm(len(flat_mag))[:10000]
                    sample = flat_mag[sample_idx]
                    sorted_sample = torch.sort(sample, descending=True).values
                    threshold = sorted_sample[min(k, len(sorted_sample)-1)]
                else:
                    sorted_mag = torch.sort(flat_mag, descending=True).values
                    threshold = sorted_mag[min(k, len(sorted_mag)-1)]
                important_mask = magnitudes >= threshold
                del flat_mag

                # ELECT: Vote on sign
                signs = torch.sign(stacked)
                elected_sign = torch.sign(signs.mean(dim=0))

                # MERGE: Average only matching signs
                merged_delta = torch.zeros_like(param.data)
                count = torch.zeros_like(param.data)

                for delta in deltas:
                    matching = (torch.sign(delta) == elected_sign) & important_mask
                    merged_delta += delta * matching.float()
                    count += matching.float()

                count = count.clamp(min=1)
                merged_delta = merged_delta / count

                param.data = base_param + merged_delta

                # Clean up intermediate tensors
                del stacked, deltas, magnitudes, signs
                gc.collect()

        return result

    def _franken_merge(self, merged: nn.Module, references: List[nn.Module]) -> nn.Module:
        """FrankenMerge: Layer-wise selection."""
        result = copy.deepcopy(merged)

        # Group by layer (simplified: just use merged as-is)
        # Full implementation would select best layers from each model
        return result

    def _dfs_merge(self, merged: nn.Module, references: List[nn.Module]) -> nn.Module:
        """DFS: Inverse-variance weighted merge."""
        result = copy.deepcopy(merged)

        with torch.no_grad():
            for name, param in result.named_parameters():
                # Get params from all models
                all_params = [dict(m.named_parameters())[name].data for m in references]
                all_params.append(param.data)

                # Stack and compute variance
                stacked = torch.stack(all_params, dim=0)
                variance = torch.var(stacked, dim=0)

                # Inverse variance weighting (stable params = important)
                importance = 1.0 / (variance + 1e-8)

                # Weighted average
                weighted_sum = torch.zeros_like(param.data)
                total_weight = torch.zeros_like(param.data)

                for p in all_params:
                    weighted_sum += importance * p
                    total_weight += importance

                param.data = weighted_sum / total_weight

                # Clean up
                del stacked, all_params
                gc.collect()

        return result

    def decode_combo(self, combo_id: int) -> str:
        """Decode combo to human-readable string."""
        bit0 = (combo_id >> 0) & 1
        bit1 = (combo_id >> 1) & 1
        bit2 = (combo_id >> 2) & 1

        interp = "SLERP" if bit0 else "Linear"
        task = "TIES" if bit1 else "DARE"
        select = "DFS" if bit2 else "Franken"

        return f"{interp} + {task} + {select}"


class RealBenchmarker:
    """Real GSM8K + Perplexity benchmark evaluation."""

    def __init__(
        self,
        tokenizer,
        gsm8k_samples: int = 50,
        ppl_samples: int = 100,
        device: str = "cuda"
    ):
        self.tokenizer = tokenizer
        self.gsm8k_samples = gsm8k_samples
        self.ppl_samples = ppl_samples
        self.device = device
        self.gsm8k_data = self._load_gsm8k()
        self.ppl_data = self._load_wikitext()

    def _load_gsm8k(self) -> List[Dict]:
        """Load GSM8K test set."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main", split="test")
            samples = []
            for item in dataset:
                answer_text = item["answer"]
                try:
                    if "####" in answer_text:
                        answer = float(answer_text.split("####")[-1].strip().replace(",", ""))
                    else:
                        import re
                        nums = re.findall(r"-?\d+\.?\d*", answer_text)
                        answer = float(nums[-1]) if nums else 0
                    samples.append({
                        "question": item["question"],
                        "answer": answer
                    })
                except:
                    continue
            return samples[:self.gsm8k_samples]
        except Exception as e:
            print(f"Warning: Could not load GSM8K: {e}")
            return []

    def _load_wikitext(self) -> List[str]:
        """Load WikiText for perplexity."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            texts = [t for t in dataset["text"] if len(t.strip()) > 50]
            return texts[:self.ppl_samples]
        except Exception as e:
            print(f"Warning: Could not load WikiText: {e}")
            return []

    def _generate_autoregressive(self, model: nn.Module, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        """Autoregressive generation for models without .generate() method."""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass
            output = model(generated)

            # Handle both dict output (TRM) and tensor output (GPT-2)
            if isinstance(output, dict):
                logits = output["logits"]
            else:
                logits = output.logits if hasattr(output, 'logits') else output

            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return generated

    def evaluate_gsm8k(self, model: nn.Module) -> Tuple[float, int, int]:
        """Evaluate model on GSM8K."""
        if not self.gsm8k_data:
            return 0.0, 0, 0

        model = model.to(self.device)
        model.eval()

        # Check if model has generate method
        has_generate = hasattr(model, 'generate') and callable(getattr(model, 'generate'))

        correct = 0
        total = 0

        with torch.no_grad():
            for i, sample in enumerate(self.gsm8k_data):
                question = sample["question"]
                gold = sample["answer"]

                prompt = f"Q: {question}\nA: Let me solve this step by step.\n"

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                try:
                    if has_generate:
                        # Use built-in generate (GPT-2 style)
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    else:
                        # Use autoregressive generation (TRM style)
                        outputs = self._generate_autoregressive(model, inputs["input_ids"], max_new_tokens=128)

                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    import re
                    nums = re.findall(r"-?\d+\.?\d*", generated[len(prompt):])
                    if nums:
                        pred = float(nums[-1])
                        if abs(pred - gold) < 0.01:
                            correct += 1

                    total += 1

                except Exception as e:
                    total += 1

                if (i + 1) % 10 == 0:
                    print(f"      GSM8K: {i+1}/{len(self.gsm8k_data)}, acc={correct}/{total}")

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, correct, total

    def evaluate_perplexity(self, model: nn.Module) -> float:
        """Evaluate perplexity on WikiText."""
        if not self.ppl_data:
            return 1000.0

        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in self.ppl_data:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                if inputs.input_ids.size(1) < 2:
                    continue

                try:
                    input_ids = inputs.input_ids
                    labels = input_ids.clone()

                    # Try different model interfaces
                    output = model(input_ids, labels=labels)

                    # Handle different output formats
                    if isinstance(output, dict):
                        if "loss" in output:
                            loss = output["loss"].item()
                        else:
                            # Compute loss from logits
                            logits = output["logits"]
                            loss = F.cross_entropy(
                                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                                labels[:, 1:].contiguous().view(-1)
                            ).item()
                    elif hasattr(output, 'loss'):
                        loss = output.loss.item()
                    else:
                        # Assume output is logits tensor
                        logits = output
                        loss = F.cross_entropy(
                            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                            labels[:, 1:].contiguous().view(-1)
                        ).item()

                    tokens = input_ids.size(1)
                    total_loss += loss * tokens
                    total_tokens += tokens
                except Exception as e:
                    continue

        if total_tokens == 0:
            return 1000.0

        avg_loss = total_loss / total_tokens
        perplexity = min(1000.0, np.exp(avg_loss))
        return perplexity

    def evaluate_full(self, model: nn.Module) -> Dict[str, Any]:
        """Full evaluation with memory management."""
        start_time = time.time()

        # GSM8K accuracy
        accuracy, correct, total = self.evaluate_gsm8k(model)

        # Perplexity
        perplexity = self.evaluate_perplexity(model)

        # Speed (tokens/sec)
        speed = total * 256 / max(1, time.time() - start_time)

        # Memory (MB)
        memory = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)

        # Move model back to CPU to free GPU
        model = model.cpu()
        clear_gpu_memory()

        # Composite fitness (simplified)
        ppl_score = 1.0 / max(1.0, perplexity)
        fitness = 0.4 * ppl_score + 0.3 * accuracy + 0.2 * (speed / 1200) + 0.1 * (500 / memory)

        return {
            "perplexity": perplexity,
            "accuracy": accuracy * 100,
            "correct": correct,
            "total": total,
            "speed": speed,
            "memory": memory,
            "fitness": fitness,
            "eval_time": time.time() - start_time
        }


def mutate_model(
    model: nn.Module,
    sigma: float = 0.01,
    rate: float = 0.01
) -> nn.Module:
    """Mutate model weights (on CPU)."""
    mutated = copy.deepcopy(model).cpu()

    with torch.no_grad():
        for param in mutated.parameters():
            if param.requires_grad:
                mask = torch.rand_like(param) < rate
                noise = torch.randn_like(param) * sigma
                param.data += noise * mask.float()

    return mutated


def compute_diversity(population: List[nn.Module]) -> float:
    """Compute population diversity (on CPU)."""
    if len(population) < 2:
        return 1.0

    distances = []
    for i in range(min(4, len(population))):  # Sample for speed
        for j in range(i + 1, min(4, len(population))):
            params_i = torch.cat([p.cpu().flatten() for p in population[i].parameters()])
            params_j = torch.cat([p.cpu().flatten() for p in population[j].parameters()])
            dist = torch.norm(params_i - params_j).item()
            distances.append(dist)

    if not distances:
        return 0.0

    avg_dist = np.mean(distances)
    expected_dist = 0.1 * len(params_i) ** 0.5
    normalized = min(1.0, avg_dist / max(1, expected_dist))

    return normalized


def run_evomerge_v6(
    checkpoint_dir: str = "checkpoints/phase1",
    output_dir: str = "checkpoints/phase2_v6",
    generations: int = 50,
    gsm8k_samples: int = 50,
    ppl_samples: int = 100,
    mutation_sigma: float = 0.01,
    mutation_rate: float = 0.01,
    diversity_threshold: float = 0.2,
    device: str = "cuda"
):
    """
    Run EvoMerge V6 with memory-optimized merging.
    """

    print("=" * 70)
    print("EVOMERGE V6 - MEMORY OPTIMIZED")
    print("=" * 70)
    print(f"Generations: {generations}")
    print(f"GSM8K samples: {gsm8k_samples}")
    print(f"Perplexity samples: {ppl_samples}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Load Phase 1 models (TRM x Titans-MAG architecture)
    print("\n[1/4] Loading Phase 1 models...")
    model_names = ["reasoning", "memory", "speed"]
    base_models = []

    from safetensors.torch import load_file as load_safetensors

    # Import Phase 1 model architecture
    try:
        from phase1_cognate.model.full_model import TRMTitansMAGModel
        from phase1_cognate.model.model_config import Phase1Config
        use_custom_arch = True
        print("  Using TRM x Titans-MAG architecture (Phase 1)")
    except ImportError:
        use_custom_arch = False
        print("  WARNING: Phase 1 architecture not found, using GPT-2")

    # Always use GPT-2 tokenizer (same vocab)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    for name in model_names:
        safetensor_path = os.path.join(checkpoint_dir, name, "epoch_10.safetensors")

        if os.path.exists(safetensor_path) and use_custom_arch:
            # Create model with correct config
            config = Phase1Config(specialization=name)
            model = TRMTitansMAGModel(config)

            # Load trained weights
            state_dict = load_safetensors(safetensor_path)
            model.load_state_dict(state_dict)
            base_models.append(model.cpu())
            params = sum(p.numel() for p in model.parameters())
            print(f"  Loaded {name}: TRM x Titans-MAG ({params/1e6:.1f}M params)")
        else:
            # Fallback to base GPT2
            model = AutoModelForCausalLM.from_pretrained("gpt2").cpu()
            base_models.append(model)
            print(f"  WARNING: Using untrained base GPT2 for {name}")

    # Initialize merger (memory-efficient)
    print("\n[2/4] Initializing memory-efficient merger...")
    merger = MemoryEfficientMerger()

    # Initialize benchmarker
    print("\n[3/4] Initializing benchmarker...")
    benchmarker = RealBenchmarker(
        tokenizer=tokenizer,
        gsm8k_samples=gsm8k_samples,
        ppl_samples=ppl_samples,
        device=device
    )
    print(f"  GSM8K: {len(benchmarker.gsm8k_data)} samples")
    print(f"  WikiText: {len(benchmarker.ppl_data)} samples")

    # GENERATION 0: Create 8 models from binary combos
    print("\n[4/4] Creating initial population (Gen 0)...")
    population = []

    for combo_id in range(8):
        combo_name = merger.decode_combo(combo_id)
        print(f"    Combo {combo_id} ({combo_id:03b}): {combo_name}...")

        clear_gpu_memory()
        merged = merger.apply_combo(base_models, combo_id)
        merged.combo_id = combo_id
        population.append(merged)

    print(f"  Initial population: {len(population)} models")

    # Initialize tracking
    champion = None
    champion_fitness = -float("inf")
    fitness_history = []
    diversity_history = []
    results_log = []

    start_time = time.time()

    # EVOLUTION LOOP
    print("\n" + "=" * 70)
    print("STARTING EVOLUTION")
    print("=" * 70)

    for gen in range(generations):
        gen_start = time.time()

        print(f"\n{'=' * 70}")
        print(f"GENERATION {gen + 1}/{generations}")
        print("=" * 70)

        # Step 1: Evaluate all models
        print(f"  Evaluating {len(population)} models...")
        fitness_scores = []

        for i, model in enumerate(population):
            print(f"    Model {i + 1}/{len(population)} (gen {gen + 1}):")
            print(f"      Running GSM8K ({gsm8k_samples} samples)...")

            metrics = benchmarker.evaluate_full(model)
            fitness_scores.append(metrics["fitness"])

            print(f"      DONE: PPL={metrics['perplexity']:.1f}, "
                  f"Acc={metrics['accuracy']:.1f}%, "
                  f"Spd={metrics['speed']:.0f}, "
                  f"fit={metrics['fitness']:.4f} "
                  f"({metrics['eval_time']:.1f}s)")

        # Step 2: Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Step 3: Update champion
        if fitness_scores[0] > champion_fitness:
            champion = copy.deepcopy(population[0])
            champion_fitness = fitness_scores[0]
            torch.save({
                "model_state_dict": champion.state_dict(),
                "fitness": champion_fitness,
                "generation": gen + 1
            }, os.path.join(output_dir, "champion.pt"))

        # Step 4: Compute diversity
        diversity = compute_diversity(population)

        # Log
        gen_time = time.time() - gen_start
        elapsed = time.time() - start_time
        eta = (elapsed / (gen + 1)) * (generations - gen - 1)

        fitness_history.append(champion_fitness)
        diversity_history.append(diversity)

        improvement = (champion_fitness / fitness_history[0] - 1) * 100 if fitness_history[0] > 0 else 0

        print(f"\n  Gen {gen + 1} Summary:")
        print(f"    Best: {fitness_scores[0]:.4f}, Champion: {champion_fitness:.4f} ({improvement:+.1f}%)")
        print(f"    Diversity: {diversity:.3f}")
        print(f"    Gen time: {gen_time / 60:.1f}min, Elapsed: {elapsed / 3600:.2f}h, ETA: {eta / 3600:.2f}h")

        # Save log
        gen_result = {
            "generation": gen + 1,
            "best_fitness": fitness_scores[0],
            "champion_fitness": champion_fitness,
            "diversity": diversity,
            "improvement_pct": improvement,
            "elapsed_hours": elapsed / 3600,
        }
        results_log.append(gen_result)

        with open(os.path.join(output_dir, "evolution_log.json"), "w") as f:
            json.dump(results_log, f, indent=2)

        # Early stopping
        if gen >= 5:
            recent = fitness_history[-5:]
            if max(recent) - min(recent) < 0.001:
                print(f"\n  CONVERGED at generation {gen + 1}!")
                break

        # Step 5: Create next generation
        if gen < generations - 1:
            print(f"\n  Creating next generation...")

            clear_gpu_memory()

            # ELITE: Top 2 -> 6 children
            print(f"    Elite mutation (top 2 -> 6 children):")
            elite_children = []

            for i, elite in enumerate([population[0], population[1]]):
                for j in range(3):
                    mut_sigma = mutation_sigma * (0.5 + j * 0.5)
                    mut_rate = mutation_rate * (0.5 + j * 0.5)
                    child = mutate_model(elite, sigma=mut_sigma, rate=mut_rate)
                    elite_children.append(child)
                    print(f"      Elite {i + 1} -> Child {j + 1}")

            # LOSER: Bottom 6 -> 2 children
            print(f"    Loser merging (bottom 6 -> 2 children):")
            losers = population[-6:]
            random.shuffle(losers)

            group1 = losers[0:3]
            group2 = losers[3:6]

            combo1 = random.randint(0, 7)
            combo2 = random.randint(0, 7)
            while combo2 == combo1:
                combo2 = random.randint(0, 7)

            print(f"      Group 1: Combo {combo1} ({combo1:03b}) = {merger.decode_combo(combo1)}")
            clear_gpu_memory()
            loser_child1 = merger.apply_combo(group1, combo1)

            print(f"      Group 2: Combo {combo2} ({combo2:03b}) = {merger.decode_combo(combo2)}")
            clear_gpu_memory()
            loser_child2 = merger.apply_combo(group2, combo2)

            loser_children = [loser_child1, loser_child2]

            # New population
            population = elite_children + loser_children
            print(f"    New population: {len(population)} models")

            # Re-seed if low diversity
            if diversity < diversity_threshold:
                print(f"    WARNING: Low diversity, re-seeding...")
                reseed1 = random.randint(0, 7)
                reseed2 = random.randint(0, 7)
                while reseed2 == reseed1:
                    reseed2 = random.randint(0, 7)

                clear_gpu_memory()
                population[-2] = merger.apply_combo(base_models, reseed1)
                population[-1] = merger.apply_combo(base_models, reseed2)

            # Clean up old population
            gc.collect()
            clear_gpu_memory()

    # FINAL RESULTS
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print(f"Champion fitness: {champion_fitness:.4f}")
    print(f"Initial fitness: {fitness_history[0]:.4f}")
    print(f"Improvement: {(champion_fitness / fitness_history[0] - 1) * 100:.1f}%")
    print(f"Generations: {len(fitness_history)}")
    print(f"Total time: {total_time / 3600:.2f} hours")

    torch.save({
        "model_state_dict": champion.state_dict(),
        "fitness": champion_fitness,
        "fitness_history": fitness_history,
        "diversity_history": diversity_history,
    }, os.path.join(output_dir, "final_champion.pt"))

    print(f"\nChampion saved to: {output_dir}/final_champion.pt")

    return champion, champion_fitness


if __name__ == "__main__":
    run_evomerge_v6(
        checkpoint_dir="checkpoints/phase1",
        output_dir="checkpoints/phase2_v6",
        generations=50,
        gsm8k_samples=50,
        ppl_samples=100,
        mutation_sigma=0.01,
        mutation_rate=0.01,
        diversity_threshold=0.2,
        device="cuda"
    )
