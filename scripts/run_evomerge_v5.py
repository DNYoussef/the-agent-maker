#!/usr/bin/env python3
"""
EvoMerge V5 - CORRECT Implementation per Phase 2 Docs

Uses the ACTUAL Phase 2 codebase with proper:
1. 8 binary combos (000-111) for Gen 0
2. Elite mutation (top 2 -> 6 children)
3. Loser merging with RANDOM combo (bottom 6 -> 2 groups -> 2 children)
4. Real GSM8K + perplexity benchmarks
5. Diversity tracking and re-seeding

Based on: docs/phases/phase2/PHASE2_COMPLETE_GUIDE.md
"""

import copy
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

# Import Phase 2 modules
from phase2_evomerge.merge import MergeTechniques
from phase2_evomerge.fitness.composite import compute_composite_fitness


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
                # Extract numeric answer from "#### answer" format
                answer_text = item["answer"]
                try:
                    if "####" in answer_text:
                        answer = float(answer_text.split("####")[-1].strip().replace(",", ""))
                    else:
                        # Try to find last number
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

    def evaluate_gsm8k(self, model: nn.Module) -> Tuple[float, int, int]:
        """
        Evaluate model on GSM8K.

        Returns: (accuracy, correct, total)
        """
        if not self.gsm8k_data:
            return 0.0, 0, 0

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i, sample in enumerate(self.gsm8k_data):
                question = sample["question"]
                gold = sample["answer"]

                # Create prompt
                prompt = f"Q: {question}\nA: Let me solve this step by step.\n"

                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                try:
                    # Generate
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    # Decode
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Extract answer (last number in response)
                    import re
                    nums = re.findall(r"-?\d+\.?\d*", generated[len(prompt):])
                    if nums:
                        pred = float(nums[-1])
                        if abs(pred - gold) < 0.01:
                            correct += 1

                    total += 1

                except Exception as e:
                    total += 1

                # Progress every 10
                if (i + 1) % 10 == 0:
                    print(f"      GSM8K: {i+1}/{len(self.gsm8k_data)}, acc={correct}/{total}")

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, correct, total

    def evaluate_perplexity(self, model: nn.Module) -> float:
        """
        Evaluate perplexity on WikiText.

        Returns: perplexity value (lower is better)
        """
        if not self.ppl_data:
            return 1000.0

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
                    outputs = model(**inputs, labels=inputs.input_ids)
                    loss = outputs.loss.item()
                    tokens = inputs.input_ids.size(1)

                    total_loss += loss * tokens
                    total_tokens += tokens
                except:
                    continue

        if total_tokens == 0:
            return 1000.0

        avg_loss = total_loss / total_tokens
        perplexity = min(1000.0, np.exp(avg_loss))
        return perplexity

    def evaluate_full(self, model: nn.Module) -> Dict[str, Any]:
        """
        Full evaluation: GSM8K + Perplexity + Speed + Memory.

        Returns: Dict with all metrics and composite fitness
        """
        start_time = time.time()

        # GSM8K accuracy
        accuracy, correct, total = self.evaluate_gsm8k(model)

        # Perplexity
        perplexity = self.evaluate_perplexity(model)

        # Speed (tokens/sec)
        speed = total * 256 / max(1, time.time() - start_time)

        # Memory (MB)
        memory = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)

        # Composite fitness
        fitness_result = compute_composite_fitness(
            perplexity=perplexity,
            accuracy=accuracy,
            speed=speed,
            memory=memory
        )

        return {
            "perplexity": perplexity,
            "accuracy": accuracy * 100,  # As percentage
            "correct": correct,
            "total": total,
            "speed": speed,
            "memory": memory,
            "fitness": fitness_result["composite"],
            "eval_time": time.time() - start_time
        }


def mutate_model(
    model: nn.Module,
    sigma: float = 0.01,
    rate: float = 0.01,
    device: str = "cuda"
) -> nn.Module:
    """
    Mutate model weights with Gaussian noise.

    Args:
        model: Model to mutate
        sigma: Noise standard deviation
        rate: Fraction of weights to mutate
        device: Target device

    Returns:
        New mutated model (deep copy)
    """
    mutated = copy.deepcopy(model)
    mutated = mutated.to(device)

    with torch.no_grad():
        for param in mutated.parameters():
            if param.requires_grad:
                # Create mask for which weights to mutate
                mask = torch.rand_like(param) < rate
                # Apply Gaussian noise to selected weights
                noise = torch.randn_like(param) * sigma
                param.data += noise * mask.float()

    return mutated


def compute_diversity(population: List[nn.Module]) -> float:
    """
    Compute population diversity as average pairwise L2 distance.

    Returns: Normalized diversity score (0 = identical, 1 = diverse)
    """
    if len(population) < 2:
        return 1.0

    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Flatten parameters
            params_i = torch.cat([p.flatten() for p in population[i].parameters()])
            params_j = torch.cat([p.flatten() for p in population[j].parameters()])

            # L2 distance
            dist = torch.norm(params_i - params_j).item()
            distances.append(dist)

    avg_dist = np.mean(distances)
    # Normalize by expected distance (rough estimate)
    expected_dist = 0.1 * params_i.numel() ** 0.5
    normalized = min(1.0, avg_dist / expected_dist)

    return normalized


def run_evomerge_v5(
    checkpoint_dir: str = "checkpoints/phase1",
    output_dir: str = "checkpoints/phase2_v5",
    generations: int = 50,
    gsm8k_samples: int = 50,
    ppl_samples: int = 100,
    mutation_sigma: float = 0.01,
    mutation_rate: float = 0.01,
    diversity_threshold: float = 0.2,
    device: str = "cuda"
):
    """
    Run EvoMerge V5 with correct implementation.

    Strategy per docs:
    - Gen 0: Create 8 models from all binary combos (000-111)
    - Each generation:
      - Elite (top 2): Mutate 3x each = 6 children
      - Loser (bottom 6): Split into 2 groups of 3, merge with RANDOM combo = 2 children
    """

    print("=" * 70)
    print("EVOMERGE V5 - CORRECT IMPLEMENTATION")
    print("=" * 70)
    print(f"Generations: {generations}")
    print(f"GSM8K samples: {gsm8k_samples}")
    print(f"Perplexity samples: {ppl_samples}")
    print(f"Mutation: sigma={mutation_sigma}, rate={mutation_rate}")
    print(f"Diversity threshold: {diversity_threshold}")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load Phase 1 models
    print("\n[1/4] Loading Phase 1 models...")
    model_names = ["reasoning", "memory", "speed"]
    base_models = []
    tokenizer = None

    for name in model_names:
        model_path = os.path.join(checkpoint_dir, f"{name}_final.pt")
        if os.path.exists(model_path):
            # Load our trained model
            checkpoint = torch.load(model_path, map_location=device)
            # For now, use a base model and load state dict
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            base_models.append(model)
            print(f"  Loaded {name} model from {model_path}")
        else:
            # Use base GPT2 as fallback
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
            base_models.append(model)
            print(f"  Using base GPT2 for {name} (checkpoint not found)")

    print(f"  Loaded {len(base_models)} models")

    # Initialize merge techniques
    print("\n[2/4] Initializing merge techniques...")
    merger = MergeTechniques()
    print("  6 merge techniques ready (Linear, SLERP, DARE, TIES, Franken, DFS)")
    print("  8 binary combos (000-111)")

    # Initialize benchmarker
    print("\n[3/4] Initializing benchmarker...")
    benchmarker = RealBenchmarker(
        tokenizer=tokenizer,
        gsm8k_samples=gsm8k_samples,
        ppl_samples=ppl_samples,
        device=device
    )
    print(f"  GSM8K: {len(benchmarker.gsm8k_data)} samples loaded")
    print(f"  WikiText: {len(benchmarker.ppl_data)} samples loaded")

    # GENERATION 0: Create 8 models from all binary combos
    print("\n[4/4] Creating initial population (Gen 0)...")
    print("  Creating 8 models from binary combos 000-111")

    population = []
    for combo_id in range(8):
        combo_name = merger.decode_combo(combo_id)
        print(f"    Combo {combo_id} ({combo_id:03b}): {combo_name}...")

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

        # Step 2: Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Step 3: Update champion
        if fitness_scores[0] > champion_fitness:
            champion = copy.deepcopy(population[0])
            champion_fitness = fitness_scores[0]
            # Save champion
            torch.save({
                "model_state_dict": champion.state_dict(),
                "fitness": champion_fitness,
                "generation": gen + 1
            }, os.path.join(output_dir, "champion.pt"))

        # Step 4: Compute diversity
        diversity = compute_diversity(population)

        # Log results
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

        # Save generation results
        gen_result = {
            "generation": gen + 1,
            "best_fitness": fitness_scores[0],
            "champion_fitness": champion_fitness,
            "diversity": diversity,
            "improvement_pct": improvement,
            "elapsed_hours": elapsed / 3600,
            "fitness_scores": fitness_scores
        }
        results_log.append(gen_result)

        # Save log
        with open(os.path.join(output_dir, "evolution_log.json"), "w") as f:
            json.dump(results_log, f, indent=2)

        # Early stopping check
        if gen >= 5:
            recent = fitness_history[-5:]
            if max(recent) - min(recent) < 0.001:
                print(f"\n  CONVERGED at generation {gen + 1}!")
                break

        # Step 5: Create next generation
        if gen < generations - 1:
            print(f"\n  Creating next generation...")

            # ELITE PRESERVATION: Top 2 -> mutate 3x each = 6 children
            print(f"    Elite mutation (top 2 -> 6 children):")
            elite1, elite2 = population[0], population[1]
            elite_children = []

            for i, elite in enumerate([elite1, elite2]):
                for j in range(3):
                    # Vary mutation strength
                    mut_sigma = mutation_sigma * (0.5 + j * 0.5)
                    mut_rate = mutation_rate * (0.5 + j * 0.5)

                    child = mutate_model(elite, sigma=mut_sigma, rate=mut_rate, device=device)
                    elite_children.append(child)
                    print(f"      Elite {i + 1} -> Child {j + 1} (sigma={mut_sigma:.3f}, rate={mut_rate:.3f})")

            # LOSER MERGING: Bottom 6 -> 2 groups of 3 -> merge with RANDOM combo = 2 children
            print(f"    Loser merging (bottom 6 -> 2 children):")
            losers = population[-6:]

            # Shuffle losers for random grouping
            random.shuffle(losers)

            group1 = losers[0:3]
            group2 = losers[3:6]

            # Random combo selection (KEY DIVERSITY SOURCE!)
            combo1 = random.randint(0, 7)
            combo2 = random.randint(0, 7)
            while combo2 == combo1:  # Ensure different combos
                combo2 = random.randint(0, 7)

            print(f"      Group 1: Combo {combo1} ({combo1:03b}) = {merger.decode_combo(combo1)}")
            loser_child1 = merger.apply_combo(group1, combo1)
            loser_child1.combo_id = combo1

            print(f"      Group 2: Combo {combo2} ({combo2:03b}) = {merger.decode_combo(combo2)}")
            loser_child2 = merger.apply_combo(group2, combo2)
            loser_child2.combo_id = combo2

            loser_children = [loser_child1, loser_child2]

            # NEW POPULATION: 6 elite children + 2 loser children = 8
            population = elite_children + loser_children
            print(f"    New population: {len(population)} models (6 elite + 2 loser)")

            # Diversity re-seeding if needed
            if diversity < diversity_threshold:
                print(f"    WARNING: Low diversity ({diversity:.3f} < {diversity_threshold})")
                print(f"    Re-seeding bottom 2 with fresh random combos...")

                reseed_combo1 = random.randint(0, 7)
                reseed_combo2 = random.randint(0, 7)
                while reseed_combo2 == reseed_combo1:
                    reseed_combo2 = random.randint(0, 7)

                population[-2] = merger.apply_combo(base_models, reseed_combo1)
                population[-1] = merger.apply_combo(base_models, reseed_combo2)
                print(f"    Re-seeded with combos {reseed_combo1} and {reseed_combo2}")

    # FINAL RESULTS
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print(f"Champion fitness: {champion_fitness:.4f}")
    print(f"Initial fitness: {fitness_history[0]:.4f}")
    print(f"Improvement: {(champion_fitness / fitness_history[0] - 1) * 100:.1f}%")
    print(f"Generations: {len(fitness_history)}")
    print(f"Final diversity: {diversity_history[-1]:.3f}")
    print(f"Total time: {total_time / 3600:.2f} hours")

    # Save final champion
    torch.save({
        "model_state_dict": champion.state_dict(),
        "fitness": champion_fitness,
        "fitness_history": fitness_history,
        "diversity_history": diversity_history,
        "generations": len(fitness_history),
        "improvement_pct": (champion_fitness / fitness_history[0] - 1) * 100
    }, os.path.join(output_dir, "final_champion.pt"))

    print(f"\nChampion saved to: {output_dir}/final_champion.pt")
    print("=" * 70)

    return champion, champion_fitness


if __name__ == "__main__":
    run_evomerge_v5(
        checkpoint_dir="checkpoints/phase1",
        output_dir="checkpoints/phase2_v5",
        generations=50,
        gsm8k_samples=50,
        ppl_samples=100,
        mutation_sigma=0.01,
        mutation_rate=0.01,
        diversity_threshold=0.2,
        device="cuda"
    )
