"""
Phase 7: Expert Discovery System

Model-driven expert discovery through self-analysis.
The model analyzes its own capabilities and determines expert count.

Research: Transformer^2 SVF, NSGA-II ADAS
Key insight: Self-guided discovery (N=3-10 experts) vs manual design.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExpertProfile:
    """Profile of a discovered expert."""

    id: int
    name: str
    capabilities: List[str]
    strength_score: float
    activation_pattern: List[float]  # Which layers activate for this expert


@dataclass
class DiscoveryConfig:
    """Configuration for expert discovery."""

    min_experts: int = 3
    max_experts: int = 10
    discovery_samples: int = 100
    clustering_threshold: float = 0.7
    capability_categories: List[str] = field(
        default_factory=lambda: [
            "reasoning",
            "coding",
            "math",
            "writing",
            "analysis",
            "creativity",
            "instruction_following",
        ]
    )


class ExpertDiscovery:
    """
    Self-Guided Expert Discovery.

    Process:
    1. Model self-analyzes via diverse prompts
    2. Cluster activation patterns to find natural expert groupings
    3. Determine optimal expert count (N=3-10)
    4. Profile each discovered expert

    This is the key V2 innovation: model determines its own experts.
    """

    def __init__(self, config: DiscoveryConfig = None):
        """
        Initialize expert discovery system.

        Args:
            config: Discovery configuration
        """
        self.config = config or DiscoveryConfig()
        self.discovered_experts: List[ExpertProfile] = []
        self.activation_cache: Dict[str, List[float]] = {}

    def discover(self, model: nn.Module, tokenizer: Any) -> Tuple[int, List[ExpertProfile]]:
        """
        Discover experts through model self-analysis.

        Args:
            model: Model to analyze
            tokenizer: Tokenizer

        Returns:
            Tuple of (num_experts, list of ExpertProfiles)
        """
        print("Stage 1: Expert Discovery")
        print("-" * 40)

        # Step 1: Generate diverse prompts for each capability
        print("  Generating discovery prompts...")
        discovery_prompts = self._generate_discovery_prompts()

        # Step 2: Collect activation patterns
        print("  Collecting activation patterns...")
        activations = self._collect_activations(model, tokenizer, discovery_prompts)

        # Step 3: Cluster activations to find natural groupings
        print("  Clustering activation patterns...")
        clusters = self._cluster_activations(activations)

        # Step 4: Determine expert count
        num_experts = self._determine_expert_count(clusters)
        print(f"  Discovered {num_experts} natural expert groupings")

        # Step 5: Profile each expert
        print("  Profiling discovered experts...")
        self.discovered_experts = self._profile_experts(clusters, num_experts)

        for expert in self.discovered_experts:
            print(f"    Expert {expert.id}: {expert.name} (strength: {expert.strength_score:.2f})")
            print(f"      Capabilities: {', '.join(expert.capabilities[:3])}")

        return num_experts, self.discovered_experts

    def _generate_discovery_prompts(self) -> Dict[str, List[str]]:
        """Generate prompts for each capability category."""
        prompts = {}

        for category in self.config.capability_categories:
            if category == "reasoning":
                prompts[category] = [
                    "Explain step by step why the sky is blue.",
                    "What is the logical flaw in this argument: All birds can fly. Penguins are birds. Therefore penguins can fly.",
                    "If A implies B, and B implies C, what can we conclude about A and C?",
                ]
            elif category == "coding":
                prompts[category] = [
                    "Write a Python function to find the nth Fibonacci number.",
                    "Explain what this code does: def f(x): return x if x <= 1 else f(x-1) + f(x-2)",
                    "What is the time complexity of binary search?",
                ]
            elif category == "math":
                prompts[category] = [
                    "Solve: 2x + 5 = 17",
                    "What is the derivative of x^3?",
                    "Calculate 15% of 240.",
                ]
            elif category == "writing":
                prompts[category] = [
                    "Write a haiku about autumn.",
                    "Summarize the main idea in one sentence: The quick brown fox jumps over the lazy dog.",
                    "Write an opening paragraph for a mystery story.",
                ]
            elif category == "analysis":
                prompts[category] = [
                    "Compare and contrast apples and oranges.",
                    "What are the pros and cons of remote work?",
                    "Analyze this data trend: 10, 15, 22, 31, 42",
                ]
            elif category == "creativity":
                prompts[category] = [
                    "Invent a new word and define it.",
                    "What if gravity worked in reverse?",
                    "Design a logo for a time travel company.",
                ]
            elif category == "instruction_following":
                prompts[category] = [
                    "List exactly 3 fruits.",
                    "Respond with only the word 'yes' or 'no': Is 2+2=4?",
                    "Write a sentence using exactly 5 words.",
                ]
            else:
                prompts[category] = [f"Test prompt for {category}"]

        return prompts

    def _collect_activations(
        self, model: nn.Module, tokenizer: Any, prompts: Dict[str, List[str]]
    ) -> Dict[str, List[Dict]]:
        """Collect activation patterns for each prompt."""
        activations = {}
        model.eval()

        # Hook to capture activations
        layer_activations = []

        def activation_hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # Store mean activation per layer
            if hasattr(out, "mean"):
                layer_activations.append(out.mean().item())

        # Register hooks on transformer layers
        hooks = []
        for name, module in model.named_modules():
            if "layer" in name.lower() or "block" in name.lower():
                hook = module.register_forward_hook(activation_hook)
                hooks.append(hook)

        device = next(model.parameters()).device

        with torch.no_grad():
            for category, prompt_list in prompts.items():
                activations[category] = []

                for prompt in prompt_list:
                    layer_activations = []  # Reset

                    try:
                        if hasattr(tokenizer, "__call__"):
                            inputs = tokenizer(
                                prompt,
                                return_tensors="pt",
                                max_length=128,
                                truncation=True,
                                padding=True,
                            )
                        else:
                            inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

                        inputs = {
                            k: v.to(device)
                            for k, v in inputs.items()
                            if isinstance(v, torch.Tensor)
                        }

                        _ = model(**inputs)

                        activations[category].append(
                            {
                                "prompt": prompt,
                                "activations": layer_activations.copy(),
                                "category": category,
                            }
                        )

                    except Exception:
                        continue

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations

    def _cluster_activations(self, activations: Dict[str, List[Dict]]) -> Dict[int, List[Dict]]:
        """Cluster activation patterns to find natural groupings."""
        # Flatten all activations
        all_patterns = []
        for category, patterns in activations.items():
            for pattern in patterns:
                all_patterns.append(
                    {
                        "category": category,
                        "activations": pattern["activations"],
                        "prompt": pattern["prompt"],
                    }
                )

        if not all_patterns:
            return {0: all_patterns}

        # Simple clustering based on activation similarity
        # (In production, would use k-means or hierarchical clustering)

        clusters = {}
        cluster_id = 0

        for pattern in all_patterns:
            assigned = False

            for cid, cluster_patterns in clusters.items():
                if cluster_patterns:
                    # Compare to cluster centroid
                    centroid = self._compute_centroid(cluster_patterns)
                    similarity = self._compute_similarity(pattern["activations"], centroid)

                    if similarity > self.config.clustering_threshold:
                        clusters[cid].append(pattern)
                        assigned = True
                        break

            if not assigned:
                clusters[cluster_id] = [pattern]
                cluster_id += 1

        return clusters

    def _compute_centroid(self, patterns: List[Dict]) -> List[float]:
        """Compute centroid of activation patterns."""
        if not patterns:
            return []

        all_acts = [p["activations"] for p in patterns if p["activations"]]
        if not all_acts:
            return []

        max_len = max(len(a) for a in all_acts)
        centroid = [0.0] * max_len

        for acts in all_acts:
            for i, a in enumerate(acts):
                centroid[i] += a

        centroid = [c / len(all_acts) for c in centroid]
        return centroid

    def _compute_similarity(self, acts1: List[float], acts2: List[float]) -> float:
        """Compute cosine similarity between activation patterns."""
        if not acts1 or not acts2:
            return 0.0

        # Pad to same length
        max_len = max(len(acts1), len(acts2))
        a1 = acts1 + [0.0] * (max_len - len(acts1))
        a2 = acts2 + [0.0] * (max_len - len(acts2))

        # Cosine similarity
        dot = sum(x * y for x, y in zip(a1, a2))
        norm1 = sum(x * x for x in a1) ** 0.5
        norm2 = sum(x * x for x in a2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def _determine_expert_count(self, clusters: Dict[int, List[Dict]]) -> int:
        """Determine optimal number of experts."""
        # Number of significant clusters
        significant_clusters = [
            cid for cid, patterns in clusters.items() if len(patterns) >= 2  # At least 2 patterns
        ]

        num_experts = len(significant_clusters)

        # Clamp to valid range
        num_experts = max(self.config.min_experts, num_experts)
        num_experts = min(self.config.max_experts, num_experts)

        return num_experts

    def _profile_experts(
        self, clusters: Dict[int, List[Dict]], num_experts: int
    ) -> List[ExpertProfile]:
        """Create profiles for discovered experts."""
        experts = []

        # Sort clusters by size
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[
            :num_experts
        ]

        for i, (cluster_id, patterns) in enumerate(sorted_clusters):
            # Determine capabilities from patterns
            capabilities = list(set(p["category"] for p in patterns))

            # Calculate strength score
            strength = len(patterns) / max(1, sum(len(c) for _, c in sorted_clusters))

            # Get activation pattern (centroid)
            activation_pattern = self._compute_centroid(patterns)

            # Generate expert name
            primary_cap = capabilities[0] if capabilities else "general"
            name = f"{primary_cap}_expert_{i+1}"

            experts.append(
                ExpertProfile(
                    id=i,
                    name=name,
                    capabilities=capabilities,
                    strength_score=strength,
                    activation_pattern=activation_pattern,
                )
            )

        return experts

    def get_expert_assignments(self) -> Dict[str, int]:
        """Get mapping of capabilities to expert IDs."""
        assignments = {}

        for expert in self.discovered_experts:
            for cap in expert.capabilities:
                if cap not in assignments:
                    assignments[cap] = expert.id

        return assignments


__all__ = ["ExpertDiscovery", "ExpertProfile", "DiscoveryConfig"]
