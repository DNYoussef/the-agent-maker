"""
Spectral Gap Monitoring for Agent Forge V2

Spectral gap analysis provides insights into representation diversity
and training stability. Key applications:

    - Phase 2 (EvoMerge): Ensure merged models preserve diversity
    - Phase 3 (Quiet-STaR): Prevent thought collapse in parallel generation
    - Phase 7 (Experts): Ensure expert diversity
    - Phase 8 (Compression): Validate compression preserves representations

Key Result from Meta-Calculus:
    gap(P_mix) >= min(gaps)

This means composed/mixed operators preserve at least the minimum
spectral gap - crucial for ensuring stability through transformations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class SpectralGapConfig:
    """Configuration for spectral gap computation."""

    # Computation settings
    top_k: int = 10  # Number of top eigenvalues to compute
    use_svd: bool = True  # Use SVD instead of eigendecomposition
    normalize: bool = True  # Normalize by largest eigenvalue

    # Thresholds
    collapse_threshold: float = 0.01  # Gap below this indicates collapse
    healthy_threshold: float = 0.1  # Gap above this is healthy

    # Monitoring
    track_history: bool = True
    history_window: int = 100


class SpectralGapMonitor:
    """
    Monitor spectral gap of weight matrices and embeddings.

    The spectral gap is defined as:
        gap = lambda_1 - lambda_2

    where lambda_1 > lambda_2 are the largest eigenvalues.

    A healthy gap indicates diverse representations.
    A collapsed gap indicates rank deficiency or over-smoothing.
    """

    def __init__(self, config: Optional[SpectralGapConfig] = None):
        self.config = config or SpectralGapConfig()
        self.history: Dict[str, List[float]] = {}

    def compute_gap(self, matrix: torch.Tensor, name: Optional[str] = None) -> Dict[str, float]:
        """
        Compute spectral gap of a matrix.

        Args:
            matrix: 2D tensor (weight matrix or embedding matrix)
            name: Optional name for tracking

        Returns:
            Dictionary with gap metrics
        """
        if matrix.dim() == 1:
            matrix = matrix.unsqueeze(0)

        if matrix.dim() > 2:
            # Flatten to 2D
            matrix = matrix.reshape(matrix.size(0), -1)

        # Compute singular values (more stable than eigenvalues)
        if self.config.use_svd:
            try:
                U, S, V = torch.linalg.svd(matrix, full_matrices=False)
                eigenvalues = S**2  # Squared singular values = eigenvalues of M^T M
            except RuntimeError:
                # Fallback for numerical issues
                eigenvalues = torch.zeros(min(matrix.shape))
        else:
            # Compute covariance matrix eigenvalues
            if matrix.size(0) < matrix.size(1):
                cov = matrix @ matrix.T
            else:
                cov = matrix.T @ matrix
            eigenvalues = torch.linalg.eigvalsh(cov)

        # Sort descending
        eigenvalues = torch.sort(eigenvalues, descending=True)[0]

        # Normalize
        if self.config.normalize and eigenvalues[0] > 0:
            eigenvalues = eigenvalues / eigenvalues[0]

        # Compute metrics
        lambda_1 = eigenvalues[0].item() if len(eigenvalues) > 0 else 0
        lambda_2 = eigenvalues[1].item() if len(eigenvalues) > 1 else 0
        gap = lambda_1 - lambda_2

        # Effective rank (using entropy)
        if len(eigenvalues) > 0 and eigenvalues.sum() > 0:
            p = eigenvalues / eigenvalues.sum()
            p = p[p > 1e-10]  # Filter zeros
            entropy = -(p * torch.log(p)).sum().item()
            effective_rank = np.exp(entropy)
        else:
            effective_rank = 0

        result = {
            "gap": gap,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "ratio": lambda_2 / (lambda_1 + 1e-10),
            "effective_rank": effective_rank,
            "is_healthy": gap >= self.config.healthy_threshold,
            "is_collapsed": gap < self.config.collapse_threshold,
        }

        # Track history
        if name and self.config.track_history:
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(gap)
            if len(self.history[name]) > self.config.history_window:
                self.history[name] = self.history[name][-self.config.history_window :]

        return result

    def compute_model_gaps(
        self,
        model: torch.nn.Module,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute spectral gaps for all weight matrices in a model.

        Args:
            model: PyTorch model
            include_patterns: Only include params matching these patterns
            exclude_patterns: Exclude params matching these patterns

        Returns:
            Dictionary mapping param names to gap metrics
        """
        results = {}

        for name, param in model.named_parameters():
            # Filter by patterns
            if include_patterns:
                if not any(p in name for p in include_patterns):
                    continue
            if exclude_patterns:
                if any(p in name for p in exclude_patterns):
                    continue

            # Skip 1D parameters (biases, layer norms)
            if param.dim() < 2:
                continue

            results[name] = self.compute_gap(param.data, name)

        return results

    def compute_embedding_diversity(
        self, embeddings: torch.Tensor, name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute diversity metrics for embeddings.

        Useful for:
            - Phase 3: Thought embedding diversity
            - Phase 7: Expert output diversity

        Args:
            embeddings: [N, D] embedding tensor
            name: Optional name for tracking

        Returns:
            Diversity metrics
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        N, D = embeddings.shape

        # Covariance matrix
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (N - 1 + 1e-8)

        # Spectral gap of covariance
        gap_result = self.compute_gap(cov, name)

        # Additional diversity metrics
        if N < 2:
            gap_result["avg_pairwise_similarity"] = 1.0
            gap_result["diversity_score"] = 0.0
            gap_result["is_collapsed"] = True
            gap_result["is_healthy"] = False
            return gap_result

        # Pairwise cosine similarity
        normalized = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = normalized @ normalized.T

        # Average pairwise similarity (excluding diagonal)
        mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
        avg_similarity = sim_matrix[mask].mean().item()

        gap_result["avg_pairwise_similarity"] = avg_similarity
        gap_result["diversity_score"] = 1 - avg_similarity  # Higher is more diverse

        return gap_result

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for tracked matrices."""
        summary = {}

        for name, gaps in self.history.items():
            if not gaps:
                continue

            gaps_array = np.array(gaps)
            summary[name] = {
                "current": gaps[-1],
                "mean": float(gaps_array.mean()),
                "std": float(gaps_array.std()),
                "min": float(gaps_array.min()),
                "max": float(gaps_array.max()),
                "trend": float(gaps_array[-10:].mean() - gaps_array[:10].mean())
                if len(gaps) >= 20
                else 0,
            }

        return summary


# =============================================================================
# Specialized Functions for Agent Forge Phases
# =============================================================================


def compute_expert_diversity(
    expert_weights: List[torch.Tensor], config: Optional[SpectralGapConfig] = None
) -> Dict[str, float]:
    """
    Compute diversity metrics across multiple experts (Phase 7).

    Args:
        expert_weights: List of expert weight tensors
        config: Optional configuration

    Returns:
        Diversity metrics for expert set
    """
    config = config or SpectralGapConfig()
    monitor = SpectralGapMonitor(config)

    # Flatten each expert to 1D and stack
    flattened = []
    for w in expert_weights:
        flattened.append(w.flatten())

    # Pad to same length if needed
    max_len = max(f.shape[0] for f in flattened)
    padded = []
    for f in flattened:
        if f.shape[0] < max_len:
            pad = torch.zeros(max_len - f.shape[0], device=f.device)
            f = torch.cat([f, pad])
        padded.append(f)

    expert_matrix = torch.stack(padded)  # [N_experts, D]

    # Compute diversity
    result = monitor.compute_embedding_diversity(expert_matrix, "experts")
    result["n_experts"] = len(expert_weights)

    return result


def compute_thought_diversity(
    thoughts: torch.Tensor, config: Optional[SpectralGapConfig] = None
) -> Dict[str, float]:
    """
    Compute diversity metrics for parallel thoughts (Phase 3 Quiet-STaR).

    Args:
        thoughts: [N_thoughts, D] thought embedding tensor
        config: Optional configuration

    Returns:
        Thought diversity metrics
    """
    config = config or SpectralGapConfig()
    monitor = SpectralGapMonitor(config)

    result = monitor.compute_embedding_diversity(thoughts, "thoughts")

    # Add thought-specific warning
    if result["is_collapsed"]:
        result["warning"] = "Thought collapse detected - parallel thoughts too similar"

    return result


def compute_merge_diversity_change(
    models_before: List[torch.nn.Module],
    model_after: torch.nn.Module,
    config: Optional[SpectralGapConfig] = None,
) -> Dict[str, float]:
    """
    Compute how merging affected spectral gaps (Phase 2 EvoMerge).

    Args:
        models_before: List of models before merging
        model_after: Merged model
        config: Optional configuration

    Returns:
        Diversity change metrics
    """
    config = config or SpectralGapConfig()
    monitor = SpectralGapMonitor(config)

    # Compute gaps for models before merge
    before_gaps = []
    for i, model in enumerate(models_before):
        model_gaps = monitor.compute_model_gaps(model)
        avg_gap = np.mean([v["gap"] for v in model_gaps.values()]) if model_gaps else 0
        before_gaps.append(avg_gap)

    # Compute gaps for merged model
    after_gaps = monitor.compute_model_gaps(model_after)
    avg_after = np.mean([v["gap"] for v in after_gaps.values()]) if after_gaps else 0

    # For arbitrary neural weights this is an empirical retention check, not a theorem.
    min_before = min(before_gaps) if before_gaps else 0
    max_before = max(before_gaps) if before_gaps else 0
    mean_before = np.mean(before_gaps) if before_gaps else 0

    return {
        "min_gap_before": min_before,
        "max_gap_before": max_before,
        "mean_gap_before": mean_before,
        "gap_after": avg_after,
        "gap_preservation_ratio": avg_after / (min_before + 1e-10),
        "satisfies_bound": avg_after >= min_before * 0.95,  # 5% tolerance
    }


def compute_compression_gap_retention(
    model_original: torch.nn.Module,
    model_compressed: torch.nn.Module,
    config: Optional[SpectralGapConfig] = None,
) -> Dict[str, float]:
    """
    Compute spectral gap retention after compression (Phase 8).

    Args:
        model_original: Original model
        model_compressed: Compressed model
        config: Optional configuration

    Returns:
        Gap retention metrics
    """
    config = config or SpectralGapConfig()
    monitor = SpectralGapMonitor(config)

    # Compute gaps for both models
    original_gaps = monitor.compute_model_gaps(model_original)
    compressed_gaps = monitor.compute_model_gaps(model_compressed)

    # Match by parameter name
    retention_ratios = []
    for name in original_gaps:
        if name in compressed_gaps:
            orig = original_gaps[name]["gap"]
            comp = compressed_gaps[name]["gap"]
            if orig > 1e-10:
                retention_ratios.append(comp / orig)

    if not retention_ratios:
        return {"error": "No matching parameters found"}

    return {
        "mean_retention": float(np.mean(retention_ratios)),
        "min_retention": float(np.min(retention_ratios)),
        "max_retention": float(np.max(retention_ratios)),
        "layer_retentions": [float(r) for r in retention_ratios],
        "n_params_compared": len(retention_ratios),
        "passes_threshold": float(np.min(retention_ratios)) >= 0.90,
    }


# =============================================================================
# Spectral Gap as Loss/Regularizer
# =============================================================================


class SpectralGapRegularizer(torch.nn.Module):
    """
    Regularizer that encourages healthy spectral gaps.

    Can be added to loss to prevent representation collapse.
    """

    def __init__(
        self, target_gap: float = 0.1, weight: float = 0.01, layers: Optional[List[str]] = None
    ):
        """
        Initialize regularizer.

        Args:
            target_gap: Target spectral gap value
            weight: Regularization weight
            layers: Layer name patterns to regularize (None = all)
        """
        super().__init__()
        self.target_gap = target_gap
        self.weight = weight
        self.layers = layers
        self.monitor = SpectralGapMonitor()

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Compute spectral gap regularization loss.

        Args:
            model: Model to regularize

        Returns:
            Regularization loss tensor
        """
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        count = 0

        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue

            if self.layers is not None:
                if not any(layer in name for layer in self.layers):
                    continue

            # Compute gap (need gradients)
            if param.size(0) < param.size(1):
                matrix = param @ param.T
            else:
                matrix = param.T @ param

            # Get top 2 eigenvalues
            eigenvalues = torch.linalg.eigvalsh(matrix)
            eigenvalues = torch.sort(eigenvalues, descending=True)[0]

            lambda_1 = eigenvalues[0]
            lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else torch.tensor(0.0)

            # Normalize
            gap = (lambda_1 - lambda_2) / (lambda_1 + 1e-8)

            # Anti-collapse loss: penalize only gaps below target.
            gap_loss = torch.relu(self.target_gap - gap) ** 2
            total_loss = total_loss + gap_loss
            count += 1

        if count > 0:
            total_loss = total_loss / count

        return self.weight * total_loss


def thought_diversity_loss(thoughts: torch.Tensor, target_diversity: float = 0.5) -> torch.Tensor:
    """
    Loss function to encourage diverse parallel thoughts.

    Args:
        thoughts: [N, D] thought embeddings
        target_diversity: Target diversity score (0-1)

    Returns:
        Diversity loss
    """
    N = thoughts.size(0)
    if N < 2:
        return torch.ones((), device=thoughts.device, dtype=thoughts.dtype)

    # Normalize
    normalized = thoughts / (thoughts.norm(dim=1, keepdim=True) + 1e-8)

    # Pairwise similarity
    sim_matrix = normalized @ normalized.T

    # Mean off-diagonal similarity (want this low)
    mask = ~torch.eye(N, dtype=torch.bool, device=thoughts.device)
    avg_sim = sim_matrix[mask].mean()

    # Loss: penalize similarity above (1 - target_diversity)
    target_sim = 1 - target_diversity
    loss = torch.relu(avg_sim - target_sim) ** 2

    return loss


# =============================================================================
# Demo and Testing
# =============================================================================


def run_demo():
    """Run spectral gap monitoring demo."""
    print("\n" + "=" * 60)
    print("SPECTRAL GAP MONITORING DEMO")
    print("=" * 60)

    # Create test model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.Linear(256, 256),
        torch.nn.Linear(256, 10),
    )

    # Initialize monitor
    print("\n1. Computing model spectral gaps...")
    monitor = SpectralGapMonitor()
    gaps = monitor.compute_model_gaps(model)

    for name, metrics in gaps.items():
        status = "HEALTHY" if metrics["is_healthy"] else "WARNING"
        print(f"   {name}: gap={metrics['gap']:.4f} [{status}]")

    # Test thought diversity
    print("\n2. Testing thought diversity (simulating Phase 3)...")

    # Good diversity
    thoughts_good = torch.randn(8, 256)
    diversity_good = compute_thought_diversity(thoughts_good)
    print(f"   Random thoughts: diversity={diversity_good['diversity_score']:.4f}")

    # Bad diversity (similar thoughts)
    base_thought = torch.randn(1, 256)
    thoughts_bad = base_thought + torch.randn(8, 256) * 0.1
    diversity_bad = compute_thought_diversity(thoughts_bad)
    print(f"   Similar thoughts: diversity={diversity_bad['diversity_score']:.4f}")

    # Test expert diversity
    print("\n3. Testing expert diversity (simulating Phase 7)...")
    expert_weights = [torch.randn(64, 64) for _ in range(5)]
    expert_div = compute_expert_diversity(expert_weights)
    print(f"   Expert diversity score: {expert_div['diversity_score']:.4f}")
    print(f"   Spectral gap: {expert_div['gap']:.4f}")

    # Test regularizer
    print("\n4. Testing spectral gap regularizer...")
    regularizer = SpectralGapRegularizer(target_gap=0.1, weight=0.01)
    reg_loss = regularizer(model)
    print(f"   Regularization loss: {reg_loss.item():.6f}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
