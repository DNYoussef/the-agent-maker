"""
Phase 7: Transformer2 Two-Pass Inference Architecture

Implements the Transformer^2 architecture with dynamic expert routing.
Pass 1: Compute routing weights z_i (which experts to use)
Pass 2: Apply weighted expert combination W = sum(z_i * W_i)

Research: "Transformer^2: Self-adaptive LLMs" (arXiv)
Key insight: Two-pass enables dynamic task-specific adaptation without retraining.

Expert Composition Formula:
    W_combined = sum_{i=1}^{N} z_i * W_i

Where:
    z_i = routing weight for expert i (from Pass 1)
    W_i = expert adapter weights (low-rank)
    N = number of experts
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Transformer2Config:
    """Configuration for Transformer2 architecture."""

    num_experts: int = 8  # Number of expert adapters
    expert_rank: int = 64  # Low-rank dimension for experts
    routing_hidden_dim: int = 256  # Hidden dimension for router
    routing_temperature: float = 1.0  # Temperature for routing softmax
    sparsity_coeff: float = 0.01  # L1 sparsity regularization for z-vectors
    load_balancing_coeff: float = 0.01  # Load balancing loss coefficient
    expert_dropout: float = 0.1  # Dropout for expert adapters
    use_residual: bool = True  # Add residual connection


@dataclass
class Transformer2Result:
    """Result from Transformer2 forward pass."""

    logits: torch.Tensor
    routing_weights: torch.Tensor  # z-vectors for each input
    expert_contributions: Dict[int, float]  # Per-expert usage stats
    auxiliary_loss: float  # Sparsity + load balancing loss
    metrics: Dict[str, float]


class ExpertAdapter(nn.Module):
    """
    Low-rank expert adapter for efficient task-specific modification.

    Uses down-projection followed by up-projection:
        x -> down(x) -> up(down(x))

    This is parameter-efficient: instead of d*d parameters,
    we only need d*r + r*d = 2*d*r parameters (where r << d).
    """

    def __init__(self, hidden_size: int, expert_rank: int, dropout: float = 0.1):
        """
        Initialize expert adapter.

        Args:
            hidden_size: Model hidden dimension
            expert_rank: Low-rank bottleneck dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_rank = expert_rank

        # Down projection: hidden_size -> expert_rank
        self.down_proj = nn.Linear(hidden_size, expert_rank, bias=False)

        # Up projection: expert_rank -> hidden_size
        self.up_proj = nn.Linear(expert_rank, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Initialize for near-identity behavior at start
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)  # Start with zero output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply expert adapter.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            adapter_output: [batch, seq_len, hidden_size]
        """
        down = self.down_proj(x)
        down = F.gelu(down)  # Non-linearity in bottleneck
        down = self.dropout(down)
        up = self.up_proj(down)
        return up


class Router(nn.Module):
    """
    Router network that computes routing weights (z-vectors).

    Takes pooled hidden states and produces a distribution over experts.
    The z-vector determines how much each expert contributes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        routing_hidden_dim: int = 256,
        temperature: float = 1.0,
    ):
        """
        Initialize router.

        Args:
            hidden_size: Model hidden dimension
            num_experts: Number of expert adapters
            routing_hidden_dim: Router hidden layer dimension
            temperature: Softmax temperature (lower = sharper routing)
        """
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature

        self.router = nn.Sequential(
            nn.Linear(hidden_size, routing_hidden_dim),
            nn.LayerNorm(routing_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(routing_hidden_dim, num_experts),
        )

    def forward(
        self, hidden_states: torch.Tensor, return_logits: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute routing weights from hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            return_logits: Also return raw logits before softmax

        Returns:
            routing_weights: [batch, num_experts]
            logits (optional): [batch, num_experts]
        """
        # Pool hidden states (mean over sequence)
        pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]

        # Compute routing logits
        logits = self.router(pooled)  # [batch, num_experts]

        # Apply temperature and softmax
        routing_weights = F.softmax(logits / self.temperature, dim=-1)

        if return_logits:
            return routing_weights, logits
        return routing_weights


class Transformer2(nn.Module):
    """
    Transformer^2: Two-Pass Inference with Dynamic Expert Routing.

    Architecture:
    - Base model (frozen): Original transformer weights
    - Expert adapters (trainable): Low-rank modifications per expert
    - Router (trainable): Determines expert weights per input

    Two-Pass Process:
    1. Pass 1: Run base model, compute routing weights z_i
    2. Pass 2: Apply weighted expert combination, generate output

    The expert composition formula is:
        output = base_output + sum(z_i * expert_i(hidden_states))

    Benefits:
    - Dynamic task-specific adaptation
    - No retraining of base model needed
    - Sparse expert activation possible
    - Interpretable routing decisions
    """

    def __init__(self, base_model: nn.Module, config: Transformer2Config = None):
        """
        Initialize Transformer2.

        Args:
            base_model: Pre-trained base model (will be frozen)
            config: Transformer2 configuration
        """
        super().__init__()
        self.config = config or Transformer2Config()
        self.base_model = base_model

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        # Detect hidden size from base model
        self.hidden_size = self._detect_hidden_size(base_model)

        # Initialize router
        self.router = Router(
            hidden_size=self.hidden_size,
            num_experts=self.config.num_experts,
            routing_hidden_dim=self.config.routing_hidden_dim,
            temperature=self.config.routing_temperature,
        )

        # Initialize expert adapters
        self.experts = nn.ModuleList(
            [
                ExpertAdapter(
                    hidden_size=self.hidden_size,
                    expert_rank=self.config.expert_rank,
                    dropout=self.config.expert_dropout,
                )
                for _ in range(self.config.num_experts)
            ]
        )

        # Track expert usage for load balancing
        self.register_buffer("expert_usage", torch.zeros(self.config.num_experts))
        self.usage_count = 0

    def _detect_hidden_size(self, model: nn.Module) -> int:
        """Detect hidden size from model config or architecture."""
        # Try common config attributes
        if hasattr(model, "config"):
            config = model.config
            for attr in ["hidden_size", "d_model", "n_embd", "dim"]:
                if hasattr(config, attr):
                    return getattr(config, attr)

        # Try to find from embedding layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                return module.embedding_dim
            if isinstance(module, nn.Linear) and "embed" in name.lower():
                return module.out_features

        # Default fallback
        return 768

    def compute_routing(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass 1: Compute routing weights from hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            routing_weights: [batch, num_experts]
            routing_logits: [batch, num_experts]
        """
        return self.router(hidden_states, return_logits=True)

    def apply_experts(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Pass 2: Apply weighted expert combination.

        Implements: output = sum(z_i * expert_i(hidden_states))

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            routing_weights: [batch, num_experts]

        Returns:
            expert_output: [batch, seq_len, hidden_size]
            expert_contributions: Dict mapping expert_id -> mean contribution
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        expert_output = torch.zeros_like(hidden_states)
        expert_contributions = {}

        for i, expert in enumerate(self.experts):
            # Compute this expert's output
            contribution = expert(hidden_states)  # [batch, seq, hidden]

            # Weight by routing
            weight = routing_weights[:, i].view(batch_size, 1, 1)
            weighted_contribution = weight * contribution

            expert_output = expert_output + weighted_contribution

            # Track contribution magnitude
            expert_contributions[i] = weighted_contribution.abs().mean().item()

        return expert_output, expert_contributions

    def compute_auxiliary_loss(
        self, routing_weights: torch.Tensor, routing_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute auxiliary losses for training.

        Includes:
        1. Sparsity loss: Encourage sparse expert activation (L1 on z)
        2. Load balancing: Encourage even expert usage

        Args:
            routing_weights: [batch, num_experts]
            routing_logits: [batch, num_experts]

        Returns:
            aux_loss: Scalar auxiliary loss
            loss_breakdown: Dict with individual loss components
        """
        # Sparsity loss: L1 norm encourages sparse routing
        sparsity_loss = routing_weights.abs().sum(dim=-1).mean()

        # Load balancing loss: Encourage even distribution across experts
        # Compute fraction of tokens routed to each expert
        expert_usage = routing_weights.mean(dim=0)  # [num_experts]
        # Target is uniform: 1/num_experts
        target = 1.0 / self.config.num_experts
        # MSE from uniform distribution
        load_balance_loss = ((expert_usage - target) ** 2).sum()

        # Update running expert usage stats
        with torch.no_grad():
            self.expert_usage = 0.99 * self.expert_usage + 0.01 * expert_usage
            self.usage_count += 1

        # Total auxiliary loss
        aux_loss = (
            self.config.sparsity_coeff * sparsity_loss
            + self.config.load_balancing_coeff * load_balance_loss
        )

        return aux_loss, {
            "sparsity_loss": sparsity_loss.item(),
            "load_balance_loss": load_balance_loss.item(),
            "aux_loss": aux_loss.item(),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_routing: bool = True,
        **kwargs,
    ) -> Transformer2Result:
        """
        Full two-pass forward.

        Pass 1: Get base model hidden states, compute routing weights
        Pass 2: Apply expert combination, generate output

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            return_routing: Include routing info in result
            **kwargs: Additional args for base model

        Returns:
            Transformer2Result with logits, routing, and metrics
        """
        # Pass 1: Get base model hidden states
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs
            )

        # Extract final hidden states
        if hasattr(base_outputs, "hidden_states"):
            hidden_states = base_outputs.hidden_states[-1]
        elif hasattr(base_outputs, "last_hidden_state"):
            hidden_states = base_outputs.last_hidden_state
        else:
            # Fallback: assume output is tuple with hidden states
            hidden_states = base_outputs[0]

        # Compute routing weights (Pass 1 result)
        routing_weights, routing_logits = self.compute_routing(hidden_states)

        # Apply experts (Pass 2)
        expert_output, expert_contributions = self.apply_experts(hidden_states, routing_weights)

        # Combine: residual connection
        if self.config.use_residual:
            adapted_hidden = hidden_states + expert_output
        else:
            adapted_hidden = expert_output

        # Generate logits
        if hasattr(self.base_model, "lm_head"):
            logits = self.base_model.lm_head(adapted_hidden)
        elif hasattr(self.base_model, "head"):
            logits = self.base_model.head(adapted_hidden)
        else:
            # Return adapted hidden states if no head found
            logits = adapted_hidden

        # Compute auxiliary loss
        aux_loss, loss_breakdown = self.compute_auxiliary_loss(routing_weights, routing_logits)

        # Compute metrics
        metrics = {
            "routing_entropy": self._routing_entropy(routing_weights),
            "routing_max": routing_weights.max(dim=-1).values.mean().item(),
            "routing_sparsity": (routing_weights < 0.1).float().mean().item(),
            **loss_breakdown,
        }

        return Transformer2Result(
            logits=logits,
            routing_weights=routing_weights if return_routing else None,
            expert_contributions=expert_contributions,
            auxiliary_loss=aux_loss.item(),
            metrics=metrics,
        )

    def _routing_entropy(self, routing_weights: torch.Tensor) -> float:
        """Compute entropy of routing distribution."""
        # H = -sum(p * log(p))
        eps = 1e-8
        entropy = -(routing_weights * torch.log(routing_weights + eps)).sum(dim=-1)
        return entropy.mean().item()

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about expert usage."""
        if self.usage_count == 0:
            return {}

        usage = self.expert_usage.cpu().numpy()
        return {
            "expert_usage": {i: float(u) for i, u in enumerate(usage)},
            "most_used_expert": int(usage.argmax()),
            "least_used_expert": int(usage.argmin()),
            "usage_std": float(usage.std()),
            "total_updates": self.usage_count,
        }

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_expert_weights(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Get expert adapter weights for analysis/saving."""
        weights = {}
        for i, expert in enumerate(self.experts):
            weights[i] = {
                "down_proj": expert.down_proj.weight.data.clone(),
                "up_proj": expert.up_proj.weight.data.clone(),
            }
        return weights

    def set_expert_weights(self, weights: Dict[int, Dict[str, torch.Tensor]]):
        """Set expert adapter weights from saved state."""
        for i, expert in enumerate(self.experts):
            if i in weights:
                expert.down_proj.weight.data = weights[i]["down_proj"]
                expert.up_proj.weight.data = weights[i]["up_proj"]


class ZVectorSparseRouter(Router):
    """
    Router with explicit sparsity control via top-k selection.

    Produces sparse z-vectors where only top-k experts are active.
    This reduces computation and improves interpretability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        routing_hidden_dim: int = 256,
        temperature: float = 1.0,
    ):
        """
        Initialize sparse router.

        Args:
            hidden_size: Model hidden dimension
            num_experts: Total number of experts
            top_k: Number of experts to activate per input
            routing_hidden_dim: Router hidden dimension
            temperature: Softmax temperature
        """
        super().__init__(hidden_size, num_experts, routing_hidden_dim, temperature)
        self.top_k = top_k

    def forward(
        self, hidden_states: torch.Tensor, return_logits: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute sparse routing weights with top-k selection.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            return_logits: Also return raw logits

        Returns:
            routing_weights: [batch, num_experts] (sparse)
            logits (optional): [batch, num_experts]
        """
        # Pool hidden states
        pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]

        # Compute routing logits
        logits = self.router(pooled)  # [batch, num_experts]

        # Select top-k experts
        topk_values, topk_indices = torch.topk(logits, self.top_k, dim=-1)

        # Create sparse routing weights
        routing_weights = torch.zeros_like(logits)

        # Softmax over top-k only
        topk_weights = F.softmax(topk_values / self.temperature, dim=-1)

        # Scatter back to full routing tensor
        routing_weights.scatter_(dim=-1, index=topk_indices, src=topk_weights)

        if return_logits:
            return routing_weights, logits
        return routing_weights


__all__ = [
    "Transformer2",
    "Transformer2Config",
    "Transformer2Result",
    "ExpertAdapter",
    "Router",
    "ZVectorSparseRouter",
]
