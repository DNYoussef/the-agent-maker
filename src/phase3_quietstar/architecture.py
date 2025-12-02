"""
Phase 3 Quiet-STaR Architecture Components

Core components for thought generation and reasoning:
1. ThoughtGenerator - Generate 4-8 parallel thoughts per token
2. CoherenceScorer - Score thought quality (semantic, syntactic, predictive)
3. MixingHead - Attention-based thought integration
4. ThoughtInjector - Identify difficult positions for thought injection
5. QuietSTaRModel - Complete Quiet-STaR model wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ThoughtOutput:
    """Output from thought generation."""

    thoughts: torch.Tensor  # (batch, num_thoughts, thought_len)
    thought_ids: List[List[int]]  # Generated token IDs
    log_probs: torch.Tensor  # Log probabilities
    attention_weights: Optional[torch.Tensor] = None


@dataclass
class CoherenceScores:
    """Coherence scoring output."""

    semantic: torch.Tensor  # (batch, num_thoughts)
    syntactic: torch.Tensor  # (batch, num_thoughts)
    predictive: torch.Tensor  # (batch, num_thoughts)
    composite: torch.Tensor  # Weighted average


class ThoughtGenerator(nn.Module):
    """
    Generate parallel thought continuations at each token position.

    Uses nucleus sampling (top-p) with temperature for diversity.
    Generates 4-8 structured thoughts per position.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_thoughts: int = 4,
        max_length: int = 20,
        min_length: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_thoughts = num_thoughts
        self.max_length = max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_p = top_p

    def forward(
        self,
        input_ids: torch.Tensor,
        position: int,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> ThoughtOutput:
        """Generate thoughts at specified position."""
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize storage
        all_thoughts = []
        all_log_probs = []
        all_thought_ids = []

        # Generate each thought independently
        for _ in range(self.num_thoughts):
            thought, log_prob, ids = self._generate_single(
                input_ids, position, hidden_states
            )
            all_thoughts.append(thought)
            all_log_probs.append(log_prob)
            all_thought_ids.append(ids)

        # Stack results
        thoughts = torch.stack(all_thoughts, dim=1)
        log_probs = torch.stack(all_log_probs, dim=1)

        return ThoughtOutput(
            thoughts=thoughts,
            thought_ids=all_thought_ids,
            log_probs=log_probs,
        )

    def _generate_single(
        self,
        input_ids: torch.Tensor,
        position: int,
        hidden_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Generate a single thought continuation."""
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Start from position
        current_ids = input_ids[:, : position + 1].clone()
        generated_ids = []
        log_probs_list = []

        # Adaptive length (10-20 tokens)
        thought_length = torch.randint(
            self.min_length, self.max_length + 1, (1,)
        ).item()

        # Generate tokens
        for step in range(thought_length):
            outputs = self.base_model(current_ids)
            logits = outputs.logits[:, -1, :] / self.temperature

            # Nucleus sampling
            probs = self._nucleus_sampling(logits)
            next_token = torch.multinomial(probs, num_samples=1)

            # Store results
            # ISS-005: Handle batch_size > 1 (use first batch item for IDs)
            generated_ids.append(next_token[0, 0].item())
            log_probs_list.append(
                torch.log(probs.gather(1, next_token))
            )

            # Append token
            current_ids = torch.cat([current_ids, next_token], dim=1)

        # Aggregate
        thought_hidden = outputs.last_hidden_state[:, -thought_length:, :]
        # ISS-005: Sum log probs per batch, then reduce to scalar for compatibility
        log_prob_sum = torch.stack(log_probs_list).sum(dim=0).squeeze(-1).mean()

        return thought_hidden, log_prob_sum, generated_ids

    def _nucleus_sampling(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply nucleus (top-p) sampling."""
        probs = F.softmax(logits, dim=-1)

        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(
            probs, descending=True, dim=-1
        )
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff
        cutoff_mask = cumsum_probs > self.top_p
        cutoff_mask[:, 0] = False  # Keep at least one

        # Zero out low probability tokens
        sorted_probs[cutoff_mask] = 0.0

        # Renormalize
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Unsort
        unsorted_probs = torch.zeros_like(probs)
        unsorted_probs.scatter_(1, sorted_indices, sorted_probs)

        return unsorted_probs


class CoherenceScorer(nn.Module):
    """
    Score thought quality across 3 dimensions:
    - Semantic: 40% weight (embedding similarity)
    - Syntactic: 30% weight (grammar validity)
    - Predictive: 30% weight (helps next-token prediction)

    Returns composite score = 0.4×semantic + 0.3×syntactic + 0.3×predictive
    """

    def __init__(
        self,
        hidden_size: int,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Default weights
        self.weights = weights or {
            "semantic": 0.4,
            "syntactic": 0.3,
            "predictive": 0.3,
        }

        # Semantic scorer
        self.semantic_projection = nn.Linear(hidden_size, hidden_size)

        # Syntactic scorer
        self.syntactic_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Predictive scorer
        self.predictive_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        base_hidden: torch.Tensor,
        thought_hiddens: torch.Tensor,
        next_token_logits: Optional[torch.Tensor] = None,
    ) -> CoherenceScores:
        """
        Score thoughts against base hidden state.

        Args:
            base_hidden: (batch, hidden_size) - Base representation
            thought_hiddens: (batch, num_thoughts, thought_len, hidden_size)
            next_token_logits: (batch, vocab_size) - For predictive scoring
        """
        batch_size, num_thoughts = thought_hiddens.shape[:2]

        # Average thought representations
        thought_avg = thought_hiddens.mean(dim=2)

        # Semantic coherence (cosine similarity)
        semantic = self._semantic_coherence(base_hidden, thought_avg)

        # Syntactic coherence (grammar validity)
        syntactic = self._syntactic_coherence(thought_avg)

        # Predictive coherence (helps prediction)
        predictive = self._predictive_coherence(
            thought_avg, next_token_logits
        )

        # Composite score
        composite = (
            self.weights["semantic"] * semantic
            + self.weights["syntactic"] * syntactic
            + self.weights["predictive"] * predictive
        )

        return CoherenceScores(
            semantic=semantic,
            syntactic=syntactic,
            predictive=predictive,
            composite=composite,
        )

    def _semantic_coherence(
        self, base: torch.Tensor, thoughts: torch.Tensor
    ) -> torch.Tensor:
        """Semantic similarity via cosine distance."""
        base_proj = self.semantic_projection(base)
        base_norm = F.normalize(base_proj, p=2, dim=-1)

        thought_norm = F.normalize(thoughts, p=2, dim=-1)

        # Cosine similarity
        similarity = torch.bmm(
            thought_norm, base_norm.unsqueeze(-1)
        ).squeeze(-1)

        # Scale to [0, 1]
        return (similarity + 1.0) / 2.0

    def _syntactic_coherence(
        self, thoughts: torch.Tensor
    ) -> torch.Tensor:
        """Grammar validity via learned MLP."""
        return self.syntactic_mlp(thoughts).squeeze(-1)

    def _predictive_coherence(
        self, thoughts: torch.Tensor, logits: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """How much thought helps next-token prediction."""
        if logits is None:
            return torch.ones(
                thoughts.size(0), thoughts.size(1), device=thoughts.device
            )

        # Predict utility
        utility = self.predictive_head(thoughts).squeeze(-1)
        return torch.sigmoid(utility)


class MixingHead(nn.Module):
    """
    Attention-based integration of thoughts with base representation.

    Uses 8 attention heads with gating to blend base + thought hiddens.
    Includes residual connection and layer normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"

        # Multi-head attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        base_hidden: torch.Tensor,
        thought_hiddens: torch.Tensor,
        coherence_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix thoughts with base representation.

        Args:
            base_hidden: (batch, hidden_size)
            thought_hiddens: (batch, num_thoughts, hidden_size)
            coherence_scores: (batch, num_thoughts) - Attention weights

        Returns:
            mixed_hidden: (batch, hidden_size)
        """
        batch_size = base_hidden.size(0)

        # Query from base
        query = self.query(base_hidden).unsqueeze(1)

        # Keys and values from thoughts
        keys = self.key(thought_hiddens)
        values = self.value(thought_hiddens)

        # Reshape for multi-head attention
        query = self._split_heads(query)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scaled dot-product attention
        attention_logits = torch.matmul(
            query, keys.transpose(-2, -1)
        ) / (self.head_dim**0.5)

        # Apply coherence scores as bias
        attention_logits = attention_logits + coherence_scores.unsqueeze(
            1
        ).unsqueeze(1)

        # Attention weights
        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        context = torch.matmul(attention_weights, values)
        context = self._merge_heads(context).squeeze(1)

        # Output projection
        thought_representation = self.output(context)

        # Gating mechanism
        gate_input = torch.cat([base_hidden, thought_representation], dim=-1)
        gate_value = self.gate(gate_input)

        # Blend base + thoughts
        mixed = gate_value * thought_representation + (
            1 - gate_value
        ) * base_hidden

        # Residual + layer norm
        output = self.layer_norm(base_hidden + mixed)

        return output

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split into multiple heads."""
        batch_size = x.size(0)
        seq_len = x.size(1)
        return x.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge multiple heads."""
        batch_size = x.size(0)
        seq_len = x.size(2)
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )


class ThoughtInjector(nn.Module):
    """
    Identify difficult token positions for thought injection.

    Uses 3 difficulty metrics:
    1. Entropy (high uncertainty)
    2. Attention dispersion (spread attention)
    3. Loss (high prediction error)

    Injects thoughts when composite difficulty > threshold.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        min_interval: int = 3,
    ):
        super().__init__()
        self.threshold = threshold
        self.min_interval = min_interval
        self.last_injection = -min_interval

    def forward(
        self,
        logits: torch.Tensor,
        attention_weights: Optional[torch.Tensor],
        loss: Optional[torch.Tensor],
        position: int,
    ) -> bool:
        """Determine if thought injection needed at position."""
        # Check minimum interval
        if position - self.last_injection < self.min_interval:
            return False

        # Compute difficulty metrics
        entropy = self._compute_entropy(logits)
        dispersion = self._compute_dispersion(attention_weights)
        error = loss.item() if loss is not None else 0.0

        # Composite difficulty (normalized to [0, 1])
        difficulty = (
            0.4 * entropy + 0.3 * dispersion + 0.3 * min(error, 10.0) / 10.0
        )

        # Inject if above threshold
        if difficulty > self.threshold:
            self.last_injection = position
            return True

        return False

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute prediction entropy (high = uncertain)."""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        # Normalize to [0, 1]
        max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float32))
        return (entropy / max_entropy).mean().item()

    def _compute_dispersion(
        self, attention_weights: Optional[torch.Tensor]
    ) -> float:
        """Compute attention dispersion (high = spread out)."""
        if attention_weights is None:
            return 0.5  # Neutral value

        # Compute entropy of attention distribution
        entropy = -(
            attention_weights * torch.log(attention_weights + 1e-10)
        ).sum(dim=-1)

        # Normalize
        max_entropy = torch.log(torch.tensor(attention_weights.size(-1), dtype=torch.float32))
        return (entropy / max_entropy).mean().item()


class QuietSTaRModel(nn.Module):
    """
    Complete Quiet-STaR model wrapper.

    Integrates all components:
    - ThoughtGenerator
    - CoherenceScorer
    - MixingHead
    - ThoughtInjector
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        num_thoughts: int = 4,
        max_thought_length: int = 20,
        injection_threshold: float = 0.6,
        coherence_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size

        # Components
        self.thought_generator = ThoughtGenerator(
            base_model, num_thoughts=num_thoughts, max_length=max_thought_length
        )
        self.coherence_scorer = CoherenceScorer(
            hidden_size, weights=coherence_weights
        )
        self.mixing_head = MixingHead(hidden_size)
        self.thought_injector = ThoughtInjector(
            threshold=injection_threshold
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_thoughts: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional thought generation.

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) - For loss computation
            use_thoughts: Whether to generate and use thoughts

        Returns:
            Dictionary with logits, loss, and optional thought metrics
        """
        batch_size, seq_len = input_ids.shape

        # Base model forward
        outputs = self.base_model(input_ids)
        base_logits = outputs.logits
        base_hidden = outputs.last_hidden_state

        if not use_thoughts:
            loss = self._compute_loss(base_logits, labels) if labels is not None else None
            return {"logits": base_logits, "loss": loss}

        # Generate thoughts at difficult positions
        enhanced_hidden = base_hidden.clone()
        thought_positions = []
        coherence_scores_list = []

        for pos in range(seq_len - 1):
            # Check if injection needed
            inject = self.thought_injector(
                base_logits[:, pos, :],
                None,  # Attention weights optional
                None,  # Loss optional
                pos,
            )

            if inject:
                # Generate thoughts
                thought_output = self.thought_generator(
                    input_ids, pos, base_hidden[:, pos, :]
                )

                # Score coherence
                coherence = self.coherence_scorer(
                    base_hidden[:, pos, :],
                    thought_output.thoughts,
                    base_logits[:, pos + 1, :],
                )

                # Mix thoughts
                mixed = self.mixing_head(
                    base_hidden[:, pos, :],
                    thought_output.thoughts.mean(dim=2),
                    coherence.composite,
                )

                # Update hidden state
                enhanced_hidden[:, pos, :] = mixed

                thought_positions.append(pos)
                coherence_scores_list.append(
                    coherence.composite.mean().item()
                )

        # Final logits from enhanced hidden states
        final_logits = self.base_model.lm_head(enhanced_hidden)

        # Compute loss
        loss = self._compute_loss(final_logits, labels) if labels is not None else None

        return {
            "logits": final_logits,
            "loss": loss,
            "thought_positions": thought_positions,
            "avg_coherence": (
                sum(coherence_scores_list) / len(coherence_scores_list)
                if coherence_scores_list
                else 0.0
            ),
            "num_thoughts_used": len(thought_positions),
        }

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
