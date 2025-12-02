"""
Phase 3 Quiet-STaR Thought Generator

Generate parallel thought continuations at each token position.
Uses nucleus sampling (top-p) with temperature for diversity.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataclasses import ThoughtOutput


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

        # Generate thought length ONCE for all thoughts (ensures stackable tensors)
        thought_length = torch.randint(self.min_length, self.max_length + 1, (1,)).item()

        # Initialize storage
        all_thoughts = []
        all_log_probs = []
        all_thought_ids = []

        # Generate each thought independently (with same length)
        for _ in range(self.num_thoughts):
            thought, log_prob, ids = self._generate_single(
                input_ids, position, hidden_states, thought_length
            )
            all_thoughts.append(thought)
            all_log_probs.append(log_prob)
            all_thought_ids.append(ids)

        # Stack results
        thoughts = torch.stack(all_thoughts, dim=1)
        # Stack scalars with dim=0 to get [num_thoughts] tensor
        log_probs = torch.stack(all_log_probs, dim=0)

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
        thought_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Generate a single thought continuation."""
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Start from position
        current_ids = input_ids[:, : position + 1].clone()
        generated_ids = []
        log_probs_list = []

        # Generate tokens (using provided thought_length for consistent sizes)
        for step in range(thought_length):
            outputs = self.base_model(current_ids)
            logits = outputs.logits[:, -1, :] / self.temperature

            # Nucleus sampling
            probs = self._nucleus_sampling(logits)
            next_token = torch.multinomial(probs, num_samples=1)

            # Store results
            # ISS-005: Handle batch_size > 1 (use first batch item for IDs)
            generated_ids.append(next_token[0, 0].item())
            log_probs_list.append(torch.log(probs.gather(1, next_token)))

            # Append token
            current_ids = torch.cat([current_ids, next_token], dim=1)

        # Aggregate
        thought_hidden = outputs.last_hidden_state[:, -thought_length:, :]
        # ISS-005: Sum log probs per batch, then reduce to scalar for compatibility
        log_prob_sum = torch.stack(log_probs_list).sum(dim=0).squeeze(-1).mean()

        return thought_hidden, log_prob_sum, generated_ids

    def _nucleus_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply nucleus (top-p) sampling."""
        probs = F.softmax(logits, dim=-1)

        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
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
