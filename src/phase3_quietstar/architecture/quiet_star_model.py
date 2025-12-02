"""
Phase 3 Quiet-STaR Model

Complete Quiet-STaR model wrapper integrating all components.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .coherence_scorer import CoherenceScorer
from .mixing_head import MixingHead
from .thought_generator import ThoughtGenerator
from .thought_injector import ThoughtInjector


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
        self.coherence_scorer = CoherenceScorer(hidden_size, weights=coherence_weights)
        self.mixing_head = MixingHead(hidden_size)
        self.thought_injector = ThoughtInjector(threshold=injection_threshold)

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
                thought_output = self.thought_generator(input_ids, pos, base_hidden[:, pos, :])

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
                coherence_scores_list.append(coherence.composite.mean().item())

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

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
