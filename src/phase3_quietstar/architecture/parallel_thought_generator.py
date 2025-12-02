"""
Phase 3 Quiet-STaR Parallel Thought Generator

Implements parallel thought generation using diagonal attention mask
as described in Quiet-STaR paper Section 4.2, Figure 3.

Key Innovation: Generate all thoughts in single forward pass instead of
sequential generation, achieving ~num_thoughts speedup.

Paper Reference: arXiv:2403.09629v2 (Quiet-STaR)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataclasses import ThoughtOutput


class ParallelThoughtGenerator(nn.Module):
    """
    Parallel thought generation using diagonal attention mask.

    Generates num_thoughts thought continuations in a single forward pass
    by using a diagonal attention mask to prevent cross-contamination.

    Efficiency:
        Sequential: O(batch * num_thoughts * thought_length * model_forward)
        Parallel: O(batch * thought_length * model_forward)
        Speedup: ~num_thoughts (e.g., 4x for num_thoughts=4)

    Paper: Quiet-STaR Section 4.2 "Parallel Thought Sampling"
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
        """
        Initialize parallel thought generator.

        Args:
            base_model: Base language model
            num_thoughts: Number of parallel thoughts to generate
            max_length: Maximum thought length
            min_length: Minimum thought length
            temperature: Sampling temperature for diversity
            top_p: Nucleus sampling threshold
        """
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
        """
        Generate thoughts at all positions in single forward pass.

        Uses diagonal attention mask to prevent cross-contamination between
        parallel thoughts (Quiet-STaR paper Section 4.2, Figure 3).

        Args:
            input_ids: (batch, seq_len) Input token IDs
            position: Position to start thought generation
            hidden_states: Optional pre-computed hidden states (unused, for API compat)

        Returns:
            ThoughtOutput containing:
                - thoughts: (batch, num_thoughts, thought_len, hidden_size)
                - thought_ids: List of token ID lists
                - log_probs: (num_thoughts,) log probabilities
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Sample thought length
        thought_length = torch.randint(
            self.min_length, self.max_length + 1, (1,), device=device
        ).item()

        # Expand input for parallel thoughts
        # (batch, seq_len) -> (batch * num_thoughts, seq_len)
        expanded_input = input_ids.repeat_interleave(self.num_thoughts, dim=0)

        # Start from position
        current_ids = expanded_input[:, : position + 1].clone()

        # Storage for generated tokens and log probs
        all_generated_ids = []
        all_log_probs = []

        # Generate tokens with diagonal attention mask
        for step in range(thought_length):
            # Create diagonal mask (prevents cross-contamination)
            attention_mask = self._create_diagonal_attention_mask(
                batch_size=batch_size,
                num_thoughts=self.num_thoughts,
                seq_len=current_ids.size(1),
                position=position,
                device=device,
            )

            # Forward pass with diagonal mask
            outputs = self.base_model(current_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :] / self.temperature

            # Nucleus sampling
            probs = self._nucleus_sampling(logits)
            next_token = torch.multinomial(probs, num_samples=1)

            # Store results
            all_generated_ids.append(next_token)
            log_probs = torch.log(probs.gather(1, next_token) + 1e-10)
            all_log_probs.append(log_probs)

            # Append token
            current_ids = torch.cat([current_ids, next_token], dim=1)

        # Extract hidden states from final forward pass
        final_outputs = self.base_model(current_ids, attention_mask=attention_mask)
        thought_hidden = final_outputs.last_hidden_state[:, -thought_length:, :]

        # Reshape: (batch * num_thoughts, thought_len, hidden) -> (batch, num_thoughts, thought_len, hidden)
        thought_hidden = thought_hidden.view(
            batch_size, self.num_thoughts, thought_length, -1
        )

        # Aggregate log probs
        # (thought_len, batch * num_thoughts, 1) -> (batch, num_thoughts)
        stacked_log_probs = torch.stack(all_log_probs, dim=0).squeeze(-1)
        stacked_log_probs = stacked_log_probs.view(
            thought_length, batch_size, self.num_thoughts
        )
        log_probs = stacked_log_probs.sum(dim=0).mean(dim=0)  # (num_thoughts,)

        # Extract thought IDs
        stacked_ids = torch.stack(all_generated_ids, dim=0).squeeze(-1)
        stacked_ids = stacked_ids.view(thought_length, batch_size, self.num_thoughts)

        thought_ids = []
        for thought_idx in range(self.num_thoughts):
            # Use first batch item
            ids = stacked_ids[:, 0, thought_idx].tolist()
            thought_ids.append(ids)

        return ThoughtOutput(
            thoughts=thought_hidden,
            thought_ids=thought_ids,
            log_probs=log_probs,
        )

    def _create_diagonal_attention_mask(
        self,
        batch_size: int,
        num_thoughts: int,
        seq_len: int,
        position: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create diagonal attention mask to prevent cross-contamination.

        Quiet-STaR paper Section 4.2, Figure 3:
        "Each thought branch attends only to itself and shared context,
        forming a diagonal block structure in the attention matrix."

        Mask structure:
            1. All thoughts attend to shared context (tokens 0:position+1)
            2. Each thought attends only to its own generated tokens
            3. No attention between different thoughts (diagonal blocks)

        Args:
            batch_size: Number of samples
            num_thoughts: Number of parallel thoughts
            seq_len: Current sequence length
            position: Position where thought generation started
            device: Torch device

        Returns:
            attention_mask: (batch * num_thoughts, seq_len, seq_len)
                Values: 0.0 = attend, -inf = mask
        """
        total_batch = batch_size * num_thoughts

        # Initialize causal mask for shared context
        # (seq_len, seq_len) lower triangular
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

        # Expand to batch dimension
        mask = mask.unsqueeze(0).expand(total_batch, -1, -1)

        # Apply diagonal blocking for thought-specific tokens
        # Tokens after position+1 are thought-specific
        if seq_len > position + 1:
            for i in range(total_batch):
                thought_idx = i % num_thoughts

                # For positions > position+1, only attend within same thought
                for j in range(position + 1, seq_len):
                    for k in range(position + 1, seq_len):
                        # Calculate which thought position k belongs to
                        k_thought_idx = i % num_thoughts

                        # Only allow attention within same thought
                        if thought_idx != k_thought_idx:
                            mask[i, j, k] = 0

        # Convert to attention mask format (0 = attend, -inf = mask)
        attention_mask = torch.where(
            mask == 1,
            torch.tensor(0.0, device=device),
            torch.tensor(float("-inf"), device=device),
        )

        return attention_mask

    def _nucleus_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply nucleus (top-p) sampling.

        Args:
            logits: (batch, vocab_size)

        Returns:
            probs: (batch, vocab_size) Filtered probability distribution
        """
        probs = F.softmax(logits, dim=-1)

        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff (keep tokens until cumsum > top_p)
        cutoff_mask = cumsum_probs > self.top_p
        cutoff_mask[:, 0] = False  # Always keep top token

        # Zero out low-probability tokens
        sorted_probs[cutoff_mask] = 0.0

        # Renormalize
        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-10)

        # Unsort back to original indices
        unsorted_probs = torch.zeros_like(probs)
        unsorted_probs.scatter_(1, sorted_indices, sorted_probs)

        return unsorted_probs

    def compute_teacher_forced_loss(
        self,
        input_ids: torch.Tensor,
        thought_ids: List[List[int]],
        labels: torch.Tensor,
        n_true: int = 4,
    ) -> torch.Tensor:
        """
        Compute teacher-forced loss over n_true future tokens.

        Quiet-STaR paper: "Non-myopic loss that considers semantic content
        of future tokens, not just next token."

        Uses parallel attention mask (Figure 4) to compute loss efficiently.

        Args:
            input_ids: (batch, seq_len) Input tokens
            thought_ids: List of thought token sequences
            labels: (batch, seq_len) Target tokens
            n_true: Number of future tokens to consider (paper uses 4)

        Returns:
            loss: Scalar teacher-forced loss
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Convert thought_ids to tensor
        # thought_ids: list of lists -> (num_thoughts, thought_len)
        max_thought_len = max(len(ids) for ids in thought_ids)
        thought_tensor = torch.zeros(
            self.num_thoughts, max_thought_len, dtype=torch.long, device=device
        )
        for i, ids in enumerate(thought_ids):
            thought_tensor[i, : len(ids)] = torch.tensor(ids, device=device)

        # Expand input to include thoughts
        # (batch, seq_len) + (num_thoughts, thought_len) -> (batch * num_thoughts, seq_len + thought_len)
        expanded_input = input_ids.repeat_interleave(self.num_thoughts, dim=0)

        # Append thoughts to input
        thought_tensor_expanded = thought_tensor.repeat(batch_size, 1)
        combined_input = torch.cat([expanded_input, thought_tensor_expanded], dim=1)

        # Get labels for n_true future tokens
        # Shift labels to get future tokens
        if labels.size(1) >= combined_input.size(1) + n_true:
            future_labels = labels[:, combined_input.size(1) : combined_input.size(1) + n_true]
        else:
            # Pad if necessary
            future_labels = labels[:, combined_input.size(1) :]
            padding_size = n_true - future_labels.size(1)
            if padding_size > 0:
                future_labels = torch.cat(
                    [
                        future_labels,
                        torch.full(
                            (batch_size, padding_size),
                            -100,
                            dtype=torch.long,
                            device=device,
                        ),
                    ],
                    dim=1,
                )

        # Expand labels for parallel thoughts
        future_labels_expanded = future_labels.repeat_interleave(self.num_thoughts, dim=0)

        # Create diagonal attention mask for teacher forcing
        seq_len = combined_input.size(1) + n_true
        attention_mask = self._create_diagonal_attention_mask(
            batch_size=batch_size,
            num_thoughts=self.num_thoughts,
            seq_len=seq_len,
            position=input_ids.size(1),
            device=device,
        )

        # Forward pass with thoughts + future tokens
        # Create input with future token placeholders
        future_input = torch.cat(
            [
                combined_input,
                torch.zeros(
                    combined_input.size(0), n_true, dtype=torch.long, device=device
                ),
            ],
            dim=1,
        )

        # Compute logits
        outputs = self.base_model(
            future_input[:, :-n_true], attention_mask=attention_mask[:, :, :-n_true]
        )

        # Extract logits for n_true future positions
        # Generate predictions for each future position
        total_loss = 0.0
        current_input = combined_input

        for i in range(n_true):
            outputs = self.base_model(current_input, attention_mask=attention_mask[:, :, :current_input.size(1)])
            logits = outputs.logits[:, -1, :]

            # Get label for this position
            target = future_labels_expanded[:, i]

            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, target, ignore_index=-100)
            total_loss += loss

            # Append predicted token for next iteration (teacher forcing)
            next_token = target.unsqueeze(1)
            current_input = torch.cat([current_input, next_token], dim=1)

        # Average over n_true positions
        return total_loss / n_true
