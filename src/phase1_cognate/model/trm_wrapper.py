"""
TRM (Transformer Recursive Memory) Wrapper

Multi-pass reasoning with iterative refinement of latent state (z) and answer (y).

Components:
- g_φ (Refiner): Updates latent state via micro-steps
- h_ψ (Updater): Updates answer from refined latent
- Recursion Controller: Manages T_max iterations with early stopping

Key Features:
- Detached recursion for memory efficiency
- Deep supervision (loss at each step)
- Micro-step refinement
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .model_config import TRMConfig


class LatentRefiner(nn.Module):
    """
    g_φ: Refine latent state via micro-steps

    Uses cross-attention to features and previous answer.
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Cross-attention to features
        self.feature_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Cross-attention to answer
        self.answer_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # MLP for update
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
        )

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, z: torch.Tensor, features: torch.Tensor, y: torch.Tensor, n_steps: int = 2
    ) -> torch.Tensor:
        """
        Refine latent via n micro-steps

        Args:
            z: [batch, seq_len, d_model] latent state
            features: [batch, seq_len, d_model] backbone features
            y: [batch, seq_len, d_model] previous answer
            n_steps: Number of micro-steps

        Returns:
            z_refined: [batch, seq_len, d_model]
        """
        for _ in range(n_steps):
            # Attend to features
            feat_context, _ = self.feature_attn(z, features, features)

            # Attend to answer
            ans_context, _ = self.answer_attn(z, y, y)

            # Combine contexts
            combined = torch.cat([z, feat_context, ans_context], dim=-1)

            # Update z
            delta = self.mlp(combined)
            z = self.norm(z + delta)

        return z


class AnswerUpdater(nn.Module):
    """
    h_ψ: Update answer from refined latent

    Simple MLP that maps refined z to new answer y.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, y_prev: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Update answer from latent

        Args:
            y_prev: [batch, seq_len, d_model] previous answer
            z: [batch, seq_len, d_model] refined latent

        Returns:
            y_new: [batch, seq_len, d_model]
        """
        combined = torch.cat([y_prev, z], dim=-1)
        delta = self.mlp(combined)
        y_new = self.norm(y_prev + delta)
        return y_new


class TRMWrapper(nn.Module):
    """
    Complete TRM Wrapper

    Manages multi-pass reasoning loop with:
    - Latent refinement (g_φ)
    - Answer update (h_ψ)
    - Detached recursion
    """

    def __init__(self, d_model: int, config: TRMConfig):
        super().__init__()
        self.config = config
        self.d_model = d_model

        # Components
        self.refiner = LatentRefiner(d_model)
        self.updater = AnswerUpdater(d_model)

        # Initial projection (features → z0)
        self.z0_proj = nn.Linear(d_model, d_model)

        # Initial projection (z0 → y0)
        self.y0_proj = nn.Linear(d_model, d_model)

    def forward(
        self, features: torch.Tensor, max_steps: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Run multi-pass reasoning

        Args:
            features: [batch, seq_len, d_model] from backbone
            max_steps: Override T_max (optional)

        Returns:
            y_history: List of [batch, seq_len, d_model] answer states
            z_history: List of [batch, seq_len, d_model] latent states
        """
        if max_steps is None:
            max_steps = self.config.T_max

        # Initialize z0 and y0
        z = self.z0_proj(features)
        y = self.y0_proj(z)

        y_history = [y]
        z_history = [z]

        # For recursion, we'll use features but need to avoid graph reuse issues
        # We pass features to the first refiner call normally, then detached versions after
        features_first = features
        features_detached = features.detach()

        # Recursion loop
        for t in range(max_steps):
            # Refine latent
            # Use non-detached features for first iteration, detached for rest
            feat_to_use = features_first if t == 0 else features_detached
            z_refined = self.refiner(z, feat_to_use, y, n_steps=self.config.micro_steps)

            # Update answer
            y_new = self.updater(y, z_refined)

            # Store
            y_history.append(y_new)
            z_history.append(z_refined)

            # Detach for next iteration (if configured)
            if self.config.detach_between_steps and t < max_steps - 1:
                y = y_new.detach()
                z = z_refined.detach()
            else:
                y = y_new
                z = z_refined

        return y_history, z_history

    def get_step_weights(self) -> List[float]:
        """Get loss weights for deep supervision"""
        return self.config.step_weights

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
