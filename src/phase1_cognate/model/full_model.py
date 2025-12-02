"""
Complete TRM × Titans-MAG Model

Integrates all components:
1. Titans-MAG Backbone (8 layers, Sliding Window + LMM + MAG)
2. TRM Wrapper (multi-pass reasoning)
3. ACT Head (adaptive computation)
4. LM Head (vocabulary projection)

Target: ~25M parameters, fits in 6GB VRAM
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .act_head import ACTHead
from .model_config import Phase1Config
from .titans_mag import TitansMAGBackbone
from .trm_wrapper import TRMWrapper


class TRMTitansMAGModel(nn.Module):
    """
    Complete Phase 1 Model

    Architecture:
        Input → Titans-MAG Backbone → TRM Wrapper → LM Head → Output
                                     ↓
                                 ACT Head (halt decision)
    """

    def __init__(self, config: Phase1Config):
        super().__init__()
        self.config = config

        d_model = config.titans_config.d_model
        vocab_size = config.titans_config.vocab_size

        # Titans-MAG Backbone
        self.backbone = TitansMAGBackbone(config.titans_config)

        # TRM Wrapper
        self.trm = TRMWrapper(d_model, config.trm_config)

        # ACT Head
        self.act_head = ACTHead(d_model, config.act_config)

        # Language Modeling Head (tied with embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        """Tie LM head with token embeddings"""
        self.lm_head.weight = self.backbone.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-step reasoning

        Args:
            input_ids: [batch, seq_len] token IDs
            labels: [batch, seq_len] target tokens (optional)
            mask: Optional attention mask
            return_all_steps: Return intermediate steps

        Returns:
            dict with keys:
                - logits: [batch, seq_len, vocab_size] final output
                - loss: scalar (if labels provided)
                - step_losses: List[scalar] (if labels + deep_supervision)
                - halting_steps: [batch] number of steps taken
                - gate_loss: scalar MAG gate entropy loss
        """
        batch_size, seq_len = input_ids.shape

        # 1. Backbone forward pass
        features, loss_gate = self.backbone(input_ids, mask)

        # 2. TRM multi-pass reasoning
        y_history, z_history = self.trm(features)

        # 3. Compute outputs for each step
        step_logits = []
        halt_probs = []

        for t, (y_t, z_t) in enumerate(zip(y_history, z_history)):
            # Compute logits
            logits_t = self.lm_head(y_t)
            step_logits.append(logits_t)

            # Compute ACT halt probability
            q_t = self.act_head(z_t)
            halt_probs.append(q_t.mean(dim=[1, 2]))  # [batch]

        # 4. Determine halting (for inference)
        halting_steps = self._compute_halting_steps(halt_probs)

        # 5. Final output (last step)
        logits = step_logits[-1]

        # 6. Compute loss with optional deep supervision
        output = {"logits": logits}

        if labels is not None:
            vocab_size = self.config.titans_config.vocab_size

            # Deep supervision: weighted loss across all steps
            if self.config.trm_config.deep_supervision and len(step_logits) > 1:
                step_weights = self.config.trm_config.step_weights
                total_weight = sum(step_weights[: len(step_logits)])
                loss_ce = torch.tensor(0.0, device=labels.device)

                step_losses = []
                for t, (logits_t, weight) in enumerate(zip(step_logits, step_weights)):
                    # Clone logits to avoid graph reuse issues (M4 fix)
                    logits_clone = logits_t.clone()
                    step_loss = nn.functional.cross_entropy(
                        logits_clone.view(-1, vocab_size), labels.view(-1), reduction="mean"
                    )
                    step_losses.append(step_loss.item())
                    loss_ce = loss_ce + weight * step_loss

                # Normalize by total weight
                loss_ce = loss_ce / total_weight
                output["step_losses"] = step_losses
            else:
                # Final step only (standard training)
                loss_ce = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), labels.view(-1), reduction="mean"
                )

            # Add ACT loss (encourage halting at appropriate time)
            loss_act = self.config.act_config.act_loss_weight * (halting_steps.float().mean())

            # Total loss
            loss_total = loss_ce + loss_act + loss_gate

            output["loss"] = loss_total
            output["loss_ce"] = loss_ce
            output["loss_act"] = loss_act
            output["loss_gate"] = loss_gate

        output["halting_steps"] = halting_steps
        output["halt_probs"] = halt_probs

        if return_all_steps:
            output["all_logits"] = step_logits
            output["all_y"] = y_history
            output["all_z"] = z_history

        return output

    def _compute_halting_steps(self, halt_probs: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute number of steps taken per sample

        Args:
            halt_probs: List of [batch] halt probabilities

        Returns:
            halting_steps: [batch] number of steps (1 to T_max)
        """
        batch_size = halt_probs[0].shape[0]
        threshold = self.config.act_config.halt_threshold

        halting_steps = torch.full((batch_size,), len(halt_probs), device=halt_probs[0].device)

        # Find first step where halt_prob > threshold
        for t, q_t in enumerate(halt_probs):
            halted = q_t > threshold
            # Set halting_steps for samples that haven't halted yet
            not_halted = halting_steps == len(halt_probs)
            halting_steps[halted & not_halted] = t + 1

        return halting_steps

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters by component

        Returns:
            dict: {component_name: param_count}
        """
        counts = {
            "backbone": self.backbone.count_parameters(),
            "trm": self.trm.count_parameters(),
            "act_head": sum(p.numel() for p in self.act_head.parameters() if p.requires_grad),
            "lm_head": 0,  # Tied with embeddings
        }
        counts["total"] = sum(counts.values())
        return counts

    def reset_memory(self):
        """Reset LTM state (call between batches)"""
        self.backbone.reset_memory()

    def get_config_dict(self) -> dict:
        """Get configuration for W&B logging"""
        return self.config.to_dict()


def create_phase1_models(device: str = "cuda") -> Dict[str, TRMTitansMAGModel]:
    """
    Create all 3 specialized Phase 1 models

    Args:
        device: Device to place models on

    Returns:
        dict: {"reasoning": model1, "memory": model2, "speed": model3}
    """
    from .model_config import create_model_configs

    configs = create_model_configs()
    models = {}

    for spec, config in configs.items():
        # Set random seed
        torch.manual_seed(config.get_seed())

        # Create model
        model = TRMTitansMAGModel(config)

        # Move to device
        model = model.to(device)

        models[spec] = model

        # Log parameter count
        param_counts = model.count_parameters()
        print(f"\n{spec.upper()} Model:")
        for component, count in param_counts.items():
            print(f"  {component}: {count:,} params")

    return models


if __name__ == "__main__":
    # Test model creation
    print("Creating Phase 1 models...")

    models = create_phase1_models(device="cpu")

    print("\n✅ All 3 models created successfully!")
    print(f"Total models: {len(models)}")

    # Test forward pass
    print("\nTesting forward pass...")
    test_input = torch.randint(0, 32768, (2, 64))  # [batch=2, seq=64]

    for spec, model in models.items():
        output = model(test_input)
        print(f"\n{spec}:")
        print(f"  Output logits shape: {output['logits'].shape}")
        print(f"  Halting steps: {output['halting_steps']}")
