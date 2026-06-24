"""
Complete TRM × Titans-MAG Model

Integrates all components:
1. Titans-MAG Backbone (n_layers, causal Sliding Window + EMA-pool LMM + MAG)
2. TRM Wrapper (multi-pass reasoning, causal)
3. ACT Head (halt-probability head; reports halting but does NOT skip compute)
4. LM Head (vocabulary projection, tied with embeddings)

Size: ~222M parameters (resized 2026-06-24). Trains on an 8 GB GPU at b2/s128
(fp32 ~4.9 GB or bf16 ~4.5 GB); larger batch/seq needs the deferred efficiency work
(gradient checkpointing + the TRM's O(seq^2) attention).
"""

from typing import Dict, List, Optional

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

        # ACT Head (EMA buffers sized to the recursion depth: y0 + T_max steps)
        self.act_head = ACTHead(d_model, config.act_config, max_steps=config.trm_config.T_max + 1)

        # Language Modeling Head (tied with embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
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
        # Truncate input AND labels together to max_seq_len: the backbone truncates
        # input internally, so unsynced labels caused a CE/ACT shape mismatch (Codex #9).
        max_len = self.config.titans_config.max_seq_len
        if input_ids.shape[1] > max_len:
            input_ids = input_ids[:, :max_len]
            if labels is not None:
                labels = labels[:, :max_len]

        batch_size, seq_len = input_ids.shape

        # 1. Backbone forward pass
        features, loss_gate = self.backbone(input_ids, mask)

        # 2. TRM multi-pass reasoning
        y_history, z_history = self.trm(features)

        # 3. Compute outputs for each step
        step_logits = []
        halt_probs = []
        step_q = []  # full [batch, seq, 1] halt probs, kept for the ACT loss

        for t, (y_t, z_t) in enumerate(zip(y_history, z_history)):
            # Compute logits
            logits_t = self.lm_head(y_t)
            step_logits.append(logits_t)

            # Compute ACT halt probability
            q_t = self.act_head(z_t)
            step_q.append(q_t)
            halt_probs.append(q_t.mean(dim=[1, 2]))  # [batch]

        # 4. Determine halting (for inference)
        # halting_steps is REPORTED for inference but does NOT skip compute: every step
        # always runs and the final step's logits are always used (see act_head docstring).
        # Range is 1..len(halt_probs) = 1..T_max+1 because y0/z0 count as step 1.
        halting_steps = self._compute_halting_steps(halt_probs)

        # 5. Final output (last step)
        logits = step_logits[-1]

        # 6. Compute loss with optional deep supervision
        output = {"logits": logits}
        if labels is not None:
            output.update(self._supervised_loss(step_logits, step_q, labels, loss_gate))

        output["halting_steps"] = halting_steps
        output["halt_probs"] = halt_probs

        if return_all_steps:
            output["all_logits"] = step_logits
            output["all_y"] = y_history
            output["all_z"] = z_history

        return output

    def _supervised_loss(
        self,
        step_logits: List[torch.Tensor],
        step_q: List[torch.Tensor],
        labels: torch.Tensor,
        loss_gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """NEXT-token (shifted) CE + differentiable ACT loss + MAG gate loss.

        Same-position CE is degenerate for a causal LM (the model copies the token it
        sees via self-attention - Codex #2); logits[t] is scored against token t+1.
        ignore_index=-100 masks padding; the ACT correctness target uses the same
        shifted, padding-masked alignment (Codex #15).
        """
        vocab_size = self.config.titans_config.vocab_size
        shift_labels = labels[:, 1:].reshape(-1)

        out: Dict[str, torch.Tensor] = {}
        # Degenerate supervision: no next token exists (seq_len<=1) or every shifted
        # target is padding. CE-mean would be nan; return graph-connected zeros so the
        # step neither crashes nor poisons training (Codex #1). The gate loss still flows.
        if step_logits[-1].shape[1] <= 1 or (shift_labels != -100).sum() == 0:
            zero = step_logits[-1].sum() * 0.0
            return {
                "loss": zero + loss_gate,
                "loss_ce": zero,
                "loss_act": zero,
                "loss_gate": loss_gate,
            }

        def _ce(logits_t: torch.Tensor) -> torch.Tensor:
            return nn.functional.cross_entropy(
                logits_t[:, :-1, :].reshape(-1, vocab_size),
                shift_labels,
                reduction="mean",
                ignore_index=-100,
            )

        if self.config.trm_config.deep_supervision and len(step_logits) > 1:
            step_weights = self.config.trm_config.step_weights
            total_weight = sum(step_weights[: len(step_logits)])
            loss_ce = torch.tensor(0.0, device=labels.device)
            step_losses = []
            for logits_t, weight in zip(step_logits, step_weights):
                step_loss = _ce(logits_t)
                step_losses.append(step_loss.item())
                loss_ce = loss_ce + weight * step_loss
            loss_ce = loss_ce / total_weight
            out["step_losses"] = step_losses
        else:
            loss_ce = _ce(step_logits[-1])

        # Per-sample NEXT-token correctness (shifted, padding-masked) drives the ACT
        # target; the halt head trains via differentiable compute_act_loss. Rows with no
        # supervised target are EXCLUDED so they are not trained as "wrong" (Codex #2);
        # at least one row has a target here (the all-padding case returned early above).
        valid = (labels[:, 1:] != -100).float()
        has_target = valid.sum(dim=1) > 0
        denom = valid.sum(dim=1).clamp(min=1.0)
        act_step_losses = []
        for t, q_t in enumerate(step_q):
            pred = step_logits[t][:, :-1, :].argmax(dim=-1)
            is_correct = ((pred == labels[:, 1:]).float() * valid).sum(dim=1) / denom
            act_step_losses.append(
                self.act_head.compute_act_loss(q_t[has_target], t, is_correct[has_target])
            )
        loss_act = self.config.act_config.act_loss_weight * torch.stack(act_step_losses).mean()

        out["loss"] = loss_ce + loss_act + loss_gate
        out["loss_ce"] = loss_ce
        out["loss_act"] = loss_act
        out["loss_gate"] = loss_gate
        return out

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

    def reset_memory(self) -> None:
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
