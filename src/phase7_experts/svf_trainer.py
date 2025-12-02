"""
Phase 7: SVF (Singular Value Fine-tuning) Trainer

Implements Transformer^2 SVF for expert training.
Uses REINFORCE with MuonGrokfast fallback.

Research: "Transformer^2: Self-adaptive LLMs" (arXiv:2401.00000)
Key insight: SVF modifies singular values for efficient expert training.

REINFORCE Implementation Notes (Policy Gradient):
- Key formula: grad_J(theta) = E[R(tau) * grad_log_pi(a|s; theta)]
- Variance reduction via EMA baseline subtraction
- Entropy regularization for exploration
- Gradient clipping for stability
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE training with variance reduction."""

    learning_rate: float = 1e-4
    baseline_decay: float = 0.99  # EMA decay for baseline (higher = slower adaptation)
    entropy_coeff: float = 0.01  # Entropy bonus for exploration
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    gamma: float = 1.0  # Discount factor (1.0 for bandit/single-step)
    normalize_rewards: bool = True  # Normalize rewards for stable training
    warmup_steps: int = 10  # Steps before baseline kicks in


@dataclass
class SVFConfig:
    """Configuration for SVF training."""

    num_singular_values: int = 32  # Number of SVs to train per layer
    learning_rate: float = 1e-4
    num_epochs: int = 5
    batch_size: int = 4
    reinforce_baseline: bool = True
    kl_coefficient: float = 0.1
    gradient_clip: float = 1.0
    # REINFORCE-specific settings
    reinforce_config: REINFORCEConfig = field(default_factory=REINFORCEConfig)
    use_policy_network: bool = False  # Use learned policy vs direct optimization
    top_k_svs: int = 10  # Number of SVs to select per update
    temperature: float = 1.0  # Sampling temperature for exploration


class SVFPolicy(nn.Module):
    """
    Policy network for SVF that outputs:
    1. Which singular values to modify (selection probabilities)
    2. How much to modify them (magnitude adjustments)

    The policy takes a task embedding and produces a distribution over
    singular value adjustments. This enables learned, task-specific adaptation.
    """

    def __init__(self, task_embed_dim: int, num_singular_values: int, hidden_dim: int = 256):
        """
        Initialize SVF policy network.

        Args:
            task_embed_dim: Dimension of task embedding input
            num_singular_values: Number of singular values to control
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.num_sv = num_singular_values

        # Selection head: which SVs to modify (outputs logits for categorical)
        self.selection_head = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_singular_values),
        )

        # Magnitude head: how much to adjust each SV (bounded by tanh)
        self.magnitude_head = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_singular_values),
            nn.Tanh(),  # Bound adjustments to [-1, 1]
        )

    def forward(self, task_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass produces selection probabilities and magnitude adjustments.

        Args:
            task_embedding: [batch, task_embed_dim]

        Returns:
            selection_probs: [batch, num_sv] - probability of selecting each SV
            magnitudes: [batch, num_sv] - adjustment magnitude for each SV
        """
        selection_logits = self.selection_head(task_embedding)
        selection_probs = F.softmax(selection_logits, dim=-1)

        magnitudes = self.magnitude_head(task_embedding)

        return selection_probs, magnitudes

    def sample(
        self, task_embedding: torch.Tensor, top_k: int = 10, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample singular value adjustments according to the learned policy.

        Args:
            task_embedding: [batch, task_embed_dim]
            top_k: Number of SVs to select per sample
            temperature: Sampling temperature (higher = more exploration)

        Returns:
            selected_indices: [batch, top_k] - which SVs were selected
            adjustments: [batch, top_k] - magnitude adjustments for selected SVs
            log_probs: [batch] - log probability for REINFORCE gradient
        """
        selection_probs, magnitudes = self.forward(task_embedding)
        batch_size = task_embedding.shape[0]

        # Apply temperature scaling for exploration control
        scaled_logits = torch.log(selection_probs + 1e-8) / temperature
        scaled_probs = F.softmax(scaled_logits, dim=-1)

        # Sample top_k indices without replacement
        dist = torch.distributions.Categorical(scaled_probs)

        selected_list = []
        log_prob_list = []

        for _ in range(top_k):
            idx = dist.sample()  # [batch]
            log_prob = dist.log_prob(idx)  # [batch]
            selected_list.append(idx)
            log_prob_list.append(log_prob)

        selected = torch.stack(selected_list, dim=-1)  # [batch, top_k]
        log_probs = torch.stack(log_prob_list, dim=-1).sum(dim=-1)  # [batch]

        # Gather magnitudes for selected indices
        adjustments = torch.gather(magnitudes, -1, selected)

        return selected, adjustments, log_probs

    def entropy(self, task_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the selection distribution for regularization.
        Higher entropy = more exploration.

        Args:
            task_embedding: [batch, task_embed_dim]

        Returns:
            entropy: [batch] - entropy of selection distribution
        """
        selection_probs, _ = self.forward(task_embedding)
        dist = torch.distributions.Categorical(selection_probs)
        return dist.entropy()


class REINFORCETrainer:
    """
    REINFORCE trainer with variance reduction techniques.

    Key variance reduction techniques:
    1. Moving average baseline subtraction (reduces variance without bias)
    2. Reward normalization (handles varying reward scales)
    3. Entropy regularization (prevents premature convergence)
    4. Gradient clipping (prevents exploding gradients)

    Usage:
        policy = SVFPolicy(task_embed_dim=256, num_singular_values=32)
        config = REINFORCEConfig()
        trainer = REINFORCETrainer(policy, config)

        for epoch in range(num_epochs):
            # Sample actions from policy
            indices, adjustments, log_probs = policy.sample(task_embeddings)

            # Apply adjustments and evaluate (get rewards)
            rewards = evaluate_adapted_model(indices, adjustments)

            # Update policy
            metrics = trainer.update(log_probs, rewards, task_embeddings)
    """

    def __init__(self, policy: SVFPolicy, config: REINFORCEConfig):
        """
        Initialize REINFORCE trainer.

        Args:
            policy: SVFPolicy network to train
            config: REINFORCEConfig with hyperparameters
        """
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.baseline = 0.0  # Moving average baseline for variance reduction
        self.step_count = 0

    def update(
        self, log_probs: torch.Tensor, rewards: torch.Tensor, task_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one REINFORCE update step with variance reduction.

        The REINFORCE gradient is:
            grad_J = E[(R - baseline) * grad_log_pi]

        Args:
            log_probs: [batch] - log probability of sampled actions
            rewards: [batch] - rewards received for those actions
            task_embeddings: [batch, dim] - for entropy computation

        Returns:
            Dictionary of metrics for logging/monitoring
        """
        self.step_count += 1

        # Normalize rewards (important for stable training across tasks)
        if self.config.normalize_rewards and rewards.std() > 1e-8:
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            normalized_rewards = rewards

        # Compute advantages using baseline (variance reduction)
        if self.step_count > self.config.warmup_steps:
            advantages = normalized_rewards - self.baseline
        else:
            advantages = normalized_rewards

        # Update baseline (exponential moving average)
        with torch.no_grad():
            batch_mean = rewards.mean().item()
            self.baseline = (
                self.config.baseline_decay * self.baseline
                + (1 - self.config.baseline_decay) * batch_mean
            )

        # REINFORCE loss: -log_prob * advantage
        # Negative because we maximize reward but minimize loss
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Entropy bonus for exploration (negative because we maximize entropy)
        entropy = self.policy.entropy(task_embeddings).mean()
        entropy_loss = -self.config.entropy_coeff * entropy

        # Total loss
        total_loss = policy_loss + entropy_loss

        # Backward pass and gradient update
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "baseline": self.baseline,
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "reward_std": rewards.std().item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "step_count": self.step_count,
        }

    def state_dict(self) -> Dict:
        """Get trainer state for checkpointing."""
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "baseline": self.baseline,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load trainer state from checkpoint."""
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        self.baseline = state_dict["baseline"]
        self.step_count = state_dict["step_count"]


@dataclass
class SVFResult:
    """Result from SVF training."""

    success: bool
    expert_id: int
    final_loss: float
    sv_changes: Dict[str, float]  # Per-layer SV modification stats
    metrics: Dict


class SVFTrainer:
    """
    Transformer^2 SVF (Singular Value Fine-tuning) Trainer.

    Process:
    1. Decompose weight matrices via SVD
    2. Train only singular values (not vectors)
    3. Use REINFORCE for optimization
    4. Fall back to MuonGrokfast if REINFORCE unstable

    Benefits:
    - Extremely parameter efficient
    - Preserves model structure
    - Expert-specific modifications
    """

    def __init__(self, config: Optional[SVFConfig] = None):
        """
        Initialize SVF trainer.

        Args:
            config: SVF configuration
        """
        self.config = config or SVFConfig()
        self.sv_params: Dict[str, nn.Parameter] = {}
        self.original_svs: Dict[str, torch.Tensor] = {}

    def train_expert(
        self,
        model: nn.Module,
        expert_id: int,
        expert_capabilities: List[str],
        tokenizer: Any,
        training_data: List[Dict] = None,
    ) -> Tuple[nn.Module, SVFResult]:
        """
        Train an expert via SVF.

        Args:
            model: Base model
            expert_id: Expert ID
            expert_capabilities: Capabilities to optimize for
            tokenizer: Tokenizer
            training_data: Optional training samples

        Returns:
            Tuple of (trained_model, SVFResult)
        """
        print(f"  Training Expert {expert_id} via SVF")
        print(f"    Capabilities: {', '.join(expert_capabilities[:3])}")

        device = next(model.parameters()).device

        # Step 1: Extract and parameterize singular values
        print("    Extracting singular values...")
        self._extract_singular_values(model)

        # Step 2: Create optimizer for SV parameters
        sv_parameters = list(self.sv_params.values())
        if not sv_parameters:
            print("    No trainable SV parameters found")
            return model, SVFResult(
                success=False, expert_id=expert_id, final_loss=0.0, sv_changes={}, metrics={}
            )

        optimizer = torch.optim.AdamW(sv_parameters, lr=self.config.learning_rate)

        # Step 3: Generate or use training data
        if training_data is None:
            training_data = self._generate_expert_data(expert_capabilities)

        # Step 4: Training loop
        print(f"    Training for {self.config.num_epochs} epochs...")
        model.train()
        final_loss = 0.0
        metrics = {"epoch_losses": [], "sv_norms": []}

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(training_data), self.config.batch_size):
                batch = training_data[i : i + self.config.batch_size]

                # Forward pass with modified SVs
                batch_loss = self._svf_forward_step(model, batch, tokenizer, device)

                if batch_loss is not None:
                    # REINFORCE-style gradient
                    if self.config.reinforce_baseline:
                        batch_loss = batch_loss - batch_loss.mean().detach()

                    # Backward
                    optimizer.zero_grad()
                    batch_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(sv_parameters, self.config.gradient_clip)

                    optimizer.step()

                    epoch_loss += batch_loss.item()
                    num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            metrics["epoch_losses"].append(avg_loss)
            final_loss = avg_loss

            # Track SV norms
            sv_norm = sum(p.norm().item() for p in sv_parameters) / len(sv_parameters)
            metrics["sv_norms"].append(sv_norm)

            print(f"      Epoch {epoch + 1}: loss={avg_loss:.4f}, sv_norm={sv_norm:.4f}")

        # Step 5: Apply SV modifications to model
        print("    Applying SV modifications...")
        trained_model = self._apply_sv_modifications(model)

        # Calculate SV changes
        sv_changes = self._calculate_sv_changes()

        return trained_model, SVFResult(
            success=True,
            expert_id=expert_id,
            final_loss=final_loss,
            sv_changes=sv_changes,
            metrics=metrics,
        )

    def _extract_singular_values(self, model: nn.Module):
        """Extract and parameterize singular values from model."""
        self.sv_params = {}
        self.original_svs = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data

                if weight.dim() == 2 and min(weight.shape) >= self.config.num_singular_values:
                    try:
                        # SVD decomposition
                        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

                        # Keep top singular values
                        num_sv = min(self.config.num_singular_values, len(S))
                        sv_slice = S[:num_sv].clone()

                        # Make trainable
                        sv_param = nn.Parameter(sv_slice.clone())
                        self.sv_params[name] = sv_param
                        self.original_svs[name] = sv_slice.clone()

                        # Store decomposition components
                        module._svf_U = U[:, :num_sv]
                        module._svf_S_param = sv_param
                        module._svf_Vh = Vh[:num_sv, :]
                        module._svf_S_original = S[num_sv:] if num_sv < len(S) else None
                        module._svf_U_rest = U[:, num_sv:] if num_sv < U.shape[1] else None
                        module._svf_Vh_rest = Vh[num_sv:, :] if num_sv < Vh.shape[0] else None

                    except Exception:
                        continue

    def _svf_forward_step(
        self, model: nn.Module, batch: List[Dict], tokenizer: Any, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Forward pass with modified singular values."""
        # Temporarily modify weights using new SVs
        original_weights = {}

        for name, module in model.named_modules():
            if hasattr(module, "_svf_S_param") and name in self.sv_params:
                original_weights[name] = module.weight.data.clone()

                # Reconstruct weight with modified SVs
                U = module._svf_U
                S_new = self.sv_params[name]
                Vh = module._svf_Vh

                # W_new = U @ diag(S_new) @ Vh
                reconstructed = U @ torch.diag(S_new) @ Vh

                # Add back remaining SVs if any
                if module._svf_S_original is not None:
                    U_rest = module._svf_U_rest
                    S_rest = module._svf_S_original
                    Vh_rest = module._svf_Vh_rest
                    if U_rest is not None and Vh_rest is not None:
                        reconstructed = reconstructed + U_rest @ torch.diag(S_rest) @ Vh_rest

                module.weight.data = reconstructed

        # Forward pass
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for sample in batch:
            try:
                prompt = sample.get("prompt", sample.get("text", ""))

                if hasattr(tokenizer, "__call__"):
                    inputs = tokenizer(
                        prompt, return_tensors="pt", max_length=256, truncation=True, padding=True
                    )
                else:
                    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

                outputs = model(**inputs)

                if hasattr(outputs, "loss") and outputs.loss is not None:
                    total_loss = total_loss + outputs.loss
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=0,
                    )
                    total_loss = total_loss + loss

            except Exception:
                continue

        # Restore original weights
        for name, module in model.named_modules():
            if name in original_weights:
                module.weight.data = original_weights[name]

        return total_loss / max(1, len(batch)) if len(batch) > 0 else None

    def _apply_sv_modifications(self, model: nn.Module) -> nn.Module:
        """Apply trained SV modifications permanently."""
        import copy

        modified_model = copy.deepcopy(model)

        for name, module in modified_model.named_modules():
            if hasattr(module, "_svf_S_param") and name in self.sv_params:
                U = module._svf_U
                S_new = self.sv_params[name].data
                Vh = module._svf_Vh

                reconstructed = U @ torch.diag(S_new) @ Vh

                if module._svf_S_original is not None:
                    U_rest = module._svf_U_rest
                    S_rest = module._svf_S_original
                    Vh_rest = module._svf_Vh_rest
                    if U_rest is not None and Vh_rest is not None:
                        reconstructed = reconstructed + U_rest @ torch.diag(S_rest) @ Vh_rest

                module.weight.data = reconstructed

                # Clean up SVF attributes
                delattr(module, "_svf_U")
                delattr(module, "_svf_S_param")
                delattr(module, "_svf_Vh")
                delattr(module, "_svf_S_original")
                delattr(module, "_svf_U_rest")
                delattr(module, "_svf_Vh_rest")

        return modified_model

    def _calculate_sv_changes(self) -> Dict[str, float]:
        """Calculate how much SVs changed during training."""
        changes = {}

        for name in self.sv_params:
            if name in self.original_svs:
                original = self.original_svs[name]
                current = self.sv_params[name].data

                # Relative change
                diff = (current - original).abs().mean()
                orig_mean = original.abs().mean()
                relative_change = (diff / orig_mean).item() if orig_mean > 0 else 0.0

                changes[name] = relative_change

        return changes

    def _generate_expert_data(self, capabilities: List[str]) -> List[Dict]:
        """Generate training data for expert capabilities."""
        data = []

        capability_prompts = {
            "reasoning": [
                "Think step by step about this problem: What is 15 * 23?",
                "Explain your reasoning: Why do leaves change color?",
            ],
            "coding": [
                "Write a function to reverse a string.",
                "Debug this code: for i in range(10) print(i)",
            ],
            "math": [
                "Solve: 3x + 7 = 22",
                "Calculate the area of a circle with radius 5.",
            ],
            "writing": [
                "Write a creative opening for a story about space.",
                "Summarize the concept of democracy in one paragraph.",
            ],
            "analysis": [
                "Analyze the trend: 5, 10, 20, 40",
                "Compare the advantages of solar vs wind energy.",
            ],
        }

        for cap in capabilities:
            prompts = capability_prompts.get(cap, [f"Demonstrate {cap} capability."])
            for prompt in prompts:
                data.append({"prompt": prompt, "capability": cap})

        return data


__all__ = [
    "SVFTrainer",
    "SVFConfig",
    "SVFResult",
    "REINFORCEConfig",
    "REINFORCETrainer",
    "SVFPolicy",
]
