# MuonGrokfast Developer Guide

**Version**: 1.0.0
**Authors**: Agent Forge V2 Team
**Last Updated**: 2025-01-15

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [API Reference](#api-reference)
5. [Phase-Specific Usage](#phase-specific-usage)
6. [Advanced Topics](#advanced-topics)
7. [Testing](#testing)
8. [W&B Integration](#wb-integration)
9. [Troubleshooting](#troubleshooting)

---

## Overview

**MuonGrokfast** is a unified optimizer system that synergistically combines three complementary optimization techniques:

### 1. Grokfast (Time-Spectrum Filtering)
**Paper**: "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"

**Algorithm** (Grokfast-EMA):
```python
μ_t = α·μ_{t-1} + (1-α)·g_t  # EMA of gradients
ĝ_t = g_t + λ·μ_t            # Filtered gradient
```

**Effect**: Emphasizes slow (low-frequency) gradient components, dampens noise, accelerates "grokking" phenomenon (sudden generalization after overfitting).

**Hyperparameters**:
- `α` (alpha): EMA decay ∈ [0.9, 0.99], default: 0.98
- `λ` (lambda): Amplification factor ∈ [0.05, 2.0], default: varies by phase

### 2. Muon (Space-Geometry Orthogonalization)
**Paper**: "Muon: Matrix-Structured Optimizer for Neural Networks" (2505.23737v1.pdf)

**Algorithm**:
```python
m_t = β·m_{t-1} + (1-β)·ĝ_t  # Momentum accumulation
M_ortho = NewtonSchulz(m_t)  # Semi-orthogonalization
W_{t+1} = W_t - η·M_ortho    # Update
```

**Effect**: Spreads updates across rare spatial directions via semi-orthogonalization, prevents low-rank collapse, maintains diversity in parameter space.

**Hyperparameters**:
- `η` (muon_lr): Learning rate ∈ [1e-4, 0.01], default: varies by phase
- `β` (momentum): Momentum coefficient, default: 0.95
- `k` (ns_steps): Newton-Schulz iterations, default: 5

### 3. QK-Clip / MuonClip (Attention Safety Rails)
**User Innovation**: Bounds attention logits during aggressive optimization.

**Algorithm**:
```python
for head h in attention_heads:
    if ||W_h + ΔW_h|| > τ:
        ΔW_h ← (τ / ||W_h + ΔW_h||) · ΔW_h
```

**Effect**: Prevents unbounded attention logits when optimizer dynamics push W_Q or W_K norms too high, critical for RL training (Phase 3, 7).

**Hyperparameters**:
- `τ` (qk_clip_threshold): Norm threshold, default: 30.0
- `per_head`: Boolean, respects MLA/decoupled-RoPE if True

### Synergy

**Time-Spectrum (Grokfast)** + **Space-Geometry (Muon)** = **Faster generalization, reduced low-rank collapse, smoother training trajectories**.

- Grokfast filters temporal history → stable, persistent gradients
- Muon orthogonalizes spatial directions → diverse, full-rank updates
- QK-clip provides safety → prevents RL instability

---

## Quick Start

### Installation

```bash
# MuonGrokfast is in cross-phase/ directory
# Ensure it's in your Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/agent-forge-v2/cross-phase"
```

### Basic Usage

```python
from mugrokfast_optimizer import MuonGrokfast
from mugrokfast_config import MuGrokConfig

# Option 1: Use phase-specific preset
config = MuGrokConfig.from_phase(1)  # Phase 1: Cognate

# Option 2: Custom configuration
config = MuGrokConfig(
    muon_lr=0.01,
    grokfast_alpha=0.98,
    grokfast_lambda=0.05,
    use_qk_clip=True,
    phase_name="custom"
)

# Create optimizer
optimizer = MuonGrokfast(model.parameters(), config=config)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Phase-Specific Quick Start

```python
# Phase 1: Cognate (TinyTitan pretraining)
config = MuGrokConfig.from_phase(1)
optimizer = MuonGrokfast(model.parameters(), config=config)

# Phase 3: Quiet-STaR (REINFORCE training)
config = MuGrokConfig.from_phase(3)
optimizer = MuonGrokfast(model.parameters(), config=config)

# Phase 5: Forge (BitNet + STE)
config = MuGrokConfig.from_phase(5)  # STE mode enabled automatically
optimizer = MuonGrokfast(model.parameters(), config=config)

# Phase 6: Baking (Prompt tuning)
config = MuGrokConfig.from_phase(6)
config.enable_muon = False  # Stage 1: 1-D prompt embeddings
optimizer = MuonGrokfast(model.parameters(), config=config)

# Phase 7: Transformer² (SVF training)
config = MuGrokConfig.from_phase(7)  # Fallback-only (z vectors are 1-D)
optimizer = MuonGrokfast(model.parameters(), config=config)
```

---

## Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Model Parameters + Raw Gradients                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: GROKFAST PREFILTER (Time-Spectrum)                    │
│                                                                  │
│   μ_t = α·μ_{t-1} + (1-α)·g_t   (EMA)                          │
│   ĝ_t = g_t + λ·μ_t             (Filtered gradient)            │
│                                                                  │
│   Effect: Emphasizes slow gradient components                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: PARAMETER ROUTING                                      │
│                                                                  │
│   if param.ndim >= 2 and enable_muon:                          │
│       → PATH A: Muon (space-geometry)                           │
│   else:                                                          │
│       → PATH B: Fallback (AdamW/Lion)                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│ PATH A: Muon     │    │ PATH B: Fallback │
│                  │    │                  │
│ 1. Momentum      │    │ AdamW / Lion     │
│ 2. Newton-Schulz │    │ (1-D params)     │
│ 3. Orthogonalize │    │                  │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: QK-CLIP SAFETY RAILS (Attention)                      │
│                                                                  │
│   if is_qk_projection and ||W_h + ΔW_h|| > τ:                  │
│       ΔW_h ← (τ / ||W_h + ΔW_h||) · ΔW_h                        │
│                                                                  │
│   Effect: Bounds attention logits, prevents RL instability      │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Apply Updates                                           │
│                                                                  │
│   W_{t+1} = W_t - lr · ΔW                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Parameter Routing Logic

```python
def route_parameter(param):
    """
    Automatic routing based on parameter dimensionality.

    Returns:
        "muon"     : 2-D parameter (weight matrices)
        "fallback" : 1-D parameter (bias, LayerNorm, embeddings)
    """
    if param.ndim >= 2 and config.enable_muon:
        return "muon"
    else:
        return "fallback"
```

**Examples**:
- `W_Q` (4096 × 4096) → Muon
- `W_K` (4096 × 4096) → Muon
- `W_MLP` (4096 × 16384) → Muon
- `bias` (4096,) → Fallback
- `LayerNorm.weight` (4096,) → Fallback
- `z_vector` (4096,) → Fallback (Phase 7 SVF)

---

## API Reference

### MuGrokConfig

```python
@dataclass
class MuGrokConfig:
    """Configuration for MuonGrokfast optimizer."""

    # Muon settings
    muon_lr: float = 0.01
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_ns_coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    muon_ste_mode: bool = False  # Phase 5 STE compatibility

    # Grokfast settings
    grokfast_alpha: float = 0.98
    grokfast_lambda: float = 0.05

    # QK-Clip settings
    use_qk_clip: bool = True
    qk_clip_threshold: float = 30.0
    qk_clip_per_head: bool = True

    # Fallback optimizer
    fallback_type: Literal["adamw", "lion"] = "adamw"
    fallback_lr: float = 1e-4
    fallback_betas: Tuple[float, float] = (0.9, 0.999)
    fallback_weight_decay: float = 0.01

    # Ablations
    enable_grokfast: bool = True
    enable_muon: bool = True
    enable_muon_clip: bool = True

    # Phase-specific
    kl_coefficient: Optional[float] = None
    phase_name: str = "unknown"

    @classmethod
    def from_phase(cls, phase: int, **overrides) -> "MuGrokConfig":
        """Create phase-specific config."""
        ...
```

### MuonGrokfast

```python
class MuonGrokfast(Optimizer):
    """Muon × Grokfast optimizer."""

    def __init__(self, params, config: Optional[MuGrokConfig] = None, **kwargs):
        """
        Initialize optimizer.

        Args:
            params: Model parameters
            config: MuGrokConfig instance
            **kwargs: Override config values
        """
        ...

    def step(self, closure=None):
        """
        Perform optimization step.

        Algorithm:
        1. Grokfast prefilter (all params)
        2. Parameter routing (2-D → Muon, 1-D → fallback)
        3. QK-clip safety rails

        Args:
            closure: Optional closure for re-evaluating loss

        Returns:
            Loss value if closure provided
        """
        ...

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients."""
        ...

    def state_dict(self) -> dict:
        """Return state dict for checkpointing."""
        ...

    def load_state_dict(self, state_dict: dict):
        """Load state dict from checkpoint."""
        ...
```

---

## Phase-Specific Usage

### Phase 1: Cognate (TinyTitan Pretraining)

**Configuration**:
```python
config = MuGrokConfig.from_phase(1)
# muon_lr=0.01, grokfast_lambda=0.05 (gentle filtering)
```

**Training Loop**:
```python
import torch
from mugrokfast_optimizer import MuonGrokfast
from mugrokfast_config import MuGrokConfig

# Initialize
config = MuGrokConfig.from_phase(1)
optimizer = MuonGrokfast(model.parameters(), config=config)

# Training
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, labels = batch

        # Forward
        outputs = model(input_ids)
        loss = F.cross_entropy(outputs.view(-1, vocab_size), labels.view(-1))

        # Backward + optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log (W&B)
        wandb.log({
            'loss': loss.item(),
            'epoch': epoch,
            'step': optimizer.state['step']
        })
```

**Expected Behavior**:
- Muon orthogonalizes attention + MLP weights (2-D)
- Grokfast filters with λ=0.05 (gentle)
- QK-clip prevents attention explosion
- Convergence: ~10% faster than baseline AdamW

---

### Phase 3: Quiet-STaR (REINFORCE Thought Training)

**Configuration**:
```python
config = MuGrokConfig.from_phase(3)
# muon_lr=5e-4 (lower for RL), grokfast_lambda=0.1, kl_coefficient=0.1
```

**Training Loop**:
```python
config = MuGrokConfig.from_phase(3)
optimizer = MuonGrokfast(model.parameters(), config=config)

for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, labels = batch

        # Generate thoughts + predictions
        thoughts = model.generate_thoughts(input_ids)
        outputs = model.forward_with_thoughts(input_ids, thoughts)

        # REINFORCE reward
        correct = (outputs.argmax(-1) == labels).float()
        reward = 2 * correct - 1  # {-1, +1}

        # Policy gradient + KL penalty
        log_prob = model.log_prob(outputs, labels)
        kl_div = compute_kl_divergence(model, model_base)
        loss = -reward * log_prob + config.kl_coefficient * kl_div

        # Optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Expected Behavior**:
- Lower Muon LR (5e-4) for RL stability
- Tighter QK-clip (τ=25.0) prevents RL divergence
- KL penalty keeps model close to base (λ_KL=0.1)
- Convergence: Stable RL training, no logit explosions

---

### Phase 5: Forge Training (BitNet + Grokfast with STE)

**Configuration**:
```python
config = MuGrokConfig.from_phase(5)
# muon_ste_mode=True (CRITICAL), grokfast_lambda=2.0 (aggressive)
```

**Training Loop with STE**:
```python
config = MuGrokConfig.from_phase(5)
optimizer = MuonGrokfast(model.parameters(), config=config)

# Replace layers with STE versions
model = replace_with_ste_layers(model)

for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, labels = batch

        # Forward (STE: quantized weights, full-precision gradients)
        outputs = model(input_ids)
        loss = F.cross_entropy(outputs.view(-1, vocab_size), labels.view(-1))

        # Backward (STE preserves gradient flow)
        loss.backward()

        # Optimize (Muon applies to full-precision weights)
        optimizer.step()
        optimizer.zero_grad()
```

**STE Compatibility Details**:
```python
# In QuantizedLinearSTE module
class QuantizedLinearSTE(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_full_precision = nn.Parameter(torch.randn(...))
        self.weight_quantized = None  # Computed in forward

    def forward(self, x):
        # Quantize for forward pass
        self.weight_quantized = quantize(self.weight_full_precision)

        # STE: Forward uses quantized, backward uses full-precision
        weight = self.weight_full_precision + \
                 (self.weight_quantized - self.weight_full_precision).detach()

        return F.linear(x, weight)

# MuonGrokfast detects STE mode and applies to weight_full_precision
if config.muon_ste_mode and hasattr(param, 'weight_full_precision'):
    param.weight_full_precision.data.add_(update, alpha=-lr)
```

**Expected Behavior**:
- Muon updates full-precision weights
- STE gradient flow preserved through quantization
- Aggressive Grokfast (λ=2.0) accelerates convergence
- Convergence: 50% faster than baseline (per original Phase 5 claims)

---

### Phase 6: Tool & Persona Baking (Two-Stage)

**Stage 1: Prompt Tuning** (1-D embeddings)
```python
config = MuGrokConfig.from_phase(6)
config.enable_muon = False  # 1-D prompt embeddings → fallback only
optimizer = MuonGrokfast(prompt_params, config=config)

for step in range(100):
    output = model(input_with_prompt=prompt_emb)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Stage 2: Weight Merging** (2-D weights)
```python
config = MuGrokConfig.from_phase(6)
config.enable_muon = True  # Merging into weight matrices
optimizer = MuonGrokfast(model.parameters(), config=config)

# Merge prompt embedding into first layer
model.embeddings.weight.data += alpha * optimized_prompt_emb

# Optional: Fine-tune merged weights
for step in range(50):
    loss = evaluate(model)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Expected Behavior**:
- Stage 1: Fallback optimizer for 1-D prompts
- Stage 2: Muon for 2-D weight merging
- Convergence: Personas successfully baked into weights

---

### Phase 7: Transformer² SVF Training

**Configuration**:
```python
config = MuGrokConfig.from_phase(7)
# enable_muon=False (z vectors are 1-D), kl_coefficient=0.2
```

**SVF Training Loop**:
```python
config = MuGrokConfig.from_phase(7)
optimizer = MuonGrokfast(z_vectors.values(), config=config)

for epoch in range(max_epochs):
    for batch in dataset:
        # Apply SVF to model
        W_adapted = {}
        for name, param in model.named_parameters():
            if param.ndim == 2:
                U, S, Vt = torch.linalg.svd(param, full_matrices=False)
                z = z_vectors[name]
                W_adapted[name] = U @ (S * z).diag() @ Vt

        # Forward with adapted weights
        output = model.forward(batch.input, weights=W_adapted)

        # REINFORCE reward
        reward = 1.0 if correct(output, batch.target) else -1.0

        # Loss: policy gradient + KL
        log_prob = model.log_prob(output, batch.input)
        kl_div = compute_kl(model, model_base)
        loss = -reward * log_prob + 0.2 * kl_div

        # Optimize z vectors (1-D → fallback path)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Expected Behavior**:
- Fallback optimizer for 1-D z vectors
- Grokfast prefilter stabilizes RL
- QK-clip prevents attention divergence
- Convergence: K experts trained, each >85% accuracy

---

## Advanced Topics

### Custom Hyperparameter Tuning

```python
# Start from phase preset, then override
config = MuGrokConfig.from_phase(1)
config.grokfast_lambda = 0.1  # Increase filtering
config.muon_ns_steps = 3      # Faster Newton-Schulz
config.qk_clip_threshold = 25.0  # Tighter attention bounds

optimizer = MuonGrokfast(model.parameters(), config=config)
```

### Ablation Studies

```python
# Disable Grokfast (test Muon alone)
config = MuGrokConfig.from_phase(1)
config.enable_grokfast = False
optimizer = MuonGrokfast(model.parameters(), config=config)

# Disable Muon (test Grokfast alone)
config.enable_muon = False

# Disable QK-clip (test optimization alone)
config.enable_muon_clip = False
```

### Checkpointing

```python
# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'config': config.to_dict()
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
```

### Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
optimizer = MuonGrokfast(model.parameters(), config=config)

for batch in dataloader:
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

## Testing

### Unit Tests

```python
# test_mugrokfast.py
import torch
from mugrokfast_optimizer import MuonGrokfast
from mugrokfast_config import MuGrokConfig

def test_parameter_routing():
    """Test 2-D → Muon, 1-D → fallback routing."""
    config = MuGrokConfig.from_phase(1)

    # Create params
    W_2d = nn.Parameter(torch.randn(100, 100))
    b_1d = nn.Parameter(torch.randn(100))

    optimizer = MuonGrokfast([W_2d, b_1d], config=config)

    # Check routing
    assert optimizer._route_parameter(W_2d) == "muon"
    assert optimizer._route_parameter(b_1d) == "fallback"

def test_newton_schulz_orthogonality():
    """Test Newton-Schulz produces semi-orthogonal matrices."""
    config = MuGrokConfig.from_phase(1)
    optimizer = MuonGrokfast([], config=config)

    W = torch.randn(100, 100)
    W_ortho = optimizer._newton_schulz_ortho(W, steps=5)

    # Check semi-orthogonality: W @ W.T ≈ I
    product = W_ortho @ W_ortho.T
    identity = torch.eye(100)

    assert torch.allclose(product, identity, atol=1e-2)

def test_qk_clip():
    """Test QK-clip rescales updates correctly."""
    config = MuGrokConfig(use_qk_clip=True, qk_clip_threshold=10.0)
    optimizer = MuonGrokfast([], config=config)

    # Create large update
    W = torch.randn(100, 100)
    update = torch.randn(100, 100) * 100  # Large update

    # Apply QK-clip
    clipped = optimizer._apply_qk_clip(W, update, "q_proj")

    # Check norm bounded
    new_norm = (W + clipped).norm()
    assert new_norm <= 10.0 + 1e-5

def test_ste_compatibility():
    """Test STE mode preserves gradient flow."""
    config = MuGrokConfig.from_phase(5)  # STE mode enabled

    # Create quantized param with full-precision copy
    param = nn.Parameter(torch.randn(100, 100))
    param.weight_full_precision = param.clone()

    optimizer = MuonGrokfast([param], config=config)

    # Simulate gradient
    param.grad = torch.randn_like(param)

    # Step
    optimizer.step()

    # Check update applied to full-precision
    assert hasattr(param, 'weight_full_precision')
```

### Integration Tests

```python
def test_phase1_integration():
    """Test Phase 1 training loop."""
    model = TinyTitan(vocab_size=1000, d_model=256)
    config = MuGrokConfig.from_phase(1)
    optimizer = MuonGrokfast(model.parameters(), config=config)

    for _ in range(10):
        batch = torch.randint(0, 1000, (32, 128))
        outputs = model(batch)
        loss = F.cross_entropy(outputs.view(-1, 1000), batch.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert loss.item() < initial_loss
```

---

## W&B Integration

### Metrics Tracking

```python
import wandb

# Initialize W&B
wandb.init(project="agent-forge-v2", config=config.to_dict())

# Training loop with logging
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)

        # Log optimizer state
        wandb.log({
            'loss': loss.item(),
            'epoch': epoch,
            'global_step': optimizer.state['step'],

            # Grokfast metrics
            'grokfast/ema_norm': compute_ema_norm(optimizer),
            'grokfast/filter_ratio': compute_filter_ratio(optimizer),

            # Muon metrics
            'muon/ortho_error': compute_ortho_error(optimizer),
            'muon/ns_iterations': config.muon_ns_steps,

            # QK-clip metrics
            'qk_clip/num_clipped': count_clipped_params(optimizer),
            'qk_clip/max_norm': compute_max_qk_norm(model),
        })
```

### Dashboard Configuration

```python
# wandb_config.yaml
metrics:
  - loss
  - grokfast/ema_norm
  - muon/ortho_error
  - qk_clip/num_clipped

charts:
  - title: "Training Loss"
    metric: loss
    type: line

  - title: "Grokfast Filter Ratio"
    metric: grokfast/filter_ratio
    type: line

  - title: "Muon Orthogonality Error"
    metric: muon/ortho_error
    type: line
```

---

## Troubleshooting

### Issue: Training Loss Explodes

**Symptoms**: Loss → NaN or ∞ after few steps

**Possible Causes**:
1. Learning rate too high
2. QK-clip disabled or threshold too high
3. Grokfast λ too aggressive

**Solutions**:
```python
# Reduce learning rates
config.muon_lr = 1e-3  # Lower from 0.01
config.fallback_lr = 1e-5  # Lower from 1e-4

# Enable/tighten QK-clip
config.use_qk_clip = True
config.qk_clip_threshold = 20.0  # Lower from 30.0

# Reduce Grokfast amplification
config.grokfast_lambda = 0.01  # Lower from 0.05
```

### Issue: RL Training Unstable (Phase 3, 7)

**Symptoms**: Reward variance high, no learning progress

**Solutions**:
```python
# Use Phase 3/7 presets (already tuned for RL)
config = MuGrokConfig.from_phase(3)  # or 7

# Tighter QK-clip
config.qk_clip_threshold = 25.0

# Lower Grokfast amplification
config.grokfast_lambda = 0.05

# Increase KL penalty
config.kl_coefficient = 0.2  # Keep close to base model
```

### Issue: STE Gradient Flow Broken (Phase 5)

**Symptoms**: Quantized model doesn't train, gradients all zeros

**Solutions**:
```python
# Ensure STE mode enabled
config = MuGrokConfig.from_phase(5)
assert config.muon_ste_mode == True

# Check param has full-precision copy
for param in model.parameters():
    if param.ndim == 2:
        assert hasattr(param, 'weight_full_precision')

# Verify STE implementation
# Forward: use quantized
# Backward: use full-precision (via .detach())
```

### Issue: Slow Training Speed

**Symptoms**: Training much slower than baseline AdamW

**Possible Causes**:
1. Too many Newton-Schulz iterations
2. SVD orthogonalization instead of Newton-Schulz

**Solutions**:
```python
# Reduce NS iterations
config.muon_ns_steps = 3  # Lower from 5

# Ensure NS enabled (not SVD)
assert config.muon_ns_steps > 0

# Profile to find bottleneck
import torch.profiler
with torch.profiler.profile() as prof:
    optimizer.step()
print(prof.key_averages())
```

---

## References

1. **Grokfast Paper**: "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"
2. **Muon Paper**: "Muon: Matrix-Structured Optimizer" (2505.23737v1.pdf)
3. **Transformer² Paper**: "Transformer-Squared: Self-Adaptive LLMs" (2501.06252v3.pdf)
4. **ADAS Paper**: "Automated Design of Agentic Systems"

---

## License

Agent Forge V2 - Internal Research Project

---

## Support

For issues, questions, or contributions:
- GitHub Issues: [agent-forge-v2/issues]
- Documentation: [docs/v2-specification/]
- Contact: Agent Forge V2 Team
