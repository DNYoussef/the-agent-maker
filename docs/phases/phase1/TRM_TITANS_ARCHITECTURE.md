# TRM × Titans-MAG Architecture - Phase 1 Complete Specification

**Version**: 2.0 (Agent Forge V2)
**Phase**: 1 (Cognate)
**Target**: 25M±1M parameters
**Status**: ✅ Ready for Implementation

---

## Overview

Phase 1 (Cognate) creates a **Transformer Recursive Memory (TRM)** system built on the **Titans-MAG** backbone. This architecture combines:

1. **TitansMAG Backbone**: 8-layer transformer with Sliding Window Attention + LMM memory
2. **MAG Gate**: Convex blend between current output (y) and long-range memory (m)
3. **TRM Wrapper**: Recursive refinement with g_φ (latent refine) and h_ψ (answer update)
4. **ACT Head**: Adaptive Computation Time with EMA-based calibration

**Key Innovation**: Multi-pass reasoning with detached recursion steps for memory efficiency and deep supervision at each refinement stage.

---

## Architecture Components

### 1. TitansMAG Backbone

The base transformer uses **8 layers** with **d=512** to fit the 25M parameter budget.

**Layer Specifications**:
```yaml
d_model: 512
n_layers: 8
n_heads: 8
head_dim: 64  # d_model / n_heads
vocab_size: 32768  # Standard BPE tokenizer
max_seq_len: 2048
```

**Attention Mechanism**: Sliding Window Attention
```yaml
sw_window: 1024  # Each token attends to ±512 tokens
# Benefits:
# - O(n·w) complexity instead of O(n²)
# - Enables 2048 seq length on 6GB VRAM
# - Preserves local context for reasoning
```

**MLP Architecture**: SwiGLU with 4× expansion
```python
class SwiGLUMLP(nn.Module):
    def __init__(self, d_model=512):
        self.w_gate = nn.Linear(d_model, 4 * d_model, bias=False)
        self.w_up = nn.Linear(d_model, 4 * d_model, bias=False)
        self.w_down = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)
```

**Normalization**: Pre-LayerNorm with RMSNorm
```python
# Per-layer structure:
x = x + attention(RMSNorm(x))
x = x + mlp(RMSNorm(x))
```

---

### 2. LMM (Long-Range Memory Module)

**Factorized Memory Design** (from Titans paper):
```yaml
memory:
  type: "factorized"
  d_mem: 256        # Memory dimension (half of d_model)
  decay: 0.99       # Exponential decay for temporal weighting
  init: "zeros"     # Initialize memory state
```

**Memory Update Rule**:
```python
# At each layer:
m_t = decay * m_{t-1} + (1 - decay) * project_to_mem(x_t)

# Factorized projection (reduces parameters):
W_mem_down: [d_model=512, d_mem=256]  # Compress
W_mem_up: [d_mem=256, d_model=512]    # Expand

# Usage:
m_compressed = W_mem_down @ x
m_t = decay * m_{t-1} + (1 - decay) * m_compressed
m_contribution = W_mem_up @ m_t
```

**Parameter Budget**:
- Down projection: 512 × 256 = 131K per layer
- Up projection: 256 × 512 = 131K per layer
- Total: 262K × 8 layers = **2.1M params** for memory system

---

### 3. MAG Gate (Memory-Augmented Gate)

**Purpose**: Dynamically blend current transformer output (y) with long-range memory (m).

**Gate Equation**:
```python
# Gating function:
g = σ(W_g @ [y || m] + b_g)  # Sigmoid, shape: [batch, seq, d_model]

# Convex blend:
output = g ⊙ y + (1 - g) ⊙ m

# With entropy regularization:
loss_gate_entropy = -λ_ent * mean(g * log(g) + (1-g) * log(1-g))
# Prevents saturation (g → 0 or g → 1)
```

**Architecture**:
```yaml
gate:
  hidden: 256           # Hidden layer for gating network
  entropy_reg: 0.001    # Regularization coefficient

# Parameter count:
W_concat: [2*d_model, hidden] = 1024 × 256 = 262K
W_gate: [hidden, d_model] = 256 × 512 = 131K
Total: ~400K params
```

**Implementation**:
```python
class MAGGate(nn.Module):
    def __init__(self, d_model=512, hidden=256, entropy_reg=0.001):
        self.w_concat = nn.Linear(2 * d_model, hidden)
        self.w_gate = nn.Linear(hidden, d_model)
        self.entropy_reg = entropy_reg

    def forward(self, y, m):
        concat = torch.cat([y, m], dim=-1)  # [batch, seq, 2*d_model]
        hidden = F.relu(self.w_concat(concat))
        g = torch.sigmoid(self.w_gate(hidden))

        # Convex blend
        output = g * y + (1 - g) * m

        # Entropy regularization
        eps = 1e-8
        entropy = -(g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps))
        loss_entropy = -self.entropy_reg * entropy.mean()

        return output, loss_entropy, g.mean().item()  # Return g stats for W&B
```

---

### 4. TRM Wrapper (Transformer Recursive Memory)

**Purpose**: Multi-pass reasoning with iterative refinement of latent state (z) and answer (y).

**Components**:
- **g_φ** (Refiner): Updates latent state z via micro-steps
- **h_ψ** (Updater): Updates answer y from refined z
- **Recursion Controller**: Manages T_max iterations with early stopping

**Recursion Flow**:
```python
# Initialization:
z_0 = encoder(input_tokens)  # Initial latent
y_0 = decoder_init(z_0)      # Initial answer

# Recursion loop (T_max iterations):
for t in range(T_max):
    # 1. Refine latent (n micro-steps inside g_φ)
    z_t = g_φ(z_{t-1}, y_{t-1}, features, n=2)

    # 2. Update answer
    y_t = h_ψ(y_{t-1}, z_t)

    # 3. Compute loss for this step
    logits_t = lm_head(y_t)
    loss_t = ce_loss(logits_t, labels) * step_weight(t)

    # 4. ACT head (halt/continue decision)
    q_t = act_head(z_t)
    if should_halt(q_t):
        break

    # 5. Detach for next iteration (memory + stability)
    y_t = y_t.detach()
    z_t = z_t.detach()
```

**Micro-Step Refinement** (inside g_φ):
```python
def g_phi_refine(z, y, features, n=2):
    """Refine latent z via n micro-steps."""
    for i in range(n):
        # Attend to features and previous answer
        context = cross_attention(z, features)
        answer_context = cross_attention(z, y)

        # Update z
        z = z + mlp(RMSNorm([z, context, answer_context]))
    return z
```

**Configuration**:
```yaml
trm:
  T_max: 3                     # Start with 3 recursion steps
  micro_steps: 2               # Refinement steps per iteration
  deep_supervision: true       # Loss at each step
  detach_between_steps: true   # Memory efficiency
  step_weights: [0.5, 0.75, 1.0]  # Earlier steps weighted lower
```

**Parameter Budget**:
- g_φ cross-attention: ~1M params
- h_ψ update network: ~500K params
- Total: **~1.5M params** for TRM wrapper

---

### 5. ACT Head (Adaptive Computation Time)

**Purpose**: Learn when to stop recursion based on confidence and correctness.

**Head Architecture**:
```python
class ACTHead(nn.Module):
    def __init__(self, d_model=512):
        self.w_halt = nn.Linear(d_model, 1)  # Single logit per token

    def forward(self, z):
        halt_logit = self.w_halt(z)  # [batch, seq, 1]
        q = torch.sigmoid(halt_logit)  # Halt probability
        return q
```

**EMA Calibration** (prevent pathological halting):
```python
# During training, track EMA of step accuracies:
ema_step_acc = {0: 0.6, 1: 0.75, 2: 0.85}  # Example after warmup

# Target halt probability based on EMA:
target_halt = (acc > ema_step_acc[t])  # Binary target

# ACT loss (per step):
loss_act = bce_loss(q, target_halt)
```

**Configuration**:
```yaml
act:
  halt_thresh: 0.5       # Probability threshold for halting
  ema_teacher: 0.98      # EMA decay for step accuracy tracking
  entropy_reg: 0.001     # Prevent q → 0.5 saturation
```

**Halting Logic**:
```python
def should_halt(q, threshold=0.5):
    """
    Halt if average halt probability > threshold.

    Args:
        q: [batch, seq, 1] halt probabilities
        threshold: Halt threshold (0.5 default)

    Returns:
        bool: True if should halt
    """
    avg_halt_prob = q.mean().item()
    return avg_halt_prob > threshold
```

**Parameter Budget**: 512 × 1 = **512 params** (negligible)

---

## Complete Parameter Budget

| Component | Params | Calculation |
|-----------|--------|-------------|
| **Embeddings** | 16.8M | 32768 vocab × 512 dim |
| **8 Transformer Layers** | 5.2M | (4 × 512² attention + 4 × 512 × 2048 MLP) × 8 |
| **LMM Memory** | 2.1M | (512×256 + 256×512) × 8 layers |
| **MAG Gate** | 400K | 1024×256 + 256×512 |
| **TRM Wrapper** | 1.5M | g_φ + h_ψ cross-attention |
| **LM Head** | 16.8M | 512 × 32768 (tied with embeddings) |
| **ACT Head** | 512 | 512 × 1 |
| **Total** | **~25M** | ✅ Within 25±1M budget |

**Notes**:
- LM head weight tying: Embeddings and output projection share weights
- Sliding Window Attention saves memory, not parameters
- Factorized LMM reduces memory params by 50%

---

## Training Configuration

### Optimizer: MuonGrokfast

**Configuration** (from Developer Guide #1):
```yaml
optim:
  kind: "muon_grokfast"
  alpha: 0.98              # Grokfast EMA decay
  lam: 0.3                 # Grokfast amplification (aggressive)
  muon_k: 3                # Newton-Schulz iterations
  beta: 0.9                # Muon momentum
  weight_decay: 0.05       # Decoupled weight decay
  lr: 0.001                # Base learning rate
  qkclip_tau: 30.0         # QK-clip threshold
  qkclip_per_head: true    # Per-head attention clipping
  mlarope_shared_k_protected: true  # Protect rotary-K
  fallback: "adamw"        # For 1-D params (bias, LayerNorm)
```

**Parameter Routing**:
- 2-D matrices (attention, MLP weights) → Muon path
- 1-D params (bias, RMSNorm scales, embeddings) → AdamW fallback

**QK-Clip Safety** (per-head):
```python
# After optimizer step, check attention logits:
for head in range(n_heads):
    logits_max = estimate_qk_max(W_Q[head], W_K[head])
    if logits_max > qkclip_tau:
        gamma_head = (qkclip_tau / logits_max) ** 2
        W_Q[head].data.mul_(gamma_head ** 0.5)
        W_K[head].data.mul_(gamma_head ** 0.5)
```

### Training Loop

**Pseudo-code** (from Developer Guide #2):
```python
for batch in dataloader:
    loss_total = 0
    z = z0_init(batch)
    y = y0_init(batch)
    act_trace = []

    for t in range(T_max):
        # Forward pass through backbone
        features = model.forward_once(batch.tokens)

        # TRM refinement
        z = trm.refine_latent(features, y, z, micro_steps=2)
        y = trm.update_answer(y, z)

        # Heads & losses
        logits = lm_head(y)
        q = act_head(z)

        # Step loss (deep supervision)
        step_loss = ce_loss(logits, batch.labels) * step_weight(t)
        act_loss = bce_loss(q, target_halt[t])

        loss_total += step_loss + act_loss

        # ACT early stop
        if should_halt(q):
            act_trace.append(t)
            break
        else:
            act_trace.append(t)
            # Detach for next iteration
            y = y.detach()
            z = z.detach()

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss_total.backward()
    optimizer.step()

    # Log to W&B
    wandb.log({
        "loss/total": loss_total.item(),
        "act/steps": np.mean(act_trace),
        "gate/entropy": gate_entropy,
        ...
    })
```

**Curriculum**:
```yaml
train:
  # Sequence length curriculum
  seq_len_schedule:
    - [0, 10000]: 512        # First 10K steps
    - [10000, 50000]: 1024   # Next 40K steps
    - [50000, None]: 2048    # Final stage

  # Recursion depth curriculum
  T_max_schedule:
    - [0, 20000]: 3          # Start with 3 steps
    - [20000, None]: 6       # Increase to 6 after plateau

  # Batch size
  batch_size: 32             # Micro-batch (gradient accumulation: 4)
  effective_batch: 128       # 32 × 4 accumulation steps
```

---

## W&B Integration

### Metrics to Track

**Loss Breakdown**:
```python
wandb.log({
    "loss/total": loss_total,
    "loss/step_0": step_losses[0],
    "loss/step_1": step_losses[1],
    "loss/step_2": step_losses[2],
    "loss/act": act_loss,
    "loss/gate_entropy": gate_entropy_loss,
})
```

**ACT Metrics**:
```python
wandb.log({
    "act/avg_steps": np.mean(act_trace),          # Average recursion depth
    "act/halt_prob_step0": q_values[0].mean(),
    "act/halt_prob_step1": q_values[1].mean(),
    "act/halt_prob_step2": q_values[2].mean(),
    "act/ema_acc_step0": ema_step_acc[0],
    "act/ema_acc_step1": ema_step_acc[1],
})
```

**Gate Metrics**:
```python
wandb.log({
    "gate/entropy": gate_entropy,
    "gate/avg_g": g_values.mean(),                # Average gate value
    "gate/y_usage": (g_values > 0.5).float().mean(),  # % using y over m
    "gate/m_usage": (g_values < 0.5).float().mean(),  # % using m over y
})
```

**Optimizer Metrics**:
```python
wandb.log({
    "optim/ns_time_ms": ns_time,                  # Newton-Schulz timing
    "optim/update_norm": update_norm,
    "optim/qk_clip_count": qk_clip_count,         # Times QK-clip activated
    "optim/grad_ema_norm": grad_ema_norm,
})
```

**Learning Curves**:
```python
wandb.log({
    "train/loss_vs_tokens": (total_tokens, loss),
    "train/perplexity": torch.exp(loss),
    "valid/loss": valid_loss,
    "valid/accuracy": valid_accuracy,
})
```

---

## Implementation Checklist

### Before First Run

- [ ] **Config file**: Create `phases/phase1/configs/phase1_config.yaml` with all hyperparameters
- [ ] **Model config**: Create `phases/phase1/model/model_config.py` dataclasses
- [ ] **Parameter budget**: Verify 25±1M params with `sum(p.numel() for p in model.parameters())`
- [ ] **W&B project**: Initialize with `wandb.init(project="agent-forge-v2", name="phase1_cognate")`
- [ ] **Premortem**: Review `PREMORTEM_CHECKLIST.md` for failure modes

### During Training

- [ ] **Monitor ACT**: Check `act/avg_steps` stays 2-4 (not 0 or T_max)
- [ ] **Monitor gate**: Check `gate/entropy > 0.5` (not saturated)
- [ ] **Monitor optimizer**: Check `optim/qk_clip_count < 100/step` (not clipping too often)
- [ ] **Check loss**: Per-step losses should decrease: `loss/step_0 > loss/step_1 > loss/step_2`
- [ ] **Check memory**: VRAM usage < 6GB with batch_size=32, seq_len=512

### Debugging

**If ACT always halts at step 0**:
- Lower `act/halt_thresh` to 0.3
- Check EMA calibration: `ema_step_acc[0]` should be < 0.5
- Increase entropy regularization: `act/entropy_reg: 0.01`

**If gate saturates (g → 0 or g → 1)**:
- Increase `gate/entropy_reg` to 0.01
- Check `gate/y_usage` and `gate/m_usage` (should be 30-70%)
- Reduce learning rate for gate params

**If loss spikes**:
- Check `optim/qk_clip_count` (QK-clip should activate)
- Reduce `optim/lam` from 0.3 to 0.1 (less aggressive Grokfast)
- Check gradient norms (clip if > 1.0)

**If out of memory**:
- Reduce `batch_size` from 32 to 16
- Reduce `seq_len` to 512 (delay curriculum)
- Enable gradient checkpointing for TRM recursion

---

## Success Criteria

- ✅ **Parameter budget**: 25±1M params
- ✅ **Training stability**: Loss decreases smoothly, no spikes
- ✅ **ACT convergence**: `act/avg_steps` stabilizes at 2-4
- ✅ **Gate balance**: `gate/y_usage` between 30-70%
- ✅ **Perplexity**: < 20 on validation set (TinyStories or similar)
- ✅ **Memory**: Fits in 6GB VRAM with batch_size=32, seq_len=1024
- ✅ **Speed**: > 5000 tokens/sec on GTX 1660

---

## References

**Developer Guides**:
- Developer Guide #1: Muon × Grokfast (modular optimizer stack)
- Developer Guide #2: TRM × Titans × ACT (model system)

**Research Papers**:
- Titans: Learning to Memorize at Test Time (LMM, MAG gate)
- Adaptive Computation Time (ACT halting)
- Grokfast: Accelerated Grokking (EMA gradient filtering)
- Muon: Momentum Orthogonalized by Newton-Schulz (optimizer)

**Related Phase 1 Documents**:
- [PREMORTEM_CHECKLIST.md](PREMORTEM_CHECKLIST.md) - Failure modes and mitigations
- [LOGICAL_UNDERSTANDING.md](LOGICAL_UNDERSTANDING.md) - Conceptual overview
- [PHASE1_COMPLETE_GUIDE.md](PHASE1_COMPLETE_GUIDE.md) - Step-by-step implementation

---

**Phase 1 Status**: ✅ Architecture Specification Complete
**Next Step**: Implement model in `phases/phase1/model/titans_mag.py`
