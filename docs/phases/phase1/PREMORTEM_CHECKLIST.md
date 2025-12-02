# Phase 1 Premortem Checklist - TRM × Titans × MuonGrokfast

**Version**: 2.0 (Agent Forge V2)
**Phase**: 1 (Cognate)
**Purpose**: Identify failure modes BEFORE implementation
**Status**: ✅ Complete MECE Failure Tree

---

## Premortem Overview

This document catalogs all potential failure modes for Phase 1 (TRM × Titans-MAG) implementation, organized in a **MECE (Mutually Exclusive, Collectively Exhaustive)** tree. Each failure mode includes:

1. **Root cause**
2. **Symptoms**
3. **Mitigation strategy**
4. **Detection method**

**Source**: Developer Guide #2 (Dr. Synthara's Premortem Analysis)

---

## Failure Tree (MECE)

### A. Objective & Data Failures

#### A1. Objective Mismatch
**Root Cause**: Metrics don't align with downstream tasks (Phase 2+)

**Symptoms**:
- Low perplexity but poor Phase 2 merge fitness
- Model can't adapt in Quiet-STaR (Phase 3)
- High pretraining loss but good eval performance (overfitting proxy)

**Mitigation**:
```yaml
# Use composite objective:
loss_total = (
    0.7 * next_token_loss +
    0.2 * sequence_level_coherence +
    0.1 * diversity_penalty
)

# Track downstream proxies:
metrics:
  - perplexity (primary)
  - embedding_diversity
  - attention_entropy
  - reasoning_probes (simple QA)
```

**Detection**:
- Monitor `valid/perplexity` vs `valid/reasoning_accuracy` divergence
- Run Phase 2 merge preview every 10K steps (quick 10-gen EvoMerge)

---

#### A2. Data Quality Issues
**Root Cause**: Training data doesn't support multi-step reasoning

**Symptoms**:
- ACT always halts at step 0 (no benefit from recursion)
- Gate always prefers y over m (memory unused)
- Flat loss curves (no learning)

**Mitigation**:
```yaml
# Use reasoning-rich datasets:
datasets:
  - TinyStories (30%)          # Narrative coherence
  - GSM8K-subset (20%)         # Math reasoning
  - Code-simple (20%)          # Structured logic
  - Wikipedia-chunks (30%)     # Factual recall

# Filter criteria:
min_length: 128 tokens         # Ensure multi-step context
max_length: 2048 tokens
quality_score: > 0.7           # Perplexity filter
```

**Detection**:
- Inspect `gate/m_usage` (should be > 20%)
- Check `act/avg_steps` (should be > 1.5)
- Sample model outputs (qualitative check)

---

### B. Architecture Failures (TRM × Titans)

#### B1. Parameter Budget Overrun
**Root Cause**: 8 layers × 512 dim × memory/gate components exceed 25M

**Symptoms**:
- `sum(p.numel()) > 26M` during initialization
- OOM on 6GB VRAM even with batch_size=1

**Mitigation**:
```python
# Budget guardrails (before training):
def check_param_budget(model, target=25e6, tolerance=1e6):
    total = sum(p.numel() for p in model.parameters())
    assert abs(total - target) <= tolerance, \
        f"Param budget: {total/1e6:.2f}M (target: {target/1e6:.2f}M)"

# If over budget, reduce:
# - d_mem: 256 → 192 (saves 1M)
# - gate/hidden: 256 → 128 (saves 200K)
# - n_layers: 8 → 7 (saves 3M)
```

**Detection**:
- Pre-training assertion in `model/__init__.py`
- W&B config logging: `config/param_count`

---

#### B2. Attention Instability (SW-Attn)
**Root Cause**: Sliding Window Attention breaks with long-range dependencies

**Symptoms**:
- Loss plateaus after 20K steps
- `gate/m_usage → 0` (memory compensates for broken attention)
- QK-clip activates frequently (`optim/qk_clip_count > 500/step`)

**Mitigation**:
```yaml
# Hybrid attention:
attention:
  type: "sliding_window_with_global"
  sw_window: 1024
  global_tokens: 64        # Every 16th token attends globally

# QK-clip per-head:
qkclip_tau: 30.0
qkclip_per_head: true      # Isolate broken heads
```

**Detection**:
- Plot attention heatmaps at steps [1K, 10K, 50K]
- Monitor `optim/qk_clip_count` (should be < 100/step)
- Check `valid/long_range_acc` (custom eval on 2048-token sequences)

---

### C. Recursion & ACT Failures

#### C1. ACT Pathological Halting
**Root Cause**: Model always halts at step 0 or never halts (runs to T_max)

**Symptoms (Always Halt)**:
- `act/avg_steps = 0.1` (always stops immediately)
- `act/halt_prob_step0 > 0.9`
- No benefit from recursion (wasted compute)

**Symptoms (Never Halt)**:
- `act/avg_steps = T_max` (runs to limit every time)
- `act/halt_prob_stepN < 0.1` for all N
- Training is T_max× slower

**Mitigation**:
```yaml
# EMA calibration (from Developer Guide #2):
act:
  halt_thresh: 0.5
  ema_teacher: 0.98          # Track step accuracies
  warmup_steps: 5000         # Don't enforce targets early

# Target halting:
# - Step 0: acc ≈ 0.6 → target_halt = False (continue)
# - Step 2: acc ≈ 0.85 → target_halt = True (halt)

# Loss:
loss_act = bce(q_t, target_halt[t])
```

**Detection**:
- Plot `act/avg_steps` histogram (should be 2-4, not 0 or T_max)
- Check `act/ema_acc_stepN` (should increase: step 0 < step 1 < step 2)

---

#### C2. Recursion Instability
**Root Cause**: Gradient explosion across recursion steps

**Symptoms**:
- Loss spikes after 5K-10K steps
- `grad/norm_z > 100` (latent gradients explode)
- NaN in `loss/step_2`

**Mitigation**:
```python
# 1. Detach between steps (from Developer Guide #2):
for t in range(T_max):
    z = trm.refine(z, y)
    y = trm.update(y, z)

    # Compute loss
    loss_t = ce_loss(lm_head(y), labels)
    loss_total += loss_t

    if not should_halt(act_head(z)):
        y = y.detach()  # CRITICAL: Break gradient flow
        z = z.detach()

# 2. Gradient clipping:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Step weights (reduce early step influence):
step_weights = [0.5, 0.75, 1.0]  # Earlier steps matter less
```

**Detection**:
- Monitor `grad/norm_y`, `grad/norm_z` (should be < 10)
- Check `loss/step_0` vs `loss/step_2` (should both decrease, not diverge)

---

### D. Optimization Failures (Muon × Grokfast)

#### D1. Grokfast Over-Amplification
**Root Cause**: λ=0.3 is too aggressive, amplifies noise

**Symptoms**:
- Loss oscillates (±0.5 every 100 steps)
- `optim/grad_ema_norm` much larger than `optim/grad_raw_norm`
- Unstable training (NaN after 20K steps)

**Mitigation**:
```yaml
# Reduce amplification:
grokfast:
  alpha: 0.98              # Keep EMA decay
  lam: 0.1                 # Lower from 0.3 → 0.1

# Or disable Grokfast temporarily:
enable_grokfast: false
```

**Detection**:
- Plot `train/loss` with rolling mean (window=100)
- Monitor `optim/grad_ema_norm / optim/grad_raw_norm` (should be 1.0-1.5, not > 2.0)

---

#### D2. Muon Newton-Schulz Divergence
**Root Cause**: k=3 iterations insufficient for orthogonalization

**Symptoms**:
- `optim/ortho_error > 0.1` (M @ M.T deviates from I)
- Low-rank collapse (`optim/effective_rank < 50%`)
- Performance worse than AdamW baseline

**Mitigation**:
```yaml
# Increase NS iterations:
muon:
  k: 5                     # Up from 3

# Or fall back to SVD (slower but exact):
muon:
  use_svd: true
  k: 0  # Disabled NS
```

**Detection**:
- Compute `ortho_error = ||M @ M.T - I||_F` every 1000 steps
- Track effective rank: `sum(singular_values > 0.01 * max_singular_value)`

---

#### D3. QK-Clip Over-Activation
**Root Cause**: Attention logits exceed τ=30 frequently

**Symptoms**:
- `optim/qk_clip_count > 500/step` (clipping every batch)
- Attention heads suppressed (`attn/head_entropy < 2.0`)
- Loss plateaus (model can't learn attention)

**Mitigation**:
```yaml
# Option 1: Increase threshold
qkclip_tau: 50.0           # Up from 30.0

# Option 2: Per-head protection
qkclip_per_head: true
mlarope_shared_k_protected: true  # Don't clip rotary-K

# Option 3: Lower learning rate for Q/K
param_groups:
  - {params: [W_Q, W_K], lr: 0.0005}  # Half of base LR
  - {params: [others], lr: 0.001}
```

**Detection**:
- Monitor `optim/qk_clip_count` (should be < 100/step)
- Plot `attn/logit_max_per_head` heatmap (identify problematic heads)

---

### E. Attention Stability Failures

#### E1. MLA/RoPE Key Protection Failure
**Root Cause**: QK-clip rescales shared rotary-K, breaking position encoding

**Symptoms**:
- Position encoding degrades (`eval/position_accuracy < 0.5`)
- Model loses track of token positions
- `attn/rope_norm` deviates from 1.0

**Mitigation**:
```yaml
# From Developer Guide #1:
qkclip:
  mlarope_shared_k_protected: true  # CRITICAL

# Implementation:
if is_shared_rotary_k(W_K[head]):
    # Only rescale W_Q, leave W_K unchanged
    W_Q[head].data.mul_(gamma)
else:
    # Standard per-head rescaling
    W_Q[head].data.mul_(gamma ** 0.5)
    W_K[head].data.mul_(gamma ** 0.5)
```

**Detection**:
- Check `attn/rope_norm` (should stay ≈ 1.0)
- Run position probe: "The [MASK] word is..." at positions [10, 100, 1000]

---

#### E2. Attention Sink (Outlier Tokens)
**Root Cause**: Some tokens accumulate all attention mass

**Symptoms**:
- `attn/sink_count > 10` tokens with > 50% attention
- `attn/entropy < 3.0` (low diversity)
- Poor long-context performance

**Mitigation**:
```python
# Attention entropy regularization:
def attn_entropy_loss(attn_weights):
    # attn_weights: [batch, heads, seq, seq]
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(-1)
    return -0.01 * entropy.mean()  # Maximize entropy

loss_total += attn_entropy_loss(attn_weights)
```

**Detection**:
- Monitor `attn/entropy` (should be > 4.0 for seq_len=512)
- Visualize attention heatmaps (check for vertical lines = sinks)

---

### F. MLOps & Observability Failures

#### F1. W&B Logging Overhead
**Root Cause**: Too many metrics (>1000) slow training by 2×

**Symptoms**:
- Training speed < 2500 tokens/sec (expected: 5000+)
- W&B upload queue > 1000 items
- High CPU usage (logging thread bottleneck)

**Mitigation**:
```yaml
# Reduce logging frequency:
logging:
  log_every_n_steps: 100      # Up from 10
  log_histograms: false        # Disable expensive histograms
  log_gradients: false         # Only enable for debugging

# Essential metrics only:
metrics:
  - loss/*                     # All loss components
  - act/avg_steps              # ACT monitoring
  - gate/entropy               # Gate health
  - optim/qk_clip_count        # Optimizer safety
```

**Detection**:
- Benchmark training speed with/without W&B
- Check `wandb.run.summary['_wandb']['upload_queue']`

---

#### F2. Checkpoint Corruption
**Root Cause**: OOM during checkpoint save, partial write

**Symptoms**:
- Training crashes at step 50K
- Latest checkpoint is 0 bytes or incomplete
- Can't resume from checkpoint

**Mitigation**:
```python
# Atomic checkpoint saving:
import tempfile
import shutil

def save_checkpoint(model, optimizer, step, path):
    # Save to temp file first
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
        }, tmp.name)

        # Atomic move (overwrites only if save succeeded)
        shutil.move(tmp.name, path)

# Save multiple checkpoints:
checkpoints:
  keep_last_n: 3               # Keep 3 most recent
  save_every_n_steps: 5000
```

**Detection**:
- Verify checkpoint size (should be ≈ 100MB for 25M params)
- Test restore: `model.load_state_dict(torch.load(ckpt))`

---

### G. Integration & Reuse Failures

#### G1. Phase 2 Merge Incompatibility
**Root Cause**: TRM recursion doesn't merge well (stateful components)

**Symptoms**:
- Phase 2 merge fitness < 0.5 (worse than single model)
- Merged model has broken ACT (always halts or never halts)
- Gate saturation (g → 0 or g → 1)

**Mitigation**:
```yaml
# Design for mergeability:
model:
  trm:
    stateless_mode: true       # Option for Phase 2
    # In stateless mode:
    # - No z/y carry-over between batches
    # - ACT head outputs same for same input
    # - Gate blending is deterministic

# Phase 2 preview:
# - Every 10K steps, merge 2 checkpoints with SLERP
# - Evaluate merged model (should be ≥ avg of parents)
```

**Detection**:
- Run Phase 2 preview merge at steps [10K, 20K, 30K]
- Check merged model perplexity vs parent average

---

#### G2. Optimizer Config Reuse Failure
**Root Cause**: Phase 1 MuonGrokfast config doesn't work for Phase 3 (RL)

**Symptoms**:
- Phase 3 training unstable (Policy Gradient explodes)
- `kl_coefficient` missing in Phase 1 config
- Need to retune all hyperparameters

**Mitigation**:
```python
# Modular config (from Developer Guide #1):
@dataclass
class MuGrokConfig:
    # Core (reusable)
    alpha: float = 0.98
    lam: float = 0.3
    muon_k: int = 3

    # Phase-specific (optional)
    kl_coefficient: Optional[float] = None  # For RL (Phase 3, 7)
    ste_mode: bool = False                  # For quantization (Phase 5)

# Phase 1 config:
config_p1 = MuGrokConfig(lam=0.3, kl_coefficient=None)

# Phase 3 config (reuse + extend):
config_p3 = MuGrokConfig(lam=0.1, kl_coefficient=0.1)  # Lower lam for RL
```

**Detection**:
- Verify `MuGrokConfig` works in phases [1, 3, 5, 6, 7]
- Check for missing required fields when loading Phase 3 config

---

## Before-First-Run Checklist

Run this checklist **before** starting training:

### Configuration
- [ ] `phases/phase1/configs/phase1_config.yaml` exists with all hyperparameters
- [ ] Parameter budget check: `sum(p.numel()) ∈ [24M, 26M]`
- [ ] W&B project initialized: `wandb.init(project="agent-forge-v2", name="phase1_cognate")`
- [ ] Datasets downloaded and verified (TinyStories, GSM8K-subset, Code-simple, Wikipedia)

### Architecture
- [ ] TitansMAG backbone: 8 layers, d=512, SW-Attn (window=1024)
- [ ] LMM memory: d_mem=256, decay=0.99, factorized
- [ ] MAG gate: hidden=256, entropy_reg=0.001
- [ ] TRM wrapper: T_max=3, micro_steps=2, detach_between_steps=true
- [ ] ACT head: halt_thresh=0.5, ema_teacher=0.98

### Optimizer
- [ ] MuonGrokfast config: α=0.98, λ=0.3, k=3, β=0.9, wd=0.05, lr=1e-3
- [ ] QK-clip: τ=30.0, per_head=true, mlarope_protected=true
- [ ] Fallback optimizer: AdamW for 1-D params
- [ ] Gradient clipping: max_norm=1.0

### Observability
- [ ] W&B metrics: loss/*, act/*, gate/*, optim/*
- [ ] Checkpoint saving: every 5000 steps, keep_last_n=3
- [ ] Logging frequency: log_every_n_steps=100
- [ ] Phase 2 merge preview: every 10K steps

---

## During-Training Checklist

Monitor these metrics **during** training (first 1000 steps):

### Loss & Learning
- [ ] `train/loss` decreasing smoothly (no spikes)
- [ ] `loss/step_0 > loss/step_1 > loss/step_2` (deep supervision working)
- [ ] `valid/perplexity` decreasing (generalization)

### ACT Monitoring
- [ ] `act/avg_steps ∈ [1.5, 4.0]` (not 0 or T_max)
- [ ] `act/halt_prob_step0 < 0.5` (model continues past first step)
- [ ] `act/ema_acc_step0 < act/ema_acc_step2` (later steps more accurate)

### Gate Monitoring
- [ ] `gate/entropy > 0.5` (not saturated)
- [ ] `gate/y_usage ∈ [0.3, 0.7]` (balanced y/m usage)
- [ ] `gate/m_usage > 0.2` (memory being used)

### Optimizer Monitoring
- [ ] `optim/qk_clip_count < 100/step` (not clipping too often)
- [ ] `optim/grad_ema_norm / optim/grad_raw_norm ∈ [1.0, 1.5]` (Grokfast not over-amplifying)
- [ ] `optim/ortho_error < 0.1` (Muon orthogonalizing correctly)

### Performance
- [ ] Training speed > 5000 tokens/sec on GTX 1660
- [ ] VRAM usage < 6GB with batch_size=32, seq_len=512
- [ ] W&B upload queue < 100 items

---

## Debugging Decision Tree

### Problem: Loss Not Decreasing

**Check**:
1. `train/loss` flat for > 1000 steps?
   - → Data quality issue (A2): Check `gate/m_usage`, `act/avg_steps`
   - → Mitigation: Verify datasets, increase min_length to 256

2. `loss/step_0` decreasing but `loss/step_2` not?
   - → Recursion instability (C2): Check `grad/norm_z`
   - → Mitigation: Ensure detach between steps, add gradient clipping

3. Loss oscillating (spikes every 100 steps)?
   - → Grokfast over-amplification (D1): Check `optim/grad_ema_norm`
   - → Mitigation: Reduce `lam` from 0.3 to 0.1

---

### Problem: ACT Pathological Halting

**Check**:
1. `act/avg_steps < 0.5`?
   - → Always halting at step 0 (C1)
   - → Mitigation: Lower `halt_thresh` to 0.3, check EMA calibration

2. `act/avg_steps = T_max`?
   - → Never halting (C1)
   - → Mitigation: Increase `halt_thresh` to 0.7, check `act/ema_acc_stepN`

3. `act/halt_prob_step0 > 0.9`?
   - → Model thinks step 0 is enough
   - → Mitigation: Increase entropy_reg to 0.01, use harder eval tasks

---

### Problem: Gate Saturation

**Check**:
1. `gate/y_usage > 0.9`?
   - → Memory unused (gate prefers y)
   - → Mitigation: Check `gate/entropy`, increase `entropy_reg` to 0.01

2. `gate/m_usage < 0.1`?
   - → Same as above
   - → Mitigation: Verify LMM memory is updating (`memory/decay` not too high)

3. `gate/entropy < 0.3`?
   - → Gate saturated (g → 0 or g → 1)
   - → Mitigation: Increase `entropy_reg` to 0.01, reduce LR for gate params

---

### Problem: Out of Memory

**Check**:
1. VRAM usage > 6GB with batch_size=1?
   - → Parameter budget overrun (B1)
   - → Mitigation: Verify `sum(p.numel())`, reduce d_mem or n_layers

2. VRAM spikes during recursion?
   - → Gradient accumulation across T_max steps
   - → Mitigation: Ensure `detach_between_steps=true`, enable gradient checkpointing

3. VRAM usage grows over time?
   - → Memory leak (cache not cleared)
   - → Mitigation: Add `torch.cuda.empty_cache()` every 1000 steps

---

### Problem: Training Too Slow

**Check**:
1. Speed < 2500 tokens/sec?
   - → W&B logging overhead (F1)
   - → Mitigation: Increase `log_every_n_steps` to 100, disable histograms

2. Speed varies (fast → slow)?
   - → Checkpoint saving blocking training
   - → Mitigation: Use async checkpoint saving, reduce `save_every_n_steps`

3. Speed degraded after 10K steps?
   - → Sequence length curriculum increased
   - → Expected: 512 → 1024 tokens halves speed

---

## Success Criteria Summary

| Metric | Target | Critical? |
|--------|--------|-----------|
| **Parameter Budget** | 25±1M | ✅ Yes |
| **Training Loss** | Smooth decrease, no spikes | ✅ Yes |
| **ACT Avg Steps** | 2-4 (not 0 or T_max) | ✅ Yes |
| **Gate Entropy** | > 0.5 | ✅ Yes |
| **Gate Y Usage** | 30-70% | ⚠️ Monitor |
| **QK-Clip Count** | < 100/step | ⚠️ Monitor |
| **Valid Perplexity** | < 20 on TinyStories | ✅ Yes |
| **Training Speed** | > 5000 tokens/sec | ⚠️ Monitor |
| **VRAM Usage** | < 6GB (batch=32, seq=512) | ✅ Yes |
| **Phase 2 Merge Fitness** | ≥ parent average | ⚠️ Monitor |

---

## References

**Source Documents**:
- Developer Guide #1: Muon × Grokfast (modular optimizer stack)
- Developer Guide #2: TRM × Titans × ACT (premortem + model system)

**Related Phase 1 Documents**:
- [TRM_TITANS_ARCHITECTURE.md](TRM_TITANS_ARCHITECTURE.md) - Architecture specification
- [LOGICAL_UNDERSTANDING.md](LOGICAL_UNDERSTANDING.md) - Conceptual overview
- [PHASE1_COMPLETE_GUIDE.md](PHASE1_COMPLETE_GUIDE.md) - Implementation guide

---

**Premortem Status**: ✅ Complete MECE Failure Tree (A1-G2)
**Next Step**: Review checklist before first training run
