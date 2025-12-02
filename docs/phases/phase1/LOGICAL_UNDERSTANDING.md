# Phase 1: Cognate - Logical Understanding

**Version**: 2.0 (V2 Rebuild)
**Date**: 2025-10-15
**Phase Purpose**: Create 25M parameter TRM × Titans-MAG model with recursive reasoning

---

## ⚠️ UPDATED ARCHITECTURE

**This document provides a conceptual overview**. For complete technical details, see:

**Primary Documentation**:
- **[TRM_TITANS_ARCHITECTURE.md](TRM_TITANS_ARCHITECTURE.md)** - Complete architecture specification
- **[PREMORTEM_CHECKLIST.md](PREMORTEM_CHECKLIST.md)** - Failure modes and mitigations (A1-G2)
- **[configs/phase1_config.yaml](configs/phase1_config.yaml)** - Full configuration with all hyperparameters
- **[model/model_config.py](model/model_config.py)** - Dataclass configurations

**Developer Guides Referenced**:
- Developer Guide #1: Muon × Grokfast (modular optimizer stack)
- Developer Guide #2: TRM × Titans × ACT (model system)

---

## What This Phase Does (In Plain English)

Phase 1 creates a **25-million-parameter AI model** with recursive reasoning capabilities. Think of it like a model that can "think multiple times" before answering:

**Key Components**:
- **TitansMAG Backbone** (8 layers, 512 dim): Base transformer with sliding window attention + long-range memory
- **MAG Gate**: Decides when to use current thoughts vs stored memories
- **TRM Wrapper**: Multi-pass reasoning (think → refine → think again → answer)
- **ACT Head**: Knows when to stop thinking (adaptive computation)

**Why 25M parameters?** Small enough to run on a consumer GPU (GTX 1660 with 6GB VRAM), yet powerful enough for multi-step reasoning. Compare to GPT-3 (175 billion parameters) - our models are 7,000× smaller!

**Output**: 1 trained model (~25M params) with recursive reasoning, long-range memory, and adaptive computation.

---

## Why It Exists (Purpose in Pipeline)

Phase 1 is the **foundation** of Agent Forge V2. Without diverse starting models, Phase 2's evolutionary optimization has nothing to work with.

**Analogy**: If Agent Forge is like breeding champion dogs:
- **Phase 1** = Start with 3 different breeds (diversity is key)
- **Phase 2** = Breed them over 50 generations to create the perfect hybrid
- **Phases 3-8** = Train and optimize that hybrid

**Critical Insight**: The 25M parameter size enables **local deployment**. This is V2's killer feature - everything runs on your own hardware, not a cloud server.

---

## Key Research Papers

### 1. TinyTitans: "Neural Memory at Scale"
**What We Take From It**:
- **Long-Term Memory (LTM)**: Store important information across many tokens
- **Surprise-Based Gating**: Only save "surprising" (novel) information to memory
- **Memory Cross-Attention**: Let the model access stored memories when needed

**Paper's Contribution**: Showed you can add effective memory to small models without exploding parameter count.

**V2 Implementation**: We implement LTM with surprise gating exactly as described, targeting 2048-8192 memory slots.

### 2. HRM (Hierarchical Reasoning Models): "Two-Timescale Learning Without Intermediate Supervision"
**What We Take From It**:
- **No Intermediate Supervision**: Train end-to-end, don't need labeled reasoning steps
- **Fast/Slow Processing**: Some tokens need more compute (slow), some less (fast)
- **Adaptive Computation Time (ACT)**: Model decides when it's "done thinking" about a token

**Paper's Contribution**: Proved you can train reasoning models without expensive step-by-step labels.

**V2 Implementation**: We use ACT halting mechanism to let models think adaptively. This makes 25M param models "punch above their weight."

---

## Our Implementation Approach (V2)

### Architecture: TinyTitan + HRM Hybrid

```
Input Tokens
    ↓
Embedding Layer (512 dims)
    ↓
12× Transformer Blocks (ACT-enabled)
    ↓
ACT Halting Head ──→ "Should I keep thinking?" (Yes/No)
    ↓
Memory Gate ──→ "Is this surprising?" (Novelty score)
    ↓
Long-Term Memory Store (2048-8192 slots)
    ↓
Output Layer
    ↓
Logits (predictions)
```

**Key Parameters**:
- **Hidden Dim**: 512 (keeps model small)
- **Attention Heads**: 8 (standard)
- **Transformer Blocks**: 12 (balance depth vs size)
- **Total Parameters**: ~25,069,534 (target: 25M ±10%)

### What We're Copying from Papers

1. **From TinyTitans**:
   - ✅ Surprise-based memory gating (Equation 3 from paper)
   - ✅ Memory cross-attention mechanism
   - ✅ LTM size scaling (proportional to model size)

2. **From HRM**:
   - ✅ ACT halting with learned threshold
   - ✅ Two-timescale processing (fast tokens skip computation)
   - ✅ End-to-end training (no intermediate supervision)

### What We're Adding (Our Insights)

1. **Model Specialization**:
   - **Innovation**: Train 3 models with different hyperparameters (not in papers)
   - **Rationale**: Phase 2 evolution needs diversity, monoculture = poor evolution
   - **Example**:
     - Model 1: ACT threshold=0.95, LTM=4096, surprise=0.7 (reasoning-focused)
     - Model 2: ACT threshold=0.90, LTM=8192, surprise=0.5 (memory-focused)
     - Model 3: ACT threshold=0.99, LTM=2048, surprise=0.3 (speed-focused)

2. **Local Hardware Optimization**:
   - **Innovation**: Target exactly 25M params to fit in 6GB VRAM
   - **Rationale**: Papers assume cloud GPUs, we assume GTX 1660
   - **Implementation**: Tune hidden_dim + num_layers to hit 25M target

3. **Grokfast Pre-Training** (if time permits in Week 2):
   - **Innovation**: Use Grokfast (from Phase 5) during initial training
   - **Rationale**: Faster convergence for Phase 1 models
   - **Note**: Papers don't mention this, it's a V2 optimization

### What We're Simplifying (For Local Use)

1. **Training Dataset Size**:
   - **Papers**: Train on billions of tokens
   - **V2**: Train on 200,000+ samples (16 datasets from HuggingFace)
   - **Rationale**: Proof-of-concept with diverse data, not production-scale training
   - **Impact**: Models will be less capable than full-scale, but pipeline still works
   - **See**: [PHASE1_DATASET_SPECIFICATION.md](../../docs/PHASE1_DATASET_SPECIFICATION.md) for complete dataset details

2. **Memory Size**:
   - **TinyTitans Paper**: LTM up to 65,536 slots
   - **V2**: LTM capped at 8,192 slots (Model 2 only)
   - **Rationale**: Fits in 6GB VRAM, good enough for proof-of-concept

3. **Training Time**:
   - **Papers**: Train for days/weeks
   - **V2**: Train for hours (3-8 hours per model on GTX 1660)
   - **Rationale**: Fast iteration, not state-of-the-art performance

---

## Technical Flow

### Step 1: Initialize Model Architecture

```python
class TinyTitanModel(nn.Module):
    def __init__(
        self,
        vocab_size=50257,        # GPT-2 tokenizer
        hidden_dim=512,          # Embedding dimension
        num_layers=12,           # Transformer blocks
        num_heads=8,             # Attention heads
        ltm_size=4096,           # Memory slots
        act_threshold=0.95,      # Halting threshold
        surprise_threshold=0.7,  # Memory gating threshold
    ):
        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        # 12 Transformer blocks with ACT
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockWithACT(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # ACT halting head
        self.act_head = nn.Linear(hidden_dim, 1)
        self.act_threshold = act_threshold

        # Memory gate (surprise detector)
        self.memory_gate = MemoryGate(hidden_dim, surprise_threshold)

        # Long-term memory
        self.ltm = LongTermMemory(ltm_size, hidden_dim)

        # Output layer
        self.output = nn.Linear(hidden_dim, vocab_size)
```

**Target Parameters**: ~25,069,534 (verify with `sum(p.numel() for p in model.parameters())`)

### Step 2: Implement ACT (Adaptive Computation Time)

```python
def forward_with_act(self, x):
    """
    Process tokens with adaptive computation.
    Some tokens halt early (fast), others continue (slow).
    """
    batch_size, seq_len, hidden_dim = x.shape
    halting_probs = torch.zeros(batch_size, seq_len)

    for block in self.transformer_blocks:
        # Transform
        x = block(x)

        # Compute halting probability
        halt = torch.sigmoid(self.act_head(x)).squeeze(-1)
        halting_probs += halt

        # Stop if threshold reached
        if (halting_probs >= self.act_threshold).all():
            break

    return x, halting_probs
```

**ACT Insight**: Tokens that are "easy" halt early (few blocks), "hard" tokens use all 12 blocks.

### Step 3: Implement Memory Gating (Surprise-Based)

```python
def update_memory(self, hidden_state):
    """
    Store hidden state in LTM only if it's "surprising" (novel).
    """
    # Compute surprise score (how different from existing memories)
    surprise = self.memory_gate.compute_surprise(hidden_state, self.ltm)

    # Gate: only store if surprise > threshold
    if surprise > self.surprise_threshold:
        self.ltm.store(hidden_state, surprise_score=surprise)
```

**Memory Insight**: Not all information is worth remembering. Only novel/surprising states get stored.

### Step 4: Train Model with HRM Approach (No Intermediate Supervision)

```python
def train_epoch(model, dataloader, optimizer):
    """
    Train end-to-end, no intermediate supervision needed.
    """
    for batch in dataloader:
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Forward pass
        logits, halting_probs = model(input_ids)

        # Loss: Cross-entropy + ACT regularization
        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        act_loss = halting_probs.mean()  # Penalize excessive computation
        loss = ce_loss + 0.01 * act_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Training Insight**: We only need (input, output) pairs. No need for "show your reasoning" labels.

### Step 5: Create 3 Specialized Models

```python
# Model 1: Reasoning-focused
model1 = TinyTitanModel(
    act_threshold=0.95,   # Think longer (slower halting)
    ltm_size=4096,        # Medium memory
    surprise_threshold=0.7  # Store less (only very novel)
)

# Model 2: Memory-focused
model2 = TinyTitanModel(
    act_threshold=0.90,   # Think even longer
    ltm_size=8192,        # Large memory
    surprise_threshold=0.5  # Store more (less selective)
)

# Model 3: Speed-focused
model3 = TinyTitanModel(
    act_threshold=0.99,   # Halt quickly
    ltm_size=2048,        # Small memory
    surprise_threshold=0.3  # Store even more (fast recall)
)

# Train all 3 (3-8 hours each on GTX 1660)
for model in [model1, model2, model3]:
    train(model, dataset='GSM8K', epochs=3)
```

---

## Expected Inputs

**Phase 1 has NO inputs** - it creates models from scratch.

**What We Need**:
1. **Training Datasets**: 16 datasets (~200,000+ samples)
   - **Math**: GSM8K, SVAMP, ASDiv
   - **Code**: Mini-MBPP, CodeXGLUE
   - **Science**: ARC-Easy, ARC-Challenge
   - **Multi-Hop QA**: HotpotQA, DROP, StrategyQA
   - **Commonsense**: PIQA, HellaSwag, BoolQ
   - **Language**: WikiText, FineWeb-Edu, OpenWebText (optional)
   - **Storage**: ~1.35 GB (or ~50 GB with full OpenWebText)
   - **Download Time**: ~10 minutes (or 1-2 hours with OpenWebText)
   - **Complete Specification**: See [PHASE1_DATASET_SPECIFICATION.md](../../docs/PHASE1_DATASET_SPECIFICATION.md)

2. **Compute Resources**:
   - GTX 1660 or better (6GB+ VRAM)
   - 16GB+ RAM
   - 50GB disk space (for datasets + models)

3. **Time**: 3-8 hours per model = 9-24 hours total

---

## Expected Outputs

**3 Trained Models**, each saved as:
```
phase1_outputs/
├── model1_reasoning.pt      # 25M params, reasoning-focused
├── model2_memory.pt         # 25M params, memory-focused
├── model3_speed.pt          # 25M params, speed-focused
├── model1_config.json       # Hyperparameters
├── model2_config.json
├── model3_config.json
└── training_logs.json       # W&B logs
```

**Validation Metrics** (logged to W&B):
- **Parameter Count**: ~25M each (±10%)
- **GPU Memory**: <6GB per model
- **Inference Latency**: <100ms per forward pass
- **Training Loss**: Decreasing over epochs
- **GSM8K Accuracy**: >10% (baseline: ~3% random guessing)

---

## Critical Implementation Notes

### 1. Parameter Count Must Be Exact

**Problem**: Easy to accidentally create 30M or 20M param model.

**Solution**: Calculate parameters after initialization:
```python
model = TinyTitanModel(hidden_dim=512, num_layers=12, ...)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # Should be ~25,069,534

if total_params > 26_000_000:
    raise ValueError(f"Model too large ({total_params} params), reduce hidden_dim or num_layers")
```

**Tuning**: Adjust `hidden_dim` (512 → 480) or `num_layers` (12 → 11) to hit 25M target.

### 2. VRAM Usage Must Fit in 6GB

**Problem**: Training uses more memory than inference (gradients, optimizer states).

**Solution**: Enable gradient checkpointing:
```python
model = TinyTitanModel(...).cuda()
model.gradient_checkpointing_enable()  # Trades compute for memory

# Test VRAM usage
torch.cuda.reset_peak_memory_stats()
output = model(input_ids)
peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
assert peak_memory < 6.0, f"Model uses {peak_memory:.1f}GB, need <6GB"
```

### 3. ACT Threshold Tuning

**Problem**: Wrong threshold = models all halt at same time (no diversity).

**Solution**: Validate diversity:
```python
# After training all 3 models
halting_steps = [
    compute_avg_halting_steps(model1),
    compute_avg_halting_steps(model2),
    compute_avg_halting_steps(model3)
]
# Should be different (e.g., [8.2, 9.5, 6.1])
assert max(halting_steps) - min(halting_steps) > 2.0, "Models not diverse enough"
```

### 4. Dataset Format and Curriculum

**Problem**: 16 different datasets with different formats.

**Solution**: Use standardized format + 3-stage curriculum:
```python
# Standardized format for all datasets
{
    "text": "Q: ... A: ...",  # Full text
    "input": "<question>",
    "output": "<answer>",
    "metadata": {"dataset": "GSM8K", "type": "math_reasoning"}
}

# 3-stage curriculum
# Stage 1 (Epochs 1-3): Foundation - GSM8K, SVAMP, Mini-MBPP
# Stage 2 (Epochs 4-6): Reasoning - ARC-Easy, ARC-Challenge, PIQA, WikiText
# Stage 3 (Epochs 7-10): Advanced - HotpotQA, DROP, HellaSwag, FineWeb-Edu
```
**See**: [PHASE1_DATASET_SPECIFICATION.md](../../docs/PHASE1_DATASET_SPECIFICATION.md) for complete details

---

## Success Criteria

### Technical Validation
- ✅ 3 models created
- ✅ Each model ~25M params (±10%)
- ✅ Each model fits in 6GB VRAM (test on GTX 1660)
- ✅ ACT halting works (models halt at different times)
- ✅ LTM stores memories (surprise gating functional)
- ✅ Inference latency <100ms per forward pass
- ✅ ≥90% test coverage for Phase 1 code

### Research Validation
- ✅ Models show different behaviors (reasoning vs memory vs speed)
- ✅ ACT reduces computation on "easy" tokens (faster than fixed 12 layers)
- ✅ LTM stores only surprising information (not everything)
- ✅ GSM8K accuracy >10% (proves models learned something)

### Deliverables
- ✅ 3 trained models (`.pt` files)
- ✅ Config files (hyperparameters)
- ✅ Training logs (W&B)
- ✅ Unit tests (≥90% coverage)
- ✅ Documentation (`docs/phase1-architecture.md`)

---

## Integration Points

### Input from Previous Phase
**None** - Phase 1 is the starting point.

### Output to Next Phase
**Phase 2 (EvoMerge)** receives:
- 3 trained models (starting population for evolution)
- Model configs (for mutation/crossover)
- Performance metrics (for fitness evaluation)

**Interface**:
```python
# Phase 1 output format
phase1_output = {
    "models": [model1, model2, model3],
    "configs": [config1, config2, config3],
    "metrics": {
        "model1": {"accuracy": 0.12, "latency_ms": 85},
        "model2": {"accuracy": 0.15, "latency_ms": 95},
        "model3": {"accuracy": 0.10, "latency_ms": 75}
    }
}

# Phase 2 input expectation
phase2_input = phase1_output  # Direct handoff
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Models Don't Fit in 6GB VRAM
**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size (32 → 16 → 8)
2. Enable gradient checkpointing
3. Reduce `hidden_dim` (512 → 480)
4. Use mixed precision training (`torch.cuda.amp`)

### Pitfall 2: ACT Never Halts (Infinite Loop)
**Symptom**: Training hangs, GPU at 100% indefinitely

**Solutions**:
1. Add max iterations cap: `for block in blocks[:max_iters]`
2. Lower ACT threshold (0.95 → 0.90)
3. Debug ACT head output: `print(halt.mean())` (should increase over blocks)

### Pitfall 3: Models All Behave Identically
**Symptom**: Phase 2 evolution stagnates (no diversity)

**Solutions**:
1. Widen hyperparameter ranges (ACT: 0.90-0.99, LTM: 2048-8192)
2. Use different random seeds for initialization
3. Train on different subsets of GSM8K (model1: easy problems, model2: hard problems)

### Pitfall 4: Training Loss Doesn't Decrease
**Symptom**: Loss stays flat after 1 epoch

**Solutions**:
1. Check learning rate (try 1e-4 instead of 1e-3)
2. Verify dataset format (input/output pairs correct?)
3. Check gradient flow: `print([p.grad.abs().mean() for p in model.parameters()])`
4. Simplify: Train vanilla transformer first (no ACT/LTM), add features incrementally

---

## Validation Checklist

Before moving to Phase 2, verify:
- [ ] 3 models created and saved (`.pt` files exist)
- [ ] Each model ~25M params (check with `sum(p.numel())`)
- [ ] Each model fits in 6GB VRAM (test with batch_size=1 on GTX 1660)
- [ ] ACT halting works (print halting steps, should vary)
- [ ] LTM stores memories (check `len(model.ltm.memories) > 0` after training)
- [ ] Models are diverse (different halting steps, different accuracies)
- [ ] Inference fast (<100ms per forward pass)
- [ ] Unit tests pass (≥90% coverage)
- [ ] Documentation complete (`docs/phase1-architecture.md`)
- [ ] W&B logs uploaded (training curves visible)

---

## V2-Specific Notes

### Why Phase 1 is Different in V2

**V1 Phase 1**:
- ❌ Incomplete implementation (missing `execute()` method)
- ❌ No clear model size target
- ❌ No local hardware optimization

**V2 Phase 1**:
- ✅ Complete clean implementation from scratch
- ✅ Explicit 25M param target for local deployment
- ✅ GTX 1660 (6GB VRAM) as target hardware
- ✅ 3 specialized models (diversity by design)

### Key V2 Innovations
1. **Local-First Design**: 25M params fits in consumer GPU
2. **Diversity by Design**: 3 models with different hyperparameters
3. **Grokfast Integration** (optional): Use Phase 5 technique in Phase 1
4. **Validation-First**: Test VRAM usage immediately, don't wait until Phase 2

---

## User Interface Specification

**Complete UI documentation**: [PHASE1-4_UI_SPECIFICATION_V2.md](../../docs/PHASE1-4_UI_SPECIFICATION_V2.md#phase-1-cognate-ui)

### UI Components for Phase 1

#### 1. Configuration Panel (Pre-Launch)

**Purpose**: Configure 3 models before training starts

**Key Controls**:
- **Model Specializations**: Configure ACT threshold, LTM slots, dataset mix for each model
- **Optimizer Settings**: Learning rate, Grokfast lambda, QK clip threshold
- **Hardware Configuration**: Auto-detected GPU, VRAM allocation, batch size
- **Dataset Selection**: 16 datasets with validation checkboxes
- **Failure Detection Settings**: Enable/disable automatic interventions

**User Actions**:
```
1. Adjust ACT thresholds for diversity (Model 1: 0.95, Model 2: 0.90, Model 3: 0.99)
2. Set LTM sizes (Model 1: 2048, Model 2: 8192, Model 3: 2048)
3. Choose dataset mix percentages
4. Enable OOM prevention (dynamic batch sizing)
5. Click "🚀 Start Phase 1"
```

#### 2. Real-Time Training View

**Purpose**: Monitor 3-model training with intervention controls

**Key Visualizations**:
- **Loss Curves**: Real-time streaming for all 3 models (Plotly)
- **Diversity Metrics**: Halting diversity, memory diversity, speed diversity (updated every epoch)
- **Resource Usage**: VRAM, RAM, GPU utilization, temperature
- **Training Progress**: Per-model progress bars with ETA

**User Actions**:
```
1. Monitor loss convergence (should decrease to ~2.5)
2. Check diversity metrics (halting >2.0, memory >0.3, speed >10ms)
3. View VRAM usage (should stay <5GB per model)
4. Pause/resume/checkpoint at any time
```

#### 3. Detailed Model View

**Purpose**: Deep dive into single model training

**Key Metrics**:
- **Training Metrics**: Loss, perplexity, learning rate, grad norm
- **ACT Analysis**: Halting step distribution, average halting steps
- **LTM Usage**: Memory slots used, retrieval rate, write rate
- **Dataset Progress**: Current curriculum stage (Foundation/Reasoning/Advanced)

**User Actions**:
```
1. Click "🔍 Details" on any model card
2. View ACT heatmap by token type
3. Visualize memory contents
4. Export model early if needed
```

#### 4. Failure Intervention Modals

**Auto-Triggers**:
- **Diversity Check Failed** (after each epoch): Models converging to similar behavior
- **Out of Memory** (during training): VRAM exceeded, auto-recovery applied
- **Training Doesn't Converge** (loss plateau): Loss not decreasing for 50 steps

**User Response Options**:
```
For Diversity Failure:
  [⚡ Apply Aggressive Divergence] - Increase config differences, restart from checkpoint
  [🔧 Manual Adjustment] - Edit configs manually
  [▶️ Continue Anyway] - Accept lower diversity (not recommended)
  [⏹ Stop Training] - Investigate manually

For OOM:
  [✅ Accept Auto-Recovery] - Reduced batch size, gradient accumulation enabled
  [🔧 Manual Config] - Adjust settings yourself
  [⏹ Stop] - Abort training

For Convergence Issues:
  [↻ Adjust Learning Rate] - Increase/decrease LR automatically
  [⏸ Checkpoint & Investigate] - Pause and review logs
  [▶️ Continue Training] - Give it more time
```

#### 5. Completion Summary

**Purpose**: Show Phase 1 results before proceeding to Phase 2

**Key Information**:
- **Model Comparison**: Side-by-side metrics (perplexity, accuracy, halting, LTM usage)
- **Training Summary**: Total time, epochs, steps, interventions
- **Diversity Score**: Overall diversity rating (0.0-1.0, target >0.7)
- **Next Steps**: Ready for Phase 2 button

**User Actions**:
```
1. Review model diversity (should be high)
2. Check accuracy metrics (GSM8K >10%, ARC >25%)
3. Export models if needed
4. Click [▶️ Start Phase 2] to proceed
```

### API Endpoints Used

```
Configuration:
POST /api/phases/1/configure
  Body: {
    "models": [
      {"specialization": "reasoning", "act_threshold": 0.95, "ltm_slots": 2048, ...},
      {"specialization": "memory", "act_threshold": 0.90, "ltm_slots": 8192, ...},
      {"specialization": "speed", "act_threshold": 0.99, "ltm_slots": 2048, ...}
    ],
    "optimizer": {"lr": 1e-3, "grokfast_lambda": 0.3, ...},
    "datasets": ["gsm8k", "svamp", "arc-easy", ...],
    "failure_detection": {"diversity": true, "oom": true, "convergence": true}
  }

Training Control:
POST /api/phases/1/start
POST /api/phases/1/pause
POST /api/phases/1/resume
POST /api/phases/1/checkpoint

Monitoring:
GET /api/phases/1/status
  Response: {
    "phase": "cognate",
    "status": "training",
    "models": [
      {"id": "model1", "epoch": 8, "loss": 2.34, "progress": 0.80},
      {"id": "model2", "epoch": 8, "loss": 2.28, "progress": 0.82},
      {"id": "model3", "epoch": 7, "loss": 2.51, "progress": 0.75}
    ],
    "diversity": {"halting": 2.4, "memory": 0.47, "speed": 14.2},
    "vram_usage": 4.8
  }

GET /api/phases/1/diversity
  Response: {
    "halting_diversity": 2.4,
    "memory_diversity": 0.47,
    "speed_diversity": 14.2,
    "overall_score": 0.82
  }

Interventions:
POST /api/phases/1/intervene
  Body: {
    "type": "diversity_failure",
    "action": "apply_aggressive_divergence",
    "checkpoint": "epoch_5"
  }
```

### WebSocket Events Emitted

```javascript
// Real-time training updates (every step)
socket.on('phase:progress', (data) => {
  // data = { phase: 'cognate', model: 'model1', step: 2847, loss: 2.34, ... }
});

// Metric updates (every 100 steps)
socket.on('phase:metric', (data) => {
  // data = { phase: 'cognate', metric: 'loss', value: 2.34, timestamp: ... }
});

// Failure alerts (immediate)
socket.on('phase:alert', (data) => {
  // data = { phase: 'cognate', type: 'oom', severity: 'high', ... }
  // Show intervention modal
});

// Phase completion (once)
socket.on('phase:complete', (data) => {
  // data = { phase: 'cognate', models: 3, duration: 22.5hrs, ... }
  // Show completion summary
});

// Hardware updates (every 10 seconds)
socket.on('hardware:update', (data) => {
  // data = { vram: 4.8, ram: 8.2, gpu_util: 95, temp: 72 }
});
```

### Weights & Biases Integration

**Logged Metrics** (see [PHASE1-4_WANDB_INTEGRATION.md](../../docs/PHASE1-4_WANDB_INTEGRATION.md#phase-1-cognate)):
- Per step: `loss`, `learning_rate`, `grad_norm`
- Per epoch: `epoch_loss`, `diversity/halting`, `diversity/memory`, `diversity/speed`
- Per model: `model1/loss`, `model2/loss`, `model3/loss`
- Final: `total_params`, `training_time`, `diversity_score`

**UI Integration**:
```
[📊 Open W&B Dashboard] button links to:
  https://wandb.ai/agent-forge-v2/phase1-cognate/runs/{run_id}
```

### Mobile Responsiveness

**Desktop (≥1440px)**: 3-column layout (one per model)
**Laptop (1024-1439px)**: 2-column layout with scrollable third
**Tablet (768-1023px)**: Stacked cards, one model per row
**Mobile (<768px)**: Single column, swipeable model cards

### Accessibility

- **Keyboard Navigation**: All buttons accessible via Tab, Enter, Space
- **Screen Readers**: ARIA labels on all controls
- **Status Icons**: ✅ ⚠️ ❌ ⏳ ⏸ (don't rely on color alone)
- **High Contrast**: Support for high contrast mode
- **Font Size**: Adjustable (default 16px)

---

**Next Phase**: [Phase 2: EvoMerge - Logical Understanding](../phase2/LOGICAL_UNDERSTANDING.md)

**Version**: 2.0
**Last Updated**: 2025-10-16
**Status**: ✅ Ready for Implementation (with UI Spec)
