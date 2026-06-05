# Phase 3: Quiet-STaR Implementation Plan

## Overview

Phase 3 implements **Self-Taught Reasoning** through a **TWO-STEP PROCESS**:
1. **Step 1: Prompt Baking** - Embed thinking tokens + reasoning patterns into weights
2. **Step 2: Quiet-STaR** - Parallel thought generation with REINFORCE training

## Architecture Summary

```
Phase 2 Champion Model
    |
    v
STEP 1: PROMPT BAKING (~5 min)
    |-- Add 8 thinking tokens: <think>, </think>, <step>, <reason>, <mece>, <falsify>, <expert>, <doubt>
    |-- Create 1600+ reasoning examples (7 advanced strategies)
    |-- Fine-tune with MuGrokfast (SUPERVISED config: lr=1e-4, lambda=0.2)
    |-- Validate convergence (>=85% accuracy)
    |
    v
Reasoning-Baked Model
    |
    v
STEP 2: QUIET-STAR (~2-4 hours)
    |-- Generate 4-8 parallel thought continuations (now uses <think> tags!)
    |-- Score thoughts (semantic 40% + syntactic 30% + predictive 30%)
    |-- Mix with attention-based fusion
    |-- Train with REINFORCE + MuGrokfast (RL config: lr=5e-4, lambda=0.1, QK-clip=25.0, KL=0.1)
    |
    v
Final Model: Baked Reasoning + Parallel Thoughts
```

---

## Step 1: Prompt Baking (FIRST!)

### 1.1 Add Thinking Tokens

```python
special_tokens = [
    '<think>',    # Start thinking
    '</think>',   # End thinking
    '<step>',     # Reasoning step
    '<reason>',   # Explicit reasoning
    '<mece>',     # MECE decomposition
    '<falsify>',  # Falsification check
    '<expert>',   # Expert perspective
    '<doubt>',    # Self-doubt check
]
```

### 1.2 Create Reasoning Dataset (1600+ examples)

| Strategy | Examples | Tag |
|----------|----------|-----|
| Chain-of-Thought | 400 | `<step>` |
| MECE Decomposition | 200 | `<mece>` |
| Falsification Testing | 200 | `<falsify>` |
| Expert Perspective | 200 | `<expert>` |
| Orthogonal Wisdom | 200 | (no tag) |
| Self-Doubt | 200 | `<doubt>` |
| Bayesian Reasoning | 200 | (no tag) |
| **Total** | **1600** | |

### 1.3 MuGrokfast Config (SUPERVISED/Baking)

```python
MuGrokConfig(
    muon_lr=1e-4,              # Lower for fine-tuning (not pretraining)
    fallback_lr=5e-5,
    grokfast_alpha=0.98,
    grokfast_lambda=0.2,       # Moderate filtering (supervised is less noisy)
    qk_clip_threshold=30.0,    # Standard clip
    kl_coefficient=0.0,        # No KL (we WANT to change model)
    phase=3,
    subphase="prompt_baking"
)
```

### 1.4 Validation Criteria

- `<think>` tag usage: >80% of responses
- `<step>` tag usage: >70% of responses
- MECE application: >60% on categorization problems
- Falsification usage: >60% on belief problems
- Self-doubt patterns: >50% of responses
- **Overall accuracy: >=85%**

---

## Step 2: Quiet-STaR (BUILDS ON BAKED FOUNDATION)

### 2.1 Thought Generator

```python
def generate_thoughts(baked_model, hidden_state, num_thoughts=4):
    thoughts = []
    for _ in range(num_thoughts):
        # Sample with temperature=1.0 for diversity
        thought = baked_model.generate(
            hidden_state,
            max_length=20,
            do_sample=True,
            temperature=1.0,
            top_p=0.9
        )
        # Baked model outputs: "<think><step>...</step></think>"
        thoughts.append(thought)
    return thoughts
```

### 2.2 Coherence Scorer (3 Metrics)

```python
coherence = (
    0.4 * semantic_coherence +   # Embedding similarity to context
    0.3 * syntactic_coherence +  # Grammar validity
    0.3 * predictive_utility     # How much thought helps next-token prediction
)
```

### 2.3 Mixing Head

```python
class MixingHead(nn.Module):
    def __init__(self, d_model, num_thoughts):
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)
        self.layer_norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, 1)

    def forward(self, base_hidden, thought_hidden):
        # Attention over thoughts
        attn_output, attn_weights = self.attention(
            query=base_hidden.unsqueeze(0),
            key=thought_hidden,
            value=thought_hidden
        )
        # Gating: blend thoughts with base
        gate = torch.sigmoid(self.gate(base_hidden))
        mixed = gate * attn_output + (1 - gate) * base_hidden
        return self.layer_norm(mixed + base_hidden)
```

### 2.4 MuGrokfast Config (RL/Quiet-STaR)

```python
MuGrokConfig(
    muon_lr=5e-4,              # HIGHER for RL exploration
    fallback_lr=1e-4,
    grokfast_alpha=0.98,
    grokfast_lambda=0.1,       # LOWER (RL gradients noisier, need more filtering)
    qk_clip_threshold=25.0,    # TIGHTER (RL causes attention spikes)
    kl_coefficient=0.1,        # NEW: Prevent drift from baked baseline
    phase=3,
    subphase="quietstar_rl"
)
```

### 2.5 REINFORCE Training

```python
for batch in dataloader:
    # Generate thoughts at difficult positions
    thoughts = generate_thoughts(baked_model, hidden_state)

    # Score thoughts
    scores = [score_coherence(t, hidden_state) for t in thoughts]

    # Mix thoughts
    enhanced = mix_thoughts(hidden_state, thoughts, scores)

    # Predict next token
    logits = baked_model.predict(enhanced)
    correct = (logits.argmax() == labels)

    # REINFORCE: Reward thoughts if prediction correct
    reward = 1.0 if correct else 0.0
    loss = -reward * torch.log(torch.tensor(scores).mean())

    # KL regularization (prevent drift from baked baseline)
    kl_loss = F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(base_logits, dim=-1)
    )

    total_loss = loss + 0.1 * kl_loss
    total_loss.backward()
    optimizer.step()
```

---

## Config Comparison: Baking vs RL

| Parameter | Prompt Baking (Step 1) | Quiet-STaR RL (Step 2) | Why? |
|-----------|------------------------|------------------------|------|
| `muon_lr` | 1e-4 | 5e-4 | RL needs higher LR to explore |
| `grokfast_lambda` | 0.2 | 0.1 | RL gradients noisier, need more filtering |
| `qk_clip_threshold` | 30.0 | 25.0 | RL causes attention spikes |
| `kl_coefficient` | 0.0 | 0.1 | RL can drift, need regularization |

---

## Anti-Theater Validation (CRITICAL)

Ensure thoughts are genuine, not empty "theater":

```python
def validate_no_theater(model, test_set):
    # Test 1: Thoughts differ from direct continuations
    divergence = mean([edit_distance(thought, direct) for thought in thoughts])
    assert divergence > 0.3, "Thoughts are trivial"

    # Test 2: Removing thoughts degrades performance
    acc_with = evaluate(model, use_thoughts=True)
    acc_without = evaluate(model, use_thoughts=False)
    assert acc_with > acc_without, "Thoughts don't help"

    # Test 3: Coherence-utility correlation
    correlation = pearson(utilities, coherences)
    assert correlation > 0.5, "Coherence doesn't predict utility"
```

---

## Expected Outcomes

### After Step 1 (Prompt Baking)
- Thinking token usage: >80%
- Structured reasoning: >70%
- Advanced strategies: >60%
- Convergence: >=85%

### After Step 2 (Quiet-STaR)
- GSM8K improvement: +5-10% (from ~4% to ~9-14%)
- Inference latency: <200ms (with thoughts)
- Anti-theater validation: PASS

---

## Implementation Steps

### Day 1: Prompt Baking Setup
1. Add 8 thinking tokens to tokenizer
2. Create reasoning dataset (1600+ examples)
3. Configure MuGrokfast (supervised config)
4. Run baking training (~5 min)
5. Validate convergence (>=85%)

### Day 2: Quiet-STaR Training
1. Implement ThoughtGenerator (parallel sampling)
2. Implement CoherenceScorer (3 metrics)
3. Implement MixingHead (attention-based)
4. Configure MuGrokfast (RL config)
5. Train with REINFORCE (~2-4 hours)
6. Run anti-theater validation

### Day 3: Evaluation & Handoff
1. Benchmark on GSM8K (full set)
2. Measure inference latency
3. Compare baked-only vs Quiet-STaR
4. Save model for Phase 4

---

## Files to Create/Modify

```
src/phase3_quietstar/
    __init__.py
    config.py                    # QuietSTaRConfig
    thought_generator.py         # ThoughtGenerator class
    coherence_scorer.py          # CoherenceScorer class
    mixing_head.py              # MixingHead class
    trainer.py                  # REINFORCE training loop
    theater_detector.py         # Anti-theater validation

scripts/
    run_phase3_baking.py        # Step 1: Prompt baking
    run_phase3_quietstar.py     # Step 2: Quiet-STaR training
    benchmark_phase3.py         # Evaluation script

data/
    reasoning_examples.json     # 1600+ examples
```

---

## Dependencies

- PEFT (for LoRA adapters in prompt baking)
- MuGrokfast optimizer (already implemented)
- Prompt baking module (already implemented)
- Tokenizer with special token support

---

## Estimated Time

| Step | Duration |
|------|----------|
| Prompt Baking (Step 1) | ~5 min |
| Quiet-STaR Training (Step 2) | 2-4 hours |
| Evaluation & Validation | ~30 min |
| **Total** | **~3-5 hours** |

---

## Key Insight: Why Two Steps?

**Without Prompt Baking (V1):**
- Quiet-STaR starts from scratch
- High RL variance (unstable training)
- Thoughts are unstructured: "and then 5", "plus 3 equals"
- Slow convergence

**With Prompt Baking (V2):**
- Model already knows how to "think" with tags
- RL training is 30-50% faster (jumpstart effect)
- Thoughts are structured: `<think><step>Subtract 5</step></think>`
- Stable convergence, better generalization

---

## Next Phase Preview

**Phase 4: BitNet** - 1.58-bit quantization
- Target: 8.2x compression, 3.8x speedup
- STE (Straight-Through Estimator) enabled via MuGrokfast
- Uses `muon_ste_mode=True`
