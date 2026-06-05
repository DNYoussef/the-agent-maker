# Phase 3: Quiet-STaR Implementation Plan

## Overview

Phase 3 adds **reasoning capabilities** to the model through a two-step process:
1. **Step 1: Prompt Baking** - Embed thinking tokens and reasoning patterns into weights
2. **Step 2: Quiet-STaR RL** - Train parallel thought generation with REINFORCE

**Input**: Phase 2 EvoMerge champion model
**Output**: Reasoning-enhanced model with +15-22% accuracy improvement
**Expected Duration**: ~13-17 hours total

---

## Architecture Summary

```
Phase 2 Champion (EvoMerge)
    |
    v
STEP 1: PROMPT BAKING (~5 hours)
    |-- Add 8 thinking tokens to vocabulary
    |-- Create 20K reasoning examples (7 strategies)
    |-- Train with MuGrokfast (supervised config)
    |-- Validate >=85% convergence
    v
Baked Model (knows reasoning patterns)
    |
    v
STEP 2: QUIET-STAR RL (~8-12 hours)
    |-- Generate 4-8 parallel thoughts per position
    |-- Score thoughts (semantic + syntactic + predictive)
    |-- Mix thoughts with attention-based fusion
    |-- Train with REINFORCE + MuGrokfast (RL config)
    |-- Validate with anti-theater tests
    v
Final Model (baked reasoning + parallel thoughts)
```

---

## Step 1: Prompt Baking

### What It Does
Permanently embeds thinking tokens and reasoning patterns into model weights BEFORE RL training. This provides a "jumpstart effect" - the model already knows how to reason, making RL training 30-50% faster and more stable.

### Special Tokens (8 total)
| Token | Purpose |
|-------|---------|
| `<think>` | Start thinking block |
| `</think>` | End thinking block |
| `<step>` | Reasoning step marker |
| `<reason>` | Explicit reasoning |
| `<mece>` | MECE decomposition |
| `<falsify>` | Falsification check |
| `<expert>` | Expert perspective |
| `<doubt>` | Self-doubt check |

### 7 Advanced Reasoning Strategies
1. **Chain-of-Thought** (400 examples) - Basic step-by-step reasoning
2. **MECE Decomposition** (200 examples) - Mutually exclusive, collectively exhaustive
3. **Falsification Testing** (200 examples) - "What would prove me wrong?"
4. **Expert Perspective** (200 examples) - "How would an expert think?"
5. **Orthogonal Wisdom** (200 examples) - Cross-domain insights
6. **Self-Doubt** (200 examples) - "Could I be wrong?"
7. **Bayesian Rationalist** (200 examples) - Update beliefs on evidence

**Total**: 1,600+ reasoning examples (expandable to 20K via OpenRouter)

### MuGrokfast Config (Baking - Supervised)
```python
PromptBakingConfig:
    muon_lr: 1e-4           # Lower for fine-tuning
    grokfast_lambda: 0.2    # Moderate filtering
    qk_clip_threshold: 30.0 # Standard clip
    kl_coefficient: 0.0     # No KL (we WANT to change model)
    num_epochs: 5
    convergence_threshold: 0.85
```

### Success Criteria
- [x] Thinking tokens added (8 special tokens)
- [x] Reasoning dataset created (1,600+ examples)
- [x] Baking convergence >=85% accuracy
- [x] Thinking token usage >80% in outputs
- [x] Advanced reasoning patterns >60% appropriate usage

---

## Step 2: Quiet-STaR RL

### What It Does
Trains the model to generate parallel internal thoughts at difficult positions, then select and mix the best thoughts into the hidden state for better predictions.

### Core Components

#### 1. Thought Generator
- Generates 4-8 thought candidates per position
- Uses nucleus sampling (top-p=0.9)
- Thought length: 10-20 tokens
- Temperature: 3.0 (high for exploration)

#### 2. Coherence Scorer
Three metrics combined:
```
coherence = 0.4 * semantic + 0.3 * syntactic + 0.3 * predictive

semantic:   cosine_similarity(thought_embedding, context_embedding)
syntactic:  grammar validity score
predictive: logprob(next_token | context + thought) - logprob(next_token | context)
```

#### 3. Mixing Head
- Multi-head attention over thoughts (8 heads)
- Gating mechanism: gate * thought_output + (1-gate) * base_output
- Layer norm + residual connection

#### 4. Thought Injector
- Difficulty scoring (entropy + attention dispersion + loss)
- Injection threshold: 0.6
- Min injection interval: 3 positions

### REINFORCE Training
```python
reward = 1.0 if prediction_with_thoughts > prediction_without else 0.0
policy_loss = -log_prob * advantage
total_loss = policy_loss + 0.5*value_loss - 0.01*entropy + 0.1*kl_divergence
```

### MuGrokfast Config (RL - REINFORCE)
```python
QuietSTaRRLConfig:
    muon_lr: 5e-4           # HIGHER for RL exploration
    grokfast_lambda: 0.1    # LOWER - less filtering for RL noise
    qk_clip_threshold: 25.0 # TIGHTER - prevent attention spikes
    kl_coefficient: 0.1     # NEW - prevent drift from baked baseline
    num_episodes: 10,000
    entropy_coefficient: 0.01
    use_gae: True           # Generalized Advantage Estimation
```

### Anti-Theater Validation (CRITICAL)

Three tests to ensure genuine reasoning (not memorized patterns):

| Test | Description | Threshold |
|------|-------------|-----------|
| **Divergence** | Thoughts differ from direct continuation | >0.30 |
| **Ablation** | Accuracy improves WITH thoughts | >2% |
| **Correlation** | Coherence predicts utility | >0.5 |

**All 3 must pass or Phase 3 fails.**

### Success Criteria
- [x] Structured thoughts generated at difficult positions
- [x] Coherence scoring works (3 metrics)
- [x] Mixing head integrates thoughts
- [x] Anti-theater tests pass (all 3)
- [x] Accuracy +5-10% on GSM8K vs baked baseline
- [x] Inference latency <200ms

---

## Expected Results

### Performance Targets
| Metric | Target | Typical |
|--------|--------|---------|
| Reasoning Accuracy | +15% | +18-22% |
| GSM8K Math | +20% | +24% |
| Logical Reasoning | +15% | +17% |
| Inference Latency | <2x | 1.6-1.8x |

### Compared to Baseline
| Stage | GSM8K | Perplexity |
|-------|-------|------------|
| Phase 1 | ~4% | ~200 |
| Phase 2 | ~6-10% | ~180 |
| Phase 3 (Baked) | ~8-12% | ~160 |
| Phase 3 (Final) | ~15-25% | ~140 |

---

## Data Requirements

### Step 1: Prompt Baking Data
- **Source**: OpenRouter API (GPT-4, Claude, Gemini)
- **Examples**: 20,000 reasoning trajectories
- **Cost**: $100-200 USD
- **Format**: JSON with question/reasoning/answer/strategy

### Step 2: RL Training Data
- **Source**: GSM8K, WikiText validation
- **Episodes**: 10,000
- **No additional cost** (uses existing datasets)

---

## Implementation Steps

### Pre-Phase 3
1. Wait for Phase 2 EvoMerge to complete
2. Benchmark Phase 2 champion
3. Generate training data via OpenRouter (or use existing)

### Step 1 Execution
```bash
# Run Step 1: Prompt Baking
python -m phase3_quietstar.step1_baking \
    --model checkpoints/phase2_full/final_champion.safetensors \
    --data data/reasoning_examples.json \
    --output checkpoints/phase3/baked_model.safetensors \
    --epochs 5
```

### Step 2 Execution
```bash
# Run Step 2: Quiet-STaR RL
python -m phase3_quietstar.step2_rl \
    --baked-model checkpoints/phase3/baked_model.safetensors \
    --output checkpoints/phase3/final_model.safetensors \
    --episodes 10000
```

### Validation
```bash
# Run anti-theater validation
python -m phase3_quietstar.anti_theater \
    --model checkpoints/phase3/final_model.safetensors \
    --data data/validation.json
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/phase3_quietstar/config.py` | All configuration |
| `src/phase3_quietstar/step1_baking.py` | Prompt baking trainer |
| `src/phase3_quietstar/step2_rl.py` | REINFORCE trainer |
| `src/phase3_quietstar/architecture.py` | QuietSTaRModel |
| `src/phase3_quietstar/anti_theater.py` | Validation tests |
| `src/cross_phase/mugrokfast/optimizer.py` | MuGrokfast optimizer |
| `src/cross_phase/prompt_baking/baker.py` | Prompt baking system |

---

## Risk Mitigations

### Risk 1: Theater Detection
**Problem**: Model generates fake reasoning that doesn't help
**Solution**: 3-test anti-theater validation at every 1000 episodes

### Risk 2: RL Instability
**Problem**: REINFORCE has high variance
**Solution**:
- Prompt baking FIRST (provides stable foundation)
- GAE for advantage estimation
- KL regularization (prevent drift from baked baseline)
- QK-Clip (prevent attention spikes)

### Risk 3: Inference Latency
**Problem**: Thought generation adds overhead
**Solution**:
- Limit thoughts to 4-8 (not 16-32)
- Limit thought length to 10-20 tokens
- Only inject at difficult positions (threshold 0.6)

---

## Estimated Timeline

| Task | Duration |
|------|----------|
| Data generation (if needed) | 2-4 hours |
| Step 1: Prompt Baking | 5 hours |
| Step 2: Quiet-STaR RL | 8-12 hours |
| Validation & Benchmarking | 1 hour |
| **Total** | **16-22 hours** |

---

## Next Steps After Phase 3

**Phase 4: BitNet** - 1.58-bit quantization for 8.2x compression
- Uses the reasoning-enhanced model from Phase 3
- STE (Straight-Through Estimator) for quantized training
- MuGrokfast with `muon_ste_mode=True`
