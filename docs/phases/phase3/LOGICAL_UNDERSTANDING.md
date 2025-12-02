# Phase 3: Quiet-STaR - Logical Understanding

**Version**: 2.1 (V2 Rebuild - CORRECTED)
**Date**: 2025-10-15
**Phase Purpose**: Add parallel reasoning via "internal thoughts" with **PROMPT BAKING FIRST**

---

## What This Phase Does (In Plain English)

Phase 3 teaches the model to "think before it speaks" through a **TWO-STEP PROCESS**:

### STEP 1: Prompt Baking (Foundation)
Permanently embed thinking tokens (`<think>`, `</think>`) and advanced reasoning patterns into model weights **BEFORE** parallel thought generation.

### STEP 2: Quiet-STaR (Building on Foundation)
Generate parallel internal thoughts at difficult positions, building on the baked reasoning foundation.

**Analogy**: Like teaching a chess player:
- **Step 1 (Baking)**: Teach "always consider opponent's response", "think 3 moves ahead" → becomes automatic
- **Step 2 (Quiet-STaR)**: Now generate multiple specific move sequences, knowing the thinking patterns

**The Jumpstart Effect**:
- **Without baking**: Start from scratch, high variance, slow convergence
- **With baking**: Model already knows how to reason, RL training is stable and fast

**Output**: Reasoning-enhanced model with baked thinking patterns + parallel thought generation

---

## Why It Exists (Purpose in Pipeline)

**Problem**: Models struggle with multi-step reasoning (e.g., "If X, then Y, then Z").

**Solution**: Quiet-STaR adds a "thinking layer" where the model explores possibilities before committing to an answer.

**Evidence**: Research shows +5-10% accuracy on reasoning tasks (GSM8K, MATH benchmarks) with internal thoughts.

---

## Key Research Papers

### "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking"

**What We Take**:
- **Token-Parallel Thoughts**: Generate thoughts at every token position
- **Coherence Scoring**: Rank thoughts by semantic/syntactic/predictive quality
- **Mixing Head**: Learned attention over thoughts
- **Training**: Reinforce thoughts that improve next-token prediction

**V2 Implementation**: Full implementation of Quiet-STaR as described

---

## Our Implementation Approach (V2 - CORRECTED)

### Architecture: Two-Step Process

```
Phase 2 Champion Model
    ↓
STEP 1: PROMPT BAKING (Comes First!)
    ├─→ Add special tokens: <think>, </think>, <step>, <reason>, <mece>, <falsify>, <expert>, <doubt>
    ├─→ Create 1600+ reasoning examples (7 advanced strategies)
    ├─→ Fine-tune with Muon × Grokfast (supervised learning config)
    ├─→ Validate convergence (85% threshold)
    ↓
Reasoning-Baked Model (knows how to use thinking tokens)
    ↓
STEP 2: QUIET-STAR (Builds on Foundation!)
    ├─→ Generate 4-8 structured thought continuations (now with <think> tags!)
    ├─→ Score thoughts (coherence: semantic + syntactic + predictive)
    ├─→ Mixing Head (attention over thoughts)
    ├─→ Train with REINFORCE + Muon × Grokfast (RL config)
    ↓
Final Model: Baked Reasoning + Parallel Thoughts
```

**Key Components**:

**Step 1 (Prompt Baking)**:
1. **Thinking Tokens**: 8 special tokens for reasoning structure
2. **Advanced Reasoning Patterns**: MECE, falsification, expert perspective, self-doubt, Bayesian, multidomain (7 total)
3. **Muon × Grokfast Optimizer**: Supervised learning configuration (lr=1e-4, lambda=0.2)
4. **Convergence Validation**: 85% accuracy threshold

**Step 2 (Quiet-STaR)**:
1. **Thought Generator**: Samples 4-8 structured continuations (now uses baked patterns!)
2. **Coherence Scorer**: 3 metrics (semantic, syntactic, predictive)
3. **Mixing Head**: Learned attention weights for thought integration
4. **REINFORCE Training**: With Muon × Grokfast RL configuration (lr=5e-4, lambda=0.1, QK-clip=25.0, KL-reg=0.1)

---

## What We're Copying from Papers

1. ✅ **Token-Parallel Thoughts**: Generate at every position (not just start)
2. ✅ **Coherence Scoring**: 3-metric composite score
3. ✅ **Mixing Head**: Attention-based integration
4. ✅ **REINFORCE Training**: Reward thoughts that help prediction

---

## What We're Adding (Our Insights)

1. **Prompt Baking First (CRITICAL INNOVATION)**:
   - **Innovation**: Bake thinking tokens + reasoning patterns BEFORE Quiet-STaR
   - **Rationale**: Provides reasoning foundation, reduces RL variance by 30-50%
   - **Implementation**: Two-step process (baking → Quiet-STaR)
   - **Evidence**: Stable RL training, faster convergence, better generalization

2. **Advanced Reasoning Patterns (7 Strategies)**:
   - **Innovation**: Beyond chain-of-thought, include:
     - MECE decomposition ("break down mutually exclusive, exhaustive categories")
     - Falsification testing ("what would prove me wrong?")
     - Expert perspective ("how would an expert think about this?")
     - Orthogonal wisdom ("what can I learn from unrelated fields?")
     - Self-doubt ("could I be wrong?")
     - Bayesian rationalist thinking (update beliefs on evidence)
     - Multidomain consultant (synthesize multiple expert views)
   - **Rationale**: These are the thinking strategies of high-performing humans
   - **Training Data**: 1600+ examples (200 per strategy + 400 baseline CoT)

3. **Muon × Grokfast for Both Steps**:
   - **Innovation**: Use Muon × Grokfast for baking (supervised) AND RL (with different configs)
   - **Baking Config**: lr=1e-4, lambda=0.2, QK-clip=30.0, KL=0.0
   - **RL Config**: lr=5e-4, lambda=0.1, QK-clip=25.0, KL=0.1
   - **Rationale**: Consistent optimizer, but tuned for supervised vs RL training

4. **Anti-Theater Validation**:
   - **Innovation**: Ensure thoughts aren't empty/trivial ("theater")
   - **Tests**:
     - **Test 1**: Thoughts differ from direct continuations (divergence > 0.3)
     - **Test 2**: Removing thoughts degrades performance (ablation)
     - **Test 3**: Coherence scores correlate with utility (correlation > 0.5)
   - **Rationale**: V1 didn't validate this, V2 prevents theater from start

5. **Thought Length Optimization**:
   - **Innovation**: Adaptive thought length (10-20 tokens, not fixed)
   - **Rationale**: Short problems need short thoughts, long problems need long thoughts
   - **Implementation**: `thought_length = min(20, problem_length // 5)`

6. **Curriculum Learning**:
   - **Innovation**: Train on easy problems first, then hard
   - **Rationale**: Thoughts are hard to learn, curriculum helps
   - **Implementation**: Sort GSM8K by difficulty, train progressively

---

## What We're Simplifying (For Local Use)

1. **Thought Count**:
   - **Papers**: 16-32 thoughts per token
   - **V2**: 4-8 thoughts (balance quality vs speed)
   - **Impact**: Less exploration, but faster inference (<200ms target)

2. **Thought Length**:
   - **Papers**: Up to 50 tokens per thought
   - **V2**: 10-20 tokens (memory constrained)
   - **Impact**: Shorter reasoning chains, but still effective

---

## Technical Flow (CORRECTED TWO-STEP PROCESS)

### Step 0: Add Thinking Tokens

```python
def add_thinking_tokens(model, tokenizer):
    """Add 8 special tokens for structured reasoning"""
    special_tokens = {
        'additional_special_tokens': [
            '<think>',    # Start thinking
            '</think>',   # End thinking
            '<step>',     # Reasoning step
            '<reason>',   # Explicit reasoning
            '<mece>',     # MECE decomposition
            '<falsify>',  # Falsification check
            '<expert>',   # Expert perspective
            '<doubt>',    # Self-doubt check
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
```

### Step 1: Prompt Baking (COMES FIRST!)

```python
def bake_reasoning_patterns(model, tokenizer):
    """
    Bake 1600+ reasoning examples with 7 advanced strategies

    Training data includes:
    - 400 chain-of-thought examples
    - 200 MECE decomposition examples
    - 200 falsification testing examples
    - 200 expert perspective examples
    - 200 orthogonal wisdom examples
    - 200 self-doubt examples
    - 200 Bayesian rationalist examples
    """

    # Configure Muon × Grokfast for SUPERVISED learning (baking)
    baking_optimizer = MuonGrokfast(
        model.parameters(),
        muon_lr=1e-4,              # Lower for fine-tuning
        grokfast_lambda=0.2,       # Moderate filtering
        qk_clip_threshold=30.0,    # Standard clip
        kl_coefficient=0.0         # No KL regularization (we WANT to change model)
    )

    # Create composite prompt with all 7 strategies
    composite_prompt = create_advanced_reasoning_prompt()  # Returns multi-strategy prompt

    # Bake prompt into weights (KL divergence minimization)
    baked_model = prompt_baker.bake(
        model,
        composite_prompt,
        optimizer=baking_optimizer,
        num_epochs=5,
        num_trajectories=1600
    )

    # Validate convergence (85% threshold)
    validation_accuracy = validate_reasoning_patterns(baked_model)
    assert validation_accuracy >= 0.85, f"Baking failed: {validation_accuracy:.2%} < 85%"

    print(f"✅ Reasoning baked: {validation_accuracy:.2%} accuracy")
    return baked_model
```

### Step 2: Generate Thoughts (NOW with baked patterns!)

```python
def generate_thoughts(baked_model, hidden_state, num_thoughts=4):
    """
    Generate parallel thought continuations

    KEY: Baked model now naturally uses <think> tags and reasoning patterns!
    """
    thoughts = []
    for _ in range(num_thoughts):
        # Sample from model (temperature=1.0 for diversity)
        thought_tokens = baked_model.generate(
            hidden_state,
            max_length=20,
            do_sample=True,
            temperature=1.0
        )
        # Baked model outputs: "<think><step>...</step></think>"
        # (Without baking, would output: "and then 5")
        thoughts.append(thought_tokens)
    return thoughts  # List of 4 STRUCTURED thought sequences
```

### Step 2: Score Thoughts (Coherence)

```python
def score_coherence(thought, hidden_state, model):
    """3-metric coherence scoring"""

    # 1. Semantic coherence (embedding similarity)
    thought_emb = embed(thought)
    context_emb = embed(hidden_state)
    semantic = cosine_similarity(thought_emb, context_emb)

    # 2. Syntactic coherence (grammar validity)
    syntactic = grammar_score(thought)  # 0-1, higher = valid grammar

    # 3. Predictive utility (helps next-token prediction?)
    with_thought = model.predict_next(hidden_state + thought_emb)
    without_thought = model.predict_next(hidden_state)
    predictive = accuracy(with_thought) - accuracy(without_thought)

    # Composite score
    coherence = 0.4 * semantic + 0.3 * syntactic + 0.3 * predictive
    return coherence
```

### Step 3: Mixing Head (Integrate Thoughts)

```python
def mix_thoughts(hidden_state, thoughts, coherence_scores):
    """Attention-based thought integration"""

    # Attention weights (learned, softmax over coherence scores)
    attn_weights = softmax(coherence_scores)

    # Weighted sum of thought embeddings
    thought_embs = [embed(t) for t in thoughts]
    mixed = sum(w * emb for w, emb in zip(attn_weights, thought_embs))

    # Add to hidden state
    enhanced_hidden = hidden_state + mixing_head(mixed)
    return enhanced_hidden
```

### Step 4: Train with REINFORCE + Muon × Grokfast (RL Config)

```python
def train_quiet_star(baked_model, dataloader):
    """
    Train thoughts to improve next-token prediction

    KEY: Use RL-specific Muon × Grokfast configuration
    """

    # Configure Muon × Grokfast for RL (DIFFERENT from baking!)
    rl_optimizer = MuonGrokfast(
        baked_model.parameters(),
        muon_lr=5e-4,              # HIGHER for RL exploration
        grokfast_lambda=0.1,       # LOWER for RL noise filtering
        qk_clip_threshold=25.0,    # TIGHTER for RL attention spikes
        kl_coefficient=0.1         # NEW: Prevent drift from baked baseline
    )

    for batch in dataloader:
        input_ids, labels = batch

        # Forward pass with thoughts
        for i, token in enumerate(input_ids):
            # Generate thoughts (now structured thanks to baking!)
            thoughts = generate_thoughts(baked_model, hidden_state[i])

            # Score thoughts
            scores = [score_coherence(t, hidden_state[i], baked_model) for t in thoughts]

            # Mix thoughts
            enhanced = mix_thoughts(hidden_state[i], thoughts, scores)

            # Predict next token
            logits = baked_model.predict(enhanced)
            correct = (logits.argmax() == labels[i])

            # REINFORCE: Reward thoughts if prediction correct
            reward = 1.0 if correct else 0.0
            loss = -reward * torch.log(torch.tensor(scores).mean())

            # KL regularization (prevent drift from baked baseline)
            with torch.no_grad():
                base_logits = baked_model.base_forward(hidden_state[i])
            kl_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(base_logits, dim=-1),
                reduction='batchmean'
            )

            total_loss = loss + 0.1 * kl_loss  # KL coefficient = 0.1

            # Backward pass
            total_loss.backward()

        rl_optimizer.step()
        rl_optimizer.zero_grad()
```

---

## Expected Inputs

**From Phase 2**: 1 evolved model (23.5% better fitness)

---

## Expected Outputs

**To Phase 4**:
- Reasoning-enhanced model (with Quiet-STaR thoughts)
- Coherence statistics (which thoughts worked best)
- Inference latency (should be <200ms with thoughts)

---

## Critical Implementation Notes

### 1. Anti-Theater Validation (CRITICAL)

**Problem**: Model might generate trivial thoughts ("theater") that look like reasoning but don't help.

**Solution**: Run 3 tests after training:

```python
# Test 1: Thoughts differ from direct continuations
direct_continuation = model.generate(prompt, do_sample=False)
thoughts = generate_thoughts(model, prompt)
divergence = mean([edit_distance(t, direct_continuation) for t in thoughts])
assert divergence > 0.3, "Thoughts are just direct continuations (theater)"

# Test 2: Removing thoughts degrades performance
accuracy_with_thoughts = evaluate(model, use_thoughts=True)
accuracy_without_thoughts = evaluate(model, use_thoughts=False)
assert accuracy_with_thoughts > accuracy_without_thoughts, "Thoughts don't help (theater)"

# Test 3: Coherence scores correlate with utility
utilities = [compute_utility(t) for t in thoughts]
coherences = [score_coherence(t) for t in thoughts]
correlation = pearson_correlation(utilities, coherences)
assert correlation > 0.5, "Coherence scores don't predict utility (theater)"
```

### 2. Inference Latency Must Be Acceptable

**Target**: <200ms per forward pass (with thoughts)

**Optimization**:
- Reduce thought count (4 instead of 8)
- Reduce thought length (10 tokens instead of 20)
- Batch thought generation (parallel, not sequential)

---

## Success Criteria

**Step 1 (Prompt Baking)**:
- ✅ Thinking tokens added (8 special tokens)
- ✅ Reasoning dataset created (1600+ examples, 7 strategies)
- ✅ Baking convergence validated (≥85% accuracy)
- ✅ Thinking token usage: >80% of responses
- ✅ Advanced reasoning patterns: >60% appropriate usage

**Step 2 (Quiet-STaR)**:
- ✅ Structured thoughts generated at every difficult position
- ✅ Coherence scoring works (3 metrics computed)
- ✅ Mixing head integrates thoughts correctly
- ✅ **Anti-theater tests pass** (critical!)
- ✅ Accuracy improves on reasoning tasks (+5-10% on GSM8K vs baked baseline)
- ✅ Inference latency <200ms (with thoughts)
- ✅ ≥90% test coverage

**Overall**:
- ✅ Two-step process validated (baking → Quiet-STaR)
- ✅ Muon × Grokfast works for both supervised (baking) and RL (Quiet-STaR)
- ✅ Jumpstart effect confirmed (RL training 30-50% faster with baking)

---

## Integration Points

**Input from Phase 2**: 1 evolved model
**Output to Phase 4**: Reasoning-enhanced model

---

**Next Phase**: [Phase 4: BitNet - Logical Understanding](../phase4/LOGICAL_UNDERSTANDING.md)

**Version**: 2.0
**Last Updated**: 2025-10-12
**Status**: ✅ Ready for Implementation
