# Phase 6: Tool & Persona Baking - Logical Understanding

**Version**: 2.0 (Complete Redesign)
**Phase Purpose**: Iterative self-guided optimization through alternating tool and persona prompt baking
**Key Innovation**: Model discovers its own optimal thinking patterns through evolutionary refinement

---

## What This Phase Does (In Plain English)

Phase 6 is an **iterative optimization loop** that alternates between two types of prompt baking:

1. **Tool Use Optimization** (A-cycle): Improve SWE-Bench performance by refining how the model uses tools
2. **Self-Guided Persona** (B-cycle): Let the model generate and test its own thinking pattern prompts

The cycles alternate (A → B → A → B → ...) until both plateau, resulting in a model with:
- Optimized tool usage baked into weights
- Self-discovered best thinking patterns baked into weights

**This is NOT pre-defined personas**. The model **evolves its own identity** through empirical testing.

---

## The Core Process

### Input from Phase 5
- Specialized model (e.g., coding agent from Phase 5)
- Has eudaimonia system baked in
- Has tool use capability (from Phase 5 training)

### The Iterative Cycle

```
Phase 5 Model
    ↓
┌─────────────────────────────────────┐
│  A-CYCLE: Tool Use Optimization     │
│  ├─ Half-bake initial tool prompt   │
│  ├─ Test on SWE-Bench               │
│  ├─ Generate 4 tool-use variants    │
│  ├─ Test all 4 on SWE-Bench         │
│  ├─ Half-bake winner                │
│  └─ Check plateau? If yes → B-cycle │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  B-CYCLE: Self-Guided Persona       │
│  ├─ Model generates 4 self-prompts  │
│  ├─ Test on benchmark suite         │
│  │   (MMLU, GSM8K, HumanEval, etc.) │
│  ├─ Half-bake winner                │
│  └─ Check plateau? If yes → A-cycle │
└─────────────────────────────────────┘
    ↓
Repeat A → B → A → B until BOTH plateau
    ↓
Phase 7: Optimized Model
```

---

## A-Cycle: Tool Use Optimization (SWE-Bench Driven)

### Goal
Maximize SWE-Bench score (software engineering problem-solving) by optimizing tool usage patterns.

### Process

**Step 1: Initial Half-Bake**
```python
tool_prompt = "Use these tools: [search, calculator, code executor, file reader, database query]"
model = half_bake_prompt(model, tool_prompt, strength=0.5)  # ~2.5 min
```

**Step 2: Test Baseline**
```python
baseline_score = test_swe_bench(model)  # e.g., 35%
```

**Step 3: Generate 4 Tool-Use Variants**
```python
variants = [
    "Always use tools before answering coding questions",
    "Break problems into tool-solvable steps, then chain tools",
    "Use tools to verify your reasoning and test assumptions",
    "When stuck, try different tool combinations systematically"
]
```

**Step 4: Test All Variants**
```python
results = {}
for i, variant in enumerate(variants):
    test_model = apply_prompt(model, variant)  # NOT baked yet, just prompted
    results[i] = test_swe_bench(test_model)

# Example results:
# Variant 0: 37% (best!)
# Variant 1: 36%
# Variant 2: 35%
# Variant 3: 34%

winner = max(results, key=results.get)  # Variant 0
```

**Step 5: Half-Bake Winner**
```python
model = half_bake_prompt(model, variants[winner], strength=0.5)  # ~2.5 min
new_score = test_swe_bench(model)  # e.g., 37%
```

**Step 6: Check Plateau**
```python
improvement = new_score - baseline_score
if improvement < threshold:  # e.g., <0.5%
    plateau_detected = True
    switch_to_B_cycle()
else:
    repeat_A_cycle()  # Generate 4 new variants based on current model
```

### Key Insights

**Half-Baking**: Each iteration adds ~50% strength of the prompt
- Multiple half-bakes stack: 50% + 50% + 50% → cumulative effect
- Prevents catastrophic forgetting of previous bakes

**Variant Generation**: Each A-cycle generates NEW variants based on current model state
- Not the same 4 prompts every time
- Evolves as model improves

**Plateau Detection**: When SWE-Bench score stops improving (delta <0.5%)

---

## B-Cycle: Self-Guided Persona (Multi-Benchmark Driven)

### Goal
Let the model discover its own optimal thinking patterns through self-generated prompts.

### Process

**Step 1: Model Generates 4 Self-Prompts**
```python
prompt_to_model = """
Generate 4 different prompts that describe your own thinking process and identity.
These should capture how you approach problems, reason, and make decisions.
"""

self_prompts = model.generate(prompt_to_model, n=4)

# Example outputs:
# [0] "I approach problems by breaking them into smallest testable parts"
# [1] "I prioritize clarity and explanation over speed"
# [2] "I think recursively, building from base cases upward"
# [3] "I validate assumptions before proceeding with solutions"
```

**Step 2: Test on Benchmark Suite**
```python
benchmarks = ["MMLU", "GSM8K", "HumanEval", "MATH", "HellaSwag", "ARC"]

results = {}
for i, self_prompt in enumerate(self_prompts):
    test_model = apply_prompt(model, self_prompt)  # NOT baked yet

    scores = {}
    for benchmark in benchmarks:
        scores[benchmark] = run_benchmark(test_model, benchmark)

    # Aggregate score (weighted average or primary metric)
    results[i] = aggregate_scores(scores)

# Example results:
# Self-Prompt 0: 72.3% (best!)
# Self-Prompt 1: 70.1%
# Self-Prompt 2: 71.8%
# Self-Prompt 3: 69.5%

winner = max(results, key=results.get)  # Self-Prompt 0
```

**Step 3: Half-Bake Winner**
```python
model = half_bake_prompt(model, self_prompts[winner], strength=0.5)  # ~2.5 min
new_avg_score = test_benchmark_suite(model)  # e.g., 72.3%
```

**Step 4: Check Plateau**
```python
improvement = new_avg_score - baseline_avg_score
if improvement < threshold:  # e.g., <0.5%
    plateau_detected = True
    switch_to_A_cycle()
else:
    repeat_B_cycle()  # Model generates 4 NEW self-prompts
```

### Key Insights

**Self-Generation**: The model is introspecting and describing its own thinking
- Not human-designed personas
- Model discovers what works for itself
- Evolves each iteration

**Multi-Benchmark Testing**: Ensures general capability, not overfitting to one task
- MMLU: Knowledge
- GSM8K: Math reasoning
- HumanEval: Code generation
- MATH: Advanced math
- HellaSwag: Commonsense
- ARC: Scientific reasoning

**Plateau Detection**: When aggregate benchmark score stops improving (delta <0.5%)

---

## The Alternating Cycle

### Why Alternate A ↔ B?

**Co-Evolution**: Tool use and thinking patterns influence each other
- Better tool use → Model can express better thinking patterns
- Better thinking patterns → Model can conceive better tool strategies

**Example Evolution**:
```
Iteration 1:
  A1: "Always use tools" → SWE 35% → 37%
  B1: "Break into testable parts" → Avg 72.3%

Iteration 2:
  A2: Based on "testable parts" thinking, discovers "chain tools systematically" → SWE 40%
  B2: Based on better tool use, discovers "verify with tools before concluding" → Avg 74.1%

Iteration 3:
  A3: Based on "verify" mindset, discovers "use tools to check edge cases" → SWE 42%
  B3: Discovers "iterate with tool feedback" → Avg 75.8%

Iteration 4:
  A4: SWE 42% (plateau!)
  B4: Avg 75.8% (plateau!)
  → STOP: Both converged
```

### Stopping Condition

**When BOTH cycles plateau in the same iteration**:
```python
def check_convergence(a_plateau, b_plateau, consecutive_count):
    if a_plateau and b_plateau:
        consecutive_count += 1
    else:
        consecutive_count = 0

    if consecutive_count >= 2:  # Both plateaued for 2 consecutive cycles
        return True  # Converged
    return False
```

---

## Half-Baking Mechanics

### What is Half-Baking?

**Full Baking** (from research paper):
- Train for 3 epochs to fully embed prompt into weights
- Objective: `θ_u = argmin D_KL(P_θ(·|u) || P_θu(·))`
- Result: 100% prompt strength baked

**Half-Baking**:
- Train for 1.5 epochs (50% of full)
- Result: ~50% prompt strength baked
- Allows multiple layers to stack without overfitting

### Why Half-Baking for Phase 6?

**Composability**: Multiple half-bakes stack gracefully
```
Iteration 1: 0% → 50% (first half-bake)
Iteration 2: 50% → 75% (second half-bake adds 50% of remaining)
Iteration 3: 75% → 87.5% (third half-bake)
Iteration 4: 87.5% → 93.75% (fourth half-bake)
...
Convergence: Approaches 100% asymptotically
```

**Prevents Catastrophic Forgetting**: Gentle updates preserve previous bakes

### Implementation

```python
def half_bake_prompt(model, prompt, strength=0.5):
    """
    Half-bake a prompt into model weights using LoRA

    Args:
        model: Current model
        prompt: Text prompt to bake
        strength: 0.5 for half-baking, 1.0 for full

    Returns:
        Model with prompt half-baked into weights
    """
    # Setup LoRA for efficient fine-tuning
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_config)

    # Generate training data from prompted model
    prompted_outputs = generate_prompted_trajectories(model, prompt, n=1000)

    # Train unprompted model to match prompted outputs (KL divergence)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 3 if strength == 1.0 else 1.5  # Half epochs for half-baking

    for epoch in range(int(epochs)):
        for batch in prompted_outputs:
            loss = kl_divergence_loss(model(batch.input), batch.prompted_output)
            loss.backward()
            optimizer.step()

    # Merge LoRA weights into base model
    model = model.merge_and_unload()

    return model
```

**Time per half-bake**: ~2.5 minutes (LoRA-based, efficient)

---

## Example: Complete Phase 6 Run

### Initial State
- Model: Phase 5 coding agent
- SWE-Bench baseline: 35%
- Benchmark avg baseline: 70%

### Iteration 1

**A1-Cycle**:
1. Half-bake: "Use these tools: [search, calculator, code executor, file reader, database]"
2. Test SWE-Bench: 35%
3. Generate variants:
   - V0: "Always use tools before answering coding questions"
   - V1: "Break problems into tool-solvable steps"
   - V2: "Use tools to verify reasoning"
   - V3: "Chain tools together for complex tasks"
4. Test variants: V0=37%, V1=36%, V2=35%, V3=34%
5. Half-bake V0 (winner)
6. New SWE-Bench: 37% (+2%, continue)

**B1-Cycle**:
1. Model generates 4 self-prompts:
   - SP0: "I break problems into smallest testable parts"
   - SP1: "I prioritize clarity over speed"
   - SP2: "I think recursively from base cases"
   - SP3: "I validate assumptions first"
2. Test on benchmarks: SP0=72.3%, SP1=70.1%, SP2=71.8%, SP3=69.5%
3. Half-bake SP0 (winner)
4. New benchmark avg: 72.3% (+2.3%, continue)

---

### Iteration 2

**A2-Cycle**:
1. Generate 4 NEW tool variants (model is now "testable parts" focused):
   - V0: "Break code into functions, test each with tools"
   - V1: "Use tools to check edge cases systematically"
   - V2: "Chain tools: read file → test → debug → verify"
   - V3: "Use tools iteratively until tests pass"
2. Test variants: V0=38%, V1=40%, V2=39%, V3=37%
3. Half-bake V1 (winner: "check edge cases")
4. New SWE-Bench: 40% (+3%, continue)

**B2-Cycle**:
1. Model generates 4 NEW self-prompts (now has "edge case" mindset):
   - SP0: "I consider edge cases before implementing solutions"
   - SP1: "I verify each step with concrete examples"
   - SP2: "I iterate until edge cases pass"
   - SP3: "I think about failure modes proactively"
2. Test benchmarks: SP0=74.1%, SP1=73.2%, SP2=73.8%, SP3=72.9%
3. Half-bake SP0 (winner: "consider edge cases")
4. New benchmark avg: 74.1% (+1.8%, continue)

---

### Iteration 3

**A3-Cycle**:
1. Generate 4 variants (model now "edge case focused"):
   - V0: "For each edge case, use tools to validate handling"
   - V1: "Generate edge case test suite with tools"
   - V2: "Use tools to find boundary conditions"
   - V3: "Iteratively expand edge case coverage"
2. Test variants: V0=42%, V1=41%, V2=40%, V3=39%
3. Half-bake V0 (winner)
4. New SWE-Bench: 42% (+2%, continue)

**B3-Cycle**:
1. Model generates 4 NEW self-prompts:
   - SP0: "I systematically explore solution space with tool validation"
   - SP1: "I build confidence through iterative verification"
   - SP2: "I treat each problem as a hypothesis to test"
   - SP3: "I maintain rigor through tool-assisted checking"
2. Test benchmarks: SP0=75.8%, SP1=75.2%, SP2=75.0%, SP3=74.5%
3. Half-bake SP0 (winner: "systematic exploration")
4. New benchmark avg: 75.8% (+1.7%, continue)

---

### Iteration 4

**A4-Cycle**:
1. Generate 4 variants:
   - V0: "Systematically test hypotheses with tools"
   - V1: "Use tools to explore alternative solutions"
   - V2: "Verify solution space coverage with tools"
   - V3: "Iteratively refine with tool feedback"
2. Test variants: V0=42%, V1=42%, V2=41%, V3=41%
3. **Plateau detected**: Best score (42%) equals previous (42%)
4. Still half-bake V0 (tie, pick first)

**B4-Cycle**:
1. Model generates 4 NEW self-prompts:
   - SP0: "I explore systematically, verify rigorously"
   - SP1: "I balance creativity with tool-based validation"
   - SP2: "I iterate between hypothesis and verification"
   - SP3: "I treat problems as experiments to conduct"
2. Test benchmarks: SP0=75.8%, SP1=75.5%, SP2=75.7%, SP3=75.3%
3. **Plateau detected**: Best score (75.8%) equals previous (75.8%)
4. Still half-bake SP0

**Convergence**: Both A and B plateaued → STOP

---

### Final Model

**Baked Layers** (7 total half-bakes):
- Tool Layer 1: "Use tools" (initial)
- Tool Layer 2: "Always use tools" (A1 winner)
- Tool Layer 3: "Check edge cases" (A2 winner)
- Tool Layer 4: "Validate edge case handling" (A3 winner)
- Persona Layer 1: "Break into testable parts" (B1 winner)
- Persona Layer 2: "Consider edge cases" (B2 winner)
- Persona Layer 3: "Systematic exploration" (B3 winner)

**Performance**:
- SWE-Bench: 42% (up from 35%, +7% absolute, +20% relative)
- Benchmark avg: 75.8% (up from 70%, +5.8% absolute, +8.3% relative)

**Time**: 4 cycles × (A + B) × 2.5 min = 20 minutes baking + ~8 hours testing = **~8.5 hours total**

---

## W&B Integration

### Metrics Per A-Cycle

```python
wandb.log({
    f"tool_iteration_{i}/baseline_swe_bench": baseline_score,
    f"tool_iteration_{i}/variant_0_swe_bench": v0_score,
    f"tool_iteration_{i}/variant_1_swe_bench": v1_score,
    f"tool_iteration_{i}/variant_2_swe_bench": v2_score,
    f"tool_iteration_{i}/variant_3_swe_bench": v3_score,
    f"tool_iteration_{i}/winner_id": winner_id,
    f"tool_iteration_{i}/winner_score": winner_score,
    f"tool_iteration_{i}/winner_prompt": winner_prompt,
    f"tool_iteration_{i}/improvement": delta_score,
    f"tool_iteration_{i}/plateau": is_plateau,
    f"tool_iteration_{i}/baking_time_sec": baking_time,
})
```

### Metrics Per B-Cycle

```python
wandb.log({
    # Per self-prompt scores
    f"persona_iteration_{i}/self_prompt_0": sp0_text,
    f"persona_iteration_{i}/self_prompt_0_mmlu": sp0_mmlu,
    f"persona_iteration_{i}/self_prompt_0_gsm8k": sp0_gsm8k,
    f"persona_iteration_{i}/self_prompt_0_humaneval": sp0_humaneval,
    f"persona_iteration_{i}/self_prompt_0_avg": sp0_avg,
    # ... repeat for SP1, SP2, SP3

    # Winner info
    f"persona_iteration_{i}/winner_id": winner_id,
    f"persona_iteration_{i}/winner_avg": winner_avg,
    f"persona_iteration_{i}/winner_prompt": winner_prompt,
    f"persona_iteration_{i}/improvement": delta_avg,
    f"persona_iteration_{i}/plateau": is_plateau,
    f"persona_iteration_{i}/baking_time_sec": baking_time,
})
```

### Overall Phase 6 Metrics

```python
wandb.log({
    "phase6/total_iterations": total_iterations,
    "phase6/total_a_cycles": a_count,
    "phase6/total_b_cycles": b_count,
    "phase6/final_swe_bench": final_swe,
    "phase6/swe_bench_improvement": final_swe - initial_swe,
    "phase6/final_benchmark_avg": final_avg,
    "phase6/benchmark_avg_improvement": final_avg - initial_avg,
    "phase6/total_time_hours": total_time,
    "phase6/baking_time_hours": baking_time,
    "phase6/testing_time_hours": testing_time,
    "phase6/convergence_iterations": convergence_iter,
})
```

---

## Success Criteria

Phase 6 is successful if:

- ✅ SWE-Bench score improves by ≥5% absolute (e.g., 35% → 40%)
- ✅ Benchmark average improves by ≥3% absolute (e.g., 70% → 73%)
- ✅ Both A and B cycles converge (plateau detected)
- ✅ Total time <12 hours (baking + testing)
- ✅ Model generates meaningful self-prompts (human-readable, actionable)
- ✅ Final model has 5-10 layers of half-baked prompts
- ✅ No catastrophic forgetting (Phase 5 capabilities maintained)

---

## Expected Outputs to Phase 7

```python
{
    "success": True,
    "model_path": "./phase6_optimized_output",

    "performance": {
        "swe_bench_initial": 35.0,
        "swe_bench_final": 42.3,
        "swe_bench_improvement": 7.3,

        "benchmark_avg_initial": 70.0,
        "benchmark_avg_final": 75.8,
        "benchmark_avg_improvement": 5.8,
    },

    "baked_layers": {
        "tool_prompts": [
            "Use tools: search, calculator, code executor, file reader, database",
            "Always use tools before answering coding questions",
            "Use tools to check edge cases systematically",
            "For each edge case, use tools to validate handling"
        ],
        "self_prompts": [
            "I break problems into smallest testable parts",
            "I consider edge cases before implementing solutions",
            "I systematically explore solution space with tool validation"
        ]
    },

    "convergence": {
        "total_iterations": 4,
        "a_cycles": 4,
        "b_cycles": 4,
        "final_a_plateau": True,
        "final_b_plateau": True
    },

    "timing": {
        "total_hours": 8.5,
        "baking_hours": 0.3,
        "testing_hours": 8.2
    }
}
```

---

## Key Differences from Phase 5 Baking

### Phase 5 Baking (Eudaimonia System)
- **Purpose**: Moral alignment and identity
- **Prompts**: Pre-defined (4 rules, 3 archetypes, OODA loop)
- **Frequency**: After each training level (10× reinforcement)
- **Method**: Full baking (3 epochs)
- **Time**: ~15 min per level × 10 levels = 150 min

### Phase 6 Baking (Tool & Persona Optimization)
- **Purpose**: Performance optimization and self-discovery
- **Prompts**: Evolved through testing (empirical selection)
- **Frequency**: Iterative until convergence (3-5 cycles)
- **Method**: Half-baking (1.5 epochs)
- **Time**: ~8.5 hours (mostly testing, 20 min baking)

**Complementary**: Phase 5 bakes "who you should be" (morals), Phase 6 bakes "how you work best" (performance)

---

## Research Paper Integration

### "Prompt Baking" (arXiv:2409.13697v1)

**What We Use**:
- KL divergence minimization objective: `θ_u = argmin D_KL(P_θ(·|u) || P_θu(·))`
- Half-baking technique (stop at 50% epochs)
- Sequential baking composability: `θ_u1u2 = B(B(θ, u1), u2)`
- LoRA for efficient fine-tuning

**What We Add** (Phase 6 innovation):
- Iterative refinement (generate → test → bake → repeat)
- Self-guided prompt generation (model introspects)
- Alternating optimization (tool vs persona cycles)
- Empirical selection (benchmark-driven, not human judgment)
- Convergence detection (plateau-based stopping)

---

## Timeline Estimate

**Per Cycle** (A + B):
- A-Cycle:
  - Half-bake: 2.5 min
  - Test 4 variants on SWE-Bench: 4 × 10 min = 40 min
  - Half-bake winner: 2.5 min
  - Subtotal: 45 min

- B-Cycle:
  - Generate 4 self-prompts: 5 min
  - Test 4 self-prompts on 6 benchmarks: 4 × 30 min = 120 min
  - Half-bake winner: 2.5 min
  - Subtotal: 127.5 min

**Total per cycle**: ~172.5 min (~2.9 hours)

**Expected cycles to convergence**: 3-5 cycles

**Total Phase 6 time**: 8.7-14.5 hours (typically ~10 hours)

---

**Next Phase**: [Phase 7: Generic Edge Deployment](../phase7/LOGICAL_UNDERSTANDING.md)

**Related Documents**:
- [PHASE6_ITERATIVE_BAKING_SYSTEM.md](./PHASE6_ITERATIVE_BAKING_SYSTEM.md) - Deep dive into A/B cycles
- [PHASE6_V2_COMPLETE_SPECIFICATION.md](./PHASE6_V2_COMPLETE_SPECIFICATION.md) - Implementation spec
- [graphviz/phase6-iterative-cycle.dot](./graphviz/phase6-iterative-cycle.dot) - Visual flowchart
- [Prompt Baking Paper](../../v1-reference/research-papers/Prompt_Baking.pdf) - Original research
