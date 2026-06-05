# Phase 6 Integration Guide

**Target**: Integrate 4 new modules into existing `baking_engine.py`
**Timeline**: 4 weeks (1 week per module)
**Difficulty**: LOW - All modules are production-ready

---

## Overview

This guide provides step-by-step instructions for integrating the four newly audited Phase 6 modules:

1. **Week 1**: Prompt Pursuit → A-Cycle optimization
2. **Week 2**: Cross-Task Validation → Post-baking verification
3. **Week 3**: Drift Meter → B-Cycle persona consistency
4. **Week 4**: Monte Carlo KL → Baking quality assessment

---

## Prerequisites

✅ All modules passed functionality audit (100% pass rate)
✅ `baking_engine.py` is functional
✅ PyTorch environment set up
✅ GPU available (GTX 1660+ recommended)

---

## Week 1: Prompt Pursuit Integration

### Objective
Add iterative re-baking to A-Cycle tool optimization for 15-40% accuracy gains.

### File to Modify
`src/phase6_baking/a_cycle_tool.py`

### Step 1: Import Prompt Pursuit

```python
# Add to top of a_cycle_tool.py
from .prompt_pursuit import PromptPursuitOptimizer, PursuitConfig
```

### Step 2: Add Config Parameter

```python
class ACycleOptimizer:
    def __init__(
        self,
        tool_prompts: List[str],
        lora_r: int = 16,
        lora_alpha: int = 32,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        # NEW: Add pursuit config
        use_prompt_pursuit: bool = False,
        pursuit_rounds: int = 3,
    ):
        self.tool_prompts = tool_prompts
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # NEW: Initialize pursuit optimizer
        self.use_prompt_pursuit = use_prompt_pursuit
        if use_prompt_pursuit:
            pursuit_config = PursuitConfig(
                pursuit_rounds=pursuit_rounds,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
            )
            self.pursuit_optimizer = PromptPursuitOptimizer(pursuit_config)
```

### Step 3: Modify `optimize()` Method

```python
def optimize(
    self,
    model: nn.Module,
    tokenizer: Any,
    evaluator: Any = None,
) -> Tuple[nn.Module, float]:
    """
    Optimize tool use via prompt baking.

    NEW: Optionally uses prompt pursuit for iterative re-baking.
    """
    print(f"A-Cycle: Tool optimization")

    # Select prompt
    prompt = self.tool_prompts[0]  # Or select best prompt

    # NEW: Use prompt pursuit if enabled
    if self.use_prompt_pursuit:
        print(f"  Using prompt pursuit (iterative re-baking)")
        result = self.pursuit_optimizer.pursue(
            model=model,
            prompt=prompt,
            tokenizer=tokenizer,
            evaluator=evaluator or self._default_evaluator,
        )

        if result.success:
            print(f"  Pursuit rounds: {result.rounds_completed}")
            print(f"  Total improvement: {result.improvements_per_round}")
            return result.final_model, result.scores_per_round[-1]
        else:
            print(f"  Pursuit failed: {result.error}, falling back to single bake")
            # Fall through to single bake

    # Original single-bake implementation
    baked_model = self._bake_single_prompt(model, prompt, tokenizer)
    score = evaluator(baked_model) if evaluator else 0.0

    return baked_model, score
```

### Step 4: Update BakingConfig

```python
# In baking_engine.py
@dataclass
class BakingConfig:
    # ... existing fields ...

    # NEW: Prompt pursuit settings
    use_prompt_pursuit: bool = False  # Enable iterative re-baking
    pursuit_rounds: int = 3  # Number of pursuit iterations
```

### Step 5: Pass Config to ACycleOptimizer

```python
# In BakingEngine.run()
a_optimizer = ACycleOptimizer(
    tool_prompts=self.config.tool_prompts,
    lora_r=self.config.lora_r,
    lora_alpha=self.config.lora_alpha,
    num_epochs=self.config.baking_epochs,
    learning_rate=self.config.learning_rate,
    # NEW: Pass pursuit config
    use_prompt_pursuit=self.config.use_prompt_pursuit,
    pursuit_rounds=self.config.pursuit_rounds,
)
```

### Step 6: Test

```python
# Test script
from phase6_baking.baking_engine import BakingEngine, BakingConfig

config = BakingConfig(
    use_prompt_pursuit=True,  # Enable pursuit
    pursuit_rounds=3,
    a_cycle_iterations=3,
)

engine = BakingEngine(config)
result = engine.run(model, tokenizer, tool_evaluator, persona_evaluator)

print(f"Final tool score: {result.final_tool_score:.3f}")
```

**Expected**: 15-40% improvement over single baking

---

## Week 2: Cross-Task Validation Integration

### Objective
Validate that baking doesn't cause catastrophic forgetting (max 3.4% degradation).

### File to Modify
`src/phase6_baking/baking_engine.py`

### Step 1: Import Validator

```python
# Add to top of baking_engine.py
from .validation import CrossTaskValidator, ValidationConfig, create_standard_benchmark_suite
```

### Step 2: Add Validation to Config

```python
@dataclass
class BakingConfig:
    # ... existing fields ...

    # NEW: Cross-task validation settings
    enable_cross_task_validation: bool = False
    validation_tasks: Optional[Dict[str, Callable]] = None
    max_degradation_threshold: float = 0.034  # 3.4% from paper
```

### Step 3: Add Validation Step

```python
class BakingEngine:
    def run(
        self,
        model: nn.Module,
        tokenizer: Any,
        tool_evaluator: Any = None,
        persona_evaluator: Any = None,
    ) -> BakingResult:
        # ... existing code ...

        # A/B cycle optimization loop
        while total_iterations < self.config.max_total_iterations:
            # ... existing cycle logic ...
            pass

        # NEW: Post-baking validation
        if self.config.enable_cross_task_validation:
            print("\n--- Cross-Task Validation ---")
            validator = CrossTaskValidator(
                ValidationConfig(max_acceptable_degradation=self.config.max_degradation_threshold)
            )

            # Use provided tasks or create standard suite
            tasks = self.config.validation_tasks or create_standard_benchmark_suite()

            val_result = validator.validate_cross_task_forgetting(
                base_model=model,  # Original model
                baked_model=current_model,  # Final baked model
                baked_task="tool_and_persona",
                all_tasks=tasks,
                tokenizer=tokenizer,
            )

            # Log validation results
            self._log_wandb({
                "validation/max_degradation": val_result.max_degradation * 100,
                "validation/avg_degradation": val_result.avg_degradation * 100,
                "validation/tasks_passed": val_result.tasks_passed,
                "validation/tasks_failed": val_result.tasks_failed,
                "validation/success": val_result.success,
            })

            # Store in metrics
            self.metrics["validation_result"] = val_result

            print(f"  Max degradation: {val_result.max_degradation*100:.2f}%")
            print(f"  Tasks passed: {val_result.tasks_passed}/{val_result.tasks_passed + val_result.tasks_failed}")

            if not val_result.success:
                print(f"  WARNING: Catastrophic forgetting detected!")
                # Optionally rollback to previous model

        # Return final result
        return BakingResult(...)
```

### Step 4: Test

```python
# Define custom tasks
tasks = {
    "swe_bench": lambda m: evaluate_swe_bench(m),
    "math_qa": lambda m: evaluate_math(m),
    "commonsense": lambda m: evaluate_commonsense(m),
}

config = BakingConfig(
    enable_cross_task_validation=True,
    validation_tasks=tasks,
    max_degradation_threshold=0.034,  # 3.4%
)

engine = BakingEngine(config)
result = engine.run(model, tokenizer, tool_evaluator, persona_evaluator)

# Check validation
val_result = result.metrics["validation_result"]
assert val_result.success, "Validation failed!"
print(f"All tasks passed with <3.4% degradation")
```

**Expected**: Validation confirms <3.4% degradation on all tasks

---

## Week 3: Drift Meter Integration

### Objective
Measure persona consistency over 30+ turns to verify baking maintains persona.

### File to Modify
`src/phase6_baking/b_cycle_persona.py`

### Step 1: Import Drift Meter

```python
# Add to top of b_cycle_persona.py
from .drift_meter import PersonaDriftMeter, DriftConfig
```

### Step 2: Add to BCycleOptimizer

```python
class BCycleOptimizer:
    def __init__(
        self,
        persona_prompts: List[str],
        lora_r: int = 16,
        lora_alpha: int = 32,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        # NEW: Add drift measurement
        measure_drift: bool = False,
        drift_num_turns: int = 30,
    ):
        self.persona_prompts = persona_prompts
        # ... existing initialization ...

        # NEW: Initialize drift meter
        self.measure_drift = measure_drift
        if measure_drift:
            drift_config = DriftConfig(num_turns=drift_num_turns)
            self.drift_meter = PersonaDriftMeter(drift_config)
```

### Step 3: Add Drift Measurement

```python
def optimize(
    self,
    model: nn.Module,
    tokenizer: Any,
    evaluator: Any = None,
) -> Tuple[nn.Module, float]:
    """
    Optimize persona via prompt baking.

    NEW: Optionally measures persona drift over multi-turn conversations.
    """
    print(f"B-Cycle: Persona optimization")

    # Bake persona prompt
    persona = self.persona_prompts[0]
    baked_model = self._bake_persona(model, persona, tokenizer)

    # NEW: Measure drift if enabled
    if self.measure_drift:
        print(f"  Measuring persona drift over {self.drift_meter.config.num_turns} turns...")

        # Compare baked vs prompted model
        prompted_model = self._create_prompted_model(model, persona)

        comparison = self.drift_meter.compare_baked_vs_prompted(
            baked_model=baked_model,
            prompted_model=prompted_model,
            persona=persona,
            keywords=["careful", "thorough", "verify"],  # Extract from persona
            tokenizer=tokenizer,
            num_turns=self.drift_meter.config.num_turns,
        )

        print(f"  Baked avg drift: {comparison['baked_avg_drift']:.4f}")
        print(f"  Prompted avg drift: {comparison['prompted_avg_drift']:.4f}")
        print(f"  Drift reduction: {comparison['drift_reduction_percent']:.1f}%")

        # Store drift metrics
        self._drift_metrics = comparison

    # Evaluate
    score = evaluator(baked_model) if evaluator else 0.0

    return baked_model, score
```

### Step 4: Update BakingConfig

```python
@dataclass
class BakingConfig:
    # ... existing fields ...

    # NEW: Drift measurement settings
    measure_persona_drift: bool = False
    drift_num_turns: int = 30  # Number of conversation turns
```

### Step 5: Test

```python
config = BakingConfig(
    measure_persona_drift=True,
    drift_num_turns=30,
    b_cycle_iterations=3,
)

engine = BakingEngine(config)
result = engine.run(model, tokenizer, tool_evaluator, persona_evaluator)

# Check drift reduction
# Expect: Baked model has <5% drift, prompted model has 15-30% drift
```

**Expected**: Baked model maintains persona 6x better than prompted model

---

## Week 4: Monte Carlo KL Integration

### Objective
Replace calibration-sample KL estimation with trajectory-based MC-KL for more accurate baking quality assessment.

### File to Modify
`src/phase6_baking/half_baking.py` (or wherever KL is computed)

### Step 1: Import MC-KL

```python
# Add to relevant file
from .monte_carlo_kl import monte_carlo_kl_from_trajectories, compute_baking_quality_score
```

### Step 2: Replace Calibration-Sample KL

```python
# OLD: Calibration-sample based KL
def compute_kl_divergence(prompted_model, baked_model, calibration_samples):
    """Old method using calibration samples."""
    kl_values = []
    for sample in calibration_samples:
        prompted_logits = prompted_model(sample)
        baked_logits = baked_model(sample)
        kl = kl_divergence_loss(prompted_logits, baked_logits)
        kl_values.append(kl)
    return sum(kl_values) / len(kl_values)

# NEW: Trajectory-based MC-KL
def compute_kl_divergence(prompted_model, baked_model, tokenizer, num_trajectories=100):
    """New method using Monte Carlo trajectory sampling."""
    kl = monte_carlo_kl_from_trajectories(
        model_prompted=prompted_model,
        model_baked=baked_model,
        tokenizer=tokenizer,
        num_trajectories=num_trajectories,
        seq_length=256,
    )
    return kl
```

### Step 3: Add Quality Assessment

```python
def assess_baking_quality(prompted_model, baked_model, tokenizer):
    """Comprehensive baking quality assessment."""
    quality = compute_baking_quality_score(
        model_prompted=prompted_model,
        model_baked=baked_model,
        tokenizer=tokenizer,
        num_trajectories=50,  # Faster for quality check
        seq_length=128,
    )

    print(f"Baking Quality:")
    print(f"  KL divergence: {quality['kl_divergence']:.4f}")
    print(f"  Quality score: {quality['quality_score']:.3f}")
    print(f"  Confidence: {quality['confidence']:.3f}")

    # Good baking: KL < 0.1
    if quality['kl_divergence'] < 0.1:
        print(f"  Status: Good baking")
    else:
        print(f"  Status: May need more baking epochs")

    return quality
```

### Step 4: Update BakingConfig

```python
@dataclass
class BakingConfig:
    # ... existing fields ...

    # NEW: MC-KL settings
    use_mc_kl: bool = True  # Use trajectory-based KL
    mc_kl_trajectories: int = 100  # Number of trajectories
```

### Step 5: Test

```python
# Compare old vs new KL estimation
from phase6_baking.monte_carlo_kl import monte_carlo_kl_from_trajectories
from phase6_baking.loss_functions import kl_divergence_loss

# Old method
kl_old = compute_kl_old(prompted, baked, calibration_samples)

# New method
kl_new = monte_carlo_kl_from_trajectories(prompted, baked, tokenizer, num_trajectories=100)

print(f"Old KL: {kl_old:.4f}")
print(f"New KL: {kl_new:.4f}")
print(f"Difference: {abs(kl_new - kl_old):.4f}")
```

**Expected**: MC-KL is more accurate, typically lower variance

---

## Complete Integration Example

### Final `BakingConfig` with All Features

```python
@dataclass
class BakingConfig:
    # A-Cycle (Tool) settings
    a_cycle_iterations: int = 5
    tool_prompts: List[str] = field(default_factory=lambda: [...])

    # B-Cycle (Persona) settings
    b_cycle_iterations: int = 5
    persona_prompts: List[str] = field(default_factory=lambda: [...])

    # Half-baking settings
    half_bake_strength: float = 0.5
    baking_epochs: int = 3
    learning_rate: float = 5e-5

    # Convergence settings
    plateau_window: int = 3
    plateau_threshold: float = 0.01
    max_total_iterations: int = 20

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32

    # NEW FEATURES
    # Week 1: Prompt Pursuit
    use_prompt_pursuit: bool = False
    pursuit_rounds: int = 3

    # Week 2: Cross-Task Validation
    enable_cross_task_validation: bool = False
    validation_tasks: Optional[Dict[str, Callable]] = None
    max_degradation_threshold: float = 0.034

    # Week 3: Drift Measurement
    measure_persona_drift: bool = False
    drift_num_turns: int = 30

    # Week 4: MC-KL
    use_mc_kl: bool = True
    mc_kl_trajectories: int = 100
```

### Full-Featured Usage

```python
from phase6_baking.baking_engine import BakingEngine, BakingConfig

# Configure with all features
config = BakingConfig(
    # Core settings
    a_cycle_iterations=5,
    b_cycle_iterations=5,
    max_total_iterations=20,

    # Enable all new features
    use_prompt_pursuit=True,
    pursuit_rounds=3,
    enable_cross_task_validation=True,
    measure_persona_drift=True,
    drift_num_turns=30,
    use_mc_kl=True,
    mc_kl_trajectories=100,
)

# Define validation tasks
validation_tasks = {
    "swe_bench": lambda m: evaluate_swe_bench(m),
    "math_qa": lambda m: evaluate_math(m),
    "commonsense": lambda m: evaluate_commonsense(m),
}
config.validation_tasks = validation_tasks

# Run full pipeline
engine = BakingEngine(config, use_wandb=True)
result = engine.run(
    model=phase5_model,
    tokenizer=tokenizer,
    tool_evaluator=swe_bench_evaluator,
    persona_evaluator=persona_evaluator,
)

# Check results
print(f"\nPhase 6 Complete:")
print(f"  Tool score: {result.final_tool_score:.3f}")
print(f"  Persona score: {result.final_persona_score:.3f}")
print(f"  Pursuit rounds: {result.metrics.get('pursuit_rounds', 0)}")
print(f"  Max degradation: {result.metrics.get('validation_result').max_degradation*100:.2f}%")
print(f"  Persona drift: {result.metrics.get('drift_reduction', 0):.1f}%")
```

---

## Testing Checklist

### Week 1: Prompt Pursuit
- [ ] Pursuit optimizer initializes
- [ ] 3 rounds complete successfully
- [ ] Improvement per round logged
- [ ] Convergence detection works
- [ ] Falls back to single bake on error
- [ ] W&B logs pursuit metrics

### Week 2: Cross-Task Validation
- [ ] Validator initializes with tasks
- [ ] All tasks evaluated (base + baked)
- [ ] Degradation calculated correctly
- [ ] <3.4% threshold enforced
- [ ] Pass/fail determination works
- [ ] W&B logs validation results

### Week 3: Drift Measurement
- [ ] Drift meter initializes
- [ ] 30-turn conversation generates
- [ ] Drift scores computed per turn
- [ ] Baked vs prompted comparison works
- [ ] Keyword tracking functional
- [ ] W&B logs drift metrics

### Week 4: MC-KL
- [ ] Trajectory sampling works
- [ ] KL divergence computed
- [ ] Quality score calculated
- [ ] Lower KL than calibration-sample method
- [ ] Handles device placement correctly
- [ ] W&B logs KL metrics

---

## Performance Tuning

### Optimization Tips

1. **Reduce Trajectories**: Start with 50 instead of 100 for MC-KL
2. **Shorter Conversations**: Use 10 turns for drift during development
3. **Fewer Validation Tasks**: Test with 2-3 tasks, expand to 5+ in production
4. **Lower Pursuit Rounds**: Use 2 rounds instead of 3 for faster iteration

### Expected Runtimes (GTX 1660, 25M params)

| Configuration | Time/Iteration | Total Time |
|---------------|----------------|------------|
| Baseline (no new features) | 5 min | 50 min (10 iter) |
| + Prompt Pursuit | 15 min | 150 min |
| + Validation | 25 min | 250 min |
| + Drift Measurement | 10 min | 100 min |
| + MC-KL | 5 min | 50 min |
| **Full Integration** | **55 min** | **550 min (9 hrs)** |

**Optimization**: Run validation/drift every 5 iterations instead of every iteration → 3-4 hours total

---

## Troubleshooting

### Issue: "Trajectories fail to generate"
**Solution**: Check tokenizer has `bos_token_id` and `pad_token_id`

### Issue: "Drift meter runs out of memory"
**Solution**: Reduce `max_tokens_per_turn` from 128 to 64

### Issue: "Validation takes too long"
**Solution**: Use fewer validation samples (50 instead of 100)

### Issue: "MC-KL divergence is NaN"
**Solution**: Check for division by zero, increase `epsilon` to 1e-6

---

## W&B Metrics Added

### Prompt Pursuit
- `a_cycle/pursuit_rounds`
- `a_cycle/pursuit_improvement`
- `b_cycle/pursuit_rounds`

### Cross-Task Validation
- `validation/max_degradation`
- `validation/avg_degradation`
- `validation/tasks_passed`
- `validation/tasks_failed`

### Drift Measurement
- `b_cycle/baked_drift`
- `b_cycle/prompted_drift`
- `b_cycle/drift_reduction_percent`

### MC-KL
- `baking/mc_kl_divergence`
- `baking/quality_score`
- `baking/confidence`

**Total New Metrics**: 13

---

## Success Criteria

✅ Prompt pursuit completes 3 rounds
✅ Final tool score improves by 15-40%
✅ Cross-task validation passes (<3.4% degradation)
✅ Baked persona drift <5%, prompted >15%
✅ MC-KL divergence <0.1 (good baking)
✅ All W&B metrics logged
✅ No memory errors
✅ Total runtime <4 hours (with optimizations)

---

## Conclusion

This 4-week integration plan adds:
- ✅ 15-40% accuracy gains (prompt pursuit)
- ✅ Catastrophic forgetting protection (validation)
- ✅ 6x better persona consistency (drift meter)
- ✅ More accurate baking quality (MC-KL)

**Total Code Changes**: ~400 lines across 4 files
**Total New Metrics**: 13 W&B metrics
**Estimated Effort**: 20-30 hours (5-8 hours per week)

**Next Steps**: Begin Week 1 integration (prompt pursuit)
