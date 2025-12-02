# Phase 2: EvoMerge - Logical Understanding

**Version**: 2.0 (V2 Rebuild - CORRECTED)
**Date**: 2025-10-15
**Phase Purpose**: Evolve 3 models from Phase 1 into single optimized model via 50 generations

---

## 🎯 What This Phase Does (HYPER-CLARITY)

Phase 2 takes **3 models from Phase 1** (each ~25M params) and evolves them through **genetic algorithms** over **50 generations** to create **1 optimized merged model** that combines the best characteristics of all three.

**Analogy**: Breeding dogs over 50 generations to create offspring with:
- Greyhound speed (Model 1's reasoning)
- Retriever memory (Model 2's memory integration)
- Border Collie intelligence (Model 3's adaptive computation)

**Expected Result**: 1 evolved model with **23.5% better fitness** than Phase 1 models in ~90 minutes on GTX 1660.

---

## 🧬 The 6 Merge Techniques (3 Mutually Exclusive Pairs)

**Critical Insight**: Techniques are **paired** and applied **sequentially in 3 stages**.

### **Pair 1: Interpolation Methods** (Stage 1)

#### **Linear (0)**: Simple weighted average
```python
merged = 0.33*model1 + 0.33*model2 + 0.33*model3
```
- **When**: Baseline, always works
- **Benefit**: Fast, predictable
- **Risk**: May average out specialized features

#### **SLERP (1)**: Spherical Linear Interpolation
```python
# Interpolate along hypersphere curve
theta = arccos(dot(w1, w2))
merged = (sin((1-t)*theta)/sin(theta)) * w1 + (sin(t*theta)/sin(theta)) * w2
```
- **When**: Models are similar (small angle θ)
- **Benefit**: Preserves parameter magnitude better than linear
- **Risk**: Undefined if θ=0 (identical models)

---

### **Pair 2: Task Arithmetic** (Stage 2)

#### **DARE (0)**: Drop And REscale
```python
# 1. Compute delta (fine-tuned - base)
delta = model_finetuned - model_base

# 2. Drop 90% of delta parameters randomly
mask = bernoulli(p=0.1)  # Keep 10%
sparse_delta = delta * mask

# 3. Rescale by 10× to compensate
rescaled_delta = sparse_delta * 10

# 4. Add back to base
result = model_base + rescaled_delta
```
- **When**: Want sparse, efficient updates
- **Benefit**: Reduces interference between tasks
- **Risk**: May drop important parameters

#### **TIES (1)**: TrIm, Elect Sign, Merge
```python
# 1. TRIM: Keep only top 20% magnitude parameters
important_params = topk(abs(delta), k=0.2)

# 2. ELECT: Vote on sign (+/-) for each parameter
elected_sign = sign(mean([delta1, delta2, delta3]))

# 3. MERGE: Average only params with matching elected sign
merged = mean([delta_i for delta_i if sign(delta_i) == elected_sign])
```
- **When**: Models have conflicting updates
- **Benefit**: Resolves sign conflicts intelligently
- **Risk**: May over-filter if models disagree heavily

---

### **Pair 3: Selection Methods** (Stage 3)

#### **FrankenMerge (0)**: Layer-wise selection
```python
merged = Model()
for i, layer in enumerate(range(num_layers)):
    # Pick best layer from each model
    if fitness(model1.layers[i]) > fitness(model2.layers[i]):
        merged.layers[i] = model1.layers[i]
    else:
        merged.layers[i] = model2.layers[i]
```
- **When**: Layers have independent performance
- **Benefit**: Mix-and-match best layers
- **Risk**: Layer compatibility issues (dimension mismatch)

#### **DFS (1)**: Deep Feature Selection
```python
# 1. Calculate importance (inverse variance)
importance = 1 / variance(param_across_models)

# 2. Weight parameters by importance during merge
merged_param = sum(importance[i] * param[i] for i in models) / sum(importance)
```
- **When**: Want to preserve stable features
- **Benefit**: Important (stable) features get higher weight
- **Risk**: May ignore beneficial variation

---

## 📊 Binary Combination Strategy (8 Initial Models)

**Generation 0**: Create **all 8 combinations** of merge techniques (3-bit binary):

```
Bit 0 = Interpolation    (0=Linear, 1=SLERP)
Bit 1 = Task Arithmetic  (0=DARE, 1=TIES)
Bit 2 = Selection        (0=FrankenMerge, 1=DFS)

Binary 000 → Linear + DARE + FrankenMerge
Binary 001 → Linear + DARE + DFS
Binary 010 → Linear + TIES + FrankenMerge
Binary 011 → Linear + TIES + DFS
Binary 100 → SLERP + DARE + FrankenMerge
Binary 101 → SLERP + DARE + DFS
Binary 110 → SLERP + TIES + FrankenMerge
Binary 111 → SLERP + TIES + DFS
```

**Sequential Application**:
```
3 Phase 1 models
  ↓
Stage 1: Interpolation (Linear OR SLERP)
  → Combines 3 models into 1
  ↓
Stage 2: Task Arithmetic (DARE OR TIES)
  → Refines merged model
  ↓
Stage 3: Selection (FrankenMerge OR DFS)
  → Final feature selection
  ↓
Result: 1 merged model
```

**Example (Binary 110)**:
```python
# Stage 1: SLERP (bit 0 = 1)
intermediate1 = slerp_merge(model1, model2, t=0.5)
intermediate2 = slerp_merge(intermediate1, model3, t=0.33)

# Stage 2: TIES (bit 1 = 1)
intermediate3 = ties_merge(intermediate2)

# Stage 3: FrankenMerge (bit 2 = 0)
final_model = frankenmerge(intermediate3)
```

**Population at Generation 0**: 8 models (one for each binary combination)

---

## 🔄 The Evolution Loop (50 Generations)

### **Step 1: Fitness Evaluation**

Evaluate all 8 models on **composite fitness**:

```python
composite_fitness = (
    0.4 * (1 / perplexity) +        # Lower perplexity = better (40%)
    0.3 * accuracy +                 # Higher accuracy = better (30%)
    0.2 * (1 / inference_time) +    # Faster = better (20%)
    0.1 * (1 / memory_usage)        # Less memory = better (10%)
)
```

**Weights reflect priorities**:
- **40%** Language modeling quality (perplexity)
- **30%** Task performance (accuracy)
- **20%** Speed (inference time)
- **10%** Efficiency (memory usage)

Rank models 1-8 by fitness score.

---

### **Step 2: Selection & Reproduction**

#### **Elite Preservation** (Top 2 → 6 children)
```python
# Take best 2 models
elite1, elite2 = population[:2]  # Sorted by fitness

# Mutate each elite 3 times
elite1_children = [mutate(elite1, sigma=0.01) for _ in range(3)]
elite2_children = [mutate(elite2, sigma=0.01) for _ in range(3)]

elite_children = elite1_children + elite2_children  # 6 total
```

**Mutation**:
- Small weight perturbation (σ=0.01)
- Applied to 1% of weights (mutation_rate=0.01)
- Adds slight variation to winning strategies

#### **Loser Merging** (Bottom 6 → 2 children)
```python
# Take worst 6 models
losers = population[-6:]  # Bottom 6 by fitness

# Split into 2 groups of 3
group1 = losers[0:3]
group2 = losers[3:6]

# Merge each group using one of the 8 combo strategies
combo1 = random.choice(range(8))  # Binary 000-111
combo2 = random.choice(range(8))

loser_child1 = apply_merge_combo(group1, combo1)
loser_child2 = apply_merge_combo(group2, combo2)

loser_children = [loser_child1, loser_child2]  # 2 total
```

**New Population**:
```python
next_generation = elite_children + loser_children  # 6 + 2 = 8 models
```

---

### **Step 3: Fitness Re-Evaluation**

```python
# Evaluate fitness of all 8 new models
fitness_scores = [evaluate_fitness(model) for model in next_generation]

# Update best-so-far model
best_so_far = max(next_generation, key=evaluate_fitness)
if evaluate_fitness(best_so_far) > evaluate_fitness(current_champion):
    current_champion = best_so_far  # New champion
```

---

### **Step 4: Repeat**

Continue for **50 generations** (or until diminishing returns: improvement < 0.1% for 5 consecutive generations).

---

## 💾 Storage Strategy (Memory Efficient)

**At any point in time, only save**:

```python
stored_models = {
    "original_phase1": [model1, model2, model3],  # 3 models (archival)
    "current_champion": best_model,                # 1 model (best so far)
    "current_generation": population,              # 8 models (working population)
}
# Total: 12 models max in memory
```

**After 50 generations**:
```python
# Keep only the champion
final_output = best_model

# Delete all intermediate models
delete(original_phase1)
delete(all_generation_populations)

# Total: 1 model (passed to Phase 3)
```

---

## 📈 Why This Works

1. **Exploration via Binary Combos**: 8 different merge strategies explore solution space
2. **Exploitation via Elitism**: Top 2 models always preserved (no regression)
3. **Refinement via Mutation**: Small perturbations refine winning strategies
4. **Diversity via Loser Merging**: Merging failures creates new diversity
5. **Efficiency**: Only 12 models in memory (fits 6GB VRAM)

**Expected Dynamics**:
- **Generations 1-15**: Exploration (fitness variance high, all 8 combos tried)
- **Generations 16-35**: Exploitation (best combos dominate, fitness climbs steeply)
- **Generations 36-50**: Refinement (diminishing returns, small improvements)

---

## 🔗 Phase Handoff (Integration with Phase 1)

### **Input from Phase 1**:
```python
# Phase 1 (Cognate) produces 3 models
cognate_output = {
    "models": ["model_id_1", "model_id_2", "model_id_3"],
    "configs": [config1, config2, config3],
    "metrics": {
        "model1": {"perplexity": 15.2, "accuracy": 0.12},
        "model2": {"perplexity": 14.8, "accuracy": 0.13},
        "model3": {"perplexity": 16.1, "accuracy": 0.11},
    }
}
```

### **Input Validation**:
```python
# Phase 2 receives and validates
evomerge = EvoMerge()
evomerge.set_input_models(cognate_output["models"])

# Validates:
✓ Must be exactly 3 models
✓ Must be TRM × Titans-MAG architecture (NEW in V2)
✓ Must have ~25M parameters (±5%)
✓ Models must load without errors
```

### **Run Evolution**:
```python
result = await evomerge.evolve(
    generations=50,
    population_size=8,
    mutation_rate=0.01,
    fitness_weights={
        "perplexity": 0.4,
        "accuracy": 0.3,
        "speed": 0.2,
        "memory": 0.1,
    }
)
```

### **Output to Phase 3**:
```python
evomerge_output = {
    "model": "best_model.pt",
    "fitness": 0.185,  # 23.5% better than 0.15 initial avg
    "improvement": 0.235,  # 23.5%
    "evolution_log": {
        "generations": 50,
        "best_fitness_per_gen": [0.15, 0.152, ..., 0.185],
        "diversity_per_gen": [0.45, 0.42, ..., 0.35],
    },
    "merge_combo_usage": {
        "000": 12, "001": 8, "010": 5, "011": 3,
        "100": 15, "101": 4, "110": 2, "111": 1,
    },
}
```

---

## 🛠️ Technical Flow (Complete Implementation)

### **Generation 0: Initialization**

```python
# Load Phase 1 models
model1, model2, model3 = load_phase1_models()

# Create 8 initial models using binary combos
population = []
for combo_id in range(8):  # 000 to 111
    bit0 = (combo_id >> 0) & 1  # Interpolation
    bit1 = (combo_id >> 1) & 1  # Task arithmetic
    bit2 = (combo_id >> 2) & 1  # Selection

    # Stage 1: Interpolation
    if bit0 == 0:
        stage1 = linear_merge(model1, model2, model3)
    else:
        stage1 = slerp_merge(model1, model2, model3)

    # Stage 2: Task arithmetic
    if bit1 == 0:
        stage2 = dare_merge(stage1)
    else:
        stage2 = ties_merge(stage1)

    # Stage 3: Selection
    if bit2 == 0:
        stage3 = frankenmerge(stage2)
    else:
        stage3 = dfs_merge(stage2)

    population.append(stage3)

# Population now has 8 models
```

---

### **Generations 1-50: Evolution**

```python
best_champion = None
champion_fitness = -inf

for generation in range(1, 51):
    # Step 1: Fitness evaluation
    fitness_scores = [evaluate_fitness(model) for model in population]

    # Sort population by fitness (descending)
    sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    population = [model for model, _ in sorted_pop]

    # Step 2: Elite preservation (top 2 → 6 children)
    elite1, elite2 = population[0], population[1]
    elite_children = []
    for elite in [elite1, elite2]:
        for _ in range(3):
            child = mutate(elite, sigma=0.01, rate=0.01)
            elite_children.append(child)

    # Step 3: Loser merging (bottom 6 → 2 children)
    losers = population[-6:]
    group1, group2 = losers[0:3], losers[3:6]

    combo1 = random.randint(0, 7)  # Binary 000-111
    combo2 = random.randint(0, 7)

    loser_child1 = apply_merge_combo(group1, combo1)
    loser_child2 = apply_merge_combo(group2, combo2)

    loser_children = [loser_child1, loser_child2]

    # Step 4: New population
    population = elite_children + loser_children  # 8 models

    # Step 5: Update champion
    best_this_gen = population[0]  # Sorted, so [0] is best
    best_fitness = evaluate_fitness(best_this_gen)

    if best_fitness > champion_fitness:
        best_champion = best_this_gen
        champion_fitness = best_fitness

    # Step 6: Log to W&B
    wandb.log({
        "generation": generation,
        "best_fitness": champion_fitness,
        "diversity": compute_diversity(population),
        "elite1_fitness": fitness_scores[0],
        "elite2_fitness": fitness_scores[1],
    })

    # Step 7: Early stopping (optional)
    if generation > 5:
        recent_improvements = fitness_history[-5:]
        if max(recent_improvements) - min(recent_improvements) < 0.001:
            print(f"Converged at generation {generation}")
            break

# After 50 generations
final_model = best_champion
```

---

### **Cleanup & Handoff**

```python
# Delete intermediate models
delete(model1, model2, model3)  # Original Phase 1 models
delete(all_population_histories)  # Generation 1-49 populations

# Keep only champion
save(final_model, "phase2_output.pt")

# Validate improvement
initial_fitness = mean([eval(m) for m in [model1, model2, model3]])
final_fitness = evaluate_fitness(final_model)
improvement = (final_fitness - initial_fitness) / initial_fitness

assert improvement > 0.20, f"Evolution didn't improve enough ({improvement:.1%})"
print(f"✅ Evolution achieved {improvement:.1%} fitness improvement")

# Pass to Phase 3
phase3_input = final_model
```

---

## 📊 Expected Performance Metrics

| Metric | Target | Typical Achievement |
|--------|--------|---------------------|
| **Fitness Improvement** | >20% | **23.5%** |
| **Convergence Generation** | <50 | 35-40 |
| **Diversity Range** | >0.3 | 0.35-0.45 |
| **Processing Time** | <2 hours | **90 minutes** (GTX 1660) |
| **Memory Usage** | <6GB VRAM | 5.2GB (12 models × 25M params) |

---

## ✅ Success Criteria

- ✅ **50 generations complete** (or early stop with convergence)
- ✅ **Fitness improvement ≥20%** (target: 23.5%)
- ✅ **Diversity maintained** (>0.3 throughout evolution)
- ✅ **Evolution time** <90 minutes on GTX 1660
- ✅ **All 8 merge combos** used at least once
- ✅ **W&B logs** show fitness improvement curve
- ✅ **Champion model** beats all Phase 1 models individually

---

## 🚨 Critical Implementation Notes

### 1. **Fitness Function Must Be Composite**

❌ **Bad** (Single Metric):
```python
fitness = accuracy  # Ignores speed, memory
```

✅ **Good** (Composite):
```python
fitness = (
    0.4 * (1 / perplexity) +
    0.3 * accuracy +
    0.2 * (1 / inference_time) +
    0.1 * (1 / memory_usage)
)
```

---

### 2. **Diversity Must Be Maintained**

**Problem**: Population converges to single solution (stagnation).

**Solution**: Track diversity, re-seed if too low:
```python
diversity = compute_pairwise_distance(population)
if diversity < 0.3:
    # Re-seed bottom 2 with random hybrids
    population[-2] = random_merge_combo([model1, model2, model3])
    population[-1] = random_merge_combo([model1, model2, model3])
```

---

### 3. **Merge Techniques Can Fail**

**Problem**: Some merges create degenerate models (NaN weights, dimension mismatches).

**Solution**: Validate after merge, fallback to linear:
```python
child = apply_merge_combo(models, combo_id)
if not is_valid_model(child):  # Check for NaN, Inf, dimension errors
    # Fallback to safe linear merge
    child = linear_merge(models[0], models[1], models[2])
```

---

### 4. **Memory Management**

**Problem**: 8 models × 25M params × 50 generations = OOM.

**Solution**: Only keep 12 models at a time:
```python
# Don't save all generations
# Only keep: 3 original + 1 champion + 8 current gen = 12 total
```

---

## 🔗 Integration Points

**Input**: 3 models from Phase 1 (Cognate)
**Output**: 1 evolved model to Phase 3 (Quiet-STaR)

**V2-Specific Notes**:
- V1 Phase 2 was production-ready (23.5% gain validated)
- V2 uses **binary combination strategy** (NEW)
- V2 uses **elite mutation + loser merging** (NEW)
- TRM × Titans-MAG architecture (NEW in Phase 1)

---

**Next Phase**: [Phase 3: Quiet-STaR - Logical Understanding](../phase3/LOGICAL_UNDERSTANDING.md)

**Version**: 2.0 (CORRECTED)
**Last Updated**: 2025-10-15
**Status**: ✅ Ready for Implementation
