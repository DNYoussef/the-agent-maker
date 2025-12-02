# Generation 0: 8 Unique Model Configurations

## Overview

Generation 0 creates **8 completely unique models** from **3 base TinyTitan models** using all 6 merge techniques with distinct configurations. Each model uses a different merging strategy to maximize population diversity.

**Input:** 3 TinyTitan models (Reasoning, Memory Integration, Adaptive Computation)
**Output:** 8 evolved models with unique characteristics
**Techniques Used:** All 6 (Linear, SLERP, TIES, DARE, FrankenMerge, DFS)

---

## The 8 Models

### Model 0: Linear Merge - Equal Weights (Baseline)
**Technique:** Linear
**Configuration:** `weights = [0.333, 0.333, 0.333]`

**Purpose:** Balanced baseline combining all 3 models equally

**Expected Characteristics:**
- Smooth blend of all capabilities
- No particular bias toward any input model
- General-purpose performance

**Formula:**
```
merged = 0.333×Model_A + 0.333×Model_B + 0.333×Model_C
```

---

### Model 1: SLERP - Spherical Interpolation
**Technique:** SLERP
**Configuration:** `t = 0.5, models = [Model_A, Model_B]`

**Purpose:** Geometric midpoint between first two models on hypersphere

**Expected Characteristics:**
- Preserves magnitude better than linear interpolation
- Combines Reasoning + Memory capabilities
- Smooth transition in parameter space

**Formula:**
```
merged = SLERP(Model_A, Model_B, 0.5)
       = [sin(0.5θ)/sin(θ)] × Model_A + [sin(0.5θ)/sin(θ)] × Model_B
```

---

### Model 2: TIES - Trim, Elect, Merge (Research-Grade)
**Technique:** TIES
**Configuration:** `k_percent = 20%, base_model = Model_A`

**Purpose:** Resolve parameter conflicts using sign voting

**Expected Characteristics:**
- Keeps only top 20% most influential parameters
- Eliminates sign conflicts between models
- High-quality task arithmetic merge
- Likely best overall performance

**Algorithm:**
1. **TRIM:** Keep top 20% of deltas by magnitude
2. **ELECT SIGN:** Vote on positive/negative for each parameter
3. **DISJOINT MERGE:** Average only same-sign parameters

---

### Model 3: DARE - High Sparsity (90% Dropout)
**Technique:** DARE
**Configuration:** `drop_rate = 0.9, base_model = Model_A`

**Purpose:** Extreme sparsification - keep only 10% of deltas

**Expected Characteristics:**
- Very sparse parameter updates
- Focuses on most critical changes from base
- Fast inference due to sparsity
- May sacrifice some accuracy for efficiency

**Algorithm:**
1. Compute deltas: Δ = models - base
2. Drop 90% of delta parameters randomly
3. Rescale remaining by 10
4. Merge and add back to base

---

### Model 4: FrankenMerge - Sequential Pattern "ABC"
**Technique:** FrankenMerge
**Configuration:** `layer_pattern = "ABC"`

**Purpose:** Interleave layers from all 3 models sequentially

**Expected Characteristics:**
- Layer 0: Model A (Reasoning)
- Layer 1: Model B (Memory)
- Layer 2: Model C (Adaptive)
- Layer 3: Model A (Reasoning)
- ...repeating pattern

**Unique Capability:** Combines depth-specific strengths from each model

---

### Model 5: DFS - Feature Importance Weighting
**Technique:** DFS (Deep Feature Selection)
**Configuration:** Automatic inverse-variance importance

**Purpose:** Weight parameters by their stability across models

**Expected Characteristics:**
- Stable parameters (low variance) get higher weight
- Unstable parameters (high variance) get lower weight
- Emphasizes consensus features
- Robust to outlier parameters

**Algorithm:**
1. Calculate variance of each parameter across models
2. Importance = 1 / (1 + variance)
3. Weight merge by importance scores

---

### Model 6: Linear Merge - Biased Weights
**Technique:** Linear
**Configuration:** `weights = [0.5, 0.3, 0.2]`

**Purpose:** Favor first model (Reasoning) while including others

**Expected Characteristics:**
- 50% Reasoning model influence
- 30% Memory model influence
- 20% Adaptive model influence
- Strong reasoning with supporting capabilities

**Use Case:** When reasoning performance is most critical

---

### Model 7: FrankenMerge - Symmetric Pattern "ABBA"
**Technique:** FrankenMerge
**Configuration:** `layer_pattern = "ABBA"`

**Purpose:** Symmetric layer distribution creating balanced architecture

**Expected Characteristics:**
- Layer 0: Model A
- Layer 1: Model B
- Layer 2: Model B (emphasis on memory)
- Layer 3: Model A
- ...repeating ABBA pattern

**Unique Capability:** Double emphasis on middle model's capabilities in center layers

---

## Configuration Summary Table

| Model | Technique | Key Parameter | Base Model | Uniqueness |
|-------|-----------|---------------|------------|------------|
| 0 | Linear | weights=[0.33,0.33,0.33] | N/A | Balanced baseline |
| 1 | SLERP | t=0.5 | N/A | Geometric interpolation |
| 2 | TIES | k=20% | Model A | Sign voting, top-k trim |
| 3 | DARE | drop=90% | Model A | Extreme sparsity |
| 4 | FrankenMerge | "ABC" | N/A | Sequential layers |
| 5 | DFS | auto-importance | N/A | Variance weighting |
| 6 | Linear | weights=[0.5,0.3,0.2] | N/A | Reasoning-biased |
| 7 | FrankenMerge | "ABBA" | N/A | Symmetric layers |

---

## Diversity Analysis

### Technique Coverage
- ✅ Linear: 2 variants (equal, biased)
- ✅ SLERP: 1 variant (t=0.5)
- ✅ TIES: 1 variant (20%)
- ✅ DARE: 1 variant (90% dropout)
- ✅ FrankenMerge: 2 variants (ABC, ABBA)
- ✅ DFS: 1 variant (auto)

### Parameter Space Coverage
- **Weight Distribution:** Equal (0), Biased (6)
- **Interpolation:** Spherical (1)
- **Task Arithmetic:** TIES (2), DARE (3)
- **Layer Mixing:** Sequential (4), Symmetric (7)
- **Feature Selection:** Importance-weighted (5)

### Base Model Usage
- **Models using base for deltas:** 2, 3 (TIES, DARE)
- **Models without base:** 0, 1, 4, 5, 6, 7

---

## Expected Performance Characteristics

### Likely Top Performers
1. **Model 2 (TIES)** - Research shows best task arithmetic performance
2. **Model 5 (DFS)** - Consensus features often robust
3. **Model 0 (Linear Equal)** - Simple baseline often competitive

### Likely Experimental/Exploratory
1. **Model 3 (DARE)** - High sparsity may hurt accuracy
2. **Model 4, 7 (FrankenMerge)** - Highly experimental, variable results

### Likely Specialized
1. **Model 6 (Linear Biased)** - Good for reasoning-heavy tasks
2. **Model 1 (SLERP)** - Good for tasks benefiting from A+B combination

---

## Fitness Evaluation

All 8 models will be evaluated on:
1. **Perplexity** (lower is better)
2. **Accuracy** (higher is better)
3. **Inference Speed** (higher tokens/sec is better)
4. **Memory Usage** (lower is better)

**Composite Fitness:**
```
fitness = w1×(1/perplexity) + w2×accuracy + w3×(1/latency) + w4×(1/memory)
```

---

## Tournament Selection

After evaluation:
- **Top 2 models** (winners) → Preserved + create 3 offspring each
- **Bottom 6 models** (losers) → Used for chaos merging (2 offspring)
- **Generation 1** → New population of 8 evolved models

---

## Rationale for Configuration Choices

### Why these specific parameters?

**Model 0 (Equal weights):** Standard baseline for comparison

**Model 1 (SLERP t=0.5):** Midpoint maximizes balance, paper recommendation

**Model 2 (TIES k=20%):** NeurIPS paper default, proven optimal

**Model 3 (DARE drop=0.9):** Research shows 90-99% dropout acceptable for SFT

**Model 4 (ABC pattern):** Simple rotation ensures all models contribute

**Model 5 (DFS auto):** Automatic importance eliminates manual tuning

**Model 6 (50-30-20 weights):** Fibonacci-like ratio, common in ensemble methods

**Model 7 (ABBA pattern):** Symmetric pattern from SOLAR-10.7B approach

---

## Code Implementation

```python
# Generation 0 initialization (simplified)
def initialize_population(base_models):
    return [
        linear(base_models, [0.33, 0.33, 0.33]),           # Model 0
        slerp(base_models[:2], t=0.5),                     # Model 1
        ties(base_models, k=20%, base=base_models[0]),     # Model 2
        dare(base_models, drop=0.9, base=base_models[0]),  # Model 3
        frankenmerge(base_models, "ABC"),                  # Model 4
        dfs(base_models),                                  # Model 5
        linear(base_models, [0.5, 0.3, 0.2]),              # Model 6
        frankenmerge(base_models, "ABBA"),                 # Model 7
    ]
```

---

## Verification Checklist

✅ All 8 models use different configurations
✅ All 6 techniques represented
✅ No duplicate models
✅ Covers diverse merging strategies
✅ Includes both simple (Linear) and complex (TIES) approaches
✅ Balanced exploration vs exploitation
✅ Research-backed parameter choices

---

## Next Steps: Generation 1

Top 2 models from Generation 0 will:
1. Be preserved unchanged (elitism)
2. Create 3 offspring each via mutation
3. Bottom 6 create 2 chaos-merged offspring
4. Total: 2 elite + 6 mutated + 2 chaos = 10 models → select best 8

This process repeats for 5 generations, progressively improving the population.
