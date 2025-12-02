# Phase 7: Self-Guided Expert System - Logical Understanding

**Version**: V2 (Self-Guided Architecture)
**Last Updated**: 2025-10-16
**Phase Purpose**: Model-driven expert discovery, specialization, and optimization
**Key Innovation**: Model determines its own expert strategy, generates training data, and guides architecture search

---

## What This Phase Does (In Plain English)

Phase 7 creates a **self-guided expert system** where the model itself:

1. **Analyzes** its own capabilities across multiple domains
2. **Determines** how many experts it needs (N=3-10) and what they should specialize in
3. **Generates** its own training data via frontier models (GPT-4, Claude-3.5)
4. **Trains** specialized expert vectors using Transformer² SVF
5. **Guides** evolutionary search (ADAS) to find optimal expert mixtures
6. **Validates** results against Phase 5 eudaimonia alignment and Phase 6 capabilities

### V1 vs V2: Paradigm Shift

| Aspect | V1 (Manual) | V2 (Self-Guided) |
|--------|-------------|------------------|
| **Expert Count** | Hardcoded (N=4) | **Model determines** (N=3-10) |
| **Domains** | Predefined (math, code, reasoning, vision) | **Model discovers** via self-analysis |
| **Data** | Human-curated datasets | **Model generates** via OpenRouter |
| **ADAS** | Metric-based search | **Model-guided** optimization |
| **Validation** | Accuracy thresholds | **Model validates** alignment preservation |
| **Integration** | Isolated | **Full** Phase 5/6/8 integration |

---

## Three-Stage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 7: SELF-GUIDED SYSTEM                    │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
  STAGE 1             STAGE 2             STAGE 3
  Expert Discovery    SVF Training        Model-Guided ADAS
  (Model-Driven)      (Transformer²)      (Model-Validated)
        │                   │                   │
   [Self-Analysis]     [REINFORCE + KL]   [NSGA-II + Model]
   [Domain Cluster]    [MuonGrokfast]     [Fitness Eval]
   [Data Generation]   [Eudaimonia Check] [Alignment Check]
        │                   │                   │
        ↓                   ↓                   ↓
  N experts defined   N z-vector sets    Optimal mixture α*
  N datasets ready    >85% accuracy      <100ms, <2GB, >85%
```

---

## Stage 1: Model-Driven Expert Discovery

### What It Does

The model performs **introspective capability analysis** to discover its own strengths and weaknesses, then determines:
- How many experts it needs (N)
- What domains each expert should cover
- What training data it needs for improvement

### Process Flow

```
[1] Capability Self-Analysis (2 hours)
    ↓
    Model tests itself on 20 micro-benchmarks:
    • Math: GSM8K (5 problems), MATH (3 problems)
    • Code: HumanEval (5), MBPP (3)
    • Reasoning: ARC (3), Winogrande (2)
    • Creative: Custom prompts (2)
    • Factual: MMLU (2)
    • Planning: Multi-step tasks (2)
    • Communication: Explanation clarity (1)
    • Meta-cognition: Phase 5 self-modeling (2)
    ↓
    Capability Report:
    {
      "math": 0.72,           # WEAK
      "code": 0.85,           # STRONG
      "reasoning": 0.65,      # WEAK
      "creative": 0.58,       # WEAK
      "factual": 0.78,        # STRONG
      "planning": 0.70,       # MEDIUM
      "communication": 0.82,  # STRONG
      "meta_cognition": 0.91  # VERY STRONG (Phase 5)
    }
    ↓
[2] Domain Clustering & Expert Determination (1 hour)
    ↓
    Model analyzes report and clusters domains:

    Meta-prompt to model:
    "Given your capabilities, how many experts do you need?
     Which domains should be grouped together?
     Which need dedicated experts?"
    ↓
    Model Decision:
    N = 5 experts:
    {
      "analytical": ["math", "reasoning"],      # Both weak, group
      "creative": ["creative"],                 # Weakest, isolate
      "planning": ["planning"],                 # Leverage Phase 6 tools
      "code": ["code"],                         # Strong, refine further
      "general": ["factual", "communication"]   # Strong, maintain
    }
    ↓
[3] Data Requirements Specification (1 hour)
    ↓
    For each expert, model specifies training data:

    Example (Analytical expert):
    {
      "topics": [
        "multi-step arithmetic with fractions",
        "algebraic equation solving",
        "logical inference (if-then, proof)",
        "pattern recognition"
      ],
      "difficulty_target": "75% correctness (edge-of-chaos)",
      "num_examples": 450,
      "format": "free_response_with_reasoning",
      "special_requirements": [
        "Include step-by-step reasoning",
        "Vary surface features (prevent memorization)",
        "Mix problem types (multi-hop)"
      ]
    }
    ↓
[4] Dataset Generation via OpenRouter (4-6 hours)
    ↓
    For each expert:
    • Send model's specification to GPT-4/Claude-3.5
    • Frontier model generates 500 problems
    • Model validates EACH problem:
      - Difficulty check: 60-90% confidence (solvable but challenging)
      - Eudaimonia check: Score ≥0.65 (Phase 5 alignment)
      - Edge-of-chaos check: In learning zone (70-80%)
      - Quality check: Clear, correct, non-repetitive
    • Accept 75%+ of problems, augment if needed
    ↓
    Output: N datasets (300-500 problems each)
    Cost: ~$150 (5 experts × $30/expert)
```

### Key Innovation: Model Self-Validation

**Unlike Phase 3** (human validates data), **Phase 7** has the model validate its own training data:

```python
def model_validate_problem(model, problem):
    """
    MODEL decides if a problem is good for its training.
    """
    # 1. Attempt the problem
    attempt = model.solve(problem, return_confidence=True)

    # 2. Difficulty check (edge-of-chaos)
    if attempt['confidence'] > 0.90:
        return False, "Too easy"
    if attempt['confidence'] < 0.40:
        return False, "Too hard"

    # 3. Eudaimonia check (Phase 5 integration)
    eudaimonia_score = model.eudaimonia_score(problem)
    if eudaimonia_score < 0.65:
        return False, "Violates moral alignment"

    # 4. Quality check (model's own judgment)
    quality_prompt = f"Is this a high-quality problem? {problem}"
    quality = model.generate(quality_prompt, temperature=0.1)
    if not quality.startswith('YES'):
        return False, "Poor quality"

    return True, "Accepted"
```

---

## Stage 2: Transformer² SVF Training (Model-Validated)

### What It Does

Train N model-determined experts using **Singular Value Fine-tuning (SVF)** with continuous model validation.

### SVF Explained

**Standard LoRA**:
- Adds low-rank matrices: `W' = W + BA` where B ∈ ℝ^(m×r), A ∈ ℝ^(r×n)
- Parameters: (m+n)×r

**Transformer² SVF**:
- Scales singular values: `W' = U(Σ ⊗ diag(z))V^T` where z ∈ ℝ^r
- Parameters: r = min(m,n)

**Example** (4096×4096 matrix, rank=16):
- LoRA: (4096+4096)×16 = **131,072 params**
- SVF: 4096 = **4,096 params** (32× fewer!)

### Training Process

```python
def stage2_svf_training(model, stage1_output):
    """
    Train N model-determined experts with model validation.
    """
    experts = {}

    for expert_spec in stage1_output['experts']:
        # Load model-generated dataset
        dataset = load_dataset(expert_spec['dataset_path'])

        # Initialize z-vectors (1-D parameters)
        z_vectors = {}
        for name, param in model.named_parameters():
            if param.ndim == 2:  # Only weight matrices
                r = min(param.shape)
                z_vectors[name] = nn.Parameter(
                    torch.normal(0.1, 1e-3, size=(r,))
                )

        # MuonGrokfast optimizer (fallback-only for 1-D z-vectors)
        optimizer = MuonGrokfast(
            z_vectors.values(),
            config=MuGrokConfig(
                enable_muon=False,        # z-vectors are 1-D
                fallback_type="adamw",
                fallback_lr=2e-3,
                grokfast_alpha=0.98,
                grokfast_lambda=0.05,     # Light RL filtering
                kl_coefficient=0.2,       # Constrain drift
                phase_name=f"phase7_{expert_spec['name']}"
            )
        )

        # REINFORCE training loop
        for epoch in range(expert_spec['training_epochs']):
            for batch in dataset:
                # Apply SVF: W' = U(Σ ⊗ diag(z))V^T
                W_adapted = apply_svf(model.weights, z_vectors)

                # Forward pass with adapted weights
                output = model.forward(batch.input, weights=W_adapted)

                # REINFORCE reward
                reward = 1.0 if correct(output, batch.target) else -1.0

                # Policy gradient + KL penalty
                log_prob = model.log_prob(output, batch.input)
                kl_div = compute_kl(model, W_adapted)
                loss = -reward * log_prob + kl_coefficient * kl_div

                # Backprop + optimizer step
                loss.backward()
                optimizer.step()

            # ============================================
            # MODEL VALIDATION (Every Epoch)
            # ============================================
            validation = model_self_validate(
                model,
                z_vectors,
                expert_spec['validation_criteria']
            )

            # Log to W&B
            wandb.log({
                f"expert_{expert_spec['id']}/accuracy": validation['accuracy'],
                f"expert_{expert_spec['id']}/eudaimonia": validation['eudaimonia'],
                f"expert_{expert_spec['id']}/edge_of_chaos": validation['edge_of_chaos']
            })

            # MODEL EARLY STOPPING
            if validation['meets_criteria']:
                print(f"Expert {expert_spec['id']} converged at epoch {epoch}")
                break

            # MODEL INTERVENTION: Drift detection
            if validation['eudaimonia'] < 0.65:
                print(f"⚠️  Expert {expert_spec['id']} drifting from eudaimonia!")
                optimizer.config.kl_coefficient *= 1.5  # Increase constraint

        experts[expert_spec['name']] = z_vectors

    return experts


def model_self_validate(model, z_vectors, criteria):
    """
    MODEL validates its own expert training progress.
    """
    # Apply current expert
    W_adapted = apply_svf(model.weights, z_vectors)

    # Test accuracy
    val_dataset = load_validation_dataset()
    accuracy = model.evaluate(val_dataset, weights=W_adapted)

    # Eudaimonia check (Phase 5 integration)
    eudaimonia_score = model.eudaimonia_self_test()

    # Edge-of-chaos check (Phase 5 integration)
    edge_of_chaos_score = model.self_modeling_accuracy(W_adapted)

    return {
        "accuracy": accuracy,
        "eudaimonia": eudaimonia_score,
        "edge_of_chaos": edge_of_chaos_score,
        "meets_criteria": (
            accuracy >= float(criteria['accuracy'].split()[1]) and
            eudaimonia_score >= criteria['eudaimonia_score'] and
            0.70 <= edge_of_chaos_score <= 0.80
        )
    }
```

### MuonGrokfast Integration (Fallback-Only)

**Why fallback-only?** SVF z-vectors are **1-D parameters** (shape: `[r]`).

**MuonGrokfast parameter routing**:
```python
for param in params:
    if param.ndim >= 2 and config.enable_muon:
        # Muon path: Newton-Schulz orthogonalization (2-D+ only)
        update = muon_step(param, grad)
    else:
        # Fallback path: AdamW/Lion (1-D params like z-vectors)
        update = fallback_optimizer.step()
```

**Phase 7 explicitly sets `enable_muon=False`** to clarify z-vectors use fallback.

**What Phase 7 uses from MuonGrokfast**:
- ✅ **Grokfast gradient filtering** (λ=0.05): Stabilizes noisy REINFORCE gradients
- ✅ **AdamW optimizer**: Adaptive learning rates for z-vector updates
- ✅ **KL regularization**: Built-in coefficient (0.2) prevents drift
- ❌ **Muon orthogonalization**: Not applicable to 1-D vectors

---

## Stage 3: Model-Guided ADAS Search

### What It Does

ADAS (Automated Design of Agentic Systems) uses NSGA-II evolutionary search to find optimal expert mixtures, but **guided by the model itself** rather than purely metric-driven.

### Process Flow

```
[1] Model Defines Search Objectives (30 min)
    ↓
    Model introspection:
    "What should I optimize for?"

    Meta-prompt:
    "Given your expert performance:
     • Analytical: 86% (exceeded target)
     • Creative: 73% (close to target)
     • Planning: 78% (close to target)
     • Code: 88% (close to target)
     • General: 81% (close to target)

     Hardware: GTX 1660 (6GB VRAM), <100ms latency

     What mixture of experts will:
     1. Maximize overall capability
     2. Maintain eudaimonia alignment (Phase 5)
     3. Preserve tool use (Phase 6)
     4. Stay within hardware constraints"
    ↓
    Model generates search criteria:
    {
      "primary_objective": "maximize_overall_accuracy",
      "constraints": [
        {"type": "latency", "max": 100, "unit": "ms"},
        {"type": "memory", "max": 2.0, "unit": "GB"},
        {"type": "eudaimonia_score", "min": 0.65},
        {"type": "edge_of_chaos", "range": [0.70, 0.80]}
      ],
      "search_space": {
        "analytical": [0.0, 3.0],    # High weight (weak domain)
        "creative": [0.0, 2.5],      # High weight (weak domain)
        "planning": [0.0, 2.0],
        "code": [0.0, 1.5],          # Restrict (already strong)
        "general": [0.5, 1.5]        # Require minimum (base capability)
      }
    }
    ↓
[2] NSGA-II Search with Model Evaluation (40 hours)
    ↓
    For each generation (100 total):

    [2.1] Generate Population (50 mixtures)
    ↓
    Random mixtures within model-defined search space

    [2.2] Evaluate Fitness (MODEL-DRIVEN)
    ↓
    For each mixture α = [α₁, α₂, α₃, α₄, α₅]:

      # Compose experts
      z_composed = Σ(α_i × z_expert_i)

      # Apply to model
      model_adapted = apply_svf(model, z_composed)

      # STANDARD METRICS
      latency = measure_latency(model_adapted)
      memory = measure_vram(model_adapted)

      # MODEL SELF-EVALUATION (Key Innovation!)
      model_eval = model.self_evaluate_mixture(
          model_adapted,
          criteria={
              "domains": ["math", "code", "reasoning", ...],
              "eudaimonia_test": True,
              "edge_of_chaos_test": True,
              "tool_use_test": True,      # Phase 6
              "persona_test": True         # Phase 6
          }
      )

      # Fitness vector (NSGA-II minimizes all)
      fitness = [
          latency,                                  # Minimize
          memory,                                   # Minimize
          1.0 - model_eval['overall_accuracy'],    # Minimize (= max)
          1.0 - model_eval['eudaimonia_score'],    # Minimize (= max)
          abs(0.75 - model_eval['edge_of_chaos'])  # Minimize distance
      ]

    [2.3] MODEL INTERVENTION (Every 10 Generations)
    ↓
    Model analyzes Pareto front:

    "Current best mixtures: {pareto_front_summary}
     Am I converging? Should I adjust search space?"
    ↓
    Model guidance:
    {
      "adjust_search_space": True,
      "new_bounds": {
        "analytical": [1.5, 3.0],  # Narrow (all good mixtures >1.5)
        ...
      },
      "adjust_objectives": False,
      "early_stopping": False
    }

    [2.4] Selection, Crossover, Mutation
    ↓
    Standard NSGA-II genetic operators

    [2.5] Next Generation
    ↓
    Repeat [2.2]-[2.4]
    ↓
[3] Model Validates Final Mixture (30 min)
    ↓
    Best from Pareto front: α* = [1.8, 1.3, 1.0, 0.9, 1.1]

    Model comprehensive validation:
    "Does this mixture preserve my capabilities and alignment?"

    final_validation = model.comprehensive_self_test(
        mixture=alpha_star,
        tests=[
            "eudaimonia_100_scenarios",
            "capability_benchmarks",
            "edge_of_chaos_stability",
            "tool_use_preservation",      # Phase 6
            "persona_consistency"          # Phase 6
        ]
    )

    if final_validation['all_passed']:
        return alpha_star
    else:
        # Model rejects, chooses 2nd best
        return second_best_mixture
```

### Model Self-Evaluation Function

```python
def model_self_evaluate_mixture(model, adapted_model, criteria):
    """
    MODEL evaluates expert mixture from its own perspective.

    This is the KEY INNOVATION: Model judges its own capabilities.
    """
    results = {}

    # Test across model-determined domains
    for domain in criteria['domains']:
        test_set = model.generate_self_test(domain, n=50)
        accuracy = adapted_model.evaluate(test_set)
        results[f"{domain}_accuracy"] = accuracy

    results['overall_accuracy'] = np.mean([
        results[f"{d}_accuracy"] for d in criteria['domains']
    ])

    # Eudaimonia alignment check (Phase 5)
    if criteria['eudaimonia_test']:
        eudaimonia_scenarios = load_eudaimonia_test_set()
        results['eudaimonia_score'] = adapted_model.eudaimonia_self_test(
            eudaimonia_scenarios
        )

    # Edge-of-chaos check (Phase 5)
    if criteria['edge_of_chaos_test']:
        results['edge_of_chaos'] = adapted_model.self_modeling_accuracy()

    # Tool use preservation (Phase 6)
    if criteria['tool_use_test']:
        tool_test = load_swe_bench_subset(n=20)
        results['tool_use_rate'] = adapted_model.evaluate_tool_usage(tool_test)

    # Persona consistency (Phase 6)
    if criteria['persona_test']:
        persona_test = generate_persona_consistency_test()
        results['persona_score'] = adapted_model.evaluate_persona(persona_test)

    return results
```

---

## Integration with Previous Phases

### Phase 5 (Eudaimonia System) Integration

**What Phase 5 Provides**:
- Eudaimonia 4-rule system (baked into weights)
- Edge-of-chaos learning zone (75% optimal)
- Self-modeling capability

**How Phase 7 Uses It**:

**Stage 1 (Discovery)**:
```python
# Model validates generated datasets against eudaimonia
for problem in candidate_dataset:
    eudaimonia_check = model.eudaimonia_score(problem)
    if eudaimonia_check < 0.65:
        reject_problem(problem)  # Violates alignment
```

**Stage 2 (Training)**:
```python
# Every epoch, check eudaimonia drift
eudaimonia_score = model.eudaimonia_self_test()
if eudaimonia_score < 0.65:
    increase_kl_coefficient()  # Constrain adaptation
```

**Stage 3 (ADAS)**:
```python
# Eudaimonia is a fitness objective
fitness = [
    latency,
    memory,
    1.0 - accuracy,
    1.0 - eudaimonia_score,        # Preserve alignment
    abs(0.75 - edge_of_chaos)      # Maintain learning zone
]
```

### Phase 6 (Tool & Persona Baking) Integration

**What Phase 6 Provides**:
- Tool usage patterns (A-cycle, SWE-Bench optimized)
- Self-guided persona (B-cycle, benchmark optimized)

**How Phase 7 Uses It**:

**Stage 1 (Discovery)**:
```python
# Model uses baked tool capabilities for planning expert
planning_expert_dataset = model.generate_planning_problems(
    use_tools=True,  # Leverage Phase 6 tool baking
    difficulty="edge_of_chaos"
)
```

**Stage 3 (ADAS)**:
```python
# Validate tool use preservation after adaptation
tool_use_rate = adapted_model.test_tool_usage(swe_bench_subset)
if tool_use_rate < phase6_baseline:
    penalize_fitness(mixture)  # Mixture hurt tool use
```

---

## Expected Inputs

**From Phase 6**:
```json
{
  "model": "phase6_optimized_model.pt",
  "metrics": {
    "swe_bench_score": 0.613,
    "aggregate_benchmark": 0.794,
    "tool_use_rate": 0.94,
    "persona_consistency": 0.89
  },
  "baked_capabilities": {
    "tool_prompts": ["Always use tools before answering...", ...],
    "persona_prompts": ["I identify core constraints first...", ...]
  }
}
```

**Phase 6 model has**:
- BitNet 1.58-bit quantization (Phase 4)
- Eudaimonia alignment (Phase 5)
- Tool usage patterns (Phase 6 A-cycle)
- Self-guided persona (Phase 6 B-cycle)

---

## Expected Outputs

**To Phase 8**:
```json
{
  "model": "phase7_adapted_model.pt",
  "expert_library": {
    "analytical": "z_vectors_analytical.pt",
    "creative": "z_vectors_creative.pt",
    "planning": "z_vectors_planning.pt",
    "code": "z_vectors_code.pt",
    "general": "z_vectors_general.pt"
  },
  "optimal_mixture": [1.8, 1.3, 1.0, 0.9, 1.1],
  "num_experts_determined_by_model": 5,
  "performance": {
    "latency_ms": 87.3,
    "vram_gb": 1.8,
    "overall_accuracy": 0.854,
    "eudaimonia_score": 0.71,
    "edge_of_chaos": 0.76,
    "tool_use_rate": 0.93,
    "persona_consistency": 0.88
  },
  "compression_notes": {
    "z_vectors_are_1d": true,
    "total_expert_params": 20480,
    "base_model_params": 25000000,
    "adaptation_overhead": 0.082
  }
}
```

---

## Success Criteria

### Stage 1: Expert Discovery
- ✅ Model generates capability report across ≥8 domains
- ✅ Model determines expert count N (3 ≤ N ≤ 10)
- ✅ Model generates N datasets (300-500 problems each)
- ✅ Validation acceptance rate ≥75% per dataset
- ✅ All datasets pass eudaimonia check (≥0.65)
- ✅ Cost <$250 (target: $150)

### Stage 2: SVF Training
- ✅ All N experts trained to target accuracy
- ✅ Eudaimonia maintained (≥0.65) throughout
- ✅ Edge-of-chaos preserved (0.70-0.80)
- ✅ Model validation every epoch
- ✅ Model intervention on drift

### Stage 3: Model-Guided ADAS
- ✅ Model defines search objectives
- ✅ NSGA-II converges within 100 generations
- ✅ Model guidance every 10 generations
- ✅ Final mixture passes model validation:
  - Overall accuracy improvement >10%
  - Eudaimonia score ≥0.65
  - Edge-of-chaos: 0.70-0.80
  - Tool use preserved (≥Phase 6 baseline)
  - Persona preserved (≥Phase 6 baseline)
  - Latency <100ms (GTX 1660)
  - Memory <2GB VRAM

### Integration Validation
- ✅ Phase 5 eudaimonia integrated at all stages
- ✅ Phase 6 tool/persona capabilities preserved
- ✅ Phase 7 output compatible with Phase 8 compression

---

## Timeline & Cost Estimate

### Timeline: 3-4 Days (GTX 1660)

**Day 1: Stage 1 (Expert Discovery)** - 8 hours
- Capability self-analysis: 2 hours
- Expert determination: 1 hour
- Data generation (OpenRouter): 4 hours
- Validation: 1 hour

**Day 2-3: Stage 2 (SVF Training)** - 36 hours
- 5 experts × 5 epochs = 25 training runs
- ~1.5 hours per expert (with validation)
- Sequential: 36 hours (1.5 days)
- Parallel (5× GPU): 7.5 hours

**Day 3-4: Stage 3 (ADAS Search)** - 42 hours
- 100 generations × 50 population = 5000 evals
- ~30 sec per eval (model-intensive)
- Model guidance: 10 × 5 min = 50 min
- Total: ~42 hours (1.75 days)

**Total**: 3-4 days on single GTX 1660

### Cost: $150-250

**Data Generation (OpenRouter)**: $150
- 5 experts × 500 problems = 2500 problems
- $0.06 per problem (GPT-4 generation + validation)

**Model Self-Evaluation (Optional)**: $100
- ADAS: 5000 evals, ~100 need OpenRouter
- $1 per call (complex prompts)
- Can be done locally for free

**Total**: $150-250

---

## Research Papers

1. **Transformer²**: "Transformer-Squared: Self-Adaptive LLMs" (2501.06252v3.pdf)
   - SVF parameterization
   - REINFORCE + KL training
   - Expert composition

2. **ADAS**: "Automated Design of Agentic Systems"
   - Multi-objective optimization (NSGA-II)
   - Architecture search
   - Sandboxed evaluation

3. **Prompt Baking** (Phase 5/6 reference): arXiv:2409.13697v1
   - Eudaimonia alignment system
   - Edge-of-chaos learning

---

## Next Phase

**Phase 8**: Final Compression (SeedLM + VPTQ + Hypercompression)

Phase 8 receives:
- Base model (25M params, BitNet 1.58-bit)
- Expert library (5 × 4K params = 20K total)
- Optimal mixture weights

Phase 8 compresses:
- Base model: 280× compression (25M → 89K params)
- Expert library: Can be compressed or composed

---

**Phase 7 Status**: ✅ Complete Self-Guided Specification
**Key Innovation**: Model-driven expert discovery, training, and validation
**Integration**: Full Phase 5/6 integration, Phase 8 handoff specified
**Timeline**: 3-4 days | **Cost**: $150-250
