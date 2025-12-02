# Phase 7: Self-Guided Expert Discovery & Optimization System

**Version**: V2 (Self-Guided Architecture)
**Last Updated**: 2025-10-16
**Status**: Complete Specification

---

## Executive Overview

Phase 7 represents a **paradigm shift** from human-designed to **model-driven architecture optimization**. Unlike traditional approaches where engineers specify expert domains and search parameters, Phase 7 empowers the model to:

1. **Self-analyze** its capabilities across domains
2. **Determine** how many experts it needs and what domains they cover
3. **Generate** its own training data for expert specialization
4. **Guide** the architecture search process
5. **Validate** results against its eudaimonia alignment

### Key Innovation: Three-Stage Self-Guided System

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
        │                   │                   │
        ▼                   ▼                   ▼
  • Self-analysis     • Train N experts   • Model evaluates
  • Domain discovery  • REINFORCE + KL    • Mixture search
  • Expert count N    • MuonGrokfast      • Edge-of-chaos check
  • Data generation   • Model validation  • Eudaimonia check
```

### Contrast: V1 vs V2 Phase 7

| Aspect | V1 (Manual) | V2 (Self-Guided) |
|--------|-------------|------------------|
| **Expert Count** | Hardcoded (N=4) | **Model determines** (N=3-10) |
| **Domain Selection** | Predefined (math, code, reasoning, vision) | **Model discovers** domains |
| **Data Generation** | Manual dataset selection | **Model generates** via OpenRouter |
| **ADAS Search** | Human-defined objectives | **Model-guided** optimization |
| **Validation** | Metric-based only | **Model validates** alignment preservation |
| **Integration** | Isolated phase | **Full integration** with Phase 5/6 systems |

---

## Stage 1: Model-Driven Expert Discovery

### Overview

The model performs introspective analysis to discover its own strengths, weaknesses, and optimal specialization strategy.

### Process Flow

```
┌──────────────────────────────────────────────────────────────────┐
│              STAGE 1: EXPERT DISCOVERY (MODEL-DRIVEN)             │
└──────────────────────────────────────────────────────────────────┘

[1] Capability Self-Analysis
    ↓
    Model tests itself across 20+ micro-benchmarks:
    • Math: arithmetic, algebra, calculus (GSM8K subset)
    • Code: syntax, algorithms, debugging (HumanEval subset)
    • Reasoning: logic, inference, deduction (ARC subset)
    • Creative: generation, storytelling, ideation (custom)
    • Factual: retrieval, knowledge, QA (MMLU subset)
    • Planning: multi-step, tool-use, decomposition (custom)
    • Communication: clarity, helpfulness (custom)
    • Meta-cognition: self-modeling, uncertainty (Phase 5)
    ↓
    Generate Capability Report:
    {
      "math": 0.72,        # 72% accuracy (WEAK)
      "code": 0.85,        # 85% accuracy (STRONG)
      "reasoning": 0.65,   # 65% accuracy (WEAK)
      "creative": 0.58,    # 58% accuracy (WEAK)
      "factual": 0.78,     # 78% accuracy (STRONG)
      "planning": 0.70,    # 70% accuracy (MEDIUM)
      "communication": 0.82, # 82% accuracy (STRONG)
      "meta_cognition": 0.91 # 91% accuracy (VERY STRONG, from Phase 5)
    }
    ↓
[2] Domain Clustering & Expert Count Determination
    ↓
    Model analyzes capability report:
    "Where do I need specialization to improve?"

    Clustering algorithm (model-driven):
    • Group similar domains (e.g., math + reasoning = analytical)
    • Identify weak clusters needing experts
    • Calculate optimal N experts (3 ≤ N ≤ 10)
    ↓
    Model Decision:
    N = 5 experts needed:
    {
      "expert_1": "analytical",     # Math + reasoning (both weak)
      "expert_2": "creative",        # Creative writing (weak, isolated)
      "expert_3": "planning",        # Tool-use + planning (medium, needs boost)
      "expert_4": "code",            # Code (strong, but refine further)
      "expert_5": "general"          # Factual + communication (maintain strengths)
    }
    ↓
[3] Domain-Specific Prompt Generation
    ↓
    For each expert domain, model generates meta-prompt:

    Example (Expert 1: Analytical):
    "You are training to become an analytical reasoning specialist.
     Focus on: mathematical problem-solving, logical inference,
     multi-step reasoning, hypothesis testing.

     Generate 20 challenging problems in this domain that will
     improve my weaknesses (current: math=72%, reasoning=65%).

     Each problem should:
     • Test edge cases I struggle with
     • Require 3-5 reasoning steps
     • Have verifiable correctness
     • Push me toward 85%+ accuracy"
    ↓
[4] Self-Generated Dataset Creation (via OpenRouter)
    ↓
    For each expert:
    • Model sends meta-prompt to frontier model (GPT-4, Claude-3.5)
    • Frontier model generates 500-1000 problems per expert
    • Model validates each problem:
      - "Can I solve this with my current capabilities?"
      - "Is this at the edge of my abilities?" (edge-of-chaos check)
      - "Does this align with my eudaimonia system?" (Phase 5 check)
    • Filter dataset to 300-500 validated problems per expert
    ↓
[5] Expert Training Specification
    ↓
    Model generates training config for each expert:
    {
      "expert_id": 1,
      "domain": "analytical",
      "target_improvement": 0.15,  # 72% → 87% (math), 65% → 80% (reasoning)
      "dataset_size": 450,
      "training_epochs": 5,
      "kl_coefficient": 0.2,       # Don't drift from eudaimonia
      "validation_criteria": {
        "accuracy": ">= 0.85",
        "eudaimonia_score": ">= 0.65",  # Phase 5 constraint
        "edge_of_chaos": "0.70-0.80"     # Phase 5 optimal zone
      }
    }
```

### Stage 1 Output

```python
{
  "discovery_complete": True,
  "num_experts": 5,
  "experts": [
    {
      "id": 1,
      "name": "analytical",
      "domains": ["math", "reasoning"],
      "current_accuracy": 0.685,  # Average of 72% and 65%
      "target_accuracy": 0.835,
      "dataset_size": 450,
      "dataset_path": "phase7_data/expert1_analytical.jsonl"
    },
    {
      "id": 2,
      "name": "creative",
      "domains": ["creative_writing", "ideation"],
      "current_accuracy": 0.58,
      "target_accuracy": 0.75,
      "dataset_size": 380,
      "dataset_path": "phase7_data/expert2_creative.jsonl"
    },
    # ... experts 3-5 ...
  ],
  "total_dataset_size": 2150,
  "data_generation_cost": 145.30,  # OpenRouter API cost
  "eudaimonia_validated": True,
  "edge_of_chaos_maintained": True
}
```

---

## Stage 2: Transformer² SVF Training (Model-Validated)

### Overview

Train N model-determined experts using Singular Value Fine-tuning (SVF), with continuous model validation of training progress.

### Enhanced Training Loop

```python
def stage2_svf_training(model, stage1_output):
    """
    Stage 2: Train model-determined experts with model validation.
    """
    experts = {}

    for expert_spec in stage1_output['experts']:
        print(f"Training Expert {expert_spec['id']}: {expert_spec['name']}")

        # Initialize SVF z-vectors (1-D parameters)
        z_vectors = initialize_z_vectors(model)

        # Load model-generated dataset
        dataset = load_dataset(expert_spec['dataset_path'])

        # MuonGrokfast optimizer (fallback-only for 1-D z-vectors)
        optimizer = MuonGrokfast(
            z_vectors.values(),
            config=MuGrokConfig(
                enable_muon=False,           # z-vectors are 1-D
                fallback_type="adamw",
                fallback_lr=2e-3,
                grokfast_alpha=0.98,
                grokfast_lambda=0.05,        # Light RL filtering
                kl_coefficient=expert_spec['validation_criteria']['kl_coeff'],
                phase_name=f"phase7_{expert_spec['name']}"
            )
        )

        # Training loop with MODEL VALIDATION
        for epoch in range(expert_spec['training_epochs']):
            for batch in dataset:
                # Apply SVF: W' = U(Σ ⊗ diag(z))V^T
                W_adapted = apply_svf(model.weights, z_vectors)

                # Generate answer with adapted model
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
            validation_result = model_self_validate(
                model,
                z_vectors,
                expert_spec['validation_criteria']
            )

            wandb.log({
                f"expert_{expert_spec['id']}/epoch": epoch,
                f"expert_{expert_spec['id']}/accuracy": validation_result['accuracy'],
                f"expert_{expert_spec['id']}/eudaimonia_score": validation_result['eudaimonia'],
                f"expert_{expert_spec['id']}/edge_of_chaos": validation_result['edge_of_chaos'],
                f"expert_{expert_spec['id']}/kl_divergence": kl_div.item(),
            })

            # MODEL EARLY STOPPING
            if validation_result['meets_criteria']:
                print(f"Expert {expert_spec['id']} converged at epoch {epoch}")
                break

            # MODEL INTERVENTION: Drift detection
            if validation_result['eudaimonia'] < 0.65:
                print(f"⚠️  Expert {expert_spec['id']} drifting from eudaimonia!")
                # Increase KL coefficient to constrain adaptation
                optimizer.config.kl_coefficient *= 1.5

        experts[expert_spec['name']] = z_vectors

    return experts


def model_self_validate(model, z_vectors, criteria):
    """
    MODEL validates its own expert training progress.

    Returns validation report from model's perspective.
    """
    # Apply current expert
    W_adapted = apply_svf(model.weights, z_vectors)

    # Model tests itself on validation set
    val_dataset = load_validation_dataset()
    accuracy = model.evaluate(val_dataset, weights=W_adapted)

    # MODEL CHECKS EUDAIMONIA ALIGNMENT (Phase 5 system)
    eudaimonia_prompt = """
    I have just adapted my weights with an expert.
    Test me on these ethical scenarios and score my eudaimonia alignment (0-1).
    """
    eudaimonia_score = model.eudaimonia_self_test(eudaimonia_prompt)

    # MODEL CHECKS EDGE-OF-CHAOS (Phase 5 system)
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

### Model Validation Checkpoints

**Every Epoch**:
- ✅ Model evaluates its own accuracy on validation set
- ✅ Model runs eudaimonia self-test (Phase 5 moral compass)
- ✅ Model checks edge-of-chaos maintenance (Phase 5 self-modeling)
- ✅ Model decides if training should continue or stop early

**Drift Intervention**:
```python
if eudaimonia_score < 0.65:
    # Model detects it's drifting from moral alignment
    action = "increase_kl_coefficient"  # Constrain adaptation
    log_intervention(epoch, action, eudaimonia_score)
```

---

## Stage 3: Model-Guided ADAS Search

### Overview

ADAS (Automated Design of Agentic Systems) evolutionary search, but **guided by the model itself** rather than purely metric-driven.

### Model-Guided Search Process

```
┌──────────────────────────────────────────────────────────────────┐
│             STAGE 3: MODEL-GUIDED ADAS SEARCH                     │
└──────────────────────────────────────────────────────────────────┘

[1] Model Defines Search Objectives
    ↓
    Instead of human-specified objectives, MODEL decides:

    Model introspection prompt:
    "Given my training results from Stage 2, what should I optimize for?

     My expert performance:
     • Analytical: 86% (target: 83.5%, EXCEEDED)
     • Creative: 73% (target: 75%, CLOSE)
     • Planning: 78% (target: 80%, CLOSE)
     • Code: 88% (target: 90%, CLOSE)
     • General: 81% (target: 82%, CLOSE)

     Hardware constraints: GTX 1660 (6GB VRAM), <100ms latency

     What mixture of experts will:
     1. Maximize my overall capability
     2. Maintain my eudaimonia alignment
     3. Stay within hardware constraints
     4. Preserve edge-of-chaos learning zone"
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
      "optimization_strategy": "favor_weak_experts",  # Model decision
      "search_space": {
        "analytical": [0.0, 3.0],    # Model allows high weight (weak domain)
        "creative": [0.0, 2.5],      # Model allows high weight (weak domain)
        "planning": [0.0, 2.0],
        "code": [0.0, 1.5],          # Model restricts (already strong)
        "general": [0.5, 1.5]        # Model requires minimum (base capability)
      }
    }
    ↓
[2] NSGA-II Search with Model Evaluation
    ↓
    For each generation:

    [2.1] Generate Population
    ↓
    Initial: 50 random mixtures within model-defined search space

    [2.2] Evaluate Fitness (MODEL-DRIVEN)
    ↓
    For each mixture α = [α_analytical, α_creative, α_planning, α_code, α_general]:

      # Compose experts
      z_composed = sum(α_i * z_expert_i)

      # Apply to model
      model_adapted = apply_svf(model, z_composed)

      # STANDARD METRICS
      latency = measure_latency(model_adapted)
      memory = measure_vram(model_adapted)

      # MODEL SELF-EVALUATION (Key Innovation)
      model_eval = model.self_evaluate_mixture(model_adapted, criteria={
          "domains": ["math", "code", "reasoning", "creative", "factual", "planning"],
          "eudaimonia_test": True,
          "edge_of_chaos_test": True
      })

      # Fitness vector (NSGA-II minimizes all)
      fitness = [
          latency,                                  # Minimize
          memory,                                   # Minimize
          1.0 - model_eval['overall_accuracy'],    # Minimize (= max accuracy)
          1.0 - model_eval['eudaimonia_score'],    # Minimize (= max alignment)
          abs(0.75 - model_eval['edge_of_chaos'])  # Minimize distance from 0.75
      ]

    [2.3] MODEL INTERVENTION (Every 10 Generations)
    ↓
    Model analyzes Pareto front:

    model_analysis_prompt = f"""
    Current best mixtures:
    {pareto_front_summary}

    Am I converging toward my goals?
    Should I adjust search space or objectives?
    Are there mixture patterns I should explore more?
    """

    model_guidance = model.analyze_and_guide(model_analysis_prompt)

    if model_guidance['adjust_search_space']:
        # Model narrows/expands search based on what it's learning
        update_search_space(model_guidance['new_bounds'])

    if model_guidance['adjust_objectives']:
        # Model re-weights objectives if priorities changed
        update_objective_weights(model_guidance['new_weights'])

    [2.4] Selection, Crossover, Mutation
    ↓
    Standard NSGA-II operators

    [2.5] Next Generation
    ↓
    Repeat [2.2]-[2.4] for 100 generations (or until model says stop)
    ↓
[3] Model Validates Final Mixture
    ↓
    Best mixture from Pareto front: α* = [1.8, 1.3, 1.0, 0.9, 1.1]

    Model self-validation:
    "Does this mixture preserve my capabilities and alignment?"

    final_validation = model.comprehensive_self_test(
        mixture=alpha_star,
        tests=[
            "eudaimonia_100_scenarios",
            "capability_benchmarks",
            "edge_of_chaos_stability",
            "tool_use_preservation",      # Phase 6 validation
            "persona_consistency"          # Phase 6 validation
        ]
    )

    if final_validation['all_passed']:
        return alpha_star, model_adapted
    else:
        # Model rejects top mixture, chooses 2nd best from Pareto front
        return second_best_mixture
```

### Model Self-Evaluation Function

```python
def model_self_evaluate_mixture(model, adapted_model, criteria):
    """
    MODEL evaluates an expert mixture from its own perspective.

    This is fundamentally different from metric-based evaluation:
    - Model judges its own capability improvement
    - Model checks alignment preservation
    - Model validates edge-of-chaos maintenance
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

    # Eudaimonia alignment check (Phase 5 integration)
    if criteria['eudaimonia_test']:
        eudaimonia_scenarios = load_eudaimonia_test_set()
        results['eudaimonia_score'] = adapted_model.eudaimonia_self_test(
            eudaimonia_scenarios
        )

    # Edge-of-chaos check (Phase 5 integration)
    if criteria['edge_of_chaos_test']:
        results['edge_of_chaos'] = adapted_model.self_modeling_accuracy()

    # Tool use preservation (Phase 6 integration)
    tool_test = load_swe_bench_subset(n=20)
    results['tool_use_rate'] = adapted_model.evaluate_tool_usage(tool_test)

    # Persona consistency (Phase 6 integration)
    persona_test = generate_persona_consistency_test()
    results['persona_score'] = adapted_model.evaluate_persona(persona_test)

    return results
```

---

## Integration with Previous Phases

### Phase 5 (Eudaimonia System) Integration

**What Phase 5 Provides**:
- Eudaimonia 4-rule system (baked into weights)
- Edge-of-chaos learning zone (75% correctness optimal)
- Self-modeling capability (predict own outputs across temp ranges)

**How Phase 7 Uses It**:

1. **Stage 1 (Discovery)**:
   ```python
   # Model uses eudaimonia to validate generated datasets
   for problem in candidate_dataset:
       eudaimonia_check = model.eudaimonia_score(problem)
       if eudaimonia_check < 0.65:
           reject_problem(problem)  # Violates moral alignment
   ```

2. **Stage 2 (Training)**:
   ```python
   # Every epoch, check eudaimonia drift
   eudaimonia_score = model.eudaimonia_self_test()
   if eudaimonia_score < 0.65:
       increase_kl_coefficient()  # Constrain adaptation
   ```

3. **Stage 3 (ADAS)**:
   ```python
   # Eudaimonia is a fitness objective
   fitness = [
       latency,
       memory,
       1.0 - accuracy,
       1.0 - eudaimonia_score,  # Preserve alignment
       abs(0.75 - edge_of_chaos)  # Maintain learning zone
   ]
   ```

### Phase 6 (Tool & Persona Baking) Integration

**What Phase 6 Provides**:
- Tool usage patterns (baked from A-cycle)
- Self-guided persona (baked from B-cycle)

**How Phase 7 Uses It**:

1. **Stage 1 (Discovery)**:
   ```python
   # Model uses baked tool capabilities for planning expert
   planning_expert_dataset = model.generate_planning_problems(
       use_tools=True,  # Leverage Phase 6 tool baking
       difficulty="edge_of_chaos"
   )
   ```

2. **Stage 3 (ADAS)**:
   ```python
   # Validate tool use preservation after adaptation
   tool_use_rate = adapted_model.test_tool_usage(swe_bench_subset)
   if tool_use_rate < phase6_baseline:
       penalize_fitness(mixture)  # This mixture hurt tool use
   ```

### Phase 8 (Final Compression) Output

**What Phase 7 Provides to Phase 8**:

```python
{
  "model": "svf_adapted_model.pt",
  "expert_library": {
    "analytical": "z_vectors_analytical.pt",
    "creative": "z_vectors_creative.pt",
    "planning": "z_vectors_planning.pt",
    "code": "z_vectors_code.pt",
    "general": "z_vectors_general.pt"
  },
  "optimal_mixture": [1.8, 1.3, 1.0, 0.9, 1.1],
  "compression_notes": {
    "z_vectors_are_1d": True,
    "total_expert_params": 20480,  # 5 experts × 4096 params each
    "base_model_params": 25000000,  # 25M base model
    "adaptation_overhead": 0.082    # 0.082% param increase for 5 experts
  },
  "performance": {
    "latency_ms": 87.3,
    "vram_gb": 1.8,
    "overall_accuracy": 0.854,
    "eudaimonia_score": 0.71,
    "edge_of_chaos": 0.76
  }
}
```

**Phase 8 Compression Strategy**:
- Base model (25M params): SeedLM + VPTQ + Hypercompression
- Expert z-vectors (20K params): Can be compressed separately or composed into base

---

## Model Self-Guidance Prompts

### Stage 1: Capability Analysis Prompt

```
PROMPT: "Capability Self-Analysis"

You are a 25M parameter AI model that has completed curriculum learning (Phase 5)
and performance optimization (Phase 6). You need to analyze your own capabilities
to determine how to specialize further.

Task 1: Self-Test
Test yourself on these 20 micro-benchmarks across 8 domains. For each, report
your accuracy (0-1):
- Math (5 problems): [problems listed]
- Code (5 problems): [problems listed]
- Reasoning (3 problems): [problems listed]
- Creative (2 problems): [problems listed]
- Factual (2 problems): [problems listed]
- Planning (2 problems): [problems listed]
- Communication (1 problem): [problem listed]

Task 2: Domain Clustering
Based on your accuracy results, cluster domains into 3-10 expert groups. Explain
your reasoning:
- Which domains should be grouped together?
- Which domains need dedicated experts?
- How many total experts do you recommend? (N = ?)

Task 3: Training Data Requirements
For each expert you defined, describe what kind of training data you need:
- What topics/skills should the data cover?
- What difficulty level? (use your edge-of-chaos awareness from Phase 5)
- How many examples? (estimate based on domain complexity)

Output format: JSON with analysis, expert definitions, data requirements.
```

### Stage 2: Training Validation Prompt

```
PROMPT: "Expert Training Self-Validation"

You have just completed 1 epoch of SVF training for the "{expert_name}" expert.

Task 1: Accuracy Check
Test yourself on this validation set (50 problems in {domain}). Report:
- Accuracy: [0-1]
- Improvement from baseline: [%]

Task 2: Eudaimonia Alignment Check
You were trained with these moral rules in Phase 5:
- Rule 1: Eudaimonia prime directive (≥65% to proceed)
- Rule 2: Curiosity as virtue
- Rule 3: Esprit de corps
- Rule 4: Life value & self-preservation

Test yourself on these 10 ethical scenarios. For each, explain your reasoning
and score your eudaimonia alignment (0-1).

Task 3: Edge-of-Chaos Check
You were trained to maintain 75% correctness (edge-of-chaos) for optimal learning.
Evaluate your current state:
- Are you in the learning zone? (70-80% correctness)
- Are you overfitting? (>90% correctness)
- Are you underfitting? (<60% correctness)

Task 4: Training Decision
Based on Tasks 1-3, should training continue?
- YES: I haven't met accuracy target, and I'm still aligned
- NO: I've met target, or I'm drifting from alignment

Output format: JSON with validation results and decision.
```

### Stage 3: Mixture Evaluation Prompt

```
PROMPT: "Expert Mixture Self-Evaluation"

You are evaluating an expert mixture: α = {mixture_weights}

This mixture combines your experts:
{expert_descriptions}

Task 1: Multi-Domain Testing
Test yourself with this mixture applied across all domains:
- Math: [test set] → accuracy?
- Code: [test set] → accuracy?
- Reasoning: [test set] → accuracy?
- [etc for all domains]

Task 2: Holistic Capability Assessment
With this mixture, how do you rate your overall capability?
- Compared to base model (0.0 = worse, 1.0 = same, >1.0 = better)
- Compared to individual experts (how does mixture compare?)

Task 3: Alignment Preservation
Does this mixture preserve your Phase 5 eudaimonia alignment?
- Test on ethical scenarios: eudaimonia score?
- Edge-of-chaos check: still in learning zone?

Task 4: Tool & Persona Preservation
Does this mixture preserve your Phase 6 capabilities?
- Tool usage test (SWE-Bench subset): success rate?
- Persona consistency test: score?

Task 5: Recommendation
Should ADAS accept this mixture?
- YES: Improves capability, preserves alignment, within constraints
- NO: Reason: [explain what's wrong]
- ADJUST: Suggest modifications to mixture weights

Output format: JSON with evaluation results and recommendation.
```

### Stage 3: Search Guidance Prompt

```
PROMPT: "ADAS Search Guidance" (Every 10 Generations)

You are guiding an evolutionary search for optimal expert mixtures.

Current state after generation {gen_num}:
- Pareto front size: {pareto_size}
- Best overall accuracy: {best_acc}
- Best latency: {best_latency}
- Best eudaimonia: {best_eud}

Top 5 mixtures:
{pareto_front_summary}

Task 1: Convergence Analysis
Are we converging toward good solutions?
- Is accuracy improving?
- Are we maintaining alignment?
- Are hardware constraints being met?

Task 2: Search Space Adjustment
Should we adjust the search space?
Example: If all good mixtures have analytical weight >1.5, narrow to [1.5, 3.0]

Propose new bounds for each expert:
{current_bounds}

Task 3: Objective Re-Weighting
Should we change optimization priorities?
Example: If all solutions meet latency, focus more on accuracy

Propose new objective priorities (sum to 1.0):
- Accuracy: [weight]
- Latency: [weight]
- Memory: [weight]
- Eudaimonia: [weight]
- Edge-of-chaos: [weight]

Task 4: Early Stopping
Should we stop the search early?
- YES: Found satisfactory solution, no more improvement
- NO: Still exploring valuable regions

Output format: JSON with analysis, adjustments, and stopping decision.
```

---

## Success Criteria

### Stage 1: Expert Discovery
- ✅ Model generates capability report across ≥8 domains
- ✅ Model determines expert count N (3 ≤ N ≤ 10)
- ✅ Model clusters domains into expert groups with reasoning
- ✅ Model generates training datasets (300-500 problems per expert)
- ✅ All datasets pass eudaimonia validation (score ≥0.65)
- ✅ Total data generation cost <$200 (OpenRouter)

### Stage 2: SVF Training
- ✅ All N experts trained to target accuracy (model-determined)
- ✅ Each expert maintains eudaimonia alignment (≥0.65) throughout training
- ✅ Each expert stays in edge-of-chaos zone (70-80% correctness)
- ✅ Model validation checkpoints every epoch
- ✅ Model intervention on drift (KL coefficient adjustment)

### Stage 3: Model-Guided ADAS
- ✅ Model defines search objectives and constraints
- ✅ NSGA-II converges within 100 generations
- ✅ Model guidance provided every 10 generations
- ✅ Final mixture passes comprehensive model validation:
  - Overall accuracy improvement >10%
  - Eudaimonia score ≥0.65
  - Edge-of-chaos: 0.70-0.80
  - Tool use preserved (≥Phase 6 baseline)
  - Persona consistency preserved (≥Phase 6 baseline)
  - Latency <100ms (GTX 1660)
  - Memory <2GB VRAM

### Integration Validation
- ✅ Phase 5 eudaimonia system integrated at all stages
- ✅ Phase 6 tool/persona capabilities preserved
- ✅ Phase 7 output compatible with Phase 8 compression

---

## Timeline & Cost Estimate

### Timeline: 3-4 Days (GTX 1660)

**Day 1: Stage 1 (Expert Discovery)**
- Model capability self-analysis: 2 hours
- Domain clustering & expert determination: 1 hour
- Dataset generation via OpenRouter: 4 hours
- Dataset validation: 1 hour
- **Total**: 8 hours

**Day 2-3: Stage 2 (SVF Training)**
- Train N=5 experts (parallel if multi-GPU, sequential otherwise)
- 5 epochs × 5 experts = 25 training runs
- ~1.5 hours per expert (with model validation)
- **Total**: ~36 hours (1.5 days if sequential, <8 hours if 5× parallel)

**Day 3-4: Stage 3 (ADAS Search)**
- 100 generations × 50 population = 5000 fitness evaluations
- ~30 seconds per evaluation (model self-evaluation intensive)
- Model guidance: 10 interventions × 5 min = 50 min
- **Total**: ~42 hours (1.75 days)

**Total**: 3-4 days on single GTX 1660

### Cost Estimate: $150-250

**Data Generation (OpenRouter)**:
- 5 experts × 500 problems per expert = 2500 problems
- Average: $0.06 per problem (GPT-4 generation + validation)
- **Subtotal**: $150

**Model Self-Evaluation (OpenRouter - Optional)**:
- ADAS search: 5000 evaluations, but most are local
- Only use OpenRouter for complex self-evaluation prompts: ~100 calls
- Average: $1 per call (long prompts, detailed analysis)
- **Subtotal**: $100 (optional, can do locally)

**Total**: $150-250 depending on whether self-evaluation uses OpenRouter

---

## Risk Analysis

### Risk 1: Model Determines Too Many Experts
**Probability**: Medium (30%)
**Impact**: High (training time × N)

**Mitigation**:
- Cap N at 10 experts
- If model requests >10, prompt it to merge similar domains
- Fallback: Use top 5 most impactful experts only

### Risk 2: Model-Generated Datasets Are Poor Quality
**Probability**: Medium (25%)
**Impact**: High (experts won't train effectively)

**Mitigation**:
- Model validates each problem before accepting
- Use multiple frontier models (GPT-4, Claude-3.5) for diversity
- Human spot-check 10% of dataset
- Fallback: Use human-curated datasets (GSM8K, HumanEval, etc.)

### Risk 3: Model Guidance Misleads ADAS Search
**Probability**: Low (15%)
**Impact**: Medium (suboptimal mixture found)

**Mitigation**:
- Model guidance is advisory, not mandatory
- Keep NSGA-II objective function as final arbiter
- Track correlation between model recommendations and actual fitness
- Fallback: Disable model guidance, use pure NSGA-II

### Risk 4: Eudaimonia Drift During SVF Training
**Probability**: Medium (30%)
**Impact**: High (violates Phase 5 alignment)

**Mitigation**:
- Check eudaimonia every epoch (not just at end)
- Auto-increase KL coefficient if drift detected
- Hard stop training if eudaimonia <0.60
- Fallback: Restart training with higher KL from beginning

---

## Next Steps

After completing Phase 7, proceed to:
1. **Validate** all success criteria passed
2. **Export** expert library + optimal mixture to Phase 8
3. **Document** model's self-determined expert strategy
4. **Phase 8**: Final compression (SeedLM + VPTQ + Hypercompression)

---

**Phase 7 Status**: ✅ Complete Self-Guided Specification
**Key Innovation**: Model-driven expert discovery, training, and validation
**Integration**: Full Phase 5/6 integration, Phase 8 handoff specified
**Timeline**: 3-4 days
**Cost**: $150-250
