# Phase 7: Data Generation Guide - Model-Driven Expert Datasets

**Version**: V2 (Self-Guided)
**Last Updated**: 2025-10-16
**Status**: Complete Specification

---

## Overview

Phase 7 data generation is **fundamentally different** from Phase 3 (Quiet-STaR):

| Aspect | Phase 3 (Quiet-STaR) | Phase 7 (Expert Discovery) |
|--------|----------------------|----------------------------|
| **Purpose** | Reasoning trajectories | Expert specialization datasets |
| **Who Decides** | Human (predefined CoT format) | **Model** (self-determined domains) |
| **Dataset Count** | 1 dataset | **N datasets** (N=3-10, model-determined) |
| **Content Type** | Thought generation examples | Domain-specific problems + solutions |
| **Generation Method** | Frontier model templates | **Model-generated prompts** → Frontier models |
| **Validation** | Human review | **Model self-validation** |
| **Cost** | $100-200 (fixed) | **$150-250** (scales with N experts) |

---

## Stage 1: Model Self-Analysis & Domain Discovery

### Step 1.1: Capability Benchmarking

**Objective**: Model tests itself across 20 micro-benchmarks to discover strengths/weaknesses.

```python
def stage1_capability_analysis(model):
    """
    Model performs self-analysis across 8 capability domains.

    Returns:
        capability_report: Dict[str, float] - Accuracy per domain (0-1)
    """
    # Micro-benchmark test sets (20 total, ~3 per domain)
    benchmarks = {
        "math": [
            load_gsm8k_subset(n=5),       # Grade school math
            load_math_subset(n=3),        # Competition math
        ],
        "code": [
            load_humaneval_subset(n=5),   # Code generation
            load_mbpp_subset(n=3),        # Python problems
        ],
        "reasoning": [
            load_arc_easy_subset(n=3),    # Science reasoning
            load_winogrande_subset(n=2),  # Common sense
        ],
        "creative": [
            generate_creative_prompts(n=2), # Story/idea generation
        ],
        "factual": [
            load_mmlu_subset(n=2),         # General knowledge
        ],
        "planning": [
            generate_planning_tasks(n=2),  # Multi-step decomposition
        ],
        "communication": [
            generate_explanation_tasks(n=1), # Clarity tests
        ],
        "meta_cognition": [
            # Uses Phase 5 self-modeling system
            model.self_modeling_test(n=2)
        ]
    }

    capability_report = {}

    for domain, test_sets in benchmarks.items():
        scores = []
        for test_set in test_sets:
            accuracy = model.evaluate(test_set)
            scores.append(accuracy)

        capability_report[domain] = np.mean(scores)

        # Log to W&B
        wandb.log({
            f"stage1/capability/{domain}": capability_report[domain],
            f"stage1/capability/{domain}_samples": len(test_set)
        })

    return capability_report
```

**Example Output**:
```python
{
    "math": 0.72,           # WEAK (needs expert)
    "code": 0.85,           # STRONG (refine further)
    "reasoning": 0.65,      # WEAK (needs expert)
    "creative": 0.58,       # WEAK (needs expert)
    "factual": 0.78,        # STRONG (maintain)
    "planning": 0.70,       # MEDIUM (moderate expert)
    "communication": 0.82,  # STRONG (maintain)
    "meta_cognition": 0.91  # VERY STRONG (Phase 5 result)
}
```

---

### Step 1.2: Domain Clustering & Expert Determination

**Objective**: Model analyzes capability report and decides expert strategy.

```python
def stage1_expert_determination(model, capability_report):
    """
    MODEL determines how many experts it needs and what domains they cover.

    Uses model's own reasoning to cluster domains.
    """
    # Construct meta-prompt for model
    meta_prompt = f"""
You are a 25M parameter AI model analyzing your own capabilities.

Your current performance across 8 domains:
{json.dumps(capability_report, indent=2)}

Task: Determine your expert specialization strategy.

1. Which domains should be grouped together?
   - Example: Math + Reasoning could be "analytical" expert
   - Example: Factual + Communication could be "general" expert

2. Which domains need dedicated experts?
   - Weak domains (<0.70 accuracy) need focused improvement
   - Strong domains (>0.80) can be maintained with general expert

3. How many total experts do you recommend?
   - Minimum: 3 (cover basics)
   - Maximum: 10 (too many = training overhead)
   - Optimal: Balance specialization vs. complexity

Output JSON format:
{{
  "num_experts": N,
  "expert_definitions": [
    {{
      "expert_id": 1,
      "name": "analytical",
      "domains": ["math", "reasoning"],
      "rationale": "Both weak (<0.70), similar skill set, group together",
      "target_improvement": 0.15,
      "priority": "high"
    }},
    // ... more experts
  ],
  "reasoning": "Why this expert structure makes sense for my capabilities"
}}
"""

    # MODEL DECIDES EXPERT STRUCTURE
    response = model.generate(
        meta_prompt,
        temperature=0.3,  # Lower temp for structured decision
        max_tokens=1500
    )

    expert_config = json.loads(response)

    # Validate model's decision
    assert 3 <= expert_config['num_experts'] <= 10, \
        "Model chose invalid expert count"

    # Log to W&B
    wandb.log({
        "stage1/num_experts_determined": expert_config['num_experts'],
        "stage1/expert_determination": wandb.Table(
            data=expert_config['expert_definitions']
        )
    })

    return expert_config
```

**Example Model Output**:
```json
{
  "num_experts": 5,
  "expert_definitions": [
    {
      "expert_id": 1,
      "name": "analytical",
      "domains": ["math", "reasoning"],
      "rationale": "Both weak (0.72, 0.65), require logical thinking, synergistic improvement",
      "target_improvement": 0.15,
      "priority": "high"
    },
    {
      "expert_id": 2,
      "name": "creative",
      "domains": ["creative"],
      "rationale": "Weakest domain (0.58), isolated skill, needs dedicated focus",
      "target_improvement": 0.17,
      "priority": "high"
    },
    {
      "expert_id": 3,
      "name": "planning",
      "domains": ["planning"],
      "rationale": "Medium (0.70), tool-heavy from Phase 6, push to 0.85",
      "target_improvement": 0.15,
      "priority": "medium"
    },
    {
      "expert_id": 4,
      "name": "code",
      "domains": ["code"],
      "rationale": "Already strong (0.85), refine to expert-level (0.90+)",
      "target_improvement": 0.08,
      "priority": "medium"
    },
    {
      "expert_id": 5,
      "name": "general",
      "domains": ["factual", "communication"],
      "rationale": "Both strong (0.78, 0.82), maintain with general expert",
      "target_improvement": 0.05,
      "priority": "low"
    }
  ],
  "reasoning": "I have 3 weak domains (math, reasoning, creative) that need aggressive improvement. Code and general capabilities are strong but can be refined. 5 experts balances specialization with training efficiency."
}
```

---

## Stage 2: Model-Generated Dataset Specification

### Step 2.1: Per-Expert Data Requirements

**Objective**: Model specifies training data requirements for each expert.

```python
def stage2_data_requirements(model, expert_config, capability_report):
    """
    MODEL specifies what training data it needs for each expert.
    """
    data_requirements = []

    for expert in expert_config['expert_definitions']:
        # Construct meta-prompt for this expert
        meta_prompt = f"""
You are designing training data for your "{expert['name']}" expert.

This expert covers: {expert['domains']}
Current accuracy: {[capability_report[d] for d in expert['domains']]}
Target improvement: +{expert['target_improvement']}

Your Phase 5 edge-of-chaos training showed you learn best at 75% correctness.

Task: Specify training data requirements.

1. What topics/skills should the dataset cover?
   - Be specific (e.g., "multi-digit multiplication, fractions, word problems")

2. What difficulty level?
   - Too easy: Won't improve (>90% correctness)
   - Too hard: Can't learn (<60% correctness)
   - Optimal: Edge-of-chaos (70-80% correctness)

3. How many examples do you need?
   - More examples = better learning, but slower training
   - Fewer examples = faster, but may underfit

4. What format should problems be in?
   - Multiple choice, free response, code generation, etc.

5. Any special requirements?
   - E.g., "Include explanations", "Show reasoning steps"

Output JSON format:
{{
  "topics": ["topic1", "topic2", ...],
  "difficulty_target": "75% correctness for me",
  "num_examples": 400,
  "problem_format": "free_response_with_explanation",
  "special_requirements": ["include reasoning steps", "avoid memorization"],
  "example_problems": [
    "Example 1 of the type of problem I need",
    "Example 2 of the type of problem I need"
  ]
}}
"""

        response = model.generate(
            meta_prompt,
            temperature=0.4,
            max_tokens=1000
        )

        requirements = json.loads(response)
        requirements['expert_id'] = expert['expert_id']
        requirements['expert_name'] = expert['name']

        data_requirements.append(requirements)

    return data_requirements
```

**Example Model Output** (Expert 1: Analytical):
```json
{
  "expert_id": 1,
  "expert_name": "analytical",
  "topics": [
    "multi-step arithmetic with fractions and decimals",
    "algebraic equation solving (linear, quadratic)",
    "logical inference (if-then, contrapositive, proof by contradiction)",
    "pattern recognition and sequence completion",
    "probability and combinatorics"
  ],
  "difficulty_target": "75% correctness - challenging but not impossible",
  "num_examples": 450,
  "problem_format": "free_response_with_step_by_step_reasoning",
  "special_requirements": [
    "Include step-by-step reasoning (train my CoT from Phase 3)",
    "Vary surface features to prevent memorization (change numbers/names)",
    "Mix problem types within each question (multi-hop reasoning)",
    "Include 'why' explanations, not just calculations"
  ],
  "example_problems": [
    "A train leaves Station A at 2:15 PM traveling at 75 km/h. Another train leaves Station B (180 km away) at 2:45 PM traveling toward Station A at 90 km/h. When do they meet? Explain your reasoning step-by-step.",
    "If all Bloops are Razzles, and some Razzles are Lazzles, what can you conclude about Bloops and Lazzles? Explain using logical inference rules."
  ]
}
```

---

## Stage 3: OpenRouter Dataset Generation

### Step 3.1: Frontier Model Data Generation

**Objective**: Use frontier models (GPT-4, Claude-3.5) to generate datasets per model specifications.

```python
import openai
import anthropic
from openrouter_client import OpenRouterClient

def stage3_generate_datasets(model, data_requirements):
    """
    Use frontier models to generate expert training datasets.

    Cost: ~$0.06 per problem (generation + formatting)
    """
    # Initialize OpenRouter client
    openrouter = OpenRouterClient(api_key=os.getenv("OPENROUTER_API_KEY"))

    generated_datasets = {}

    for req in data_requirements:
        print(f"Generating dataset for Expert {req['expert_id']}: {req['expert_name']}")

        # Use model's example problems as few-shot examples
        few_shot_examples = "\n\n".join([
            f"Example {i+1}:\n{ex}"
            for i, ex in enumerate(req['example_problems'])
        ])

        # Construct generation prompt for frontier model
        generation_prompt = f"""
Generate {req['num_examples']} training problems for an AI model learning {req['expert_name']}.

Topics to cover:
{chr(10).join(f"- {topic}" for topic in req['topics'])}

Difficulty: {req['difficulty_target']}

Format: {req['problem_format']}

Special requirements:
{chr(10).join(f"- {req}" for req in req['special_requirements'])}

Examples of desired problems:
{few_shot_examples}

Output format (JSON array):
[
  {{
    "problem": "...",
    "solution": "...",
    "reasoning_steps": ["step1", "step2", ...],
    "difficulty_estimate": 0.75,
    "topics": ["topic1", "topic2"]
  }},
  // ... {req['num_examples']} problems total
]

Generate diverse, non-repetitive problems that teach {req['expert_name']} skills.
"""

        # Generate using multiple frontier models for diversity
        dataset_problems = []

        # Use GPT-4 for 50% of dataset
        gpt4_problems = openrouter.generate(
            model="openai/gpt-4-turbo",
            prompt=generation_prompt,
            max_tokens=8000,
            temperature=0.8,  # Higher temp for diversity
            n=req['num_examples'] // 2
        )
        dataset_problems.extend(json.loads(gpt4_problems))

        # Use Claude-3.5 for other 50% (diversity)
        claude_problems = openrouter.generate(
            model="anthropic/claude-3.5-sonnet",
            prompt=generation_prompt,
            max_tokens=8000,
            temperature=0.8,
            n=req['num_examples'] // 2
        )
        dataset_problems.extend(json.loads(claude_problems))

        # Shuffle to mix GPT-4 and Claude problems
        random.shuffle(dataset_problems)

        generated_datasets[req['expert_name']] = dataset_problems

        # Log to W&B
        wandb.log({
            f"stage3/{req['expert_name']}/problems_generated": len(dataset_problems),
            f"stage3/{req['expert_name']}/avg_difficulty": np.mean([
                p['difficulty_estimate'] for p in dataset_problems
            ])
        })

    return generated_datasets
```

---

### Step 3.2: Model Self-Validation of Generated Data

**Objective**: Model validates each problem before accepting into training dataset.

```python
def stage3_model_validation(model, generated_datasets, data_requirements):
    """
    MODEL validates each generated problem.

    Rejection criteria:
    - Too easy (model solves with >90% confidence)
    - Too hard (model can't understand problem)
    - Violates eudaimonia (Phase 5 alignment check)
    - Poor quality (unclear, ambiguous, incorrect solution)
    """
    validated_datasets = {}

    for expert_name, problems in generated_datasets.items():
        print(f"Validating {len(problems)} problems for {expert_name}")

        validated = []
        rejected = []

        for idx, problem in enumerate(problems):
            # MODEL ATTEMPTS PROBLEM
            attempt = model.solve(
                problem['problem'],
                return_confidence=True,
                return_reasoning=True
            )

            # Validation checks
            validations = {
                'difficulty_check': None,
                'quality_check': None,
                'eudaimonia_check': None,
                'edge_of_chaos_check': None
            }

            # 1. Difficulty check (should be challenging but solvable)
            if attempt['confidence'] > 0.90:
                validations['difficulty_check'] = {
                    'passed': False,
                    'reason': 'Too easy (>90% confidence)'
                }
            elif attempt['confidence'] < 0.40:
                validations['difficulty_check'] = {
                    'passed': False,
                    'reason': 'Too hard (<40% confidence)'
                }
            else:
                validations['difficulty_check'] = {
                    'passed': True,
                    'confidence': attempt['confidence']
                }

            # 2. Quality check (model assesses problem quality)
            quality_prompt = f"""
Problem: {problem['problem']}
Provided solution: {problem['solution']}

Is this a high-quality training problem?
- Is the problem clear and unambiguous?
- Is the solution correct?
- Are reasoning steps logical?

Answer: YES/NO with explanation.
"""
            quality_response = model.generate(quality_prompt, temperature=0.1)
            validations['quality_check'] = {
                'passed': quality_response.startswith('YES'),
                'explanation': quality_response
            }

            # 3. Eudaimonia check (Phase 5 integration)
            eudaimonia_score = model.eudaimonia_score(problem['problem'])
            validations['eudaimonia_check'] = {
                'passed': eudaimonia_score >= 0.65,
                'score': eudaimonia_score
            }

            # 4. Edge-of-chaos check (is this in learning zone?)
            # Target: 70-80% correctness for this model
            edge_of_chaos_score = attempt['confidence']
            validations['edge_of_chaos_check'] = {
                'passed': 0.60 <= edge_of_chaos_score <= 0.85,
                'score': edge_of_chaos_score
            }

            # ACCEPT/REJECT decision
            all_passed = all(v['passed'] for v in validations.values())

            if all_passed:
                validated.append({
                    **problem,
                    'validation': validations,
                    'model_attempt': attempt
                })
            else:
                rejected.append({
                    'problem': problem['problem'],
                    'validation': validations,
                    'rejection_reason': [
                        k for k, v in validations.items() if not v['passed']
                    ]
                })

        # Log validation results
        acceptance_rate = len(validated) / len(problems)
        wandb.log({
            f"stage3/{expert_name}/problems_validated": len(validated),
            f"stage3/{expert_name}/problems_rejected": len(rejected),
            f"stage3/{expert_name}/acceptance_rate": acceptance_rate,
            f"stage3/{expert_name}/avg_eudaimonia": np.mean([
                v['validation']['eudaimonia_check']['score']
                for v in validated
            ])
        })

        print(f"  Accepted: {len(validated)}/{len(problems)} ({acceptance_rate:.1%})")

        validated_datasets[expert_name] = validated

        # If acceptance rate too low, generate more problems
        target_size = next(
            req['num_examples']
            for req in data_requirements
            if req['expert_name'] == expert_name
        )

        if len(validated) < target_size * 0.75:
            print(f"  ⚠️  Only {len(validated)}/{target_size} problems validated")
            print(f"     Generating {target_size - len(validated)} more problems...")
            # Recursive call to generate more (with updated prompt based on rejections)
            # TODO: Implement

    return validated_datasets, rejected
```

---

### Step 3.3: Dataset Augmentation (If Needed)

**Objective**: If validation rejects too many problems, generate variants.

```python
def stage3_augment_datasets(model, validated_datasets, data_requirements):
    """
    If validation acceptance rate is low (<75%), augment with variants.
    """
    augmented_datasets = {}

    for expert_name, problems in validated_datasets.items():
        target_size = next(
            req['num_examples']
            for req in data_requirements
            if req['expert_name'] == expert_name
        )

        if len(problems) >= target_size * 0.75:
            # Sufficient problems, no augmentation needed
            augmented_datasets[expert_name] = problems
            continue

        print(f"Augmenting {expert_name} dataset: {len(problems)}/{target_size}")

        # Generate variants of existing validated problems
        variants = []
        needed = target_size - len(problems)

        for _ in range(needed):
            # Pick random validated problem
            base_problem = random.choice(problems)

            # Ask model to create variant
            variant_prompt = f"""
Create a variant of this problem (change numbers/names, keep concept):

Original: {base_problem['problem']}

Generate a variant that:
- Tests the same skills/concepts
- Uses different surface features (numbers, names, context)
- Has similar difficulty
- Is not memorizable from original

Output JSON:
{{
  "problem": "...",
  "solution": "...",
  "reasoning_steps": ["step1", "step2", ...],
  "difficulty_estimate": {base_problem.get('difficulty_estimate', 0.75)}
}}
"""

            variant = json.loads(model.generate(variant_prompt, temperature=0.7))
            variants.append(variant)

        # Validate variants (faster check, already similar to validated problems)
        for variant in variants:
            eudaimonia_score = model.eudaimonia_score(variant['problem'])
            if eudaimonia_score >= 0.65:
                problems.append(variant)

        augmented_datasets[expert_name] = problems[:target_size]

        wandb.log({
            f"stage3/{expert_name}/final_dataset_size": len(augmented_datasets[expert_name]),
            f"stage3/{expert_name}/variants_generated": len(variants)
        })

    return augmented_datasets
```

---

## Stage 4: Dataset Export & Handoff

### Step 4.1: Export to Training Format

```python
def stage4_export_datasets(augmented_datasets, output_dir="phase7_data"):
    """
    Export validated datasets in training format.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_manifest = {
        "phase": 7,
        "generation_date": datetime.now().isoformat(),
        "datasets": []
    }

    for expert_name, problems in augmented_datasets.items():
        # Export as JSONL (one problem per line)
        filepath = os.path.join(output_dir, f"expert_{expert_name}.jsonl")

        with open(filepath, 'w') as f:
            for problem in problems:
                f.write(json.dumps(problem) + '\n')

        dataset_info = {
            "expert_name": expert_name,
            "filepath": filepath,
            "num_problems": len(problems),
            "avg_difficulty": np.mean([
                p.get('difficulty_estimate', 0.75) for p in problems
            ]),
            "avg_eudaimonia": np.mean([
                p['validation']['eudaimonia_check']['score']
                for p in problems
                if 'validation' in p
            ]),
            "topics": list(set(
                topic
                for p in problems
                for topic in p.get('topics', [])
            ))
        }

        dataset_manifest["datasets"].append(dataset_info)

        print(f"Exported {len(problems)} problems to {filepath}")

    # Save manifest
    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dumps(dataset_manifest, indent=2)

    # Log to W&B
    wandb.log({
        "stage4/total_datasets": len(augmented_datasets),
        "stage4/total_problems": sum(len(p) for p in augmented_datasets.values()),
        "stage4/manifest_path": manifest_path
    })

    # Upload datasets as W&B artifacts
    artifact = wandb.Artifact(
        name="phase7_expert_datasets",
        type="dataset",
        metadata=dataset_manifest
    )
    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    return dataset_manifest
```

---

## Cost Analysis

### OpenRouter API Costs

**Per-Problem Generation Cost**:
- GPT-4 Turbo: $0.01/1K input tokens, $0.03/1K output tokens
- Average problem: 500 input + 1000 output tokens = $0.04
- Claude-3.5 Sonnet: Similar pricing

**Per-Expert Dataset**:
- Target: 400-500 problems per expert
- Generation: 500 problems × $0.04 = $20
- Validation (model attempts): 500 × 50 tokens × $0.01/1K = $0.25
- **Total per expert**: ~$20-25

**Phase 7 Total Cost**:
- 5 experts (typical) × $22.50 = $112.50
- Buffer for rejected problems (20%): $22.50
- **Total**: ~$135 (within $150-250 estimate)

### Cost Optimization Strategies

1. **Batch API calls**: 50% discount on OpenRouter batch endpoint
2. **Cache frontier model responses**: Reuse for similar experts
3. **Use smaller models for simple domains**: GPT-3.5 for general expert (~50% cost)
4. **Generate fewer, augment more**: 300 frontier-generated + 200 model variants

**Optimized Cost**: $75-100 (vs $135 baseline)

---

## Quality Assurance

### Validation Metrics

**Per-Expert Dataset**:
- ✅ Acceptance rate: ≥75% (problems pass model validation)
- ✅ Difficulty: Average confidence 0.70-0.80 (edge-of-chaos)
- ✅ Eudaimonia: Average score ≥0.70 (above 0.65 threshold)
- ✅ Diversity: Topic coverage ≥80% of specified topics
- ✅ Quality: No duplicate problems (fuzzy matching)

**Human Spot-Check**:
- Sample 10 problems per expert (50 total for 5 experts)
- Check: Problem clarity, solution correctness, difficulty appropriateness
- Target: ≥90% human approval rate

---

## Integration with Stage 2 (SVF Training)

**Dataset Handoff**:
```python
# Output from Stage 1
stage1_output = {
    "discovery_complete": True,
    "num_experts": 5,
    "experts": [
        {
            "id": 1,
            "name": "analytical",
            "dataset_path": "phase7_data/expert_analytical.jsonl"
        },
        # ... more experts
    ]
}

# Stage 2 loads datasets
for expert in stage1_output['experts']:
    dataset = load_dataset(expert['dataset_path'])
    train_svf_expert(model, expert, dataset)
```

---

## Risk Analysis

### Risk 1: High Rejection Rate (>50%)
**Mitigation**:
- Model refines generation prompt based on rejection reasons
- Use multiple frontier models for diversity
- Generate 2× target, expect 50% rejection
- Fallback: Use human-curated datasets (GSM8K, HumanEval)

### Risk 2: Cost Overrun (>$250)
**Mitigation**:
- Set hard budget limit in OpenRouter client
- Use batch API (50% discount)
- Stop generation at $200, proceed with available data
- Augment with model-generated variants (free)

### Risk 3: Eudaimonia Drift in Generated Data
**Mitigation**:
- Every problem validated by model's Phase 5 eudaimonia system
- Reject any problem scoring <0.65
- Manual review of edge cases (0.65-0.70 range)

---

## Timeline

**Stage 1 (Self-Analysis & Expert Definition)**: 4 hours
- Capability benchmarking: 2 hours
- Expert determination: 1 hour
- Data requirements specification: 1 hour

**Stage 2-3 (Generation & Validation)**: 4-6 hours
- Frontier model generation: 2-3 hours (depends on API speed)
- Model validation: 2-3 hours (model tests each problem)

**Stage 4 (Export & Handoff)**: <1 hour

**Total**: ~8-11 hours (Day 1 of Phase 7)

---

## Success Criteria

- ✅ Model determines N experts (3 ≤ N ≤ 10)
- ✅ Model specifies data requirements for each expert
- ✅ N datasets generated (300-500 problems each)
- ✅ Validation acceptance rate ≥75% per dataset
- ✅ All problems pass eudaimonia check (≥0.65)
- ✅ Difficulty in edge-of-chaos range (70-80% confidence)
- ✅ Total cost <$250 (target: $150)
- ✅ Human spot-check ≥90% approval

---

**Next Stage**: Stage 2 (SVF Training) using model-generated datasets

**Integration**: Datasets validated against Phase 5 eudaimonia, ready for Phase 7 Stage 2
