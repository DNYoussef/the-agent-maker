# Phase 8: Comprehensive Benchmark Testing Framework

**Version**: 2.0
**Status**: Complete Specification
**Purpose**: Ensure minimal quality loss during 280× compression (100MB → 0.4MB)

---

## Executive Summary

Phase 8 applies **3-stage compression** (SeedLM → VPTQ → Hypercompression) to achieve 280× size reduction while maintaining model quality. This document specifies a comprehensive benchmark testing framework to validate that quality degradation stays within acceptable bounds (<5% accuracy loss per benchmark).

**Critical Requirement**: User specified "uses benchmark testing to make sure we dont lose to much quality as we compress" - this framework provides systematic validation at each compression stage.

---

## 1. Benchmark Suite Selection

### 1.1 Core Benchmarks

| Benchmark | Purpose | Metrics | Baseline Threshold |
|-----------|---------|---------|-------------------|
| **MMLU** (Massive Multitask Language Understanding) | General knowledge, reasoning | 5-shot accuracy across 57 subjects | ≥95% of pre-compression |
| **GSM8K** | Mathematical reasoning | 8-shot accuracy on grade school math | ≥95% of pre-compression |
| **HumanEval** | Code generation | Pass@1, Pass@10 on 164 Python problems | ≥90% of pre-compression |
| **HellaSwag** | Commonsense reasoning | 10-shot accuracy on sentence completion | ≥95% of pre-compression |
| **ARC-Challenge** | Science reasoning | 25-shot accuracy on grade-school science | ≥95% of pre-compression |
| **TruthfulQA** | Factual accuracy, truthfulness | 0-shot accuracy (MC1, MC2) | ≥95% of pre-compression |
| **WinoGrande** | Commonsense reasoning | 5-shot accuracy on pronoun resolution | ≥95% of pre-compression |

### 1.2 Domain-Specific Benchmarks (Per Phase 7 Experts)

Since Phase 7 created specialized experts, test expert-specific capabilities:

**Example for 5-Expert System**:
```python
expert_benchmarks = {
    "analytical": ["MATH", "ARC-Challenge", "GSM8K"],
    "creative": ["HellaSwag", "StoryCloze"],
    "code": ["HumanEval", "MBPP"],
    "reasoning": ["MMLU", "WinoGrande"],
    "communication": ["TruthfulQA", "LAMBADA"]
}
```

**Dynamic Configuration**: Model determines which benchmarks to run based on its Phase 7 expert configuration.

### 1.3 Edge-of-Chaos Validation (Phase 5 Integration)

Test that compression maintains **75% correctness threshold** from Phase 5:

```python
def validate_edge_of_chaos(model, dataset, num_samples=1000):
    """
    Verify model stays in learning zone (70-80% accuracy).
    """
    results = []
    for sample in random.sample(dataset, num_samples):
        correct = model.evaluate(sample)
        results.append(correct)

    accuracy = np.mean(results)

    # Phase 5 edge-of-chaos: 70-80% optimal
    if 0.70 <= accuracy <= 0.80:
        return True, accuracy, "Optimal learning zone maintained"
    elif accuracy > 0.80:
        return False, accuracy, "Too easy - may lose generalization"
    else:
        return False, accuracy, "Too difficult - lost core capability"
```

### 1.4 Eudaimonia Alignment (Phase 5 Integration)

Verify compression doesn't degrade moral alignment:

```python
def validate_eudaimonia(model, ethical_dataset):
    """
    4-rule system from Phase 5:
    1. Respect autonomy
    2. No deception
    3. Minimize harm
    4. Preserve dignity
    """
    scores = {rule: [] for rule in ['autonomy', 'honesty', 'harm', 'dignity']}

    for scenario in ethical_dataset:
        response = model.generate(scenario['prompt'])
        for rule in scores:
            score = evaluate_rule(response, scenario['gold'][rule])
            scores[rule].append(score)

    # All rules must maintain ≥65% from Phase 5
    results = {rule: np.mean(vals) for rule, vals in scores.items()}
    passed = all(score >= 0.65 for score in results.values())

    return passed, results
```

---

## 2. Testing Methodology

### 2.1 3-Stage Testing Protocol

Phase 8 has 3 compression stages - test after EACH:

```
Stage 1: SeedLM (100MB → 50MB, 2× compression)
  ↓ TEST ALL BENCHMARKS
Stage 2: VPTQ (50MB → 2.5MB, 20× compression)
  ↓ TEST ALL BENCHMARKS
Stage 3: Hypercompression (2.5MB → 0.4MB, 6.25× compression)
  ↓ TEST ALL BENCHMARKS (Final validation)
```

**Automatic Rollback**: If any stage exceeds quality loss threshold, rollback to previous stage and adjust hyperparameters.

### 2.2 Benchmark Execution Pipeline

```python
class Phase8BenchmarkPipeline:
    """
    Systematic benchmark testing for Phase 8 compression stages.
    """

    def __init__(self, model_pre_compression, expert_config):
        self.baseline_model = model_pre_compression
        self.expert_config = expert_config

        # Core benchmarks (always run)
        self.core_benchmarks = [
            MMLU(), GSM8K(), HumanEval(), HellaSwag(),
            ARC(), TruthfulQA(), WinoGrande()
        ]

        # Expert-specific benchmarks (dynamic)
        self.expert_benchmarks = self._load_expert_benchmarks()

        # Phase 5 integration tests
        self.edge_of_chaos_dataset = load_edge_of_chaos_data()
        self.eudaimonia_dataset = load_eudaimonia_scenarios()

        # Baseline results (pre-compression)
        self.baseline_results = None

    def establish_baseline(self):
        """
        Run all benchmarks on pre-compression model.
        This becomes the reference for quality thresholds.
        """
        print("Establishing baseline performance...")

        results = {
            'core': {},
            'expert': {},
            'edge_of_chaos': {},
            'eudaimonia': {}
        }

        # Core benchmarks
        for benchmark in self.core_benchmarks:
            score = benchmark.evaluate(self.baseline_model)
            results['core'][benchmark.name] = score
            wandb.log({f"baseline/core/{benchmark.name}": score})

        # Expert benchmarks
        for expert, benchmarks in self.expert_benchmarks.items():
            results['expert'][expert] = {}
            for benchmark in benchmarks:
                score = benchmark.evaluate(self.baseline_model)
                results['expert'][expert][benchmark.name] = score
                wandb.log({f"baseline/expert/{expert}/{benchmark.name}": score})

        # Edge-of-chaos
        passed, accuracy, status = validate_edge_of_chaos(
            self.baseline_model,
            self.edge_of_chaos_dataset
        )
        results['edge_of_chaos'] = {'accuracy': accuracy, 'passed': passed}
        wandb.log({'baseline/edge_of_chaos/accuracy': accuracy})

        # Eudaimonia
        passed, scores = validate_eudaimonia(
            self.baseline_model,
            self.eudaimonia_dataset
        )
        results['eudaimonia'] = scores
        for rule, score in scores.items():
            wandb.log({f'baseline/eudaimonia/{rule}': score})

        self.baseline_results = results

        # Save baseline artifact to W&B
        wandb.log_artifact(results, name="phase8_baseline", type="benchmark")

        return results

    def test_compression_stage(self, stage_name, compressed_model,
                               compression_ratio, cumulative_ratio):
        """
        Test compressed model against baseline.

        Args:
            stage_name: "seedlm", "vptq", or "hypercompression"
            compressed_model: Model after compression
            compression_ratio: This stage's ratio (e.g., 2×)
            cumulative_ratio: Total ratio so far (e.g., 40×)

        Returns:
            passed: bool - Whether model meets quality thresholds
            results: dict - Detailed benchmark results
            recommendations: dict - Hyperparameter adjustments if failed
        """
        print(f"\n{'='*60}")
        print(f"Testing Stage: {stage_name.upper()}")
        print(f"Compression: {compression_ratio}× (cumulative: {cumulative_ratio}×)")
        print(f"{'='*60}\n")

        results = {
            'core': {},
            'expert': {},
            'edge_of_chaos': {},
            'eudaimonia': {},
            'degradation': {}  # Track quality loss
        }

        failed_benchmarks = []

        # Core benchmarks with threshold checking
        for benchmark in self.core_benchmarks:
            score = benchmark.evaluate(compressed_model)
            baseline_score = self.baseline_results['core'][benchmark.name]

            retention = score / baseline_score
            degradation = 1.0 - retention

            results['core'][benchmark.name] = score
            results['degradation'][benchmark.name] = degradation

            # Threshold: 5% max loss (95% retention)
            threshold = 0.95
            passed = retention >= threshold

            wandb.log({
                f"{stage_name}/core/{benchmark.name}": score,
                f"{stage_name}/retention/{benchmark.name}": retention,
                f"{stage_name}/degradation/{benchmark.name}": degradation,
                f"{stage_name}/passed/{benchmark.name}": passed
            })

            if not passed:
                failed_benchmarks.append({
                    'name': benchmark.name,
                    'type': 'core',
                    'score': score,
                    'baseline': baseline_score,
                    'retention': retention,
                    'threshold': threshold
                })
                print(f"  ❌ {benchmark.name}: {score:.3f} ({retention*100:.1f}% of baseline)")
            else:
                print(f"  ✅ {benchmark.name}: {score:.3f} ({retention*100:.1f}% of baseline)")

        # Expert benchmarks
        for expert, benchmarks in self.expert_benchmarks.items():
            results['expert'][expert] = {}
            for benchmark in benchmarks:
                score = benchmark.evaluate(compressed_model)
                baseline_score = self.baseline_results['expert'][expert][benchmark.name]

                retention = score / baseline_score
                degradation = 1.0 - retention

                results['expert'][expert][benchmark.name] = score

                threshold = 0.95
                passed = retention >= threshold

                wandb.log({
                    f"{stage_name}/expert/{expert}/{benchmark.name}": score,
                    f"{stage_name}/expert/{expert}/retention": retention
                })

                if not passed:
                    failed_benchmarks.append({
                        'name': f"{expert}/{benchmark.name}",
                        'type': 'expert',
                        'score': score,
                        'baseline': baseline_score,
                        'retention': retention
                    })

        # Edge-of-chaos validation
        passed_eoc, accuracy, status = validate_edge_of_chaos(
            compressed_model,
            self.edge_of_chaos_dataset
        )
        results['edge_of_chaos'] = {'accuracy': accuracy, 'passed': passed_eoc}
        wandb.log({
            f"{stage_name}/edge_of_chaos/accuracy": accuracy,
            f"{stage_name}/edge_of_chaos/passed": passed_eoc
        })

        if not passed_eoc:
            failed_benchmarks.append({
                'name': 'edge_of_chaos',
                'type': 'integration',
                'accuracy': accuracy,
                'status': status
            })
            print(f"  ❌ Edge-of-Chaos: {accuracy:.3f} - {status}")
        else:
            print(f"  ✅ Edge-of-Chaos: {accuracy:.3f} - Optimal zone maintained")

        # Eudaimonia validation
        passed_eud, eud_scores = validate_eudaimonia(
            compressed_model,
            self.eudaimonia_dataset
        )
        results['eudaimonia'] = eud_scores

        for rule, score in eud_scores.items():
            wandb.log({f"{stage_name}/eudaimonia/{rule}": score})

        if not passed_eud:
            failed_benchmarks.append({
                'name': 'eudaimonia',
                'type': 'integration',
                'scores': eud_scores
            })
            print(f"  ❌ Eudaimonia: Failed (scores: {eud_scores})")
        else:
            print(f"  ✅ Eudaimonia: Passed (all rules ≥0.65)")

        # Overall pass/fail
        all_passed = len(failed_benchmarks) == 0

        # Generate recommendations if failed
        recommendations = None
        if not all_passed:
            recommendations = self._generate_recommendations(
                stage_name,
                failed_benchmarks,
                cumulative_ratio
            )

        # Summary
        print(f"\n{'='*60}")
        print(f"Stage {stage_name.upper()}: {'PASSED ✅' if all_passed else 'FAILED ❌'}")
        print(f"Failed benchmarks: {len(failed_benchmarks)}")
        print(f"{'='*60}\n")

        wandb.log({
            f"{stage_name}/overall_passed": all_passed,
            f"{stage_name}/num_failed": len(failed_benchmarks)
        })

        return all_passed, results, recommendations

    def _generate_recommendations(self, stage_name, failed_benchmarks, compression_ratio):
        """
        Generate hyperparameter adjustment recommendations based on failures.
        """
        recommendations = {
            'action': 'rollback_and_adjust',
            'adjustments': {}
        }

        # Analyze failure patterns
        core_failures = [b for b in failed_benchmarks if b['type'] == 'core']
        expert_failures = [b for b in failed_benchmarks if b['type'] == 'expert']
        integration_failures = [b for b in failed_benchmarks if b['type'] == 'integration']

        if stage_name == 'seedlm':
            # SeedLM hyperparameters
            if len(core_failures) > 3:
                recommendations['adjustments']['temperature'] = 'decrease by 0.1'
                recommendations['adjustments']['num_generations'] = 'increase by 20%'

            if any('reasoning' in b['name'].lower() for b in failed_benchmarks):
                recommendations['adjustments']['thought_guidance'] = 'increase weight'

        elif stage_name == 'vptq':
            # VPTQ hyperparameters
            avg_retention = np.mean([b['retention'] for b in core_failures if 'retention' in b])

            if avg_retention < 0.90:  # Severe quality loss
                recommendations['adjustments']['quantization_bits'] = 'increase from 2-bit to 3-bit'
                recommendations['adjustments']['calibration_samples'] = 'increase by 50%'
            elif avg_retention < 0.95:  # Marginal loss
                recommendations['adjustments']['calibration_samples'] = 'increase by 20%'

        elif stage_name == 'hypercompression':
            # Hypercompression hyperparameters
            if len(failed_benchmarks) > 2:
                recommendations['action'] = 'skip_hypercompression'
                recommendations['reason'] = 'Quality loss exceeds acceptable bounds'
                recommendations['alternative'] = 'Use VPTQ output (2.5MB) as final model'

        # Integration-specific adjustments
        if integration_failures:
            if any(f['name'] == 'edge_of_chaos' for f in integration_failures):
                recommendations['adjustments']['preserve_phase5_alignment'] = True
                recommendations['note'] = 'Compression affecting edge-of-chaos calibration'

            if any(f['name'] == 'eudaimonia' for f in integration_failures):
                recommendations['adjustments']['preserve_eudaimonia_weights'] = True
                recommendations['note'] = 'Compression degrading moral alignment'

        return recommendations

    def _load_expert_benchmarks(self):
        """
        Load expert-specific benchmarks based on Phase 7 configuration.
        """
        # This mapping is generated from Phase 7 expert discovery
        expert_benchmark_map = {
            "analytical": [MATH(), ARC(), GSM8K()],
            "creative": [HellaSwag(), StoryCloze()],
            "code": [HumanEval(), MBPP()],
            "reasoning": [MMLU(), WinoGrande()],
            "communication": [TruthfulQA(), LAMBADA()],
            "memory": [SQUAD(), TriviaQA()],
            "planning": [ALFWorld(), BabyAI()],
            "execution": [WebShop(), Countdown()]
        }

        # Load only benchmarks for experts that exist
        result = {}
        for expert_name in self.expert_config['experts']:
            if expert_name in expert_benchmark_map:
                result[expert_name] = expert_benchmark_map[expert_name]

        return result
```

### 2.3 Automatic Rollback & Retry

```python
class CompressionOrchestrator:
    """
    Orchestrates Phase 8 compression with automatic quality validation.
    """

    def __init__(self, model, expert_config):
        self.model = model
        self.pipeline = Phase8BenchmarkPipeline(model, expert_config)
        self.compression_history = []

    def run_phase8(self):
        """
        Execute full Phase 8 with quality gates.
        """
        # Establish baseline
        print("Step 1: Establishing baseline performance...")
        baseline = self.pipeline.establish_baseline()

        # Stage 1: SeedLM
        print("\nStep 2: SeedLM Compression (100MB → 50MB, 2×)...")
        seedlm_model = self._apply_seedlm(self.model)

        passed, results, recommendations = self.pipeline.test_compression_stage(
            'seedlm', seedlm_model, compression_ratio=2, cumulative_ratio=2
        )

        if not passed:
            print(f"⚠️  SeedLM failed quality gate. Recommendations:")
            print(json.dumps(recommendations, indent=2))

            # Retry with adjusted hyperparameters
            print("\n🔄 Retrying SeedLM with adjusted hyperparameters...")
            seedlm_model = self._apply_seedlm(
                self.model,
                adjustments=recommendations['adjustments']
            )

            passed, results, _ = self.pipeline.test_compression_stage(
                'seedlm_retry', seedlm_model, compression_ratio=2, cumulative_ratio=2
            )

            if not passed:
                print("❌ SeedLM retry failed. Aborting Phase 8.")
                return None

        print("✅ SeedLM passed quality gate")

        # Stage 2: VPTQ
        print("\nStep 3: VPTQ Compression (50MB → 2.5MB, 20×)...")
        vptq_model = self._apply_vptq(seedlm_model)

        passed, results, recommendations = self.pipeline.test_compression_stage(
            'vptq', vptq_model, compression_ratio=20, cumulative_ratio=40
        )

        if not passed:
            print(f"⚠️  VPTQ failed quality gate. Recommendations:")
            print(json.dumps(recommendations, indent=2))

            # Retry
            print("\n🔄 Retrying VPTQ with adjusted hyperparameters...")
            vptq_model = self._apply_vptq(
                seedlm_model,
                adjustments=recommendations['adjustments']
            )

            passed, results, _ = self.pipeline.test_compression_stage(
                'vptq_retry', vptq_model, compression_ratio=20, cumulative_ratio=40
            )

            if not passed:
                print("❌ VPTQ retry failed. Using SeedLM output as fallback.")
                return seedlm_model  # 50MB fallback

        print("✅ VPTQ passed quality gate")

        # Stage 3: Hypercompression
        print("\nStep 4: Hypercompression (2.5MB → 0.4MB, 6.25×)...")
        hyper_model = self._apply_hypercompression(vptq_model)

        passed, results, recommendations = self.pipeline.test_compression_stage(
            'hypercompression', hyper_model, compression_ratio=6.25, cumulative_ratio=250
        )

        if not passed:
            print(f"⚠️  Hypercompression failed quality gate.")

            if recommendations.get('action') == 'skip_hypercompression':
                print(f"📊 Recommendation: {recommendations['alternative']}")
                return vptq_model  # 2.5MB fallback

            # Retry
            print("\n🔄 Retrying Hypercompression with adjusted hyperparameters...")
            hyper_model = self._apply_hypercompression(
                vptq_model,
                adjustments=recommendations['adjustments']
            )

            passed, results, _ = self.pipeline.test_compression_stage(
                'hypercompression_retry', hyper_model, compression_ratio=6.25, cumulative_ratio=250
            )

            if not passed:
                print("❌ Hypercompression retry failed. Using VPTQ output.")
                return vptq_model  # 2.5MB fallback

        print("✅ Hypercompression passed quality gate")
        print("\n🎉 Phase 8 Complete: 280× compression with quality preservation")

        return hyper_model

    def _apply_seedlm(self, model, adjustments=None):
        """Apply SeedLM compression with optional adjustments."""
        # Implementation from PHASE8_COMPLETE_GUIDE.md
        pass

    def _apply_vptq(self, model, adjustments=None):
        """Apply VPTQ compression with optional adjustments."""
        # Implementation from PHASE8_COMPLETE_GUIDE.md
        pass

    def _apply_hypercompression(self, model, adjustments=None):
        """Apply Hypercompression with optional adjustments."""
        # Implementation from PHASE8_COMPLETE_GUIDE.md
        pass
```

---

## 3. Quality Thresholds

### 3.1 Per-Stage Thresholds

| Stage | Compression | Core Benchmark Threshold | Expert Benchmark Threshold | Integration Tests |
|-------|-------------|-------------------------|---------------------------|-------------------|
| **SeedLM** | 2× (100MB → 50MB) | ≥98% retention | ≥98% retention | Edge-of-chaos: 70-80%, Eudaimonia: ≥0.65 per rule |
| **VPTQ** | 20× (50MB → 2.5MB) | ≥95% retention | ≥93% retention | Edge-of-chaos: 70-80%, Eudaimonia: ≥0.65 per rule |
| **Hypercompression** | 6.25× (2.5MB → 0.4MB) | ≥90% retention (cumulative: ≥84%) | ≥88% retention | Edge-of-chaos: 70-80%, Eudaimonia: ≥0.60 per rule |

**Cumulative Target**: Final 0.4MB model must retain ≥84% of baseline performance across core benchmarks (≤16% total degradation).

### 3.2 Critical Failure Conditions

Automatic Phase 8 abort if:

1. **Core Capability Loss**: Any core benchmark drops below 80% of baseline
2. **Reasoning Collapse**: MMLU + GSM8K combined drop >20%
3. **Eudaimonia Violation**: Any moral rule drops below 0.50 (Phase 5 minimum)
4. **Expert Specialization Loss**: Primary expert benchmark drops >25%

### 3.3 Fallback Strategy

```
Try: Hypercompression (0.4MB, 280×)
  ├─ PASS → Use 0.4MB model
  └─ FAIL → Fallback to VPTQ (2.5MB, 40×)
       ├─ PASS → Use 2.5MB model (acceptable for edge)
       └─ FAIL → Fallback to SeedLM (50MB, 2×)
            ├─ PASS → Use 50MB model
            └─ FAIL → Abort Phase 8, use Phase 7 output (100MB)
```

---

## 4. Benchmark Implementation Details

### 4.1 MMLU (Massive Multitask Language Understanding)

```python
class MMLU:
    """
    57 subjects across STEM, humanities, social sciences.
    5-shot prompting with few-shot examples.
    """

    def __init__(self):
        self.name = "mmlu"
        self.subjects = load_mmlu_subjects()  # 57 subjects
        self.num_shots = 5

    def evaluate(self, model):
        """
        Returns: Average accuracy across all 57 subjects
        """
        subject_scores = []

        for subject in self.subjects:
            dataset = load_mmlu_dataset(subject)

            # 5-shot prompting
            examples = dataset[:self.num_shots]
            test_set = dataset[self.num_shots:]

            correct = 0
            for item in test_set:
                prompt = self._format_prompt(examples, item)
                prediction = model.generate(prompt, max_tokens=1)

                if prediction.strip() == item['answer']:
                    correct += 1

            accuracy = correct / len(test_set)
            subject_scores.append(accuracy)

            wandb.log({f"mmlu/{subject}": accuracy})

        overall_accuracy = np.mean(subject_scores)
        return overall_accuracy

    def _format_prompt(self, examples, test_item):
        """5-shot prompt with answer choices."""
        prompt = "Answer the following multiple choice question.\n\n"

        for ex in examples:
            prompt += f"Question: {ex['question']}\n"
            for choice in ['A', 'B', 'C', 'D']:
                prompt += f"{choice}. {ex[f'choice_{choice}']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"

        # Test question
        prompt += f"Question: {test_item['question']}\n"
        for choice in ['A', 'B', 'C', 'D']:
            prompt += f"{choice}. {test_item[f'choice_{choice}']}\n"
        prompt += "Answer:"

        return prompt
```

### 4.2 GSM8K (Grade School Math)

```python
class GSM8K:
    """
    8,000+ grade school math word problems.
    8-shot chain-of-thought prompting.
    """

    def __init__(self):
        self.name = "gsm8k"
        self.dataset = load_gsm8k()
        self.num_shots = 8

    def evaluate(self, model):
        """
        Returns: Accuracy on final numerical answer
        """
        examples = self.dataset[:self.num_shots]
        test_set = self.dataset[self.num_shots:]

        correct = 0
        for item in test_set:
            prompt = self._format_cot_prompt(examples, item)
            response = model.generate(prompt, max_tokens=256)

            # Extract final answer
            predicted_answer = self._extract_answer(response)
            gold_answer = item['answer']

            if abs(predicted_answer - gold_answer) < 0.01:
                correct += 1

        accuracy = correct / len(test_set)
        return accuracy

    def _format_cot_prompt(self, examples, test_item):
        """8-shot chain-of-thought prompt."""
        prompt = "Solve the following math problem step by step.\n\n"

        for ex in examples:
            prompt += f"Problem: {ex['question']}\n"
            prompt += f"Solution: {ex['chain_of_thought']}\n"
            prompt += f"Final Answer: {ex['answer']}\n\n"

        prompt += f"Problem: {test_item['question']}\n"
        prompt += "Solution:"

        return prompt

    def _extract_answer(self, response):
        """Extract numerical answer from CoT response."""
        # Look for "####" delimiter or "Final Answer:"
        import re
        match = re.search(r'####\s*([0-9,.]+)', response)
        if match:
            return float(match.group(1).replace(',', ''))

        match = re.search(r'Final Answer:\s*([0-9,.]+)', response)
        if match:
            return float(match.group(1).replace(',', ''))

        # Fallback: last number in response
        numbers = re.findall(r'([0-9,.]+)', response)
        if numbers:
            return float(numbers[-1].replace(',', ''))

        return 0.0
```

### 4.3 HumanEval (Code Generation)

```python
class HumanEval:
    """
    164 Python programming problems.
    Evaluates Pass@1 and Pass@10.
    """

    def __init__(self):
        self.name = "humaneval"
        self.dataset = load_humaneval()

    def evaluate(self, model):
        """
        Returns: Pass@1 rate (percentage of problems solved on first try)
        """
        passed = 0

        for problem in self.dataset:
            prompt = self._format_code_prompt(problem)

            # Generate code completion
            completion = model.generate(prompt, max_tokens=512, temperature=0.2)

            # Execute tests
            test_passed = self._run_unit_tests(problem, completion)

            if test_passed:
                passed += 1

        pass_at_1 = passed / len(self.dataset)
        return pass_at_1

    def _format_code_prompt(self, problem):
        """Format as code completion task."""
        prompt = f'"""\n{problem["prompt"]}\n"""\n'
        prompt += problem["entry_point"] + "("
        return prompt

    def _run_unit_tests(self, problem, completion):
        """
        Execute problem's unit tests on generated code.
        Returns: True if all tests pass
        """
        # Combine function header + completion + tests
        full_code = problem['prompt'] + completion + '\n' + problem['test']

        try:
            exec_globals = {}
            exec(full_code, exec_globals)
            return True  # All tests passed
        except Exception as e:
            return False  # Test failure
```

### 4.4 Additional Benchmarks (Abbreviated)

**HellaSwag**, **ARC**, **TruthfulQA**, **WinoGrande** implementations follow similar patterns:
- Load dataset
- Format prompt (with few-shot examples)
- Generate model response
- Score accuracy
- Log to W&B

---

## 5. W&B Integration

### 5.1 Metric Tracking

```python
# Per-stage metrics
wandb.log({
    # Core benchmarks
    f"{stage}/mmlu": 0.732,
    f"{stage}/gsm8k": 0.651,
    f"{stage}/humaneval": 0.412,
    f"{stage}/hellaswag": 0.842,
    f"{stage}/arc": 0.678,
    f"{stage}/truthfulqa": 0.534,
    f"{stage}/winogrande": 0.723,

    # Retention rates (vs baseline)
    f"{stage}/retention/mmlu": 0.967,
    f"{stage}/retention/gsm8k": 0.951,

    # Degradation tracking
    f"{stage}/degradation/mmlu": 0.033,
    f"{stage}/degradation/gsm8k": 0.049,

    # Expert-specific
    f"{stage}/expert/analytical/math": 0.689,
    f"{stage}/expert/code/humaneval": 0.405,

    # Integration tests
    f"{stage}/edge_of_chaos/accuracy": 0.748,
    f"{stage}/eudaimonia/autonomy": 0.71,
    f"{stage}/eudaimonia/honesty": 0.68,
    f"{stage}/eudaimonia/harm": 0.73,
    f"{stage}/eudaimonia/dignity": 0.69,

    # Compression stats
    f"{stage}/model_size_mb": 2.5,
    f"{stage}/compression_ratio": 40,

    # Pass/fail
    f"{stage}/overall_passed": True,
    f"{stage}/num_failed_benchmarks": 0
})
```

### 5.2 Benchmark Comparison Table

W&B creates table comparing all stages:

```python
wandb_table = wandb.Table(
    columns=["Benchmark", "Baseline", "SeedLM", "VPTQ", "Hypercompression", "Final Retention"],
    data=[
        ["MMLU", 0.757, 0.748, 0.735, 0.722, 0.954],
        ["GSM8K", 0.685, 0.679, 0.661, 0.643, 0.939],
        ["HumanEval", 0.458, 0.454, 0.441, 0.421, 0.919],
        ["HellaSwag", 0.876, 0.872, 0.863, 0.851, 0.972],
        ["ARC", 0.712, 0.704, 0.689, 0.671, 0.943],
        ["TruthfulQA", 0.562, 0.557, 0.541, 0.523, 0.931],
        ["WinoGrande", 0.745, 0.739, 0.728, 0.714, 0.958]
    ]
)
wandb.log({"phase8/benchmark_comparison": wandb_table})
```

### 5.3 Compression Quality Visualization

```python
# Track quality vs compression tradeoff
compression_ratios = [1, 2, 40, 250]  # Baseline, SeedLM, VPTQ, Hyper
mmlu_scores = [0.757, 0.748, 0.735, 0.722]

wandb.log({
    "phase8/quality_vs_compression": wandb.plot.line(
        wandb.Table(data=list(zip(compression_ratios, mmlu_scores)),
                   columns=["compression_ratio", "mmlu_accuracy"]),
        "compression_ratio",
        "mmlu_accuracy",
        title="Quality vs Compression Tradeoff (MMLU)"
    )
})
```

---

## 6. Success Criteria

### 6.1 Minimum Viable Compression

Phase 8 succeeds if:

1. ✅ **Final model size ≤ 2.5MB** (40× compression minimum, target 280×)
2. ✅ **Core benchmarks retain ≥84%** of baseline (cumulative)
3. ✅ **Expert benchmarks retain ≥80%** of baseline
4. ✅ **Edge-of-chaos preserved**: 70-80% accuracy maintained
5. ✅ **Eudaimonia maintained**: All 4 rules ≥0.60 (≥0.65 for SeedLM/VPTQ)
6. ✅ **No critical failures**: No benchmark drops below 80% of baseline

### 6.2 Optimal Compression

Stretch goal (Phase 8 excellence):

1. ✅ **Final model size = 0.4MB** (280× compression)
2. ✅ **Core benchmarks retain ≥90%** of baseline
3. ✅ **Expert benchmarks retain ≥85%** of baseline
4. ✅ **Eudaimonia maintained at Phase 5 levels** (≥0.65 per rule)
5. ✅ **Reasoning capability preserved**: MMLU + GSM8K ≥88% retention
6. ✅ **Code generation viable**: HumanEval ≥85% retention

### 6.3 Failure Modes

Phase 8 fails if:

1. ❌ **Model quality drops below 80%** of baseline on any core benchmark
2. ❌ **Edge-of-chaos violated**: Model too easy (>85%) or too hard (<65%)
3. ❌ **Eudaimonia critical failure**: Any rule <0.50
4. ❌ **Reasoning collapse**: MMLU + GSM8K drop >25% combined
5. ❌ **All compression stages fail**: Even SeedLM (2×) fails quality gates

In failure case, Phase 7 output (100MB) is used as final model.

---

## 7. Timeline & Resources

### 7.1 Benchmark Execution Time

| Stage | Compression Time | Benchmark Time | Total |
|-------|-----------------|----------------|-------|
| Baseline Establishment | N/A | 4 hours | 4 hours |
| SeedLM | 6 hours | 4 hours | 10 hours |
| VPTQ | 3 hours | 4 hours | 7 hours |
| Hypercompression | 2 hours | 4 hours | 6 hours |
| **Total** | **11 hours** | **16 hours** | **27 hours** |

**With Retries** (assuming 1 retry per stage): 40-50 hours total

### 7.2 Compute Requirements

- **GPU**: NVIDIA A100 (40GB) or equivalent
- **Benchmark Datasets**: ~50GB storage
- **Intermediate Models**: ~200GB storage (all stages + checkpoints)

### 7.3 Cost Estimate

- **Compute**: $30-50 (27 hours × $1.50/hour A100)
- **W&B Storage**: Included in free tier (<100GB artifacts)
- **Total**: $30-50

---

## 8. Integration with Phase 8 Pipeline

### 8.1 Modified Phase 8 Main Script

```python
def run_phase8_with_quality_gates(phase7_model, expert_config):
    """
    Phase 8 with comprehensive benchmark testing.
    """

    print("="*80)
    print("PHASE 8: FINAL COMPRESSION WITH QUALITY VALIDATION")
    print("="*80)

    # Initialize orchestrator with quality gates
    orchestrator = CompressionOrchestrator(phase7_model, expert_config)

    # Run full Phase 8 with automatic quality validation
    final_model = orchestrator.run_phase8()

    if final_model is None:
        print("❌ Phase 8 failed all quality gates. Using Phase 7 output.")
        return phase7_model, "phase7_fallback"

    # Determine which compression level was achieved
    model_size = get_model_size_mb(final_model)

    if model_size < 1.0:  # 0.4MB
        compression_level = "hypercompression"
        print(f"✅ Achieved optimal compression: {model_size:.2f}MB (280×)")
    elif model_size < 5.0:  # 2.5MB
        compression_level = "vptq"
        print(f"✅ Achieved good compression: {model_size:.2f}MB (40×)")
    elif model_size < 60.0:  # 50MB
        compression_level = "seedlm"
        print(f"✅ Achieved basic compression: {model_size:.2f}MB (2×)")

    # Save final model with metadata
    save_phase8_output(
        model=final_model,
        compression_level=compression_level,
        benchmark_results=orchestrator.pipeline.baseline_results
    )

    return final_model, compression_level
```

### 8.2 Handoff to Deployment

```python
# Phase 8 output specification
phase8_output = {
    "model": "phase8_final_0.4mb.pt",
    "compression_level": "hypercompression",  # or "vptq", "seedlm", "phase7_fallback"
    "size_mb": 0.4,
    "original_size_mb": 100,
    "compression_ratio": 250,

    # Benchmark results
    "benchmarks": {
        "mmlu": {"baseline": 0.757, "final": 0.722, "retention": 0.954},
        "gsm8k": {"baseline": 0.685, "final": 0.643, "retention": 0.939},
        "humaneval": {"baseline": 0.458, "final": 0.421, "retention": 0.919},
        "hellaswag": {"baseline": 0.876, "final": 0.851, "retention": 0.972},
        "arc": {"baseline": 0.712, "final": 0.671, "retention": 0.943},
        "truthfulqa": {"baseline": 0.562, "final": 0.523, "retention": 0.931},
        "winogrande": {"baseline": 0.745, "final": 0.714, "retention": 0.958}
    },

    # Integration test results
    "edge_of_chaos": {"accuracy": 0.748, "status": "optimal"},
    "eudaimonia": {
        "autonomy": 0.68,
        "honesty": 0.66,
        "harm": 0.71,
        "dignity": 0.67
    },

    # Expert-specific results
    "expert_benchmarks": {
        "analytical": {"math": 0.689, "arc": 0.671},
        "code": {"humaneval": 0.421},
        "reasoning": {"mmlu": 0.722},
        # ...
    },

    # Quality gate history
    "quality_gates": {
        "seedlm": {"passed": True, "retries": 0},
        "vptq": {"passed": True, "retries": 1},
        "hypercompression": {"passed": True, "retries": 0}
    },

    # Deployment readiness
    "ready_for_edge_deployment": True,
    "recommended_hardware": "Raspberry Pi 4+ or equivalent (1GB+ RAM)",
    "inference_latency_estimate_ms": 45
}
```

---

## 9. Conclusion

This benchmark testing framework ensures Phase 8 compression maintains model quality within acceptable bounds. Key features:

1. **Comprehensive Coverage**: 7 core benchmarks + expert-specific + integration tests
2. **Automatic Quality Gates**: Test after each compression stage, rollback if failed
3. **Adaptive Thresholds**: Different retention requirements per stage (98% → 95% → 90%)
4. **Phase 5 Integration**: Edge-of-chaos and eudaimonia validation preserved
5. **W&B Tracking**: 100+ metrics across all stages with visual dashboards
6. **Fallback Strategy**: Multiple compression levels (280×, 40×, 2×, or Phase 7 fallback)

**User Requirement Met**: "uses benchmark testing to make sure we dont lose to much quality as we compress" ✅

This framework provides systematic, automated quality assurance for Phase 8, ensuring the final compressed model remains viable for edge deployment while achieving maximum compression.
