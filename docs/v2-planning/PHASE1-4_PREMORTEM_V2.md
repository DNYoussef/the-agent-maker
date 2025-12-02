# Phase 1-4: Comprehensive Premortem Analysis

**Version**: 2.0
**Date**: 2025-10-15
**Purpose**: Identify failure modes, risks, and mitigations for Phases 1-4

---

## Executive Summary

This premortem identifies **critical failure modes** across Phases 1-4, with focus on the **model-size-agnostic architecture** challenge. Each phase has unique risks, but cross-phase integration failures are the highest threat.

### Risk Categories

1. **Data Issues** (Phase 1, Phase 3)
2. **Training Failures** (All phases)
3. **Integration Failures** (Phase 1→2, 2→3, 3→4)
4. **Infrastructure Failures** (Storage, VRAM, W&B)
5. **Quality Failures** (Accuracy drops, theater, compression artifacts)

### Severity Scale

- 🔴 **CRITICAL**: Project blocker, must fix immediately
- 🟡 **HIGH**: Major setback, requires significant rework
- 🟢 **MEDIUM**: Minor issue, has workaround
- ⚪ **LOW**: Inconvenience, minimal impact

---

## Phase 1: Cognate - Foundation Model Creation

### Risk 1.1: Dataset Download Failures 🔴 CRITICAL

**Failure Mode**: HuggingFace datasets fail to download (connection issues, rate limits, deprecated datasets)

**Impact**:
- Cannot train models without data
- Phase 1 blocked entirely
- Timeline delay: 1-3 days to resolve

**Probability**: 30% (HuggingFace API can be flaky)

**Mitigation**:
1. **Pre-download all datasets** before starting Phase 1
2. **Cache datasets locally** (1.35 GB)
3. **Have fallback mirrors**:
   ```python
   DATASET_MIRRORS = {
       "gsm8k": [
           ("huggingface", "gsm8k"),
           ("github", "https://github.com/openai/grade-school-math"),
           ("local", "/backup/datasets/gsm8k.json")
       ]
   }
   ```
4. **Implement retry logic** with exponential backoff
5. **Test download script** in isolation before Phase 1

**Detection**:
```python
def validate_dataset_availability():
    """Run before Phase 1 starts"""
    required_datasets = [
        "gsm8k", "ChilleD/SVAMP", "MU-NLPC/Calc-asdiv_a", "mbpp",
        "microsoft/codexglue", "ai2_arc", "hotpot_qa", "drop",
        "wics/strategy-qa", "piqa", "hellaswag", "boolq",
        "wikitext", "HuggingFaceFW/fineweb-edu"
    ]

    for dataset in required_datasets:
        try:
            load_dataset(dataset, split="train[:10]")  # Test load
            print(f"✅ {dataset}")
        except Exception as e:
            print(f"❌ {dataset}: {str(e)}")
            raise RuntimeError(f"Dataset {dataset} not available")
```

---

### Risk 1.2: Models Don't Diversify 🟡 HIGH

**Failure Mode**: Despite different dataset mixes and configs, all 3 models behave similarly (no diversity)

**Impact**:
- Phase 2 evolution has nothing to work with
- Merging similar models = no improvement
- Wasted 20+ hours of training

**Probability**: 40% (diversity is hard to enforce)

**Root Causes**:
1. **Dataset mixes too similar** (60% overlap in all 3 models)
2. **Random seed not set differently** (models converge to same solution)
3. **ACT thresholds too close** (0.90, 0.92, 0.94 instead of 0.90, 0.95, 0.99)
4. **LTM sizes too close** (2048, 3072, 4096 instead of 2048, 8192, 2048)

**Mitigation**:
1. **Enforce diversity metrics** during training:
   ```python
   def validate_diversity(model1, model2, model3):
       """Run after each epoch to detect early convergence"""

       # Test 1: Different halting steps
       halting1 = measure_avg_halting_steps(model1, test_set)
       halting2 = measure_avg_halting_steps(model2, test_set)
       halting3 = measure_avg_halting_steps(model3, test_set)

       diversity_halting = max(halting1, halting2, halting3) - min(halting1, halting2, halting3)
       assert diversity_halting > 2.0, f"ACT diversity too low: {diversity_halting:.2f}"

       # Test 2: Different memory usage
       mem1 = measure_ltm_usage(model1, test_set)
       mem2 = measure_ltm_usage(model2, test_set)
       mem3 = measure_ltm_usage(model3, test_set)

       diversity_memory = max(mem1, mem2, mem3) - min(mem1, mem2, mem3)
       assert diversity_memory > 0.3, f"Memory diversity too low: {diversity_memory:.2f}"

       # Test 3: Different inference times
       time1 = measure_inference_time(model1, test_set)
       time2 = measure_inference_time(model2, test_set)
       time3 = measure_inference_time(model3, test_set)

       diversity_time = max(time1, time2, time3) - min(time1, time2, time3)
       assert diversity_time > 10.0, f"Speed diversity too low: {diversity_time:.1f}ms"

       print(f"✅ Diversity validated: halting={diversity_halting:.2f}, mem={diversity_memory:.2f}, time={diversity_time:.1f}ms")
   ```

2. **Aggressive config differences**:
   ```python
   MODEL_CONFIGS = {
       "model1_reasoning": {
           "act_threshold": 0.95,  # Thinks LONG (12+ blocks)
           "ltm_slots": 2048,      # SMALL memory
           "dataset_weights": {"math": 0.5, "qa": 0.3, "code": 0.1, "commonsense": 0.05, "language": 0.05}
       },
       "model2_memory": {
           "act_threshold": 0.85,  # Thinks MODERATE (8-10 blocks)
           "ltm_slots": 8192,      # LARGE memory
           "dataset_weights": {"language": 0.5, "qa": 0.3, "math": 0.1, "code": 0.05, "commonsense": 0.05}
       },
       "model3_speed": {
           "act_threshold": 0.99,  # Thinks SHORT (3-5 blocks)
           "ltm_slots": 2048,      # SMALL memory
           "dataset_weights": {"commonsense": 0.5, "math": 0.3, "code": 0.1, "qa": 0.05, "language": 0.05}
       }
   }
   ```

3. **Set different random seeds**:
   ```python
   torch.manual_seed(42 + model_id)  # 42, 43, 44
   ```

4. **Train on different epochs**:
   - Model 1: 10 epochs
   - Model 2: 12 epochs (slight overtraining)
   - Model 3: 8 epochs (slight undertraining)

**Detection**: Run diversity validation after Phase 1 completes, **BEFORE** starting Phase 2.

---

### Risk 1.3: Out of Memory (OOM) During Training 🟡 HIGH

**Failure Mode**: Model doesn't fit in 6GB VRAM, crashes during training

**Impact**:
- Training fails after hours of work
- Data loss if not checkpointed
- Requires smaller model or more GPU memory

**Probability**: 25% (depends on actual model size at runtime)

**Root Causes**:
1. **Model larger than expected** (e.g., 50M params instead of 25M)
2. **Batch size too large** (32 instead of 16)
3. **Gradient accumulation not working**
4. **Memory leaks** (old activations not freed)

**Mitigation**:
1. **Dynamic batch size**:
   ```python
   def calculate_safe_batch_size(model, device_vram_gb):
       """Calculate batch size that fits in VRAM"""
       model_size_mb = get_model_size(model)["size_mb"]

       # Rule of thumb: 4x model size for training (model + optimizer + gradients + activations)
       required_vram_mb = model_size_mb * 4

       # Leave 10% headroom
       available_vram_mb = device_vram_gb * 1024 * 0.9

       if required_vram_mb > available_vram_mb:
           # Won't fit, need gradient accumulation
           batch_size = 1
           accumulation_steps = math.ceil(required_vram_mb / available_vram_mb)
           print(f"⚠️ Using gradient accumulation: batch_size=1, accumulation={accumulation_steps}")
       else:
           # Fits, calculate optimal batch size
           overhead_per_sample = model_size_mb * 0.1  # 10% overhead per sample
           batch_size = int((available_vram_mb - required_vram_mb) / overhead_per_sample)
           batch_size = min(batch_size, 32)  # Cap at 32
           accumulation_steps = 1

       return batch_size, accumulation_steps
   ```

2. **Gradient checkpointing**:
   ```python
   model.gradient_checkpointing_enable()  # Trade compute for memory
   ```

3. **Mixed precision training**:
   ```python
   scaler = torch.cuda.amp.GradScaler()
   with torch.cuda.amp.autocast():
       outputs = model(inputs)
       loss = criterion(outputs, labels)
   scaler.scale(loss).backward()
   ```

4. **Regular checkpoints**:
   ```python
   # Checkpoint every 1000 steps
   if step % 1000 == 0:
       torch.save(model.state_dict(), f"checkpoint_step{step}.pt")
   ```

**Detection**:
```python
try:
    # Test forward pass
    test_batch = torch.randn(batch_size, seq_len, hidden_dim).to(device)
    with torch.no_grad():
        output = model(test_batch)
    print(f"✅ Batch size {batch_size} fits in VRAM")
except torch.cuda.OutOfMemoryError:
    print(f"❌ Batch size {batch_size} too large for VRAM")
    # Retry with smaller batch size
```

---

### Risk 1.4: Training Doesn't Converge 🟡 HIGH

**Failure Mode**: Loss plateaus or diverges, model doesn't learn

**Impact**:
- Wasted training time (hours)
- Models perform no better than random
- May need to restart with different hyperparameters

**Probability**: 30% (training instability is common)

**Root Causes**:
1. **Learning rate too high** (1e-3 causes divergence)
2. **Learning rate too low** (1e-6 learns too slowly)
3. **Grokfast lambda too high** (0.5+ over-filters gradients)
4. **Data corruption** (NaN values in dataset)
5. **ACT threshold too extreme** (0.999 causes collapse)

**Mitigation**:
1. **Learning rate warmup**:
   ```python
   scheduler = get_linear_schedule_with_warmup(
       optimizer,
       num_warmup_steps=1000,  # Warm up for 1000 steps
       num_training_steps=total_steps
   )
   ```

2. **Gradient clipping**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **Monitor loss trends**:
   ```python
   def detect_training_issues(loss_history):
       """Detect divergence early"""
       last_100 = loss_history[-100:]

       # Issue 1: Divergence (loss increasing)
       if len(last_100) > 10:
           recent_trend = np.polyfit(range(len(last_100)), last_100, 1)[0]
           if recent_trend > 0.01:  # Loss increasing
               raise RuntimeError(f"Loss diverging: trend={recent_trend:.4f}")

       # Issue 2: Plateau (no improvement for 50 steps)
       if len(last_100) >= 50:
           recent_variance = np.var(last_100[-50:])
           if recent_variance < 0.001:  # No change
               print(f"⚠️ Loss plateaued: variance={recent_variance:.6f}")

       # Issue 3: NaN
       if np.isnan(last_100[-1]):
           raise RuntimeError("Loss is NaN")
   ```

4. **Adaptive learning rate**:
   ```python
   scheduler = ReduceLROnPlateau(
       optimizer,
       mode='min',
       factor=0.5,
       patience=5,
       verbose=True
   )
   ```

5. **Restart from checkpoint**:
   ```python
   if training_fails:
       # Load last good checkpoint
       model.load_state_dict(torch.load("checkpoint_step_X.pt"))
       # Reduce LR by 2x
       for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.5
       # Resume training
   ```

**Detection**: Monitor W&B dashboard for loss trends in real-time.

---

## Phase 2: EvoMerge - Evolutionary Optimization

### Risk 2.1: Population Converges Prematurely 🟡 HIGH

**Failure Mode**: All 8 models become identical after 15 generations, no further improvement

**Impact**:
- Evolution stagnates
- Remaining 35 generations wasted
- Final model not significantly better than initial

**Probability**: 35% (genetic algorithms prone to premature convergence)

**Root Causes**:
1. **Mutation rate too low** (0.001 = barely any variation)
2. **Elite preservation too aggressive** (top 6 always kept)
3. **Loser merging not diverse enough** (same combos repeated)
4. **Selection pressure too high** (only top 2 reproduce)

**Mitigation**:
1. **Diversity-based selection**:
   ```python
   def select_diverse_elites(population, fitness_scores, diversity_threshold=0.3):
       """Select elites that are both fit AND diverse"""

       sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)

       elites = [sorted_pop[0][0]]  # Best model always included

       for model, fitness in sorted_pop[1:]:
           # Check diversity from existing elites
           diversity = min(compute_distance(model, elite) for elite in elites)

           if diversity > diversity_threshold:
               elites.append(model)

           if len(elites) >= 2:
               break

       return elites
   ```

2. **Adaptive mutation rate**:
   ```python
   def calculate_mutation_rate(generation, diversity):
       """Increase mutation when diversity drops"""
       base_rate = 0.01

       if diversity < 0.3:
           # Low diversity, increase mutation
           rate = base_rate * 2.0
       elif diversity > 0.5:
           # High diversity, decrease mutation
           rate = base_rate * 0.5
       else:
           rate = base_rate

       # Also increase mutation later in evolution (exploration)
       if generation > 30:
           rate *= 1.5

       return rate
   ```

3. **Inject random models**:
   ```python
   if generation % 10 == 0 and diversity < 0.3:
       # Every 10 generations, if diversity low, inject random model
       print(f"⚠️ Diversity too low ({diversity:.2f}), injecting random model")
       random_combo = random.randint(0, 7)
       population[-1] = apply_merge_combo([model1, model2, model3], random_combo)
   ```

4. **Monitor diversity metric**:
   ```python
   def compute_diversity(population):
       """Average pairwise distance"""
       distances = []
       for i, model_i in enumerate(population):
           for j, model_j in enumerate(population):
               if i < j:
                   dist = cosine_distance(get_weights_flat(model_i), get_weights_flat(model_j))
                   distances.append(dist)

       return np.mean(distances)
   ```

**Detection**: Plot diversity over generations in W&B. If drops below 0.25, trigger intervention.

---

### Risk 2.2: Merge Techniques Create Degenerate Models 🟡 HIGH

**Failure Mode**: SLERP or TIES creates model with NaN weights or dimension mismatches

**Impact**:
- Crash during fitness evaluation
- Invalid model in population
- Need to regenerate, wastes time

**Probability**: 20% (merge math can fail edge cases)

**Root Causes**:
1. **SLERP with identical models** (θ=0, division by zero)
2. **TIES with all-zero deltas** (no parameters to merge)
3. **FrankenMerge with incompatible layers** (different hidden dims)
4. **Numerical instability** (float32 overflow)

**Mitigation**:
1. **Validate after merge**:
   ```python
   def is_valid_model(model):
       """Check if model is valid after merge"""

       for name, param in model.named_parameters():
           # Check for NaN
           if torch.isnan(param).any():
               print(f"❌ {name} has NaN values")
               return False

           # Check for Inf
           if torch.isinf(param).any():
               print(f"❌ {name} has Inf values")
               return False

           # Check for zero variance (all same value)
           if param.numel() > 1 and param.var() < 1e-8:
               print(f"⚠️ {name} has zero variance")
               # Not fatal, but suspicious

       # Test forward pass
       try:
           test_input = torch.randn(1, 512).to(model.device)
           with torch.no_grad():
               output = model(test_input)
           if torch.isnan(output).any():
               print(f"❌ Output is NaN")
               return False
       except Exception as e:
           print(f"❌ Forward pass failed: {str(e)}")
           return False

       return True
   ```

2. **Fallback to safe merge**:
   ```python
   def safe_merge(models, combo_id):
       """Try combo_id, fallback to linear if fails"""
       try:
           merged = apply_merge_combo(models, combo_id)
           if is_valid_model(merged):
               return merged
           else:
               print(f"⚠️ Combo {combo_id} created invalid model, falling back to linear")
               return linear_merge(models[0], models[1], models[2])
       except Exception as e:
           print(f"❌ Combo {combo_id} failed: {str(e)}, falling back to linear")
           return linear_merge(models[0], models[1], models[2])
   ```

3. **Numerical stability**:
   ```python
   # SLERP with epsilon
   theta = torch.arccos(torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7))
   if theta < 1e-6:  # Models too similar
       print("⚠️ Models nearly identical, using linear instead of SLERP")
       return linear_merge(model1, model2)
   ```

**Detection**: Run `is_valid_model()` after every merge, before fitness evaluation.

---

### Risk 2.3: Fitness Evaluation is Too Slow 🟢 MEDIUM

**Failure Mode**: Evaluating 8 models takes >10 minutes per generation, 50 generations = 8+ hours

**Impact**:
- Evolution takes much longer than expected (90 min → 8 hours)
- Blocks subsequent phases
- Expensive GPU time

**Probability**: 40% (fitness eval can be slow)

**Root Causes**:
1. **Full test set evaluated** (10,000 samples)
2. **No caching** (re-evaluate same models)
3. **Sequential evaluation** (1 model at a time)
4. **Slow perplexity calculation** (full forward pass)

**Mitigation**:
1. **Sample-based fitness**:
   ```python
   def fast_fitness_evaluation(model, test_set, sample_size=1000):
       """Evaluate on subset for speed"""
       sampled_test = random.sample(test_set, sample_size)
       return evaluate_fitness(model, sampled_test)
   ```

2. **Cache fitness scores**:
   ```python
   fitness_cache = {}  # model_hash → fitness

   def cached_fitness(model):
       model_hash = hash_model_weights(model)
       if model_hash in fitness_cache:
           return fitness_cache[model_hash]  # Already evaluated
       else:
           fitness = evaluate_fitness(model)
           fitness_cache[model_hash] = fitness
           return fitness
   ```

3. **Parallel evaluation**:
   ```python
   from concurrent.futures import ThreadPoolExecutor

   with ThreadPoolExecutor(max_workers=4) as executor:
       fitness_scores = list(executor.map(evaluate_fitness, population))
   ```

4. **Approximate perplexity**:
   ```python
   # Use last 100 tokens instead of full sequence
   def fast_perplexity(model, text):
       tokens = tokenizer(text)[-100:]  # Last 100 only
       return calculate_perplexity(model, tokens)
   ```

**Detection**: Time each fitness evaluation. If >1 minute per model, trigger optimization.

---

## Phase 3: Quiet-STaR - Reasoning Enhancement

### Risk 3.1: Frontier Model Data Generation Fails 🔴 CRITICAL

**Failure Mode**: OpenRouter API fails, returns errors, or costs exceed budget

**Impact**:
- No reasoning data for Step 1 (prompt baking)
- Phase 3 blocked entirely
- Timeline delay: 1-2 weeks to resolve

**Probability**: 25% (API reliability unknown)

**Root Causes**:
1. **API rate limits** (60 RPM exceeded)
2. **API downtime** (Claude or GPT-4 unavailable)
3. **Cost explosion** ($127 → $500+)
4. **Poor quality outputs** (models don't follow format)

**Mitigation**:
1. **Pre-generate data** before Phase 3:
   ```python
   # Run data generation BEFORE Phase 3 starts
   # Store locally, don't regenerate each time
   python phase3_data_generator.py --output data/reasoning_25k.json
   ```

2. **Implement retry with exponential backoff**:
   ```python
   @retry(tries=5, delay=2, backoff=2, max_delay=60)
   async def api_call_with_retry(prompt):
       response = await openrouter_client.generate(prompt)
       return response
   ```

3. **Rate limiting**:
   ```python
   rate_limiter = AsyncRateLimiter(max_rate=50, time_period=60)  # 50 per minute
   async with rate_limiter:
       response = await api_call(prompt)
   ```

4. **Cost monitoring**:
   ```python
   total_cost = 0.0
   cost_limit = 200.0  # $200 budget

   for batch in batches:
       batch_cost = await generate_batch(batch)
       total_cost += batch_cost

       if total_cost > cost_limit:
           raise RuntimeError(f"Cost limit exceeded: ${total_cost:.2f} > ${cost_limit}")

       print(f"💰 Cost so far: ${total_cost:.2f} / ${cost_limit}")
   ```

5. **Quality validation**:
   ```python
   def validate_generated_example(example):
       """Ensure example follows format"""

       # Check required fields
       assert "reasoning" in example, "Missing reasoning"
       assert "answer" in example, "Missing answer"

       # Check token structure
       assert "[thinking]" in example["reasoning"], "Missing [thinking] token"
       assert "[/endthinking]" in example["reasoning"], "Missing [/endthinking] token"

       # Check strategy tokens
       strategy_tokens = ["<step>", "<mece>", "<falsify>", "<expert>", "<orthogonal>", "<doubt>", "<bayesian>", "<multidomain>", "<correct>", "<uncertain>"]
       has_strategy = any(token in example["reasoning"] for token in strategy_tokens)
       assert has_strategy, "Missing strategy tokens"

       return True
   ```

6. **Fallback to local generation**:
   ```python
   if openrouter_fails:
       # Use local Llama 3 70B or Mixtral 8x7B to generate examples
       # Quality lower, but unblocked
       print("⚠️ Using local model for generation (lower quality)")
       examples = generate_locally(prompts)
   ```

**Detection**: Monitor API success rate. If <90%, investigate immediately.

---

### Risk 3.2: Prompt Baking Doesn't Converge 🟡 HIGH

**Failure Mode**: Step 1 training doesn't reach 85% accuracy threshold after 5 epochs

**Impact**:
- Cannot proceed to Step 2 (Quiet-STaR)
- Need to retrain with different hyperparameters
- Wasted 6 hours of training

**Probability**: 30% (baking is supervised, but can fail)

**Root Causes**:
1. **Learning rate too low** (1e-5 learns too slowly)
2. **Data quality poor** (frontier models generated bad examples)
3. **Model capacity insufficient** (25M params not enough)
4. **Convergence threshold too strict** (85% impossible for this task)

**Mitigation**:
1. **Staged convergence threshold**:
   ```python
   # Relax threshold for harder strategies
   CONVERGENCE_THRESHOLDS = {
       "chain_of_thought": 0.90,  # Easy
       "mece": 0.85,
       "falsification": 0.80,     # Harder
       "bayesian": 0.75,          # Hardest
       "overall": 0.85            # Average
   }

   def validate_convergence(model, reasoning_data):
       accuracies = {}
       for strategy in STRATEGIES:
           strategy_data = filter_by_strategy(reasoning_data, strategy)
           accuracy = evaluate(model, strategy_data)
           accuracies[strategy] = accuracy

           threshold = CONVERGENCE_THRESHOLDS.get(strategy, 0.85)
           if accuracy < threshold:
               print(f"⚠️ {strategy}: {accuracy:.2%} < {threshold:.2%}")

       overall = np.mean(list(accuracies.values()))
       print(f"Overall: {overall:.2%}")

       return overall >= 0.85
   ```

2. **Learning rate finder**:
   ```python
   from torch_lr_finder import LRFinder

   lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
   lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
   lr_finder.plot()  # Find optimal LR
   optimal_lr = lr_finder.lr_suggestion()
   ```

3. **Curriculum learning**:
   ```python
   # Train on easy strategies first
   easy_strategies = ["chain_of_thought", "self_doubt"]
   hard_strategies = ["bayesian", "falsification"]

   # Epochs 1-2: Easy strategies
   train(model, filter_by_strategies(data, easy_strategies), epochs=2)

   # Epochs 3-5: All strategies
   train(model, data, epochs=3)
   ```

4. **Data augmentation**:
   ```python
   # If convergence slow, augment data
   augmented_data = []
   for example in reasoning_data:
       augmented_data.append(example)  # Original

       # Paraphrase
       paraphrased = paraphrase(example["prompt"])
       augmented_data.append({...paraphrased...})

   # Train on 2x data
   ```

**Detection**: Monitor validation accuracy after each epoch. If not improving for 2 epochs, intervene.

---

### Risk 3.3: Quiet-STaR Generates "Theater" (Empty Reasoning) 🔴 CRITICAL

**Failure Mode**: Step 2 RL training succeeds, but thoughts are trivial/useless (e.g., "[thinking] hmm [/endthinking]")

**Impact**:
- Model appears to reason, but doesn't actually
- Passes naive tests, fails anti-theater validation
- Need to restart Phase 3 entirely

**Probability**: 35% (RL can find shortcuts)

**Root Causes**:
1. **Reward function too lenient** (any thought gets positive reward)
2. **KL coefficient too low** (model drifts from baked baseline)
3. **Thought length too short** (1-2 tokens, not meaningful)
4. **No diversity penalty** (all thoughts identical)

**Mitigation**:
1. **Strict anti-theater tests** (run DURING training, not just after):
   ```python
   def anti_theater_check(model, test_samples):
       """Run every 1000 steps during RL training"""

       # Test 1: Divergence from direct continuation
       divergence_scores = []
       for sample in test_samples:
           direct = model.generate(sample, use_thoughts=False)
           with_thoughts = model.generate(sample, use_thoughts=True)
           divergence = edit_distance(direct, with_thoughts) / len(direct)
           divergence_scores.append(divergence)

       avg_divergence = np.mean(divergence_scores)
       if avg_divergence < 0.3:
           print(f"🚨 THEATER DETECTED: divergence={avg_divergence:.2f} < 0.3")
           return False

       # Test 2: Thought length
       thought_lengths = []
       for sample in test_samples:
           thoughts = generate_thoughts(model, sample)
           avg_len = np.mean([len(t) for t in thoughts])
           thought_lengths.append(avg_len)

       avg_thought_length = np.mean(thought_lengths)
       if avg_thought_length < 5.0:  # At least 5 tokens
           print(f"🚨 THEATER DETECTED: thought_length={avg_thought_length:.1f} < 5.0")
           return False

       # Test 3: Thought diversity (within a single generation)
       diversity_scores = []
       for sample in test_samples:
           thoughts = generate_thoughts(model, sample, num=4)
           pairwise_dists = [edit_distance(thoughts[i], thoughts[j])
                             for i in range(len(thoughts))
                             for j in range(i+1, len(thoughts))]
           diversity = np.mean(pairwise_dists)
           diversity_scores.append(diversity)

       avg_diversity = np.mean(diversity_scores)
       if avg_diversity < 3.0:  # Thoughts differ by at least 3 tokens
           print(f"🚨 THEATER DETECTED: diversity={avg_diversity:.1f} < 3.0")
           return False

       print(f"✅ Anti-theater passed: divergence={avg_divergence:.2f}, length={avg_thought_length:.1f}, diversity={avg_diversity:.1f}")
       return True

   # Run every 1000 steps
   if step % 1000 == 0:
       if not anti_theater_check(model, validation_set):
           print("⚠️ Stopping RL training, theater detected")
           # Rollback to previous checkpoint
           model.load_state_dict(torch.load("checkpoint_before_theater.pt"))
           # Adjust hyperparameters
           kl_coefficient *= 2.0  # Increase KL (tie model to baseline)
           reward_threshold *= 1.5  # Make rewards stricter
           # Resume training
   ```

2. **Reward shaping**:
   ```python
   def calculate_reward(correct, thought_length, thought_diversity, coherence):
       """Multi-component reward"""

       if correct:
           base_reward = 1.0
       else:
           base_reward = 0.0

       # Bonus for longer thoughts (up to a point)
       length_bonus = min(thought_length / 10.0, 1.0) * 0.2

       # Bonus for diverse thoughts
       diversity_bonus = min(thought_diversity / 5.0, 1.0) * 0.2

       # Penalty for low coherence
       coherence_penalty = -0.5 if coherence < 0.5 else 0.0

       total_reward = base_reward + length_bonus + diversity_bonus + coherence_penalty

       return total_reward
   ```

3. **Minimum thought length**:
   ```python
   def generate_thoughts(model, hidden_state, num_thoughts=4, min_length=5):
       """Enforce minimum length"""
       thoughts = []
       for _ in range(num_thoughts):
           thought = model.generate(hidden_state, max_length=20)

           # Reject if too short
           if len(thought) < min_length:
               # Regenerate with higher temperature
               thought = model.generate(hidden_state, max_length=20, temperature=1.5)

           thoughts.append(thought)
       return thoughts
   ```

**Detection**: Run anti-theater tests every 1000 steps. If fails, stop immediately.

---

## Phase 4: BitNet - Compression

### Risk 4.1: Compression Ratio Lower Than Expected 🟢 MEDIUM

**Failure Mode**: Achieve 5x compression instead of 8x due to model size composition

**Impact**:
- Model still works, but larger than planned
- May not fit in target deployment (e.g., mobile)
- Need to adjust Phase 5-8 expectations

**Probability**: 50% (compression varies by model architecture)

**Root Causes**:
1. **Embeddings are large % of model** (20% for tiny models vs 5% for large)
2. **Embeddings preserved in FP16** (not quantized)
3. **More special tokens than expected** (50,269 vocab vs 50,257)
4. **Low sparsity** (25% instead of 35%)

**Mitigation**:
1. **Accept lower compression for tiny models**:
   ```python
   def adjust_compression_expectations(model_size_category):
       """Set realistic targets based on size"""
       if model_size_category == "tiny":
           target_compression = 6.0  # Lower target
           print("ℹ️ Tiny model: embedding overhead limits compression to ~6x")
       elif model_size_category == "small":
           target_compression = 8.0  # Standard
       elif model_size_category == "medium":
           target_compression = 10.0  # Higher
       else:
           target_compression = 12.0  # Highest

       return target_compression
   ```

2. **Quantize embeddings to int8** (risky but higher compression):
   ```python
   if compression_ratio < target and model_size_category == "tiny":
       print("⚠️ Quantizing embeddings to int8 for higher compression (may hurt accuracy)")
       embeddings = model.get_input_embeddings()
       quantized_embeddings = quantize_to_int8(embeddings.weight)
       embeddings.weight.data = quantized_embeddings
   ```

3. **Increase sparsity threshold**:
   ```python
   if sparsity < 0.30:
       print(f"⚠️ Low sparsity ({sparsity:.1%}), increasing threshold")
       sparsity_threshold *= 1.5  # 0.1 → 0.15
       # Re-quantize
   ```

4. **Report actual compression**:
   ```python
   # Don't fail if compression is lower, just report
   print(f"Compression: {actual_compression:.2f}x (target: {target_compression}x)")
   wandb.log({"compression_ratio": actual_compression, "target": target_compression})
   ```

**Detection**: Calculate compression after quantization, compare to target.

---

### Risk 4.2: Accuracy Drop Exceeds 10% 🔴 CRITICAL

**Failure Mode**: After quantization, GSM8K accuracy drops from 18% → 8% (>50% relative drop)

**Impact**:
- Model unusable
- Need to re-compress with conservative settings
- Wasted 4 hours of compression + fine-tuning

**Probability**: 20% (aggressive quantization can hurt)

**Root Causes**:
1. **Sparsity threshold too high** (0.2 → 45% of weights become 0)
2. **Insufficient calibration** (500 samples not enough)
3. **Important layers quantized** (output layer should be preserved)
4. **Fine-tuning failed** (didn't recover from quantization error)

**Mitigation**:
1. **Pre-quantization evaluation**:
   ```python
   baseline_accuracy = evaluate_accuracy(model_before_quantization)
   print(f"Baseline accuracy: {baseline_accuracy:.2%}")

   # Set acceptable drop threshold
   max_acceptable_drop = baseline_accuracy * 0.10  # 10% relative drop
   ```

2. **Staged quantization**:
   ```python
   # Quantize one layer at a time, check accuracy after each
   for i, layer in enumerate(model.layers):
       layer_before = copy.deepcopy(layer)
       quantize_layer(layer)

       accuracy_after = evaluate_accuracy(model)
       drop = baseline_accuracy - accuracy_after

       if drop > max_acceptable_drop:
           print(f"⚠️ Layer {i} caused {drop:.2%} drop, reverting")
           model.layers[i] = layer_before  # Revert
       else:
           print(f"✅ Layer {i} quantized: accuracy drop {drop:.2%}")
   ```

3. **Conservative fallback**:
   ```python
   if accuracy_drop > 0.10:
       print(f"❌ Accuracy drop too high ({accuracy_drop:.2%}), using conservative config")

       # Revert to full precision
       model.load_state_dict(torch.load("model_before_quantization.pt"))

       # Re-quantize with conservative settings
       config = {
           "sparsity_threshold": 0.05,  # Very low
           "preserve_layers": ["embeddings", "lm_head", "layer_norm", "output_layers[-1]"],  # Preserve more
           "target_compression": 4.0  # Lower target
       }

       model = compress(model, config)
       accuracy_after = evaluate_accuracy(model)
       print(f"Conservative compression: {accuracy_after:.2%} (drop: {baseline_accuracy - accuracy_after:.2%})")
   ```

4. **Extended fine-tuning**:
   ```python
   if accuracy_drop > 0.05:
       print(f"Accuracy drop {accuracy_drop:.2%} > 5%, fine-tuning for longer")
       fine_tune(model, epochs=5, lr=1e-5)  # More epochs, lower LR
   ```

**Detection**: Evaluate accuracy immediately after quantization, before saving.

---

## Infrastructure Failures

### Risk I.1: Weights & Biases (W&B) Connection Fails 🟢 MEDIUM

**Failure Mode**: W&B offline/unavailable, logs not saved

**Impact**:
- No experiment tracking
- Cannot visualize training progress
- Loss of metrics data

**Probability**: 15% (W&B usually reliable)

**Mitigation**:
1. **Offline mode**:
   ```python
   wandb.init(project="agent-forge-v2", mode="offline")
   # Logs saved locally, synced later
   ```

2. **Fallback to local logging**:
   ```python
   if not wandb.run:
       # W&B failed, use local logging
       logger = LocalLogger("logs/phase1_run1.json")
       logger.log({"step": step, "loss": loss})
   ```

3. **Retry sync**:
   ```python
   # After run completes
   if wandb.run.mode == "offline":
       wandb.sync("wandb/offline-run-xyz")  # Sync later
   ```

**Detection**: Check `wandb.run` is not None after init.

---

### Risk I.2: Storage Full During Training 🔴 CRITICAL

**Failure Mode**: Disk runs out of space mid-training, crashes

**Impact**:
- Training halts immediately
- Checkpoint save fails
- Data loss if not handled

**Probability**: 25% (storage management often overlooked)

**Mitigation**:
1. **Pre-flight check**:
   ```python
   def check_storage_requirements(phase):
       """Estimate storage needed"""
       requirements = {
           "phase1": 10.0,  # GB (3 models + datasets + checkpoints)
           "phase2": 5.0,
           "phase3": 8.0,
           "phase4": 3.0
       }

       required_gb = requirements[phase]
       available_gb = shutil.disk_usage("/").free / (1024**3)

       if available_gb < required_gb * 1.5:  # 1.5x safety margin
           raise RuntimeError(f"Insufficient storage: {available_gb:.1f} GB available, {required_gb * 1.5:.1f} GB required")

       print(f"✅ Storage OK: {available_gb:.1f} GB available, {required_gb:.1f} GB required")
   ```

2. **Monitor during training**:
   ```python
   def monitor_storage():
       """Check every 1000 steps"""
       free_gb = shutil.disk_usage("/").free / (1024**3)

       if free_gb < 2.0:  # <2GB free
           print(f"🚨 Low storage: {free_gb:.1f} GB free")
           # Delete old checkpoints
           delete_old_checkpoints(keep_last=3)

       if free_gb < 0.5:  # <500MB free
           raise RuntimeError(f"Out of storage: {free_gb:.1f} GB free")
   ```

3. **Delete intermediate files**:
   ```python
   # After Phase 2 completes
   delete(phase1_models)  # 5 GB freed

   # After Phase 4 completes
   delete(phase2_champion)  # 1.7 GB freed
   delete(phase3_reasoning_uncompressed)  # 1.8 GB freed
   ```

**Detection**: Monitor disk usage in system metrics.

---

### Risk I.3: GPU Availability Issues 🟡 HIGH

**Failure Mode**: GPU claimed by another process, or driver crash

**Impact**:
- Training blocked until GPU available
- Timeline delay
- Possible data loss if crash during training

**Probability**: 30% (GPU sharing issues common)

**Mitigation**:
1. **GPU availability check**:
   ```python
   def wait_for_gpu(timeout=3600):
       """Wait up to 1 hour for GPU"""
       start_time = time.time()

       while time.time() - start_time < timeout:
           if torch.cuda.is_available():
               # Check if GPU has enough memory
               gpu_mem_free = torch.cuda.get_device_properties(0).total_memory
               gpu_mem_allocated = torch.cuda.memory_allocated(0)
               gpu_mem_available = gpu_mem_free - gpu_mem_allocated

               if gpu_mem_available > 4 * 1024**3:  # >4GB free
                   print(f"✅ GPU available: {gpu_mem_available / 1024**3:.1f} GB free")
                   return True

           print(f"⏳ Waiting for GPU... ({int(time.time() - start_time)}s elapsed)")
           time.sleep(60)  # Check every minute

       raise RuntimeError("GPU not available after timeout")
   ```

2. **CPU fallback**:
   ```python
   device = "cuda" if torch.cuda.is_available() else "cpu"
   if device == "cpu":
       print("⚠️ GPU not available, using CPU (will be slower)")
   model.to(device)
   ```

3. **Auto-resume**:
   ```python
   # Save checkpoint before GPU operations
   torch.save(model.state_dict(), "checkpoint_pre_training.pt")

   try:
       train(model)
   except RuntimeError as e:
       if "CUDA" in str(e):
           print(f"🚨 GPU error: {str(e)}")
           print("Reloading checkpoint and retrying on CPU")
           model.load_state_dict(torch.load("checkpoint_pre_training.pt"))
           model.to("cpu")
           train(model)  # Resume on CPU
   ```

**Detection**: Catch CUDA errors, implement retry logic.

---

## Summary: Critical Risks by Severity

### 🔴 CRITICAL (Project Blockers)
1. **Phase 1**: Dataset download failures
2. **Phase 3**: Frontier model data generation fails
3. **Phase 3**: Quiet-STaR generates theater
4. **Phase 4**: Accuracy drop >10%
5. **Infrastructure**: Storage full during training

### 🟡 HIGH (Major Setbacks)
1. **Phase 1**: Models don't diversify
2. **Phase 1**: OOM during training
3. **Phase 1**: Training doesn't converge
4. **Phase 2**: Population converges prematurely
5. **Phase 2**: Merge techniques create degenerate models
6. **Phase 3**: Prompt baking doesn't converge
7. **Infrastructure**: GPU availability issues

### 🟢 MEDIUM (Minor Issues)
1. **Phase 2**: Fitness evaluation too slow
2. **Phase 4**: Compression ratio lower than expected
3. **Infrastructure**: W&B connection fails

---

**Version**: 2.0
**Date**: 2025-10-15
**Status**: ✅ Complete - Ready for Implementation

**Next Actions**:
1. Review all mitigations
2. Implement detection code for each risk
3. Test fallback strategies before Phase 1 start
4. Set up monitoring dashboards (W&B, system metrics)
5. Create runbooks for each critical failure mode
