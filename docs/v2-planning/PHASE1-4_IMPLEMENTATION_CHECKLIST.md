# Phase 1-4: Implementation Checklist with Premortem Checkpoints

**Version**: 2.0
**Date**: 2025-10-15
**Purpose**: Step-by-step implementation guide with built-in failure detection and mitigation

---

## Pre-Implementation (Do BEFORE Phase 1)

### Environment Setup

- [ ] **GPU Environment Validated**
  ```bash
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
  ```
  - Minimum: 6GB VRAM (GTX 1660 or better)
  - Recommended: 8GB+ VRAM

- [ ] **Python Dependencies Installed**
  ```bash
  pip install torch transformers datasets wandb numpy scipy
  pip install -U huggingface_hub  # Latest version
  ```

- [ ] **Storage Validated**
  ```python
  import shutil
  free_gb = shutil.disk_usage("/").free / (1024**3)
  assert free_gb > 15, f"Need 15GB free, have {free_gb:.1f}GB"
  ```
  - Minimum: 15GB free
  - Recommended: 50GB free

- [ ] **W&B Account Setup**
  ```bash
  wandb login
  wandb init --project agent-forge-v2
  ```

### Dataset Pre-Download (🔴 CRITICAL)

- [ ] **Test HuggingFace Connection**
  ```python
  from datasets import load_dataset
  test_dataset = load_dataset("gsm8k", "main", split="train[:10]")
  print(f"✅ HuggingFace accessible, loaded {len(test_dataset)} samples")
  ```

- [ ] **Download All 16 Datasets**
  ```bash
  python phases/phase1/download_datasets.py --output data/cognate_datasets
  ```
  - Expected time: ~10 minutes (50 Mbps)
  - Expected storage: ~1.35 GB

- [ ] **Validate Dataset Integrity**
  ```python
  # Check all 16 datasets present
  required = [
      "GSM8K", "SVAMP", "ASDiv", "Mini-MBPP", "CodeXGLUE",
      "ARC-Easy", "ARC-Challenge", "HotpotQA", "DROP", "StrategyQA",
      "PIQA", "HellaSwag", "BoolQ", "WikiText", "FineWeb-Edu"
  ]

  for dataset_name in required:
      path = f"data/cognate_datasets/{dataset_name}_train.json"
      assert os.path.exists(path), f"Missing {dataset_name}"
      with open(path) as f:
          data = json.load(f)
      print(f"✅ {dataset_name}: {len(data)} samples")
  ```

- [ ] **Backup Datasets**
  ```bash
  # Create backup on external drive or cloud
  tar -czf cognate_datasets_backup.tar.gz data/cognate_datasets/
  ```

### Utilities Implementation

- [ ] **Model Size Detection**
  ```python
  # Implement get_model_size() from comprehensive plan
  # Test with dummy model
  dummy_model = nn.Linear(512, 512)
  size_info = get_model_size(dummy_model)
  assert "params" in size_info
  assert "size_mb" in size_info
  assert "size_category" in size_info
  ```

- [ ] **Diversity Validation**
  ```python
  # Implement validate_diversity() from comprehensive plan
  # Test with 3 dummy models
  ```

- [ ] **VRAM Calculator**
  ```python
  # Implement calculate_safe_batch_size() from comprehensive plan
  # Test with current GPU
  batch_size, acc_steps = calculate_safe_batch_size(dummy_model, 6)  # 6GB GPU
  print(f"Safe batch size: {batch_size}, accumulation: {acc_steps}")
  ```

---

## Phase 1: Cognate Implementation

### Pre-Flight Checks

- [ ] **Dataset Availability** (🔴 CRITICAL)
  ```python
  validate_dataset_availability()  # From comprehensive plan
  # Must pass before proceeding
  ```

- [ ] **Storage Check**
  ```python
  check_storage_requirements("phase1")  # Need 10GB
  ```

- [ ] **GPU Check**
  ```python
  wait_for_gpu(timeout=3600)  # Wait up to 1 hour
  ```

### Model 1: Reasoning-Focused

- [ ] **Initialize Model**
  ```python
  model1 = TRMTitansMAG(
      num_layers=8,
      hidden_dim=512,
      num_heads=8,
      act_threshold=0.95,  # Thinks LONG
      ltm_slots=2048,       # SMALL memory
      vocab_size=50257
  )

  size_info = get_model_size(model1)
  print(f"Model 1: {size_info['params']:,} params ({size_info['size_mb']:.1f} MB)")
  assert size_info['size_category'] == "tiny"  # Expected for 25M model
  ```

- [ ] **Calculate Safe Batch Size**
  ```python
  batch_size, acc_steps = calculate_safe_batch_size(model1, device_vram_gb=6)
  print(f"Model 1 batch config: batch_size={batch_size}, accumulation={acc_steps}")
  ```

- [ ] **Initialize Optimizer**
  ```python
  optimizer = MuonGrokfast(
      model1.parameters(),
      muon_lr=1e-3,
      grokfast_lambda=0.3,
      qk_clip_threshold=30.0,
      kl_coefficient=0.0
  )
  ```

- [ ] **Initialize W&B**
  ```python
  wandb.init(
      project="agent-forge-v2",
      name="phase1-cognate-model1-reasoning",
      config={
          "model_id": 1,
          "act_threshold": 0.95,
          "ltm_slots": 2048,
          "batch_size": batch_size,
          # ... full config
      }
  )
  ```

- [ ] **Load Training Data (Curriculum Stage 1)**
  ```python
  # Foundation: GSM8K, SVAMP, Mini-MBPP
  dataset = Phase1Dataset(
      ["data/cognate_datasets/GSM8K_train.json",
       "data/cognate_datasets/SVAMP_train.json",
       "data/cognate_datasets/Mini-MBPP_train.json"],
      tokenizer,
      dataset_weights={"math": 0.4, "qa": 0.3, "code": 0.1, "commonsense": 0.1, "language": 0.1}
  )
  ```

- [ ] **Training Loop (Epochs 1-3)**
  ```python
  loss_history = []

  for epoch in range(1, 4):  # Stage 1: Foundation
      for batch in dataloader:
          # Forward
          output = model1(batch["input_ids"])
          loss = criterion(output, batch["labels"])
          loss_history.append(loss.item())

          # Detect issues
          if len(loss_history) >= 100:
              detect_training_issues(loss_history)  # Will raise if diverging

          # Backward
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
          optimizer.step()
          optimizer.zero_grad()

          # Log to W&B
          if step % 100 == 0:
              wandb.log({
                  "train/loss": loss.item(),
                  "train/perplexity": torch.exp(loss).item(),
                  # ... all metrics from W&B integration plan
              })

          # Checkpoint
          if step % 1000 == 0:
              torch.save(model1.state_dict(), f"checkpoints/model1_step{step}.pt")

      # End of epoch validation
      val_loss, val_acc = validate(model1, val_dataset)
      wandb.log({"val/loss": val_loss, "val/accuracy": val_acc})
  ```

- [ ] **Curriculum Stage 2 (Epochs 4-6)**
  ```python
  # Load Reasoning datasets: ARC-Easy, ARC-Challenge, PIQA, WikiText
  dataset = Phase1Dataset([...], tokenizer, ...)
  # Train for 3 more epochs
  ```

- [ ] **Curriculum Stage 3 (Epochs 7-10)**
  ```python
  # Load Advanced datasets: HotpotQA, DROP, HellaSwag, FineWeb-Edu
  dataset = Phase1Dataset([...], tokenizer, ...)
  # Train for 4 more epochs
  ```

- [ ] **Final Model 1 Validation**
  ```python
  final_metrics = {
      "perplexity": evaluate_perplexity(model1),
      "gsm8k_acc": evaluate_gsm8k(model1),
      "arc_acc": evaluate_arc(model1),
      "halting_steps": measure_avg_halting_steps(model1),
      "ltm_usage": measure_ltm_usage(model1),
      "inference_time": measure_inference_time(model1)
  }

  print(f"Model 1 Final: {final_metrics}")
  wandb.log({"final/": final_metrics})

  # Save final model
  torch.save(model1.state_dict(), "phase1_outputs/model1_reasoning.pt")
  ```

### Model 2: Memory-Focused

- [ ] **Initialize Model** (ACT 0.85, LTM 8192)
- [ ] **Calculate Safe Batch Size**
- [ ] **Training (same process as Model 1)**
  - Different dataset mix (language-heavy)
  - Different random seed (43)
  - Different epochs (12 instead of 10)

### Model 3: Speed-Focused

- [ ] **Initialize Model** (ACT 0.99, LTM 2048)
- [ ] **Calculate Safe Batch Size**
- [ ] **Training (same process as Model 1)**
  - Different dataset mix (commonsense-heavy)
  - Different random seed (44)
  - Different epochs (8 instead of 10)

### Phase 1 Post-Training Validation

- [ ] **Diversity Check** (🟡 HIGH RISK)
  ```python
  validate_diversity(model1, model2, model3)
  # Will raise if diversity <0.3
  # If fails: Retrain with more aggressive config differences
  ```

- [ ] **Success Criteria**
  ```python
  assert final_metrics["gsm8k_acc"] > 0.10  # >10%
  assert final_metrics["halting_diversity"] > 2.0
  assert final_metrics["memory_diversity"] > 0.3
  assert final_metrics["speed_diversity"] > 10.0  # ms
  ```

- [ ] **W&B Summary**
  ```python
  # Create diversity comparison table
  diversity_table = wandb.Table(...)
  wandb.log({"diversity/model_comparison": diversity_table})
  wandb.finish()
  ```

---

## Phase 2: EvoMerge Implementation

### Pre-Flight Checks

- [ ] **Load Phase 1 Models**
  ```python
  model1 = load_model("phase1_outputs/model1_reasoning.pt")
  model2 = load_model("phase1_outputs/model2_memory.pt")
  model3 = load_model("phase1_outputs/model3_speed.pt")
  ```

- [ ] **Validate Diversity** (must have passed Phase 1 check)
  ```python
  diversity = compute_diversity([model1, model2, model3])
  assert diversity > 0.3, f"Insufficient diversity: {diversity}"
  ```

- [ ] **VRAM Check**
  ```python
  # 8 models × 1.7GB = ~13.6GB
  # If <13.6GB available, use CPU fallback
  device = manage_population_memory(population, device_vram_gb=6)
  # Returns "cuda" or "cpu"
  ```

### Evolution Loop

- [ ] **Generation 0: Binary Combinations**
  ```python
  population = []
  for combo_id in range(8):  # 000 to 111
      merged = apply_merge_combo([model1, model2, model3], combo_id)

      # Validate merged model
      if not is_valid_model(merged):
          print(f"⚠️ Combo {combo_id} invalid, using linear fallback")
          merged = linear_merge(model1, model2, model3)

      population.append(merged)

  # Evaluate fitness
  fitness_scores = [evaluate_fitness(m, get_model_size(m)["size_mb"]) for m in population]

  # Log to W&B
  wandb.log({"generation": 0, "best_fitness": max(fitness_scores)})
  ```

- [ ] **Generations 1-50**
  ```python
  for gen in range(1, 51):
      # Elite preservation
      sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
      elites = [sorted_pop[0][0], sorted_pop[1][0]]

      # Check diversity (🟡 HIGH RISK: premature convergence)
      diversity = compute_diversity(population)
      if diversity < 0.25:
          print(f"⚠️ Gen {gen}: Diversity too low ({diversity:.2f}), injecting random model")
          population[-1] = apply_merge_combo([model1, model2, model3], random.randint(0, 7))

      # Loser merging
      losers = [sorted_pop[i][0] for i in range(6, 8)]
      merged_loser = merge(losers[0], losers[1])

      # New population
      population = elites + [merged_loser] + generate_new_models(5)

      # Evaluate
      fitness_scores = [evaluate_fitness(m, ...) for m in population]

      # Early stopping
      improvement = (max(fitness_scores) - initial_fitness) / initial_fitness
      if improvement > 0.30:  # 30% improvement, stop early
          print(f"Early stopping at gen {gen}: {improvement:.1%} improvement")
          break

      # Log
      wandb.log({
          "generation": gen,
          "best_fitness": max(fitness_scores),
          "diversity": diversity,
          "improvement": improvement
      })
  ```

- [ ] **Select Champion**
  ```python
  champion_idx = np.argmax(fitness_scores)
  champion = population[champion_idx]

  # Save
  torch.save(champion.state_dict(), "phase2_outputs/champion_evolved.pt")

  # Validate success criteria
  assert improvement >= 0.20, f"Improvement too low: {improvement:.1%} < 20%"
  ```

---

## Phase 3: Quiet-STaR Implementation

### Pre-Flight Checks

- [ ] **Load Phase 2 Champion**
  ```python
  model = load_model("phase2_outputs/champion_evolved.pt")
  base_size = get_model_size(model)
  print(f"Base model: {base_size['params']:,} params")
  ```

- [ ] **Add Special Tokens**
  ```python
  special_tokens = {
      "additional_special_tokens": [
          "[thinking]", "[/endthinking]",  # Outer wrapper
          "<step>", "<mece>", "<falsify>", "<expert>",  # Strategies 1-4
          "<orthogonal>", "<doubt>", "<bayesian>", "<multidomain>",  # Strategies 5-8
          "<correct>", "<uncertain>"  # Strategies 9-10
      ]
  }

  tokenizer.add_special_tokens(special_tokens)
  model.resize_token_embeddings(len(tokenizer))

  new_size = get_model_size(model)
  print(f"With tokens: {new_size['params']:,} params (+{new_size['params'] - base_size['params']})")
  ```

### Step 0: Data Generation (🔴 CRITICAL)

- [ ] **Pre-Generate Reasoning Data**
  ```bash
  # Run BEFORE Phase 3 starts
  python phases/phase3/phase3_data_generator.py \
      --output data/phase3_reasoning_25k.json \
      --cost-limit 200
  ```
  - Expected cost: $100-200
  - Expected time: 3-4 hours
  - Expected output: 25,000 examples

- [ ] **Validate Generated Data**
  ```python
  with open("data/phase3_reasoning_25k.json") as f:
      data = json.load(f)

  # Check counts
  assert data["metadata"]["total_examples"] >= 20000
  assert len(data["metadata"]["strategies"]) == 10
  assert len(data["special_tokens"]) == 12  # 2 outer + 10 inner

  # Check quality
  for example in data["examples"][:100]:
      assert "[thinking]" in example["reasoning"]
      assert "[/endthinking]" in example["reasoning"]
      # Check has at least one strategy token
      has_strategy = any(tok in example["reasoning"]
                         for tok in ["<step>", "<mece>", "<falsify>", ...])
      assert has_strategy, f"No strategy tokens in {example['id']}"

  print(f"✅ Data validated: {len(data['examples'])} examples")
  ```

### Step 1: Prompt Baking

- [ ] **Load Reasoning Data**
  ```python
  baking_dataset = load_json("data/phase3_reasoning_25k.json")
  train_data, val_data = train_test_split(baking_dataset["examples"], test_size=0.1)
  ```

- [ ] **Initialize Optimizer** (Baking Config)
  ```python
  optimizer = MuonGrokfast(
      model.parameters(),
      muon_lr=1e-4,  # Lower for fine-tuning
      grokfast_lambda=0.2,
      qk_clip_threshold=30.0,
      kl_coefficient=0.0  # No KL for baking
  )
  ```

- [ ] **Training (5 epochs)**
  ```python
  for epoch in range(5):
      for batch in DataLoader(train_data, batch_size=batch_size):
          logits = model(batch["input_ids"])
          loss = F.cross_entropy(logits.view(-1, vocab_size), batch["labels"].view(-1))
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          wandb.log({"baking/loss": loss.item(), "baking/epoch": epoch})

      # Validate convergence
      val_acc = evaluate(model, val_data)
      wandb.log({"baking/val_accuracy": val_acc})

      # Strategy-specific accuracy
      for strategy in STRATEGIES:
          strategy_acc = evaluate(model, filter_by_strategy(val_data, strategy))
          wandb.log({f"baking/accuracy_{strategy}": strategy_acc})
  ```

- [ ] **Convergence Check** (🟡 HIGH RISK)
  ```python
  final_accuracy = evaluate(model, val_data)

  if final_accuracy < 0.85:
      print(f"⚠️ Baking failed: {final_accuracy:.2%} < 85%")
      # Mitigation: Train longer or adjust LR
      optimizer.param_groups[0]['lr'] *= 0.5  # Reduce LR
      train_more_epochs(model, 3)  # Train 3 more epochs
      final_accuracy = evaluate(model, val_data)

  assert final_accuracy >= 0.85, f"Baking failed: {final_accuracy:.2%}"

  # Save baked model
  torch.save(model.state_dict(), "phase3_outputs/baked_model.pt")
  ```

### Step 2: Quiet-STaR (RL)

- [ ] **Load Baked Model**
  ```python
  model = load_model("phase3_outputs/baked_model.pt")
  ```

- [ ] **Initialize Optimizer** (RL Config)
  ```python
  optimizer = MuonGrokfast(
      model.parameters(),
      muon_lr=5e-4,  # HIGHER for RL
      grokfast_lambda=0.1,  # LOWER for RL
      qk_clip_threshold=25.0,  # TIGHTER
      kl_coefficient=0.1  # NEW: Prevent drift
  )
  ```

- [ ] **RL Training Loop**
  ```python
  for episode in range(10000):
      for batch in dataloader:
          for token_pos in batch:
              # Generate thoughts
              thoughts = generate_thoughts(model, token_pos, num=num_thoughts)

              # Score & mix
              scores = [score_coherence(t, token_pos, model) for t in thoughts]
              enhanced_hidden = mix_thoughts(token_pos, thoughts, scores)

              # Predict & reward
              logits = model.predict(enhanced_hidden)
              correct = (logits.argmax() == labels[token_pos])
              reward = 1.0 if correct else 0.0

              # REINFORCE + KL
              loss = -reward * torch.log(torch.tensor(scores).mean())
              kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(base_logits, dim=-1))
              total_loss = loss + 0.1 * kl_loss

              total_loss.backward()

          optimizer.step()
          optimizer.zero_grad()

          # Log
          wandb.log({"rl/reward": reward, "rl/coherence": scores.mean()})

      # Anti-theater check every 1000 episodes (🔴 CRITICAL)
      if episode % 1000 == 0:
          theater_results = anti_theater_check(model, test_set)

          if not theater_results["all_passed"]:
              print(f"🚨 THEATER DETECTED at episode {episode}")
              print(f"   Divergence: {theater_results['divergence']:.2f} (need >0.3)")
              print(f"   Ablation: {theater_results['ablation_drop']:.2f} (need >0.02)")
              print(f"   Correlation: {theater_results['correlation']:.2f} (need >0.5)")

              # Rollback to previous checkpoint
              model.load_state_dict(torch.load(f"checkpoints/rl_episode{episode-1000}.pt"))

              # Adjust hyperparameters
              kl_coefficient *= 2.0  # Increase KL (tie to baseline)
              reward_threshold *= 1.5

              print(f"   Rolled back and adjusted: kl={kl_coefficient}, reward_thresh={reward_threshold}")
          else:
              wandb.log({
                  "anti_theater/divergence": theater_results["divergence"],
                  "anti_theater/ablation": theater_results["ablation_drop"],
                  "anti_theater/correlation": theater_results["correlation"]
              })
  ```

- [ ] **Final Validation**
  ```python
  final_metrics = {
      "gsm8k_acc": evaluate_gsm8k(model),
      "coherence": measure_coherence(model),
      "thinking_token_usage": measure_token_usage(model, "[thinking]"),
  }

  # Success criteria
  assert final_metrics["gsm8k_acc"] > phase2_metrics["gsm8k_acc"]  # Improvement
  assert final_metrics["coherence"] > 0.6
  assert final_metrics["thinking_token_usage"] > 0.7  # 70% of outputs

  # Save
  torch.save(model.state_dict(), "phase3_outputs/reasoning_enhanced.pt")
  ```

---

## Phase 4: BitNet Compression

### Pre-Flight Checks

- [ ] **Load Phase 3 Model**
  ```python
  model = load_model("phase3_outputs/reasoning_enhanced.pt")
  base_size = get_model_size(model)
  size_category = base_size["size_category"]

  print(f"Input: {base_size['params']:,} params ({base_size['size_mb']:.1f} MB)")
  print(f"Category: {size_category}")
  ```

- [ ] **Set Adaptive Compression Targets**
  ```python
  config = get_compression_config(size_category)
  # Returns sparsity_threshold, target_compression, preserve_layers, calibration_samples
  ```

### Compression

- [ ] **Calibration**
  ```python
  calibration_data = load_calibration_data(config["calibration_samples"])
  scale_factors = calibrate_quantization(model, calibration_data)
  ```

- [ ] **Layer-by-Layer Quantization**
  ```python
  accuracy_before = evaluate_accuracy(model)

  for i, layer in enumerate(model.layers):
      # Quantize
      quantize_layer(layer, scale_factors[i], config["sparsity_threshold"])

      # Check accuracy after each layer (🔴 CRITICAL: staged quantization)
      accuracy_after = evaluate_accuracy(model)
      drop = (accuracy_before - accuracy_after) / accuracy_before

      if drop > 0.10:  # >10% drop
          print(f"⚠️ Layer {i} caused {drop:.1%} drop, reverting")
          # Revert this layer
          layer.load_state_dict(layer_backup)
      else:
          print(f"✅ Layer {i}: {drop:.1%} drop")
  ```

- [ ] **Final Compression Check**
  ```python
  compressed_size = get_model_size(model)
  compression_ratio = base_size["size_mb"] / compressed_size["size_mb"]

  print(f"Compressed: {compressed_size['size_mb']:.1f} MB")
  print(f"Compression: {compression_ratio:.2f}x")

  if compression_ratio < config["target_compression"] * 0.75:
      print(f"⚠️ Low compression: {compression_ratio:.2f}x < {config['target_compression']:.2f}x target")
      # Accept it (expected for tiny models)
  ```

- [ ] **Accuracy Check** (🔴 CRITICAL)
  ```python
  accuracy_final = evaluate_accuracy(model)
  accuracy_drop = (accuracy_before - accuracy_final) / accuracy_before

  if accuracy_drop > 0.10:
      print(f"❌ Accuracy drop too high: {accuracy_drop:.1%} > 10%")
      # Revert and use conservative config
      model.load_state_dict(torch.load("phase3_outputs/reasoning_enhanced.pt"))
      config = get_conservative_compression_config(size_category)
      # Re-compress with conservative settings
  ```

- [ ] **Fine-Tuning** (if accuracy drop >5%)
  ```python
  if accuracy_drop > 0.05:
      print(f"Fine-tuning to recover from {accuracy_drop:.1%} drop")
      fine_tune_compressed(model, epochs=2, lr=1e-5)
      accuracy_recovered = evaluate_accuracy(model)
      final_drop = (accuracy_before - accuracy_recovered) / accuracy_before
      print(f"After fine-tuning: {final_drop:.1%} drop")
  ```

- [ ] **Final Validation**
  ```python
  final_metrics = {
      "compression_ratio": compression_ratio,
      "accuracy_drop": final_drop,
      "sparsity": measure_sparsity(model),
      "inference_speedup": measure_speedup(model)
  }

  # Success criteria (size-dependent)
  min_compression = 6.0 if size_category == "tiny" else 8.0
  assert final_metrics["compression_ratio"] >= min_compression
  assert final_metrics["accuracy_drop"] < 0.10
  assert final_metrics["inference_speedup"] >= 2.0

  # Save
  torch.save(model.state_dict(), "phase4_outputs/compressed_1.58bit.pt")
  ```

---

## Post-Phase 4 Validation

- [ ] **End-to-End Test**
  ```python
  # Load compressed model
  model = load_model("phase4_outputs/compressed_1.58bit.pt")

  # Test inference
  test_input = "What is 2+2?"
  output = model.generate(test_input)
  print(f"Input: {test_input}")
  print(f"Output: {output}")

  # Check thinking tokens present
  assert "[thinking]" in output
  assert "[/endthinking]" in output
  ```

- [ ] **Benchmark Suite**
  ```python
  benchmark_results = {
      "gsm8k": evaluate_gsm8k(model),
      "arc": evaluate_arc(model),
      "piqa": evaluate_piqa(model),
      "inference_time": measure_inference_time(model),
      "model_size_mb": get_model_size(model)["size_mb"],
      "vram_usage_mb": measure_vram_usage(model)
  }

  print("Final Benchmark Results:")
  for metric, value in benchmark_results.items():
      print(f"  {metric}: {value}")
  ```

- [ ] **Cleanup Intermediate Files**
  ```python
  # Delete Phase 1-3 models to save space
  os.remove("phase1_outputs/model1_reasoning.pt")
  os.remove("phase1_outputs/model2_memory.pt")
  os.remove("phase1_outputs/model3_speed.pt")
  os.remove("phase2_outputs/champion_evolved.pt")
  os.remove("phase3_outputs/reasoning_enhanced.pt")

  # Keep only:
  # - phase4_outputs/compressed_1.58bit.pt (310 MB)
  # - data/cognate_datasets/ (1.35 GB)
  # - wandb logs (500 MB)
  # Total: ~2.2 GB
  ```

---

## Summary Checklist

### Phase 1
- [x] Pre-download datasets
- [x] Validate diversity
- [x] Monitor loss convergence
- [x] 3 models created

### Phase 2
- [x] Load Phase 1 models
- [x] Monitor diversity during evolution
- [x] Validate merged models
- [x] Champion selected

### Phase 3
- [x] Pre-generate reasoning data
- [x] Baking converges ≥85%
- [x] Anti-theater checks pass
- [x] Reasoning model created

### Phase 4
- [x] Staged quantization
- [x] Accuracy drop <10%
- [x] Compression ≥6x
- [x] Compressed model saved

---

**Version**: 2.0
**Date**: 2025-10-15
**Status**: ✅ Ready for Implementation

**Estimated Timeline**: 40-50 hours on GTX 1660
**Estimated Cost**: $100-225
**Final Storage**: ~2.2 GB (minimal) or ~13 GB (with intermediates)
