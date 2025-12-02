# Phase 1-4: Comprehensive Plan & Integration Specification

**Version**: 2.0
**Date**: 2025-10-15
**Purpose**: Complete specification for Phases 1-4 with model-size-agnostic architecture

---

## Executive Summary

This document provides a **complete, model-size-agnostic** plan for Phases 1-4 of Agent Forge V2. The critical insight is that **we don't know the final model size** until Phase 3 completes, so all downstream phases must be **adaptive**.

### Phase Outputs (Sizes Unknown Until Runtime)

| Phase | Output | Expected Size | Actual Size (Unknown) |
|-------|--------|---------------|----------------------|
| **Phase 1** | 3 models | ~25M params each | **TBD at training time** |
| **Phase 2** | 1 merged model | ~25M params | **TBD (depends on Phase 1)** |
| **Phase 3** | 1 reasoning model | ~30-40M params (adds tokens) | **TBD (depends on Phase 2 + tokens)** |
| **Phase 4** | 1 compressed model | 8x smaller | **TBD (depends on Phase 3)** |

**Key Design Principle**: All phases must query model size at runtime and adapt compression ratios, batch sizes, memory allocation, etc.

---

## Phase 1: Cognate (Create 3 Foundation Models)

### Purpose
Create **3 specialized models** (25M params each) with TRM × Titans-MAG architecture, each trained on different dataset mixes for diversity.

### Architecture (Fixed)

**TRM × Titans-MAG** (Test-time Reasoning Model × Titans with Memory-Augmented Gate):
- **Titans-MAG Backbone**: 8 layers, 512 hidden dim, 8 attention heads
- **TRM Wrapper**: Multi-pass reasoning (up to 3 passes)
- **ACT Head**: Adaptive computation time (decides when to stop thinking)
- **LTM**: Long-term memory (2048-8192 slots, configurable)
- **MAG Gate**: Memory-Augmented Gate (decides when to use memory)

**Parameters**:
- Model 1 (Reasoning): ~25M params, ACT threshold 0.95, LTM 2048
- Model 2 (Memory): ~25M params, ACT threshold 0.90, LTM 8192
- Model 3 (Speed): ~25M params, ACT threshold 0.99, LTM 2048

### Training Data

**16 datasets** (~200,000 samples):
- Math: GSM8K, SVAMP, ASDiv
- Code: Mini-MBPP, CodeXGLUE
- Science: ARC-Easy, ARC-Challenge
- Multi-Hop QA: HotpotQA, DROP, StrategyQA
- Commonsense: PIQA, HellaSwag, BoolQ
- Language: WikiText, FineWeb-Edu, OpenWebText (optional)

**3-Stage Curriculum**:
1. **Foundation** (Epochs 1-3): GSM8K, SVAMP, Mini-MBPP
2. **Reasoning** (Epochs 4-6): ARC-Easy, ARC-Challenge, PIQA, WikiText
3. **Advanced** (Epochs 7-10): HotpotQA, DROP, HellaSwag, FineWeb-Edu

**Dataset Mixing** (for diversity):
```python
# Model 1 (Reasoning-focused)
model1_weights = {
    "math": 0.4,      # Heavy on math
    "qa": 0.3,        # Heavy on multi-hop
    "code": 0.1,
    "commonsense": 0.1,
    "language": 0.1
}

# Model 2 (Memory-focused)
model2_weights = {
    "language": 0.4,  # Heavy on long documents
    "qa": 0.3,        # Heavy on multi-hop
    "math": 0.1,
    "code": 0.1,
    "commonsense": 0.1
}

# Model 3 (Speed-focused)
model3_weights = {
    "commonsense": 0.4,  # Heavy on quick tasks
    "math": 0.3,         # Simple math
    "code": 0.1,
    "qa": 0.1,
    "language": 0.1
}
```

### Optimizer

**Muon × Grokfast** (Phase 1 config):
```python
optimizer = MuonGrokfast(
    model.parameters(),
    muon_lr=1e-3,              # Standard pretraining LR
    grokfast_lambda=0.3,       # Moderate filtering
    qk_clip_threshold=30.0,    # Standard attention clip
    kl_coefficient=0.0         # No KL (pretraining from scratch)
)
```

### Outputs

```python
phase1_output = {
    "models": [
        {"path": "model1_reasoning.pt", "params": 24_876_512, "vram": "1.2 GB"},
        {"path": "model2_memory.pt", "params": 25_134_208, "vram": "1.3 GB"},
        {"path": "model3_speed.pt", "params": 24_912_384, "vram": "1.2 GB"}
    ],
    "metrics": {
        "model1": {"perplexity": 15.2, "gsm8k_acc": 0.12, "arc_acc": 0.28},
        "model2": {"perplexity": 14.8, "gsm8k_acc": 0.11, "arc_acc": 0.27},
        "model3": {"perplexity": 16.1, "gsm8k_acc": 0.13, "arc_acc": 0.29}
    },
    "diversity": {
        "halting_steps": [8.2, 9.5, 6.1],  # ACT diversity
        "memory_usage": [0.35, 0.68, 0.21],  # LTM usage
        "inference_time": [45, 52, 38]  # ms per forward pass
    },
    "training_time": "22.5 hours (7.5 hrs per model)",
    "storage": "~5 GB (3 models × ~1.7 GB each)"
}
```

### Model Size Detection (Runtime)

```python
def get_model_size(model):
    """Detect model size at runtime"""
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 ** 2)  # FP32
    return {
        "params": total_params,
        "size_mb": size_mb,
        "size_category": categorize_size(total_params)
    }

def categorize_size(params):
    """Categorize for adaptive strategies"""
    if params < 50_000_000:  # <50M
        return "tiny"
    elif params < 500_000_000:  # <500M
        return "small"
    elif params < 2_000_000_000:  # <2B
        return "medium"
    else:
        return "large"
```

### Critical Failure Modes & Mitigations (Phase 1)

#### 🔴 CRITICAL: Dataset Download Failures

**Detection**:
```python
def validate_dataset_availability():
    """Run BEFORE Phase 1 starts"""
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

**Mitigation**:
- Pre-download all datasets before starting Phase 1
- Cache locally (1.35 GB)
- Implement retry logic with exponential backoff
- Have fallback mirrors ready

#### 🟡 HIGH: Models Don't Diversify

**Detection**:
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

**Mitigation**:
- Aggressive config differences (ACT: 0.95, 0.85, 0.99)
- Different dataset mixes (60% vs 40% vs 50% main category)
- Different random seeds (42, 43, 44)
- Different epoch counts (10, 12, 8)

#### 🟡 HIGH: Out of Memory During Training

**Detection & Mitigation**:
```python
def calculate_safe_batch_size(model, device_vram_gb):
    """Calculate batch size that fits in VRAM"""
    model_size_mb = get_model_size(model)["size_mb"]

    # Rule of thumb: 4x model size for training
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
        overhead_per_sample = model_size_mb * 0.1
        batch_size = int((available_vram_mb - required_vram_mb) / overhead_per_sample)
        batch_size = min(batch_size, 32)  # Cap at 32
        accumulation_steps = 1

    # Test it
    try:
        test_batch = torch.randn(batch_size, 512, 512).to("cuda")
        with torch.no_grad():
            output = model(test_batch)
        print(f"✅ Batch size {batch_size} fits in VRAM")
    except torch.cuda.OutOfMemoryError:
        print(f"❌ Batch size {batch_size} too large, reducing...")
        batch_size = batch_size // 2
        accumulation_steps *= 2

    return batch_size, accumulation_steps
```

**Additional Mitigations**:
- Enable gradient checkpointing
- Use mixed precision training (FP16)
- Regular checkpoints every 1000 steps

#### 🟡 HIGH: Training Doesn't Converge

**Detection**:
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

**Mitigation**:
- Learning rate warmup (1000 steps)
- Gradient clipping (max_norm=1.0)
- Monitor loss trends in W&B
- Adaptive learning rate (ReduceLROnPlateau)

---

## Phase 2: EvoMerge (Evolve 3 → 1 via Genetic Algorithm)

### Purpose
Merge 3 Phase 1 models into **1 optimized model** through 50 generations of evolution.

### Process (Model-Size-Agnostic)

**Input**: 3 models of **unknown size** (detected at runtime)

**Evolution**:
1. **Generation 0**: Create 8 models using binary merge combinations
2. **Generations 1-50**: Elite preservation + loser merging
3. **Fitness evaluation**: Composite score (perplexity, accuracy, speed, memory)

**6 Merge Techniques** (3 pairs):
- **Pair 1 (Interpolation)**: Linear (0) vs SLERP (1)
- **Pair 2 (Task Arithmetic)**: DARE (0) vs TIES (1)
- **Pair 3 (Selection)**: FrankenMerge (0) vs DFS (1)

**Binary Combinations** (8 total):
```
000: Linear + DARE + FrankenMerge
001: Linear + DARE + DFS
010: Linear + TIES + FrankenMerge
011: Linear + TIES + DFS
100: SLERP + DARE + FrankenMerge
101: SLERP + DARE + DFS
110: SLERP + TIES + FrankenMerge
111: SLERP + TIES + DFS
```

### Fitness Function (Model-Size-Aware)

```python
def evaluate_fitness(model, model_size_mb):
    """Composite fitness with size-adaptive weights"""

    # Standard metrics
    perplexity = evaluate_perplexity(model)
    accuracy = evaluate_accuracy(model)
    inference_time = measure_inference_time(model)
    memory_usage = measure_memory_usage(model)

    # Size-adaptive weights
    if model_size_mb < 100:  # Tiny models (<100MB)
        weights = {"perplexity": 0.5, "accuracy": 0.3, "speed": 0.1, "memory": 0.1}
    elif model_size_mb < 1000:  # Small models (<1GB)
        weights = {"perplexity": 0.4, "accuracy": 0.3, "speed": 0.2, "memory": 0.1}
    else:  # Large models (>1GB)
        weights = {"perplexity": 0.4, "accuracy": 0.3, "speed": 0.15, "memory": 0.15}

    # Composite fitness
    fitness = (
        weights["perplexity"] * (1 / perplexity) +
        weights["accuracy"] * accuracy +
        weights["speed"] * (1 / inference_time) +
        weights["memory"] * (1 / memory_usage)
    )

    return fitness
```

### Memory Management (Size-Agnostic)

```python
def manage_population_memory(population, device_vram_gb):
    """Adaptive memory management based on model size and available VRAM"""

    model_size_mb = get_model_size(population[0])["size_mb"]
    total_models = len(population)  # 8 models
    required_vram_mb = model_size_mb * total_models
    available_vram_mb = device_vram_gb * 1024 * 0.9  # 90% utilization

    if required_vram_mb > available_vram_mb:
        # Offload to CPU
        print(f"⚠️ Models too large for VRAM ({required_vram_mb:.0f}MB > {available_vram_mb:.0f}MB)")
        print(f"   Offloading to CPU for evolution")
        for model in population:
            model.cpu()
        return "cpu"
    else:
        print(f"✅ Models fit in VRAM ({required_vram_mb:.0f}MB < {available_vram_mb:.0f}MB)")
        return "cuda"
```

### Critical Failure Modes & Mitigations (Phase 2)

#### 🟡 HIGH: Population Converges Prematurely

**Detection**:
```python
def compute_diversity(population):
    """Average pairwise distance - run every generation"""
    distances = []
    for i, model_i in enumerate(population):
        for j, model_j in enumerate(population):
            if i < j:
                dist = cosine_distance(get_weights_flat(model_i), get_weights_flat(model_j))
                distances.append(dist)

    return np.mean(distances)

# Monitor during evolution
if generation % 5 == 0:
    diversity = compute_diversity(population)
    wandb.log({"diversity": diversity})

    if diversity < 0.25:
        print(f"⚠️ Low diversity ({diversity:.2f}), triggering intervention")
```

**Mitigation**:
```python
def maintain_diversity(population, generation, diversity):
    """Diversity-based interventions"""

    # Intervention 1: Adaptive mutation rate
    if diversity < 0.3:
        mutation_rate = 0.02  # Double from 0.01
    else:
        mutation_rate = 0.01  # Standard

    # Intervention 2: Inject random model
    if generation % 10 == 0 and diversity < 0.3:
        print(f"Injecting random model (generation {generation})")
        random_combo = random.randint(0, 7)
        population[-1] = apply_merge_combo([model1, model2, model3], random_combo)

    # Intervention 3: Diversity-based selection
    elites = select_diverse_elites(population, fitness_scores, diversity_threshold=0.3)

    return population, mutation_rate
```

#### 🟡 HIGH: Merge Techniques Create Degenerate Models

**Detection**:
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

**Mitigation**:
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

# Numerical stability for SLERP
theta = torch.arccos(torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7))
if theta < 1e-6:  # Models too similar
    print("⚠️ Models nearly identical, using linear instead of SLERP")
    return linear_merge(model1, model2)
```

#### 🟢 MEDIUM: Fitness Evaluation Too Slow

**Mitigation**:
```python
def fast_fitness_evaluation(model, test_set, sample_size=1000):
    """Evaluate on subset for speed"""
    sampled_test = random.sample(test_set, sample_size)
    return evaluate_fitness(model, sampled_test)

# Cache fitness scores
fitness_cache = {}  # model_hash → fitness

def cached_fitness(model):
    model_hash = hash_model_weights(model)
    if model_hash in fitness_cache:
        return fitness_cache[model_hash]  # Already evaluated
    else:
        fitness = fast_fitness_evaluation(model)
        fitness_cache[model_hash] = fitness
        return fitness

# Parallel evaluation
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    fitness_scores = list(executor.map(cached_fitness, population))
```

### Outputs

```python
phase2_output = {
    "model": {
        "path": "champion_evolved.pt",
        "params": 25_012_384,  # Similar to Phase 1 (slight variation from merging)
        "vram": "1.3 GB"
    },
    "metrics": {
        "fitness": 0.185,  # 23.5% better than Phase 1 average (0.15)
        "perplexity": 13.2,  # Improved from 15.4 average
        "gsm8k_acc": 0.15,  # Improved from 0.12 average
        "arc_acc": 0.31  # Improved from 0.28 average
    },
    "evolution": {
        "generations": 50,
        "improvement": 0.235,  # 23.5%
        "best_combo": "100 (SLERP + DARE + FrankenMerge)",
        "combo_usage": {"100": 15, "000": 12, "001": 8, ...}
    },
    "training_time": "90 minutes",
    "storage": "~1.7 GB (1 model)"
}
```

---

## Phase 3: Quiet-STaR (Add Reasoning via Two-Step Process)

### Purpose
Add parallel reasoning capabilities through **prompt baking** (Step 1) then **Quiet-STaR** (Step 2).

### **CRITICAL**: Token Addition Increases Model Size

```python
# Phase 2 model
base_model_params = 25_012_384

# Step 0: Add special tokens
special_tokens = 12  # 2 outer ([thinking], [/endthinking]) + 10 inner strategies
vocab_size_before = 50_257  # GPT-2 vocab
vocab_size_after = 50_257 + 12 = 50_269

# Embedding layer size increase
embedding_params_before = vocab_size_before * hidden_dim = 50_257 * 512 = 25_731_584
embedding_params_after = vocab_size_after * hidden_dim = 50_269 * 512 = 25_737_728
embedding_increase = 25_737_728 - 25_731_584 = 6_144 params

# LM head size increase (same as embedding)
lm_head_increase = 6_144 params

# Total increase
total_increase = embedding_increase + lm_head_increase = 12_288 params

# New model size
phase3_model_params = 25_012_384 + 12_288 = 25_024_672 params  # ~25M still
```

**Key Insight**: Adding 12 tokens adds ~12K params. Negligible for 25M models, but scales with hidden_dim.

### Process (Model-Size-Agnostic)

#### **Step 0: Detect Base Model Size**

```python
def prepare_phase3(phase2_model):
    """Detect model size and configure accordingly"""

    model_info = get_model_size(phase2_model)
    base_params = model_info["params"]
    base_size_mb = model_info["size_mb"]
    hidden_dim = phase2_model.config.hidden_size

    # Calculate token addition impact
    num_new_tokens = 12
    params_per_token = hidden_dim * 2  # Embedding + LM head
    token_params = num_new_tokens * params_per_token

    new_params = base_params + token_params
    new_size_mb = new_params * 4 / (1024 ** 2)  # FP32

    print(f"📊 Phase 3 Model Size Projection:")
    print(f"   Base model: {base_params:,} params ({base_size_mb:.1f} MB)")
    print(f"   + {num_new_tokens} tokens: {token_params:,} params ({token_params * 4 / (1024**2):.2f} MB)")
    print(f"   = New model: {new_params:,} params ({new_size_mb:.1f} MB)")

    return {
        "base_params": base_params,
        "new_params": new_params,
        "token_overhead": token_params / base_params,  # % increase
        "size_mb": new_size_mb
    }
```

#### **Step 1: Prompt Baking** (Supervised Learning)

**Data Generation** (OpenRouter API):
- 5 frontier models × 10 strategies × 500 examples = **25,000 examples**
- Cost: ~$100-200
- Batch size: 100 examples per API call (250 calls total)

**Training**:
```python
def phase3_step1_baking(model, reasoning_data):
    """Bake reasoning patterns (supervised)"""

    # Size-adaptive batch size
    model_size_mb = get_model_size(model)["size_mb"]
    if model_size_mb < 100:
        batch_size = 32
    elif model_size_mb < 500:
        batch_size = 16
    elif model_size_mb < 2000:
        batch_size = 8
    else:
        batch_size = 4

    # Muon × Grokfast (baking config)
    optimizer = MuonGrokfast(
        model.parameters(),
        muon_lr=1e-4,              # Lower for fine-tuning
        grokfast_lambda=0.2,       # Moderate filtering
        qk_clip_threshold=30.0,    # Standard
        kl_coefficient=0.0         # No KL (we WANT to change model)
    )

    # Train for 5 epochs
    for epoch in range(5):
        for batch in DataLoader(reasoning_data, batch_size=batch_size):
            # Supervised learning (standard cross-entropy)
            logits = model(batch["input_ids"])
            loss = F.cross_entropy(logits.view(-1, vocab_size), batch["labels"].view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Validate convergence
    accuracy = validate_reasoning(model, reasoning_data)
    assert accuracy >= 0.85, f"Baking failed: {accuracy:.2%} < 85%"

    return model  # Baked model
```

#### **Step 2: Quiet-STaR** (Reinforcement Learning)

```python
def phase3_step2_quietstar(baked_model):
    """Train parallel thoughts (RL)"""

    # Size-adaptive thought generation
    model_size_mb = get_model_size(baked_model)["size_mb"]
    if model_size_mb < 100:
        num_thoughts = 8  # Tiny models can handle more
        thought_length = 20
    elif model_size_mb < 500:
        num_thoughts = 6
        thought_length = 15
    elif model_size_mb < 2000:
        num_thoughts = 4
        thought_length = 12
    else:
        num_thoughts = 4  # Large models limited by memory
        thought_length = 10

    # Muon × Grokfast (RL config)
    optimizer = MuonGrokfast(
        baked_model.parameters(),
        muon_lr=5e-4,              # HIGHER for RL exploration
        grokfast_lambda=0.1,       # LOWER for RL noise
        qk_clip_threshold=25.0,    # TIGHTER for RL attention spikes
        kl_coefficient=0.1         # NEW: Prevent drift from baked baseline
    )

    # RL training loop
    for batch in dataloader:
        for token_pos in batch:
            # Generate thoughts (size-adaptive count)
            thoughts = generate_thoughts(baked_model, token_pos, num=num_thoughts)

            # Score & mix thoughts
            scores = [score_coherence(t, token_pos, baked_model) for t in thoughts]
            enhanced_hidden = mix_thoughts(token_pos, thoughts, scores)

            # Predict next token
            logits = baked_model.predict(enhanced_hidden)
            correct = (logits.argmax() == labels[token_pos])

            # REINFORCE reward
            reward = 1.0 if correct else 0.0
            loss = -reward * torch.log(torch.tensor(scores).mean())

            # KL regularization (prevent drift)
            base_logits = baked_model.base_forward(token_pos)
            kl_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(base_logits, dim=-1),
                reduction='batchmean'
            )

            total_loss = loss + 0.1 * kl_loss
            total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return baked_model  # Reasoning-enhanced model
```

### Critical Failure Modes & Mitigations (Phase 3)

#### 🔴 CRITICAL: Frontier Model Data Generation Fails

**Detection**:
```python
# Pre-generation validation
def validate_api_access():
    """Run BEFORE Phase 3 data generation"""
    try:
        response = openrouter_client.generate("Test prompt", max_tokens=10)
        print("✅ API access confirmed")
        return True
    except Exception as e:
        print(f"❌ API access failed: {str(e)}")
        return False

# During generation
total_cost = 0.0
cost_limit = 200.0
success_rate = 0.0

for batch_idx, batch in enumerate(batches):
    batch_cost = await generate_batch(batch)
    total_cost += batch_cost

    if total_cost > cost_limit:
        raise RuntimeError(f"Cost limit exceeded: ${total_cost:.2f}")

    print(f"💰 Batch {batch_idx}: ${batch_cost:.2f} (total: ${total_cost:.2f}/${cost_limit})")
```

**Mitigation**:
```python
# Pre-generate data BEFORE Phase 3
python phase3_data_generator.py --output data/reasoning_25k.json

# Retry with exponential backoff
@retry(tries=5, delay=2, backoff=2, max_delay=60)
async def api_call_with_retry(prompt):
    response = await openrouter_client.generate(prompt)
    return response

# Rate limiting
rate_limiter = AsyncRateLimiter(max_rate=50, time_period=60)  # 50 per minute
async with rate_limiter:
    response = await api_call(prompt)

# Quality validation
def validate_generated_example(example):
    assert "reasoning" in example, "Missing reasoning"
    assert "[thinking]" in example["reasoning"], "Missing [thinking] token"
    assert "[/endthinking]" in example["reasoning"], "Missing [/endthinking] token"

    strategy_tokens = ["<step>", "<mece>", "<falsify>", "<expert>", "<orthogonal>", "<doubt>", "<bayesian>", "<multidomain>", "<correct>", "<uncertain>"]
    has_strategy = any(token in example["reasoning"] for token in strategy_tokens)
    assert has_strategy, "Missing strategy tokens"
    return True
```

#### 🟡 HIGH: Prompt Baking Doesn't Converge

**Detection**:
```python
# Monitor validation accuracy after each epoch
for epoch in range(5):
    train(model, reasoning_data)
    val_accuracy = validate_reasoning(model, val_data)

    wandb.log({"baking/val_accuracy": val_accuracy, "epoch": epoch})

    if val_accuracy < 0.85 and epoch == 4:
        print(f"⚠️ Baking failed to converge: {val_accuracy:.2%} < 85%")
```

**Mitigation**:
```python
# Staged convergence thresholds
CONVERGENCE_THRESHOLDS = {
    "chain_of_thought": 0.90,  # Easy
    "mece": 0.85,
    "falsification": 0.80,     # Harder
    "bayesian": 0.75,          # Hardest
    "overall": 0.85
}

# Learning rate finder
from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
optimal_lr = lr_finder.lr_suggestion()

# Curriculum learning
easy_strategies = ["chain_of_thought", "self_doubt"]
hard_strategies = ["bayesian", "falsification"]

# Epochs 1-2: Easy strategies
train(model, filter_by_strategies(data, easy_strategies), epochs=2)

# Epochs 3-5: All strategies
train(model, data, epochs=3)
```

#### 🔴 CRITICAL: Quiet-STaR Generates Theater (Empty Reasoning)

**Detection & Mitigation**:
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

    # Test 3: Thought diversity
    diversity_scores = []
    for sample in test_samples:
        thoughts = generate_thoughts(model, sample, num=4)
        pairwise_dists = [edit_distance(thoughts[i], thoughts[j])
                         for i in range(len(thoughts))
                         for j in range(i+1, len(thoughts))]
        diversity = np.mean(pairwise_dists)
        diversity_scores.append(diversity)

    avg_diversity = np.mean(diversity_scores)
    if avg_diversity < 3.0:
        print(f"🚨 THEATER DETECTED: diversity={avg_diversity:.1f} < 3.0")
        return False

    print(f"✅ Anti-theater passed: divergence={avg_divergence:.2f}, length={avg_thought_length:.1f}, diversity={avg_diversity:.1f}")
    return True

# Run during RL training
if step % 1000 == 0:
    if not anti_theater_check(model, validation_set):
        print("⚠️ Stopping RL training, theater detected")
        # Rollback to previous checkpoint
        model.load_state_dict(torch.load("checkpoint_before_theater.pt"))
        # Adjust hyperparameters
        kl_coefficient *= 2.0  # Increase KL (tie model to baseline)
        reward_threshold *= 1.5  # Make rewards stricter

# Reward shaping
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

### Outputs

```python
phase3_output = {
    "model": {
        "path": "reasoning_enhanced.pt",
        "params": 25_024_672,  # Base (25M) + tokens (12K)
        "vram": "1.4 GB",
        "size_increase": "0.05%"  # Negligible
    },
    "metrics": {
        "gsm8k_acc": 0.18,  # +3% from Phase 2 (0.15)
        "arc_acc": 0.34,    # +3% from Phase 2 (0.31)
        "perplexity": 12.8,  # Slight improvement
        "inference_time": 185  # ms (with 4 thoughts)
    },
    "reasoning": {
        "thinking_token_usage": 0.82,  # 82% of outputs use [thinking]
        "avg_thought_count": 4.2,
        "avg_thought_length": 12.5,
        "coherence_score": 0.73
    },
    "anti_theater": {
        "divergence": 0.42,  # Thoughts differ from direct (>0.3 ✅)
        "ablation_drop": 0.03,  # Removing thoughts drops accuracy 3% ✅
        "correlation": 0.61  # Coherence correlates with utility (>0.5 ✅)
    },
    "training_time": "14 hours (Step 1: 6hrs, Step 2: 8hrs)",
    "data_generation_cost": "$127.50",
    "storage": "~1.8 GB"
}
```

---

## Phase 4: BitNet (Compress to 1.58-bit)

### Purpose
Compress Phase 3 model 8x using BitNet 1.58-bit quantization.

### **CRITICAL**: Model-Size-Agnostic Compression

```python
def phase4_compress(phase3_model, target_compression=8.0):
    """Compress model with adaptive strategies"""

    # Detect model size
    model_info = get_model_size(phase3_model)
    base_params = model_info["params"]
    base_size_mb = model_info["size_mb"]
    size_category = model_info["size_category"]

    print(f"📦 Phase 4 Compression Plan:")
    print(f"   Input: {base_params:,} params ({base_size_mb:.1f} MB)")
    print(f"   Target: {target_compression}x compression")
    print(f"   Category: {size_category}")

    # Size-adaptive compression config
    if size_category == "tiny":  # <50M params
        config = {
            "sparsity_threshold": 0.05,  # Conservative (fewer zeros)
            "target_compression": 6.0,    # Lower target (embeddings are larger %)
            "preserve_layers": ["embeddings", "lm_head", "layer_norm"],
            "calibration_samples": 500
        }
    elif size_category == "small":  # <500M params
        config = {
            "sparsity_threshold": 0.1,    # Moderate
            "target_compression": 8.0,    # Standard target
            "preserve_layers": ["embeddings", "lm_head"],
            "calibration_samples": 1000
        }
    elif size_category == "medium":  # <2B params
        config = {
            "sparsity_threshold": 0.15,   # Aggressive
            "target_compression": 10.0,   # Higher target
            "preserve_layers": ["embeddings", "lm_head"],
            "calibration_samples": 2000
        }
    else:  # Large (>2B params)
        config = {
            "sparsity_threshold": 0.2,    # Very aggressive
            "target_compression": 12.0,   # Highest target
            "preserve_layers": ["embeddings"],  # Only preserve embeddings
            "calibration_samples": 5000
        }

    # Run calibration
    calibration_data = load_calibration_data(config["calibration_samples"])
    scale_factors = calibrate_quantization(phase3_model, calibration_data)

    # Compress layer by layer
    compressed_params = 0
    preserved_params = 0

    for name, layer in phase3_model.named_modules():
        if any(preserve_name in name for preserve_name in config["preserve_layers"]):
            # Preserve in FP16
            layer_params = sum(p.numel() for p in layer.parameters())
            preserved_params += layer_params
            layer.half()  # FP32 → FP16 (2x compression)
        elif isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            # Quantize to 1.58-bit
            layer_params = sum(p.numel() for p in layer.parameters())
            compressed_params += layer_params

            for param in layer.parameters():
                # Calculate scale factor
                scale = scale_factors[name]

                # Normalize
                normalized = param.data / scale

                # Apply sparsity threshold
                mask = torch.abs(normalized) < config["sparsity_threshold"]

                # Quantize to {-1, 0, +1}
                quantized = torch.sign(normalized)
                quantized[mask] = 0

                # Store as int8
                param.data = quantized.to(torch.int8)
                param.scale = scale  # Store scale for dequantization

    # Calculate actual compression
    total_params = base_params
    compressed_size_mb = (
        compressed_params * 1 / 8 +  # 1.58-bit ≈ 1 bit (int8 storage)
        preserved_params * 2          # FP16
    ) / (1024 ** 2)

    actual_compression = base_size_mb / compressed_size_mb

    print(f"✅ Compression Complete:")
    print(f"   Compressed: {compressed_params:,} params ({compressed_params/total_params:.1%})")
    print(f"   Preserved: {preserved_params:,} params ({preserved_params/total_params:.1%})")
    print(f"   Final size: {compressed_size_mb:.1f} MB")
    print(f"   Actual compression: {actual_compression:.2f}x")

    # Validate quality
    accuracy_before = evaluate_accuracy(phase3_model)
    accuracy_after = evaluate_accuracy(phase3_model)  # Compressed
    accuracy_drop = (accuracy_before - accuracy_after) / accuracy_before

    print(f"   Accuracy drop: {accuracy_drop:.1%}")

    # Fine-tune if needed
    if accuracy_drop > 0.05:  # >5% drop
        print(f"⚠️ Accuracy drop too high ({accuracy_drop:.1%}), fine-tuning...")
        phase3_model = fine_tune_compressed(phase3_model, epochs=2)
        accuracy_final = evaluate_accuracy(phase3_model)
        accuracy_drop_final = (accuracy_before - accuracy_final) / accuracy_before
        print(f"   After fine-tuning: {accuracy_drop_final:.1%}")

    return {
        "model": phase3_model,
        "compression_ratio": actual_compression,
        "accuracy_drop": accuracy_drop,
        "sparsity": (compressed_params - torch.count_nonzero(quantized)) / compressed_params
    }
```

### Sparsity Analysis (Size-Dependent)

```python
def analyze_sparsity(model, size_category):
    """Expected sparsity by model size"""

    # Tiny models (<50M): Embeddings are ~15-20% of total params
    # Large models (>2B): Embeddings are ~2-5% of total params

    if size_category == "tiny":
        expected_sparsity = 0.25  # 25% (less compression due to embeddings)
        quantized_percent = 0.80  # 80% of params quantized
        expected_compression = 6.0  # Lower overall
    elif size_category == "small":
        expected_sparsity = 0.35  # 35%
        quantized_percent = 0.85  # 85% of params quantized
        expected_compression = 8.0
    elif size_category == "medium":
        expected_sparsity = 0.40  # 40%
        quantized_percent = 0.90  # 90% of params quantized
        expected_compression = 10.0
    else:  # large
        expected_sparsity = 0.45  # 45%
        quantized_percent = 0.95  # 95% of params quantized
        expected_compression = 12.0

    return {
        "expected_sparsity": expected_sparsity,
        "quantized_percent": quantized_percent,
        "expected_compression": expected_compression
    }
```

### Critical Failure Modes & Mitigations (Phase 4)

#### 🟢 MEDIUM: Compression Ratio Lower Than Expected

**Detection**:
```python
# After compression
target_compression = adjust_compression_expectations(model_size_category)
actual_compression = base_size_mb / compressed_size_mb

if actual_compression < target_compression * 0.8:  # 80% of target
    print(f"⚠️ Low compression: {actual_compression:.2f}x < {target_compression * 0.8:.2f}x")
else:
    print(f"✅ Compression achieved: {actual_compression:.2f}x (target: {target_compression}x)")
```

**Mitigation**:
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

# Increase sparsity threshold if needed
if compression_ratio < target and sparsity < 0.30:
    print(f"⚠️ Low sparsity ({sparsity:.1%}), increasing threshold")
    sparsity_threshold *= 1.5  # 0.1 → 0.15
    # Re-quantize

# Report actual compression (don't fail)
print(f"Compression: {actual_compression:.2f}x (target: {target_compression}x)")
wandb.log({"compression_ratio": actual_compression, "target": target_compression})
```

#### 🔴 CRITICAL: Accuracy Drop Exceeds 10%

**Detection**:
```python
# Pre-quantization baseline
baseline_accuracy = evaluate_accuracy(model_before_quantization)
print(f"Baseline accuracy: {baseline_accuracy:.2%}")

# Post-quantization validation
accuracy_after = evaluate_accuracy(model_after_quantization)
accuracy_drop = (baseline_accuracy - accuracy_after) / baseline_accuracy

print(f"Accuracy drop: {accuracy_drop:.1%}")

if accuracy_drop > 0.10:
    print(f"❌ CRITICAL: Accuracy drop {accuracy_drop:.1%} > 10%")
elif accuracy_drop > 0.05:
    print(f"⚠️ Warning: Accuracy drop {accuracy_drop:.1%} > 5%")
else:
    print(f"✅ Accuracy drop acceptable: {accuracy_drop:.1%} < 5%")
```

**Mitigation**:
```python
# Staged quantization (layer-by-layer)
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

# Conservative fallback
if accuracy_drop > 0.10:
    print(f"❌ Accuracy drop too high ({accuracy_drop:.2%}), using conservative config")

    # Revert to full precision
    model.load_state_dict(torch.load("model_before_quantization.pt"))

    # Re-quantize with conservative settings
    config = {
        "sparsity_threshold": 0.05,  # Very low
        "preserve_layers": ["embeddings", "lm_head", "layer_norm", "output_layers[-1]"],
        "target_compression": 4.0  # Lower target
    }

    model = compress(model, config)
    accuracy_after = evaluate_accuracy(model)
    print(f"Conservative compression: {accuracy_after:.2%} (drop: {baseline_accuracy - accuracy_after:.2%})")

# Extended fine-tuning
if accuracy_drop > 0.05:
    print(f"Accuracy drop {accuracy_drop:.2%} > 5%, fine-tuning for longer")
    fine_tune(model, epochs=5, lr=1e-5)  # More epochs, lower LR
```

### Outputs

```python
phase4_output = {
    "model": {
        "path": "compressed_1.58bit.pt",
        "params": 25_024_672,  # Same number, but 1.58-bit
        "size_mb": 310,  # 8x smaller (1800 MB → 310 MB for 25M model)
        "vram": "350 MB",  # Fits easily
        "compression": 7.85  # Actual (close to 8x target)
    },
    "metrics": {
        "gsm8k_acc": 0.17,  # -1% from Phase 3 (0.18)
        "arc_acc": 0.32,    # -2% from Phase 3 (0.34)
        "accuracy_drop": 0.04,  # 4% drop (acceptable <10%)
        "inference_speedup": 2.6,  # 2.6x faster
        "inference_time": 71  # ms (down from 185ms)
    },
    "compression_details": {
        "quantized_params": 21_270_971,  # 85% of total
        "preserved_params": 3_753_701,   # 15% of total (embeddings, lm_head)
        "sparsity": 0.36,  # 36% of weights = 0
        "zero_count": 7_657_549
    },
    "training_time": "4 hours (calibration 1hr, compression 1hr, fine-tuning 2hrs)",
    "storage": "~310 MB"
}
```

---

## Cross-Phase Integration

### Data Flow

```
Phase 1 (3 models, ~75M params total)
    ↓ (models saved as .pt files)
Phase 2 (1 model, ~25M params)
    ↓ (model + tokenizer saved)
Phase 3 (1 model, ~25M params + 12K for tokens)
    ↓ (reasoning-enhanced model saved)
Phase 4 (1 model, same params but 8x smaller storage)
    ↓ (compressed model saved as .pt with int8 weights)
Phase 5 (ready for main training loop)
```

### Storage Requirements (Cumulative)

```python
# Worst case (keep all intermediate models)
storage_breakdown = {
    "phase1_models": 5.1,      # 3 models × 1.7 GB
    "phase1_datasets": 1.35,   # 16 datasets (without OpenWebText)
    "phase2_champion": 1.7,    # 1 model
    "phase3_reasoning": 1.8,   # 1 model (slightly larger)
    "phase3_data": 0.5,        # Generated reasoning examples
    "phase4_compressed": 0.31, # 1 compressed model
    "wandb_logs": 0.5,         # Experiment logs
    "checkpoints": 2.0         # Various checkpoints
}

total_storage_gb = sum(storage_breakdown.values())  # ~13 GB

# Minimal (delete intermediates)
minimal_storage = {
    "phase1_datasets": 1.35,
    "phase4_final": 0.31,
    "wandb_logs": 0.5
}

minimal_storage_gb = sum(minimal_storage.values())  # ~2.2 GB
```

### VRAM Requirements (Concurrent)

```python
def calculate_vram_requirements(phase, model_size_mb):
    """Calculate VRAM needed for each phase"""

    if phase == "phase1":
        # 1 model training + optimizer state + gradients + activations
        vram_per_model = model_size_mb * 4  # Model + optimizer + gradients
        return vram_per_model  # ~5.5 GB for 25M model

    elif phase == "phase2":
        # 8 models in population (can offload to CPU if needed)
        vram_population = model_size_mb * 8 * 1.2  # Small overhead
        return vram_population  # ~16 GB for 25M models (or CPU fallback)

    elif phase == "phase3_step1":
        # 1 model + optimizer + gradients
        return model_size_mb * 4  # ~5.5 GB

    elif phase == "phase3_step2":
        # 1 model + 4-8 thoughts (parallel generation)
        thoughts_overhead = model_size_mb * 0.3  # Thoughts are activations
        return (model_size_mb * 4) + thoughts_overhead  # ~6.5 GB

    elif phase == "phase4":
        # 1 FP32 model + 1 int8 model (during compression)
        return model_size_mb * 2  # ~3.5 GB

    else:
        return model_size_mb * 4  # Default
```

### Weights & Biases Integration

```python
# Phase 1: Track 3 models
wandb.init(project="agent-forge-v2", name="phase1-cognate")
for i, model in enumerate([model1, model2, model3]):
    wandb.log({
        f"model{i+1}/perplexity": perplexity,
        f"model{i+1}/accuracy": accuracy,
        f"model{i+1}/training_time": time_hours
    })

# Phase 2: Track evolution
wandb.init(project="agent-forge-v2", name="phase2-evomerge")
for generation in range(50):
    wandb.log({
        "generation": generation,
        "best_fitness": fitness_scores[0],
        "diversity": compute_diversity(population),
        "improvement": (current_fitness - initial_fitness) / initial_fitness
    })

# Phase 3: Track two-step process
wandb.init(project="agent-forge-v2", name="phase3-quietstar")
# Step 1: Baking
wandb.log({
    "step": "baking",
    "epoch": epoch,
    "loss": loss,
    "accuracy": accuracy
})
# Step 2: RL
wandb.log({
    "step": "rl",
    "episode": episode,
    "reward": reward,
    "coherence": coherence_score
})

# Phase 4: Track compression
wandb.init(project="agent-forge-v2", name="phase4-bitnet")
wandb.log({
    "compression_ratio": actual_compression,
    "accuracy_before": accuracy_before,
    "accuracy_after": accuracy_after,
    "sparsity": sparsity_percent
})
```

---

## Success Criteria (Phases 1-4)

### Phase 1
- ✅ 3 models created (~25M params each)
- ✅ Diverse behaviors (halting steps, memory usage, speed)
- ✅ GSM8K accuracy >10% (baseline 3% random, 5-shot 12%)
- ✅ Training time <30 hours total (3 models)
- ✅ Storage <10 GB (models + datasets)

### Phase 2
- ✅ Fitness improvement ≥20% (target 23.5%)
- ✅ Evolution time <2 hours
- ✅ Diversity maintained (>0.3 throughout)
- ✅ All 8 merge combos used at least once

### Phase 3
- ✅ Thinking tokens added (12 total)
- ✅ Baking convergence ≥85% accuracy
- ✅ Reasoning accuracy +3-5% over Phase 2
- ✅ Anti-theater tests pass (divergence >0.3, ablation drop >2%, correlation >0.5)
- ✅ Inference time <200ms with thoughts

### Phase 4
- ✅ Compression ratio ≥6x (target 8x for small models)
- ✅ Accuracy drop <10% (ideally <5%)
- ✅ Inference speedup ≥2x
- ✅ Sparsity >30%

---

## Next Steps

After Phase 4, the compressed model proceeds to:
- **Phase 5**: Forge Training (main training loop with Grokfast)
- **Phase 6**: Tool & Persona Baking
- **Phase 7**: Generic Edge Deployment
- **Phase 8**: Final Compression (SeedLM + VPTQ + Hyper)

---

## UI Integration

**Complete UI specification available at**: [PHASE1-4_UI_SPECIFICATION_V2.md](PHASE1-4_UI_SPECIFICATION_V2.md)

The UI provides:
- **Adaptive controls** that adjust to runtime model sizes
- **Real-time monitoring** with failure detection visualization
- **Interactive intervention** for all 12 critical failure modes
- **Progressive disclosure** (simple by default, advanced on demand)
- **Mobile responsive** design with accessibility compliance

---

**Version**: 2.0
**Date**: 2025-10-16
**Status**: ✅ Ready for Implementation (Plan + UI Spec Complete)
