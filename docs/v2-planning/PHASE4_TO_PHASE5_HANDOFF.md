# Phase 4 → Phase 5 Handoff: BitNet Quantization to Curriculum Learning

**Purpose**: Solve the compatibility issue between Phase 4's 1.58-bit quantization and Phase 5's gradient-based curriculum learning

**Status**: Solution specified, ready for implementation

---

## Problem Statement

### Phase 4 Output
- **Format**: 1.58-bit quantized model
- **Weights**: Discrete values ∈ {-1, 0, +1}
- **Storage**: int8 with per-channel scaling factors
- **Size**: ~12MB (8.2× compression from FP32)
- **Benefit**: Efficient inference, low memory footprint

### Phase 5 Requirement (Stage 1: Edge-of-Chaos Assessment)
- **Process**: Gradient-based difficulty calibration
- **Needs**: Continuous gradients for variance calculation
- **Purpose**: Find optimal curriculum difficulty (75% accuracy threshold)
- **Formula**: `difficulty_score = compute_gradient_variance(gradients)`

### The Conflict

**Quantized weights are discrete → No continuous gradients → Assessment fails**

```python
# Phase 4 output
quantized_weights = {-1, 0, +1}  # Discrete values

# Phase 5 needs this:
loss = model(sample)
gradients = torch.autograd.grad(loss, model.parameters())  # ❌ BREAKS
difficulty = compute_gradient_variance(gradients)
```

**Root cause**: Quantized weights break the computational graph needed for backpropagation.

---

## Technical Analysis

### Why This Matters

Phase 5's 7-stage curriculum pipeline requires:
1. **Stage 1 (Assessment)**: Gradient-based difficulty calibration
2. **Stage 3 (Training Loop)**: Backprop for recursive thinking + tool use
3. **Stage 4 (Prompt Baking)**: LoRA fine-tuning (needs gradients)
4. **Stage 5 (Self-Modeling)**: Temperature prediction training
5. **Stage 6 (Dream Consolidation)**: Reconstruction loss minimization

**All 5 stages need gradients** → Quantization incompatible with entire Phase 5 pipeline.

### Alternative Solutions Considered

#### ❌ Solution 1: Inference-Based Assessment (No Gradients)
```python
def find_edge_of_chaos_inference_only(quantized_model, dataset):
    # Use accuracy as difficulty proxy
    for level in range(1, 11):
        accuracy = quantized_model.evaluate(dataset[level])
    optimal_level = find_closest(accuracies, target=0.75)
```

**Pros**: Preserves Phase 4 compression
**Cons**:
- Less precise than gradient-based assessment
- Doesn't work for Stages 3-6 (training requires backprop)
- **VERDICT**: Not viable

#### ❌ Solution 2: Partial Quantization (Keep Embeddings FP32)
**Cons**: Still breaks gradient flow through quantized layers
**VERDICT**: Not viable

---

## ✅ Recommended Solution: Partial Dequantization (FP16)

### Strategy

**Dequantize Phase 4 model to FP16 for Phase 5-7 training, then optionally re-quantize before Phase 8.**

### Rationale

1. **Phase 5-7 need gradients**: Cannot work with quantized weights
2. **Phase 8 compresses anyway**: Final compression target is 280× (100MB → 0.4MB)
   - Starting from FP16 (50MB) still achieves **125× compression** to 0.4MB
   - Phase 4's quantization is **temporary optimization**, not final state
3. **Memory is acceptable**: 50MB FP16 fits in 2-3GB VRAM during training

### Implementation

```python
# phases/phase4/bitnet_phase.py

async def finalize(self):
    """Phase 4 completion with dual output"""

    # 1. Save quantized model (for reference/inference)
    quantized_path = f"{self.storage_path}/bitnet_quantized_model.pt"
    torch.save(self.quantized_model, quantized_path)

    # 2. Dequantize to FP16 for Phase 5-7 training
    dequantized_model = self.dequantize_to_fp16(self.quantized_model)
    dequantized_path = f"{self.storage_path}/bitnet_dequantized_fp16.pt"
    torch.save(dequantized_model, dequantized_path)

    # 3. Save quantization metadata for optional re-quantization
    metadata = {
        "scales": self.quantization_scales,
        "zero_points": self.zero_points,
        "quantization_scheme": "1.58-bit",
        "sparsity_threshold": self.sparsity_threshold,
    }
    metadata_path = f"{self.storage_path}/quantization_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # 4. Validate dequantization quality
    dequant_accuracy = self.validate_dequantization(
        self.original_model,
        dequantized_model
    )

    assert dequant_accuracy >= 0.995, f"Dequantization accuracy too low: {dequant_accuracy}"

    return {
        "quantized_model": quantized_path,         # 12MB, for reference
        "dequantized_model": dequantized_path,     # 50MB, for Phase 5-7
        "quantization_metadata": metadata_path,
        "compression_ratio": 8.2,
        "dequantization_accuracy": dequant_accuracy,
        "format": "FP16 dequantized for training",
        "note": "Use dequantized_model for Phase 5-7, quantized_model for inference testing"
    }

def dequantize_to_fp16(self, quantized_model):
    """Convert 1.58-bit quantized model back to FP16"""
    model_fp16 = copy.deepcopy(quantized_model)

    for name, param in model_fp16.named_parameters():
        if hasattr(param, 'quantized'):
            # Retrieve original scale factors
            scale = self.quantization_scales[name]
            zero_point = self.zero_points.get(name, 0)

            # Dequantize: float_value = (int8_value - zero_point) * scale
            param.data = ((param.data - zero_point) * scale).half()  # Convert to FP16

    return model_fp16

def validate_dequantization(self, original_model, dequantized_model):
    """Test dequantization accuracy on validation set"""
    original_model.eval()
    dequantized_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in self.validation_loader:
            inputs, labels = batch

            # Compare predictions
            orig_out = original_model(inputs).argmax(dim=-1)
            dequant_out = dequantized_model(inputs).argmax(dim=-1)

            correct += (orig_out == dequant_out).sum().item()
            total += labels.size(0)

    return correct / total
```

---

## Phase 5 Integration

```python
# phases/phase5/curriculum_phase.py

async def initialize(self):
    """Phase 5 initialization with FP16 model from Phase 4"""

    # Load Phase 4 handoff
    phase4_output = self.load_phase_handoff("phase4")

    # Use dequantized FP16 model (NOT quantized)
    model_path = phase4_output["dequantized_model"]
    self.model = torch.load(model_path).to(self.device)

    # Verify gradients work
    self.test_gradient_flow(self.model)

    # Stage 1: Edge-of-chaos assessment (gradient-based)
    optimal_level = self.find_edge_of_chaos(
        self.model,
        self.assessment_dataset
    )

    print(f"Edge-of-chaos assessment complete: optimal level = {optimal_level}")
    print(f"Model format: FP16, size: {self.get_model_size_mb(self.model)}MB")

def test_gradient_flow(self, model):
    """Verify model supports gradient computation"""
    model.train()

    # Test forward + backward pass
    dummy_input = torch.randn(1, 512).to(self.device)
    dummy_labels = torch.randint(0, 100, (1,)).to(self.device)

    output = model(dummy_input)
    loss = F.cross_entropy(output, dummy_labels)

    try:
        loss.backward()
        print("✅ Gradient flow test PASSED")
    except Exception as e:
        raise RuntimeError(f"❌ Gradient flow test FAILED: {e}")

def find_edge_of_chaos(self, model, dataset):
    """Gradient-based difficulty assessment"""
    model.train()
    gradient_variances = []

    for level in range(1, 11):
        level_data = dataset.filter_by_difficulty(level)

        for batch in level_data:
            inputs, labels = batch

            # Forward pass
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)

            # Compute gradients
            loss.backward()

            # Calculate gradient variance (proxy for difficulty)
            grad_var = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_var += param.grad.var().item()

            gradient_variances.append((level, grad_var))
            model.zero_grad()

    # Find level with gradient variance closest to target
    # (High variance = too hard, low variance = too easy)
    optimal_level = self.find_optimal_variance_level(gradient_variances)

    return optimal_level
```

---

## Optional: Re-Quantization After Phase 7

```python
# phases/phase7/adas_phase.py

async def finalize(self):
    """Phase 7 completion with optional re-quantization"""

    # Phase 7 model is currently FP16

    # Option 1: Keep FP16 for Phase 8 compression (RECOMMENDED)
    # Phase 8 will compress FP16 → 0.4MB (280× from FP32 baseline)
    # This is the standard path

    if self.config.get("requantize_before_phase8", False):
        # Option 2: Re-quantize to BitNet before Phase 8 (OPTIONAL)
        # Only if user wants to compare compression starting points

        metadata = self.load_quantization_metadata("phase4/quantization_metadata.json")
        self.model = self.requantize_to_bitnet(self.model, metadata)

        model_format = "BitNet 1.58-bit"
        model_size_mb = 12
    else:
        model_format = "FP16"
        model_size_mb = 50

    return {
        "model": self.model,
        "format": model_format,
        "size_mb": model_size_mb,
        "note": "Phase 8 will compress to 0.4MB regardless of starting format"
    }
```

---

## Storage Impact Analysis

| Phase | Model Format | Size (25M params) | VRAM (Training) | Notes |
|-------|--------------|-------------------|-----------------|-------|
| **Phase 4 Output (Quantized)** | 1.58-bit int8 | **12MB** | N/A | Inference-only |
| **Phase 4 Output (Dequantized)** | FP16 | **50MB** | 2-3GB | For Phase 5-7 |
| **Phase 5-7 Training** | FP16 | 50MB | 4-6GB | Working format |
| **Phase 8 Input** | FP16 | 50MB | N/A | Standard path |
| **Phase 8 Output (Final)** | SeedLM+VPTQ+Hyper | **0.4MB** | <1GB inference | 125× compression |

### Key Insight

**Phase 8's 280× compression target is measured from FP32 baseline (~100MB), not Phase 4 quantized size.**

- Starting Phase 8 with FP16 (50MB) → 0.4MB = **125× compression**
- Starting Phase 8 with FP32 (100MB) → 0.4MB = **280× compression**

Both achieve the **same 0.4MB final size**, meeting the project goal.

**Phase 4 quantization is a temporary optimization** for inference testing, not the final compression strategy.

---

## W&B Metrics

```python
# Track handoff quality
wandb.log({
    # Phase 4 → 5 handoff
    "phase4_to_5/quantized_model_size_mb": 12,
    "phase4_to_5/dequantized_model_size_mb": 50,
    "phase4_to_5/dequantization_accuracy_retention": 0.998,  # Should be >99.5%
    "phase4_to_5/gradient_flow_test_passed": True,

    # Phase 5 Stage 1 (Assessment)
    "phase5/edge_of_chaos_assessment_completed": True,
    "phase5/edge_of_chaos_optimal_level": optimal_level,
    "phase5/edge_of_chaos_target_accuracy": 0.75,
    "phase5/gradient_variance_optimal": optimal_variance,

    # Model format tracking
    "phase5/model_format": "FP16",
    "phase5/model_size_mb": 50,
    "phase5/vram_usage_gb": torch.cuda.memory_allocated() / 1024**3,
})
```

---

## Success Criteria

### Phase 4 → 5 Handoff
- ✅ Phase 5 receives FP16 model with working gradients
- ✅ Dequantization accuracy retention ≥99.5%
- ✅ Gradient flow test passes (forward + backward)
- ✅ Model size ~50MB (fits in training memory)

### Phase 5 Stage 1 (Assessment)
- ✅ Edge-of-chaos assessment finds 70-80% accuracy zone
- ✅ Gradient variance calculation succeeds
- ✅ Optimal curriculum level determined (1-10)

### Phases 5-7 Training
- ✅ All 7 curriculum stages train successfully with backprop
- ✅ Dream consolidation works (autoencoder reconstruction)
- ✅ Self-modeling trains (temperature prediction)
- ✅ Phase 6 A/B cycles complete (tool + persona optimization)
- ✅ Phase 7 SVF training succeeds (REINFORCE + KL)

### Phase 8 Compression
- ✅ Phase 8 still achieves ≥100× compression (0.4MB target)
- ✅ Quality retention ≥84% (cumulative after 3 compression stages)
- ✅ Benchmark testing validates model capabilities

---

## Implementation Checklist

### Phase 4 Updates
- [ ] Implement `dequantize_to_fp16()` method
- [ ] Update `finalize()` to output both quantized and dequantized models
- [ ] Save quantization metadata (scales, zero_points)
- [ ] Add dequantization validation test (≥99.5% accuracy)
- [ ] Update W&B logging with dual outputs

### Phase 5 Updates
- [ ] Update `initialize()` to load dequantized model
- [ ] Add `test_gradient_flow()` method
- [ ] Verify Stage 1 assessment works with FP16 model
- [ ] Test all 7 stages with gradient-based training
- [ ] Validate dream consolidation with FP16

### Documentation Updates
- [ ] Update Phase 4 README with dual output explanation
- [ ] Update Phase 5 README with FP16 model requirement
- [ ] Add handoff validation guide
- [ ] Update storage estimates in project documentation

### Testing
- [ ] Unit test: Quantize → Dequantize → Verify accuracy
- [ ] Integration test: Phase 4 finalize → Phase 5 initialize
- [ ] End-to-end test: Phase 4 → Phase 5 Stage 1 assessment
- [ ] Performance test: FP16 training VRAM usage

---

## FAQ

### Q: Why not keep quantized weights for Phase 5?
**A**: Phase 5-7 require gradient computation for:
- Edge-of-chaos assessment (gradient variance)
- Curriculum training (backpropagation)
- Prompt baking (LoRA fine-tuning)
- Self-modeling (temperature prediction training)
- Dream consolidation (reconstruction loss)

Quantized weights break gradient flow → Cannot train.

### Q: Doesn't this waste Phase 4's compression work?
**A**: No. Phase 4 validates the quantization approach and provides:
1. Inference-optimized model for testing (12MB)
2. Quantization metadata for optional re-quantization
3. Proof that 8.2× compression is achievable

Phase 8 performs **final compression** (280×), making Phase 4's temporary compression unnecessary for the final model.

### Q: Can we skip Phase 4 entirely then?
**A**: No. Phase 4 serves important purposes:
1. Tests quantization compatibility early (fail-fast)
2. Provides inference baseline (12MB model for speed testing)
3. Validates STE (Straight-Through Estimator) implementation
4. Documents compression metadata for Phase 8 reference

Phase 4 is **validation and preparation**, not the final compression step.

### Q: What about memory usage during Phase 5-7?
**A**: FP16 (50MB model) requires 4-6GB VRAM during training, which fits the project's hardware requirements (GTX 1660 with 6GB VRAM).

### Q: Why FP16 instead of FP32?
**A**:
- FP16 is 2× smaller than FP32 (50MB vs 100MB)
- Modern GPUs have hardware FP16 support (faster training)
- Sufficient precision for 25M parameter models
- Fits in 6GB VRAM for training

### Q: Should we re-quantize before Phase 8?
**A**: **No** (not recommended). Phase 8 compresses FP16 → 0.4MB (125×), which meets the project goal. Re-quantizing adds complexity without benefit.

---

## Related Documents

- [phases/phase4/PHASE4_COMPLETE_GUIDE.md](../../phases/phase4/PHASE4_COMPLETE_GUIDE.md) - BitNet quantization
- [phases/phase5/PHASE5_LOGICAL_UNDERSTANDING_V2.md](../../phases/phase5/PHASE5_LOGICAL_UNDERSTANDING_V2.md) - Curriculum system
- [phases/phase5/PHASE5_V2_IMPLEMENTATION_SUMMARY.md](../../phases/phase5/PHASE5_V2_IMPLEMENTATION_SUMMARY.md) - Stage 1 assessment
- [phases/phase8/PHASE8_COMPLETE_GUIDE.md](../../phases/phase8/PHASE8_COMPLETE_GUIDE.md) - Final compression
- [docs/v2-planning/PHASE5-8_V1_VS_V2_RECONCILIATION.md](PHASE5-8_V1_VS_V2_RECONCILIATION.md) - V1 vs V2 comparison
