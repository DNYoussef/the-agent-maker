# Phase 7: BitNet 1.58-bit Integration Analysis

**Critical Issue Identified**: Phase 7 SVF training operates on BitNet quantized model
**Date**: 2025-10-16
**Status**: ⚠️ Requires Design Decision

---

## The Problem

Phase 7 receives a **BitNet 1.58-bit quantized model** from Phase 4, where all weights are ternary {-1, 0, +1}.

**SVF formula**: `W' = U(Σ ⊗ diag(z))V^T`

**Questions**:
1. Is W quantized or full precision during SVF?
2. Are z-vectors constrained to ternary or float?
3. Does adaptation preserve 1.58-bit compression?
4. What's the memory footprint during training/inference?

---

## Research: Transformer² + BitNet Compatibility

### Transformer² Paper Assumptions

The Transformer² paper (2501.06252v3) assumes **full-precision weights**:
- SVF operates on singular value decomposition of W
- z-vectors are continuous (not quantized)
- No mention of compatibility with quantized models

### BitNet Paper Constraints

BitNet (1.58-bit LLMs) requires:
- Weights stored as ternary {-1, 0, +1}
- Scaling factors α per layer (full precision)
- Forward pass: quantized, Backward pass: full precision (STE)

---

## Three Possible Approaches

### Option 1: Dequantize for Phase 7 (Simplest)

**Approach**:
- Phase 4 → Phase 5-6: Model stays 1.58-bit (8.2x compression)
- **Phase 7: Temporarily dequantize** to full precision for SVF training
- Phase 8: Re-quantize + further compress

**Memory during Phase 7**:
- Base model: 25M params × 32 bits = 100MB (dequantized)
- Expert z-vectors: 5 experts × 4K params × 32 bits = 80KB
- **Total**: ~100MB (vs 12MB if stayed quantized)

**Pros**:
- ✅ SVF training works as designed (full precision)
- ✅ No research risk (proven technique)
- ✅ Simple implementation
- ✅ Experts are full precision (high quality)

**Cons**:
- ❌ Temporarily loses 8.2x compression
- ❌ Higher memory during Phase 7 (100MB vs 12MB)
- ❌ Needs re-quantization in Phase 8

**Workflow**:
```python
# Phase 6 → Phase 7 handoff
bitnet_model = load("phase6_bitnet_model.pt")  # 12MB, ternary weights

# Dequantize for Phase 7
full_precision_model = dequantize_bitnet(bitnet_model)  # 100MB, float32

# Train SVF experts (full precision)
experts = train_svf_experts(full_precision_model, datasets)  # z-vectors are float32

# Phase 7 → Phase 8 handoff
save("phase7_full_precision_model.pt", full_precision_model)  # 100MB
save("phase7_experts.pt", experts)  # 80KB

# Phase 8 re-quantizes
phase8_requantize_and_compress(full_precision_model, experts)
```

---

### Option 2: BitNet-Aware SVF (Novel, Higher Risk)

**Approach**:
- Keep model 1.58-bit throughout
- Adapt SVF to work with quantized weights
- z-vectors modulate scaling factors α, not W directly

**Modified SVF formula**:
```
Standard SVF: W' = U(Σ ⊗ diag(z))V^T

BitNet-Aware SVF:
  W_ternary = {-1, 0, +1}  (stays quantized)
  α_adapted = α_base × z    (z modulates scaling factor)
  W' = α_adapted × W_ternary
```

**Memory during Phase 7**:
- Base model: 25M params × 1.58 bits = 5MB (stays quantized!)
- Scaling factors α: 25M × 32 bits = 100MB (full precision)
- Expert z-vectors: 5 × 4K × 32 bits = 80KB
- **Total**: ~105MB (but base weights stay compressed)

**Pros**:
- ✅ Maintains 1.58-bit compression
- ✅ Novel research contribution
- ✅ No re-quantization needed
- ✅ True "adapt without decompressing"

**Cons**:
- ❌ Unproven technique (Transformer² doesn't cover this)
- ❌ Research risk (may not work as well)
- ❌ z-vectors are smaller (only modulate α, not W)
- ❌ Limited expressiveness (α is per-layer, not per-param)

**Implementation**:
```python
class BitNetAwareSVF:
    def __init__(self, bitnet_model):
        self.W_ternary = bitnet_model.quantized_weights  # {-1, 0, +1}
        self.alpha_base = bitnet_model.scaling_factors   # float32

        # Expert z-vectors modulate alpha (not W)
        self.z_vectors = nn.ParameterDict({
            layer_name: nn.Parameter(torch.ones(1))  # 1 value per layer!
            for layer_name in self.W_ternary.keys()
        })

    def forward(self, x, expert_name):
        for layer_name, W_ternary in self.W_ternary.items():
            # Modulate scaling factor
            alpha_adapted = self.alpha_base[layer_name] * self.z_vectors[layer_name]

            # Apply quantized weights with adapted scaling
            W = alpha_adapted * W_ternary  # Still ternary structure
            x = F.linear(x, W, bias=None)

        return x
```

**Problem**: z-vectors are now **per-layer** (1 value), not **per-singular-value** (4K values).
- This drastically reduces expert expressiveness
- May not be enough to specialize effectively

---

### Option 3: Hybrid - Quantized Base + Float Experts (Compromise)

**Approach**:
- Base model stays 1.58-bit
- **During expert training**: Dequantize temporarily for SVF
- **At inference**: Store experts as float deltas, compose at runtime

**Memory**:
- **Storage**:
  - Base model: 5MB (quantized)
  - Expert z-vectors: 5 × 4K × 32 bits = 80KB (full precision)
  - **Total storage**: ~5.1MB
- **Inference** (when expert applied):
  - Dequantize base: 100MB
  - Apply SVF with z-vectors: 100MB
  - **Runtime**: ~100MB

**Pros**:
- ✅ Storage stays compressed (5.1MB)
- ✅ SVF training works as designed (full precision)
- ✅ No research risk
- ✅ Experts are high quality (full precision)

**Cons**:
- ❌ Inference requires dequantization (100MB runtime)
- ❌ Defeats purpose of BitNet for deployment
- ❌ Latency hit from dequantization

**When this makes sense**:
- If Phase 8 will re-quantize anyway (280× compression)
- If Phase 7 is just an intermediate step
- If we care more about expert quality than compression

---

## Recommendation: Option 1 (Dequantize for Phase 7)

### Rationale

1. **Phase 8 Re-Quantizes Anyway**:
   - Phase 8 applies SeedLM + VPTQ + Hypercompression (280× total)
   - Whether Phase 7 is 1.58-bit or float32 doesn't matter for Phase 8 input
   - Phase 8 will compress both base model + experts together

2. **SVF Quality Matters**:
   - Phase 7 is about creating **high-quality expert specializations**
   - Full-precision z-vectors have more expressiveness
   - Research risk is high for Option 2 (BitNet-Aware SVF)

3. **Memory is Acceptable**:
   - 100MB for Phase 7 training is manageable (GTX 1660 has 6GB)
   - Temporary decompression for 3-4 days is acceptable
   - Final deployment (Phase 8) will be 0.4MB anyway

4. **Alignment with Pipeline**:
   - Phase 4 quantizes: 100MB → 12MB (8.2×)
   - Phase 5-6: Stay at 12MB
   - **Phase 7: Temporarily 100MB** (dequantize for SVF)
   - Phase 8: 100MB → 0.4MB (280×, final compression)

### Implementation

```python
# Phase 6 → Phase 7
bitnet_model = torch.load("phase6_output/bitnet_model.pt")  # 12MB

# Dequantize for Phase 7 SVF training
print("Dequantizing BitNet model for Phase 7 SVF training...")
full_precision_model = dequantize_bitnet(bitnet_model)  # 100MB

# Verify model functionality
assert test_model_accuracy(full_precision_model) >= 0.85, "Dequantization broke model"

# Train SVF experts on full-precision model
stage1_output = stage1_expert_discovery(full_precision_model)
experts = stage2_svf_training(full_precision_model, stage1_output)
optimal_mixture = stage3_adas_search(full_precision_model, experts)

# Phase 7 → Phase 8
output = {
    "model": full_precision_model,  # 100MB (float32)
    "experts": experts,              # 80KB (float32)
    "optimal_mixture": optimal_mixture,
    "note": "Model dequantized for SVF training, ready for Phase 8 re-quantization"
}

torch.save(output, "phase7_output/model_and_experts.pt")
```

---

## Updated Phase 7 Pipeline

```
Phase 6 Output:
├─ BitNet 1.58-bit model: 12MB
├─ Eudaimonia baked: ✅
├─ Tool/persona baked: ✅
└─ Performance: 61.3% SWE-Bench, 79.4% aggregate

    ↓ DEQUANTIZE

Phase 7 Training (Full Precision):
├─ Stage 1: Model self-analysis, expert discovery
│   • Model: 100MB (dequantized)
│   • Generate datasets via OpenRouter ($150)
│
├─ Stage 2: SVF training (REINFORCE + KL)
│   • Train N=5 experts (z-vectors: float32)
│   • Model validation every epoch
│   • Total expert params: 20KB
│
└─ Stage 3: ADAS search (NSGA-II + model guidance)
    • Compose experts, search optimal mixture
    • Model self-evaluation at each generation
    • Validate eudaimonia + edge-of-chaos preservation

Phase 7 Output (Full Precision):
├─ Base model: 100MB (float32)
├─ Expert library: 80KB (5 experts, float32)
├─ Optimal mixture: α* = [1.8, 1.3, 1.0, 0.9, 1.1]
├─ Performance: 85.4% overall accuracy
└─ Note: "Ready for Phase 8 re-quantization + final compression"

    ↓ PHASE 8

Phase 8 Output (280× Compression):
├─ SeedLM: 100MB → 25MB (4×)
├─ VPTQ: 25MB → 2.5MB (10×)
├─ Hypercompression: 2.5MB → 0.36MB (7×)
└─ Final: 0.36MB (280× total from 100MB)
```

---

## Memory Analysis

### Phase-by-Phase Memory Footprint

| Phase | Model Size | Quantization | VRAM (Training) | Notes |
|-------|------------|--------------|-----------------|-------|
| Phase 1 | 100MB | Float32 | 2GB | 3× 25M models |
| Phase 2 | 100MB | Float32 | 2GB | Evolutionary merge |
| Phase 3 | 112MB | Float32 | 2.5GB | + reasoning thoughts |
| **Phase 4** | **12MB** | **1.58-bit** | 1GB | **8.2× compression** |
| Phase 5 | 12MB | 1.58-bit | 1GB | Curriculum learning |
| Phase 6 | 12MB | 1.58-bit | 1GB | Prompt baking |
| **Phase 7** | **100MB** | **Float32** | 2GB | **Dequantize for SVF** |
| **Phase 8** | **0.4MB** | **Custom** | 500MB | **280× compression** |

**Key Insight**: Phase 7 is the **only phase that temporarily decompresses** for expert training quality.

---

## Risk Assessment

### Risk 1: Dequantization Loses BitNet Benefits (Medium)

**Impact**: Phase 7 temporarily uses 8× more memory

**Mitigation**:
- Only for 3-4 days (training duration)
- Still fits in 6GB VRAM (GTX 1660)
- Phase 8 re-compresses to 0.4MB (final deployment)

### Risk 2: Re-Quantization in Phase 8 Hurts Experts (Low)

**Impact**: Expert quality might degrade during Phase 8 compression

**Mitigation**:
- Expert z-vectors are small (20KB total)
- Can store experts uncompressed in Phase 8 (negligible size)
- Test: Compare expert performance before/after Phase 8

### Risk 3: Model Changes Between Quantize/Dequantize (Low)

**Impact**: Dequantization might not perfectly reconstruct original

**Mitigation**:
- Test dequantization accuracy: Should recover ≥99% of pre-quantization accuracy
- If <99%, use Phase 3 output (pre-quantization) for Phase 7 instead

---

## Alternative: Skip BitNet in Phase 4 (Not Recommended)

**If dequantization is unacceptable**, we could:
- Skip Phase 4 BitNet quantization
- Phase 3 → Phase 5-6-7 (stay float32)
- Only quantize in Phase 8 (final compression)

**Why not recommended**:
- Phase 4 BitNet is valuable research validation (1.58-bit LLMs)
- Phase 5-6 benefit from smaller model (faster training)
- Phase 7 dequantization is acceptable trade-off

---

## Documentation Updates Needed

1. **PHASE7_LOGICAL_UNDERSTANDING.md**: Add "BitNet Dequantization" section
2. **PHASE7_COMPLETE_GUIDE.md**: Add dequantization step in handoff
3. **PHASE7_SELF_GUIDED_SYSTEM.md**: Update memory estimates (12MB → 100MB)
4. **Phase 4 → Phase 7 handoff**: Document dequantization requirement
5. **Phase 7 → Phase 8 handoff**: Clarify model is full precision

---

## Conclusion

**Decision**: **Option 1 - Dequantize for Phase 7**

**Justification**:
- SVF requires full precision for quality expert specialization
- Phase 8 will re-compress anyway (280×)
- 100MB is acceptable for 3-4 days of training
- No research risk (proven technique)

**Action Items**:
1. Update all Phase 7 docs to reflect dequantization step
2. Add dequantization utility function to codebase
3. Test dequantization accuracy (target: ≥99% recovery)
4. Update memory estimates in timelines
5. Document Phase 4 → Phase 7 handoff clearly

---

**Status**: ⚠️ **Design decision made, documentation updates in progress**
