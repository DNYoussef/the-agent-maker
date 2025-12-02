# Phase 4: BitNet - Logical Understanding

**Version**: 2.0 (V2 Rebuild)
**Phase Purpose**: Quantize model to 1.58 bits for 8.2x compression, 2-4x speedup

---

## What This Phase Does (In Plain English)

Phase 4 compresses the model from Phase 3 by quantizing weights to **ternary values** {-1, 0, +1} (1.58 bits).

**Analogy**: Like compressing a high-res photo to a sketch:
- **Before**: Weights are 32-bit floats (e.g., 0.523, -1.847, 0.002)
- **After**: Weights are ternary (e.g., +1, -1, 0)
- **Result**: 8.2x smaller file, 2-4x faster inference, <10% accuracy loss

**Key Technique**: **Straight-Through Estimator (STE)** - quantize forward pass, keep full precision for gradients in backward pass.

---

## Key Research Papers

### "BitNet: Scaling 1-bit Transformers for Large Language Models"
### "The Era of 1-bit LLMs: Fine-tuning to 1.58-bit"

**What We Take**:
- Ternary quantization: `Q(w) = sign(w) if |w| > threshold else 0`
- Scaling factors α per layer
- STE for gradient flow
- Fine-tuning post-quantization

---

## Technical Flow

```python
def quantize_to_ternary(weight, threshold=0.1):
    """Quantize weight to {-1, 0, +1}"""
    quantized = torch.zeros_like(weight)
    quantized[weight > threshold] = 1.0
    quantized[weight < -threshold] = -1.0
    return quantized  # 0 for |w| < threshold

def forward_quantized(model, x):
    """Forward pass with quantized weights"""
    for layer in model.layers:
        # Quantize weights (forward only)
        w_quantized = quantize_to_ternary(layer.weight)

        # Scale by learned α
        w_scaled = layer.alpha * w_quantized

        # Standard forward pass
        x = F.linear(x, w_scaled, layer.bias)

    return x

def backward_full_precision(loss):
    """Backward pass with full precision (STE)"""
    # Gradients computed on full-precision weights
    loss.backward()  # Autograd handles STE automatically

def train_bitnet(model, dataloader):
    """Fine-tune quantized model"""
    for batch in dataloader:
        # Forward (quantized)
        output = forward_quantized(model, batch['input'])
        loss = criterion(output, batch['labels'])

        # Backward (full precision)
        backward_full_precision(loss)
        optimizer.step()
```

---

## Expected Inputs

**From Phase 3**: Reasoning-enhanced model (full precision)

---

## Expected Outputs

**To Phase 5**:
- Quantized model (1.58-bit weights)
- Compression ratio: 8.2x (e.g., 16GB → 2GB)
- Speedup: 2-4x inference
- Accuracy: <10% loss

---

## Success Criteria

- ✅ Model quantized to ternary {-1, 0, +1}
- ✅ 8.2x compression achieved
- ✅ 2-4x inference speedup on GTX 1660
- ✅ Accuracy degradation <10%
- ✅ Fine-tuning recovers most accuracy
- ✅ Model fits in 2GB (down from ~16GB)

---

**Next Phase**: [Phase 5: Forge Training](../phase5/LOGICAL_UNDERSTANDING.md)
