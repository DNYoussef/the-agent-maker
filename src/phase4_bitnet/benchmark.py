"""
Inference Benchmark for Phase 4 (BitNet)

Measures:
1. Inference speedup from quantization (target: 2-4x)
2. Memory reduction
3. Throughput (tokens/second)
4. Latency (ms per forward pass)

Note: Actual speedup depends on hardware support for binary operations.
Consumer GPUs may not see full theoretical speedup.
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.phase4_bitnet.compressed_model import CompressedModel
from src.phase4_bitnet.config import Phase4Config
from src.phase4_bitnet.quantizer import BitNetQuantizer


@dataclass
class BenchmarkResult:
    """Result of inference benchmark."""

    # Timing metrics (in milliseconds)
    fp32_latency_ms: float
    quantized_latency_ms: float
    speedup_ratio: float

    # Memory metrics (in MB)
    fp32_memory_mb: float
    quantized_memory_mb: float
    memory_reduction_ratio: float

    # Throughput (tokens per second)
    fp32_throughput: float
    quantized_throughput: float

    # Metadata
    num_warmup_runs: int
    num_benchmark_runs: int
    batch_size: int
    sequence_length: int
    device: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timing": {
                "fp32_latency_ms": self.fp32_latency_ms,
                "quantized_latency_ms": self.quantized_latency_ms,
                "speedup_ratio": self.speedup_ratio,
            },
            "memory": {
                "fp32_memory_mb": self.fp32_memory_mb,
                "quantized_memory_mb": self.quantized_memory_mb,
                "memory_reduction_ratio": self.memory_reduction_ratio,
            },
            "throughput": {
                "fp32_tokens_per_sec": self.fp32_throughput,
                "quantized_tokens_per_sec": self.quantized_throughput,
            },
            "config": {
                "num_warmup_runs": self.num_warmup_runs,
                "num_benchmark_runs": self.num_benchmark_runs,
                "batch_size": self.batch_size,
                "sequence_length": self.sequence_length,
                "device": self.device,
            },
        }


@contextmanager
def torch_inference_mode():
    """Context manager for inference (no gradients)."""
    with torch.no_grad():
        yield


def measure_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Tuple[float, float]:
    """
    Measure inference latency.

    Args:
        model: Model to benchmark
        input_ids: Input tensor
        attention_mask: Optional attention mask
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs

    Returns:
        Tuple of (mean_latency_ms, std_latency_ms)
    """
    model.eval()

    # Warmup
    with torch_inference_mode():
        for _ in range(num_warmup):
            if attention_mask is not None:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                _ = model(input_ids=input_ids)

    # Sync if CUDA
    if input_ids.is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch_inference_mode():
        for _ in range(num_runs):
            start = time.perf_counter()

            if attention_mask is not None:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                _ = model(input_ids=input_ids)

            if input_ids.is_cuda:
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    mean_latency = sum(latencies) / len(latencies)
    variance = sum((x - mean_latency) ** 2 for x in latencies) / len(latencies)
    std_latency = variance**0.5

    return mean_latency, std_latency


def measure_memory(model: nn.Module) -> float:
    """
    Measure model memory in MB.

    Args:
        model: Model to measure

    Returns:
        Memory in MB
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.nelement() * buffer.element_size()

    return total_bytes / (1024**2)


def benchmark_model(
    base_model: nn.Module,
    config: Optional[Phase4Config] = None,
    batch_size: int = 1,
    sequence_length: int = 128,
    num_warmup: int = 5,
    num_runs: int = 20,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Run full inference benchmark comparing FP32 and quantized models.

    Args:
        base_model: Base model to benchmark
        config: Phase 4 configuration
        batch_size: Batch size for benchmark
        sequence_length: Sequence length for benchmark
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        device: Device to run on

    Returns:
        BenchmarkResult with all metrics
    """
    if config is None:
        config = Phase4Config()

    # Create input
    input_ids = torch.randn(batch_size, sequence_length).to(device)

    # Move base model to device
    base_model = base_model.to(device)
    base_model.eval()

    # Measure FP32 performance
    print("Benchmarking FP32 model...")
    fp32_latency, fp32_std = measure_latency(
        base_model, input_ids, num_warmup=num_warmup, num_runs=num_runs
    )
    fp32_memory = measure_memory(base_model)

    # Create and compress model
    print("Quantizing model...")
    quantizer = BitNetQuantizer(config)
    compressed_model = CompressedModel(base_model, quantizer, config)
    compressed_model.compress()
    compressed_model = compressed_model.to(device)
    compressed_model.eval()

    # Measure quantized performance
    print("Benchmarking quantized model...")
    quant_latency, quant_std = measure_latency(
        compressed_model, input_ids, num_warmup=num_warmup, num_runs=num_runs
    )
    quant_memory = measure_memory(compressed_model)

    # Get compression stats for accurate memory comparison
    compression_stats = compressed_model.get_compression_stats()
    actual_quantized_mb = compression_stats.get("quantized_size_mb", quant_memory)

    # Calculate metrics
    speedup = fp32_latency / quant_latency if quant_latency > 0 else 0
    memory_reduction = fp32_memory / actual_quantized_mb if actual_quantized_mb > 0 else 0

    # Throughput (tokens per second)
    total_tokens = batch_size * sequence_length
    fp32_throughput = total_tokens / (fp32_latency / 1000)
    quant_throughput = total_tokens / (quant_latency / 1000)

    result = BenchmarkResult(
        fp32_latency_ms=fp32_latency,
        quantized_latency_ms=quant_latency,
        speedup_ratio=speedup,
        fp32_memory_mb=fp32_memory,
        quantized_memory_mb=actual_quantized_mb,
        memory_reduction_ratio=memory_reduction,
        fp32_throughput=fp32_throughput,
        quantized_throughput=quant_throughput,
        num_warmup_runs=num_warmup,
        num_benchmark_runs=num_runs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
    )

    return result


def print_benchmark_report(result: BenchmarkResult) -> None:
    """Print a human-readable benchmark report."""
    print("=" * 60)
    print("BITNET INFERENCE BENCHMARK REPORT")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Device: {result.device}")
    print(f"  Batch size: {result.batch_size}")
    print(f"  Sequence length: {result.sequence_length}")
    print(f"  Warmup runs: {result.num_warmup_runs}")
    print(f"  Benchmark runs: {result.num_benchmark_runs}")

    print(f"\nLatency:")
    print(f"  FP32:      {result.fp32_latency_ms:.3f} ms")
    print(f"  Quantized: {result.quantized_latency_ms:.3f} ms")
    print(f"  Speedup:   {result.speedup_ratio:.2f}x")

    # Evaluate speedup against target
    target_min, target_max = 2.0, 4.0
    if result.speedup_ratio >= target_min:
        speedup_status = "PASS"
    elif result.speedup_ratio >= 1.0:
        speedup_status = "PARTIAL (below 2x target)"
    else:
        speedup_status = "FAIL (slower than FP32)"
    print(f"  Target (2-4x): {speedup_status}")

    print(f"\nMemory:")
    print(f"  FP32:      {result.fp32_memory_mb:.2f} MB")
    print(f"  Quantized: {result.quantized_memory_mb:.2f} MB")
    print(f"  Reduction: {result.memory_reduction_ratio:.2f}x")

    print(f"\nThroughput:")
    print(f"  FP32:      {result.fp32_throughput:.0f} tokens/sec")
    print(f"  Quantized: {result.quantized_throughput:.0f} tokens/sec")

    print("=" * 60)

    # Explanation note
    print("\nNote: BitNet speedup depends on hardware support for")
    print("binary/ternary operations. Consumer GPUs may not achieve")
    print("full theoretical 2-4x speedup without custom kernels.")
    print("Memory reduction should be closer to theoretical 8x.")


def validate_speedup_target(
    result: BenchmarkResult, min_speedup: float = 1.5, min_memory_reduction: float = 2.0
) -> Tuple[bool, List[str]]:
    """
    Validate benchmark results against targets.

    Args:
        result: Benchmark result
        min_speedup: Minimum acceptable speedup (relaxed from 2x)
        min_memory_reduction: Minimum acceptable memory reduction

    Returns:
        Tuple of (passed: bool, issues: List[str])
    """
    issues = []

    if result.speedup_ratio < min_speedup:
        issues.append(f"Speedup {result.speedup_ratio:.2f}x below minimum {min_speedup}x")

    if result.memory_reduction_ratio < min_memory_reduction:
        issues.append(
            f"Memory reduction {result.memory_reduction_ratio:.2f}x "
            f"below minimum {min_memory_reduction}x"
        )

    return len(issues) == 0, issues


if __name__ == "__main__":
    print("Running BitNet inference benchmark...")

    # Create a simple test model that handles dtype conversion
    class SimpleTestModel(nn.Module):
        def __init__(self, hidden_size=256):
            super().__init__()
            self.hidden_size = hidden_size
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size)

        def forward(self, input_ids, attention_mask=None, labels=None):
            # Ensure input has correct shape
            x = input_ids
            if x.dim() == 2:
                # [batch, seq_len] -> [batch, seq_len, hidden]
                x = x.unsqueeze(-1).expand(-1, -1, self.hidden_size)

            # Match dtype of first layer weight
            x = x.to(self.linear1.weight.dtype)

            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            logits = self.output(x)

            if labels is not None:
                # Simple MSE loss for benchmark
                loss = nn.MSELoss()(logits, labels.unsqueeze(-1).expand_as(logits).to(logits.dtype))
                return type("Output", (), {"loss": loss, "logits": logits})()

            return logits

    # Create model and run benchmark
    config = Phase4Config(preserve_embedding_precision=False)  # Keep all layers same dtype
    model = SimpleTestModel()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Run benchmark
    result = benchmark_model(
        base_model=model,
        config=config,
        batch_size=4,
        sequence_length=64,
        num_warmup=3,
        num_runs=10,
        device=device,
    )

    # Print report
    print_benchmark_report(result)

    # Validate against targets
    passed, issues = validate_speedup_target(result)
    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
