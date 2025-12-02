"""
Phase 8: Hypercompression

Final stage of triple compression pipeline.
Applies parametric curve fitting for extreme compression.

Research: "Hyper-Compression of LLM Weights"
Target: 6.25x additional compression with >90% retention.
Total pipeline: 2x * 20x * 6.25x = 250x compression

Curve Fitting Quality Metrics:
- R^2 (coefficient of determination): 1 - SS_res/SS_tot
- RMSE (root mean squared error): sqrt(mean((y - y_hat)^2))
- MAE (mean absolute error): mean(|y - y_hat|)

Expected R^2 > 0.95 for production quality compression.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import copy
import math

import torch
import torch.nn as nn


@dataclass
class CurveFitMetrics:
    """Detailed metrics for curve fitting quality."""
    r_squared: float           # Coefficient of determination (target > 0.95)
    rmse: float               # Root mean squared error
    mae: float                # Mean absolute error
    max_error: float          # Maximum reconstruction error
    compression_ratio: float  # Size reduction ratio
    num_parameters: int       # Number of curve parameters


@dataclass
class HyperConfig:
    """Configuration for hypercompression."""
    num_params: int = 8  # Parameters per curve
    curve_type: str = "bezier"  # bezier, polynomial, spline
    num_segments: int = 16  # Segments per layer
    optimization_steps: int = 100
    target_retention: float = 0.90
    preserve_layers: List[str] = None
    min_r_squared: float = 0.90        # Minimum acceptable R^2
    early_stop_r_squared: float = 0.99  # Stop optimization early if R^2 exceeds this


@dataclass
class HyperResult:
    """Result from hypercompression."""
    success: bool
    compressed_state: Dict[str, Any]
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    retention_score: float
    curve_stats: Dict
    # New: detailed fit metrics
    mean_r_squared: float = 0.0
    mean_rmse: float = 0.0
    layer_r_squared: Dict[str, float] = field(default_factory=dict)


class HyperCompressor:
    """
    Hypercompression: Parametric curve fitting.

    Process:
    1. Fit parametric curves to weight distributions
    2. Store only curve parameters
    3. Reconstruct weights via curve evaluation

    Benefits:
    - 6.25x additional compression
    - Smooth approximations
    - Stacks with SeedLM and VPTQ
    """

    def __init__(self, config: HyperConfig = None):
        """
        Initialize hypercompressor.

        Args:
            config: Hypercompression configuration
        """
        self.config = config or HyperConfig()
        if self.config.preserve_layers is None:
            self.config.preserve_layers = ['embed', 'norm', 'ln_', 'layernorm', 'bias']

    def compress(
        self,
        model: nn.Module,
        calibration_data: List[Any] = None,
        tokenizer: Any = None
    ) -> Tuple[nn.Module, HyperResult]:
        """
        Compress model using hypercompression.

        Args:
            model: Model to compress
            calibration_data: Optional calibration samples
            tokenizer: Optional tokenizer

        Returns:
            Tuple of (compressed_model, HyperResult)
        """
        print("  Hypercompression Stage: Parametric curve fitting")

        # Get original size
        original_state = model.state_dict()
        original_size = self._calculate_size(original_state)
        print(f"    Original size: {original_size:.2f} MB")

        # Compress each layer
        compressed_state = {}
        curve_stats = {}
        layer_r_squared = {}
        all_r_squared = []
        all_rmse = []

        for name, param in original_state.items():
            should_preserve = any(p in name.lower() for p in self.config.preserve_layers)

            if should_preserve or param.dim() < 2 or param.numel() < 100:
                # Preserve layer
                compressed_state[name] = {
                    'type': 'preserved',
                    'data': param.half()
                }
            else:
                # Apply hypercompression with R^2 metrics
                curve_params, retention, fit_metrics = self._fit_curves(param)
                compressed_state[name] = {
                    'type': 'hyper',
                    'curve_params': curve_params,
                    'shape': param.shape,
                    'curve_type': self.config.curve_type
                }

                curve_stats[name] = {
                    'num_params': len(curve_params.flatten()),
                    'original_params': param.numel(),
                    'compression': param.numel() / max(len(curve_params.flatten()), 1),
                    'retention': retention,
                    'r_squared': fit_metrics.r_squared,
                    'rmse': fit_metrics.rmse,
                    'mae': fit_metrics.mae,
                    'max_error': fit_metrics.max_error
                }

                layer_r_squared[name] = fit_metrics.r_squared
                all_r_squared.append(fit_metrics.r_squared)
                all_rmse.append(fit_metrics.rmse)

        # Calculate compressed size
        compressed_size = self._calculate_compressed_size(compressed_state)
        compression_ratio = original_size / max(compressed_size, 0.01)
        print(f"    Compressed size: {compressed_size:.2f} MB")
        print(f"    Compression ratio: {compression_ratio:.2f}x")

        # Create compressed model
        compressed_model = self._create_compressed_model(model, compressed_state)

        # Calculate retention
        retention = self._calculate_retention(curve_stats)
        print(f"    Retention score: {retention:.2%}")

        # Calculate mean R^2 and RMSE
        mean_r_squared = sum(all_r_squared) / max(len(all_r_squared), 1) if all_r_squared else 0.0
        mean_rmse = sum(all_rmse) / max(len(all_rmse), 1) if all_rmse else 0.0
        print(f"    Mean R^2: {mean_r_squared:.4f}")
        print(f"    Mean RMSE: {mean_rmse:.6f}")

        return compressed_model, HyperResult(
            success=True,
            compressed_state=compressed_state,
            original_size_mb=original_size,
            compressed_size_mb=compressed_size,
            compression_ratio=compression_ratio,
            retention_score=retention,
            curve_stats=curve_stats,
            mean_r_squared=mean_r_squared,
            mean_rmse=mean_rmse,
            layer_r_squared=layer_r_squared
        )

    def compute_r_squared(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> CurveFitMetrics:
        """
        Compute R^2 and other quality metrics for curve fitting.

        R^2 = 1 - SS_res / SS_tot
        where:
            SS_res = sum((y - y_hat)^2)  # Residual sum of squares
            SS_tot = sum((y - y_mean)^2) # Total sum of squares

        Args:
            original: Original tensor values
            reconstructed: Reconstructed values from curve

        Returns:
            CurveFitMetrics with R^2, RMSE, MAE, etc.
        """
        original = original.float()
        reconstructed = reconstructed.float()

        # Ensure same length
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]

        # Residuals
        residuals = original - reconstructed

        # SS_res (residual sum of squares)
        ss_res = (residuals ** 2).sum().item()

        # SS_tot (total sum of squares)
        y_mean = original.mean()
        ss_tot = ((original - y_mean) ** 2).sum().item()

        # R^2 (coefficient of determination)
        r_squared = 1.0 - (ss_res / max(ss_tot, 1e-10))
        r_squared = max(0.0, min(1.0, r_squared))

        # RMSE (root mean squared error)
        mse = (residuals ** 2).mean().item()
        rmse = math.sqrt(mse)

        # MAE (mean absolute error)
        mae = residuals.abs().mean().item()

        # Max error
        max_error = residuals.abs().max().item()

        # Compression ratio (original params / curve params)
        # This will be calculated at the caller level
        compression_ratio = 0.0

        return CurveFitMetrics(
            r_squared=r_squared,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            compression_ratio=compression_ratio,
            num_parameters=0
        )

    def _fit_curves(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, float, CurveFitMetrics]:
        """
        Fit parametric curves to tensor with R^2 metrics.

        Args:
            tensor: Weight tensor

        Returns:
            Tuple of (curve_parameters, retention, metrics)
        """
        flat = tensor.flatten().float()

        # Divide into segments
        segment_size = max(1, len(flat) // self.config.num_segments)
        num_segments = (len(flat) + segment_size - 1) // segment_size

        all_params = []
        all_reconstructed = []
        segment_r_squared = []

        for i in range(num_segments):
            start = i * segment_size
            end = min(start + segment_size, len(flat))
            segment = flat[start:end]

            # Fit curve to segment
            params = self._fit_segment(segment)
            all_params.append(params)

            # Reconstruct and calculate metrics
            reconstructed = self._evaluate_curve(params, len(segment))
            all_reconstructed.append(reconstructed)

            # Per-segment R^2
            metrics = self.compute_r_squared(segment, reconstructed)
            segment_r_squared.append(metrics.r_squared)

        # Stack parameters
        curve_params = torch.stack(all_params)

        # Reconstruct full tensor for overall metrics
        full_reconstructed = torch.cat(all_reconstructed)[:len(flat)]

        # Calculate overall metrics
        overall_metrics = self.compute_r_squared(flat, full_reconstructed)

        # Update compression ratio
        overall_metrics.compression_ratio = len(flat) / max(curve_params.numel(), 1)
        overall_metrics.num_parameters = curve_params.numel()

        # Retention is same as R^2 for consistency
        retention = overall_metrics.r_squared

        return curve_params.half(), retention, overall_metrics

    def _fit_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """Fit a parametric curve to a segment."""
        n = len(segment)

        if self.config.curve_type == "bezier":
            return self._fit_bezier(segment)
        elif self.config.curve_type == "polynomial":
            return self._fit_polynomial(segment)
        else:
            return self._fit_bezier(segment)  # Default

    def _fit_bezier(self, segment: torch.Tensor) -> torch.Tensor:
        """Fit Bezier curve to segment."""
        n = len(segment)
        num_control = self.config.num_params

        # Initialize control points
        control_points = torch.zeros(num_control)
        indices = torch.linspace(0, n - 1, num_control).long()
        for i, idx in enumerate(indices):
            if idx < n:
                control_points[i] = segment[idx]

        # Optimize control points
        control_points.requires_grad_(True)
        optimizer = torch.optim.Adam([control_points], lr=0.1)

        for _ in range(self.config.optimization_steps):
            reconstructed = self._evaluate_bezier(control_points, n)
            loss = ((segment - reconstructed) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return control_points.detach()

    def _evaluate_bezier(self, control_points: torch.Tensor, n: int) -> torch.Tensor:
        """Evaluate Bezier curve at n points."""
        t = torch.linspace(0, 1, n)
        num_control = len(control_points)
        degree = num_control - 1

        result = torch.zeros(n)
        for i in range(num_control):
            # Bernstein basis polynomial
            binom = math.comb(degree, i)
            basis = binom * (t ** i) * ((1 - t) ** (degree - i))
            result = result + basis * control_points[i]

        return result

    def _fit_polynomial(self, segment: torch.Tensor) -> torch.Tensor:
        """Fit polynomial to segment."""
        n = len(segment)
        degree = self.config.num_params - 1

        # Create Vandermonde matrix
        x = torch.linspace(-1, 1, n)
        V = torch.vander(x, N=self.config.num_params, increasing=True)

        # Least squares fit
        try:
            coeffs = torch.linalg.lstsq(V, segment.unsqueeze(1)).solution.squeeze()
        except Exception:
            coeffs = torch.zeros(self.config.num_params)

        return coeffs

    def _evaluate_curve(self, params: torch.Tensor, n: int) -> torch.Tensor:
        """Evaluate curve at n points."""
        if self.config.curve_type == "bezier":
            return self._evaluate_bezier(params, n)
        elif self.config.curve_type == "polynomial":
            return self._evaluate_polynomial(params, n)
        else:
            return self._evaluate_bezier(params, n)

    def _evaluate_polynomial(self, coeffs: torch.Tensor, n: int) -> torch.Tensor:
        """Evaluate polynomial at n points."""
        x = torch.linspace(-1, 1, n)
        result = torch.zeros(n)

        for i, c in enumerate(coeffs):
            result = result + c * (x ** i)

        return result

    def _calculate_size(self, state_dict: Dict) -> float:
        """Calculate size of state dict in MB."""
        total_bytes = 0
        for param in state_dict.values():
            if isinstance(param, torch.Tensor):
                if param.dtype == torch.float32:
                    total_bytes += param.numel() * 4
                elif param.dtype == torch.float16:
                    total_bytes += param.numel() * 2
                else:
                    total_bytes += param.numel() * 4
        return total_bytes / (1024 * 1024)

    def _calculate_compressed_size(self, compressed_state: Dict) -> float:
        """Calculate compressed state size in MB."""
        total_bytes = 0

        for name, data in compressed_state.items():
            if data['type'] == 'preserved':
                total_bytes += data['data'].numel() * 2  # FP16
            else:  # hyper
                total_bytes += data['curve_params'].numel() * 2  # FP16

        return total_bytes / (1024 * 1024)

    def _calculate_retention(self, curve_stats: Dict) -> float:
        """Calculate overall retention score."""
        retentions = [stats['retention'] for stats in curve_stats.values()]
        return sum(retentions) / max(len(retentions), 1) if retentions else 1.0

    def _create_compressed_model(
        self,
        original_model: nn.Module,
        compressed_state: Dict
    ) -> nn.Module:
        """Create model with decompressed weights."""
        model = copy.deepcopy(original_model)

        decompressed_state = {}
        for name, data in compressed_state.items():
            if data['type'] == 'preserved':
                decompressed_state[name] = data['data'].float()
            else:
                # Decompress from curves
                decompressed = self._decompress_tensor(
                    data['curve_params'],
                    data['shape'],
                    data['curve_type']
                )
                decompressed_state[name] = decompressed

        model.load_state_dict(decompressed_state)
        return model

    def _decompress_tensor(
        self,
        curve_params: torch.Tensor,
        shape: torch.Size,
        curve_type: str
    ) -> torch.Tensor:
        """Decompress tensor from curve parameters."""
        original_size = 1
        for s in shape:
            original_size *= s

        num_segments = curve_params.shape[0]
        segment_size = (original_size + num_segments - 1) // num_segments

        segments = []
        for params in curve_params:
            segment = self._evaluate_curve(params.float(), segment_size)
            segments.append(segment)

        flat = torch.cat(segments)[:original_size]
        return flat.reshape(shape)


__all__ = [
    'HyperCompressor',
    'HyperConfig',
    'HyperResult',
    'CurveFitMetrics'
]
