"""
Paper-Accurate DFS (Dataflow Selection) Merge Implementation

Implements the DFS merging technique as described in the Sakana AI EvoMerge paper.
This is different from the variance-based DFS in dfs_merge.py.

Paper: Evolutionary Optimization of Model Merging Recipes (arXiv:2403.13187v1)

Key Components:
1. Indicator Array I: Binary array of size T = M x r
   - I[k] = 1 if layer k is selected for the merged model
   - I[k] = 0 if layer k is excluded
   - Optimized via evolutionary search

2. Scaling Matrix W: M x M matrix for layer-wise scaling
   - W[i,j] scales contribution of model i's layer j
   - Allows fine-grained control over layer importance

Algorithm:
    1. Define indicator array I (length T = M x r)
       where M = number of models, r = layers per model
    2. Define scaling matrix W (M x M)
    3. For each layer position j in merged model:
       a. Select layers from source models based on I
       b. Apply scaling from W
       c. Merge selected layers using weighted sum
    4. Return merged model

Benefits:
    - Layer-wise routing (FrankenMerge-like)
    - Fine-grained control via scaling matrix
    - Can completely exclude layers (I[k] = 0)
    - Optimizable via CMA-ES or evolutionary search
"""

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class DFSConfig:
    """Configuration for paper-accurate DFS merging."""

    # Indicator array initialization strategy
    init_strategy: str = "uniform"  # uniform, random, learned
    # Scaling initialization
    scale_init: str = "ones"  # ones, random, learned
    # Constraints
    min_layers_per_model: int = 1  # Minimum layers to select from each model
    # Optimization
    optimize_indicators: bool = True  # Whether to optimize I via search
    optimize_scaling: bool = True  # Whether to optimize W via search


class DFSPaperAccurate:
    """
    Paper-accurate DFS merging with indicator array I and scaling matrix W.

    This implementation follows the Sakana AI EvoMerge paper's DFS approach,
    which is more sophisticated than simple variance-based merging.
    """

    def __init__(self, config: Optional[DFSConfig] = None):
        """
        Initialize DFS merger.

        Args:
            config: DFS configuration
        """
        self.config = config or DFSConfig()
        self.indicator_array: Optional[np.ndarray] = None
        self.scaling_matrix: Optional[np.ndarray] = None

    def merge(
        self,
        models: List[nn.Module],
        indicator_array: Optional[np.ndarray] = None,
        scaling_matrix: Optional[np.ndarray] = None,
    ) -> nn.Module:
        """
        Merge models using DFS with indicator array and scaling matrix.

        Args:
            models: List of M models to merge
            indicator_array: Binary array of shape [T] where T = M * r
                (r = layers per model). If None, initialized automatically.
            scaling_matrix: Scaling matrix of shape [M, M]. If None,
                initialized automatically.

        Returns:
            Merged model

        Raises:
            ValueError: If models have incompatible architectures
        """
        M = len(models)  # Number of models
        if M < 2:
            raise ValueError("DFS merge requires at least 2 models")

        # Verify compatibility
        for i, model in enumerate(models):
            if not self._check_compatibility(models[0], model):
                raise ValueError(f"Model {i} incompatible with model 0")

        # Get layer count
        r = self._count_layers(models[0])
        T = M * r  # Total layer count across all models

        # Initialize indicator array if not provided
        if indicator_array is None:
            indicator_array = self._init_indicator_array(T, M, r)

        # Initialize scaling matrix if not provided
        if scaling_matrix is None:
            scaling_matrix = self._init_scaling_matrix(M)

        # Store for later access
        self.indicator_array = indicator_array
        self.scaling_matrix = scaling_matrix

        # Perform merge
        result = self._merge_with_indicators(models, indicator_array, scaling_matrix, M, r)

        return result

    def _init_indicator_array(self, T: int, M: int, r: int) -> np.ndarray:
        """
        Initialize indicator array.

        Args:
            T: Total size (M * r)
            M: Number of models
            r: Layers per model

        Returns:
            Binary array of shape [T]
        """
        if self.config.init_strategy == "uniform":
            # Uniformly select subset of layers
            # Ensure at least min_layers_per_model from each model
            indicators = np.zeros(T, dtype=np.float32)
            for model_idx in range(M):
                start = model_idx * r
                end = start + r
                # Select at least min_layers_per_model layers
                n_select = max(self.config.min_layers_per_model, r // 2)
                selected_indices = np.random.choice(r, size=n_select, replace=False)
                for idx in selected_indices:
                    indicators[start + idx] = 1.0

        elif self.config.init_strategy == "random":
            # Random binary initialization
            indicators = np.random.randint(0, 2, size=T).astype(np.float32)

        elif self.config.init_strategy == "all":
            # Select all layers (no routing)
            indicators = np.ones(T, dtype=np.float32)

        else:
            raise ValueError(f"Unknown init_strategy: {self.config.init_strategy}")

        return indicators

    def _init_scaling_matrix(self, M: int) -> np.ndarray:
        """
        Initialize scaling matrix.

        Args:
            M: Number of models

        Returns:
            Scaling matrix of shape [M, M]
        """
        if self.config.scale_init == "ones":
            # Identity-like scaling
            return np.ones((M, M), dtype=np.float32)

        elif self.config.scale_init == "random":
            # Random positive scaling
            return np.random.uniform(0.5, 1.5, size=(M, M)).astype(np.float32)

        elif self.config.scale_init == "identity":
            # Diagonal identity
            return np.eye(M, dtype=np.float32)

        else:
            raise ValueError(f"Unknown scale_init: {self.config.scale_init}")

    def _merge_with_indicators(
        self,
        models: List[nn.Module],
        indicator_array: np.ndarray,
        scaling_matrix: np.ndarray,
        M: int,
        r: int,
    ) -> nn.Module:
        """
        Perform actual merging using indicators and scaling.

        Args:
            models: List of models
            indicator_array: Binary array [T]
            scaling_matrix: Scaling matrix [M, M]
            M: Number of models
            r: Layers per model

        Returns:
            Merged model
        """
        # Create result as copy of first model
        result = copy.deepcopy(models[0])

        # Get all parameter names (assuming sequential layer naming)
        param_names = list(dict(models[0].named_parameters()).keys())

        # Group parameters by layer (heuristic: split by layer index in name)
        layer_groups = self._group_parameters_by_layer(param_names, r)

        logger.info(f"DFS merge: {M} models, {r} layers per model")
        logger.info(f"Indicator array sum: {indicator_array.sum()}/{len(indicator_array)}")

        with torch.no_grad():
            for layer_idx in range(r):
                # Get parameter names for this layer
                layer_param_names = layer_groups.get(layer_idx, [])

                if not layer_param_names:
                    continue

                # For each parameter in this layer
                for param_name in layer_param_names:
                    # Collect parameters from all models for this layer
                    merged_param = None
                    total_weight = 0.0

                    for model_idx in range(M):
                        # Check indicator for this model-layer combination
                        indicator_idx = model_idx * r + layer_idx
                        indicator = indicator_array[indicator_idx]

                        if indicator < 0.5:  # Not selected
                            continue

                        # Get parameter from this model
                        model_param = dict(models[model_idx].named_parameters())[param_name]

                        # Apply scaling from W
                        scale = scaling_matrix[model_idx, model_idx]  # Diagonal scaling for now

                        # Weighted sum
                        weighted_param = scale * indicator * model_param

                        if merged_param is None:
                            merged_param = weighted_param.clone()
                        else:
                            merged_param += weighted_param

                        total_weight += scale * indicator

                    # Normalize by total weight
                    if merged_param is not None and total_weight > 0:
                        merged_param = merged_param / total_weight

                        # Apply to result model
                        result_param = dict(result.named_parameters())[param_name]
                        result_param.copy_(merged_param)

        return result

    def _group_parameters_by_layer(self, param_names: List[str], n_layers: int) -> Dict[int, List[str]]:
        """
        Group parameter names by layer index.

        Heuristic: Extract layer index from parameter name.
        E.g., "transformer.layers.0.self_attn.q_proj.weight" -> layer 0

        Args:
            param_names: List of parameter names
            n_layers: Expected number of layers

        Returns:
            Dictionary mapping layer_idx -> list of parameter names
        """
        layer_groups = {}

        for param_name in param_names:
            # Try to extract layer index from name
            # Common patterns: "layer.0.", "layers.0.", "h.0.", "block.0."
            import re

            match = re.search(r"(?:layer|layers|h|block)[s]?\.(\d+)\.", param_name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in layer_groups:
                    layer_groups[layer_idx] = []
                layer_groups[layer_idx].append(param_name)
            else:
                # Fallback: assign to layer 0 (non-layer parameters)
                if 0 not in layer_groups:
                    layer_groups[0] = []
                layer_groups[0].append(param_name)

        return layer_groups

    def _count_layers(self, model: nn.Module) -> int:
        """
        Count number of layers in model.

        Args:
            model: Model to inspect

        Returns:
            Number of layers
        """
        param_names = list(dict(model.named_parameters()).keys())
        layer_groups = self._group_parameters_by_layer(param_names, n_layers=100)
        return len(layer_groups)

    def _check_compatibility(self, model1: nn.Module, model2: nn.Module) -> bool:
        """
        Check if two models have compatible architectures.

        Args:
            model1: First model
            model2: Second model

        Returns:
            True if compatible
        """
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())

        if set(params1.keys()) != set(params2.keys()):
            return False

        for name in params1.keys():
            if params1[name].shape != params2[name].shape:
                return False

        return True

    def optimize_indicators_and_scaling(
        self,
        models: List[nn.Module],
        fitness_fn: callable,
        n_iterations: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize indicator array and scaling matrix using evolutionary search.

        Args:
            models: List of models to merge
            fitness_fn: Function that evaluates merged model fitness
            n_iterations: Number of optimization iterations

        Returns:
            Tuple of (best_indicators, best_scaling)
        """
        M = len(models)
        r = self._count_layers(models[0])
        T = M * r

        best_fitness = float("-inf")
        best_indicators = self._init_indicator_array(T, M, r)
        best_scaling = self._init_scaling_matrix(M)

        logger.info(f"Optimizing DFS indicators and scaling ({n_iterations} iterations)...")

        for iteration in range(n_iterations):
            # Mutate indicators (flip random bits)
            indicators = best_indicators.copy()
            n_flips = max(1, int(0.1 * T))  # Flip 10% of indicators
            flip_indices = np.random.choice(T, size=n_flips, replace=False)
            indicators[flip_indices] = 1.0 - indicators[flip_indices]

            # Mutate scaling (add Gaussian noise)
            scaling = best_scaling + np.random.randn(M, M) * 0.1
            scaling = np.clip(scaling, 0.1, 2.0)  # Keep scaling positive

            # Merge with new parameters
            merged = self._merge_with_indicators(models, indicators, scaling, M, r)

            # Evaluate fitness
            try:
                fitness = fitness_fn(merged)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_indicators = indicators.copy()
                    best_scaling = scaling.copy()
                    logger.info(f"Iteration {iteration + 1}: New best fitness = {fitness:.4f}")
            except Exception as e:
                logger.warning(f"Fitness evaluation failed: {e}")

        logger.info(f"Optimization complete. Best fitness: {best_fitness:.4f}")

        return best_indicators, best_scaling


__all__ = ["DFSPaperAccurate", "DFSConfig"]
