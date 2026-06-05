"""
k-Formula Utilities (Layer 2)

k(L)-only utilities that DO NOT require MOO or other heavy dependencies.
These provide adaptive scaling based on the verified k(L) = -0.0137*log10(L) + 0.1593 formula.

Submodules:
    scheduling: Learning rate schedules, difficulty curves
    layer_ratios: Merge ratios, sparsity, compression ratios per layer
    routing: Expert routing temperature and top-k selection
    adaptive: Thought count, baking strength based on input properties

Usage:
    from src.cross_phase.meta_calculus.k_utils import (
        k_learning_rate_schedule,
        get_layer_merge_ratio,
        get_routing_temperature,
        get_thought_count,
    )

Design:
    - All functions are stateless and pure
    - No dependencies on other Layer 2 modules
    - Only imports from Layer 1 (k_formula.py)
"""

from .scheduling import (
    k_learning_rate_schedule,
    create_k_lr_scheduler,
    k_difficulty_schedule,
    k_warmup_schedule,
    KScheduleConfig,
)

from .layer_ratios import (
    get_layer_merge_ratio,
    get_all_merge_ratios,
    get_layer_sparsity,
    get_all_sparsities,
    get_layer_compression_ratio,
    get_all_compression_ratios,
    LayerRatioConfig,
)

from .routing import (
    get_routing_temperature,
    get_adaptive_top_k,
    get_routing_noise_scale,
    RoutingConfig,
)

from .adaptive import (
    get_thought_count,
    get_baking_strength,
    get_half_baking_ratio,
    get_quantization_threshold_scale,
    AdaptiveConfig,
)

__all__ = [
    # scheduling
    "k_learning_rate_schedule",
    "create_k_lr_scheduler",
    "k_difficulty_schedule",
    "k_warmup_schedule",
    "KScheduleConfig",
    # layer_ratios
    "get_layer_merge_ratio",
    "get_all_merge_ratios",
    "get_layer_sparsity",
    "get_all_sparsities",
    "get_layer_compression_ratio",
    "get_all_compression_ratios",
    "LayerRatioConfig",
    # routing
    "get_routing_temperature",
    "get_adaptive_top_k",
    "get_routing_noise_scale",
    "RoutingConfig",
    # adaptive
    "get_thought_count",
    "get_baking_strength",
    "get_half_baking_ratio",
    "get_quantization_threshold_scale",
    "AdaptiveConfig",
]
