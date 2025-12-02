"""
Fitness caching system for performance optimization.

This module provides an LRU cache for fitness evaluation results, using
model parameter hashing as the cache key. This significantly speeds up
evolutionary optimization by avoiding redundant fitness evaluations.
"""

import hashlib
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class FitnessCache:
    """
    LRU cache for fitness evaluation results.

    Uses SHA256 hash of model parameters as cache key. Implements
    least-recently-used (LRU) eviction policy when cache reaches max size.

    Example:
        >>> cache = FitnessCache(max_size=100)
        >>> fitness = {'composite': 0.185, 'components': {...}}
        >>> cache.put(model, fitness)
        >>> cached = cache.get(model)  # Returns fitness dict or None
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize LRU cache with maximum size.

        Args:
            max_size: Maximum number of cached entries (default: 100)
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def hash_model(self, model: nn.Module) -> str:
        """
        Compute SHA256 hash of model parameters.

        This creates a unique identifier for the model's parameter values.
        Two models with identical parameters will have the same hash.

        Args:
            model: PyTorch model to hash

        Returns:
            SHA256 hash string (64 hex characters)

        Example:
            >>> model = SimpleModel()
            >>> hash1 = cache.hash_model(model)
            >>> hash2 = cache.hash_model(model)
            >>> assert hash1 == hash2  # Same model, same hash
        """
        # Flatten all parameters into single tensor
        param_list = []
        for param in model.parameters():
            param_list.append(param.data.cpu().flatten())

        if not param_list:
            # No parameters (empty model)
            return hashlib.sha256(b"").hexdigest()

        # Concatenate all parameters
        all_params = torch.cat(param_list)

        # Convert to bytes
        param_bytes = all_params.numpy().tobytes()

        # Compute SHA256 hash
        hash_obj = hashlib.sha256(param_bytes)
        hash_str = hash_obj.hexdigest()

        return hash_str

    def get(self, model: nn.Module) -> Optional[Dict[str, Any]]:
        """
        Get cached fitness for model, or None if not found.

        Args:
            model: Model to lookup in cache

        Returns:
            Cached fitness dict, or None if not in cache

        Example:
            >>> fitness = cache.get(model)
            >>> if fitness:
            ...     print(f"Cache hit! Fitness: {fitness['composite']}")
            ... else:
            ...     print("Cache miss, need to evaluate")
        """
        model_hash = self.hash_model(model)

        if model_hash in self._cache:
            # Move to end (mark as recently used)
            self._cache.move_to_end(model_hash)
            return self._cache[model_hash]

        return None

    def put(self, model: nn.Module, fitness: Dict[str, Any]) -> None:
        """
        Store fitness in cache for given model.

        If cache is full, evicts least-recently-used entry.

        Args:
            model: Model to cache fitness for
            fitness: Fitness dict to store

        Example:
            >>> fitness = {'composite': 0.185, 'components': {...}}
            >>> cache.put(model, fitness)
        """
        model_hash = self.hash_model(model)

        # If already in cache, update and mark as recently used
        if model_hash in self._cache:
            self._cache[model_hash] = fitness
            self._cache.move_to_end(model_hash)
            return

        # Add new entry
        self._cache[model_hash] = fitness

        # Evict LRU if over capacity
        if len(self._cache) > self.max_size:
            # Remove oldest (first) item
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """
        Clear all cached entries.

        Example:
            >>> cache.clear()
            >>> assert cache.size() == 0
        """
        self._cache.clear()

    def size(self) -> int:
        """
        Return number of cached entries.

        Returns:
            Number of entries in cache

        Example:
            >>> cache.put(model1, fitness1)
            >>> cache.put(model2, fitness2)
            >>> assert cache.size() == 2
        """
        return len(self._cache)
