"""
Secure Checkpoint Utilities using SafeTensors + JSON.

ISS-004 REMEDIATION: This module provides secure checkpoint save/load
that eliminates pickle-based arbitrary code execution vulnerabilities.

Architecture:
- Model weights: SafeTensors format (binary, no code execution)
- Config/metadata: JSON format (human-readable, no code execution)
- Optimizer state: SafeTensors for tensor values + JSON for non-tensor metadata

Security Guarantee:
- NO torch.load() with weights_only=False
- NO pickle deserialization of untrusted data
- All checkpoint loading is safe against malicious files
"""

import json
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file, save_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    warnings.warn(
        "safetensors not installed. Run: pip install safetensors>=0.4.0\n"
        "Checkpoint operations will fail until safetensors is installed."
    )


def _serialize_config(config: Any) -> Dict[str, Any]:
    """Convert config (dataclass or dict) to JSON-serializable dict."""
    if config is None:
        return {}
    if is_dataclass(config) and not isinstance(config, type):
        # Recursively convert nested dataclasses
        result = {}
        for key, value in asdict(config).items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif hasattr(value, "__dict__") and not isinstance(value, (dict, list)):
                result[key] = _serialize_config(value)
            else:
                result[key] = value
        return result
    if isinstance(config, dict):
        result = {}
        for key, value in config.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif is_dataclass(value) and not isinstance(value, type):
                result[key] = _serialize_config(value)
            else:
                result[key] = value
        return result
    return {"value": str(config)}


def _extract_optimizer_tensors(
    optimizer_state: Dict[str, Any]
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Separate optimizer state into tensors (for SafeTensors) and metadata (for JSON).

    Returns:
        (tensors_dict, metadata_dict)
    """
    tensors = {}
    metadata = {"state": {}, "param_groups": optimizer_state.get("param_groups", [])}

    for param_id, param_state in optimizer_state.get("state", {}).items():
        param_key = str(param_id)
        metadata["state"][param_key] = {}

        for key, value in param_state.items():
            if isinstance(value, torch.Tensor):
                tensor_key = f"opt_state_{param_key}_{key}"
                tensors[tensor_key] = value
                metadata["state"][param_key][key] = {"_tensor_ref": tensor_key}
            else:
                # Scalar or other serializable value
                if isinstance(value, (int, float, bool, str, type(None))):
                    metadata["state"][param_key][key] = value
                else:
                    metadata["state"][param_key][key] = str(value)

    return tensors, metadata


def _reconstruct_optimizer_state(
    tensors: Dict[str, torch.Tensor], metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Reconstruct optimizer state from tensors and metadata."""
    state: Dict[str, Any] = {}

    for param_key, param_state in metadata.get("state", {}).items():
        param_id = int(param_key) if param_key.isdigit() else param_key
        state[param_id] = {}

        for key, value in param_state.items():
            if isinstance(value, dict) and "_tensor_ref" in value:
                tensor_key = value["_tensor_ref"]
                if tensor_key in tensors:
                    state[param_id][key] = tensors[tensor_key]
            else:
                state[param_id][key] = value

    return {"state": state, "param_groups": metadata.get("param_groups", [])}


def save_checkpoint(
    model: nn.Module,
    output_path: Union[str, Path],
    config: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    optimizer_state: Optional[Dict] = None,
    extra_tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> Path:
    """
    Save model checkpoint securely using SafeTensors + JSON.

    File structure created:
    - {output_path}.safetensors: Model weights + any extra tensors
    - {output_path}.json: Config + metadata
    - {output_path}.optimizer.safetensors: Optimizer tensor state (if provided)
    - {output_path}.optimizer.json: Optimizer non-tensor state (if provided)

    Args:
        model: PyTorch model to save
        output_path: Base path for checkpoint files (without extension)
        config: Configuration (dataclass or dict) - will be JSON serialized
        metadata: Additional metadata dict
        optimizer_state: Optimizer state_dict() for resuming training
        extra_tensors: Additional tensors to save (e.g., tokenizer embeddings)

    Returns:
        Path to the main safetensors file

    Raises:
        ImportError: If safetensors is not installed
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError(
            "safetensors required for secure checkpointing. "
            "Install with: pip install safetensors>=0.4.0"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare model state dict
    tensors = {k: v for k, v in model.state_dict().items()}

    # Add extra tensors if provided
    if extra_tensors:
        for key, tensor in extra_tensors.items():
            tensors[f"extra_{key}"] = tensor

    # Save model weights with SafeTensors
    safetensors_path = output_path.with_suffix(".safetensors")
    save_file(tensors, str(safetensors_path))

    # Save config + metadata as JSON
    json_data = {
        "config": _serialize_config(config),
        "metadata": metadata or {},
        "has_optimizer": optimizer_state is not None,
        "has_extra_tensors": extra_tensors is not None,
        "extra_tensor_keys": list(extra_tensors.keys()) if extra_tensors else [],
    }

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)

    # Save optimizer state if provided
    if optimizer_state:
        opt_tensors, opt_metadata = _extract_optimizer_tensors(optimizer_state)

        if opt_tensors:
            opt_tensors_path = output_path.with_suffix(".optimizer.safetensors")
            save_file(opt_tensors, str(opt_tensors_path))

        opt_json_path = output_path.with_suffix(".optimizer.json")
        with open(opt_json_path, "w", encoding="utf-8") as f:
            json.dump(opt_metadata, f, indent=2, default=str)

    return safetensors_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    load_optimizer: bool = False,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model checkpoint securely from SafeTensors + JSON.

    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Base path to checkpoint (with or without extension)
        device: Device to load tensors to
        load_optimizer: Whether to load optimizer state
        strict: Whether to enforce strict state_dict loading

    Returns:
        Dict containing:
        - config: Loaded config dict
        - metadata: Loaded metadata dict
        - optimizer_state_dict: Optimizer state (if load_optimizer=True and exists)
        - extra_tensors: Any extra tensors saved with checkpoint

    Raises:
        FileNotFoundError: If checkpoint files don't exist
        ImportError: If safetensors not installed
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError(
            "safetensors required for secure checkpointing. "
            "Install with: pip install safetensors>=0.4.0"
        )

    checkpoint_path = Path(checkpoint_path)

    # Handle extension variations
    if checkpoint_path.suffix == ".safetensors":
        base_path = checkpoint_path.with_suffix("")
    elif checkpoint_path.suffix == ".json":
        base_path = checkpoint_path.with_suffix("")
    else:
        base_path = checkpoint_path

    safetensors_path = base_path.with_suffix(".safetensors")
    json_path = base_path.with_suffix(".json")

    if not safetensors_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {safetensors_path}")

    # Load model weights
    all_tensors = load_file(str(safetensors_path), device=device)

    # Separate model weights from extra tensors
    model_state = {}
    extra_tensors = {}

    for key, tensor in all_tensors.items():
        if key.startswith("extra_"):
            extra_tensors[key[6:]] = tensor  # Remove "extra_" prefix
        else:
            model_state[key] = tensor

    model.load_state_dict(model_state, strict=strict)

    # Load JSON metadata
    config = {}
    metadata = {}
    has_optimizer = False

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        config = json_data.get("config", {})
        metadata = json_data.get("metadata", {})
        has_optimizer = json_data.get("has_optimizer", False)

    result = {
        "config": config,
        "metadata": metadata,
        "extra_tensors": extra_tensors if extra_tensors else None,
    }

    # Load optimizer state if requested
    if load_optimizer and has_optimizer:
        opt_tensors_path = base_path.with_suffix(".optimizer.safetensors")
        opt_json_path = base_path.with_suffix(".optimizer.json")

        opt_tensors = {}
        opt_metadata = {}

        if opt_tensors_path.exists():
            opt_tensors = load_file(str(opt_tensors_path), device=device)

        if opt_json_path.exists():
            with open(opt_json_path, "r", encoding="utf-8") as f:
                opt_metadata = json.load(f)

        result["optimizer_state_dict"] = _reconstruct_optimizer_state(opt_tensors, opt_metadata)

    return result


def migrate_legacy_checkpoint(
    legacy_path: Union[str, Path],
    output_path: Union[str, Path],
    model: Optional[nn.Module] = None,
) -> Path:
    """
    Migrate a legacy torch.save checkpoint to secure SafeTensors format.

    WARNING: This function uses torch.load with weights_only=False ONCE
    to read the legacy checkpoint. Only use this on TRUSTED checkpoints
    that you created yourself.

    Args:
        legacy_path: Path to legacy .pt/.pth checkpoint
        output_path: Path for new secure checkpoint
        model: Optional model to validate state dict compatibility

    Returns:
        Path to new safetensors file
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors required for migration")

    legacy_path = Path(legacy_path)
    output_path = Path(output_path)

    # Load legacy checkpoint (TRUSTED sources only)
    # ISS-004: This is the ONLY place weights_only=False is acceptable,
    # and only for migrating YOUR OWN legacy checkpoints
    print(f"WARNING: Loading legacy checkpoint from {legacy_path}")
    print("Only proceed if this is a checkpoint YOU created.")

    checkpoint = torch.load(legacy_path, map_location="cpu", weights_only=False)

    # Extract components
    if isinstance(checkpoint, dict):
        model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        config = checkpoint.get("config", {})
        metadata = checkpoint.get("metadata", {})
        optimizer_state = checkpoint.get("optimizer_state_dict")

        # Handle case where checkpoint IS the state dict
        if model_state is None and any(k.endswith(".weight") for k in checkpoint.keys()):
            model_state = checkpoint
            config = {}
            metadata = {}
            optimizer_state = None
    else:
        # Checkpoint is just a state dict
        model_state = checkpoint
        config = {}
        metadata = {}
        optimizer_state = None

    # Validate against model if provided
    if model is not None:
        model.load_state_dict(model_state)
        print("State dict validated against model")

    # Create a temporary module to save
    class TempModule(nn.Module):
        def __init__(self, state_dict: Dict[str, torch.Tensor]) -> Any:
            super().__init__()
            for name, param in state_dict.items():
                # Register as buffer (not parameter) to preserve exact structure
                self.register_buffer(name.replace(".", "__DOT__"), param)

        def state_dict(self, *args, **kwargs) -> Any:
            # Return original keys
            return {name.replace("__DOT__", "."): buf for name, buf in self._buffers.items()}

    temp_module = TempModule(model_state)

    # Save in new secure format
    return save_checkpoint(
        model=temp_module,
        output_path=output_path,
        config=config,
        metadata=metadata,
        optimizer_state=optimizer_state,
    )


# Convenience functions for common patterns


def save_model_only(model: nn.Module, path: Union[str, Path]) -> Path:
    """Save just model weights without config/optimizer."""
    return save_checkpoint(model, path)


def load_model_only(model: nn.Module, path: Union[str, Path], device: str = "cpu") -> nn.Module:
    """Load just model weights, return the model."""
    load_checkpoint(model, path, device=device)
    return model
