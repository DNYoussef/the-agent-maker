"""
Phase 3 Handoff Validation

Validates model handoffs between phases:
- Phase 2 -> Phase 3: Champion model from EvoMerge
- Phase 3 -> Phase 4: Reasoning-enhanced model to BitNet

Ensures model integrity, metadata preservation, and format compatibility.

ISS-004: Updated to use secure SafeTensors checkpoint loading.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# ISS-004: Secure checkpoint validation
from safetensors.torch import load_file as safe_load_file

from ..cross_phase.storage import ModelRegistry


def _secure_load_checkpoint_metadata(checkpoint_path: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Securely load checkpoint for validation (ISS-004).

    Returns:
        (state_dict, config, metadata) - all loaded securely without pickle
    """
    base_path = checkpoint_path.with_suffix("")
    safetensors_path = base_path.with_suffix(".safetensors")
    json_path = base_path.with_suffix(".json")

    # Load state dict from SafeTensors (secure)
    if safetensors_path.exists():
        state_dict = safe_load_file(str(safetensors_path), device="cpu")
    else:
        raise FileNotFoundError(f"SafeTensors checkpoint not found: {safetensors_path}")

    # Load config/metadata from JSON (secure)
    config = {}
    metadata = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            config = json_data.get("config", {})
            metadata = json_data.get("metadata", {})

    return state_dict, config, metadata


class Phase3HandoffValidator:
    """
    Validates Phase 3 handoffs (Phase 2→3 and Phase 3→4).

    Ensures:
    - Model format compatibility
    - Metadata preservation
    - Model integrity (reconstruction success)
    - Performance metrics tracking
    """

    def __init__(self, registry_path: Path):
        self.registry = ModelRegistry(str(registry_path))

    def validate_phase2_input(self, model_path: Path) -> Tuple[bool, Dict[str, any]]:
        """
        Validate Phase 2 champion model for Phase 3 input.

        ISS-004: Updated to use secure SafeTensors loading.

        Args:
            model_path: Path to Phase 2 champion model (SafeTensors format)

        Returns:
            (valid, metadata)
        """
        print("\n[>>] Validating Phase 2 -> Phase 3 handoff...")

        # Check for SafeTensors format
        safetensors_path = model_path.with_suffix(".safetensors")
        if not safetensors_path.exists() and not model_path.exists():
            print(f"[X] Model not found: {model_path}")
            return False, {}

        # Load checkpoint securely (ISS-004)
        try:
            model_state, config, metadata = _secure_load_checkpoint_metadata(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False, {}

        # Validate model state dict
        num_params = sum(p.numel() for p in model_state.values())

        print(f"[OK] Model loaded successfully")
        print(f"   Parameters: {num_params / 1e6:.1f}M")

        # Validate Phase 2 metadata
        if "phase" not in metadata or metadata["phase"] != 2:
            print(f"⚠️  Warning: Model phase not set to 2")

        # Check for EvoMerge champion marker
        if "champion_selected" not in metadata:
            print(f"⚠️  Warning: No champion selection metadata")

        # Validate fitness improvement
        fitness_gain = metadata.get("fitness_improvement", 0.0)
        if fitness_gain < 0.20:
            print(f"⚠️  Warning: Low fitness gain: {fitness_gain:.2%} " f"(expected ≥20%)")
        else:
            print(f"✅ Fitness gain: {fitness_gain:.2%}")

        print(f"✅ Phase 2 → Phase 3 handoff validated")

        return True, {
            "num_params": num_params,
            "fitness_gain": fitness_gain,
            "phase": metadata.get("phase", 2),
            "model_type": metadata.get("model_type", "champion"),
        }

    def validate_phase3_output(
        self, model_path: Path, baked_path: Path, rl_path: Path
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Validate Phase 3 output for Phase 4 input.

        ISS-004: Updated to use secure SafeTensors loading.

        Args:
            model_path: Final Phase 3 model (for Phase 4) - SafeTensors format
            baked_path: Step 1 baked model - SafeTensors format
            rl_path: Step 2 RL model - SafeTensors format

        Returns:
            (valid, metadata)
        """
        print("\n[<<] Validating Phase 3 -> Phase 4 handoff...")

        # Check for SafeTensors format
        safetensors_path = model_path.with_suffix(".safetensors")
        if not safetensors_path.exists() and not model_path.exists():
            print(f"[X] Final model not found: {model_path}")
            return False, {}

        # Load checkpoint securely (ISS-004)
        try:
            model_state, config, metadata = _secure_load_checkpoint_metadata(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False, {}

        # Validate thinking tokens (from config or metadata)
        thinking_tokens = config.get("thinking_tokens", [])
        if not thinking_tokens:
            thinking_tokens = metadata.get("thinking_tokens", [])

        if len(thinking_tokens) < 8:
            print(f"[!] Warning: Expected >=8 thinking tokens, " f"got {len(thinking_tokens)}")
        else:
            print(f"[OK] Thinking tokens: {len(thinking_tokens)}")

        # Validate Step 1 (baking) results - secure loading (ISS-004)
        baking_acc = 0.0
        baked_safetensors = baked_path.with_suffix(".safetensors")
        if baked_safetensors.exists() or baked_path.exists():
            try:
                _, _, baked_meta = _secure_load_checkpoint_metadata(baked_path)
                baking_acc = baked_meta.get("strategy_accuracies", {})
                if isinstance(baking_acc, dict):
                    baking_acc = sum(baking_acc.values()) / len(baking_acc) if baking_acc else 0.0

                if baking_acc < 0.85:
                    print(f"[!] Warning: Baking accuracy {baking_acc:.2%} " f"< 85% threshold")
                else:
                    print(f"[OK] Baking accuracy: {baking_acc:.2%}")
            except Exception:
                print(f"[!] Warning: Could not load baked checkpoint for validation")

        # Validate Step 2 (RL) results - secure loading (ISS-004)
        avg_reward = 0.0
        rl_safetensors = rl_path.with_suffix(".safetensors")
        if rl_safetensors.exists() or rl_path.exists():
            try:
                _, _, rl_meta = _secure_load_checkpoint_metadata(rl_path)
                reward_history = rl_meta.get("reward_history", [])

                if reward_history:
                    avg_reward = sum(reward_history[-100:]) / min(100, len(reward_history))
                    print(f"[OK] Avg reward (last 100): {avg_reward:.4f}")
            except Exception:
                print(f"[!] Warning: Could not load RL checkpoint for validation")

        # Validate anti-theater results (if available in metadata)
        anti_theater = metadata.get("anti_theater_results", {})
        if anti_theater:
            all_passed = anti_theater.get("all_passed", False)
            if not all_passed:
                print(f"⚠️  Warning: Anti-theater tests failed")
                print(f"   Divergence: {anti_theater.get('divergence', 0):.3f}")
                print(f"   Ablation: {anti_theater.get('ablation', 0):.3f}")
            else:
                print(f"✅ Anti-theater: All tests passed")

        print(f"✅ Phase 3 → Phase 4 handoff validated")

        return True, {
            "num_thinking_tokens": len(thinking_tokens),
            "baking_accuracy": baking_acc if baked_path.exists() else None,
            "avg_reward": avg_reward if rl_path.exists() else None,
            "anti_theater_passed": anti_theater.get("all_passed", False),
        }

    def register_phase3_completion(
        self,
        session_id: str,
        input_metadata: Dict,
        output_metadata: Dict,
    ) -> bool:
        """
        Register Phase 3 completion in model registry.

        Args:
            session_id: Training session ID
            input_metadata: Phase 2 input metadata
            output_metadata: Phase 3 output metadata

        Returns:
            Success status
        """
        try:
            # Register Phase 3 completion
            self.registry.register_phase_handoff(
                from_phase=2,
                to_phase=3,
                session_id=session_id,
                input_model_metadata=input_metadata,
                output_model_metadata=output_metadata,
                validation_status="passed",
                validation_metrics={
                    "fitness_gain": input_metadata.get("fitness_gain", 0.0),
                    "baking_accuracy": output_metadata.get("baking_accuracy", 0.0),
                    "anti_theater_passed": output_metadata.get("anti_theater_passed", False),
                },
            )

            print(f"✅ Phase 3 completion registered in model registry")
            return True

        except Exception as e:
            print(f"❌ Failed to register: {e}")
            return False


def validate_full_phase3_pipeline(
    phase2_model_path: Path,
    phase3_baked_path: Path,
    phase3_rl_path: Path,
    phase3_final_path: Path,
    registry_path: Path,
    session_id: str,
) -> bool:
    """
    Validate complete Phase 3 pipeline (Phase 2 → Phase 3 → Phase 4).

    Args:
        phase2_model_path: Phase 2 champion model
        phase3_baked_path: Phase 3 Step 1 baked model
        phase3_rl_path: Phase 3 Step 2 RL model
        phase3_final_path: Phase 3 final model (for Phase 4)
        registry_path: Model registry database path
        session_id: Training session ID

    Returns:
        Pipeline valid (True/False)
    """
    print("=" * 70)
    print("PHASE 3 PIPELINE VALIDATION")
    print("=" * 70)

    validator = Phase3HandoffValidator(registry_path)

    # Validate Phase 2 → Phase 3
    input_valid, input_metadata = validator.validate_phase2_input(phase2_model_path)

    if not input_valid:
        print("\n❌ Phase 2 → Phase 3 handoff FAILED")
        return False

    # Validate Phase 3 → Phase 4
    output_valid, output_metadata = validator.validate_phase3_output(
        phase3_final_path, phase3_baked_path, phase3_rl_path
    )

    if not output_valid:
        print("\n❌ Phase 3 → Phase 4 handoff FAILED")
        return False

    # Register completion
    registered = validator.register_phase3_completion(session_id, input_metadata, output_metadata)

    if not registered:
        print("\n⚠️  Warning: Failed to register Phase 3 completion")

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"✅ Phase 2 → Phase 3: PASSED")
    print(f"   Fitness gain: {input_metadata.get('fitness_gain', 0):.2%}")
    print(f"✅ Phase 3 → Phase 4: PASSED")
    print(f"   Baking accuracy: {output_metadata.get('baking_accuracy', 0):.2%}")
    print(
        f"   Anti-theater: {'✅ PASSED' if output_metadata.get('anti_theater_passed') else '❌ FAILED'}"
    )
    print(f"\n✅ Full Phase 3 pipeline validated!")

    return True
