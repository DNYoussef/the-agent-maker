#!/usr/bin/env python3
"""
AGM-007: Full 8-phase pipeline CLI entry point.

Usage:
    python scripts/run_pipeline.py                    # Run all 8 phases
    python scripts/run_pipeline.py --phase 1         # Run only Phase 1
    python scripts/run_pipeline.py --phase 3 --from-checkpoint  # Resume from Phase 3
    python scripts/run_pipeline.py --mock            # Run with mock models (for testing)
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cross_phase.orchestrator.pipeline import PipelineOrchestrator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent Forge V2 - 8-Phase Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline (all 8 phases)
    python scripts/run_pipeline.py

    # Run single phase
    python scripts/run_pipeline.py --phase 3

    # Run with mock models for testing
    python scripts/run_pipeline.py --mock

    # Run phases 3-8 (resume from checkpoint)
    python scripts/run_pipeline.py --start-phase 3

    # Run with custom config
    python scripts/run_pipeline.py --config configs/production.yaml
        """,
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=range(1, 9),
        help="Run only a specific phase (1-8)",
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        choices=range(1, 9),
        default=1,
        help="Start from a specific phase (default: 1)",
    )
    parser.add_argument(
        "--end-phase",
        type=int,
        choices=range(1, 9),
        default=8,
        help="End at a specific phase (default: 8)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock models for testing",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="offline",
        help="W&B logging mode (default: offline)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        help="Resume with existing session ID",
    )

    return parser.parse_args()


def load_config(config_path: str = None) -> dict:
    """Load configuration from file or use defaults."""
    import yaml

    default_config = {
        "mock_mode": False,
        "wandb": {
            "enabled": True,
            "project": "agent-forge-v2",
            "mode": "offline",
        },
        "registry": {
            "db_path": "./storage/registry/model_registry.db",
        },
        "phases": {
            "phase1": {
                "num_cognates": 4,
                "base_model": "gpt2",
            },
            "phase2": {
                "population_size": 8,
                "generations": 5,
            },
            "phase3": {
                "enable_full_rl": False,
                "baking_epochs": 3,
            },
            "phase4": {
                "quantize": True,
            },
            "phase5": {
                "curriculum_stages": 3,
            },
            "phase6": {
                "embed_prompt": True,
            },
            "phase7": {
                "num_experts": 4,
            },
            "phase8": {
                "compression_target": 0.5,
            },
        },
    }

    if config_path:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
            # Merge configs (user overrides default)
            for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value

    return default_config


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    config["mock_mode"] = args.mock
    config["wandb"]["mode"] = args.wandb_mode

    print("=" * 60)
    print("AGENT FORGE V2 - 8-PHASE PIPELINE")
    print("=" * 60)
    print(f"Mock Mode: {config['mock_mode']}")
    print(f"W&B Mode: {config['wandb']['mode']}")
    print()

    # Create orchestrator
    with PipelineOrchestrator(config, session_id=args.session_id) as orchestrator:
        if args.phase:
            # Run single phase
            print(f"Running Phase {args.phase} only...")
            result = orchestrator.run_single_phase(args.phase)
            print(f"\nPhase {args.phase} Result: {'SUCCESS' if result.success else 'FAILED'}")
            print(f"Metrics: {result.metrics}")
        else:
            # Run full pipeline (or range)
            if args.start_phase == 1 and args.end_phase == 8:
                # Full pipeline
                results = orchestrator.run_full_pipeline()
            else:
                # Custom range
                results = {}
                current_models = None
                for phase_num in range(args.start_phase, args.end_phase + 1):
                    print(f"\nRunning Phase {phase_num}...")
                    result = orchestrator.run_single_phase(phase_num, current_models)
                    results[f"phase{phase_num}"] = {
                        "success": result.success,
                        "metrics": result.metrics,
                    }
                    if isinstance(result.model, list):
                        current_models = result.model
                    else:
                        current_models = [result.model] if result.model else None

            # Summary
            print("\n" + "=" * 60)
            print("PIPELINE SUMMARY")
            print("=" * 60)
            for phase_name, phase_result in results.items():
                status = "OK" if phase_result.get("success", True) else "FAILED"
                print(f"  {phase_name}: {status}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
