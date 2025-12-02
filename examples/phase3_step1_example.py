"""
Example: Phase 3 Step 1 (Prompt Baking) Training

Shows how to run Step 1 training on Phase 2 champion model.

Usage:
    python examples/phase3_step1_example.py

Requirements:
    - Phase 2 champion model (from EvoMerge)
    - 20K reasoning examples (from data_generator.py)
    - GPU with 6GB+ VRAM
"""

import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

from src.phase3_quietstar.step1_baking import run_step1_baking
from src.phase3_quietstar.config import QuietSTaRConfig


def main():
    """Run Phase 3 Step 1 training."""

    print("=" * 70)
    print("PHASE 3 STEP 1: PROMPT BAKING EXAMPLE")
    print("=" * 70)

    # Configuration
    config = QuietSTaRConfig()

    print("\n📋 Configuration:")
    print(f"  - Optimizer: MuGrokfast")
    print(f"  - Learning Rate (Muon): {config.baking.muon_lr}")
    print(f"  - Grokfast Lambda: {config.baking.grokfast_lambda}")
    print(f"  - QK Clip Threshold: {config.baking.qk_clip_threshold}")
    print(f"  - Number of Epochs: {config.baking.num_epochs}")
    print(f"  - Batch Size: {config.baking.batch_size}")
    print(f"  - Convergence Threshold: {config.baking.convergence_threshold:.2%}")

    # Paths
    phase2_model_path = Path("models/phase2_champion_model.pt")
    reasoning_data_path = Path("data/phase3_reasoning_training_data.json")
    output_path = Path("models/phase3_baked_model.pt")

    # Check files exist
    if not reasoning_data_path.exists():
        print(f"\n❌ Error: Reasoning data not found at {reasoning_data_path}")
        print("   Please run data generation first:")
        print("   python src/phase3_quietstar/data_generator.py <api_key>")
        return

    # Load Phase 2 champion model
    print("\n📥 Loading Phase 2 champion model...")

    if phase2_model_path.exists():
        # Load from checkpoint
        # ISS-004 SECURITY WARNING: Using weights_only=False for demo purposes only.
        # This is acceptable ONLY for trusted internal files YOU created yourself.
        # NEVER use on untrusted/downloaded checkpoints (arbitrary code execution risk).
        #
        # TODO: Refactor to use SafeTensors format:
        #   - Load model architecture separately
        #   - Load state_dict with safe_load_file()
        #   - model.load_state_dict(state_dict)
        #
        # For production, use: src/cross_phase/utils/checkpoint_utils.py
        checkpoint = torch.load(phase2_model_path, weights_only=False)
        model = checkpoint["model"]
        tokenizer = checkpoint["tokenizer"]
        print(f"✅ Loaded model from {phase2_model_path}")
    else:
        # Use base model for demo
        print("⚠️  Phase 2 model not found, using base model for demo")
        model_name = "gpt2"  # Small model for demo
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Display model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params / 1e6:.1f}M")

    # Run Step 1 training
    print("\n🚀 Starting Step 1 (Prompt Baking) training...")
    print(f"   Training data: {reasoning_data_path}")
    print(f"   Output model: {output_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    if device == "cpu":
        print("⚠️  Warning: Training on CPU will be slow. GPU recommended.")

    # Run training
    final_metrics = run_step1_baking(
        model=model,
        tokenizer=tokenizer,
        data_path=reasoning_data_path,
        output_path=output_path,
        config=config,
        device=device,
    )

    # Display results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\n📊 Final Results:")
    print(f"  Overall Accuracy: {final_metrics['final_accuracy']:.4f}")
    print(f"  Converged: {'✅ Yes' if final_metrics['converged'] else '❌ No'}")

    print("\n  Strategy Accuracies:")
    for strategy, acc in final_metrics["strategy_accuracies"].items():
        emoji = "✅" if acc >= config.baking.convergence_threshold * 0.9 else "⚠️"
        print(f"    {emoji} {strategy}: {acc:.4f}")

    print("\n  Thinking Token Usage:")
    for token_type, usage in final_metrics["token_usage"].items():
        print(f"    - {token_type}: {usage:.2%}")

    print(f"\n✅ Baked model saved to: {output_path}")
    print("   Ready for Step 2 (Quiet-STaR RL training)")

    # Next steps
    print("\n📝 Next Steps:")
    print("  1. Validate baked model on test set")
    print("  2. Check thinking token usage (target: >80% thinking tags)")
    print("  3. Proceed to Step 2 (RL training) if converged")

    if not final_metrics["converged"]:
        print("\n⚠️  Convergence not achieved. Consider:")
        print("   - Increasing num_epochs (current: 5)")
        print("   - Adjusting learning rate")
        print("   - Checking data quality")


if __name__ == "__main__":
    main()
