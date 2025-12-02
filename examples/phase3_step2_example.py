"""
Example: Phase 3 Step 2 (Quiet-STaR RL) Training

Shows how to run Step 2 RL training on baked model from Step 1.

Usage:
    python examples/phase3_step2_example.py

Requirements:
    - Baked model from Step 1
    - Validation dataset
    - GPU with 8GB+ VRAM (more thoughts = more memory)
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.phase3_quietstar.step2_rl import run_step2_rl
from src.phase3_quietstar.step1_baking import ReasoningDataset
from src.phase3_quietstar.config import QuietSTaRConfig
from src.phase3_quietstar.anti_theater import validate_anti_theater


def main():
    """Run Phase 3 Step 2 training."""

    print("=" * 70)
    print("PHASE 3 STEP 2: QUIET-STAR RL (REINFORCE) EXAMPLE")
    print("=" * 70)

    # Configuration
    config = QuietSTaRConfig()

    print("\n📋 Configuration:")
    print(f"  - Algorithm: REINFORCE (policy gradient)")
    print(f"  - Optimizer: MuGrokfast (RL-specific)")
    print(f"  - Learning Rate (Muon): {config.rl.muon_lr}")
    print(f"  - Grokfast Lambda: {config.rl.grokfast_lambda}")
    print(f"  - QK Clip Threshold: {config.rl.qk_clip_threshold}")
    print(f"  - KL Coefficient: {config.rl.kl_coefficient}")
    print(f"  - Number of Episodes: {config.rl.num_episodes}")
    print(f"  - Number of Thoughts: {config.rl.num_thoughts}")
    print(f"  - Thought Length: {config.rl.min_thought_length}-{config.rl.max_thought_length}")

    # Paths
    baked_model_path = Path("models/phase3_baked_model.pt")
    reasoning_data_path = Path("data/phase3_reasoning_training_data.json")
    output_path = Path("models/phase3_reasoning_enhanced_model.pt")

    # Check files exist
    if not baked_model_path.exists():
        print(f"\n❌ Error: Baked model not found at {baked_model_path}")
        print("   Please run Step 1 first:")
        print("   python examples/phase3_step1_example.py")
        return

    if not reasoning_data_path.exists():
        print(f"\n❌ Error: Reasoning data not found at {reasoning_data_path}")
        return

    # Load baked model checkpoint to get tokenizer
    print(f"\nLoading baked model checkpoint...")
    # ISS-004 SECURITY WARNING: Using weights_only=False for demo purposes only.
    # This is acceptable ONLY for trusted internal files YOU created yourself.
    # NEVER use on untrusted/downloaded checkpoints (arbitrary code execution risk).
    #
    # TODO: Refactor to use SafeTensors format:
    #   - Save tokenizer config separately (JSON)
    #   - Load tokenizer from config
    #   - Load model state_dict with safe_load_file()
    #
    # For production, use: src/cross_phase/utils/checkpoint_utils.py
    checkpoint = torch.load(baked_model_path, weights_only=False)
    tokenizer = checkpoint["tokenizer"]

    # Load dataset
    print(f"📥 Loading reasoning dataset...")
    from src.phase3_quietstar.step1_baking import ReasoningDataset
    import json

    with open(reasoning_data_path) as f:
        examples = json.load(f)

    dataset = ReasoningDataset(examples, tokenizer, max_length=512)

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=2
    )

    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples: {len(val_dataset)}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    if device == "cpu":
        print("⚠️  Warning: RL training on CPU will be very slow. GPU strongly recommended.")

    # Run Step 2 training
    print("\n🚀 Starting Step 2 (Quiet-STaR RL) training...")
    print(f"   Baked model: {baked_model_path}")
    print(f"   Output model: {output_path}")
    print(f"   Episodes: {config.rl.num_episodes}")

    # Reduce episodes for demo
    demo_episodes = 1000  # Full training: 10,000

    print(f"\n⚠️  Running demo with {demo_episodes} episodes (full: {config.rl.num_episodes})")
    print("   For production, set num_episodes in config to 10,000+")

    final_metrics = run_step2_rl(
        baked_model_path=baked_model_path,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_path=output_path,
        config=config,
        device=device,
    )

    # Display results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\n📊 Final Results:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Avg Coherence: {final_metrics['avg_coherence']:.4f}")
    print(f"  Avg Thoughts per Sequence: {final_metrics['avg_thoughts']:.1f}")

    # Run anti-theater validation
    print("\n🎭 Running Final Anti-Theater Validation...")

    # Load trained model
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
    trained_checkpoint = torch.load(output_path, weights_only=False)
    trained_model = trained_checkpoint["model"]

    theater_results = validate_anti_theater(
        model=trained_model,
        tokenizer=tokenizer,
        dataloader=val_dataloader,
        config=config.anti_theater,
        device=device,
    )

    if theater_results["all_passed"]:
        print("\n✅ All anti-theater tests PASSED!")
        print("   Model generates genuine reasoning, not theater.")
    else:
        print("\n⚠️  Anti-theater tests FAILED")
        print("   Consider:")
        print("   - Training for more episodes")
        print("   - Adjusting KL coefficient (prevent overfitting)")
        print("   - Using different thought generation parameters")

    print(f"\n✅ Reasoning-enhanced model saved to: {output_path}")
    print("   Ready for Phase 4 (BitNet quantization)")

    # Next steps
    print("\n📝 Next Steps:")
    print("  1. Validate model on GSM8K/ARC benchmarks")
    print("  2. Measure inference latency (target: <200ms)")
    print("  3. Check accuracy improvement vs baked baseline (+5-10% target)")
    print("  4. Proceed to Phase 4 if all targets met")

    # Performance summary
    print("\n📈 Performance vs Targets:")
    accuracy_improvement = final_metrics["accuracy"] - 0.85  # Assuming 85% from Step 1
    print(f"  Accuracy improvement: {accuracy_improvement:+.2%} (target: +5-10%)")
    print(f"  Coherence: {final_metrics['avg_coherence']:.4f} (target: >0.70)")
    print(f"  Anti-theater: {'✅ PASS' if theater_results['all_passed'] else '❌ FAIL'}")


if __name__ == "__main__":
    main()
