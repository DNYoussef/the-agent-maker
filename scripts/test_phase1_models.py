"""
Test Phase 1 Trained Models - Interactive Chat

Load the trained reasoning and memory models and test them with prompts.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import GPT2Tokenizer
from safetensors.torch import load_file as safe_load_file
import json

from phase1_cognate.model.model_config import Phase1Config
from phase1_cognate.model.full_model import TRMTitansMAGModel


def load_model(specialization: str, checkpoint_path: str, device: str = "cuda"):
    """Load a trained Phase 1 model from checkpoint"""
    print(f"\nLoading {specialization} model from {checkpoint_path}...")

    # Create config
    config = Phase1Config(specialization=specialization)

    # Create model
    model = TRMTitansMAGModel(config)

    # Load checkpoint from SafeTensors format
    checkpoint_path = Path(checkpoint_path)
    safetensors_path = checkpoint_path.parent / f"{checkpoint_path.stem}.safetensors"
    metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}.json"

    if safetensors_path.exists():
        # Load from SafeTensors (secure format)
        print(f"  Loading from SafeTensors: {safetensors_path.name}")
        state_dict = safe_load_file(str(safetensors_path), device=str(device))

        # Load metadata from JSON
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
    else:
        # Fallback to legacy .pt format (with warning)
        print(f"  WARNING: SafeTensors not found, falling back to legacy .pt format")
        print(f"  Consider converting checkpoints to SafeTensors for security")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint

    # Fix LTM memory_state shape mismatch if needed
    # Checkpoints may have batch_size != 1 from training
    if "backbone.ltm.memory_state" in state_dict:
        memory_state = state_dict["backbone.ltm.memory_state"]
        if memory_state.shape[0] != 1:
            # Take only first item to match inference batch size of 1
            print(f"  Fixing memory_state shape: {memory_state.shape} -> {torch.Size([1, memory_state.shape[1], memory_state.shape[2]])}")
            state_dict["backbone.ltm.memory_state"] = memory_state[:1, :, :]

    model.load_state_dict(state_dict)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"OK - Loaded {specialization} model (epoch {metadata.get('epoch', 'N/A')}, step {metadata.get('global_step', 'N/A')})")
    print(f"     Best validation loss: {metadata.get('best_val_loss', 'N/A')}")

    return model


def generate_text(model, tokenizer, prompt: str, max_length: int = 100, temperature: float = 0.7, device: str = "cuda"):
    """Generate text from a prompt using the model"""

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate
    print(f"\nGenerating response (max {max_length} tokens)...")

    with torch.no_grad():
        generated_ids = input_ids.clone()

        for _ in range(max_length):
            # Forward pass
            outputs = model(generated_ids)
            logits = outputs["logits"]

            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            # Stop if end-of-sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def test_model(model, tokenizer, specialization: str, device: str = "cuda"):
    """Test a model with sample prompts"""

    print(f"\n{'='*70}")
    print(f"TESTING {specialization.upper()} MODEL")
    print(f"{'='*70}\n")

    # Test prompts based on specialization
    if specialization == "reasoning":
        prompts = [
            "Q: If a store has 15 apples and sells 7, how many are left? A:",
            "Q: What is 25 + 17? A:",
            "Q: A train travels 60 miles in 2 hours. What is its speed? A:",
        ]
    elif specialization == "memory":
        prompts = [
            "The capital of France is Paris. The capital of Germany is Berlin. Q: What is the capital of France? A:",
            "John has 5 apples. Mary gives him 3 more apples. Q: How many apples does John have now? A:",
            "The color of the sky is blue. The color of grass is green. Q: What color is the sky? A:",
        ]
    else:  # speed
        prompts = [
            "Q: 2 + 2 = A:",
            "Q: Is the sky blue? A:",
            "Q: What is 10 - 5? A:",
        ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'-'*70}")
        print(f"Test {i}/{len(prompts)}")
        print(f"{'-'*70}")
        print(f"\nPrompt: {prompt}")

        generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, device=device)

        # Extract just the answer (everything after the prompt)
        answer = generated[len(prompt):].strip()

        print(f"\nModel Response:\n{answer}")


def main():
    print("\n" + "="*70)
    print("PHASE 1 MODEL TESTING - INTERACTIVE CHAT")
    print("="*70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load tokenizer
    print(f"\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("OK - Tokenizer loaded")

    # Test reasoning model (use epoch_10.pt - fully trained model)
    checkpoint_path = "checkpoints/phase1/reasoning/epoch_10.pt"
    if Path(checkpoint_path).exists():
        reasoning_model = load_model("reasoning", checkpoint_path, device)
        test_model(reasoning_model, tokenizer, "reasoning", device)

        # Clean up
        del reasoning_model
        torch.cuda.empty_cache()
    else:
        print(f"\nERROR: Reasoning model checkpoint not found: {checkpoint_path}")

    # Test memory model (use epoch_10.pt - fully trained model)
    checkpoint_path = "checkpoints/phase1/memory/epoch_10.pt"
    if Path(checkpoint_path).exists():
        memory_model = load_model("memory", checkpoint_path, device)
        test_model(memory_model, tokenizer, "memory", device)

        # Clean up
        del memory_model
        torch.cuda.empty_cache()
    else:
        print(f"\nERROR: Memory model checkpoint not found: {checkpoint_path}")

    print(f"\n{'='*70}")
    print("TESTING COMPLETE!")
    print(f"{'='*70}\n")

    # Interactive mode
    print("\nINTERACTIVE MODE - Chat with the models!")
    print("Commands:")
    print("  /reasoning <prompt> - Chat with reasoning model")
    print("  /memory <prompt>    - Chat with memory model")
    print("  /quit               - Exit")
    print()

    # Load both models for interactive use
    reasoning_model = None
    memory_model = None

    if Path("checkpoints/phase1/reasoning/epoch_10.pt").exists():
        reasoning_model = load_model("reasoning", "checkpoints/phase1/reasoning/epoch_10.pt", device)

    if Path("checkpoints/phase1/memory/epoch_10.pt").exists():
        memory_model = load_model("memory", "checkpoints/phase1/memory/epoch_10.pt", device)

    # Interactive loop
    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("\nGoodbye!")
                break

            if user_input.startswith("/reasoning "):
                if reasoning_model is None:
                    print("ERROR: Reasoning model not loaded")
                    continue

                prompt = user_input[11:].strip()
                response = generate_text(reasoning_model, tokenizer, prompt, max_length=100, device=device)
                answer = response[len(prompt):].strip()
                print(f"\nReasoning Model:\n{answer}")

            elif user_input.startswith("/memory "):
                if memory_model is None:
                    print("ERROR: Memory model not loaded")
                    continue

                prompt = user_input[8:].strip()
                response = generate_text(memory_model, tokenizer, prompt, max_length=100, device=device)
                answer = response[len(prompt):].strip()
                print(f"\nMemory Model:\n{answer}")

            else:
                print("ERROR: Unknown command. Use /reasoning <prompt> or /memory <prompt>")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
