# Phase 5: Dream Consolidation Implementation Guide

**Based on**: "Dreaming is All You Need" (Ni & Liu, 2024)
**Paper Location**: [phases/phase5/DREAMING IS ALL YOU NEED.pdf](../../phases/phase5/DREAMING%20IS%20ALL%20YOU%20NEED.pdf)
**Status**: Implementation-ready specification

---

## Executive Summary

Dream consolidation is **Stage 6** of Phase 5's 7-stage curriculum pipeline. It prevents catastrophic forgetting across 10 curriculum levels by using full autoencoder reconstruction (encoder + decoder) to consolidate learned patterns into model weights.

**Key Parameters** (from paper analysis):
- **Frequency**: After each level (10 times total)
- **Epochs per consolidation**: 3 (DreamNet-3 optimal)
- **Duration per level**: 30-60 minutes
- **Total dream time**: 5-10 hours (4-8% of 120-240 hour training)
- **Temperature**: 1.2 (high-temp creative replay)
- **Architecture**: Full autoencoder (frozen weights)

---

## Paper Summary

### Key Findings from "Dreaming is All You Need"

**Architecture** (Figures 4-5, Pages 4-5):
1. **SleepNet**: Uses encoder-only from autoencoder (simpler)
2. **DreamNet**: Uses **full autoencoder** (encoder + decoder) - **Better performance**
3. **Best configuration**: 3-4 dream blocks (DreamNet-3, DreamNet-4)
4. **Frozen weights**: Autoencoder weights frozen during training (93% vs 86% unfrozen)

**Results** (Table 3, Page 10):
- DreamNet-3: **93.4% CIFAR100**, **89.6% ImageNet-tiny**
- SleepNet-3: 92.2% CIFAR100, 88.1% ImageNet-tiny
- **DreamNet > SleepNet** due to full reconstruction

**Training Protocol** (Section 4.1, Page 8):
- 30 epochs total
- ADAM optimizer (lr=0.005, β1=0.9, β2=0.999)
- Dream blocks **interspersed** in network architecture

---

## Agent Forge V2 Adaptation

### Differences from Paper

| Aspect | Paper (DreamNet) | Agent Forge V2 |
|--------|------------------|----------------|
| **Domain** | Image classification (CIFAR100) | NLP + code generation (25M params) |
| **Training** | 30 epochs continuous | 10 curriculum levels (12-24 hrs each) |
| **Dream timing** | Interspersed in forward pass | **After level completion** (consolidation) |
| **Purpose** | Feature enhancement | **Memory consolidation** + forgetting prevention |
| **Autoencoder** | Pre-trained MAE (visual) | Pre-trained BART/T5 (text) |

### Why Consolidation (Not Interspersed)?

**Agent Forge uses consolidation** (not interspersed dreams) because:

1. **Curriculum structure**: 10 discrete levels vs continuous training
2. **Catastrophic forgetting risk**: Level 10 would forget Level 1 without consolidation
3. **Long training duration**: 120-240 hours requires periodic memory reinforcement
4. **Biological analogy**: Sleep/dreams occur **after** learning episodes, not during

---

## Implementation Architecture

### Overview

```python
Phase 5: Curriculum Learning
    ↓
Level 1 Training (12-24 hrs)
    ↓
Dream Consolidation (30-60 min)  ← This document
    ↓
Level 2 Training (12-24 hrs)
    ↓
Dream Consolidation (30-60 min)
    ↓
... (repeat for levels 3-10)
    ↓
Phase 6: Tool & Persona Baking
```

### Dream Consolidation Class

```python
# phases/phase5/dream_consolidation.py

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import wandb

class DreamConsolidation:
    """
    Implements dream-based memory consolidation after each curriculum level.

    Based on "Dreaming is All You Need" (Ni & Liu, 2024), adapted for
    NLP curriculum learning with catastrophic forgetting prevention.

    Key insight: Full autoencoder (encoder + decoder) outperforms encoder-only
    for feature consolidation (93.4% vs 92.2% accuracy in paper).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        autoencoder_name: str = "facebook/bart-base",  # Or "t5-base"
        device: str = "cuda",
        config: dict = None
    ):
        """
        Initialize dream consolidation system.

        Args:
            model: The Phase 5 curriculum model to consolidate
            autoencoder_name: Pre-trained autoencoder for reconstruction
            device: Training device (cuda/cpu)
            config: Dream configuration parameters
        """
        self.model = model
        self.device = device

        # Load pre-trained autoencoder (frozen weights - paper finding)
        print(f"Loading pre-trained autoencoder: {autoencoder_name}")
        self.autoencoder = AutoModel.from_pretrained(autoencoder_name)
        self.autoencoder.to(device)
        self.autoencoder.eval()  # Freeze weights (93% vs 86% unfrozen)

        for param in self.autoencoder.parameters():
            param.requires_grad = False  # Explicit freeze

        # Dream configuration (from paper analysis)
        self.config = config or {
            "num_epochs": 3,              # DreamNet-3 from paper
            "temperature": 1.2,           # High-temp creative replay
            "consolidation_weight": 0.1,  # Loss weighting (10% of total)
            "learning_rate": 1e-5,        # Low LR for fine-tuning
            "batch_size": 32,             # Match training batch size
        }

        print(f"Dream consolidation initialized:")
        print(f"  - Epochs per consolidation: {self.config['num_epochs']}")
        print(f"  - Temperature: {self.config['temperature']}")
        print(f"  - Consolidation weight: {self.config['consolidation_weight']}")

    def dream_epoch(
        self,
        level: int,
        level_data: torch.utils.data.DataLoader,
        previous_levels_data: list = None,
        epoch: int = 0
    ) -> dict:
        """
        Single dream consolidation epoch.

        Process:
        1. Forward pass through model (current hidden states)
        2. Autoencoder reconstruction (dream states)
        3. Consolidation loss (align current with dream)
        4. Supervised loss (maintain performance)
        5. Combined update

        Args:
            level: Current curriculum level (1-10)
            level_data: Training data from current level
            previous_levels_data: Sample data from levels 1..level-1
            epoch: Current dream epoch (0-2 for 3 epochs total)

        Returns:
            dict: Epoch metrics
        """
        self.model.train()

        epoch_metrics = {
            "consolidation_losses": [],
            "supervised_losses": [],
            "total_losses": [],
            "reconstruction_mse": [],
        }

        # Replay current level + previous levels (prevent forgetting)
        all_data_loaders = [level_data]
        if previous_levels_data:
            all_data_loaders.extend(previous_levels_data)

        for data_idx, data_loader in enumerate(all_data_loaders):
            is_current_level = (data_idx == 0)

            for batch_idx, batch in enumerate(data_loader):
                inputs = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # 1. Forward pass through model (current hidden states)
                outputs = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True
                )
                current_hidden = outputs.hidden_states[-1]  # Last layer
                supervised_loss = outputs.loss

                # 2. Autoencoder reconstruction (dream states)
                with torch.no_grad():
                    # Encode then decode (full autoencoder reconstruction)
                    encoder_outputs = self.autoencoder.encoder(
                        inputs,
                        attention_mask=attention_mask
                    )
                    encoded_hidden = encoder_outputs.last_hidden_state

                    # Decode (reconstruction)
                    decoder_outputs = self.autoencoder.decoder(
                        encoder_hidden_states=encoded_hidden,
                        attention_mask=attention_mask
                    )
                    reconstructed_hidden = decoder_outputs.last_hidden_state

                # 3. Consolidation loss (align model with dream reconstruction)
                consolidation_loss = F.mse_loss(
                    current_hidden,
                    reconstructed_hidden
                )

                # 4. Reconstruction quality (for monitoring)
                reconstruction_mse = F.mse_loss(
                    encoded_hidden,
                    current_hidden
                ).item()

                # 5. Combined loss
                total_loss = (
                    supervised_loss +
                    self.config["consolidation_weight"] * consolidation_loss
                )

                # 6. Update
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Track metrics
                epoch_metrics["consolidation_losses"].append(consolidation_loss.item())
                epoch_metrics["supervised_losses"].append(supervised_loss.item())
                epoch_metrics["total_losses"].append(total_loss.item())
                epoch_metrics["reconstruction_mse"].append(reconstruction_mse)

        # Average metrics
        return {
            "consolidation_loss": torch.tensor(epoch_metrics["consolidation_losses"]).mean().item(),
            "supervised_loss": torch.tensor(epoch_metrics["supervised_losses"]).mean().item(),
            "total_loss": torch.tensor(epoch_metrics["total_losses"]).mean().item(),
            "reconstruction_mse": torch.tensor(epoch_metrics["reconstruction_mse"]).mean().item(),
        }

    def test_memory_retention(
        self,
        previous_levels_data: list,
        level: int
    ) -> dict:
        """
        Test accuracy on previous levels (catastrophic forgetting check).

        Args:
            previous_levels_data: Data from all previous levels
            level: Current level number

        Returns:
            dict: Retention metrics per previous level
        """
        self.model.eval()

        retention_metrics = {}

        with torch.no_grad():
            for prev_level_idx, data_loader in enumerate(previous_levels_data):
                prev_level = prev_level_idx + 1  # Levels are 1-indexed

                correct = 0
                total = 0

                for batch in data_loader:
                    inputs = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    outputs = self.model(inputs, attention_mask=attention_mask)
                    predictions = outputs.logits.argmax(dim=-1)

                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                accuracy = correct / total if total > 0 else 0.0
                retention_metrics[f"level_{prev_level}_retention"] = accuracy

        self.model.train()

        # Calculate average retention
        if retention_metrics:
            avg_retention = sum(retention_metrics.values()) / len(retention_metrics)
            retention_metrics["average_retention"] = avg_retention

        return retention_metrics

    def consolidate(
        self,
        level: int,
        level_data: torch.utils.data.DataLoader,
        previous_levels_data: list = None,
        wandb_logger = None
    ) -> dict:
        """
        Full dream consolidation after level completion.

        Performs 3 epochs of dream consolidation (DreamNet-3 configuration)
        with memory retention testing after each epoch.

        Args:
            level: Current curriculum level (1-10)
            level_data: Training data from current level
            previous_levels_data: Sample data from all previous levels
            wandb_logger: W&B run for logging (optional)

        Returns:
            dict: Complete consolidation metrics
        """
        import time
        start_time = time.time()

        print(f"\n{'='*70}")
        print(f"Level {level}: Dream Consolidation Starting")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - Epochs: {self.config['num_epochs']}")
        print(f"  - Temperature: {self.config['temperature']}")
        print(f"  - Learning rate: {self.config['learning_rate']}")
        print(f"  - Consolidation weight: {self.config['consolidation_weight']}")

        # Optimizer for consolidation (low LR)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            betas=(0.9, 0.999)
        )

        consolidation_metrics = {
            "level": level,
            "epochs": [],
            "epoch_metrics": [],
            "retention_metrics": [],
        }

        # Perform dream consolidation epochs
        for epoch in range(self.config["num_epochs"]):
            print(f"\n  Epoch {epoch+1}/{self.config['num_epochs']}:")

            # Dream consolidation epoch
            epoch_metrics = self.dream_epoch(
                level=level,
                level_data=level_data,
                previous_levels_data=previous_levels_data,
                epoch=epoch
            )

            consolidation_metrics["epochs"].append(epoch)
            consolidation_metrics["epoch_metrics"].append(epoch_metrics)

            # Test memory retention on previous levels
            retention_metrics = {}
            if previous_levels_data:
                retention_metrics = self.test_memory_retention(
                    previous_levels_data,
                    level
                )
                consolidation_metrics["retention_metrics"].append(retention_metrics)

            # Display metrics
            print(f"    Consolidation loss: {epoch_metrics['consolidation_loss']:.4f}")
            print(f"    Supervised loss: {epoch_metrics['supervised_loss']:.4f}")
            print(f"    Reconstruction MSE: {epoch_metrics['reconstruction_mse']:.4f}")

            if retention_metrics:
                avg_retention = retention_metrics.get("average_retention", 0.0)
                print(f"    Memory retention: {avg_retention:.2%}")

            # Log to W&B
            if wandb_logger:
                log_dict = {
                    f"level_{level}/dream_epoch": epoch,
                    f"level_{level}/consolidation_loss": epoch_metrics["consolidation_loss"],
                    f"level_{level}/supervised_loss": epoch_metrics["supervised_loss"],
                    f"level_{level}/reconstruction_mse": epoch_metrics["reconstruction_mse"],
                    f"level_{level}/temperature": self.config["temperature"],
                }

                # Add retention metrics
                for key, value in retention_metrics.items():
                    log_dict[f"level_{level}/{key}"] = value

                wandb_logger.log(log_dict)

        # Calculate final metrics
        elapsed_time = time.time() - start_time

        final_metrics = {
            "level": level,
            "duration_minutes": elapsed_time / 60,
            "num_epochs": self.config["num_epochs"],
            "final_consolidation_loss": consolidation_metrics["epoch_metrics"][-1]["consolidation_loss"],
            "final_reconstruction_mse": consolidation_metrics["epoch_metrics"][-1]["reconstruction_mse"],
        }

        if consolidation_metrics["retention_metrics"]:
            final_metrics["final_average_retention"] = consolidation_metrics["retention_metrics"][-1]["average_retention"]

        print(f"\n{'='*70}")
        print(f"Level {level}: Dream Consolidation Complete")
        print(f"  Duration: {elapsed_time/60:.1f} minutes")
        if "final_average_retention" in final_metrics:
            print(f"  Final retention: {final_metrics['final_average_retention']:.2%}")
        print(f"{'='*70}\n")

        return {
            **final_metrics,
            "detailed_metrics": consolidation_metrics
        }


# Example usage in Phase 5 curriculum training
class CurriculumTraining:
    """Phase 5 curriculum training with dream consolidation."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Initialize dream consolidation
        self.dream_consolidator = DreamConsolidation(
            model=model,
            autoencoder_name=config.get("autoencoder", "facebook/bart-base"),
            config={
                "num_epochs": 3,
                "temperature": 1.2,
                "consolidation_weight": 0.1,
                "learning_rate": 1e-5,
            }
        )

        # Store data from previous levels for replay
        self.previous_levels_data = []

    def train_level(self, level: int, level_data):
        """Train on curriculum level with dream consolidation."""
        print(f"\nLevel {level}: Training...")

        # Stage 3: Standard curriculum training (12-24 hours)
        for epoch in range(self.config["epochs_per_level"]):
            for batch in level_data:
                # ... standard training loop ...
                pass

        print(f"Level {level}: Training complete")

        # Stage 6: DREAM CONSOLIDATION (30-60 minutes)
        consolidation_metrics = self.dream_consolidator.consolidate(
            level=level,
            level_data=level_data,
            previous_levels_data=self.previous_levels_data,
            wandb_logger=wandb
        )

        # Store sample data for future consolidation
        sample_data = self.sample_level_data(level_data, n_samples=1000)
        self.previous_levels_data.append(sample_data)

        return consolidation_metrics
```

---

## W&B Metrics

### Per Dream Epoch (10 levels × 3 epochs = 30 metric sets)

```python
# Logged for each epoch of each level
wandb.log({
    # Epoch identification
    f"level_{level}/dream_epoch": epoch,  # 0, 1, 2

    # Loss metrics
    f"level_{level}/consolidation_loss": consolidation_loss,
    f"level_{level}/supervised_loss": supervised_loss,
    f"level_{level}/total_loss": total_loss,

    # Reconstruction quality
    f"level_{level}/reconstruction_mse": mse,

    # Configuration
    f"level_{level}/temperature": 1.2,
    f"level_{level}/learning_rate": 1e-5,
    f"level_{level}/consolidation_weight": 0.1,

    # Memory retention (if previous levels exist)
    f"level_{level}/level_1_retention": retention_acc_level1,
    f"level_{level}/level_2_retention": retention_acc_level2,
    # ... (for each previous level)
    f"level_{level}/average_retention": avg_retention,

    # Forgetting detection
    f"level_{level}/catastrophic_forgetting_detected": retention < 0.90,
})
```

### Total Metrics

- **30 metric sets** (10 levels × 3 epochs)
- **~10 metrics per set** = **300 total metrics**
- Plus retention metrics per previous level (grows as levels progress)

---

## Timeline

```
Phase 5 Timeline (120-240 hours total):

Level 1: Training (12-24 hrs) → Dream (30-60 min)
Level 2: Training (12-24 hrs) → Dream (30-60 min)
Level 3: Training (12-24 hrs) → Dream (30-60 min)
...
Level 10: Training (12-24 hrs) → Dream (30-60 min)

Total training: 120-240 hours
Total dream consolidation: 5-10 hours (4-8% overhead)
```

---

## Success Criteria

- ✅ **Memory retention ≥ 95%**: Level 10 maintains ≥95% accuracy on Level 1 data
- ✅ **Consolidation loss < 0.05**: High-quality dream reconstructions
- ✅ **Reconstruction MSE < 0.10**: Autoencoder reconstructs features well
- ✅ **Eudaimonia preservation**: Alignment score ≥ 65% after all 10 levels
- ✅ **No catastrophic forgetting**: Accuracy on all previous levels ≥ 90%
- ✅ **Dream efficiency**: Consolidation time ≤ 5% of total training time
- ✅ **Convergence**: Metrics stable across 3 epochs (not degrading)

---

## Troubleshooting

### Issue 1: High Consolidation Loss (>0.10)

**Symptoms**: Consolidation loss doesn't decrease across epochs

**Possible Causes**:
1. Learning rate too high/low
2. Consolidation weight too high
3. Autoencoder mismatch (visual vs text model)

**Solutions**:
```python
# Adjust learning rate
config["learning_rate"] = 5e-6  # Lower LR

# Adjust consolidation weight
config["consolidation_weight"] = 0.05  # Reduce from 0.1

# Try different autoencoder
autoencoder_name = "t5-base"  # Instead of bart-base
```

### Issue 2: Catastrophic Forgetting (retention <90%)

**Symptoms**: Accuracy on previous levels drops below 90%

**Possible Causes**:
1. Not enough dream epochs
2. Previous level data sample too small
3. Consolidation weight too low

**Solutions**:
```python
# Increase dream epochs
config["num_epochs"] = 5  # Instead of 3

# Increase previous level sample size
sample_data = self.sample_level_data(level_data, n_samples=2000)  # Instead of 1000

# Increase consolidation weight
config["consolidation_weight"] = 0.15  # Instead of 0.1
```

### Issue 3: Slow Consolidation (>90 minutes per level)

**Symptoms**: Dream consolidation takes too long

**Possible Causes**:
1. Too many epochs
2. Batch size too small
3. Previous level data too large

**Solutions**:
```python
# Reduce epochs (minimum 2)
config["num_epochs"] = 2

# Increase batch size
config["batch_size"] = 64  # Instead of 32

# Sample less data from previous levels
sample_data = self.sample_level_data(level_data, n_samples=500)  # Instead of 1000
```

---

## Related Documents

- [phases/phase5/PHASE5_LOGICAL_UNDERSTANDING_V2.md](../../phases/phase5/PHASE5_LOGICAL_UNDERSTANDING_V2.md) - Phase 5 overview
- [phases/phase5/PHASE5_EUDAIMONIA_SYSTEM.md](../../phases/phase5/PHASE5_EUDAIMONIA_SYSTEM.md) - Eudaimonia system (Stage 4)
- [phases/phase5/PHASE5_V2_IMPLEMENTATION_SUMMARY.md](../../phases/phase5/PHASE5_V2_IMPLEMENTATION_SUMMARY.md) - Implementation summary
- [phases/phase5/DREAMING IS ALL YOU NEED.pdf](../../phases/phase5/DREAMING%20IS%20ALL%20YOU%20NEED.pdf) - Research paper
- [CLAUDE.md](../../CLAUDE.md) - Master V2 overview

---

**Status**: Ready for implementation
**Paper Basis**: "Dreaming is All You Need" (Ni & Liu, 2024)
**Adaptation**: NLP curriculum learning with catastrophic forgetting prevention
