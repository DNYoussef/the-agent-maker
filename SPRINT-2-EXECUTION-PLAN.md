# Sprint 2 Execution Plan - Remaining Remediation

**Date**: 2025-11-26
**Reference**: REMEDIATION-SUMMARY-2025-11-26.md (Recommendations for Next Sprint)
**Status**: PLANNING

---

## Executive Summary

This plan addresses 4 remaining issues from the previous remediation:

| Issue | Priority | Effort | Dependencies |
|-------|----------|--------|--------------|
| E2E Tests | HIGH | Large (40+ hours) | None |
| Safetensors Migration | HIGH | Medium (8-12 hours) | None |
| MockTokenizer Cleanup | LOW | Small (2 hours) | None |
| Full RL Training (ISS-007) | MEDIUM | Large (20+ hours) | None |
| STE Fine-Tuning (ISS-008) | MEDIUM | Large (16+ hours) | None |

**Total Estimated Effort**: 86-100+ hours

---

## Issue 1: Add E2E Tests for All 8 Phases

### Current State
- **0 E2E tests exist** (tests/e2e/ directory is empty or missing)
- Integration tests exist for Phases 3-4 only
- Unit tests now exist for all phases (after Sprint 1)

### Scope
Create end-to-end tests validating the complete pipeline:

| Phase | E2E Test File | Key Validations |
|-------|---------------|-----------------|
| Phase 1 | test_e2e_phase1_cognate.py | Model creation, 25M params, training loop |
| Phase 2 | test_e2e_phase2_evomerge.py | Merge techniques, fitness evaluation |
| Phase 3 | test_e2e_phase3_quietstar.py | Baking + RL training pipeline |
| Phase 4 | test_e2e_phase4_bitnet.py | Quantization, 8.2x compression |
| Phase 5 | test_e2e_phase5_curriculum.py | Curriculum learning, level progression |
| Phase 6 | test_e2e_phase6_baking.py | A/B cycle optimization |
| Phase 7 | test_e2e_phase7_experts.py | Expert discovery, SVF training |
| Phase 8 | test_e2e_phase8_compression.py | 280x compression pipeline |
| Cross-Phase | test_e2e_pipeline.py | Phase handoff validation |

### Implementation Approach

```
tests/
  e2e/
    __init__.py
    conftest.py                    # Shared fixtures, mock models
    test_e2e_phase1_cognate.py     # ~150 lines
    test_e2e_phase2_evomerge.py    # ~150 lines
    test_e2e_phase3_quietstar.py   # ~200 lines (baking + RL)
    test_e2e_phase4_bitnet.py      # ~150 lines
    test_e2e_phase5_curriculum.py  # ~150 lines
    test_e2e_phase6_baking.py      # ~150 lines
    test_e2e_phase7_experts.py     # ~150 lines
    test_e2e_phase8_compression.py # ~150 lines
    test_e2e_pipeline.py           # ~200 lines (cross-phase)
```

### E2E Test Template

```python
"""
E2E Test for Phase N
Validates complete pipeline from input to output
"""
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

# Import phase components
from phaseN_module import PhaseNEngine, PhaseNConfig, PhaseNResult

class TestPhaseNE2E:
    """End-to-end tests for Phase N pipeline."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary directory for test outputs."""
        return tmp_path / "phase_n_output"

    @pytest.fixture
    def mock_input_model(self):
        """Create mock model from previous phase."""
        model = Mock()
        model.parameters.return_value = [torch.randn(100, 100)]
        return model

    def test_full_pipeline_execution(self, mock_input_model, temp_output_dir):
        """Test complete Phase N pipeline runs without error."""
        config = PhaseNConfig(
            # Use minimal settings for fast E2E test
            epochs=1,
            batch_size=2,
        )

        engine = PhaseNEngine(config=config)
        result = engine.run(mock_input_model, temp_output_dir)

        assert isinstance(result, PhaseNResult)
        assert result.success is True
        assert result.error is None

    def test_output_artifacts_created(self, mock_input_model, temp_output_dir):
        """Test expected output files are created."""
        # ... implementation

    def test_handoff_to_next_phase(self, mock_input_model, temp_output_dir):
        """Test output can be consumed by next phase."""
        # ... implementation

    def test_error_handling(self, mock_input_model, temp_output_dir):
        """Test graceful error handling."""
        # ... implementation

    def test_metrics_logged(self, mock_input_model, temp_output_dir):
        """Test W&B metrics are logged correctly."""
        # ... implementation
```

### Estimated Effort
- **Per Phase**: 4-6 hours
- **Cross-Phase Tests**: 6-8 hours
- **Total**: 40-56 hours

---

## Issue 2: Migrate to Safetensors Format

### Current State
Found **30+ torch.save/load locations** in source code:

**Source Files (Priority - Must Migrate)**:
1. `src/phase1_cognate/training/trainer.py` (2 locations)
2. `src/phase3_quietstar/step1_baking.py` (1 location)
3. `src/phase3_quietstar/step2_rl.py` (2 locations)
4. `src/phase3_quietstar/phase_handoff.py` (4 locations)
5. `src/phase4_bitnet/phase_controller.py` (2 locations)

**Test Files (Secondary - Update for Consistency)**:
- `tests/integration/test_phase3_integration.py` (8 locations)
- `tests/integration/test_phase4_integration.py` (3 locations)

### Safetensors Benefits
1. **Security**: No arbitrary code execution risk (pickle vulnerability eliminated)
2. **Speed**: Faster load times (memory-mapped access)
3. **Interoperability**: Compatible with HuggingFace ecosystem
4. **Size**: Slightly smaller file sizes

### Migration Pattern

**Before (torch.save/load)**:
```python
# Saving
torch.save({
    "model_state_dict": model.state_dict(),
    "config": config.to_dict(),
    "metadata": {"epoch": epoch}
}, output_path)

# Loading (ISS-004 fix)
checkpoint = torch.load(path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
```

**After (safetensors)**:
```python
from safetensors.torch import save_model, load_model
import json

# Saving (state dict only in safetensors, metadata in JSON)
save_model(model, output_path.with_suffix('.safetensors'))

# Save metadata separately
with open(output_path.with_suffix('.json'), 'w') as f:
    json.dump({
        "config": config.to_dict(),
        "metadata": {"epoch": epoch}
    }, f)

# Loading
load_model(model, path.with_suffix('.safetensors'))

with open(path.with_suffix('.json'), 'r') as f:
    metadata = json.load(f)
```

### Implementation Steps

1. **Add safetensors dependency**
   ```bash
   pip install safetensors
   # Add to requirements.txt
   ```

2. **Create utility functions** in `src/cross_phase/utils/checkpoint_utils.py`:
   ```python
   from safetensors.torch import save_model, load_model
   import json
   from pathlib import Path
   from typing import Dict, Any, Optional
   import torch.nn as nn

   def save_checkpoint(
       model: nn.Module,
       output_path: Path,
       config: Optional[Dict[str, Any]] = None,
       metadata: Optional[Dict[str, Any]] = None
   ) -> None:
       """
       Save model checkpoint using safetensors format.

       Args:
           model: PyTorch model to save
           output_path: Base path (will create .safetensors and .json files)
           config: Configuration dictionary
           metadata: Additional metadata
       """
       output_path = Path(output_path)
       output_path.parent.mkdir(parents=True, exist_ok=True)

       # Save model weights
       safetensors_path = output_path.with_suffix('.safetensors')
       save_model(model, str(safetensors_path))

       # Save metadata
       metadata_dict = {
           "config": config or {},
           "metadata": metadata or {}
       }
       json_path = output_path.with_suffix('.json')
       with open(json_path, 'w') as f:
           json.dump(metadata_dict, f, indent=2)

   def load_checkpoint(
       model: nn.Module,
       checkpoint_path: Path,
       device: str = "cpu"
   ) -> Dict[str, Any]:
       """
       Load model checkpoint from safetensors format.

       Args:
           model: PyTorch model to load weights into
           checkpoint_path: Base path (will look for .safetensors and .json)
           device: Device to load model to

       Returns:
           Metadata dictionary
       """
       checkpoint_path = Path(checkpoint_path)

       # Load model weights
       safetensors_path = checkpoint_path.with_suffix('.safetensors')
       load_model(model, str(safetensors_path), device=device)

       # Load metadata
       json_path = checkpoint_path.with_suffix('.json')
       if json_path.exists():
           with open(json_path, 'r') as f:
               return json.load(f)

       return {"config": {}, "metadata": {}}

   def migrate_torch_checkpoint(
       torch_path: Path,
       output_path: Path,
       model: nn.Module
   ) -> None:
       """
       Migrate existing torch checkpoint to safetensors format.

       For backward compatibility during transition.
       """
       # Load old checkpoint
       checkpoint = torch.load(torch_path, map_location="cpu", weights_only=False)

       # Load state dict into model
       if "model_state_dict" in checkpoint:
           model.load_state_dict(checkpoint["model_state_dict"])
       elif "state_dict" in checkpoint:
           model.load_state_dict(checkpoint["state_dict"])
       else:
           model.load_state_dict(checkpoint)

       # Extract metadata (everything except state dict)
       metadata = {k: v for k, v in checkpoint.items()
                   if k not in ["model_state_dict", "state_dict"]}

       # Save in new format
       save_checkpoint(model, output_path, metadata=metadata)
   ```

3. **Update each source file** to use new utility functions

4. **Add migration script** for existing checkpoints:
   ```python
   # scripts/migrate_checkpoints.py
   """Migrate all torch checkpoints to safetensors format."""
   ```

### Files to Modify

| File | Changes |
|------|---------|
| `src/phase1_cognate/training/trainer.py` | Replace 2 torch.save/load calls |
| `src/phase3_quietstar/step1_baking.py` | Replace 1 torch.save call |
| `src/phase3_quietstar/step2_rl.py` | Replace 2 torch.save/load calls |
| `src/phase3_quietstar/phase_handoff.py` | Replace 4 torch.load calls |
| `src/phase4_bitnet/phase_controller.py` | Replace 2 torch.save calls |

### Estimated Effort
- **Create utility module**: 2 hours
- **Update source files**: 4-6 hours
- **Testing**: 2-4 hours
- **Total**: 8-12 hours

---

## Issue 3: Clean Up Remaining MockTokenizer Duplicates

### Current State
Found **7 MockTokenizer definitions**:

| Location | Status | Action |
|----------|--------|--------|
| `src/cross_phase/utils.py:209` | CANONICAL | Keep as source of truth |
| `src/cross_phase/utils/tokenizer_utils.py:10` | LEGACY | Remove, redirect imports |
| `tests/conftest.py:71` | TEST FIXTURE | Replace with import |
| `tests/unit/test_bitnet_calibration.py:17` | DUPLICATE | Replace with import |
| `tests/integration/test_phase4_integration.py:53` | DUPLICATE | Replace with import |
| `src/phase1_cognate/train_phase1.py:60` | STANDALONE | Replace with import |
| `scripts/train_phase1_cached.py:101` | SCRIPT | Replace with import |

### Implementation Steps

1. **Ensure canonical MockTokenizer is properly exported**:
   ```python
   # src/cross_phase/utils.py
   from .utils import MockTokenizer, get_tokenizer

   __all__ = ['MockTokenizer', 'get_tokenizer', ...]
   ```

2. **Update each duplicate location**:
   ```python
   # Before (inline class)
   class MockTokenizer:
       vocab_size = 50257
       pad_token_id = 0
       eos_token_id = 50256
       bos_token_id = 50256
       # ... implementation

   # After (import)
   from cross_phase.utils import MockTokenizer
   # OR for test files:
   from cross_phase.utils import get_tokenizer
   tokenizer = get_tokenizer("gpt2")  # Returns MockTokenizer
   ```

3. **Remove legacy tokenizer_utils.py** (or keep as re-export):
   ```python
   # src/cross_phase/utils/tokenizer_utils.py
   # DEPRECATED: Use cross_phase.utils.MockTokenizer instead
   from cross_phase.utils import MockTokenizer, get_tokenizer

   __all__ = ['MockTokenizer', 'get_tokenizer']
   ```

### Estimated Effort
- **Total**: 2 hours

---

## Issue 4: Implement Full RL Training (ISS-007)

### Current State
`src/phase3_quietstar/step2_rl.py` contains `REINFORCETrainer` class with:
- Basic episode loop
- Reward calculation
- Anti-theater validation
- W&B logging

**Gap**: MVP trains for limited episodes. Full implementation needs:
- 10,000+ episode training
- Advantage estimation (baseline subtraction)
- Entropy bonus for exploration
- Gradient clipping
- Learning rate scheduling
- Early stopping on convergence

### Implementation Plan

**Current REINFORCETrainer (MVP)**:
```python
class REINFORCETrainer:
    def train(self, train_dataloader, val_dataloader) -> Dict:
        # MVP: Limited episodes
        for episode in range(self.config.rl.num_episodes):  # ~100 episodes
            # Basic REINFORCE update
            loss = -log_prob * reward
            loss.backward()
            self.optimizer.step()
```

**Full Implementation Required**:
```python
class REINFORCETrainer:
    def __init__(self, ...):
        # Add baseline network for variance reduction
        self.baseline_network = nn.Linear(hidden_size, 1)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.rl.num_episodes
        )

        # Entropy coefficient for exploration
        self.entropy_coef = config.rl.entropy_coefficient

        # Gradient clipping
        self.max_grad_norm = config.rl.max_grad_norm

    def compute_advantage(self, rewards, values):
        """Compute advantage using GAE or simple baseline subtraction."""
        advantages = rewards - values  # Simple baseline
        # OR: GAE (Generalized Advantage Estimation)
        return advantages

    def train(self, train_dataloader, val_dataloader) -> Dict:
        """Full REINFORCE training with 10,000+ episodes."""

        best_reward = float('-inf')
        patience_counter = 0

        for episode in range(self.config.rl.num_episodes):  # 10,000+
            # Sample batch
            batch = next(iter(train_dataloader))

            # Forward pass
            outputs = self.model(batch)
            log_probs = outputs.log_probs

            # Compute rewards
            rewards = self._compute_rewards(outputs, batch)

            # Compute baseline values
            with torch.no_grad():
                values = self.baseline_network(outputs.hidden_states).squeeze()

            # Compute advantages
            advantages = self.compute_advantage(rewards, values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss
            policy_loss = -(log_probs * advantages).mean()

            # Entropy bonus for exploration
            entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
            entropy_loss = -self.entropy_coef * entropy

            # Value loss (train baseline)
            value_loss = F.mse_loss(values, rewards)

            # Total loss
            total_loss = policy_loss + entropy_loss + 0.5 * value_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            # Logging
            if episode % 100 == 0:
                self._log_metrics(episode, policy_loss, entropy, rewards)

            # Validation
            if episode % 1000 == 0:
                val_reward = self._validate(val_dataloader)

                if val_reward > best_reward:
                    best_reward = val_reward
                    patience_counter = 0
                    self._save_best_model()
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= self.config.rl.patience:
                    print(f"Early stopping at episode {episode}")
                    break

        return self._compile_results()
```

### Configuration Updates

```python
# src/phase3_quietstar/config.py
@dataclass
class RLConfig:
    # Episode settings
    num_episodes: int = 10000  # Increased from MVP
    batch_size: int = 32

    # Learning settings
    learning_rate: float = 1e-4
    lr_schedule: str = "cosine"  # "constant", "cosine", "linear"

    # Exploration
    entropy_coefficient: float = 0.01
    entropy_decay: float = 0.995  # Decay entropy bonus over time

    # Stability
    max_grad_norm: float = 1.0
    value_loss_coef: float = 0.5

    # Early stopping
    patience: int = 10  # Stop if no improvement for N validations
    validation_frequency: int = 1000

    # Advantage estimation
    use_gae: bool = True
    gae_lambda: float = 0.95
    gamma: float = 0.99
```

### Estimated Effort
- **Implement advantage estimation**: 4 hours
- **Add baseline network**: 2 hours
- **Implement full training loop**: 8 hours
- **Testing and validation**: 6 hours
- **Total**: 20+ hours

---

## Issue 5: Implement STE Fine-Tuning (ISS-008)

### Current State
`src/phase4_bitnet/fine_tuner.py` contains `FineTuner` class with:
- MuGrokfast optimizer with STE mode enabled
- Basic training loop
- Perplexity monitoring

**Gap**: The STE (Straight-Through Estimator) is configured via `muon_ste_mode=True` but the actual gradient flow through quantized weights needs full implementation.

### Current Implementation

The FineTuner class is actually fairly complete. The "stub" label may be outdated. Let me verify what's missing:

**Already Implemented**:
- MuGrokfast optimizer creation with STE mode
- Training epoch loop
- Evaluation with perplexity
- should_fine_tune() decision logic

**Potentially Missing**:
1. **Custom STE backward pass** - May be handled by MuGrokfast
2. **Quantization-aware gradient scaling**
3. **Mixed-precision training support**
4. **Checkpoint saving/loading**
5. **W&B integration**

### Implementation Plan

```python
# src/phase4_bitnet/fine_tuner.py additions

class FineTuner:
    def __init__(self, ...):
        # ... existing code ...

        # Add gradient scaler for mixed precision
        self.use_amp = config.use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # W&B logging
        self.wandb_enabled = config.wandb_enabled

    def _train_epoch_with_ste(
        self,
        dataloader: DataLoader,
        log_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Train epoch with proper STE gradient flow.

        STE (Straight-Through Estimator):
        - Forward: Use quantized weights
        - Backward: Gradient flows through as if weights were not quantized
        """
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get(
                "attention_mask",
                torch.ones_like(input_ids)
            ).to(self.device)

            # Mixed precision context
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Forward pass (uses quantized weights internally)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # Backward pass (STE: gradients flow through)
            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Unscale before gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Accumulate
            total_loss += loss.item()
            num_batches += 1

            # Update progress
            pbar.set_postfix({'loss': loss.item()})

            # W&B logging
            if self.wandb_enabled and batch_idx % 10 == 0:
                wandb.log({
                    'fine_tune/batch_loss': loss.item(),
                    'fine_tune/epoch': self.current_epoch,
                    'fine_tune/batch': batch_idx,
                })

        return {
            'epoch': self.current_epoch,
            'loss': total_loss / num_batches,
            'num_batches': num_batches,
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save fine-tuning checkpoint."""
        # Use safetensors if available
        from cross_phase.utils.checkpoint_utils import save_checkpoint

        save_checkpoint(
            model=self.model,
            output_path=path,
            config=self.config.__dict__,
            metadata={
                'epoch': self.current_epoch,
                'best_perplexity': self.best_perplexity,
                'training_history': self.training_history,
            }
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load fine-tuning checkpoint."""
        from cross_phase.utils.checkpoint_utils import load_checkpoint

        metadata = load_checkpoint(
            model=self.model,
            checkpoint_path=path,
            device=self.device
        )

        self.current_epoch = metadata.get('metadata', {}).get('epoch', 0)
        self.best_perplexity = metadata.get('metadata', {}).get('best_perplexity', float('inf'))
        self.training_history = metadata.get('metadata', {}).get('training_history', [])
```

### Configuration Updates

```python
# src/phase4_bitnet/config.py additions
@dataclass
class Phase4Config:
    # ... existing fields ...

    # STE Fine-tuning
    use_mixed_precision: bool = True
    max_grad_norm: float = 1.0
    wandb_enabled: bool = True

    # Checkpointing
    save_every_n_epochs: int = 1
    checkpoint_dir: str = "checkpoints/phase4"
```

### Estimated Effort
- **Implement STE training loop**: 6 hours
- **Add mixed precision support**: 4 hours
- **Checkpoint save/load**: 2 hours
- **W&B integration**: 2 hours
- **Testing**: 4 hours
- **Total**: 16+ hours

---

## Execution Order

### Recommended Priority

1. **MockTokenizer Cleanup** (2 hours) - Quick win, low risk
2. **Safetensors Migration** (8-12 hours) - Security improvement, foundation for other work
3. **E2E Tests** (40-56 hours) - Validates all phases work together
4. **STE Fine-Tuning** (16+ hours) - Completes Phase 4 pipeline
5. **Full RL Training** (20+ hours) - Completes Phase 3 pipeline

### Dependencies

```
MockTokenizer Cleanup -----> (No dependencies)
Safetensors Migration -----> (No dependencies)
E2E Tests ----------------> (Benefits from Safetensors migration)
STE Fine-Tuning -----------> (Benefits from Safetensors migration)
Full RL Training ----------> (Benefits from Safetensors migration)
```

### Parallelization Opportunities

**Can Run in Parallel**:
- MockTokenizer Cleanup + Safetensors Migration
- E2E Tests (per-phase tests are independent)

**Must Be Sequential**:
- Safetensors Migration -> Source files before tests
- Full implementations -> Test updates

---

## Verification Commands

```bash
# After MockTokenizer cleanup
grep -r "class MockTokenizer" src/ tests/ scripts/  # Should find only 1 (canonical)

# After Safetensors migration
grep -r "torch\.save\|torch\.load" src/  # Should find 0 in core modules

# Run E2E tests
python -m pytest tests/e2e/ -v --tb=short

# Run full test suite
python -m pytest tests/ -v --tb=short --cov=src --cov-report=html
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Safetensors breaks existing checkpoints | Medium | High | Create migration script, keep backward compatibility |
| E2E tests reveal hidden bugs | High | Medium | Fix bugs as discovered, track in issues |
| Full RL training doesn't converge | Medium | High | Use proven hyperparameters from literature |
| STE gradients are incorrect | Low | High | Validate against known working implementations |

---

## Success Criteria

- [ ] All 7 MockTokenizer duplicates removed (1 canonical remains)
- [ ] All torch.save/load in src/ migrated to safetensors
- [ ] E2E tests exist for all 8 phases
- [ ] E2E tests achieve 100% pass rate
- [ ] Full RL training runs for 10,000+ episodes
- [ ] RL training shows convergence (reward improvement)
- [ ] STE fine-tuning recovers >95% of pre-quantization quality
- [ ] Overall test coverage increases to >80%

---

*Plan created: 2025-11-26*
*Ready for execution approval*
