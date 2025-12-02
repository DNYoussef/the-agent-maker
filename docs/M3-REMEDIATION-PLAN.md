# M3 TIER 0 Infrastructure Remediation Plan

**Date:** 2025-11-27
**Status:** PENDING EXECUTION
**Prerequisite:** M1 (Security) COMPLETE, M2 (Verification) COMPLETE

---

## Executive Summary

M2 Verification identified 3 blocking issues that must be fixed before TIER 0 gate passes:

| Issue | Status | Root Cause | Agent | Skill |
|-------|--------|------------|-------|-------|
| ISS-001 | BROKEN | Missing exports in `__init__.py` | dev-backend-api | backend-dev |
| ISS-003 | PARTIAL | MuonGrokfast missing `config=` kwarg | dev-backend-api | backend-dev |
| ISS-005 | PARTIAL | ThoughtGenerator `.item()` on batch>1 | automl-optimizer | machine-learning |

**Already Verified Fixed:**
- ISS-002: W&B integration (9 tests collected)
- ISS-004: torch.load vulnerabilities (SafeTensors migration complete)

---

## Task Breakdown with Agent Routing

### Task 0.16-0.18: Fix ISS-001 (Missing Utility Exports)

**Agent:** `delivery/development/backend/dev-backend-api.md`
**Skill:** `backend-dev`
**Playbook:** `simple-feature`

**Problem:**
`tests/unit/test_utils.py` imports functions that exist in `src/cross_phase/utils.py` but are NOT exported by `src/cross_phase/utils/__init__.py`:

```python
# test_utils.py imports these:
from cross_phase.utils import (
    get_model_size,              # EXISTS in utils.py:12
    calculate_safe_batch_size,   # EXISTS in utils.py:43
    validate_model_diversity,    # EXISTS in utils.py:377
    detect_training_divergence,  # EXISTS in utils.py:398
    compute_population_diversity # EXISTS in utils.py:424
)
```

**Current `__init__.py` exports:**
```python
__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "SAFETENSORS_AVAILABLE",
    "MockTokenizer",
    "get_tokenizer",
]
```

**Fix Required:**
Edit `src/cross_phase/utils/__init__.py` to export the missing functions:

```python
# Add after line 18 (MockTokenizer = _utils.MockTokenizer):
get_model_size = _utils.get_model_size
calculate_safe_batch_size = _utils.calculate_safe_batch_size
validate_model_diversity = _utils.validate_model_diversity
detect_training_divergence = _utils.detect_training_divergence
compute_population_diversity = _utils.compute_population_diversity

# Update __all__ to include new exports
__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "SAFETENSORS_AVAILABLE",
    "MockTokenizer",
    "get_tokenizer",
    "get_model_size",
    "calculate_safe_batch_size",
    "validate_model_diversity",
    "detect_training_divergence",
    "compute_population_diversity",
]
```

**Verification Command:**
```bash
python -c "from cross_phase.utils import get_model_size, calculate_safe_batch_size, validate_model_diversity, detect_training_divergence, compute_population_diversity; print('ISS-001 FIXED')"
```

---

### Task 0.16b: Fix ISS-003 (MuonGrokfast API Mismatch)

**Agent:** `delivery/development/backend/dev-backend-api.md`
**Skill:** `backend-dev`
**Playbook:** `simple-feature`

**Problem:**
Tests expect `MuonGrokfast(params, config=config)` but optimizer only accepts individual kwargs:

```python
# test_mugrokfast.py:70 expects:
optimizer = MuonGrokfast(mock_model.parameters(), config=config)
assert optimizer.config.muon_lr == 1e-3  # Expects .config attribute

# optimizer.py:33 actual signature:
def __init__(self, params, muon_lr=0.01, fallback_lr=1e-3, ...)
# NO config= parameter, NO self.config attribute
```

**Fix Required:**
Update `src/cross_phase/mugrokfast/optimizer.py` to accept `config=` kwarg:

```python
def __init__(
    self,
    params,
    config: Optional['MuGrokConfig'] = None,  # ADD THIS
    muon_lr: float = 0.01,
    fallback_lr: float = 1e-3,
    grokfast_alpha: float = 0.98,
    grokfast_lambda: float = 2.0,
    qk_clip_threshold: float = 30.0,
    kl_coefficient: float = 0.0,
    muon_ste_mode: bool = False,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
):
    # If config provided, use its values
    if config is not None:
        muon_lr = config.muon_lr
        fallback_lr = config.fallback_lr
        grokfast_alpha = config.grokfast_alpha
        grokfast_lambda = config.grokfast_lambda
        qk_clip_threshold = config.qk_clip_threshold
        kl_coefficient = config.kl_coefficient
        muon_ste_mode = config.muon_ste_mode
        momentum = config.momentum
        nesterov = config.nesterov
        ns_steps = config.ns_steps

    # Store config for attribute access
    self.config = config

    # ... rest of __init__
```

**Verification Command:**
```bash
pytest tests/unit/test_mugrokfast.py -v
# Expected: 11 passed (was 8 passed, 3 failed)
```

---

### Task 0.16c: Fix ISS-005 (ThoughtGenerator Tensor Bug)

**Agent:** `platforms/ai-ml/automl/automl-optimizer.md`
**Skill:** `machine-learning`
**Playbook:** `ml-pipeline`

**Problem:**
`ThoughtGenerator._generate_single()` calls `.item()` on a tensor with batch_size > 1:

```python
# architecture.py:128
generated_ids.append(next_token.item())  # FAILS when batch_size > 1
```

Error: `RuntimeError: a Tensor with 2 elements cannot be converted to Scalar`

**Root Cause Analysis:**
- `next_token = torch.multinomial(probs, num_samples=1)` returns shape `(batch_size, 1)`
- When `batch_size=2`, `next_token.item()` fails because tensor has 2 elements
- The code assumes batch_size=1 but tests use batch_size=2

**Fix Required:**
Update `src/phase3_quietstar/architecture.py` `_generate_single()` method:

```python
def _generate_single(
    self,
    input_ids: torch.Tensor,
    position: int,
    hidden_states: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Generate a single thought continuation."""
    device = input_ids.device
    batch_size = input_ids.size(0)

    # Start from position
    current_ids = input_ids[:, : position + 1].clone()
    generated_ids = []
    log_probs_list = []

    # Adaptive length (10-20 tokens)
    thought_length = torch.randint(
        self.min_length, self.max_length + 1, (1,)
    ).item()

    # Generate tokens
    for step in range(thought_length):
        outputs = self.base_model(current_ids)
        logits = outputs.logits[:, -1, :] / self.temperature

        # Nucleus sampling
        probs = self._nucleus_sampling(logits)
        next_token = torch.multinomial(probs, num_samples=1)

        # Store results - FIX: Handle batch_size > 1
        # Store first batch item for IDs (or store all as list of lists)
        generated_ids.append(next_token[0, 0].item())  # First batch, first sample
        log_probs_list.append(
            torch.log(probs.gather(1, next_token))
        )

        # Append token
        current_ids = torch.cat([current_ids, next_token], dim=1)

    # Aggregate
    thought_hidden = outputs.last_hidden_state[:, -thought_length:, :]
    log_prob_sum = torch.stack(log_probs_list).sum(dim=0).squeeze(-1)  # FIX: Sum per batch

    return thought_hidden, log_prob_sum, generated_ids
```

**Alternative Fix (More Robust):**
Return `generated_ids` as list of lists (one per batch item):

```python
# Instead of: generated_ids.append(next_token.item())
# Use: generated_ids.append(next_token.squeeze(-1).tolist())
```

**Verification Command:**
```bash
pytest tests/unit/test_phase3_architecture.py -v
# Expected: 39 passed (was 21 passed, 18 failed)
```

---

### Task 0.19: Pytest Collection Verification

**Agent:** `quality/testing/test-orchestrator.md`
**Skill:** `test-orchestrator`
**Playbook:** `testing-quality`

**Verification Commands:**
```bash
# Full collection (should have 0 errors)
pytest --collect-only 2>&1 | grep -E "(error|Error|ERROR)"
# Expected: Empty output

# Count collected tests
pytest --collect-only -q | tail -1
# Expected: 605+ tests collected

# Specific ISS file verification
pytest tests/unit/test_utils.py --collect-only
# Expected: Tests collected (was ImportError)

pytest tests/unit/test_mugrokfast.py -v
# Expected: 11 passed

pytest tests/unit/test_phase3_architecture.py -v
# Expected: 39 passed
```

---

### Task 0.20: TIER 0 Gate Check

**Agent:** `orchestration/consensus/byzantine-coordinator.md`
**Skill:** `gate-validation`
**Playbook:** `gate-validation`

**Gate Criteria:**

| Criterion | Command | Expected |
|-----------|---------|----------|
| ISS-001 | `python -c "from cross_phase.utils import get_model_size"` | No error |
| ISS-002 | `pytest tests/unit/test_wandb_integration.py --collect-only` | 9 tests |
| ISS-003 | `pytest tests/unit/test_mugrokfast.py -v` | 11 passed |
| ISS-004 | `grep -rn "torch\.load(" src/ \| grep -v weights_only` | Empty |
| ISS-005 | `pytest tests/unit/test_phase3_architecture.py -v` | 39 passed |
| Collection | `pytest --collect-only` | 0 errors |

**Gate Decision:**
- ALL 6 criteria PASS -> TIER 0 GATE PASSED -> Proceed to TIER 1 (M4)
- ANY criterion FAILS -> TIER 0 GATE FAILED -> Fix before proceeding

---

## Execution Order

```
1. [dev-backend-api] Task 0.16: Fix ISS-001 exports
2. [dev-backend-api] Task 0.16b: Fix ISS-003 config= API
3. [automl-optimizer] Task 0.16c: Fix ISS-005 tensor bug
4. [test-orchestrator] Task 0.19: Verify pytest collection
5. [byzantine-coordinator] Task 0.20: TIER 0 Gate Check
```

**Parallel Execution Option:**
Tasks 0.16, 0.16b, and 0.16c can run in PARALLEL (no dependencies).
Task 0.19 and 0.20 must run SEQUENTIALLY after fixes complete.

---

## Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `src/cross_phase/utils/__init__.py` | Add 5 exports + update __all__ | ~10 lines |
| `src/cross_phase/mugrokfast/optimizer.py` | Add config= param + self.config | ~15 lines |
| `src/phase3_quietstar/architecture.py` | Fix .item() for batch>1 | ~5 lines |

**Total Changes:** ~30 lines of code

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Export breaks existing imports | LOW | Only adding exports, not removing |
| config= breaks existing optimizer usage | LOW | Backwards compatible (config=None default) |
| Tensor fix changes output shape | MEDIUM | Tests will validate shape expectations |

---

## Post-Execution Verification

After all fixes applied, run comprehensive verification:

```bash
# 1. Import verification
python -c "
from cross_phase.utils import (
    get_model_size,
    calculate_safe_batch_size,
    validate_model_diversity,
    detect_training_divergence,
    compute_population_diversity
)
print('ISS-001: PASS')
"

# 2. Full test suite
pytest tests/unit/test_utils.py tests/unit/test_mugrokfast.py tests/unit/test_phase3_architecture.py -v

# 3. Collection check
pytest --collect-only 2>&1 | grep -c "error"
# Expected: 0

# 4. Security check (ISS-004 regression)
grep -rn "torch\.load(" src/ --include="*.py" | grep -v "weights_only" | grep -v "checkpoint_utils"
# Expected: Empty
```

---

## Next Steps After TIER 0 Gate Passes

1. Update Progress Tracker: Mark M3 COMPLETE
2. Proceed to M4 (TIER 1 Critical): Phase 6 + Phase 1 implementation
3. Use `P0 + M4 + A1` prompt composition for next session

---

**Document Version:** 1.0
**Created By:** M2 Verification Analysis
**Awaiting:** Execution approval
