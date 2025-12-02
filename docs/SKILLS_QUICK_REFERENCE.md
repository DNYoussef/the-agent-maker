# ML Skills Quick Reference Card

Fast reference for using ml-training-debugger and ml-expert skills.

---

## 🔍 ML Training Debugger

**Purpose**: Diagnose ML training failures with systematic evidence-based analysis

### Quick Invocation
```
/ml-training-debugger

[Describe issue + provide artifacts]
```

### When to Use
- ❌ Loss diverging (increasing instead of decreasing)
- ❌ Mode collapse (same output for all inputs)
- ❌ Gradient explosion/vanishing (NaN, inf, warnings)
- ❌ Architecture issues (parameter imbalance)
- ❌ Training warnings (variance=0, etc.)
- ❌ Model outputs gibberish/nonsense

### What to Provide
- **Symptoms**: What went wrong and when (epoch, step)
- **Logs**: Training logs, error messages
- **Code**: Relevant model/trainer files
- **Config**: Hyperparameters, settings

### What You Get Back
```json
{
  "root_causes": [
    {"issue": "...", "confidence": 0.95, "evidence": [...], "fix": "..."}
  ],
  "quick_wins": ["Fix 1", "Fix 2", "Fix 3"],
  "recommended_fixes": [
    {"priority": 1, "implementation": {...}, "expected_impact": "..."}
  ]
}
```

### Example
```
/ml-training-debugger

Training diverged at epoch 7. Loss was decreasing (3.76→0.16),
then increased (0.16→0.19). grad_norm spiked to 45.2.

Logs: phases/phase1/TRAINING_FAILURE_DIAGNOSIS.md
Code: src/phase1_cognate/training/trainer.py
```

---

## 🛠️ ML Expert

**Purpose**: Implement ML solutions with production-quality code and tests

### Quick Invocation
```
/ml-expert

[Describe what to implement]
```

### When to Use
- ✅ Implement diagnosis fixes
- ✅ Create new model architectures
- ✅ Optimize performance (speed, memory)
- ✅ Add features to existing models
- ✅ Fix bugs in ML code
- ✅ Implement research papers

### What to Provide
- **Requirements**: What to build/fix
- **Constraints**: Parameter budget, VRAM, speed
- **Context**: Existing code (if modifying)
- **References**: Papers, docs (if implementing)

### What You Get Back
```json
{
  "code_changes": [
    {"file": "...", "change": "...", "description": "..."}
  ],
  "tests": "48/48 passing",
  "verification": {"all_checks": "✓ Passing"},
  "performance": {"params": "25.6M", "inference": "45ms"}
}
```

### Example
```
/ml-expert

Implement the top 2 fixes from diagnosis:
1. Reduce muon_lr from 1e-2 to 5e-3
2. Add diversity regularization to ACT head

Ensure all tests pass.
```

---

## 🔄 Typical Workflow

### 1. Problem → Diagnosis
```
User has training failure
  ↓
/ml-training-debugger
  ↓
Systematic analysis (2-5 min)
  ↓
Diagnosis with fixes (confidence scores)
```

### 2. Diagnosis → Implementation
```
Diagnosis suggests fixes
  ↓
/ml-expert
  ↓
Implementation plan (you approve)
  ↓
Code + tests (3-10 min)
  ↓
Validated implementation
```

### 3. Implementation → Testing
```
Quick test (CPU, 1 epoch): 1-2 min
  ↓
If passing → Full retrain (GPU, 10 epochs): 4-6 hours
  ↓
Success! Or iterate if issues remain
```

**Total time (diagnosis → fix → test)**: **5-15 minutes** before full retrain

---

## 📋 Cheat Sheet

### Common Issues & Skills

| Issue | First Skill | Then |
|-------|------------|------|
| Loss divergence | /ml-training-debugger | /ml-expert (implement LR fix) |
| Mode collapse | /ml-training-debugger | /ml-expert (fix architecture) |
| Gradient explosion | /ml-training-debugger | /ml-expert (add clipping) |
| ACT variance=0 | /ml-training-debugger | /ml-expert (add diversity loss) |
| Slow inference | /ml-expert (optimize) | Test performance |
| Implement paper | /ml-expert (implement) | /ml-training-debugger (if issues) |

### Quick Commands

**Diagnose current Phase 1 failure**:
```
/ml-training-debugger

Phase 1 training failed. See:
- phases/phase1/TRAINING_FAILURE_DIAGNOSIS.md
- phases/phase1/NUCLEAR_FIX_IMPLEMENTATION_SUMMARY.md
```

**Implement ACT diversity fix**:
```
/ml-expert

Add diversity regularization to ACT head:
File: src/phase1_cognate/model/act_head.py
Method: compute_act_loss()
Add diversity loss term to encourage token variance.
```

**Verify fixes worked**:
```bash
python src/phase1_cognate/train_phase1.py --model reasoning --test
```

---

## 🎯 Success Metrics

### Debugger Quality
- ✅ Root cause identified (>80% confidence)
- ✅ Evidence from actual artifacts (not speculation)
- ✅ ≥3 prioritized fixes proposed
- ✅ Analysis completes in 2-5 minutes

### Expert Quality
- ✅ Production-quality code (clean, documented)
- ✅ All tests passing (≥90% coverage)
- ✅ Implementation validated (end-to-end)
- ✅ Performance metrics reported

### Overall Workflow
- ✅ Problem diagnosed in <5 min
- ✅ Fix implemented in <10 min
- ✅ Tests pass before full retrain
- ✅ Issue resolved (training succeeds)

---

## 🚨 Troubleshooting

### Skill Not Activating
```
# Try explicit slash command
/ml-training-debugger

# Or natural language
Use the ml-training-debugger skill to diagnose...
```

### Debugger Needs More Info
```
Provide these if available:
- Training logs (copy-paste sections)
- Error messages (full tracebacks)
- Model code (relevant files)
- Config files (hyperparameters)

Even partial info helps (lowers confidence but still useful)
```

### Implementation Breaks Tests
```
ml-expert will auto-debug and fix.

If stuck:
1. Revert changes
2. Try alternative fix
3. Provide test failure details to ml-expert
```

---

## 📚 Documentation Locations

### Skills
- `~/.claude/skills/ml-training-debugger/SKILL.md`
- `~/.claude/skills/ml-expert/SKILL.md`

### Agent Prompts
- `~/.claude/skills/ml-training-debugger/agents/ml-debugger-specialist.prompt`
- `~/.claude/skills/ml-expert/agents/ml-expert-specialist.prompt`

### Guides
- `docs/PHASE1_SKILLS_WORKFLOW.md` - Comprehensive workflow guide
- `docs/PHASE1_READY_TO_TRAIN.md` - Phase 1 status and fixes
- `docs/PHASE1_QUICK_START.md` - Quick start training guide

---

## 💡 Pro Tips

### 1. Always Diagnose First
Don't skip to implementing fixes. Debugger analysis prevents wrong solutions.

### 2. Provide Context
More artifacts = better diagnosis. Give logs, code, config when available.

### 3. Review Plans
ml-expert shows implementation plan before coding. Review and approve.

### 4. Test Incrementally
Quick test (1 epoch, CPU) before full GPU retrain (saves hours if fix is wrong).

### 5. Iterate if Needed
If first fix doesn't fully resolve, re-diagnose with new symptoms.

---

## ⚡ One-Liner Commands

**Diagnose & fix Phase 1 in one session**:
```
First: /ml-training-debugger [provide artifacts]
Wait for diagnosis...
Then: /ml-expert [implement top fixes]
Wait for implementation...
Finally: python train_phase1.py --model reasoning --test
```

**Expected total time**: 10-15 minutes (diagnosis + implementation + quick test)

If successful → Full retrain: 4-6 hours

---

**Print this card and keep it handy while working on ML projects!**
