# Agent Forge V2 - Developer Reading Order

**Version**: 1.0
**Created**: 2025-10-16
**Purpose**: Definitive ordered reading list for developers building Agent Forge V2

---

## Quick Start (5-Minute Overview)

If you have 5 minutes, read these files in order:

1. [README.md](../README.md) - Project overview (Lines 1-100)
2. [CLAUDE.md](../CLAUDE.md) - V1 vs V2 distinction (Lines 1-50, 129-182)
3. [docs/INDEX.md](INDEX.md) - Navigation guide (Quick Links section)

**After 5 minutes, you'll understand**: V2 is a local-first rebuild (not refactor), 8-phase pipeline, 25M param models, 16-week timeline.

---

## Full Reading Order (4-6 Hours Total)

### Phase 1: Project Context (30 minutes)

**Goal**: Understand what you're building and why

| # | Document | Time | Why Read This |
|---|----------|------|---------------|
| 1 | [README.md](../README.md) | 15 min | Complete project overview, V1 vs V2 comparison |
| 2 | [CLAUDE.md](../CLAUDE.md) | 10 min | Critical V1/V2 distinctions, file organization rules |
| 3 | [docs/INDEX.md](INDEX.md) | 5 min | Master navigation guide |

**Key Takeaways**:
- ✅ V2 is **NOT** a refactor of V1 code
- ✅ V2 runs **locally** on consumer hardware (6GB VRAM)
- ✅ V1 documentation is **reference only** (historical context)
- ✅ Clean build with NASA POT10 compliance from day 1

---

### Phase 2: V2 Specifications (90 minutes)

**Goal**: Understand complete system architecture and requirements

| # | Document | Lines | Time | Section Focus |
|---|----------|-------|------|---------------|
| 4 | [docs/v2-specification/SPECIFICATION_INDEX.md](v2-specification/SPECIFICATION_INDEX.md) | 363 | 10 min | Overview of 165-page spec, navigation guide |
| 5 | [docs/v2-specification/AGENT_FORGE_V2_SPECIFICATION.md](v2-specification/AGENT_FORGE_V2_SPECIFICATION.md) | 2,082 | 40 min | **Section 1**: Executive Summary<br>**Section 2**: Phases 1-4 (detailed)<br>**Section 3.1-3.2**: Infrastructure |
| 6 | [docs/v2-specification/AGENT_FORGE_V2_SPECIFICATION_PART2.md](v2-specification/AGENT_FORGE_V2_SPECIFICATION_PART2.md) | 2,226 | 40 min | **Section 2.5-2.8**: Phases 5-8 (detailed)<br>**Section 3.3**: Execution environment<br>**Section 4**: W&B metrics (603 total)<br>**Section 5**: UI/CLI<br>**Section 9**: Data schemas<br>**Section 10**: 16-week roadmap |

**Critical Sections**:
- **Executive Summary** (Part 1, Lines 1-150): System vision, success criteria, tech stack
- **Phase Specifications** (Both parts): Input/output/algorithms for all 8 phases
- **W&B Metrics** (Part 2, Section 4): 603 metrics breakdown
- **16-Week Roadmap** (Part 2, Section 10): Week-by-week implementation plan

**Key Takeaways**:
- ✅ Complete technical specifications for all 8 phases
- ✅ 603 W&B metrics across pipeline
- ✅ Local-first architecture (SQLite, filesystem, CLI/Jupyter)
- ✅ Hardware constraints: 6GB VRAM, 16GB RAM, 50GB disk

---

### Phase 3: V2 Planning & Risk Analysis (60 minutes)

**Goal**: Understand implementation strategy, risks, and timeline

| # | Document | Lines | Time | Focus |
|---|----------|-------|------|-------|
| 7 | [docs/v2-planning/PHASE5-8_V1_VS_V2_RECONCILIATION.md](v2-planning/PHASE5-8_V1_VS_V2_RECONCILIATION.md) | ~400 | 15 min | **CRITICAL**: Phase 5-8 V2 redesigns explained |
| 8 | [docs/v2-planning/ISSUE_RESOLUTION_MATRIX.md](v2-planning/ISSUE_RESOLUTION_MATRIX.md) | 1,136 | 20 min | All 34 issues resolved, 100% status |
| 9 | [docs/v2-planning/PREMORTEM_ANALYSIS.md](v2-planning/PREMORTEM_ANALYSIS.md) | 1,145 | 15 min | Infrastructure risks, hardware validation |
| 10 | [docs/v2-planning/PLAN-V2-BUILD.md](v2-planning/PLAN-V2-BUILD.md) | ~600 | 10 min | 16-week timeline, week-by-week breakdown |

**Critical Insights**:
- **Phase 5-8 Redesigns**: V2 differs significantly from V1 (curriculum learning, iterative A/B, self-guided experts, production-ready)
- **Cost Impact**: V1 $0 → V2 $750-1050 (frontier models: GPT-4o-mini, Claude-3.5 Haiku, Gemini 2.0 Flash, Qwen 2.5)
- **All Issues Resolved**: 34/34 issues addressed, 100% confidence level
- **No Blockers**: Ready for implementation

**Key Takeaways**:
- ✅ Phase 5: **Curriculum Learning** (NOT "BitNet + Grokfast")
- ✅ Phase 6: **Iterative A/B Optimization** (NOT "9 pre-defined personas")
- ✅ Phase 7: **Self-Guided Experts** (NOT "generic edge")
- ✅ Phase 8: **Production Ready** with benchmark testing
- ✅ Dream consolidation: 3 epochs × 10 levels, 5-10 hours total

---

### Phase 4: Critical Cross-Phase Documents (45 minutes)

**Goal**: Understand phase handoffs, dependencies, and integration patterns

| # | Document | Lines | Time | Why Critical |
|---|----------|-------|------|--------------|
| 11 | [docs/v2-planning/PHASE4_TO_PHASE5_HANDOFF.md](v2-planning/PHASE4_TO_PHASE5_HANDOFF.md) | ~300 | 15 min | **BLOCKING ISSUE SOLVED**: BitNet quantization → Curriculum learning dequantization |
| 12 | [docs/integration/BITNET_COMPATIBILITY_ALL_PHASES.md](integration/BITNET_COMPATIBILITY_ALL_PHASES.md) | ~400 | 15 min | Phase 4 BitNet compatibility across all phases |
| 13 | [docs/v2-planning/PHASE5_DREAM_CONSOLIDATION_IMPLEMENTATION.md](v2-planning/PHASE5_DREAM_CONSOLIDATION_IMPLEMENTATION.md) | ~500 | 15 min | Complete implementation based on "Dreaming is All You Need" paper |

**Critical Problem Solved**:
- **Phase 4 → 5 Incompatibility**: BitNet outputs 1.58-bit quantized weights, Phase 5 needs gradients
- **Solution**: Dequantize to FP16 (50MB) for Phases 5-7, re-quantize in Phase 8
- **Impact**: Phase 5 training works, Phase 8 still achieves 125× compression

**Key Takeaways**:
- ✅ Phase 4 → 5 handoff solved with dequantization
- ✅ Dream consolidation: Full autoencoder, frozen weights, temperature 1.2
- ✅ BitNet compatibility documented for all phases

---

### Phase 5: Dependencies & Infrastructure (30 minutes)

**Goal**: Understand tooling, dependencies, and code quality enforcement

| # | Document | Lines | Time | Focus |
|---|----------|-------|------|-------|
| 14 | [docs/DEPENDENCY_VERSIONS.md](DEPENDENCY_VERSIONS.md) | ~400 | 10 min | All pinned dependencies, frontier model configs, costs |
| 15 | [docs/PRE_COMMIT_SETUP_GUIDE.md](PRE_COMMIT_SETUP_GUIDE.md) | 600+ | 15 min | Complete pre-commit hooks setup (NASA POT10, type checking, security) |
| 16 | [.pre-commit-config.yaml](../.pre-commit-config.yaml) | 133 | 5 min | Pre-commit hook configuration |

**Key Infrastructure**:
- **PyTorch 2.1.0**, **Transformers 4.38.0**, **W&B 0.16.3** (all pinned)
- **Frontier Models**: GPT-4o-mini ($0.15/1M), Claude-3.5 Haiku ($0.80/1M), Gemini 2.0 Flash ($0.075/1M), Qwen 2.5 ($0.40/1M)
- **Pre-Commit Hooks**: Black, isort, Flake8, MyPy, NASA POT10 checker, test coverage, security checks

**Key Takeaways**:
- ✅ All dependencies pinned for reproducibility
- ✅ Frontier model costs budgeted: $750-1050 total
- ✅ Code quality enforced automatically (NASA POT10: ≤60 LOC/function)
- ✅ No secrets, no backup files (git branches only)

---

### Phase 6: W&B Integration (30 minutes)

**Goal**: Understand experiment tracking and metrics system

| # | Document | Lines | Time | Focus |
|---|----------|-------|------|-------|
| 17 | [docs/integration/WANDB_INTEGRATION_GUIDE.md](integration/WANDB_INTEGRATION_GUIDE.md) | ~500 | 15 min | W&B setup, offline mode, dashboard config |
| 18 | [docs/integration/PHASE1-4_WANDB_INTEGRATION.md](integration/PHASE1-4_WANDB_INTEGRATION.md) | ~300 | 10 min | Phase-specific W&B integration details |
| 19 | [v1-reference/implementation/WANDB_100_PERCENT_COMPLETE.md](../v1-reference/implementation/WANDB_100_PERCENT_COMPLETE.md) | ~600 | 5 min | **REFERENCE ONLY**: V1's 100% W&B integration (7,800+ metrics) |

**W&B Metrics Breakdown**:
- Phase 1: 37 metrics (3 model training)
- Phase 2: 370 metrics (50 generation evolution)
- Phase 3: 17 metrics (reasoning + coherence)
- Phase 4: 19 metrics (8× compression validation)
- Phase 5: 55 metrics (curriculum learning) ← **V2 DIFFERS FROM V1**
- Phase 6: 42 metrics (iterative A/B) ← **V2 DIFFERS FROM V1**
- Phase 7: 28 metrics (self-guided experts) ← **V2 DIFFERS FROM V1**
- Phase 8: 35 metrics (280× compression)
- **Total: 603 metrics**

**Key Takeaways**:
- ✅ W&B offline mode for local-first development
- ✅ 603 metrics tracked across all phases
- ✅ Cross-phase continuity tracking
- ✅ Dashboard configuration provided

---

### Phase 7: Dataset Specifications (20 minutes)

**Goal**: Understand data requirements for each phase

| # | Document | Lines | Time | Focus |
|---|----------|-------|------|-------|
| 20 | [docs/datasets/PHASE1_DATASET_SPECIFICATION.md](datasets/PHASE1_DATASET_SPECIFICATION.md) | ~300 | 10 min | Phase 1 training data (ARC-Easy, reasoning tasks) |
| 21 | [docs/datasets/PHASE3_DATA_GENERATION_GUIDE.md](datasets/PHASE3_DATA_GENERATION_GUIDE.md) | ~400 | 10 min | Phase 3 frontier model data generation ($100-200 cost) |

**Dataset Summary**:
- **Phase 1**: ARC-Easy, OBQA, reasoning datasets (500MB)
- **Phase 2**: Validation set from Phase 1 (200MB)
- **Phase 3**: Frontier model generated reasoning traces ($100-200 OpenRouter)
- **Phase 4**: 512-sample calibration set (1MB)
- **Phase 5**: OpenWebText + curriculum data (1GB) + frontier models ($600-800)
- **Phase 6**: Synthetic tool/persona data (500MB)
- **Phase 7**: Test set + frontier models ($150-250)
- **Phase 8**: Compression validation set (100MB)

**Key Takeaways**:
- ✅ Phase 1 datasets documented
- ✅ Phase 3 data generation with frontier models
- ✅ Total frontier model cost: $750-1050

---

### Phase 8: Phase-Specific Deep Dives (60-120 minutes)

**Goal**: Understand each phase's algorithms and implementation details

Read **LOGICAL_UNDERSTANDING.md** for each phase (conceptual synthesis), then **COMPLETE_GUIDE.md** (V1 implementation reference):

| Phase | LOGICAL_UNDERSTANDING | COMPLETE_GUIDE (V1 Reference) | Time | Priority |
|-------|----------------------|------------------------------|------|----------|
| **Phase 1** | [phases/phase1/LOGICAL_UNDERSTANDING.md](../phases/phase1/LOGICAL_UNDERSTANDING.md) | [phases/phase1/PHASE1_COMPLETE_GUIDE.md](../phases/phase1/PHASE1_COMPLETE_GUIDE.md) | 15 min | **HIGH** (Start here) |
| **Phase 2** | [phases/phase2/LOGICAL_UNDERSTANDING.md](../phases/phase2/LOGICAL_UNDERSTANDING.md) | [phases/phase2/PHASE2_COMPLETE_GUIDE.md](../phases/phase2/PHASE2_COMPLETE_GUIDE.md) | 15 min | **HIGH** (Proven V1 approach) |
| **Phase 3** | [phases/phase3/LOGICAL_UNDERSTANDING.md](../phases/phase3/LOGICAL_UNDERSTANDING.md) | [phases/phase3/PHASE3_COMPLETE_GUIDE.md](../phases/phase3/PHASE3_COMPLETE_GUIDE.md) | 15 min | **HIGH** (Proven V1 approach) |
| **Phase 4** | [phases/phase4/LOGICAL_UNDERSTANDING.md](../phases/phase4/LOGICAL_UNDERSTANDING.md) | [phases/phase4/PHASE4_COMPLETE_GUIDE.md](../phases/phase4/PHASE4_COMPLETE_GUIDE.md) | 15 min | **HIGH** (Proven V1 approach) |
| **Phase 5** | [phases/phase5/LOGICAL_UNDERSTANDING.md](../phases/phase5/LOGICAL_UNDERSTANDING.md) | [phases/phase5/PHASE5_COMPLETE_GUIDE.md](../phases/phase5/PHASE5_COMPLETE_GUIDE.md) | 15 min | **MEDIUM** (V2 redesign, reference V1) |
| **Phase 6** | [phases/phase6/LOGICAL_UNDERSTANDING.md](../phases/phase6/LOGICAL_UNDERSTANDING.md) | [phases/phase6/PHASE6_COMPLETE_GUIDE.md](../phases/phase6/PHASE6_COMPLETE_GUIDE.md) | 15 min | **MEDIUM** (V2 redesign, reference V1) |
| **Phase 7** | [phases/phase7/LOGICAL_UNDERSTANDING.md](../phases/phase7/LOGICAL_UNDERSTANDING.md) | [phases/phase7/PHASE7_COMPLETE_GUIDE.md](../phases/phase7/PHASE7_COMPLETE_GUIDE.md) | 15 min | **MEDIUM** (V2 redesign, reference V1) |
| **Phase 8** | [phases/phase8/LOGICAL_UNDERSTANDING.md](../phases/phase8/LOGICAL_UNDERSTANDING.md) | [phases/phase8/PHASE8_COMPLETE_GUIDE.md](../phases/phase8/PHASE8_COMPLETE_GUIDE.md) | 15 min | **LOW** (V2 production ready) |

**Reading Strategy**:
1. **LOGICAL_UNDERSTANDING.md**: Research synthesis, conceptual approach, key insights
2. **COMPLETE_GUIDE.md**: V1 implementation details (reference only for V2)
3. **Cross-reference with V2 spec**: Understand V1 vs V2 differences (especially Phases 5-8)

**Key Takeaways Per Phase**:
- **Phase 1**: TinyTitans architecture, HRM training, 25M params
- **Phase 2**: 6 merge techniques, binary pairing, 50 generations
- **Phase 3**: Quiet-STaR, thought generation, anti-theater detection
- **Phase 4**: BitNet 1.58-bit quantization, STE (not needed in V2 - dequantized)
- **Phase 5**: **V2 REDESIGN** - 7-stage curriculum learning (NOT BitNet + Grokfast)
- **Phase 6**: **V2 REDESIGN** - Iterative A/B optimization (NOT 9 personas)
- **Phase 7**: **V2 REDESIGN** - Self-guided experts (NOT manual ADAS)
- **Phase 8**: **V2 REDESIGN** - Production-ready triple compression

---

### Phase 9: V1 Modular Systems (Optional Reference - 30 minutes)

**Goal**: Understand production-ready V1 systems you can reuse in V2

| # | Document | Lines | Time | What It Provides |
|---|----------|-------|------|------------------|
| 22 | [v1-reference/implementation/MUGROKFAST_DEVELOPER_GUIDE.md](../v1-reference/implementation/MUGROKFAST_DEVELOPER_GUIDE.md) | 913 | 15 min | **REUSABLE**: Complete MuGrokfast optimizer (Grokfast × Muon) |
| 23 | [v1-reference/implementation/PROMPT_BAKING_INTEGRATION.md](../v1-reference/implementation/PROMPT_BAKING_INTEGRATION.md) | 1,339 | 15 min | **REUSABLE**: Complete prompt baking system (Phase 6) |

**Production-Ready V1 Systems** (Can be integrated into V2):
1. **MuGrokfast Optimizer**: Phase-specific presets, 10-50% speedup, STE mode
2. **Prompt Baking**: Half-baking, prompt pursuit, 5-minute baking, no decay over 30+ turns
3. **W&B Integration**: 7,800+ metrics, cross-phase continuity, dashboard configs
4. **OpenRouter Client**: Frontier model access, cost tracking, retry logic with fallback
5. **Phase Handoff Validation**: 99% reconstruction success, auto-cleanup, metadata versioning

**Key Takeaways**:
- ✅ V1 has 5 production-ready modular systems
- ✅ Can be integrated into V2 with minimal adaptation
- ✅ Saves 4-8 weeks of development time
- ✅ Focus: Read to understand approach, then adapt for V2

---

## Reading Order Summary

### Minimal Path (2 hours)
For developers who want to start coding ASAP:

1. README.md (15 min) - Project context
2. CLAUDE.md (10 min) - V1 vs V2 distinction
3. SPECIFICATION_INDEX.md (10 min) - Spec overview
4. AGENT_FORGE_V2_SPECIFICATION.md - Section 1 + Phase 1 (20 min)
5. PHASE5-8_V1_VS_V2_RECONCILIATION.md (15 min) - **CRITICAL** Phase 5-8 redesigns
6. ISSUE_RESOLUTION_MATRIX.md (20 min) - All issues resolved
7. PHASE4_TO_PHASE5_HANDOFF.md (15 min) - BitNet compatibility solution
8. DEPENDENCY_VERSIONS.md (10 min) - Pinned dependencies
9. phases/phase1/LOGICAL_UNDERSTANDING.md (15 min) - Start implementing Phase 1

**Total: 2 hours** → Ready to start Phase 1 implementation

---

### Recommended Path (4-6 hours)
For developers who want comprehensive understanding:

**Follow the full "Phase 1-9" reading order above** (4-6 hours total)

---

### Complete Path (8-12 hours)
For developers, architects, and project leads:

**Follow the full reading order** + Read all 8 phase LOGICAL_UNDERSTANDING.md + COMPLETE_GUIDE.md files

---

## Quick Reference Cheat Sheet

### V2 Differences from V1 (MEMORIZE THIS)

| Aspect | V1 | V2 |
|--------|----|----|
| **Purpose** | Server-based production | Local research platform |
| **Deployment** | FastAPI, Next.js, S3 | CLI, Jupyter, local filesystem |
| **Model Size** | Unspecified | **25M params** (Phase 1) |
| **Phase 5** | BitNet + Grokfast | **Curriculum Learning** (7 stages) |
| **Phase 6** | 9 pre-defined personas | **Iterative A/B Optimization** |
| **Phase 7** | Generic edge | **Self-Guided Experts** |
| **Phase 8** | Incomplete | **Production Ready** (280× compression) |
| **Cost** | $0 (no frontier models) | **$750-1050** (OpenRouter API) |
| **Timeline** | 26 weeks (refactor) | **16 weeks** (clean build) |

---

## File Locations Cheat Sheet

### Essential V2 Documents
```
docs/
├── v2-specification/
│   ├── AGENT_FORGE_V2_SPECIFICATION.md        # Phases 1-4, infrastructure
│   └── AGENT_FORGE_V2_SPECIFICATION_PART2.md  # Phases 5-8, W&B, UI, roadmap
├── v2-planning/
│   ├── PHASE5-8_V1_VS_V2_RECONCILIATION.md    # **CRITICAL** V2 redesigns
│   ├── ISSUE_RESOLUTION_MATRIX.md             # All 34 issues resolved
│   ├── PHASE4_TO_PHASE5_HANDOFF.md            # BitNet compatibility solution
│   ├── PHASE5_DREAM_CONSOLIDATION_IMPLEMENTATION.md  # Dream implementation
│   └── PREMORTEM_ANALYSIS.md                  # Infrastructure risks
├── integration/
│   ├── WANDB_INTEGRATION_GUIDE.md             # W&B setup
│   └── BITNET_COMPATIBILITY_ALL_PHASES.md     # BitNet cross-phase compatibility
├── datasets/
│   ├── PHASE1_DATASET_SPECIFICATION.md        # Phase 1 data
│   └── PHASE3_DATA_GENERATION_GUIDE.md        # Frontier model data gen
├── DEPENDENCY_VERSIONS.md                     # All pinned dependencies
└── PRE_COMMIT_SETUP_GUIDE.md                  # Code quality setup
```

### Phase-Specific Documents
```
phases/
├── phase1/
│   ├── LOGICAL_UNDERSTANDING.md               # Conceptual synthesis
│   ├── PHASE1_COMPLETE_GUIDE.md              # V1 implementation (reference)
│   └── TRM_TITANS_ARCHITECTURE.md            # TinyTitan architecture
├── phase2/
│   ├── LOGICAL_UNDERSTANDING.md
│   ├── PHASE2_COMPLETE_GUIDE.md
│   └── MERGE_TECHNIQUES_UPDATED.md           # 6 merge techniques
├── phase3/
│   ├── LOGICAL_UNDERSTANDING.md
│   └── PHASE3_COMPLETE_GUIDE.md
├── phase4/
│   ├── LOGICAL_UNDERSTANDING.md
│   └── PHASE4_COMPLETE_GUIDE.md
└── phase5-8/                                  # V2 redesigned phases
    ├── LOGICAL_UNDERSTANDING.md
    └── PHASEN_COMPLETE_GUIDE.md
```

### V1 Reference (Historical)
```
v1-reference/
├── implementation/
│   ├── MUGROKFAST_DEVELOPER_GUIDE.md         # Reusable optimizer
│   ├── PROMPT_BAKING_INTEGRATION.md          # Reusable baking system
│   └── WANDB_100_PERCENT_COMPLETE.md         # V1 W&B integration
└── analysis/
    └── LOOP1-COMPLETE-SUMMARY.md             # Complete V1 analysis
```

---

## Next Steps After Reading

### 1. Environment Setup (2 hours)
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Install dependencies (follow DEPENDENCY_VERSIONS.md)
pip install torch==2.1.0+cu118 transformers==4.38.0 wandb==0.16.3

# Setup OpenRouter API key (if using frontier models)
export OPENROUTER_API_KEY="your-key-here"
```

### 2. Phase 1 Implementation (Week 1-2)
- Implement TinyTitan architecture (25M params)
- Train 3 specialized models
- Validate models fit in 6GB VRAM
- Validate inference <100ms on local GPU

### 3. Continue Through Phases 2-8 (Week 3-16)
- Follow 16-week roadmap from AGENT_FORGE_V2_SPECIFICATION_PART2.md (Section 10)
- Reference phase LOGICAL_UNDERSTANDING.md for each phase
- Use V1 COMPLETE_GUIDE.md as implementation reference (adapt for V2)
- Track progress with W&B (603 metrics)

---

## Critical Warnings

### ⚠️ DO NOT:
1. **Refactor V1 code** - V2 is a clean build, not a refactor
2. **Follow V1 Phase 5-8 approaches exactly** - V2 has major redesigns
3. **Skip PHASE5-8_V1_VS_V2_RECONCILIATION.md** - Critical document explaining redesigns
4. **Skip ISSUE_RESOLUTION_MATRIX.md** - All 34 issues addressed here
5. **Save files to root directory** - Use subdirectories (docs/, src/, tests/)
6. **Commit backup files** - Use git branches only
7. **Hardcode API keys** - Use environment variables

### ✅ DO:
1. **Read PHASE5-8_V1_VS_V2_RECONCILIATION.md first** - Before implementing Phases 5-8
2. **Follow V2 specifications exactly** - Don't deviate without documenting
3. **Use pre-commit hooks** - NASA POT10, type checking, security checks
4. **Track all metrics in W&B** - 603 metrics across 8 phases
5. **Validate each phase** - Don't proceed to next phase until current validates
6. **Document deviations** - If you deviate from spec, document why
7. **Test on local GPU** - Validate all phases fit in 6GB VRAM

---

## FAQ

### Q: Should I read V1 documentation first?
**A**: No. Start with V2 specifications. V1 documentation is reference only for implementation ideas.

### Q: Do I need to read all 8 phase COMPLETE_GUIDE.md files?
**A**: Not immediately. Read LOGICAL_UNDERSTANDING.md for each phase, then COMPLETE_GUIDE.md when implementing that phase.

### Q: Which phases are different in V2?
**A**: Phases 5-8 have major redesigns. Read PHASE5-8_V1_VS_V2_RECONCILIATION.md to understand differences.

### Q: Can I skip the pre-commit hooks?
**A**: No. Code quality is enforced from day 1 (NASA POT10, type hints, security checks).

### Q: How do I know if I'm ready to start implementing?
**A**: After completing "Minimal Path" (2 hours), you can start Phase 1. For comprehensive understanding, complete "Recommended Path" (4-6 hours).

### Q: What's the single most important document?
**A**: **PHASE5-8_V1_VS_V2_RECONCILIATION.md** - Explains critical V2 redesigns for Phases 5-8.

---

## Document Status

**Last Updated**: 2025-10-16
**Version**: 1.0
**Maintained By**: Agent Forge V2 Team
**Review Status**: Ready for developer use

**Total Reading Time**:
- Minimal Path: 2 hours
- Recommended Path: 4-6 hours
- Complete Path: 8-12 hours

---

**Ready to build Agent Forge V2!** 🚀

**Start here**: [README.md](../README.md) → [CLAUDE.md](../CLAUDE.md) → [SPECIFICATION_INDEX.md](v2-specification/SPECIFICATION_INDEX.md)
