# Project Organization Plan - Agent Forge V2

**Date**: 2025-10-15
**Purpose**: Reorganize project structure for clarity between V1 (reference) and V2 (implementation)

---

## Current Issues

1. **Root directory cluttered**: 14 markdown files + 3 PDFs in root
2. **V1/V2 confusion**: V1 analysis mixed with V2 specifications
3. **Research papers scattered**: PDFs in root instead of organized folder
4. **Inconsistent naming**: Some docs have version suffixes (v1, v2, v3), others don't

---

## Proposed Structure

```
the agent maker/
├── README.md                          # Updated project overview
├── CLAUDE.md                          # Keep in root (IDE configuration)
├── FILE_MANIFEST.txt                  # Keep in root (project index)
│
├── docs/                              # V2 Documentation (NEW)
│   ├── v2-specification/              # V2 Specifications
│   │   ├── AGENT_FORGE_V2_SPECIFICATION.md
│   │   ├── AGENT_FORGE_V2_SPECIFICATION_PART2.md
│   │   ├── SPECIFICATION_INDEX.md
│   │   ├── SPECIFICATION_SUMMARY.md
│   │   └── SPECIFICATION_UPDATES_APPLIED.md
│   ├── v2-planning/                   # V2 Planning Documents
│   │   ├── PLAN-V2-BUILD.md
│   │   ├── PREMORTEM-V2-BUILD.md
│   │   └── PREMORTEM_ANALYSIS.md
│   ├── graphviz/                      # Process flow diagrams (keep location)
│   │   └── (existing .dot files)
│   └── INDEX.md                       # Navigation guide (NEW)
│
├── v1-reference/                      # V1 Historical Documentation (NEW)
│   ├── analysis/                      # V1 Analysis Documents
│   │   ├── LOOP1-COMPLETE-SUMMARY.md
│   │   ├── FINDINGS-AGGREGATION.md
│   │   ├── architecture-analysis.md
│   │   ├── code-quality-report.md
│   │   ├── phase-methodology-analysis.md
│   │   └── COMPLETE_PHASE_DOCUMENTATION_INDEX.md
│   ├── planning/                      # V1 Planning Iterations
│   │   ├── PLAN-v1.md
│   │   ├── PLAN-v2.md
│   │   ├── PLAN-v3.md
│   │   ├── PREMORTEM-v1.md
│   │   ├── PREMORTEM-v2.md
│   │   └── PREMORTEM-v3.md
│   └── README.md                      # V1 reference guide (NEW)
│
├── research-papers/                   # Research Papers (NEW)
│   ├── phase1-cognate/
│   │   └── (phase 1 papers - move from phases/phase1/)
│   ├── phase2-evomerge/
│   │   └── (phase 2 papers)
│   ├── phase3-quietstar/
│   │   └── (phase 3 papers)
│   ├── phase5-grokfast/
│   │   ├── Grokfast_Accelerated_Grokking_by_Amplifying_Slow_Gradients.pdf
│   │   └── GROKKING_GENERALIZATION_BEYOND_OVERFITTING_ON_SMALL_ALGORITHMIC_DATASETS.pdf
│   ├── phase6-baking/
│   │   └── Prompt_Baking.pdf
│   └── README.md                      # Paper index (NEW)
│
├── phases/                            # Phase-Specific Documentation
│   ├── phase1/                        # Cognate
│   │   ├── LOGICAL_UNDERSTANDING.md   # (keep)
│   │   ├── graphviz/                  # (keep)
│   │   └── (V1 implementation docs)
│   ├── phase2/                        # EvoMerge
│   ├── phase3/                        # Quiet-STaR
│   ├── phase4/                        # BitNet
│   ├── phase5/                        # Forge Training
│   ├── phase6/                        # Tool Baking
│   ├── phase7/                        # Edge Deployment
│   └── phase8/                        # Final Compression
│
└── cross-phase/                       # Cross-Phase Analysis (keep location)
    └── (existing cross-phase docs)
```

---

## File Movement Plan

### Phase 1: Create New Directories
```bash
mkdir -p "docs/v2-specification"
mkdir -p "docs/v2-planning"
mkdir -p "v1-reference/analysis"
mkdir -p "v1-reference/planning"
mkdir -p "research-papers/phase1-cognate"
mkdir -p "research-papers/phase2-evomerge"
mkdir -p "research-papers/phase3-quietstar"
mkdir -p "research-papers/phase4-bitnet"
mkdir -p "research-papers/phase5-grokfast"
mkdir -p "research-papers/phase6-baking"
mkdir -p "research-papers/phase7-adas"
mkdir -p "research-papers/phase8-compression"
```

### Phase 2: Move V2 Specifications
```bash
# From docs/ to docs/v2-specification/
mv "docs/AGENT_FORGE_V2_SPECIFICATION.md" "docs/v2-specification/"
mv "docs/AGENT_FORGE_V2_SPECIFICATION_PART2.md" "docs/v2-specification/"
mv "docs/SPECIFICATION_INDEX.md" "docs/v2-specification/"
mv "docs/SPECIFICATION_SUMMARY.md" "docs/v2-specification/"
mv "docs/SPECIFICATION_UPDATES_APPLIED.md" "docs/v2-specification/"
```

### Phase 3: Move V2 Planning
```bash
# From docs/ to docs/v2-planning/
mv "docs/PLAN-V2-BUILD.md" "docs/v2-planning/"
mv "docs/PREMORTEM-V2-BUILD.md" "docs/v2-planning/"
mv "docs/PREMORTEM_ANALYSIS.md" "docs/v2-planning/"
```

### Phase 4: Move V1 Analysis Documents
```bash
# From root to v1-reference/analysis/
mv "LOOP1-COMPLETE-SUMMARY.md" "v1-reference/analysis/"
mv "FINDINGS-AGGREGATION.md" "v1-reference/analysis/"
mv "architecture-analysis.md" "v1-reference/analysis/"
mv "code-quality-report.md" "v1-reference/analysis/"
mv "phase-methodology-analysis.md" "v1-reference/analysis/"
mv "COMPLETE_PHASE_DOCUMENTATION_INDEX.md" "v1-reference/analysis/"
```

### Phase 5: Move V1 Planning Documents
```bash
# From root to v1-reference/planning/
mv "PLAN-v1.md" "v1-reference/planning/"
mv "PLAN-v2.md" "v1-reference/planning/"
mv "PLAN-v3.md" "v1-reference/planning/"
mv "PREMORTEM-v1.md" "v1-reference/planning/"
mv "PREMORTEM-v2.md" "v1-reference/planning/"
mv "PREMORTEM-v3.md" "v1-reference/planning/"
```

### Phase 6: Move Research Papers
```bash
# From root to research-papers/phase5-grokfast/
mv "Grokfast_ Accelerated Grokking by Amplifying Slow Gradients.pdf" "research-papers/phase5-grokfast/Grokfast_Accelerated_Grokking.pdf"
mv "GROKKING_ GENERALIZATION BEYOND OVERFIT- TING ON SMALL ALGORITHMIC DATASETS.pdf" "research-papers/phase5-grokfast/Grokking_Generalization_Beyond_Overfitting.pdf"
mv "Prompt Baking.pdf" "research-papers/phase6-baking/Prompt_Baking.pdf"
```

---

## Files Staying in Root

Keep these files in root for IDE/tooling purposes:
- ✅ `README.md` (project overview)
- ✅ `CLAUDE.md` (IDE configuration)
- ✅ `FILE_MANIFEST.txt` (project index)
- ✅ `.claude/` (IDE settings folder)

---

## Documents to Create

1. **docs/INDEX.md** - Master navigation guide
2. **v1-reference/README.md** - V1 reference overview
3. **research-papers/README.md** - Paper index with phase mapping
4. **Updated README.md** - Reflect new structure

---

## Benefits

1. **Clear V1/V2 Separation**: No confusion about what's reference vs implementation
2. **Clean Root**: Only essential config files remain
3. **Organized Research**: Papers grouped by phase and topic
4. **Better Navigation**: Logical folder hierarchy with index documents
5. **Maintainability**: Easier to find and update documents

---

## Rollback Plan

If organization needs to be reverted:
```bash
# All files still in repository, can use git to revert moves
git checkout HEAD -- .
```

---

## Next Steps

1. ✅ Create folder structure
2. ✅ Move files according to plan
3. ✅ Create new index/README documents
4. ✅ Update README.md
5. ✅ Test all cross-references still work
