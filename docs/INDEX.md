# Agent Forge V2 - Documentation Index

**Last Updated**: 2025-10-16

This document provides a comprehensive navigation guide to all Agent Forge V2 documentation.

---

## Quick Links

- **[V2 Specification](v2-specification/)** - Complete technical specification
- **[V2 Planning](v2-planning/)** - Implementation plans and analysis
- **[Integration Guides](integration/)** - UI, W&B, and system integration
- **[Dataset Documentation](datasets/)** - Data generation and specifications
- **[GraphViz Flows](graphviz/)** - Visual process documentation
- **[V1 Reference](../v1-reference/README.md)** - Historical V1 documentation

---

## V2 Documentation Structure

### V2 Specification
**Location**: `docs/v2-specification/`

Technical specifications for V2 rebuild (if present)

### V2 Planning & Analysis
**Location**: `docs/v2-planning/`

Implementation plans, risk analysis, and project management documents:
- Phase 1-4 comprehensive plans
- Implementation checklists
- Premortem analysis (V2-specific risks)
- Master indexes
- UI specifications
- Consistency analysis

### Integration Guides
**Location**: `docs/integration/`

System integration documentation:
- **UI Integration**
  - UI_INTEGRATION_README.md
  - PHASES_UI_INTEGRATION_GUIDE.md
  - UI_INTEGRATION_COMPLETE.md
- **W&B Integration**
  - WANDB_INTEGRATION_GUIDE.md
  - PHASE1-4_WANDB_INTEGRATION.md

### Dataset Documentation
**Location**: `docs/datasets/`

Data generation and dataset specifications:
- **Phase 1 Datasets**
  - PHASE1_DATASET_INTEGRATION_SUMMARY.md
  - PHASE1_DATASET_SPECIFICATION.md
- **Phase 3 Data Generation**
  - PHASE3_DATA_GENERATION_GUIDE.md
  - PHASE3_DATA_GENERATOR_UPDATE_V2.md
  - PHASE3_PROMPT_BAKING_CORRECTED.md

### Visual Documentation
**Location**: `docs/graphviz/`

Process flow diagrams (GraphViz .dot files):
- agent-forge-master-flow.dot - Master workflow
- phase-integration-flow.dot - Phase integration flows
- GRAPHVIZ_UPDATE_SUMMARY.md - Documentation of GraphViz updates

### V2 Implementation Tracking
**Location**: `docs/v2-implementation/`

Implementation status and progress tracking:
- REORGANIZATION_COMPLETE.md - Repository reorganization status

---

## V1 Reference (Historical)

**⚠️ Important**: V1 documentation is for reference only. V2 is a ground-up rebuild, NOT a refactor of V1 code.

**Location**: `v1-reference/`

### V1 Analysis
**Location**: `v1-reference/analysis/`

Complete analysis of V1 implementation:
- **LOOP1-COMPLETE-SUMMARY.md** - Comprehensive V1 analysis (93% confidence)
- **FINDINGS-AGGREGATION.md** - Research findings aggregation
- **architecture-analysis.md** - V1 architecture issues and problems
- **code-quality-report.md** - V1 code quality (201 backups, 8 God objects)
- **phase-methodology-analysis.md** - Phase effectiveness analysis
- **COMPLETE_PHASE_DOCUMENTATION_INDEX.md** - V1 documentation index

### V1 Planning (Refactoring Plans - NOT V2)
**Location**: `v1-reference/planning/`

V1 refactoring plans (historical):
- PLAN-v1.md, PLAN-v2.md, PLAN-v3.md - V1 refactoring plan iterations
- PREMORTEM-v1.md, PREMORTEM-v2.md, PREMORTEM-v3.md - V1 risk analysis iterations

**Note**: These plans addressed V1 technical debt (God objects, backup files, etc.). V2 builds clean.

### V1 Implementation Documentation
**Location**: `v1-reference/implementation/`

V1 implementation details (reference only):
- API and backend architecture documents
- Frontend architecture
- W&B integration implementation
- WebSocket implementation
- Test reports and success reports
- Grokking, prompt baking, TRM implementation
- Simulation vs real analysis
- Generation 0 models documentation

### V1 Architecture
**Location**: `v1-reference/architecture/`

V1 system architecture documentation:
- API architecture and documentation
- Backend API summary
- Frontend architecture
- Complete system architecture

### Research Papers
**Location**: `v1-reference/research-papers/`

Academic papers referenced in V1:
- Grokfast_Accelerated_Grokking.pdf (Phase 5)
- Grokking_Generalization_Beyond_Overfitting.pdf (Phase 5)
- Prompt_Baking.pdf (Phase 6)
- 2505.23737v1.pdf (Additional research)

---

## Phase-Specific Documentation

**Location**: `phases/phase1/` through `phases/phase8/`

Each phase contains comprehensive documentation:

### Phase 1: Cognate - TinyTitans (25M params)
- LOGICAL_UNDERSTANDING.md - Conceptual synthesis
- PHASE1_COMPLETE_GUIDE.md - Complete implementation guide
- TRM_TITANS_ARCHITECTURE.md - TRM architecture
- PREMORTEM_CHECKLIST.md - Risk assessment
- README.md - Phase overview

### Phase 2: EvoMerge - Evolutionary Optimization
- LOGICAL_UNDERSTANDING.md
- PHASE2_COMPLETE_GUIDE.md
- MERGE_TECHNIQUES_UPDATED.md - 6 merge techniques
- BINARY_PAIRING_STRATEGY.md
- README.md

### Phase 3: Quiet-STaR - Reasoning Enhancement
- LOGICAL_UNDERSTANDING.md
- PHASE3_COMPLETE_GUIDE.md
- PHASE3-QUIET-STAR-VISUALIZATION.md
- INTEGRATION_README.md
- README.md

### Phase 4: BitNet - 1.58-bit Compression
- LOGICAL_UNDERSTANDING.md
- PHASE4_COMPLETE_GUIDE.md
- production-readiness-assessment.md
- FINAL-PRODUCTION-CI-CD-INTEGRATION-READINESS.md
- README.md

### Phase 5: Forge Training - Grokfast Acceleration
- LOGICAL_UNDERSTANDING.md
- PHASE5_COMPLETE_GUIDE.md
- phase5-training-architecture.md
- PHASE5-VALIDATION-ROLLOUT-SWARM-INIT.md
- README.md

### Phase 6: Tool & Persona Baking
- LOGICAL_UNDERSTANDING.md
- PHASE6_COMPLETE_GUIDE.md
- PHASE6_INTEGRATION_ARCHITECTURE.md
- phase6_completion_report.md
- PHASE6_PRODUCTION_VALIDATION_REPORT.md
- PHASE6_VALIDATION_EXECUTIVE_SUMMARY.md

### Phase 7: Generic Edge Deployment
- LOGICAL_UNDERSTANDING.md
- PHASE7_COMPLETE_GUIDE.md
- ADAS_ARCHITECTURE.md (V1 reference)
- HONEST_CAPABILITIES_REPORT.md
- THEATER_ELIMINATION_REPORT.md
- README.md

### Phase 8: Final Compression
- LOGICAL_UNDERSTANDING.md
- PHASE8_COMPLETE_GUIDE.md

---

## Root-Level Files

- **[README.md](../README.md)** - Project overview and introduction
- **[CLAUDE.md](../CLAUDE.md)** - Instructions for Claude Code (AI assistant configuration)
- **[FILE_MANIFEST.txt](../FILE_MANIFEST.txt)** - Complete file index

---

## Navigation Guide by Role

### For V2 Developers (Implementing V2)
1. **Start**: [CLAUDE.md](../CLAUDE.md) - Understand V1 vs V2 distinction
2. **Review**: [v2-planning/](v2-planning/) - Implementation plans and checklists
3. **Study**: `phases/phaseN/LOGICAL_UNDERSTANDING.md` - Phase concepts
4. **Reference**: [integration/](integration/) - Integration patterns
5. **Data**: [datasets/](datasets/) - Dataset specifications

### For V1 Researchers (Understanding Original)
1. **Overview**: [v1-reference/README.md](../v1-reference/README.md) - V1 introduction
2. **Analysis**: [v1-reference/analysis/LOOP1-COMPLETE-SUMMARY.md](../v1-reference/analysis/LOOP1-COMPLETE-SUMMARY.md)
3. **Architecture**: [v1-reference/architecture/](../v1-reference/architecture/) - V1 system design
4. **Implementation**: [v1-reference/implementation/](../v1-reference/implementation/) - V1 code docs

### For Project Managers
1. **Overview**: [README.md](../README.md) - High-level project summary
2. **Planning**: [v2-planning/](v2-planning/) - Implementation timeline and risk analysis
3. **Status**: [v2-implementation/REORGANIZATION_COMPLETE.md](v2-implementation/REORGANIZATION_COMPLETE.md)
4. **Reference**: [v1-reference/analysis/](../v1-reference/analysis/) - Historical context

### For Researchers (Academic)
1. **Papers**: [v1-reference/research-papers/](../v1-reference/research-papers/) - Academic papers
2. **Understanding**: `phases/phaseN/LOGICAL_UNDERSTANDING.md` - Research synthesis
3. **Implementation**: Phase-specific COMPLETE_GUIDE.md files - Applied research

---

## File Organization Rules

**NEVER save files to the root directory**. Use these subdirectories:

### Documentation
- `docs/v2-planning/` - V2 planning documents
- `docs/v2-specification/` - V2 technical specifications
- `docs/v2-implementation/` - V2 implementation tracking
- `docs/integration/` - Integration guides
- `docs/datasets/` - Dataset documentation
- `docs/graphviz/` - Process flow diagrams

### Implementation (Future)
- `src/` - V2 source code (when implementation begins)
- `tests/` - V2 test files
- `examples/` - V2 example notebooks/scripts
- `config/` - Configuration files
- `scripts/` - Utility scripts

### Reference
- `v1-reference/` - All V1 documentation (analysis, planning, implementation)
- `phases/phaseN/` - Phase-specific documentation (V1 + V2)

---

## Quick Reference Commands

### Check Documentation Structure
```bash
# View this index
cat docs/INDEX.md

# List V2 planning docs
ls docs/v2-planning/

# List integration guides
ls docs/integration/

# List phase documentation
ls phases/phase1/
```

### Navigate to Key Documents
```bash
# V2 planning
cd docs/v2-planning
ls -la

# V1 analysis (reference)
cd v1-reference/analysis
cat LOOP1-COMPLETE-SUMMARY.md

# Phase documentation
cd phases/phase5
cat LOGICAL_UNDERSTANDING.md

# Integration guides
cd docs/integration
cat WANDB_INTEGRATION_GUIDE.md
```

---

## Document Status

### Ready for Use
- ✅ **V2 Planning Documentation** - Implementation plans and checklists organized
- ✅ **Integration Guides** - UI and W&B integration documented
- ✅ **Dataset Documentation** - Data specifications organized
- ✅ **Phase Documentation** - All 8 phases documented with LOGICAL_UNDERSTANDING
- ✅ **V1 Reference** - Complete V1 analysis and documentation archived
- ✅ **GraphViz Flows** - Visual process documentation available

### In Progress
- 🔄 **V2 Implementation** - Not yet started (planned for 16-week timeline)
- 🔄 **V2 Specification** - To be created based on planning docs

---

## Version History

- **v2.0** (2025-10-16): Complete reorganization, updated structure and navigation
- **v1.0** (2025-10-15): Initial index after project reorganization

---

**Need Help?**
- Project Overview: [README.md](../README.md)
- Development Setup: [CLAUDE.md](../CLAUDE.md)
- V2 Planning: [v2-planning/](v2-planning/)
- V1 Reference: [v1-reference/README.md](../v1-reference/README.md)
