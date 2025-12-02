# Agent Forge V2 - Graphviz Diagrams

This directory contains **8 Graphviz (.dot) diagrams** that visualize the key processes, workflows, and architecture of Agent Forge V2.

---

## 🚨 Installation Issue Detected

**Chocolatey installation failed** due to lock file and permissions issues.

**Solution**: Please see **[INSTALL_GRAPHVIZ.md](INSTALL_GRAPHVIZ.md)** for manual installation instructions.

**Quick Fix** (Run PowerShell as Administrator):
```powershell
cd "c:\Users\17175\Desktop\the agent maker\docs\graphviz"
.\install_graphviz_admin.ps1
```

---

## Quick Start

### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install graphviz
```

**macOS:**
```bash
brew install graphviz
```

**Windows:**
```bash
choco install graphviz
```

Or download from: https://graphviz.org/download/

### Rendering All Diagrams

**Linux/macOS:**
```bash
cd docs/graphviz
chmod +x render_all.sh
./render_all.sh
```

**Windows:**
```cmd
cd docs\graphviz
render_all.bat
```

### Rendering Individual Diagrams

```bash
# PNG (recommended for documentation)
dot -Tpng cicd-pipeline-flow.dot -o cicd-pipeline-flow.png

# SVG (scalable, best for web)
dot -Tsvg cicd-pipeline-flow.dot -o cicd-pipeline-flow.svg

# PDF (print-ready)
dot -Tpdf cicd-pipeline-flow.dot -o cicd-pipeline-flow.pdf
```

## Diagram Files

### 1. cicd-pipeline-flow.dot
**Visualizes**: Complete CI/CD automation workflow
**Source**: WEEK_11-12_CICD_SUMMARY.md
**Nodes**: Triggers, CI jobs, quality gates, docs deployment, release workflow

### 2. testing-infrastructure-flow.dot
**Visualizes**: Testing and quality assurance setup
**Source**: WEEK_9-10_TESTING_SUMMARY.md
**Nodes**: Pre-commit hooks (10), test suite (47 tests), coverage analysis, quality tools

### 3. week-1-12-timeline.dot
**Visualizes**: 16-week project timeline and milestones
**Source**: WEEKS_1-10_COMPLETE.md, WEEK_11-12_CICD_SUMMARY.md
**Nodes**: Weeks 1-6 (core), 7-8 (UI), 9-10 (testing), 11-12 (CI/CD), 13-16 (planned)

### 4. 8-phase-pipeline-complete.dot
**Visualizes**: Complete 8-phase AI agent creation pipeline
**Source**: Architecture documentation, WEEK_11-12_CICD_SUMMARY.md
**Nodes**: All 8 phases from Cognate (Phase 1) to Final Compression (Phase 8) with time/size annotations

### 5. infrastructure-systems.dot
**Visualizes**: 6 core infrastructure systems and interactions
**Source**: INFRASTRUCTURE_SUMMARY.md
**Nodes**: Pipeline orchestrator, model registry, MuGrokfast, prompt baking, W&B, utils

### 6. quality-assurance-flow.dot
**Visualizes**: Complete QA process from code to production
**Source**: WEEK_9-10_TESTING_SUMMARY.md, WEEK_11-12_CICD_SUMMARY.md
**Nodes**: Local pre-commit, CI pipeline, quality gate, documentation, production, feedback loop

### 7. agent-forge-master-flow.dot (Existing)
**Visualizes**: High-level project workflow
**Source**: Original master workflow

### 8. phase-integration-flow.dot (Existing)
**Visualizes**: Phase-to-phase handoff validation
**Source**: Phase handoff documentation

## Color Scheme

| Color | Usage |
|-------|-------|
| `lightblue` | Entry points, start nodes |
| `lightgreen` | Success states, outputs |
| `lightyellow` | Process nodes, jobs |
| `orange` | Decision points, quality gates |
| `red` | Failure states, blocked |
| `gold` | Final outputs, production |
| `lightcyan` | Sub-processes |
| `lavender`, `mistyrose`, etc. | Phase-specific clusters |

## Integration with Documentation

### Markdown
```markdown
![CI/CD Pipeline](docs/graphviz/cicd-pipeline-flow.png)
```

### Sphinx (reStructuredText)
```rst
.. image:: ../graphviz/cicd-pipeline-flow.png
   :alt: CI/CD Pipeline Flow
   :align: center
```

## File Structure

```
docs/graphviz/
├── README.md                           # This file
├── GRAPHVIZ_DIAGRAMS_SUMMARY.md       # Complete documentation
├── render_all.sh                       # Linux/macOS batch renderer
├── render_all.bat                      # Windows batch renderer
│
├── cicd-pipeline-flow.dot              # CI/CD workflow
├── testing-infrastructure-flow.dot     # Testing workflow
├── week-1-12-timeline.dot              # Project timeline
├── 8-phase-pipeline-complete.dot       # 8-phase pipeline
├── infrastructure-systems.dot          # Core systems
├── quality-assurance-flow.dot          # QA process
├── agent-forge-master-flow.dot         # Master workflow (existing)
└── phase-integration-flow.dot          # Phase handoff (existing)
```

## Troubleshooting

### "dot: command not found"
Graphviz is not installed or not in PATH. Install using instructions above.

### "Error: syntax error in line X"
Check the .dot file for syntax errors. Graphviz is sensitive to:
- Unclosed quotes
- Missing semicolons
- Invalid node names (use quotes for special chars)

### Diagram too large/small
Adjust DPI:
```bash
dot -Tpng -Gdpi=150 diagram.dot -o diagram.png
```

### Layout issues
Try different layout engines:
```bash
# Default (dot) - hierarchical
dot -Tpng diagram.dot -o diagram.png

# Circular layout
circo -Tpng diagram.dot -o diagram.png

# Force-directed
neato -Tpng diagram.dot -o diagram.png

# Radial
twopi -Tpng diagram.dot -o diagram.png
```

## References

- **Graphviz Official Documentation**: https://graphviz.org/documentation/
- **DOT Language Guide**: https://graphviz.org/doc/info/lang.html
- **Node Shapes**: https://graphviz.org/doc/info/shapes.html
- **Color Names**: https://graphviz.org/doc/info/colors.html

---

**Created**: 2025-10-16
**Total Diagrams**: 8 (.dot files)
**Status**: ✅ Ready to render
