# Connascence Scan Summary

- Project: `the-agent-maker`
- Path: `D:\Projects\the-agent-maker`
- Git branch: `main`
- Git commit: `40fb3b5cdd270c016ef8e073ba669b848d0b58f4`
- Dirty before scan: `False`
- Scan succeeded: `True`
- Python files staged: `375`

## Commands Run
- `C:\Python312\python.exe -m analyzer C:\Users\17175\Desktop\_SCRATCH\connascence-portfolio-scan-2026-06-06\raw-results\the-agent-maker\mirror --format json --output C:\Users\17175\Desktop\_SCRATCH\connascence-portfolio-scan-2026-06-06\raw-results\the-agent-maker\connascence.raw.json --no-duplication --compliance-threshold 0 --max-god-objects 999999` (exit 0)
- `connascence_portfolio_runner.py generate-sarif-from-json D:\Projects\the-agent-maker\docs\connascence\scan-2026-06-06\connascence.json` (exit 0)
- `C:\Python312\python.exe -m analyzer.ast_engine --path C:\Users\17175\Desktop\_SCRATCH\connascence-portfolio-scan-2026-06-06\raw-results\the-agent-maker\mirror --analyzer god_object --output C:\Users\17175\Desktop\_SCRATCH\connascence-portfolio-scan-2026-06-06\raw-results\the-agent-maker\god-object.raw.json` (exit 0)

## Counts By Severity

- low: 16311
- medium: 1887
- critical: 120
- high: 71

## Counts By Type

- connascence_of_meaning: 14366
- CoV: 1862
- connascence_of_convention: 825
- CoP: 452
- connascence_of_type: 275
- connascence_of_execution: 268
- connascence_of_algorithm: 188
- god_object: 113
- connascence_of_timing: 24
- CoA: 16

## Top Files

- `D:\Projects\the-agent-maker\src\ui\pages\phase6_baking.py`: 733
- `D:\Projects\the-agent-maker\src\ui\pages\phase5_curriculum.py`: 710
- `D:\Projects\the-agent-maker\src\ui\pages\phase7_experts.py`: 673
- `D:\Projects\the-agent-maker\src\ui\pages\phase3_quietstar.py`: 544
- `D:\Projects\the-agent-maker\src\ui\pages\phase4_bitnet_upgraded.py`: 507
- `D:\Projects\the-agent-maker\src\ui\design_system.py`: 418
- `D:\Projects\the-agent-maker\src\ui\pages\phase2_evomerge.py`: 397
- `D:\Projects\the-agent-maker\docs\phases\phase8\final_compression.py`: 312
- `D:\Projects\the-agent-maker\src\cross_phase\validation\quality_validator.py`: 279
- `D:\Projects\the-agent-maker\src\ui\pages\wandb_monitor.py`: 275

## Top 10 Actionable Findings

1. `D:\Projects\the-agent-maker\src\phase8_compression\vptq.py:70` - Class 'VPTQCompressor' is a God Object (business_logic context): Lines of code (331) exceeds business_logic threshold (300); Low cohesion (0.13) in business logic class; Business logic class violates Single Responsibility Principle
2. `D:\Projects\the-agent-maker\src\phase8_compression\seedlm.py:44` - Class 'SeedLMCompressor' is a God Object (business_logic context): Low cohesion (0.31) in business logic class
3. `D:\Projects\the-agent-maker\src\phase8_compression\hypercompression.py:71` - Class 'HyperCompressor' is a God Object (business_logic context): Lines of code (305) exceeds business_logic threshold (300); Low cohesion (0.17) in business logic class; Business logic class violates Single Responsibility Principle
4. `D:\Projects\the-agent-maker\src\phase8_compression\compression_engine.py:68` - Class 'CompressionEngine' is a God Object (unknown context): Very low cohesion (0.10)
5. `D:\Projects\the-agent-maker\src\phase8_compression\benchmarks.py:79` - Class 'MMLUBenchmark' is a God Object (config context): Very low cohesion (0.10)
6. `D:\Projects\the-agent-maker\src\phase8_compression\benchmarks.py:339` - Class 'GSM8KBenchmark' is a God Object (config context): Very low cohesion (0.11)
7. `D:\Projects\the-agent-maker\src\phase8_compression\benchmarks.py:541` - Class 'BenchmarkSuite' is a God Object (unknown context): Very low cohesion (0.36)
8. `D:\Projects\the-agent-maker\src\phase7_experts\transformer2.py:171` - Class 'Transformer2' is a God Object (config context): Very low cohesion (0.17)
9. `D:\Projects\the-agent-maker\src\phase7_experts\svf_trainer.py:175` - Class 'REINFORCETrainer' is a God Object (config context): Very low cohesion (0.16)
10. `D:\Projects\the-agent-maker\src\phase7_experts\svf_trainer.py:316` - Class 'SVFTrainer' is a God Object (business_logic context): Low cohesion (0.11) in business logic class

## Tool Limitations

- Connascence currently analyzes Python files only; non-Python coupling is not covered.
- Source-bearing fields and literal values were stripped or redacted before writing artifacts.
- Excluded directories and sensitive data patterns were not staged into the scan mirror.

## Next Cleanup Recommendations

### 1. Quick Wins
- Add type annotations at public function boundaries with the highest CoT counts.
- Replace repeated or magic literals with named constants or configuration keys.

### 2. Medium Refactors
- Convert high-parameter functions to keyword-only APIs or parameter objects.
- Split complex functions and consolidate duplicated algorithmic branches.
- Start with the top files by violation count and keep each change behavior-preserving.

### 3. Large Architectural Work
- Split god objects into cohesive classes around stable domain responsibilities.
- Use module or service boundaries to isolate recurring high-count hotspots.
