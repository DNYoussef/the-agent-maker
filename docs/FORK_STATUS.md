# FORK STATUS: meta_calculus vs meta-calculus-toolkit

**Date**: 2026-06-12 (plan P6)
**Upstream**: `C:\Users\17175\Desktop\meta-calculus-toolkit` (GitHub:
DNYoussef/meta-calculus-portfolio), certified against upstream commit
`80bdc58`.
**Downstream**: `src/cross_phase/meta_calculus/` (this repo), created
Dec 2025 as an ML-focused ADAPTATION.

## Finding: the fork is semantic, not file-level

None of this module's files (`k_formula.py`, `bigeometric.py`,
`spectral_gap.py`, `meta_grokfast.py`, `moo_bridge.py`) exist in the
toolkit - the Dec 2025 integration re-implemented the shared MATHEMATICS
for the training pipeline rather than copying files. What is actually
shared, and how drift is prevented:

| Shared semantics | Toolkit home | This repo | Drift guard |
|---|---|---|---|
| k(L) = -0.0137*log10(L) + 0.1593 | `k_cosmology.SpatialKParams` | `k_formula.K_SLOPE/K_INTERCEPT` | golden vectors `k_of_L` table, asserted in BOTH repos vs the Lean `kOf` form |
| Epistemic status of those constants | `k_cosmology.K_SPATIAL_PROVENANCE` (fit currently NOT regenerable, AUDIT A11) | consumers should read the note here | this file + the provenance dict |
| Bigeometric math | `core/derivatives.BigeometricDerivative` (a derivative OPERATOR) | `bigeometric.BigeometricTransform` (a gradient TRANSFORM, `g*\|g\|^(2k-1)` - agent-maker-native, no upstream twin) | both pinned to gcc-lean-lab: `bgDerivMul_powFlow` vectors (toolkit) and `bgTransform` vectors (here) |
| Pareto-front discipline | `multiscale_moo.py` | `moo_bridge.py` | `scalarized_min_is_pareto` / `middle_not_scalar_min` (Lean); `select_from_pareto` audit logging (here) |

Agent-maker-native (NO upstream port - zero upstream call sites, R4):
the k-guardrail (`BigeometricConfig.k_min/k_max`, justified by Lean
`bgTransform_contracts` family) and the MDL objective
(`description_length_bits`). If the toolkit ever grows an optimizer-facing
gradient transform, port the guardrail then, citing the same theorems.

## Sync rule

Shared SEMANTICS (constants, formulas, claim wording) change in the
TOOLKIT first, then here, in that order, and every such change must keep
`tests/validation/test_golden_vectors.py` (here) and
`tests/test_golden_vectors.py` (toolkit) green against a single
regenerated vector file (`lake exe gccvectors` in gcc-lean-lab; update
EXPECTED_SHA256 in both consumers together). ML-specific adaptation
(optimizer wiring, phase configs, guardrails, MDL) is owned here and needs
no upstream mirror. When touching shared semantics, update this file's
certified-against commit.
