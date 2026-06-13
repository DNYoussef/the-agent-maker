# FORK STATUS: meta_calculus vs meta-calculus-toolkit

**Date**: 2026-06-12 (plan P6; corrections sync same evening)
**Upstream**: `C:\Users\17175\Desktop\meta-calculus-toolkit` (GitHub:
DNYoussef/meta-calculus-portfolio), certified against upstream commit
`3678210`.
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

## Upstream corrections sync (2026-06-12 evening, toolkit fleet sweep)

The toolkit ran every simulation + test and shipped corrections (its
`docs/HONEST-STATUS.md`, commits `88269ec`..`3678210`). Audit of this
module against each:

- **RETRACTED upstream: "Bigeometric achieves 2nd order convergence."**
  The old scheme was central difference in disguise (converting D_BG back
  to an additive derivative reproduces u' exactly); meta-k schemes measure
  ~0.84-0.95 (first-order-like). Fork impact: NONE in code -
  `BigeometricTransform` here is a gradient transform pinned to the Lean
  `bgTransform` vectors, not an advection scheme. Wording rule: never cite
  bigeometric convergence order as a feature.
- **Shu-Osher corrected**: upstream `cfd_nnc_comprehensive` silently
  reported NaN (forward-Euler + MUSCL instability); fixed with SSP-RK2,
  NNC limiter now measures 9.7% over superbee (Sod a tie). Fork impact:
  none (no CFD claims here).
- **`BigeometricDerivative` scalar-shape crash fixed upstream** (boolean
  indexing assumed vectorized f). Fork impact: none - `bigeometric.py`
  here was audited and has no such pattern.
- **GlobalMOO wiring**: upstream's `ObjectiveType.PERCENT`-toward-zero was
  degenerate; with native MINIMIZE the equal-budget baseline inverted
  (GlobalMOO 99.42% better chi2 than pymoo at 200 evals/seed 20260612).
  Fork impact: none - `moo_bridge.py` is pymoo-only, no GlobalMOO SDK
  wiring to mis-wire.
- **12-sim master suite disclosed as illustrative** (sim 1 hardcodes the
  k*=0.14 it reports; five sims hardcode accuracy numbers) and **vacuum
  suppression fails its own physics targets**. Fork impact: do not cite
  `run_all_simulations.py` outputs as evidence; cite the dedicated
  verification scripts per HONEST-STATUS.
