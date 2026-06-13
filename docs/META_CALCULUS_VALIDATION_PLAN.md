# META-CALCULUS VALIDATION PLAN

**Version**: 2.0.0 (v1 targeted the wrong repo; this version is re-anchored
after a deep audit of the real toolkit)
**Date**: 2026-06-11
**Author**: Planning session (Claude Code), informed by the GCC Lean relay
(`D:\Projects\gcc-lean-lab\docs\`), a deep audit of
`C:\Users\17175\Desktop\meta-calculus-toolkit`, and a deep audit of
`D:\Projects\the-agent-maker\src\cross_phase\meta_calculus`.
**Executor**: any competent coding agent. Every work item names exact files,
exact changes, failure modes, and tests that can fail for real reasons.

---

## 0. Context: three repos, one doctrine, one fork

| Repo | Role | State |
|---|---|---|
| `C:\Users\17175\Desktop\meta-calculus-toolkit` | **UPSTREAM SOURCE OF TRUTH.** Standalone research toolkit: 39,125 lines in `meta_calculus/` (69 modules), 347 tests passing, 67 simulations, 322-file reference library, live portfolio site on Railway (`meta-calculus-portfolio-production.up.railway.app`), GitHub `DNYoussef/meta-calculus-portfolio` (private). | ~60% complete. Core math 85%. Has its own honesty layer: `HONEST-STATUS.md` audit. |
| `D:\Projects\the-agent-maker\src\cross_phase\meta_calculus` | **DOWNSTREAM CONSUMER.** Dec 2025 stripped snapshot (7 core files) applying k(L) + bigeometric transforms to the 8-phase ML training pipeline. Has diverged from upstream (both sides have post-Dec fixes). | ~70% complete for its scope. |
| `D:\Projects\gcc-lean-lab` | **FORMAL SPEC LAYER.** Lean 4.28.0 + Mathlib v4.28.0, all green (commit `fae99d8`). Already proves the toolkit's foundational layer (see 0.1). | GREEN. |

### 0.1 What is ALREADY proved (do not re-plan it)

The gcc-lean-lab capstone unknowingly formalized the toolkit's foundation:

| Toolkit concept (file) | Published source | Lean theorem (gcc-lean-lab) |
|---|---|---|
| Generated arithmetic `x (+) y = f^-1(f(x)+f(y))` (`core/generators.py`, Czachor Eq. 18 quoted at `core/derivatives.py:120-145`) | Czachor 2016, *Relativity of Arithmetic*; Burgin-Czachor 2020 Ch. 2 | `gadd/gmul/gsub/gzero/gone` + transported laws + `razor_*` (CanonicalCore), `*_recoordRing` certificates |
| Discrete meta-FTC | Grossman 1981; Burgin-Czachor 2020 Ch. 4 | `gFTC` (sum of g-differences telescopes, all generators at once) |
| Geometric straight lines / exponentials | Grossman-Katz 1972 Ch. 3 | `const_gdiff_iff`, `geometricLine_logCoord`, `logRate_geometricLine` |
| Log/logit chart instruments with safety-by-type | - | `logChart`, `logitChart`, `multiplicativeEuler_pos`, `logisticEuler_mem` |
| Pareto-vs-argmin discipline (`multiscale_moo.py` returns fronts) | - | `scalarized_min_is_pareto` + `middle_not_scalar_min` (Triad witness) |

The plan below extends this, it does not duplicate it.

### 0.2 Published-literature anchors (all IN the toolkit's `references/`)

Primary (cite by file path in code and Lean docstrings):
- `references/NNC & Generated Arithmetics/Grossman, Non-Newtonian Calculus.pdf` (Grossman & Katz 1972) - generators, geometric/bigeometric derivatives.
- `references/NNC & Generated Arithmetics/Grossman, Meta-Calculus.pdf` (Grossman 1981) - meta-gradients, weights u/v.
- Grossman 1983 *Bigeometric Calculus* - `D_BG[x^n] = e^n` constancy.
- Czachor 2016 *Relativity of Arithmetic*; Czachor 2019 *Dark Energy from Arithmetic Mismatch* (Eq. 35 meta-Friedmann, used in `scalar_friedmann.py:80-120`); Burgin & Czachor 2020 monograph Ch. 2-6.
- Bashirov et al. 2008 (multiplicative calculus), Aniszewska (multiplicative Runge-Kutta/Lorenz), Boruah (bigeometric DEs) - all present in the references tree.
- `CITATION_INDEX.md` (toolkit root) maps equations to code locations - the implementing agent MUST consult it before writing any docstring citation.

### 0.3 Audited gaps this plan closes

Toolkit (from its own `HONEST-STATUS.md` plus the audit):
- G1. Spectral-gap preservation fails ~20% of Neumann-BC cases
  (counterexamples already sitting in `results/spectral_gap_boundary_moo.json`)
  but the failure boundary is not encoded as tests or stated as a theorem
  side-condition.
- G2. k(L) = -0.0137*log10(L) + 0.1593 rests on 8 data points across 61
  orders of magnitude (R^2 = 0.71, p = 0.008; post-bugfix 0.82), and ALL
  affine features fit equally well (acknowledged feature-degeneracy). This is
  the identifiability problem, live and acknowledged but not guarded.
- G3. No formal layer: core operators are "computer-algebra verified" only.
- G4. The fork: the-agent-maker copy and the toolkit have both moved since
  Dec 2025 with no sync policy (the GCC relay's mush-reenters-through-the-
  supply-chain failure mode).

the-agent-maker module (unchanged from v1 of this plan):
- G5. Landmines: pymoo hidden hard-dep, no k-range guardrail, GlobalMOO
  stubs, unlogged scalarization weights. [CLOSED 2026-06-12, commits
  c4f0922/8eafb07]
- G6. No MDL term in MOO objectives; only recovery tests. [CLOSED
  2026-06-12, commit 8eafb07]

Toolkit operator core (NEW - inline math audit 2026-06-12, full detail in
`meta-calculus-toolkit/docs/AUDIT-2026-06-12.md`, findings A1-A9):
- G7. CRITICAL: `Sqrt` generator is not a bijection (x -> sqrt|x| vs
  inverse y^2; round trip silently wrong for x < 0; derivative sign wrong
  for x < 0). Violates the Czachor bijection axiom the whole system rests
  on - and is exactly the object the Lean `Chart := Equiv` layer makes
  unconstructible.
- G8. Systemic sign-flipping epsilon guards (~8 sites): small negative
  values replaced by +epsilon, flipping downstream quotient signs.
- G9. Inconsistent overflow policy: GeometricDerivative clips
  unconditionally at exp(+-50) while BigeometricDerivative deliberately
  defaults to no clipping; plus Richardson extrapolation pairs wrong step
  ratios after NaN filtering; weighted chain_rule double-counts u/v; the
  "Mathematical Basis" docstring describes the conjugated operator, not
  the implemented differential form.

---

## 1. Coding principles (Torvalds discipline, applied)

- **R1 Data structures first.** Lean->Python bridge = JSON golden vectors,
  not shared code. Neumann failure boundary = a counterexample TABLE
  extracted from existing results, not prose.
- **R2 Eliminate special cases.** One k-guardrail location per repo
  (upstream `core/`-level in the toolkit, `BigeometricTransform` in
  the-agent-maker), not per-caller.
- **R3 Never break the user.** The toolkit has a LIVE Railway site and 347
  passing tests; every change keeps them green. the-agent-maker evaluator
  dicts with 5 keys keep working. Behavior changes ship behind flags
  defaulting safe.
- **R6 Show me the code.** Lean compiles after every theorem. Python test
  files run after every test added.
- **R8 Every error path is a code path.** pymoo-absent, k-out-of-range,
  Neumann-BC, NaN gradients: each gets an explicit, tested path.
- **R9 Incremental over rewrite.** The toolkit's simulations, results JSON,
  and portfolio site are accumulated knowledge. Nothing is rewritten;
  failures get encoded, not erased.
- **R10 One logical change per PR.** Each phase below is one PR.
- **Anti-theater rule** (from the relay): every new test docstring carries
  `Fails if: <specific defect>`; every phase exit includes a deliberate-break
  check (revert safeguard, watch test fail, restore). No test may assert a
  label the author set.

---

## 2. Dependency graph

```
P0 freeze + housekeeping (1h)
  |
  +--> P1 the-agent-maker landmine defusal (0.5d)        [agent-maker]
  |      |
  +--> P2 MDL objective (1-2d)                           [agent-maker]
  |
  +--> P3 Lean: meta-calculus operator layer (3-5d)      [gcc-lean-lab]
  |      |
  +--> P4 toolkit validation hardening (3-5d)            [toolkit]
  |      |        (P4 independent of P1-P3; parallel-safe)
  |      |
P3 ----->+--> P5 golden-vector bridge (1-2d)             [all three]
  |      |
P1+P2 -->+--> P6 fork-collapse / sync policy (1d)        [toolkit + agent-maker]
```

Out of scope (explicit): GlobalMOO cloud client; new physics mechanisms for
k(L) (research, not engineering); H0-tension work; GPU pipeline runs;
portfolio-site redesign.

Deferred (blocked by account spend limit as of 2026-06-12, tracked as task
#8): the remaining audit lenses - bug/efficiency scan of cli.py +
simulations + scheme_validator, results-reproducibility audit of the 8
headline claims, portfolio-site design review. Relaunch the three agent
prompts from the 2026-06-12 session when the limit is raised; their findings
slot into P4 (code/results) and a new P7 (site) on arrival.

---

## PHASE 0 - Freeze the spec layer + housekeeping (~1 hour)

Unchanged from v1 except item 4. Summary:

1. `gcc-lean-lab/README.md`: blunt non-claim ("does not prove phi governs
   physics...") + pinned versions + build instructions.
2. `gcc-lean-lab/docs/THEOREM_MAP.md`: every theorem name -> file -> meaning;
   each entry verified by grep before commit. Add a column mapping theorems
   to toolkit concepts per the table in 0.1 above.
3. Tag `v0.1.0-gcc-razor`.
4. `C:\Users\17175\.claude\CLAUDE.md` project table: fix the stale row to
   `Meta-Calculus Toolkit | C:\Users\17175\Desktop\meta-calculus-toolkit | 60%, Railway-live (GitHub: meta-calculus-portfolio)`;
   add rows for the agent-maker module and gcc-lean-lab.

Exit: lake build green, tag exists, THEOREM_MAP greps clean.

---

## PHASE 1 - the-agent-maker landmine defusal (~0.5 day)

UNCHANGED from v1 (it audited that repo correctly). Recap of the four items,
full detail in v1 text (git history of this file):

1.1 pymoo fail-fast in `moo_bridge.py::MOORunner.__init__` (line 431) +
facade grep. Test: monkeypatch `PYMOO_AVAILABLE=False`, assert ImportError.
1.2 k-range guardrail in `bigeometric.py::BigeometricTransform.transform`
(line 68): clamp k to `[1e-3, 0.5]` behind `clamp_k: bool = True` config,
warn-once. Justification: Phase 3 theorems `bgTransform_contracts` /
`bgTransform_amplifies_of_k_le_zero` (cite by name in the comment).
Mandatory pre-grep of phase configs for out-of-range k.
1.3 GlobalMOO stubs: verify `is_available()` gates all callers; no client
implementation.
1.4 `select_from_pareto` (line 496): log chosen index + weights
(governance: scalarization weights are an auditable regime choice).

Tests in `tests/unit/test_meta_calculus_guards.py`; each with `Fails if:`;
deliberate-break check for 1.1 and 1.2.

---

## PHASE 2 - MDL objective for MOO in the-agent-maker (1-2 days)

UNCHANGED from v1. Recap: perplexity is already the data-half of two-part
MDL; add the model-bits half.

- New pure fn `description_length_bits(active_components, total_components,
  param_count, residual_nll_bits=None)` (lgamma-based log-binomial; refs
  Rissanen 1978, Grunwald MDL ch. 5).
- Append `description_length` ObjectiveDefinition to `EVOMERGE_OBJECTIVES`
  and `EXPERT_DISCOVERY_OBJECTIVES`; wire into both `compute_objectives`
  via `.get` with computed default (R3).
- MANDATORY pre-step: grep all F-matrix consumers for hard-coded column
  indices (`pareto_front|objective_names|F\[`).
- `MOOConfig.include_mdl: bool = True` opt-out for fixed-architecture sweeps.

Tests in `tests/unit/test_mdl_objective.py`: independent-formula equality,
monotonicity properties, 5-key/6-key evaluator compat, frontier-survival of
the small-model solution (skipif no pymoo). Exit: consumer grep clean, old +
new tests pass, `run_demo()` runs.

---

## PHASE 3 - Lean: the meta-calculus operator layer (3-5 days)

**Goal**: formalize the toolkit's operator core in gcc-lean-lab, anchored to
the books actually in `references/`. New file `GccLeanLab/MetaCalculus.lean`
(+ extend `GccLeanLab/Tests.lean`). Compile after every theorem. Hard rules
from the relay's error log apply (no `/-!` before import; no trailing
closers; `simp only`/`change` before `omega`/`rw` on unreduced
projections; `HasDerivAt` formulations, never bare `deriv` equalities).

### 3.1 Geometric and bigeometric derivatives (Grossman-Katz 1972/1983)

Build on the capstone's `PosReal`, `logChart`, `logRate` (do not re-roll):

```
def geomDeriv (f : Real -> PosReal) (x : Real) : Real :=
  deriv (fun u => Real.log (f u).1) x                  -- log-rate; D_G = exp of this

def bgDeriv (f : PosReal -> PosReal) (x : PosReal) : Real :=
  deriv (fun u : Real => Real.log (f (logChart.symm u)).1) (Real.log x.1)

def bgDerivMul (f : PosReal -> PosReal) (x : PosReal) : Real :=
  Real.exp (bgDeriv f x)
```

| Theorem | Statement | Toolkit twin / anchor |
|---|---|---|
| `geomDeriv_expFlow` | geometric derivative of `expFlow` is the constant rate | already essentially `logRate_expFlow`; restate as a one-line corollary, no new proof |
| `bgDeriv_powFlow` | for `powFlow n x := <x.1 ^ n, rpow_pos>`, `bgDeriv (powFlow n) x = n` | `core/derivatives.py::BigeometricDerivative`; Grossman 1983. Proof: funext to affine form via `Real.log_rpow (exp_pos _)`, then `HasDerivAt.const_mul (hasDerivAt_id u)` - copy the `logRate_expFlow` proof shape. |
| `bgDerivMul_powFlow` | `bgDerivMul (powFlow n) x = Real.exp n` | the toolkit's constancy diagnostic (`constancy_analysis_v2.json`, CV=0.0000) and `verify_power_law` - the numerically-checked fact becomes a theorem. |

### 3.2 F007 unified derivative - the keystone (NEW, highest value)

`core/derivatives.py:26-40`: `D*_w f(a) = (v(f(a))/u(a)) * beta'(f(a)) * f'(a) / alpha'(a)`
with weights u, v and generators alpha, beta (Grossman 1981;
`F007_WEIGHT_DERIVATION.md` in the toolkit docs).

Lean shape - define over charts to reuse the lab's machinery:

```
structure WeightedGen where
  alpha : Chart Real Real      -- domain generator (Equiv)
  beta  : Chart Real Real      -- codomain generator
  u v   : Real -> Real         -- meta-weights

def f007Deriv (W : WeightedGen) (f : Real -> Real) (a : Real) : Real := ...
```

Theorems (specialization lemmas - prove F007 SUBSUMES the family):
- `f007_classical`: alpha = beta = Equiv.refl, u = v = 1 -> `f007Deriv = deriv f a` (under HasDerivAt hypotheses).
- `f007_geometric`: beta = log-side chart, u = v = 1 -> recovers `geomDeriv` shape.
- `f007_bigeometric`: alpha and beta log charts -> recovers `bgDeriv`.
Each as an equality of the defining expressions with explicit
differentiability/positivity side-conditions - this is exactly the kind of
bookkeeping a typechecker is for, and it is what makes the Python F007
implementation auditable.

Failure mode: chart-on-Real vs chart-on-PosReal mismatch (log needs
positivity). Mitigation: state geometric/bigeometric specializations on
PosReal-valued functions and connect via `Subtype.val`, mirroring how the
capstone handles `expFlow`.

### 3.3 Gradient-transform theorems (justify the Phase 1.2 guardrail)

Unchanged from v1: `bgTransform (k g : Real) := g * |g| ^ (2*k - 1)` with
`bgTransform_zero`, `abs_bgTransform` (`= |g| ^ (2*k)`), `sign_bgTransform`,
`bgTransform_fixed` (|g|=1), `bgTransform_contracts` (0<k<1/2, |g|>1),
`bgTransform_amplifies_of_k_le_zero`, `bgTransform_expands_of_half_lt_k`.
Lemma candidates: `Real.rpow_add (abs_pos.mpr hg)`, `Real.rpow_pos_of_pos`,
`Real.rpow_lt_rpow_left_iff`, `Real.rpow_lt_one`; fallbacks `positivity` /
`nlinarith`. Mind the rpow `0^0=1` edge (the leading `g *` factor handles
`g = 0`).

### 3.4 k(L) structural facts (NOT the fitted values)

Unchanged from v1: `kOf slope intercept L := slope * Real.logb 10 L + intercept`;
`kOf_antitone` (slope<0), `kOf_mem_Icc` on `[1, 10^12]`. The fitted
coefficients, R^2, and the 8-point dataset are EMPIRICAL - they belong to
Phase 4, never to Lean. (The toolkit's own HONEST-STATUS feature-degeneracy
note is the proof this boundary matters.)

### 3.5 Witness examples + THEOREM_MAP update

Concrete instantiations in `Tests.lean` (`bgDeriv_powFlow` at n=3;
`f007_classical` on a polynomial; `bgTransform 0.3 (-4)` by `norm_num`).
THEOREM_MAP gains a "toolkit concept -> theorem" column entry per new item.

### 3.6 The audit's tombstone: signSqrtChart (NEW, from finding A1)

The toolkit's broken `Sqrt` generator (audit A1) is unconstructible in the
lab because `Chart := Equiv` demands proved round trips. Make that concrete:
define the CORRECTED odd square root as an honest chart -

```
def signSqrt (x : Real) : Real := Real.sign x * Real.sqrt |x|   -- or via cases
def signSqrtChart : Chart Real Real := { toFun := signSqrt, invFun := fun y => Real.sign y * y^2, ... }
```

with the two round-trip proofs (case split on sign; `Real.sqrt_sq`,
`Real.sq_sqrt`, `abs_of_nonneg/abs_of_neg`). The proof obligations that Lean
forces here are exactly the ones the Python `Sqrt` silently failed. Cite
`signSqrtChart` in the P4.4 fix's docstring. If `Real.sign` API is awkward,
define piecewise via `if 0 <= x then sqrt x else -sqrt (-x)` and prove with
`split_ifs`.

### Exit criteria
- `lake build` green, zero sorry, every theorem witnessed, THEOREM_MAP
  updated. Deliberate-break: flip an exponent in a witness example, watch it
  fail, restore.

---

## PHASE 4 - Toolkit validation hardening (3-5 days)

**Goal**: encode the toolkit's KNOWN failure boundaries as permanent tests,
and guard its acknowledged identifiability weakness. All work in
`C:\Users\17175\Desktop\meta-calculus-toolkit`. Keep all 347 existing tests
green; CI matrix Python 3.8-3.12 must stay green (mind 3.8 syntax - no
`X | None` annotations in new code, use `Optional`).

### 4.1 Neumann failure boundary -> counterexample regression suite

The data already exists: `results/spectral_gap_boundary_moo.json` (947 KB,
200 tests, ~20% Neumann failures).

- New `tests/test_spectral_gap_boundaries.py`:
  - Loader extracts ALL failing Neumann cases from the JSON into a
    parametrized fixture (operator params, BC spec, expected gap violation).
  - For each: re-run the gap computation (reuse the simulation entry point
    the JSON came from - locate via `simulations/`'s spectral gap script)
    and assert the failure REPRODUCES. `Fails if:` someone "fixes" the gap
    computation in a way that silently changes the failure boundary - we
    want to KNOW when the boundary moves, in either direction.
  - Mirror set: 9 passing Dirichlet cases assert preservation holds.
- Docs: add the side-condition everywhere the claim appears - grep
  `README.md`, `HONEST-STATUS.md`, `portfolio-site/` content for
  "spectral gap" and ensure every claim says "Dirichlet/periodic BCs"
  (portfolio claim-surface tests exist per commit `6f875e4` - extend them
  to assert the qualifier string is present on the relevant page).

Failure mode: re-running 200 MOO cases is slow. Mitigation: the regression
suite re-runs only the ~40 failing + 9 canonical passing cases, marked
`@pytest.mark.slow` with a 10-case smoke subset in the default run.

### 4.2 k(L) identifiability guard (the feature-degeneracy test)

HONEST-STATUS already admits all affine features fit equally well. Encode it:

- New `tests/test_k_identifiability.py`:
  - Load the 8 fit points (source: `results/k_mechanism_discovery.json`,
    fitting code `k_of_L_master.py:62-99`).
  - Refit against log10(L) AND at least two rival affine features; assert
    the R^2 spread is reported and that the test PRINTS the comparison
    table. Assert `R^2(log10 L) > 0.6` (the published floor) - `Fails if:`
    a data regeneration quietly degrades the fit below the published claim.
  - Leave-one-out stability: refit dropping each point; assert slope sign
    never flips and slope stays within +/-50% of -0.0137. `Fails if:` the
    pattern is carried by a single point (the 8-points-over-61-orders risk
    made executable).
- `meta_calculus/k_cosmology.py`: add a `provenance` dict constant next to
  the coefficients ({n_points: 8, r2: ..., fit_file: ...}) so downstream
  consumers (the-agent-maker!) can read the epistemic status
  programmatically. R3: additive only.

### 4.3 Misspecification probes for the core operators

- New `tests/test_operator_misspecification.py`:
  - `BigeometricDerivative` on functions OUTSIDE its happy family: a
    sign-changing function (where log-space is undefined - assert the
    documented error/NaN-guard path, not silence), a non-differentiable
    kink (assert finite-difference output flagged or bounded), near-zero
    values exercising `NUMERICAL_EPSILON = 1e-12` (`core/generators.py`).
    `Fails if:` epsilon-guard removal lets a 1/0 escape as inf into results.
  - `UnifiedDerivative` (F007): verify the three specializations against
    closed forms on a test grid - the SAME identities Lean proves in 3.2;
    tolerance 1e-8. `Fails if:` the Python F007 weights wiring diverges
    from the published formula (this is the bridge's premise, tested before
    the bridge exists).

### 4.4 Operator-defect remediation (AUDIT-2026-06-12, findings A1-A9)

Fixes in `meta_calculus/core/` plus the property tests that make each defect
class unrepresentable again. Full finding detail:
`docs/AUDIT-2026-06-12.md` (same repo).

1. **A1 Sqrt bijection (CRITICAL)** - `core/generators.py:256-275`. Replace
   with the odd bijection `signSqrt(x) = sign(x) * sqrt|x|`, inverse
   `sign(y) * y^2`, derivative `0.5 / sqrt|x|` (correct for both signs,
   x != 0); raise DomainError at x = 0 or document the epsilon behavior.
   R3 pre-grep: find all `Sqrt(` constructions in simulations/tests and
   confirm none relied on the absolute-value behavior.
   TEST (the permanent guard): a round-trip PROPERTY test over EVERY
   generator in the registry - for x in a mixed-sign, mixed-magnitude grid,
   `g.inverse(g(x)) ~ x` (rtol 1e-9) or a documented domain error.
   "Fails if: any present or future generator ships without an actual
   inverse" - the A1 class, killed permanently. Mirror witness in Lean
   (P3.6): `signSqrtChart : Chart Real Real` proves the corrected version
   IS an equivalence.
2. **A2 sign-flip epsilon guards (HIGH)** - `core/derivatives.py:121,
   480-483, 706-707`, `core/generators.py:244, 249`, `composite.py:152,
   197`. Replace with `np.where(np.abs(v) < eps, np.copysign(eps, v), v)`.
   TEST: sign-preservation - a function crossing zero from below must not
   produce a positive guarded quotient at any site.
3. **A3 GeometricDerivative unconditional clip (HIGH)** -
   `core/derivatives.py:126-127`. Change `MAX_RATIO=50` to constructor
   param `max_ratio: Optional[float] = None`, matching
   BigeometricDerivative's honest default.
   TEST: identity preservation `D_G[e^(kx)] = e^k` for k in {1, 10, 60} -
   the k=60 case fails under the old clip.
4. **A4 Richardson wrong-ratio pairing (MEDIUM)** -
   `core/derivatives.py:716-720`. Keep (step, value) pairs; extrapolate
   only adjacent valid pairs, or use ratio-aware weights
   `(r^2 * D_small - D_large) / (r^2 - 1)`. Fix the backwards docstring
   (line 736) and the "two smallest" comment (line 719).
   TEST: force NaN at h/2, assert the result still has central-difference
   accuracy on x^3 (catches the silent 4/3-on-ratio-4 blend).
5. **A5 weighted chain_rule double-count (MEDIUM)** -
   `core/derivatives.py:587-607`. Assert unity weights (raise otherwise)
   and document, or implement the correct weighted composition.
6. **A6 Mathematical Basis docstring (MEDIUM)** -
   `core/derivatives.py:413-419`. Rewrite to derive the differential form;
   cite gcc-lean-lab `f007_classical/f007_geometric/f007_bigeometric` as
   the pinned semantics (they compiled against exactly the line-487
   formula).
7. **A7-A9 (LOW)** - AdaptiveMetaDerivative accepts+forwards `method`;
   `frw_scheme_robustness.py:600` bare except -> `except Exception`;
   rename/document the `stability` metric in `compare_derivative_methods`.

Exit addition: deliberate-break for item 1 (restore the old Sqrt, watch the
round-trip property test fail) and item 2 (revert one copysign, watch the
sign-preservation test fail).

### 4.5 PyCBC MVE follow-through (smallest honest increment)

`run_pycbc_mve.py` exists; `pycbc_mve_results.json` is sparse (24 KB).
Do NOT plan a LIGO research program here (out of scope). One increment:
make the existing MVE reproducible - pin its inputs, write a smoke test
that runs it end-to-end on the bundled minimal data (skipif pycbc absent),
and record its output schema. `Fails if:` bit-rot in the PyCBC integration
goes unnoticed for another six months.

### Exit criteria
- New suites pass; all 347 legacy tests pass; CI green on 3.8 and 3.12;
  portfolio claim-surface test asserts the Neumann qualifier; deliberate-
  break demonstrated for 4.1 (loosen the gap assert) and 4.2 (drop a fit
  point permanently).

---

## PHASE 5 - Golden-vector bridge (1-2 days)

As v1, with the consumer list corrected: Lean generates, BOTH Python repos
consume.

1. `gcc-lean-lab/GccLeanLab/VectorGen.lean` + `lean_exe gccvectors`:
   deterministic JSON to stdout - `bgTransform` grid, `bgDerivMul` power-law
   expectations (`exp n`), `kOf` affine table, (new) F007 specialization
   triples (function id, point, expected classical/geometric/bigeometric
   values), AND (audit A1) generator round-trip vectors: (x, signSqrt x)
   pairs over a mixed-sign grid so the toolkit's corrected Sqrt is pinned to
   the chart-certified values from P3.6.
   Header: `"provenance": "Float shadow of Lean theorems <names>; floats are not the proofs"`.
2. Vendor `golden_vectors.json` into:
   - `meta-calculus-toolkit/tests/data/golden_vectors.json` -> consumed by
     `tests/test_golden_vectors.py` exercising `core/derivatives.py`
     (GeometricDerivative, BigeometricDerivative, UnifiedDerivative) and the
     generator arithmetic.
   - `the-agent-maker/src/cross_phase/meta_calculus/golden_vectors.json` ->
     consumed by `tests/validation/test_golden_vectors.py` exercising
     `BigeometricTransform` and `compute_k`.
3. SHA-256 of the vendored file asserted in both consumers (tamper-evidence;
   update procedure in the docstring). Tolerances: 1e-6 algebraic, 1e-3
   numerical-derivative cases. Never exact-equality on derivatives
   (Float-vs-Real ulp gap).

`Fails if:` either Python implementation drifts from the Lean-proved forms -
the report-vs-code divergence class, caught by CI in three repos.

---

## PHASE 6 - Fork-collapse: sync policy for the two Python copies (1 day)

**Goal**: end the silent divergence (G4). The relay's lesson verbatim: "two
drifting copies of a single source of truth is how the mush re-enters
through the supply chain."

1. Diff the seven shared modules
   (`k_formula/bigeometric/spectral_gap/meta_grokfast/moo_bridge/...`)
   between the repos; produce `docs/FORK_STATUS.md` in the-agent-maker
   listing per-module: identical / toolkit-ahead / agent-maker-ahead / both.
2. Policy decision (single rule, documented in both repos): the TOOLKIT is
   upstream for math semantics (operator definitions, k provenance);
   the-agent-maker owns ML-specific adaptation (optimizer wiring, phase
   configs). Concretely:
   - Port Phase 1.2's guardrail + Phase 2's MDL function UPSTREAM into the
     toolkit (`meta_calculus/core/` + `multiscale_moo.py`) in the same PR
     wave, so semantics never fork again.
   - the-agent-maker imports the `provenance` dict from 4.2's pattern
     (vendored constant, since the repos do not share an environment) and
     its docs state the snapshot date + upstream commit hash.
3. Add a sync checklist to both repos' docs: "when touching shared-module
   semantics, change toolkit first, then mirror, then update FORK_STATUS."

Failure mode: porting the guardrail upstream collides with toolkit callers
that legitimately explore k outside (0, 1/2) in SIMULATIONS (cosmology
sweeps may scan k). Mitigation: upstream guardrail lives in a new
`core/safe_transforms.py` used by optimizer-facing paths only; raw
exploratory APIs stay unclamped and documented as research surfaces. (R2:
one guarded door, one explicitly unguarded lab bench - not many half-doors.)

### Exit criteria
- FORK_STATUS.md accurate (spot-check 3 modules by hand), both repos green,
  sync checklist present in both.

---

## Final acceptance (whole plan)

1. All three repos build/test green from clean checkouts (toolkit CI matrix
   included); Railway portfolio still serves (claim-surface tests green).
2. Every new test demonstrated to fail under its documented deliberate break.
3. THEOREM_MAP.md (with toolkit-concept column), META_CALCULUS_INTEGRATION.md,
   FORK_STATUS.md, and HONEST-STATUS.md updated in the same PRs as the code.
4. One-paragraph empirical summary appended to
   `gcc-lean-lab/docs/responses-to-build-log-2026-06-11.md`: what the
   validation suites actually found (Neumann boundary shape, k(L)
   leave-one-out result, F007 specialization check) - the relay gets its
   data.
