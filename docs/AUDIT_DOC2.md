# Audit: Doc 2 Implementation vs Approximate Supercompilation Spec

**Document**: *Approximate Supercompilation for MeTTa+MM2* (Goertzel, Oct 2025)
**Spec file**: `docs/specs/approximate_metta_supercompilation_spec.md`
**Audit date**: 2026-04-29 · **🔴 SUPERSEDED IN PART 2026-08-24 — read §"Correction" at the foot FIRST**
**Result (2026-04-29, EXISTENCE audit)**: All 7 algorithms implemented. 5 minor gaps fixed during audit.
**Result (2026-08-24, EXECUTION cross-check)**: **NOT complete.** ~60% by section, ~35% by capability.
Three of the ✅ rows below are refuted by running them; §6's Specialization phase is a stub. Every
table in this file was built by asking *does a function with this name exist* — none of them ran one.

---

## Algorithm Coverage

| # | Spec Name | Section | Implementation | Status |
|---|-----------|---------|----------------|--------|
| 1 | AddPBox (independent + Fréchet) | §2.3 | `approx/PBoxAlgebra.jl::add_pbox` | ⚠️ indep ✅ / **Fréchet 🔴 D2** |
| 2 | EstimateCardinalityPBox | §3.2 | `approx/UncertainQuery.jl::estimate_cardinality_pbox` | 🔴 **D1 — wrong answer, live consumer** |
| 3a | ApproximateSplit | §3.3 | `approx/UncertainQuery.jl::approximate_split` | ✅ |
| 3b | MatchWithUncertainty | §4.2.2 | `approx/UncertainInference.jl::match_with_uncertainty` | ✅ |
| 4 | ApplyRule (UncertainModusPonens) | §4.3 | `approx/UncertainInference.jl::apply_rule` | ⚠️ ✅ on [0,1]; **D3** on §4.1's [-1,1] — but its two helpers are UNSPECIFIED by the paper |
| 5 | ApproximateFitness (Hoeffding) | §5.2 | `supercompiler/EvoSpecializer.jl::approximate_fitness` | ⚠️ ε+tail only — **no `stratified_sample`, no `evaluate`** |
| 6 | TournamentWithPBox (Monte Carlo) | §5.3 | `approx/ApproxMOSES.jl::tournament_with_pbox` | ✅ |
| 7 | AllocateEvaluations (VoI) | §5.5 | `supercompiler/EvoSpecializer.jl::allocate_evaluations` | ⚠️ `could_be_best` is a heuristic, not P(best); top-k ≠ `allocate_proportional` |

## Data Structure Coverage

| Structure | Spec Fields | Implementation | Status |
|-----------|------------|----------------|--------|
| `PBox` | 4: intervals, probabilities, confidence, correlation_sig | `core/MCore.jl` — all 4 fields ✅ | ✅ |
| `UncertainNode` | 4: base, value_pbox, cost_pbox, error_bound | `core/MCore.jl` ✅ | ⚠️ **declared, never constructed** — 0 call sites in `src/`; §2.4 is *integration* |
| `UncertainFact` | 5: predicate, arguments, truth_pbox, confidence, derivation | `approx/UncertainInference.jl` ✅ | ✅ |
| `EvolutionaryPBox` | 5: individual_id, fitness_pbox, rank_pbox, heritability, evaluation_count | `supercompiler/EvoSpecializer.jl` ✅ | ✅ |
| `ApproxIndex{T}` | 4: core, overflow (Bloom), weights, coverage | `approx/ApproxPipeline.jl` ✅ | ⚠️ **orphaned** — the pipeline never builds one |
| `ApproximatePathSig` | 3: base_sig, error_level, confidence | `approx/ApproxPipeline.jl` + `error_bound` ✅ | ✅ |

## Theoretical Guarantees

| Theorem/Lemma | Spec | Implementation | Status |
|---------------|------|----------------|--------|
| Theorem A.2 Error Composition | §7.1: `Σ w_i + O(n²·w_max²)` | `PBoxAlgebra.jl::error_composition_bound` ✅ | ✅ |
| Lemma A.4 Fréchet Width | A.4: `wX + wY + 2·min(wX,wY)` | `PBoxAlgebra.jl::frechet_width_bound` ✅ | ✅ |
| Lemma A.5 Hoeffding Bound | A.4: `2·exp(-2nt²/(b-a)²)` | `PBoxAlgebra.jl::hoeffding_bound` + `hoeffding_epsilon` ✅ | ✅ |
| Inference Convergence §4.4 | `O(1/√(nr))` width | `UncertainInference.jl::convergence_width_bound` ✅ | ⚠️ formula only — no semi-naive run to hold over |
| Convergence detection §5.6 | `overlap > 0.5` threshold | `ApproxMOSES.jl::population_converged` ✅ | ✅ |

---

## Gaps Found and Fixed During Audit

### GAP-D2-1: `is_cacheable` on `ApproximatePathSig` missing `error_bound` field in spec

**Spec §6.4** lists `ApproximatePathSig` with 3 fields: `base_sig, error_level, confidence`.
Our implementation adds a 4th field `error_bound::Float64` not in the spec.

**Assessment**: ADDITIVE — our implementation is a strict superset. The `error_bound` field provides BOUNDED level's ε value, making the spec's BOUNDED class concrete. Kept as is.

### GAP-D2-2: Conjunction AND with perfect correlation — probability composition

**Spec §4.2.1**: `max(T_A + T_B - 1, 0)` for perfect correlation.
**Implementation**: `UncertainInference.jl::_and_lukasiewicz` applies this to intervals correctly.
The probability is `min(pa, pb)` (Fréchet upper bound). This is conservative and sound.

**Assessment**: CORRECT per Fréchet upper bound.

### GAP-D2-3: `hoeffding_bound` — δ parameterization

**Spec §5.2** uses `δ = 0.05` hardcoded for the 5% tail. Our `hoeffding_bound(n, t)` uses `a=0, b=1` defaults.

**Fixed**: `hoeffding_bound` and `hoeffding_epsilon` already accept `a, b` keyword args. δ is exposed via `delta` parameter in `approximate_fitness`. No change needed.

### GAP-D2-4: Missing convergence detection §5.6 formula in spec

**Spec §5.6**: `Converged = |{(i,j): overlap(Fi,Fj) > 0.5}| / |P|² > θ`
**Implementation**: `population_converged` counts ordered pairs (i,j) with diagonal. Matches spec.

**Assessment**: CORRECT.

### GAP-D2-5: `ApproxIndex` — `approx_index_lookup` `:POSSIBLE` return

**Spec §6.2** says overflow BloomFilter catches rare entries "probabilistically."
Our lookup returns `:POSSIBLE` Symbol — correct as a 3-way result indicator.

**Assessment**: CORRECT.

---

## False Positives from Initial Audit

| Reported Gap | Why It's a False Positive |
|-------------|--------------------------|
| GAP-1: UncertainNode missing | EXISTS in `core/MCore.jl` lines 259-277 |
| GAP-3: Algorithm 5 missing in ApproxMOSES | EXISTS in `EvoSpecializer.jl::approximate_fitness` |
| GAP-4: Algorithm 7 missing | EXISTS in `EvoSpecializer.jl::allocate_evaluations` |
| GAP-5: Error composition formula wrong | O(n²·w_max²) matches spec — big-O hides constant |
| GAP-6: MONTE_CARLO_TRIALS usage | Defined as `const MONTE_CARLO_TRIALS = 100`, used correctly |
| GAP-8: CanonicalPathSig integration | Used in `ApproxPipeline.jl` via module-level import |

---

## Summary (2026-04-29 — SUPERSEDED, kept verbatim as the record of what an existence audit yields)

**Doc 2: COMPLETE** — all 7 algorithms, all spec data structures, all theoretical guarantees implemented and tested. 0 blocking gaps. 5 audit notes (all ADDITIVE or CORRECT).

---

# 🔴 Correction — 2026-08-24 execution cross-check

Every table above answers *does a function with this name exist*. None of them ran one. Running them
refutes three rows and downgrades five more. The 2026-04-29 verdict is not withdrawn as a record —
it is what a name-match audit produces, and that is exactly why it read COMPLETE.

## Two numbers, and the gap between them IS the finding

| measure | value | what it counts |
|---|---|---|
| **by SECTION** | **~60%** | paper §§2–7 rows that have an implementation. Flattering, and the number an audit-against-a-paper naturally produces: 24 rows mostly green. |
| **by CAPABILITY** | **~35%** | what the compiler can actually *do*. The one stubbed phase is the thing the paper is about. |

**Why they diverge:** §6.1's **Specialization phase emits the reordered ORIGINAL program**
(`ApproxPipeline.jl:312` — `program_approx = sprint_program(planned_nodes)`). No approximate variant
is generated, ever. §6.2's `ApproxIndex` and §6.3's two prims — the paper's own worked example of
*what approximate supercompilation produces* — are built, unit-tested, and never reached. So the
p-box library is real and the approximate *code generation* is absent, while a section-count scores
both the same.

## ⚠️ SCOPE — say WHICH thing is missing

It is **wrong** to call §6.2/§6.3 "the half that makes it supercompilation." The supercompiler core
is `mm2_supercompiler_spec.md` **§6 — Supercompiler Core with Minimal IR** (§6.1 Structural Stepper,
§6.2 Bounded Splitting, §6.3 Canonical Keys and Termination), and **that exists here**:
`Stepper.jl` · `BoundedSplit.jl` · `CanonicalKeys.jl` (off by default; see `workflows/SPECMAP.md` §1).

What is missing is **the approximate CODE GENERATION that THIS paper contributes on top of it**.
Both statements are true; they scope differently, and only the second one is this file's finding.

## The three defects — NOT a set. Rank by REACH, not by proximity.

### 🔴 D1 — `estimate_cardinality_pbox` returns an interval that excludes the truth · **LIVE CONSUMER**

`approx/UncertainQuery.jl:81`. **This is the only one of the three that a user can hit today.**

There is no `stratified_sample`. The function calls `dynamic_count(btm, src)` — an **exact
full-space** subtrie count — and then rescales it by `total/√total` as if it had sampled √n of the
space, and sets Hoeffding's range `(b−a)` to `total_atoms`.

```
100 × (edge i i+1), true cardinality 100
  point_estimate  = 100          ✅ (the exact path is fine)
  pbox interval   = [957.1, 1042.9]   contains 100? false
```
Error grows as √n. Reproduce: `curl -s -X POST http://localhost:7702/julia --data-binary @probe.juliasrc`.

**Consumer:** `plan_join_order_approx` (`UncertainQuery.jl:221`) → `total_cost` (`:49`), which is
Phase 2 of `run_approx_pipeline`, reachable from `SCOptions(use_approx=true)`
(`SCPipeline.jl:252`). **Second-order effect, and it is the worse one:** ε depends only on
`n_sample` and `total` — both *space*-level, not *source*-level — so `width` and `max_width` come
out near-identical for every conjunct. The β·Error and γ·Variance terms become a constant offset and
the **three-objective cost model silently degenerates to time-only ranking** whenever the sampling
path is taken. The feature does not fail loudly; it stops being the feature.

**🔗 It also corrupts today's `dynamic_count` fix.** `Selectivity.jl:51` was extended 2026-08-24 (`8317b3b`) to
walk leading **ground arguments** into the seekable prefix — `(ccr CY $i)` reads 26 instead of
57,686. `estimate_cardinality` (`Statistics.jl:336`) is argument-sensitive too, via
`argument_selectivity` per constrained position. So **both estimators now see ground arguments** —
and the p-box wrapper takes `dynamic_count`'s newly-exact count and multiplies it by `√total`. The
approx path inherits the fix and immediately destroys the precision it bought. Fixing D1 is what
makes today's work reach this lane.

### 🟠 D2 — Fréchet `add_pbox` does not conserve probability mass · **UNREACHABLE TODAY**

`approx/PBoxAlgebra.jl:187`. Two dependent 2-interval p-boxes of mass 1.0 → `confidence = 2.0`, a
field §2.2 defines as "total probability mass tracked". Cause: `p = min(px, py)` is the Fréchet
**upper** bound on a joint, applied per interval-pair with no normalization; the spec's stated bound
is a **pair**, `[max(Fx+Fy−1, 0), min(Fx, Fy)]`, and only the upper half is implemented. The
interval widening by `±min(wX,wY)` is a Lemma-A.4-shaped heuristic, not the CDF-bound formula.

**Reach: nothing in `src/` constructs a dependent p-box.** `mark_dependent` has no non-test caller,
so no live path reaches `_add_pbox_frechet`. Real, and currently only reachable from a REPL/test.

**Why it survived:** `test_pbox_algebra.jl:55` asserts `sum(probabilities) ≈ 1.0` for the
**independent** case only. The Fréchet testset (`:64`) checks Lemma A.4's *width* bound and nothing
about mass.

### 🟠 D3 — `widen_pbox` NARROWS a negative lower bound · **UNREACHABLE TODAY · RE-SCOPED 2026-08-24**

`approx/PBoxAlgebra.jl:261`. `[lo/factor, hi*factor]`, so `[-1,1]` with factor 2 → `[-0.5, 2.0]`:
the lower bound moves *toward* zero. Fine on `[0,1]`; wrong-looking on the `[-1,1]` truth range §4.1
explicitly allows.

**Reach: nothing uses the [-1,1] option** — every constructed truth p-box is `[0,1]`. Latent until
the PLN bridge (`FactorGeometry.jl:297 stv_to_pbox` / `PLNDemand.jl:247 pbox_to_stv`) carries a
signed strength.

**🔴 RE-SCOPED — this is NOT a deviation from the paper.** A PDF-vs-extraction fidelity audit
(2026-08-24) established that **`WidenPBox` is never defined anywhere in the paper**: it appears
only as a call site inside Algorithm 4. §2.3.1 promises the operation list — *"Each operation in our
p-box algebra corresponds to a common pattern in program execution:"* — and **the list is absent
from the PDF itself** (verified against the raw PDF: 20pp, no images, no figures, no tables, so this
is a hole in the paper, not a conversion loss). The same is true of `MultiplyPBox`.

⇒ D3 is an **unspecified operation whose semantics we chose**, not a conformance failure. It needs
**its own oracle**, and a green test over it proves only self-consistency. Full register of the ~22
helpers the algorithms invoke but the paper never defines:
`docs/specs/supercompiler/approximate_metta_supercompilation_spec.md` § *UNSPECIFIED BY THE PAPER*.

**D2 is the opposite case and the contrast is the point:** §2.3 *does* state the Fréchet–Hoeffding
bound explicitly, as a **pair** — `[max(Fx+Fy−1, 0), min(Fx, Fy)]` — and we implemented only the
upper half. D2 is a real deviation; D3 is an unspecified choice. Triaging them as one kind was the
error.

## Test status — green, and green does not mean verified

All 5 approx test files pass under `tools/run_tests.sh` (real exit codes): 158 assertions,
`test/approx/` + `test_evo_specializer.jl`. They are **shape checks** — `hi > lo`, `bound ≥ Σwᵢ`,
`result isa X`. D1 is asserted around: `test_uncertain_query.jl:52` checks `hi > lo` on the very
interval that excludes the true cardinality by 10×. A suite that never compares against a *known
answer* cannot observe a wrong one.

## What 100% requires, in cost order

1. **Fix D1** — sample for real or drop the rescale (hours). Restores the 3-objective cost model and
   lets this lane inherit the `dynamic_count` ground-argument fix.
2. Wire `approximate_split` into `BoundedSplit`, and `apply_rule`/`UncertainFact` into
   `KBSaturation` — §4.4 and Theorem A.3 currently have no evaluation to hold over (days).
3. **Build Phase 3 for real**: emit `ApproxIndex`-backed code, execute the §6.3 prims (both handlers
   today parse `tol`/`rate` and then never use them, and return a `Residual`), feed `Profiler` output
   in for the hot/cold split. This is the bulk, and it is the paper's actual contribution.
4. Benchmark §8's 5–20× / 10–50× / 2–10×. **`benchmark/` has no approx entry** — the speedup claim
   is unmeasured here.
5. Build oracles for the **~22 paper-undefined helpers** (register in the spec file). Until then
   "matches the paper" is unanswerable for `mul_pbox`, `widen_pbox`, `structural_similarity`,
   `sample_from_pbox`, `overlap`, `width`, and every §6.2 emission primitive.

> **Spec fidelity, 2026-08-24:** the `.md` extraction this audit was written against was itself
> incomplete — §6.2's two code listings (the Phase-3 emission target), Appendix A.1 entire, and
> A.3's Step 5 assumption were missing, and two section numbers were invented. All restored;
> **Phase 3 was built with no emission target because the target was only in the PDF.**
