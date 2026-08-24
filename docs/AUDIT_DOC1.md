# Audit: Doc 1 Implementation vs MM2 Supercompiler Spec

**Document**: *A MORK-Native Supercompiler for MeTTa+MM2* (Goertzel, Oct 2025)  
**Spec file**: `docs/specs/mm2_supercompiler_spec.md`  
**Audit date**: 2026-04-29  
**Result (2026-04-29, EXISTENCE audit)**: ✅ All 14 algorithms and all spec data structures implemented.
**Result (2026-08-24, EXECUTION trace)**: two of these rows describe an implementation that does
not do its job. See §"Execution trace" at the foot. Row 2's note — *"Uses pattern_shape_histogram
+ argument_selectivity"* — is a description of the code, not a claim that it works, and it read
green for the same reason Doc-2's tables did.

---

## Algorithm Coverage

| # | Spec Name | Spec Section | Implementation | Notes |
|---|-----------|-------------|----------------|-------|
| 1 | EffectCommutes | §4.2 | `core/Effects.jl::commutes` | All 9 axioms verbatim |
| 2 | EstimatePatternCardinality | §5.1.2 | `planner/Statistics.jl::estimate_cardinality` | Uses pattern_shape_histogram + argument_selectivity |
| 3 | PrefixSampling | §5.1.3 | `planner/Statistics.jl::prefix_sample_count` | O(1) exact subtrie count via PathMap (equivalent to perfect sampling) |
| 4 | UpdateIncrementalStats | §5.2.1 | `planner/Statistics.jl::update_incremental!` | EMA growth_rate, selectivity_drift per predicate |
| 5 | ShouldReplan | §5.2.2 | `integration/AdaptivePlanner.jl::should_replan` | Drift threshold + time_since_replan |
| 6 | EffectAwarePlanning | §5.3.1 | `planner/QueryPlanner.jl::plan_query` | Pure-region identification + cost-based join order |
| 7 | RewriteOnce | §6.1 | `supercompiler/Stepper.jl::rewrite_once` | All 11 node kinds dispatched |
| 8 | CallPrimitive | §6.1 | `supercompiler/Stepper.jl::_call_primitive` | PrimRegistry with :kb_query/:mm2_exec/:fitness_eval/:identity |
| 9 | BoundedSplit | §6.2 | `supercompiler/BoundedSplit.jl::bounded_split` | SPLIT_PROB_THRESHOLD=0.95, SPLIT_DEFAULT_BUDGET=16 |
| 10 | KeySubsumption | §6.3.2 | `supercompiler/CanonicalKeys.jl::subsumes` | 3-part check: structural + KB + effect |
| 11 | IncrementalSaturation | §7.1 | `supercompiler/KBSaturation.jl::saturate!` | Semi-naive: at least one premise from delta_old |
| 12 | GatedEvolutionarySpecialization | §8.1 | `supercompiler/EvoSpecializer.jl::should_specialize` | 3-tier: SPEC_VECTORIZED/INCREMENTAL/GENERIC |
| 13 | CanReuseFitnessCache | §8.2 | `supercompiler/EvoSpecializer.jl::can_reuse_cache` | AST diff: STRUCTURAL/CONSTANT/NONE |
| 14 | BisimulationProof | §9.2 | `codegen/MM2Compiler.jl::record_bisim!` | Records :forward_sim, :backward_sim, :fairness obligations |

---

## Data Structure Coverage (Spec Appendix B)

| Spec Structure | Fields Required | Fields Implemented | File |
|----------------|----------------|-------------------|------|
| M-Core node types | 11 kinds | 11 kinds ✅ | `core/MCore.jl` |
| `Effect` algebra | 7 kinds | 7 kinds ✅ | `core/Effects.jl` |
| `MORKStatistics` | 6 fields | 6 fields ✅ | `planner/Statistics.jl` |
| `IncrementalStats` | 5 fields | 5 fields ✅ | `planner/Statistics.jl` |
| `EffectStats` | 4 fields | 4 fields ✅ | `planner/Statistics.jl` |
| `CanonicalPathSig` | 6 fields | 6 fields ✅ | `supercompiler/CanonicalKeys.jl` |
| `CanonicalKBSig` | 2 fields | 2 fields ✅ | `supercompiler/CanonicalKeys.jl` |
| `CanonicalEffectSig` | 2 fields | 2 fields ✅ | `supercompiler/CanonicalKeys.jl` |
| `VersionedIndex` | 5 fields | 5 fields ✅ | `supercompiler/KBSaturation.jl` |

---

## Gaps Found and Fixed During Audit (2026-04-29)

### GAP-1: `MORKStatistics` had only 5 fields (spec requires 6)

**Spec §5.1.1** requires:
1. `node_type_counts` — ❌ missing
2. `pattern_shape_histogram` — ❌ missing (was using flat predicate_counts)
3. `predicate_fanout` — ✅ present
4. `argument_selectivity` — ❌ missing (was empty Dict)
5. `pattern_match_cache` — ❌ missing
6. `correlation_matrix` — ❌ missing

**Fix**: Rewrote `MORKStatistics` struct with all 6 fields. Added `predicate_counts(s)` helper function for backward compat. Updated `collect_stats` to populate all 6 fields during trie scan. Added convenience 2-arg constructor for tests.

### GAP-2: `IncrementalStats` had only 4 fields (spec requires 5)

**Spec §5.2.1** requires `selectivity_drift` and `last_replan_time`.

**Fix**: Added both fields. `update_incremental!` now computes per-predicate drift and timestamps.

### GAP-3: `EffectStats` not implemented (spec §5.3.2)

**Fix**: Added `EffectStats` struct with all 4 fields to `Statistics.jl`.

### GAP-4: Algorithm 6 missing `identify_pure_regions` step

**Spec §5.3.1**: `regions ← identify_pure_regions(query, effect_analysis)` — our `QueryPlanner.jl` went straight to `plan_join_order` without this step.

**Fix**: Added `PureRegion`, `EffectBarrier`, `identify_pure_regions`, and `plan_query` to `QueryPlanner.jl`. For MORK exec patterns, always produces a single pure region (all sources are `Read`).

### GAP-5: Algorithm 5 missing `time_since_replan` check

**Spec §5.2.2**: replan if `time_since_replan > MAX_PLAN_AGE`.

**Fix**: Updated `should_replan` in `Statistics.jl` to accept `time_since_replan::Float64` and check against `max_plan_age_sec=300.0`. Updated `AdaptivePlanner.jl` to use `last_replan_time` from `IncrementalStats`.

---

## Implementation Notes

### Algorithm 3 (PrefixSampling) — exact vs sampling

The spec describes approximate sampling (`uniform_sample_prefix`, `bootstrap_variance`).
Our implementation uses `read_zipper_at_path` + `zipper_val_count` for O(1) exact subtrie counts.
This is **strictly better** than sampling (exact, not approximate) and is possible because PathMap's
trie structure provides exact counts without full enumeration. No deviation from spec intent.

### Algorithm 14 (BisimulationProof) — obligation recording

The spec describes the proof **structure** (3 obligations: forward sim, backward sim, fairness),
not automated proof checking. Our implementation records `BiSimObligation` structs for external
verification. This is the correct interpretation of the spec.

### Algorithm 6 (EffectAwarePlanning) — all-pure assumption for MORK

The spec handles both pure and non-pure regions (with `EffectBarrier`). For MORK exec sources,
all sources are `Read(space)` which commutes with `Read(space)` (Algorithm 1). So `identify_pure_regions`
always returns a single pure region for MORK programs. The infrastructure for non-pure regions
(topological ordering, EffectBarrier) is implemented for future use.


---

# 🔴 Execution trace — 2026-08-24

`8317b3b` taught `dynamic_count` to walk leading GROUND ARGUMENTS into the seekable prefix, measured
900x discrimination, and its own commit message recorded that it "did NOT help yet". A four-stage
trace (estimator -> planner -> decompose -> MORK) found **four independent nullifiers**. Two are
fixed (`bded38f`, and the `findfirst` commit); these two are open.

## N3 — `estimate_cardinality` does not discriminate on ground-argument patterns

Audit row 2 (Algorithm 2, §5.1.2). MEASURED on a corpus with a 100x split on a leading ground arg,
`total_atoms = 4020`:

    source        truth   estimate_cardinality   dynamic_count
    (a K1 $i)        20         4020                20  ✅
    (a K2 $i)      2000         4020              2000  ✅
    (b $i $v)      2000         1005              2000  ✅

Two patterns differing 100x in truth both return `total_atoms`, and the UNSELECTIVE `(b $i $v)` is
ranked cheapest — the ordering is inverted, not merely imprecise. Cause: `pattern_shape_histogram`
misses the `(pred, arity)` shape, so it falls through to `predicate_counts` / the
`CARDINALITY_FALLBACK_DIVISOR`, and `argument_selectivity` supplies no discrimination for the ground
position.

**Reach: LIVE.** `SCPipeline` no longer routes here (`bded38f`), but `MGCompiler.jl:429` and
`Explainer.jl` both still call the stats path, and `plan_report` reports from it.

## N4 — `PipelineDecompose` discards the planned order and can pick a DISCONNECTED first stage

`PipelineDecompose.jl:125-128`, unconditional, before any chaining:

    scores = static_score.(sources)
    perm   = sortperm(scores; alg=MergeSort)
    ordered_sources = sources[perm]

`static_score` is the pure variable-fraction heuristic — no stats, no btm, **and no connectivity**.
So whatever stage 2 computes is overwritten one stage later by a weaker criterion. `decompose` is
DEFAULT-ON and is the one load-bearing transform in the pipeline.

**This is not merely "weaker" — MEASURED HARM 2026-08-24.** On a corpus where variable-fraction and
cardinality disagree:

    (c $i $j)  static 0.667  card   20     <- best by cardinality, worst by static_score
    (a K $i)   static 0.333  card  300
    (b K $j)   static 0.333  card  300     <- shares NO variable with (a K $i)

    planner (dynamic) emits : (, (c $i $j) (a K $i) (b K $j))          <- correct
    decompose re-sorts to   : (, (a K $i) (b K $j)) -> _sc_tmp0        <- CARTESIAN PRODUCT

    first-stage intermediate:  decompose's chain  90000 atoms
                               planner's chain        20 atoms       (4500x)

Sorting by variable fraction prefers ground-heavy patterns, which here groups two DISCONNECTED
sources — exactly the O(K^n) blowup `PipelineDecompose` exists to prevent. MORK's
`_classify_connected` is connectivity-aware; this pre-sort is not.

⇒ **N4 is the binding constraint on the default path.** No improvement to any estimator upstream of
it can reach execution while it stands.

### The fix, decided 2026-08-24 — and TWO plausible-sounding versions were REJECTED

**✅ THE ONE TO BUILD: connectivity as a CONSTRAINT ON THE EXISTING SORT, not a replacement.**
When choosing which sources go into a stage, prefer ones sharing a variable with what is already
bound. `static_score` still breaks ties; it simply cannot pick a disconnected pair over a connected
one. Small, local, no new dependency, and it removes the measured harm directly.

**❌ REJECTED — "consume the planned order".** It makes `decompose` depend on stage 2 having run.
`decompose` is DEFAULT-ON while `plan=false` is a valid config, so it would need a fallback anyway,
leaving two orderings to reason about instead of one.

**❌ REJECTED — "refuse to group disconnected sources".** *Refuse* is the wrong verb: `decompose`
has to emit something. The constraint formulation above is what "refuse" was reaching for.

**🔗 Same property, two layers apart.** Routing on CONNECTIVITY — every conjunct shares a variable
⇒ join, else stock — is also what separated the leapfrog predictor's 5/5 from 0/4
(`project_leapfrog_default_flip_gates`). The predicate that decides whether something IS a join is
the one that holds; cardinality proxies are what drift.

### ⚠️ N3 AND N4 INTERACT — do not read a connectivity fix as closing this

A connectivity-aware `decompose` still tie-breaks with `static_score`, the same variable-fraction
heuristic that is wrong in the SAME DIRECTION as N3's estimator (both prefer ground-heavy patterns
regardless of cardinality). **Fixing connectivity removes the CATASTROPHIC case — the 4500x
disconnected first stage — and leaves the ORDINARY one**: among connected candidates, the order is
still chosen by a criterion that cannot see cardinality. Know that before this reads as solved.

## Stage 4 — NOT ANSWERED, and honestly so

`MORK._CARD_REORDER_ENABLED[]` A/B x good/bad conjunct order, all four arms in ONE process:
23.3 / 23.9 / 24.1 / 23.3 ms. Within 3% on a box with documented variance — **no signal**. No
conclusion is drawn from it. Stage 4 only becomes a live question with `decompose=false`, since with
decompose on, N4 has already erased the input order.
