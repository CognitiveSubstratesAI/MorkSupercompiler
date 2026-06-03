# MorkSupercompiler.jl

A formally-grounded supercompiler for MeTTa+MM2, implemented in Julia.

Implements the full specification from three design documents by Ben Goertzel (Oct–Apr 2025/2026):

| Document | Algorithms | Status |
|----------|-----------|--------|
| *A MORK-Native Supercompiler for MeTTa+MM2* (Oct 2025) | 14 | ✅ Complete — audited `docs/AUDIT_DOC1.md` |
| *Approximate Supercompilation for MeTTa+MM2* (Oct 2025) | 7 | ✅ Complete — audited `docs/AUDIT_DOC2.md` |
| *A Multi-Geometry Hyperon Methods Framework* (Apr 2026) | 5 | ✅ Complete — audited `docs/AUDIT_DOC3.md` |

**Repo**: [sivaji1012/Supercompiler](https://github.com/sivaji1012/Supercompiler)  
**Depends on**: [MORK](https://github.com/CognitiveSubstratesAI/MORK) + [PathMap](https://github.com/CognitiveSubstratesAI/PathMap) + MORKTensorNetworks + HPC  
**Tests**: 916 across ~30 test files, all passing

---

## What runs on the live execution path

The algorithms above are **implemented and unit-tested**, but "✅ Complete" means
*algorithm-complete*, not *all load-bearing*. To be precise about the shipped runtime:

**On the live path** (what `SCPipeline.execute!` actually transforms + runs): an MM2
**query planner** + **Rule-of-64 pipeline decomposer** + **approximate-rewrite
pipeline** + **geometry-aware MM2 lowering**. The compiler→runtime seam is
encoding-sound — `compile_program` output runs through `space_metta_calculus!` and
derives the expected atoms (`test/integration/test_mm2_roundtrip.jl`). (That validates
the encoding/priority *shape*, not the §9.3 Priority-Control Equivalence theorem
itself, which is about ordering across multiple priorities.)

**Built and unit-tested, but NOT yet load-bearing** (runs into a throwaway graph, or
records without discharging):
- the §6 driving/folding core (`Driver.drive!`) — `drive_results` are recorded for
  inspection; execution still proceeds via `space_metta_calculus!`,
- KB saturation (`KBSaturation.saturate!`) — derives onto a local `MCoreGraph`, never
  written back to the Space,
- the proof surfaces — MM2 `BiSimObligation`s, the approximate Phase-4 tolerance check
  (self-consistent arithmetic vs *assigned* error, not measured), and MGCompiler
  `proof_artifacts`/TyLAA `@warn` — **record or assert; they do not prove or gate**.

Wiring `drive!`/`saturate!` into the executed program — and discharging the proof
obligations against an exact reference — is scheduled phase work, gated on building a
verification surface that does not exist yet. See `docs/AUDIT_DOC1.md` + the 2026-06-03 audit.

---

## Quick Start

```julia
using MorkSupercompiler, MORK

# Load facts into a MORK Space
s = new_space()
space_add_all_sexpr!(s, "(edge 0 1) (edge 1 2) (edge 2 3)")

# Plan + execute: stats → join-order → execute
result = execute!(s, raw"(exec 0 (, (edge $x $y) (edge $y $z)) (, (path $x $z)))")
println(timing_report(result))

# Explain the join plan chosen
println(explain(s, raw"(exec 0 (, (edge $x $y) (edge $y $z)) (, (path $x $z)))"))

# Approximate supercompilation with error budget
result2 = run_approx_pipeline(s, raw"(exec 0 (, (edge $x $y) (edge $y $z)) (, (path $x $z)))";
                               error_tolerance=0.05)

# MG Framework: define a factor rule and register it
r = define_factor_rule(
    name        = :TransitivityRule,
    premises    = [:Ancestor_x_y, :Ancestor_y_z],
    conclusion  = [:Ancestor_x_z],
    truth_family= :STV)
```

### Development REPL (warm — no cold-start, Revise-enabled)

```bash
julia --project=. -i tools/repl.jl
```
```julia
t()                   # run all 580+ tests without restarting
plan(prog)            # static source reordering
report(facts, prog)   # join-plan explanation with cardinalities
run(facts, prog, n)   # plan + execute n steps
```

---

## Architecture

Nine layers covering all three specification documents:

```
src/
  frontend/      SExpr.jl              — s-expression parser
  core/          MCore.jl              — 11 M-Core IR node types + MCoreGraph
                 Effects.jl            — Effect algebra (Algorithm 1)
  planner/       Selectivity.jl        — static + O(1) dynamic cardinality
                 Statistics.jl         — MORKStatistics (all 6 spec fields)
                 QueryPlanner.jl       — Algorithm 6 EffectAwarePlanning
  rewrite/       Rewrite.jl            — source reorderer (join-order)
  supercompiler/ Stepper.jl            — Algorithm 7+8 (RewriteOnce, CallPrimitive)
                 CanonicalKeys.jl      — Algorithm 10 (KeySubsumption + FoldTable)
                 BoundedSplit.jl       — Algorithm 9 (Rule-of-64 fix)
                 KBSaturation.jl       — Algorithm 11 (IncrementalSaturation)
                 EvoSpecializer.jl     — Algorithms 12+13+5+7
  codegen/       MM2Compiler.jl        — Algorithm 14 (BisimulationProof + MM2)
  integration/   SCPipeline.jl         — execute! end-to-end pipeline
                 Profiler.jl           — baseline vs planned speedup
                 Explainer.jl          — explain + to_dot + diff_programs
                 AdaptivePlanner.jl    — Algorithm 5 drift-based replanning
  approx/        PBoxAlgebra.jl        — Doc 2 §2: AddPBox + Fréchet + theorems
                 UncertainQuery.jl     — Doc 2 §3: cost model + ApproximateSplit
                 UncertainInference.jl — Doc 2 §4: AND/OR + MatchWithUncertainty + ApplyRule
                 ApproxMOSES.jl        — Doc 2 §5: TournamentWithPBox + convergence
                 ApproxPipeline.jl     — Doc 2 §6: ApproxIndex + 4-phase pipeline
  mgfw/          SemanticObjects.jl    — Doc 3 §6.1-6.2: Pres(G,A) + TyLA coercions
                 GeometryTemplate.jl   — Doc 3 §6.3+§13: 13-field template + contracts
                 SchemaRegistry.jl     — Doc 3 §8+§11: registry + DSL + Algorithm 4
                 FactorGeometry.jl     — Doc 3 §10.1: Algorithms 1+2 + STV
                 TrieDAGGeometry.jl    — Doc 3 §10.2-10.3: Algorithm 3 + trie mining
                 MGCompiler.jl         — Doc 3 §9+§12: Algorithm 5 + late commitment
```

---

## Implemented Algorithms

### Doc 1 — MM2 Supercompiler Spec (14 algorithms)

| # | Algorithm | File |
|---|-----------|------|
| 1 | EffectCommutes | `core/Effects.jl` |
| 2 | EstimatePatternCardinality | `planner/Statistics.jl` |
| 3 | PrefixSampling | `planner/Statistics.jl` |
| 4 | UpdateIncrementalStats | `planner/Statistics.jl` |
| 5 | ShouldReplan | `integration/AdaptivePlanner.jl` |
| 6 | EffectAwarePlanning | `planner/QueryPlanner.jl` |
| 7 | RewriteOnce | `supercompiler/Stepper.jl` |
| 8 | CallPrimitive | `supercompiler/Stepper.jl` |
| 9 | BoundedSplit ← *Rule-of-64 fix* | `supercompiler/BoundedSplit.jl` |
| 10 | KeySubsumption | `supercompiler/CanonicalKeys.jl` |
| 11 | IncrementalSaturation | `supercompiler/KBSaturation.jl` |
| 12 | GatedEvolutionarySpecialization | `supercompiler/EvoSpecializer.jl` |
| 13 | CanReuseFitnessCache | `supercompiler/EvoSpecializer.jl` |
| 14 | BisimulationProof | `codegen/MM2Compiler.jl` |

### Doc 2 — Approximate Supercompilation Spec (7 algorithms)

| # | Algorithm | File |
|---|-----------|------|
| 1 | AddPBox (independent + Fréchet-Hoeffding) | `approx/PBoxAlgebra.jl` |
| 2 | EstimateCardinalityPBox (Hoeffding-bounded) | `approx/UncertainQuery.jl` |
| 3a | ApproximateSplit (error-tolerance-gated) | `approx/UncertainQuery.jl` |
| 3b | MatchWithUncertainty (structural similarity) | `approx/UncertainInference.jl` |
| 4 | ApplyRule / UncertainModusPonens | `approx/UncertainInference.jl` |
| 5 | ApproximateFitness (Hoeffding bound + 5% tail) | `supercompiler/EvoSpecializer.jl` |
| 6 | TournamentWithPBox (Monte Carlo, 100 trials) | `approx/ApproxMOSES.jl` |
| 7 | AllocateEvaluations (value of information) | `supercompiler/EvoSpecializer.jl` |

Also: Theorem A.2 (error composition), Lemma A.4 (Fréchet width), Lemma A.5 (Hoeffding).

### Doc 3 — Multi-Geometry Framework Spec (5 algorithms)

| # | Algorithm | File |
|---|-----------|------|
| 1 | Exact factor-geometry specialization (8 steps) | `mgfw/FactorGeometry.jl` |
| 2 | Approximate factor specialization (7 steps) | `mgfw/FactorGeometry.jl` |
| 3 | Canonical DAG evolutionary loop (8 steps) | `mgfw/TrieDAGGeometry.jl` |
| 4 | Human/LLM authoring workflow (8 steps) | `mgfw/SchemaRegistry.jl` |
| 5 | Geometry-aware compilation pipeline (9 steps) | `mgfw/MGCompiler.jl` |

---

## Key Data Structures

| Structure | Spec | File |
|-----------|------|------|
| 11 M-Core node types + `MCoreGraph` | Doc 1 §3 | `core/MCore.jl` |
| `MORKStatistics` (6 fields) | Doc 1 §5.1.1 | `planner/Statistics.jl` |
| `CanonicalPathSig` + `FoldTable` | Doc 1 §6.3 | `supercompiler/CanonicalKeys.jl` |
| `PBox` (4 fields incl. `correlation_sig`) | Doc 2 §2.2 | `core/MCore.jl` |
| `UncertainFact` + `ProofTree` | Doc 2 §4.1 | `approx/UncertainInference.jl` |
| `ApproximatePathSig` (EXACT/BOUNDED/STATISTICAL) | Doc 2 §6.4 | `approx/ApproxPipeline.jl` |
| `GeometryTemplate` (14 fields incl. noether_charge) | Doc 3 §6.3 | `mgfw/GeometryTemplate.jl` |
| `LocalConcurrencyContract` (8 fields) | Doc 3 §13.1 | `mgfw/GeometryTemplate.jl` |
| `DistributedExecContract` (10 fields) | Doc 3 §13.2 | `mgfw/GeometryTemplate.jl` |
| `SchemaRegistry` + 5 DSL forms | Doc 3 §8 | `mgfw/SchemaRegistry.jl` |

---

## Tests

580+ tests across 22 files organized by layer:

```
test/
  runtests.jl                 — master runner
  frontend/                   test_sexpr.jl
  core/                       test_mcore.jl  test_effects.jl
  planner/                    test_statistics.jl  test_selectivity.jl  test_query_planner.jl
  rewrite/                    test_rewrite.jl
  supercompiler/              test_stepper.jl  test_canonical_keys.jl  test_bounded_split.jl
                              test_kb_saturation.jl  test_evo_specializer.jl
  codegen/                    test_mm2_compiler.jl
  integration/                test_pipeline.jl  test_profiler.jl  test_explainer.jl
                              test_adaptive_planner.jl
  approx/                     test_pbox_algebra.jl  test_uncertain_query.jl
                              test_uncertain_inference.jl  test_approx_moses.jl
                              test_approx_pipeline.jl
  mgfw/                       test_mgfw.jl
```

---

## Public API

### Pipeline (Doc 1 integration layer)

```julia
execute!(s, program; opts)           # stats → plan → execute → SCResult
execute(facts, program; steps)       # fresh space + run
explain(s, program)              # join order + cardinalities + variable flow
to_dot(program)                     # Graphviz DOT source
profile(facts, program; steps)   # baseline vs planned speedup
speedup_report(profile)             # formatted table
ap = AdaptivePlan(s, program)
sc_run_adaptive!(s, ap; steps)      # auto-replan on cardinality drift
```

### Query Planner

```julia
collect_stats(s; sample_frac)       # MORKStatistics (Algorithms 2+3)
plan_static(program)             # static source reordering
plan_program(s, program)         # dynamic (uses btm prefix counts)
plan_query(sources, stats)          # Algorithm 6: sources + EffectBarriers
```

### Supercompiler Core

```julia
rewrite_once(g, id, env, deps, reg) # Algorithm 7 RewriteOnce
bounded_split(g, id, env, stats)    # Algorithm 9 BoundedSplit (Rule-of-64 fix)
canonical_key(g, id, depth)         # CanonicalPathSig for termination
subsumes(key1, key2)                # Algorithm 10 KeySubsumption
saturate!(kb; max_rounds)           # Algorithm 11 IncrementalSaturation
compile_program(g, root_ids)        # Algorithm 14 → MM2 exec s-expressions
```

### Approximate Supercompilation (Doc 2)

```julia
add_pbox(X, Y)                      # Algorithm 1: independent or Fréchet
match_with_uncertainty(pat, fact, ε)# Algorithm 3b: structural similarity
apply_rule(premise, rule_str, depth)# Algorithm 4: UncertainModusPonens
tournament_with_pbox(candidates, n) # Algorithm 6: MC tournament selection
run_approx_pipeline(s, program; ε)  # 4-phase approximate compilation
```

### MG Framework (Doc 3)

```julia
define_factor_rule(; name, ...)     # DSL → canonical GeometryTemplate
define_trie_miner(; name, ...)      # DSL → trie geometry template
mg_compile(region, registry)        # Algorithm 5: 9-step compilation
mg_run!(s, region)                  # compile + execute
specialize_exact(query, graph)      # Algorithm 1: exact factor specialization
specialize_approximate(query, graph, ε) # Algorithm 2: approximate with witness
evolve_demes!(demes, fitness_fn)    # Algorithm 3: canonical DAG evolution
run_trie_miner(template, data)      # 3-stage trie mining (§15.6)
```

---

## Local Setup

```bash
cd packages/MorkSupercompiler
julia --project=. -e '
  import Pkg
  Pkg.develop([Pkg.PackageSpec(path="../PathMap"),
               Pkg.PackageSpec(path="../MORK")])
'
```

---

## Documentation

| Document | Contents |
|----------|----------|
| `docs/ARCHITECTURE.md` | Layer diagram, data flow, type hierarchy, effect commutativity table |
| `docs/AUDIT_DOC1.md` | Gap analysis vs MM2 Supercompiler Spec — 5 gaps found+fixed |
| `docs/AUDIT_DOC2.md` | Gap analysis vs Approximate Supercompilation Spec — 0 blocking gaps |
| `docs/AUDIT_DOC3.md` | Gap analysis vs MG Framework Spec — 7 gaps found+fixed |

---

## What's Next

1. **Benchmark validation** — run `profile` on canonical Rule-of-64 cases (odd_even_sort, counter_machine, transitive_detect) and measure actual speedup from join-order optimization
2. **MORK integration** — wire `BoundedSplit` + nested-loop join into `space_metta_calculus!` to replace `ProductZipper` for multi-source patterns
