# MGFW Integration — Current State, Spec Intent, and Decision Point

**Date**: 2026-05-30
**Status**: substrate audit complete; integration decision pending
**Related**: `$PRIMUS/docs/specs/mg_framework_spec.md` (spec extract)
**Source paper**: Goertzel, *A Multi-Geometry Hyperon Methods Framework*, April 17, 2026
**Companion audits**: `AUDIT_DOC1.md`, `AUDIT_DOC2.md`, `AUDIT_DOC3.md`

---

## 1. Purpose

This document captures the gap between what the **Multi-Geometry Hyperon
Methods Framework (MGFW)** spec says the `src/mgfw/` subsystem SHOULD be and
what it currently IS, so a future session can pick up the integration work
without re-deriving the picture.

Recommended reading order: §2 (TL;DR) → §3 (spec recap) → §4 (current state) →
§7 (decision point).

---

## 2. TL;DR

**The spec says**: MGFW is the **shared ontology / canonical IR** that every
Hyperon algorithm package registers into. Programmers write declarative
templates (`define-factor-rule`, `define-trie-miner`, …); the supercompiler
consumes the typed templates and lowers via a 9-step pipeline (Algorithm 5).
The TyLA `F ⊣ G` adjunction maps MorkAdapter (F: MeTTa → canonical) and
**MorkSupercompiler (G: canonical → execution)** to the framework's
formal backbone.

**The code says**: `src/mgfw/` is a 2,263-LOC subsystem with all the data
structures present (SemanticObjects, GeometryTemplate, SchemaRegistry,
FactorGeometry, TrieDAGGeometry, MGCompiler) — but nothing else in PRIMUS
imports from it, and `mg_compile` step 8 silently bypasses the templates it
just computed by falling back to plain MM2 `compile_program`. So MGFW is
currently a **leaf scaffold whose framework intent is unfulfilled**.

**The decision** (§7): fix the substrate so MGFW WOULD work if anyone
registered into it, OR fix the substrate AND register one or two real
templates from Core / MORKTensorNetworks as the §15 MVP.

---

## 3. Spec recap — what MGFW is supposed to be

### 3.1 Three artifacts, not one (§2)

The spec distinguishes:

1. **Human/LLM-facing DSL** — authoring surface (`define-factor-rule`,
   `define-bgc-search`, `define-trie-miner`, `define-coercion`,
   `define-cache-contract`)
2. **Canonical normalized schema / registry** — machine-readable internal
   form; the supercompiler / linter / planner consume this
3. **Executor / lowering / backend layer** — where canonical schemas become
   running code (MM2 worklist skeletons, MORK-native kernels, trie ops,
   tensor shard kernels)

**Stability contract**: DSL can evolve; canonical schema is stable enough that
cached supercompilation results remain valid across DSL revisions; backend can
change freely behind the schema boundary.

### 3.2 Five-layer architecture (§6)

| Layer | What |
|---|---|
| 1 — Semantic objects | `Rel`, `Prog`, `Model`, `Codec`, `Sched`, `Stream` |
| 2 — Presentations | `Pres(G, A)` where `G ∈ {Factor, DAG, Trie, TensorSparse, TensorDense, Hybrid}` |
| 3 — Operator schemas & geometry templates | 13-field canonical `GeometryTemplate` record |
| 4 — Transformation and planning | supercompilers / planners / linters operate on Layer 3 |
| 5 — Execution and orchestration | MM2 worklist, MORK kernels, tensor shards |

### 3.3 The 9-step Algorithm 5 (§12.1) — what `mg_compile` should do

```text
Require: MeTTa / MeTTa-IL program region R
 1: Parse and infer semantic objects and effect regions
 2: Attach one or more initial presentations from the framework registry
 3: Compute backend affinities and candidate coercion graph
 4: Run geometry-aware planning and selective supercompilation over presentation space
 5: Choose exact or approximate kernels subject to cost and witness constraints
 6: Choose concurrency and distribution policies from registered families
 7: Select backend(s): PeTTa, MM2, tensor runtime, trie runtime, factor runtime, or hybrid
 8: Lower to residual executable code and runtime metadata
 9: Return mixed executable plus proof or witness artifacts
```

**Step 8 must use the templates from steps 1-7.** Current `mg_compile`
implementation does not (see §4.3 below).

### 3.4 TyLA `F ⊣ G` mapping (§7.3 + Appendix D)

| TyLA role | PRIMUS subsystem |
|---|---|
| `F` functor (operational spec → typed env; MeTTa → canonical) | **MorkAdapter** |
| `G` functor (typed env → canonical operational; canonical → execution) | **MorkSupercompiler** |
| Coercions `Pres(G₁, A) ⇒ Pres(G₂, A)` | registered template transitions |

This is the spec's central architectural claim: MorkSupercompiler IS the `G`
functor. The `G` functor produces an execution plan from typed templates —
which means SCPipeline.execute! is supposed to consume MGFW templates, not
short-circuit them.

### 3.5 PRIMUS cross-reference (Appendix D)

| MGFW concept | PRIMUS subsystem |
|---|---|
| Factor geometry | PLN rules (`lib/pln/pln_core_logic.metta`) + LogicBridge |
| DAG geometry | MOSES / gCoDD (MeTTaGenerator + AxiomCompiler) |
| Trie geometry | MORK PathMap / PathTrie + MORK-Miner |
| Tensor geometry | **MORKTensorNetworks** |
| Trie→Codec coercion | **WILLIAM** (Hyperseed subsystem) |
| Hybrid (DualWorklist + Factor + Trie) | GeodesicBGC-Composite (not yet implemented) |
| Exactness classes (EXACT / BOUNDED / STATISTICAL) | STV/DTV distinction in TV types |
| Cache contracts | ADR-039 OER caching with version-tuple invalidation |
| Noether-charge (evidence conservation) | paraconsistent `tv_negative::Float32` |

### 3.6 §15 MVP — six deliverables

1. **Schema registry + canonical form** with four semantic object classes
   (Rel, Prog, Model, Codec) and four geometry tags (Factor, Trie, DAG, Tensor).
   Every registered template must declare: semantic type, geometry,
   visible/internal ops, effect class, cache contract, exactness class,
   symmetries, concurrency/distribution stubs, backend affinity.
2. **Human/LLM-facing DSL** with 5 forms: `define-factor-rule`,
   `define-trie-miner`, `define-coercion`, `define-exactness`,
   `define-cache-contract`.
3. **Restricted supercompiler bridge** — parse a restricted MeTTa subset to
   M-Core, use `rewrite_once` / `Split` / `PathSig` / `SchemaKey` / global
   folding, residualize back to MeTTa-IL or direct template-runtime call graph.
4. **Exact execution slice**: STV factor geometry — producer/premise/conclusion
   role-labeled factor graphs, backward activation, forward supply on active
   subgraph, frozen boundary caches, version-tuple cache validity, MM2 scheduling.
5. **Approximate execution slice**: DTV Layer 2 — Beta projections + (μ, n)
   Jacobian norm. Tag every coercion/contract with EXACT / BOUNDED(ε) /
   STATISTICAL(confidence).
6. **Trie geometry runtime** — three PathMap stages (seed extraction by
   subtree scan, growth by prefix proximity, scoring via in-place prefix
   counters) + one factor-graph-as-trie encoding example.

**§15.3 explicit non-goals**: full TyLAA verification (SHD proofs deferred);
full MM2 lowering (direct runtime, not general MM2 code generator); full
DAG/ENF/CENF runtime; GPU training; AC/ACU unification; general distributed
execution.

---

## 4. Current state — audit findings (2026-05-30)

### 4.1 Inventory

`src/mgfw/` — 6 files, 2,263 LOC:

| File | LOC | What it claims |
|---|---:|---|
| `SemanticObjects.jl` | 237 | Six geometry-neutral base types (Layer 1) |
| `FactorGeometry.jl` | 339 | Factor presentation (Layer 2) + specialize_exact / approximate |
| `GeometryTemplate.jl` | 379 | 13-field canonical template record (Layer 3) |
| `SchemaRegistry.jl` | 376 | GLOBAL_REGISTRY + lookup / register |
| `TrieDAGGeometry.jl` | 411 | Trie miner stages + DAG ENF normalize |
| `MGCompiler.jl` | 521 | Algorithm 5 pipeline (Layer 4 → Layer 5) |

Test file: `test/mgfw/test_mgfw.jl` (360 LOC — the largest test file in the
package).

### 4.2 Module load order (per `MorkSupercompiler.jl`)

`SemanticObjects.jl` loads at line 81 *before* `using HPC` (line 84) and the
MORKTensorNetworks import (line 101); the rest of mgfw loads after at lines
105–109. Type-only file loads first; the others depend on HPC + tensor
semirings.

### 4.3 What's real

- **Data layer**: 6 semantic kinds × 5 geometry tags × 14-field
  GeometryTemplate × policy/concurrency/distribution contracts × 4 minimum
  coercions × 5-form DSL — all present, all tested by `test_mgfw.jl`.
- **`backend_neutral_optimize`** (`MGCompiler.jl:279-307`) — real template
  ranking by affinity score.
- **`affinity_analysis`** / **`select_backend`** — real heuristics from §9
  ("backend affinity").
- **Trie miner** (`TrieDAGGeometry.jl:§15.6` analog) — real seed-scan +
  growth + counter implementation.
- **`dag_normalize!`** ENF pass — real, tested.

### 4.4 What's broken or unwired

#### Bug 1 — `mg_compile` step 8 ignores the templates

`MGCompiler.jl:345-353` accepts `optimized_templates` and `coercions`
computed by steps 1-7, then **discards them** and routes through plain
`compile_program` (the regular MM2 codegen). The "geometry-aware compilation
pipeline" is **geometry-blind at the code-emission boundary**.

The test at `test_mgfw.jl:358` openly acknowledges this with the comment:

> "Space size unchanged or larger (no guarantee of new atoms from IR stub)"

This is a direct violation of Algorithm 5 step 8 ("Lower to residual
executable code AND runtime metadata") — the runtime metadata derived from
the templates is dropped.

#### Bug 2 — `GLOBAL_REGISTRY` is empty at module load

`SchemaRegistry.jl:106-109` defines `__init_registry__` which populates
GLOBAL_REGISTRY with the default coercions and template families. But Julia
auto-invokes only `__init__`, not arbitrary `__init_*`. So
`__init_registry__` is **never called**, and `GLOBAL_REGISTRY` stays empty
unless a caller explicitly initializes it.

Downstream impact: `mg_compile`'s default `registry=GLOBAL_REGISTRY` at
`MGCompiler.jl:290` operates on an empty registry; `_attach_presentations`
at `:419-439` always hits the "no candidate found" branch and synthesizes
auto-templates at `:433`. Real templates are never matched.

#### Bug 3 — TyLAA concurrency verification can never fail

`_apply_concurrency_policies!` + `_template_effect_kind` at
`MGCompiler.jl:470-496` is a guaranteed-pass: the effect-kind classifier
checks for symbols (`:never`, `:read_only`, `:append_only`, `:always`) that
never appear in the `commutes_when` fields populated by
`default_local_concurrency` at `GeometryTemplate.jl:86-124` (which uses
domain-specific tags like `:disjoint_targets_or_monotone_join`,
`:disjoint_factor_neighborhoods`, etc.).

Every template falls through to `EFF_READ`, all pairs commute, no
violations are ever produced — the TyLAA verification step (spec §7.4 +
Appendix C) is dead.

#### Bug 4 — Algorithm 1 / Algorithm 2 are wrapper-only

`FactorGeometry.specialize_exact` / `specialize_approximate` at
`FactorGeometry.jl:105-185` build a `SpecializedRegion` metadata record
without producing executable kernels. `_specialize_kernels!` at `:267-276`
admits in source comment: "In full implementation: replace generic map with
STV-specific fast path." `_get_truth_family` at `:305-307` hardcodes `:STV`.

The approximation test at `:196-211` is vacuous — passes with
`noether_charge == 1.0` from an empty witness, which is exactly what
happens for the 3-node test graph.

#### Bug 5 — Algorithm 3 mutation is a one-line head-rename

(Re-check / not yet confirmed by audit — flagged for future investigation.)

#### Bug 6 — No downstream caller

`grep -rn "<MGFW symbol>"` over `src/` returns zero non-MGFW hits for:
`mg_compile`, `mg_run`, `GeometryTemplate`, `GLOBAL_REGISTRY`,
`evolve_demes`, `run_trie_miner`, `specialize_exact`,
`specialize_approximate`, `FactorGraph`, `DAGStore`, `PatternTrie`,
`SchemaRegistry`, `build_geodesic`, `affinity_analysis`, `select_backend`.

The subsystem is INCLUDED by the module loader (lines 81 + 105-109) and the
14 main symbols are re-exported (lines 268-294), but no other layer
depends on it. It is currently a leaf.

### 4.5 What this means

The audit's surface finding ("MGFW is a leaf, nothing depends on it") is
*technically true today* but is **the inverse of the spec's design**. The
spec's intent is that the other subsystems should be registering INTO MGFW.
The current state reflects:

- Substrate bugs preventing the framework from working even if registrations
  existed (Bugs 1-3).
- Missing integration work — no algorithm package currently produces
  registry entries to feed in (Bug 6).

---

## 5. The TyLA `G` functor gap

This is a separate but related finding from the same audit:

`SCPipeline.execute!` (`src/integration/SCPipeline.jl:100-207`) is the
top-level entry point the spec maps to the TyLA `G` functor (Appendix D:
"MorkSupercompiler = G functor"). Its job per Algorithm 5 is to consume
typed templates and lower to a residual execution plan.

**Current implementation**: loaded but never invokes Stepper / CanonicalKeys
/ BoundedSplit — the §6 supercompiler core. What `execute!` actually does:

1. Stage 1 STATS → `collect_stats`
2. Stage 2 PLAN → `plan_program` + `plan_report` (Phase 0 §5)
3. Stage 2b APPROX → `run_approx_pipeline` (optional)
4. Stage 3 DECOMPOSE → `decompose_program` (Rule-of-64 fix, Phase 1)
5. Stage 4 SATURATE → builds throwaway `MCoreGraph`, runs `saturate!`,
   discards it (Phase 3 §7) — **observable runtime effect: zero**
6. Stage 4b COMPILE → optional `compile_program` via MM2Compiler — feeds the
   compiled MM2 exec atoms back into `space_metta_calculus!`
7. Stage 5 EXECUTE → `space_add_all_sexpr!` + `space_metta_calculus!`

So a user calling `run!` gets:
- Reordered MORK calculus (planner + decompose)
- NOT supercompilation in the §6 sense (no Stepper driver, no folding,
  no canonical-key whistle, no bounded split)
- NOT MGFW-mediated lowering (no template registration, no coercion graph,
  no exactness tags)

This is the same architectural gap as MGFW: the `G` functor is wired up at
the file level but doesn't actually do the `G` functor's work.

---

## 6. What it would take to close the gap

### Path A — Substrate fixes only (~150 LOC + tests)

Close the bugs in §4.4 so MGFW WOULD work if anyone registered into it.

| Item | Where | Effort |
|---|---|---|
| Fix `mg_compile` step 8 to honor `optimized_templates` and `coercions` | `MGCompiler.jl:345-353` | ~50 LOC |
| Auto-invoke `__init_registry__` from `__init__` | `SchemaRegistry.jl` | ~5 LOC |
| Align `_template_effect_kind` symbols with what `default_local_concurrency` actually produces | `MGCompiler.jl:470-496` + `GeometryTemplate.jl:86-124` | ~30 LOC |
| Tests for the above | `test/mgfw/test_mgfw.jl` | ~60 LOC |

Does NOT unblock anything externally because no algorithm package produces
registry entries yet.

### Path B — Substrate + first integration (~400 LOC + tests)

Path A plus the §15 MVP demonstration: register one factor-geometry PLN
rule and one trie-geometry MORK miner as real templates in
GLOBAL_REGISTRY, and wire `mg_compile` to use them end-to-end on a small
fixture.

| Item | Where | Effort |
|---|---|---|
| Path A | — | ~150 LOC |
| Register `HeuristicModusPonens` STV factor template (spec §10.1) | new `src/mgfw/templates/pln_stv.jl` | ~80 LOC |
| Register `FactorGraphMotifMiner` trie template (spec §10.3) | new `src/mgfw/templates/motif_miner.jl` | ~60 LOC |
| Wire `mg_run!` to dispatch to either template depending on input | `MGCompiler.jl` | ~40 LOC |
| Two end-to-end tests (one per geometry) | `test/mgfw/` | ~70 LOC |

This is what spec §15.4 calls out as MVP demonstrations.

### Path C — Substrate + integration + close the `G` functor gap (~700 LOC)

Path B plus wiring `SCPipeline.execute!` to actually consume the registry
templates and invoke the §6 supercompiler core (Stepper drives configs,
CanonicalKeys folds via subsumption, BoundedSplit handles divergence).

This is the most architecturally honest option — it makes the TyLA
`F ⊣ G` mapping (Appendix D) hold up in code. But it's a substantive
refactor of the Layer 7 integration files.

| Item | Where | Effort |
|---|---|---|
| Path B | — | ~400 LOC |
| `SCPipeline.execute!` invokes Stepper / CanonicalKeys / BoundedSplit when `opts.supercompile=true` | `src/integration/SCPipeline.jl` | ~150 LOC |
| Convert saturated `MCoreGraph` from throwaway to first-class result; thread through to Stage 5 | `SCPipeline.jl:152-173` | ~50 LOC |
| Bisimulation discharge (MM2Compiler obligations actually checked, not just recorded) | `src/codegen/MM2Compiler.jl` | ~80 LOC |
| End-to-end tests on a 3-rule benchmark | `test/integration/` | ~120 LOC |

---

## 7. Decision point — which path?

The substrate work in Path A is genuinely small and removes a class of
"the framework is silently broken" hazards. It does NOT unblock anything
externally — by itself it's hygiene work.

Path B is the smallest amount of work that produces a **runnable §15 MVP
demonstration**. That's what the spec explicitly defines as "minimum
viable framework."

Path C closes the architectural debt the audit surfaced — the `G` functor
mapping in Appendix D becomes load-bearing in code, not just documentation.
But it's a non-trivial refactor and only worth doing once an algorithm
workload (WILLIAM, MetaMo, MOSES) actually needs the supercompiler to do
work beyond the current planner + decompose flow.

**Recommendation order if budget is the constraint**:

1. Path A first (no commitment to integration).
2. Path B when WILLIAM or MetaMo session work needs a factor or trie
   geometry template.
3. Path C only when a workload requires the §6 supercompiler core to
   actually drive configurations.

---

## 8. Open questions for the spec author

- Spec §15.2 deliverable 3 mentions `PathSig` / `SchemaKey` — these are
  partial in `src/supercompiler/CanonicalKeys.jl` (audit found the KB-sig
  path was dead-code; fixed in commit `e3ef130`). Are the schema-key
  semantics now sufficient for the MVP bridge, or does the spec want
  additional fields (e.g., the `(schema-id, factor-id, subst-shape,
  evidence-ver)` 4-tuple from §12.2 Step 2)?
- Spec §15.4 demo 2 says "STV factor path returns same result as reference
  interpreter" — what's the reference interpreter for STV factor geometry?
  PLN's `lib/pln/pln_core_logic.metta` runs in MORK, not MGFW. Should
  PRIMUS adopt PLN as the reference for MVP acceptance, or stand up a
  separate STV reference?
- Spec §11.4 ("When no template fits") allows hybrids and novel geometries.
  Is the GeodesicBGC-Composite (Appendix A) a registered hybrid or a
  newly-defined one? It composes DualWorklist + Factor + Trie — but
  `DualWorklist` isn't in the four MVP geometry tags (Factor / Trie / DAG /
  Tensor). Should DualWorklist be added as a fifth tag, or modeled as a
  Sched semantic object?

---

## 9. References

- Spec extract: `$PRIMUS/docs/specs/mg_framework_spec.md`
- Audit findings: `docs/AUDIT_DOC1.md`, `docs/AUDIT_DOC2.md`, `docs/AUDIT_DOC3.md`
- Related fixes: PathMap commits `460f9b1`, `cf77708`, `a3bd948`;
  Supercompiler commit `e3ef130`; MORKTensorNetworks commit `2e25204`
- MM2 spec (companion): `$PRIMUS/docs/specs/mm2_supercompiler_spec.md`
- Approx-SC spec (companion): `$PRIMUS/docs/specs/approximate_metta_supercompilation_spec.md`
- Source paper: `docs/Super compiler/mg_framework_design_doc_v7.pdf` (Goertzel,
  April 17, 2026; 51 pages)
