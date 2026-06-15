# Architecture

MorkSupercompiler is layered: surface syntax lowers into an effect-typed core IR, which a
query planner + source rewriter + supercompiler transform, which a code generator lowers to
MM2 for execution on a MORK `Space`. The full reference is
[`docs/ARCHITECTURE.md`](https://github.com/CognitiveSubstratesAI/MorkSupercompiler/blob/main/docs/ARCHITECTURE.md).

## Layer map

```
┌─────────────────────────────────────────────────────────────┐
│  User code / warm REPL (tools/sc_repl.jl)                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 7 — Integration  (src/integration/)                   │
│  SCPipeline · Profiler · Explainer · AdaptivePlanner         │
├─────────────────────────────────────────────────────────────┤
│  Layer 6 — Code Generation  (src/codegen/)                   │
│  MM2Compiler (Algorithm 14, bisimulation, priority encoding) │
├─────────────────────────────────────────────────────────────┤
│  Layer 5 — Core Supercompiler  (src/supercompiler/)          │
│  Stepper · CanonicalKeys · BoundedSplit                      │
│  KBSaturation · EvoSpecializer                               │
├─────────────────────────────────────────────────────────────┤
│  Layer 4 — Source Rewriting  (src/rewrite/)                  │
│  Rewrite (join-order reordering of `,` source lists)         │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — Query Planner  (src/planner/)                     │
│  Selectivity · Statistics (Algorithms 2-5) · QueryPlanner    │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — Core IR + Effect algebra  (src/core/)             │
│  MCore (11 node types) · Effects (Algorithm 1)               │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — Surface Syntax  (src/frontend/)                   │
│  SExpr (M-Core parser)                                        │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Geometry Framework (mgfw)

`src/mgfw/` is the algorithmic answer to the Rule-of-64 — it represents knowledge under
multiple *geometries* (factor graph, trie, sparse tensor, DAG) with registered, exactness-
typed coercions between them, and specializes a query to the cheapest geometry that still
answers it exactly (`specialize_exact`) or within a witnessed error bound
(`specialize_approximate`).

The factor-geometry path carries the project's PLN work: `FactorGeometry.jl` (STV / PBox
truth values, role-labeled factor edges, backward-demand activation) and the
`templates/` registry (`pln_stv.jl`, `motif_miner.jl`, `references.jl`). The PLN truth-value
math reference is Core's `lib/pln`; see the package's PLN Layer-1 build notes.

## Audits

Finding-by-finding records live in-repo:
[`AUDIT_DOC1`](https://github.com/CognitiveSubstratesAI/MorkSupercompiler/blob/main/docs/AUDIT_DOC1.md) ·
[`AUDIT_DOC2`](https://github.com/CognitiveSubstratesAI/MorkSupercompiler/blob/main/docs/AUDIT_DOC2.md) ·
[`AUDIT_DOC3`](https://github.com/CognitiveSubstratesAI/MorkSupercompiler/blob/main/docs/AUDIT_DOC3.md) ·
[`MGFW_INTEGRATION`](https://github.com/CognitiveSubstratesAI/MorkSupercompiler/blob/main/docs/MGFW_INTEGRATION.md).
