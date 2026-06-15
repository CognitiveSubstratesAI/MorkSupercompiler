# MorkSupercompiler.jl

A formally-grounded supercompiler for **MeTTa + MM2**, implemented in Julia, plus the
**Multi-Geometry Framework (mgfw)** — an algorithmic answer to the Rule-of-64 blowup.

It implements the algorithms specified across three design documents by Ben Goertzel
(Oct 2025 – Apr 2026):

| Document | Algorithms | In-repo audit |
|----------|-----------|---------------|
| *A MORK-Native Supercompiler for MeTTa+MM2* (Oct 2025) | 14 | [`docs/AUDIT_DOC1.md`](https://github.com/CognitiveSubstratesAI/MorkSupercompiler/blob/main/docs/AUDIT_DOC1.md) |
| *Approximate Supercompilation for MeTTa+MM2* (Oct 2025) | 7 | [`docs/AUDIT_DOC2.md`](https://github.com/CognitiveSubstratesAI/MorkSupercompiler/blob/main/docs/AUDIT_DOC2.md) |
| *A Multi-Geometry Hyperon Methods Framework* (Apr 2026) | 5 | [`docs/AUDIT_DOC3.md`](https://github.com/CognitiveSubstratesAI/MorkSupercompiler/blob/main/docs/AUDIT_DOC3.md) |

**Depends on** [MORK](https://github.com/CognitiveSubstratesAI/MORK) +
[PathMap](https://github.com/CognitiveSubstratesAI/PathMap) +
[MORKTensorNetworks](https://github.com/CognitiveSubstratesAI/MORKTensorNetworks) + HPC,
declared as sibling path-`[sources]` (cloned next to this repo for standalone dev / CI).

## What runs on the live execution path

The algorithms are **implemented and unit-tested**, but "complete" means
*algorithm-complete*, not *all load-bearing*. To be precise about the shipped runtime:

**On the live path** (what `SCPipeline.execute!` actually transforms + runs): an MM2
**query planner** + **Rule-of-64 pipeline decomposer** + **approximate-rewrite pipeline** +
**geometry-aware MM2 lowering**. The compiler→runtime seam is encoding-sound —
`compile_program` output runs through `space_metta_calculus!` and derives the expected
atoms (`test/integration/test_mm2_roundtrip.jl`).

**Built and unit-tested, but NOT yet load-bearing** (runs into a throwaway graph, or
records without discharging): the §6 driving/folding core (`Driver.drive!`), KB saturation
(`KBSaturation.saturate!`), and the proof surfaces (MM2 `BiSimObligation`s, the approximate
Phase-4 tolerance check, MGCompiler `proof_artifacts` / TyLAA `@warn`) — they record or
assert; they do not prove or gate. Wiring those into the executed program — and discharging
the obligations against an exact reference — is scheduled phase work.

See the [Architecture](architecture.md) page for the layer map.

## Quick start

```julia
using MorkSupercompiler, MORK

# Load facts into a MORK Space
s = new_space()
space_add_all_sexpr!(s, "(edge 0 1) (edge 1 2) (edge 2 3)")

# Plan + execute: stats → join-order → execute
result = execute!(s, raw"(exec 0 (, (edge $x $y) (edge $y $z)) (, (path $x $z)))")
println(timing_report(result))
```

For static planning without a live Space:

```julia
program′ = plan_static(program)         # pure-string join reorder
space_add_all_sexpr!(s, program′)
space_metta_calculus!(s, max_steps)
```
