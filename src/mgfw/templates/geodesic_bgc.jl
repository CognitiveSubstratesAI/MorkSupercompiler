"""
GeodesicBGC-Composite — §12.2 Goertzel "Geodesic Inference Control" worked example.

Implements the spec's canonical multi-geometry hybrid: DualWorklist scheduler +
Factor guidance + Trie evidence. The composite template is the demonstration
that the framework can compose three orthogonal geometries under a single
typed contract — what §11.4 calls a registered hybrid (not a novel geometry).

Components (§12.2):
  :scheduler  → GeodesicBGC-Worklist
                priority = Δ(log f + log g) / cost   (eq. 4.1)
                splice via PathMap probe
  :guidance   → FactorFGSurrogate
                factor-message-passing over rule graph; STV truth family
  :evidence   → EvidenceCapsule (Trie)
                CRDT-style sketch with KMV-128 + Noether-charge evidence_mass

Data flow (4 edges, §12.2 step 4):
  scheduler ↔ guidance:  active-subgraph-query / fg-score-update
  scheduler ↔ evidence:  capsule-transport     / overlap-veto

This file builds on `build_geodesic_bgc_composite(reg)` already in
`MGCompiler.jl` — that returns the GeometryTemplate; here we register it
in GLOBAL_REGISTRY at module load AND attach a lowering function so
`mg_compile` step 8 can emit the routing skeleton.
"""

"""
    geodesic_bgc_lowering(t, region) → String

Spec §12.2 step 4: emit the residual routing skeleton — three sub-templates
wired via four data-flow channels. The lowering is structural: it does NOT
compile a specific BGC search instance, but emits the MeTTa-level scaffolding
that downstream rule-DSL forms (`define-bgc-search`) would plug into.

The emitted residual is consumed by `space_metta_calculus!` at execution
time. Each `(exec ...)` block is a separate scheduled rewrite that fires
when its premise pattern is in the space.
"""
function geodesic_bgc_lowering(t::GeometryTemplate, region::AbstractString) :: String
    """
    ;; mgfw:lowering GeodesicBGC_Composite
    ;; §12.2 hybrid: scheduler + guidance + evidence
    ;; Edge 1: scheduler → guidance (active-subgraph-query)
    (exec (bgc-stage scheduler-to-guidance)
          (, (bgc-frontier \$x))
          (, (active-subgraph \$x)))
    ;; Edge 2: guidance → scheduler (fg-score-update)
    (exec (bgc-stage guidance-to-scheduler)
          (, (fg-score \$x \$score))
          (, (bgc-priority \$x \$score)))
    ;; Edge 3: scheduler → evidence (capsule-transport)
    (exec (bgc-stage scheduler-to-evidence)
          (, (bgc-frontier \$x))
          (, (capsule-mint \$x)))
    ;; Edge 4: evidence → scheduler (overlap-veto)
    (exec (bgc-stage evidence-to-scheduler)
          (, (capsule-overlap \$x \$y))
          (, (bgc-veto \$x \$y)))
    """
end

export geodesic_bgc_lowering
