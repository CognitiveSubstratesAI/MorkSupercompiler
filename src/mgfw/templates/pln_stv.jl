"""
PLN STV Factor template — §15.4 MVP demonstration 2.

Implements the spec §10.1.2 / §15.4 demo: a registered factor-geometry
template for the canonical PLN HeuristicModusPonens rule (premises A and
(implies A B); conclusion B) under STV (Simple Truth Value) truth-family.

This is one of two templates that close the §15 MVP gap — the other is
the trie-geometry FactorGraphMotifMiner (`motif_miner.jl`).

Status of the supercompiler→PLN demo (per spec §15.4 acceptance):
  - "STV factor path returns same result as reference interpreter" — when
    a downstream reference (PRIMUS's lib/pln/pln_core_logic.metta running
    in MORK) is wired up, this template's lowering can be diffed against
    it. For now the lowering emits a MeTTa rewrite skeleton that the MORK
    space_metta_calculus! can execute directly.

This file is included by SchemaRegistry.jl AFTER GeometryTemplate.jl has
defined `make_template`, so all the spec §6.3 fields are filled out
correctly with sane defaults from `default_local_concurrency` etc.
"""

# Build the template — uses the existing TEMPLATE_HEURISTIC_MP as a starting
# point but with explicit STV truth-family + a registered lowering function.
const TEMPLATE_PLN_STV_MP = make_template(
    :PLN_STV_HeuristicModusPonens,
    sem_model(:Q, :Formula),
    GEOM_FACTOR;
    operators=[
        :stv_forward_map,
        :stv_backward_demand,
        :message_update,
        :boundary_refresh,
        :cache_lookup
    ],
    effects=[ReadEffect(DEFAULT_SPACE), AppendEffect(DEFAULT_SPACE)],
    laws=[:monotone, :sink_free, :delta_safe, :stv_strength_revisable],
    cache=CacheContract(
        [:schema_id, :factor_id, :subst_shape, :evidence_ver, :rule_ver, :truth_family],
        [:evidence_change, :rule_change, :truth_family_change]
    ),
    coercions=[
        Coercion(:FactorToTrie, GEOM_FACTOR, GEOM_TRIE, sem_model(:Q, :Formula)),
        Coercion(:FactorToTensor, GEOM_FACTOR, GEOM_TENSOR_SPARSE, sem_model(:Q, :Formula))
    ],
    affinity=Dict(:mm2 => :high, :mork => :high, :tensor => :medium)
)

"""
    pln_stv_lowering(t, region) → String

Spec §10.1.3 Algorithm 1 — emit the residual MeTTa code that implements
HeuristicModusPonens under STV. The residual is what MORK's
`space_metta_calculus!` will execute.

Form: an `exec`-driven rewrite that takes (A_TV, implies_AB_TV) pairs out of
the space and emits a B_TV with strength/confidence computed by the
spec's `heuristic-mp-tv` forward map:

    Bs = As * implies_s     (strength multiplies; canonical PLN MP under STV)
    Bc = min(Ac, implies_c) * 0.9   (confidence: weaker of the two, * confidence-decay)

The decay factor 0.9 matches the spec §10.1.2 default for HeuristicModusPonens
(`adjoint-need` backward demand → tightened confidence).
"""
function pln_stv_lowering(t::GeometryTemplate, region::AbstractString)::String
    # Body is independent of the input region — the template emits its own
    # canonical rewrite rule.  Downstream callers add concrete (A_TV, ...)
    # atoms into the MORK space, then run space_metta_calculus! and observe
    # the inferred (B_TV, ...) results.
    """
    ;; mgfw:lowering PLN_STV_HeuristicModusPonens
    ;; §10.1 Factor geometry: HeuristicModusPonens / STV
    ;; A_tv carries (strength confidence) for A
    ;; B_tv carries (strength confidence) for B (derived)
    (= (stv-mp \$As \$Ac \$Is \$Ic)
       (\$Bs \$Bc :where (\$Bs = (* \$As \$Is))
                       (\$Bc = (* (min \$Ac \$Ic) 0.9))))
    (= (apply-mp (\$A (stv \$As \$Ac)) (\$AimpB (stv \$Is \$Ic)))
       (\$B (stv \$Bs \$Bc) :where (\$Bs = (* \$As \$Is))
                                  (\$Bc = (* (min \$Ac \$Ic) 0.9))))
    """
end

export TEMPLATE_PLN_STV_MP, pln_stv_lowering
