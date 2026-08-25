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
        :compute_demand_field,
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
    # 🔴 THE FOUR EXECS HAVE A HARD SEQUENCING DEPENDENCY WITH NO STRUCTURAL ENFORCEMENT.
    # Step (1 4) reads `stv-mp-s` and `stv-mp-c`, which (1 1)-(1 3) must already have written, and
    # (1 3) reads `stv-mp-cmin` from (1 2). The ONLY thing ordering them is the priority numbers.
    # Renumber, reorder or merge them and the chain breaks SILENTLY — the differential in
    # test_pln_reference.jl starts failing, but nothing here says why the numbers matter.
    # Same class as the other "correctness depends on an invisible constraint" defects found
    # 2026-08-25 (`error_tolerance` accepted-and-ignored, `correlation_sig` never seeded, the PURE
    # effect tag gating reordering) — smaller stakes, identical shape. If you touch the priorities,
    # run the differential.
    #
    # Emits MM2 execs whose SINKS carry `(pure <template> $var <expr>)`. Callers add
    # `(stv <A> <s> <c>)` and `(imp <A> <B> <s> <c>)` atoms, run space_metta_calculus!,
    # and read back `(stv <B> <s> <c>)`.
    """
    ;; mgfw:lowering PLN_STV_HeuristicModusPonens
    ;; §10.1 Factor geometry: HeuristicModusPonens / STV
    ;; s_b = s_a * s_imp ; c_b = min(c_a, c_imp) * 0.9   (matches stv_mp_reference)
    (exec (STV_MP (1 1))
        (, (stv \$A \$As \$Ac) (imp \$A \$B \$Is \$Ic))
        (O (pure (stv-mp-s \$A \$B \$o) \$o
                 (f64_to_string (product_f64 (f64_from_string \$As) (f64_from_string \$Is))))))
    (exec (STV_MP (1 2))
        (, (stv \$A \$As \$Ac) (imp \$A \$B \$Is \$Ic))
        (O (pure (stv-mp-cmin \$A \$B \$o) \$o
                 (f64_to_string (min_f64 (f64_from_string \$Ac) (f64_from_string \$Ic))))))
    (exec (STV_MP (1 3))
        (, (stv-mp-cmin \$A \$B \$m))
        (O (pure (stv-mp-c \$A \$B \$o) \$o
                 (f64_to_string (product_f64 (f64_from_string \$m) (f64_from_string 0.9))))))
    (exec (STV_MP (1 4))
        (, (stv-mp-s \$A \$B \$Bs) (stv-mp-c \$A \$B \$Bc))
        (O (stv \$B \$Bs \$Bc)))
    """
end

export TEMPLATE_PLN_STV_MP, pln_stv_lowering
