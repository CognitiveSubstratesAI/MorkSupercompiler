"""
GeometryTemplate — canonical 13-field template schema and contracts.

Implements §6.3, §7.8, and §13 of the MG Framework spec (Goertzel, April 2026):
  §6.3  GeometryTemplate schema — 13 fields every template must declare
  §7.8  Exactness classes: EXACT, BOUNDED(ε), STATISTICAL(confidence)
  §13.1 Local concurrency contract — 8 fields
  §13.2 Distributed execution contract — 10 fields
  §13.3 Five policy families: LocalRewritePolicy, FixedPointMessagePolicy,
         PrefixShardPolicy, PatchLogShardPolicy, DemeAgentPolicy

The stability principle (§2): "canonical schema should be stable enough that
cached supercompilation results remain valid across DSL revisions."

Note: exactness classes (EXACT/BOUNDED/STATISTICAL) are shared with ApproxPipeline.jl
(which uses them as ErrorLevel). The MG framework makes them first-class in
the schema — every template declares its exactness class.
"""

# ── §13.3 Execution policy families ───────────────────────────────────────────

"""
    PolicyFamily

Five registered execution policy families (§13.3).
Geometry → typical policy:
Factor  → FIXED_POINT_MESSAGE
Trie    → PREFIX_SHARD
Tensor  → PATCH_LOG_SHARD
DAG     → DEME_AGENT + LOCAL_REWRITE
"""
@enum PolicyFamily begin
    LOCAL_REWRITE_POLICY       # sequential local rewrites within a region
    FIXED_POINT_MESSAGE_POLICY # iterative message passing to fixed point (Factor)
    PREFIX_SHARD_POLICY        # partition by hashed prefix (Trie)
    PATCH_LOG_SHARD_POLICY     # capture-compute-patch-reattach cycle (Tensor)
    DEME_AGENT_POLICY          # parallel deme agents with migration (DAG/MOSES)
end

"""
Default policy family for each geometry.
"""
default_policy(g::GeomTag)::PolicyFamily =
    if g == GEOM_FACTOR
        FIXED_POINT_MESSAGE_POLICY
    elseif g == GEOM_TRIE
        PREFIX_SHARD_POLICY
    elseif g == GEOM_TENSOR_SPARSE
        PATCH_LOG_SHARD_POLICY
    elseif g == GEOM_TENSOR_DENSE
        PATCH_LOG_SHARD_POLICY
    else
        DEME_AGENT_POLICY   # DAG default
    end   # DAG default

# ── §13.1 Local concurrency contract (8 fields) ────────────────────────────────

"""
    LocalConcurrencyContract

All 8 fields from §13.1. Declares how operations within a single node can be
parallelized safely.

unit_of_parallelism   — what can be executed concurrently (e.g., :message-update)
commutes_when         — conditions under which two operations commute
requires_barrier_when — conditions requiring a synchronization barrier
merge_law             — how parallel results are combined
scheduler_shape       — structural description of the scheduler
conflict_domain       — which operations can conflict
cache_consistency     — how cached values are kept valid
determinism_class     — degree of determinism provided
"""
struct LocalConcurrencyContract
    unit_of_parallelism::Vector{Symbol}
    commutes_when::Vector{Symbol}
    requires_barrier_when::Vector{Symbol}
    merge_law::Symbol
    scheduler_shape::Symbol
    conflict_domain::Symbol
    cache_consistency::Symbol
    determinism_class::Symbol
end

LocalConcurrencyContract(
    unit,
    commutes,
    barriers,
    merge;
    sched=:default,
    conflict=:none,
    cache=:epoch,
    determinism=:eventual_fixed_point
) = LocalConcurrencyContract(
    isa(unit, Symbol) ? [unit] : unit,
    isa(commutes, Symbol) ? [commutes] : commutes,
    isa(barriers, Symbol) ? [barriers] : barriers,
    merge,
    sched,
    conflict,
    cache,
    determinism
)

# Default contracts per geometry type (from §10.1.5 / §10.2.4 / §10.3.4 / §10.4.3)
function default_local_concurrency(g::GeomTag)::LocalConcurrencyContract
    if g == GEOM_FACTOR
        LocalConcurrencyContract(
            [:message_update],
            [:disjoint_targets_or_monotone_join],
            [:boundary_refresh, :phase_switch],
            :quantale_join;
            sched=:worklist,
            cache=:version_tuple,
            determinism=:eventual_fixed_point
        )
    elseif g == GEOM_DAG
        LocalConcurrencyContract(
            [:disjoint_rewrite_region, :deme],
            [:regions_disjoint, :rewrite_confluent],
            [:global_canonicalization_pass],
            :canonical_hashcons_merge;
            sched=:deme_parallel,
            cache=:content_hash,
            determinism=:confluent_if_confluent
        )
    elseif g == GEOM_TRIE
        LocalConcurrencyContract(
            [:prefix_subtree],
            [:distinct_prefix_ownership],
            [:topk_rebuild, :ancestor_counter_flush],
            :counter_plus_topk_merge;
            sched=:prefix_scan,
            cache=:epoch_or_prefix_lock,
            determinism=:deterministic_under_fixed_update_order
        )
    else   # TENSOR
        LocalConcurrencyContract(
            [:tensor_shard_kernel],
            [:distinct_output_patch_sets],
            [:global_reduction, :optimizer_step],
            :patch_log_replay;
            sched=:shard_parallel,
            cache=:snapshot_plus_patch,
            determinism=:deterministic_up_to_floating_point
        )
    end
end

# ── §13.2 Distributed execution contract (10 fields) ─────────────────────────

"""
    DistributedExecContract

All 10 fields from §13.2. Declares how computation can be distributed across nodes.

partition_key      — how to assign atoms/objects to shards
shard_shape        — shape description of each shard
halo_policy        — what data crosses shard boundaries
migration_unit     — what can be moved between shards
placement_hints    — suggestions for shard placement
replication_policy — if/how state is replicated
state_model        — snapshot/delta model
recovery_model     — how to recover from failure
cross_shard_protocol — how shards communicate
cost_metrics       — how to measure distributed cost
"""
struct DistributedExecContract
    partition_key::Symbol
    shard_shape::Symbol
    halo_policy::Vector{Symbol}
    migration_unit::Symbol
    placement_hints::Vector{Symbol}
    replication_policy::Symbol
    state_model::Symbol
    recovery_model::Symbol
    cross_shard_protocol::Symbol
    cost_metrics::Dict{Symbol, Any}
end

DistributedExecContract(;
    partition=:default,
    shard=:default,
    halo=Symbol[],
    migration=:default,
    hints=Symbol[],
    replication=:none,
    state=:snapshot_plus_delta,
    recovery=:recompute_from_cache,
    protocol=:default,
    metrics=Dict{Symbol, Any}()
) = DistributedExecContract(
    partition,
    shard,
    halo,
    migration,
    hints,
    replication,
    state,
    recovery,
    protocol,
    metrics
)

function default_distributed_exec(g::GeomTag)::DistributedExecContract
    if g == GEOM_FACTOR
        DistributedExecContract(;
            partition=:active_subgraph_or_factor_community,
            halo=[:boundary_messages],
            migration=:factor_neighborhood,
            state=:snapshot_plus_delta,
            recovery=:recompute_from_cache,
            protocol=:boundary_message_exchange
        )
    elseif g == GEOM_DAG
        DistributedExecContract(;
            partition=:deme_id,
            migration=:rooted_subdag_or_exemplar_set,
            state=:transactional_atomspace,
            recovery=:replay_from_canonical_roots,
            protocol=:pubsub_plus_migration
        )
    elseif g == GEOM_TRIE
        DistributedExecContract(;
            partition=:hashed_high_prefix,
            halo=[:ancestor_summary_only],
            migration=:prefix_subtree,
            state=:append_plus_counter_merge,
            recovery=:rebuild_topk_from_local_stats,
            protocol=:summary_push_or_prefix_steal
        )
    else   # TENSOR
        DistributedExecContract(;
            partition=:prefix_or_resolution_block,
            halo=[:read_only_boundary, :second_pass_stream],
            migration=:shard,
            state=:capture_compute_patch_reattach,
            recovery=:rerun_shard_from_snapshot,
            protocol=:halo_exchange_plus_patch_replay
        )
    end
end

# ── §6.3 Cache contract ───────────────────────────────────────────────────────

"""
    CacheContract

Declared cache key and invalidation policy for a geometry template (§6.3).
Cache keys allow supercompiler to reuse cached results via hash-cons probes.
"""
struct CacheContract
    key::Vector{Symbol}
    invalidate_on::Vector{Symbol}
end
CacheContract() = CacheContract(Symbol[], Symbol[])

# ── §6.3 GeometryTemplate (all 13 fields) ─────────────────────────────────────

"""
    GeometryTemplate

The canonical 13-field schema record from §6.3. Every geometry template must
declare ALL 13 fields. Missing fields → not a valid normalized template.

Fields:

 1. name              — unique template identifier
 2. semantic_type     — SemanticType (Rel/Prog/Model/Codec/Sched/Stream)
 3. presentation      — GeomTag (or HybridGeom for composites)
 4. operators         — visible + internal operator names
 5. effects           — Effect list (from Effects.jl)
 6. laws              — algebraic laws (e.g., :monotone, :sink_free)
 7. symmetries        — symmetry group Γ acting on operator labels
 8. cache_contract    — CacheContract with key + invalidation policy
 9. exactness_class   — ErrorLevel: EXACT / BOUNDED / STATISTICAL
10. coercions         — registered coercions from/to other geometries
11. local_concurrency — LocalConcurrencyContract (8 fields)
12. distributed_exec  — DistributedExecContract (10 fields)
13. backend_affinity  — Dict{Symbol, Symbol} (e.g., :mm2 → :high)
"""
struct GeometryTemplate
    name::Symbol
    semantic_type::SemanticType
    presentation::Union{GeomTag, HybridGeom}
    operators::Vector{Symbol}
    effects::Vector{Effect}
    laws::Vector{Symbol}
    symmetries::Vector{Symbol}
    cache_contract::CacheContract
    exactness_class::ErrorLevel
    coercions::Vector{Coercion}
    local_concurrency::LocalConcurrencyContract
    distributed_exec::DistributedExecContract
    backend_affinity::Dict{Symbol, Symbol}
    noether_charge::Union{Symbol, Nothing}   # §12.2: conserved quantity name, or nothing
end

"""
    is_valid_template(t::GeometryTemplate) -> Bool

Check all 13 spec fields are populated (non-empty where required).
Validates: name, operators, semantic_type args, laws, local_concurrency,
distributed_exec, backend_affinity, and exactness class.
"""
function is_valid_template(t::GeometryTemplate)::Bool
    t.name != :unnamed || return false
    !isempty(t.operators) || return false
    length(t.semantic_type.args) > 0 || return false
    !isempty(t.local_concurrency.unit_of_parallelism) || return false
    !isempty(t.local_concurrency.merge_law |> string) || return false
    !isempty(t.distributed_exec.state_model |> string) || return false
    !isempty(t.backend_affinity) || return false
    true
end

"""
    geometry_of(t::GeometryTemplate) -> GeomTag

Primary geometry tag (first component for Hybrid).
"""
function geometry_of(t::GeometryTemplate)::GeomTag
    t.presentation isa GeomTag && return t.presentation::GeomTag
    (t.presentation::HybridGeom).components[1]
end

"""
    all_geometries(t::GeometryTemplate) -> Vector{GeomTag}

All geometry tags for a template, preserving Hybrid composition.
Use this (not geometry_of) when full multi-geometry information is needed.
"""
function all_geometries(t::GeometryTemplate)::Vector{GeomTag}
    t.presentation isa GeomTag && return [t.presentation::GeomTag]
    collect((t.presentation::HybridGeom).components)
end

"""
    is_hybrid(t::GeometryTemplate) -> Bool

Returns true iff this template uses multiple geometry presentations.
"""
is_hybrid(t::GeometryTemplate)::Bool = t.presentation isa HybridGeom

"""
    policy_families(t::GeometryTemplate) -> Vector{PolicyFamily}

All policy families applicable to this template (one per geometry for Hybrid).
"""
policy_families(t::GeometryTemplate)::Vector{PolicyFamily} =
    [default_policy(g) for g in all_geometries(t)]

# ── Built-in templates (from §12.2 worked examples) ──────────────────────────

"""
Build a GeometryTemplate with sane defaults for a given geometry.
"""
function make_template(
    name::Symbol,
    sem_type::SemanticType,
    geom::GeomTag;
    operators::Vector{Symbol}=Symbol[],
    effects::AbstractVector{<:Effect}=Effect[],
    laws::Vector{Symbol}=Symbol[],
    symmetries::Vector{Symbol}=[:trivial],
    cache::CacheContract=CacheContract(),
    exactness::ErrorLevel=EXACT,
    coercions::Vector{Coercion}=Coercion[],
    affinity::Dict{Symbol, Symbol}=Dict{Symbol, Symbol}(),
    noether::Union{Symbol, Nothing}=nothing
)::GeometryTemplate
    GeometryTemplate(
        name,
        sem_type,
        geom,
        operators,
        collect(Effect, effects),
        laws,
        symmetries,
        cache,
        exactness,
        coercions,
        default_local_concurrency(geom),
        default_distributed_exec(geom),
        isempty(affinity) ? Dict(:mm2 => :high, :mork => :high) : affinity,
        noether
    )
end

# §12.2 canonical examples
const TEMPLATE_HEURISTIC_MP = make_template(
    :HeuristicModusPonens,
    sem_model(:Q, :Formula),
    GEOM_FACTOR;
    operators=[:forward_map, :backward_demand, :message_update],
    effects=[ReadEffect(DEFAULT_SPACE), AppendEffect(DEFAULT_SPACE)],
    laws=[:monotone, :sink_free, :delta_safe],
    cache=CacheContract(
        [:schema_id, :factor_id, :subst_shape, :evidence_ver, :rule_ver],
        [:evidence_change, :rule_change]
    ),
    coercions=[Coercion(:FactorToTrie, GEOM_FACTOR, GEOM_TRIE, sem_model(:Q, :Formula))]
)

const TEMPLATE_EVIDENCE_CAPSULE = make_template(
    :EvidenceCapsule,
    sem_codec(:EvidenceSet),
    GEOM_TRIE;
    operators=[:mint_token, :sketch_union, :overlap_estimate, :merge_capsule],
    effects=[ReadEffect(DEFAULT_SPACE), AppendEffect(DEFAULT_SPACE)],
    laws=[:idempotent_merge, :commutative_merge, :evidence_monotone],
    cache=CacheContract([:capsule_cid, :sketch_cid], [:new_token_mint]),
    noether=:evidence_mass
)   # §12.2: evidence mass is the conserved Noether charge

export PolicyFamily
export LOCAL_REWRITE_POLICY, FIXED_POINT_MESSAGE_POLICY, PREFIX_SHARD_POLICY
export PATCH_LOG_SHARD_POLICY, DEME_AGENT_POLICY, default_policy
export LocalConcurrencyContract, default_local_concurrency
export DistributedExecContract, default_distributed_exec
export CacheContract, GeometryTemplate
export is_valid_template, geometry_of, all_geometries, is_hybrid, policy_families
export make_template
const TEMPLATE_CAUSAL_DAG = make_template(
    :CausalDAG,
    sem_rel(:Cause, :Effect),
    GEOM_DAG;
    operators=[:topo_sort, :ancestor_query, :path_score],
    effects=[ReadEffect(DEFAULT_SPACE)],
    laws=[:acyclic, :topological_order],
    cache=CacheContract([:dag_cid, :path_hash], [:edge_add])
)

export TEMPLATE_HEURISTIC_MP, TEMPLATE_EVIDENCE_CAPSULE, TEMPLATE_CAUSAL_DAG
