"""
MorkSupercompiler — query planner and source-reordering supercompiler for MORK.

Implements Phase 0 of the MM2 Supercompiler design (Goertzel, Oct 2025):

  - §3    M-Core IR foundation (SExpr layer)
  - §5    Query planner with MORK-native cardinality estimation
  - §5.3  Effect-aware join ordering (all exec sources are Read → freely reorder)
  - §5.1  Statistics collection (MORKStatistics)
  - §5.2  Incremental statistics under monotonic growth

Usage (static planning, no Space needed):
program′ = plan_static(program)
space_add_all_sexpr!(s, program′)
space_metta_calculus!(s, max_steps)

Usage (dynamic planning, uses btm cardinalities):
plan!(s, program)    # adds reordered program to s and runs metta_calculus
-- or --
program′ = plan_program(s, program)

Public API:
plan_static(program)          — pure-string reorder (no Space)
plan_program(s, program)      — reorder using btm prefix counts
plan!(s, program, steps)      — add + run in one call
collect_stats(s)              — build MORKStatistics for s
plan_report(s, program)       — human-readable join-plan report

Architecture references:
SExpr.jl      — s-expression parser (M-Core surface syntax)
Statistics.jl — MORKStatistics + Algorithms 2/3
Selectivity.jl — dynamic_count / static_score
Rewrite.jl    — source reordering (pure static, backward-compat)
QueryPlanner.jl — Algorithm 6 (EffectAwarePlanning) + variable flow
"""
module MorkSupercompiler

using MORK
using PathMap

# Layer 1 — Surface syntax (s-expression parser)
include("frontend/SExpr.jl")

# Layer 2 — Core IR + Effect algebra (§3 + §4 of MM2 spec)
include("core/MCore.jl")
include("core/Effects.jl")

# Layer 3 — Query Planner (§5 of MM2 spec)
include("planner/Selectivity.jl")
include("planner/Statistics.jl")
include("planner/QueryPlanner.jl")

# Layer 4 — Source rewriting (join-order reordering)
include("rewrite/Rewrite.jl")

# Layer 5 — Core Supercompiler (§6 of MM2 spec)
include("supercompiler/Stepper.jl")
include("supercompiler/CanonicalKeys.jl")
include("supercompiler/BoundedSplit.jl")
include("supercompiler/Driver.jl")            # §6 driver — closes TyLA G functor gap
include("supercompiler/PipelineDecompose.jl")

# Layer 6 — Phase 3 Specializations + Code Generation (§7–9)
include("supercompiler/KBSaturation.jl")
include("supercompiler/RuleSpecialization.jl")   # v1 §8.3 partial instantiation + magic sets
include("supercompiler/EvoSpecializer.jl")
include("codegen/MM2Compiler.jl")
include("codegen/BisimVerifier.jl")          # Boundary #3 — discharges MM2Compiler obligations via v2 §12.2 differential testing
include("codegen/MM2Optimize.jl")            # v1 §10.6 — static scheduling + space batching + pattern fusion

# Layer 8 — Approximate Supercompilation (Doc 2, approx spec)
# Must be included before SCPipeline (SCResult references ApproxPipelineResult)
include("approx/PBoxAlgebra.jl")
include("approx/UncertainQuery.jl")
include("approx/UncertainInference.jl")
include("approx/ApproxMOSES.jl")
include("approx/ApproxPipeline.jl")

# Layer 7 — Integration + Production Hardening (§10.4)
include("integration/SCPipeline.jl")
include("integration/Profiler.jl")
include("integration/Explainer.jl")
include("integration/AdaptivePlanner.jl")

# Layer 9 — Multi-Geometry Framework (Doc 3, mg_framework_spec)
include("mgfw/SemanticObjects.jl")

# Layer 10 — Multi-Space via HPC standalone package
using HPC
# Re-export all HPC symbols so existing MorkSupercompiler users see no change
using HPC: ENABLE_MULTI_SPACE, enable_multi_space!
using HPC: LOCAL_PEER, NamedSpaceID, SpaceRegistry
using HPC: get_registry, new_space!, get_space, common_space, list_spaces, compute_cid
using HPC: TRAVERSAL_THRESHOLD, TraversalResult, space_traverse!, process_mpi_traversals!
using HPC: process_multispace_commands!
using HPC: save_space!, load_space!, checkpoint_all!
using HPC: mpi_init!, mpi_finalize!, mpi_rank, mpi_nranks, mpi_active
using HPC: mpi_send_traverse!, mpi_poll_traverse!, mpi_broadcast_traverse!, mpi_barrier!
using HPC: mpi_allreduce_sum, mpi_allgatherv_strings, mpi_bcast_bytes!
using HPC: TRAVERSE_TAG, RESULT_TAG
using HPC: ShardedSpace, new_sharded_space
using HPC: sharded_add!, sharded_flush!, sharded_query, sharded_val_count
using HPC: shard_owner, SHARD_ATOM_TAG

# MORKTensorNetworks — semiring algebra for principled geometry selection (ADR-055)
using MORKTensorNetworks:
    AbstractSemiring,
    BooleanSemiring,
    MaxPlusSemiring,
    MinPlusSemiring,
    SumProductSemiring,
    PLNSemiring,
    CostSemiring,
    semiring_matmul,
    oplus,
    otimes,
    szero
include("mgfw/GeometryTemplate.jl")
include("mgfw/SchemaRegistry.jl")
include("mgfw/FactorGeometry.jl")
include("mgfw/PLNDemand.jl")
include("mgfw/TrieDAGGeometry.jl")
include("mgfw/MGCompiler.jl")

# §15.4 MVP demonstration templates — registered with their lowerings inside
# `__init__` via the SchemaRegistry.init_registry! hook (see SchemaRegistry.jl).
include("mgfw/templates/pln_stv.jl")
include("mgfw/templates/motif_miner.jl")
include("mgfw/templates/geodesic_bgc.jl")
include("mgfw/templates/references.jl")

# GEO-EVO geodesic evolutionary control (Geo-Evo.pdf §3–6): couples the forward demes to the PLN
# backward demand field. All structure (params/subgoals/factors) is read from MORK atoms at
# runtime — see the anti-hardcode contract at the head of GeoEvo.jl.
include("mgfw/GeoEvo.jl")
export geo_params, geo_subgoals, geo_comp, geo_score, geo_sinkhorn, geo_feff, geo_effort, geo_sigma
export geo_factor_graph, geo_backward_g
export geo_deme_ops, geo_weakness, geo_subgoal_motifs, geo_cover, geo_gap, geo_reach, geo_pairing
export geo_forward_f, geo_splice_check, geo_bandit, geo_step!
export geo_align_bias!, geo_eda_sample!, geo_evolve_steered!
export geo_xover_join, geo_xover_product, geo_xover_mask, geo_mutate_add, geo_mutate_restrict, geo_recombine
export geo_metrics, geo_guidance_capsules!

# Wire the MVP-template lowerings + template registrations into the global
# registry. Registers idempotently — `register!` overwrites, so re-calling
# is safe. Invoked at `__init__` time by replacing `init_registry!` with
# a version that calls both the framework-shipped registrations AND ours.
function _register_mvp_templates!()
    register!(GLOBAL_REGISTRY, TEMPLATE_PLN_STV_MP)
    register!(GLOBAL_REGISTRY, TEMPLATE_TRIE_MOTIF_MINER)
    # GeodesicBGC composite — §12.2 worked example, registered AFTER its
    # sub-templates so the registry has the pieces in case `mg_compile`
    # walks the composite's components list.
    register!(GLOBAL_REGISTRY, build_geodesic_bgc_composite(GLOBAL_REGISTRY))
    register_lowering!(:PLN_STV_HeuristicModusPonens, pln_stv_lowering)
    register_lowering!(:FactorGraphMotifMiner, motif_miner_lowering)
    register_lowering!(:GeodesicBGC_Composite, geodesic_bgc_lowering)
    nothing
end

# Inline the framework-shipped registrations + MVP-template additions. Defined
# here (after all template files have loaded) rather than in SchemaRegistry.jl
# to avoid the Julia "Method overwriting is not permitted during precompilation"
# error that would fire if SchemaRegistry.jl defined a stub and we overrode it.
function init_registry!()
    register!(GLOBAL_REGISTRY, TEMPLATE_HEURISTIC_MP)
    register!(GLOBAL_REGISTRY, TEMPLATE_EVIDENCE_CAPSULE)
    register!(GLOBAL_REGISTRY, TEMPLATE_CAUSAL_DAG)
    _register_mvp_templates!()
    nothing
end

# Julia auto-invokes `__init__` at module load. This is the only place
# `init_registry!` is called automatically — callers depending on
# GLOBAL_REGISTRY being populated can rely on it being non-empty after
# `using MorkSupercompiler`.
function __init__()
    init_registry!()
end

# ── High-level public API ─────────────────────────────────────────────────────

"""
    plan_static(program::AbstractString) -> String

Reorder sources in all conjunction lists using static selectivity only
(variable-fraction heuristic).  No Space or statistics required.
Cheapest option; use when no background facts are loaded yet.
"""
plan_static(program::AbstractString)::String = reorder_program_static(program)

"""
    plan_program(s::Space, program::AbstractString) -> String

Reorder sources using dynamic btm-prefix cardinality counts.
Dispatches on the first argument: Space → dynamic, MORKStatistics → stats-based.
"""
plan_program(s::Space, program::AbstractString)::String =
    plan_program_dynamic(program, s.btm)

"""
    _preprocess_program(program) → String

When multi-space is enabled, strip and execute multi-space MM2 commands.
Zero overhead when ENABLE_MULTI_SPACE[] = false.
"""
@inline function _preprocess_program(program::AbstractString)::String
    ENABLE_MULTI_SPACE[] || return String(program)
    process_multispace_commands!(get_registry(), program)
end

"""
    plan!(s::Space, program::AbstractString, max_steps::Int=typemax(Int)) -> Int

Plan, decompose, add, and execute in one call:

 1. Reorder sources using btm prefix counts
 2. Decompose multi-source exec atoms (Rule-of-64 fix)
 3. Add the transformed program to `s`
 4. Run `space_metta_calculus!(s, max_steps)`
 5. Clean up `_sc_tmp*` intermediate atoms
 6. Return steps executed.
"""
function plan!(s::Space, program::AbstractString, max_steps::Int=typemax(Int))::Int
    program = _preprocess_program(program)
    program′ = plan_program(s, program)
    program′ = decompose_program(program′)
    space_add_all_sexpr!(s, program′)
    steps = space_metta_calculus!(s, max_steps)
    _cleanup_sc_tmp!(s)
    steps
end

"""
    run!(s::Space, program::AbstractString, max_steps::Int=typemax(Int)) -> SCResult

Drop-in replacement for `space_add_all_sexpr! + space_metta_calculus!`.
Runs the full supercompiler pipeline: stats → plan → decompose → execute → cleanup.

This is the primary entry point from the MM2 spec §10.5.
Use instead of calling `space_metta_calculus!` directly.
"""
function run!(s::Space, program::AbstractString, max_steps::Int=typemax(Int))::SCResult
    program = _preprocess_program(program)
    execute!(s, program; opts=SCOptions(; max_steps=max_steps))
end

# Re-export the most useful lower-level symbols
export plan_static, plan_program, plan!, plan_report
export MORKStatistics, IncrementalStats, collect_stats, merged_stats
export estimate_cardinality, prefix_sample_count
export SNode, SAtom, SVar, SList
export parse_program, parse_sexpr, sprint_sexpr, sprint_program
export plan_join_order, plan_join_order_static, plan_report
export reorder_program_static
export source_order_report
export JoinNode, build_join_nodes, build_join_nodes_dynamic
export effects_commute, EffectKind, EFF_PURE, EFF_READ, EFF_APPEND, EFF_WRITE, EFF_OBSERVE
# MCore IR (Phase 1)
export NodeID, NULL_NODE, EffectSet, SpaceID, DEFAULT_SPACE
export MCoreNode, Sym, Var, Lit, Con, App, Abs, LetNode, MatchNode, MatchClause
export Choice, ChoiceAlt, Prim, MCoreRef, UncertainNode, PBox
export MCoreGraph, get_node
export add_sym!, add_var!, add_lit!, add_con!, add_app!, add_abs!
export add_let!, add_match!, add_choice!, add_prim!, add_mref!
export compile_kb_query, compile_mm2_exec
# Effect algebra (Phase 1)
export Effect, ReadEffect, WriteEffect, AppendEffect
export CreateEffect, DeleteEffect, ObserveEffect, PureEffect, PURE
export commutes, commutes_all, is_sink_free, sink_free_check, mork_source_effects
# Stepper (Phase 2)
export StepResult, Value, Blocked, Residual
export Env, env_lookup, env_extend
export DepSet, can_proceed, add_dep
export PrimRegistry, register_prim!, lookup_prim, DEFAULT_PRIM_REGISTRY
export rewrite_once, step_to_value
# CanonicalKeys (Phase 2)
# BoundedSplit (Phase 2)
export Branch, SplitResult, bounded_split
export SPLIT_PROB_THRESHOLD, SPLIT_DEFAULT_BUDGET
# PipelineDecompose (Rule-of-64 fix)
export STAGE_MAX_SOURCES, SC_TMP_PREFIX
export DecomposedProgram, decompose_exec
export decompose_program, decompose_report, flow_vars
# KBSaturation (Phase 3 §7)
export Fact, is_base_fact, Rule
export VersionedIndex, index_insert!, index_lookup, index_delta_since, bump_version!
export KBState, kb_add_fact!, kb_add_rule!, all_facts, saturate!
export SpecializationResult, specialize_rules                   # v1 §8.3 partial instantiation
export MagicSetsResult, magic_sets_transform                    # v1 §8.3 Alg 4 magic sets
# EvoSpecializer (Phase 3 §8)
export SpecLevel, SPEC_GENERIC, SPEC_INCREMENTAL, SPEC_VECTORIZED
export SpecDecision, should_specialize
export ChangeKind, CHANGE_NONE, CHANGE_CONSTANT, CHANGE_STRUCTURAL
export ASTDiff, compute_ast_diff, CacheMetadata, can_reuse_cache
export EvolutionaryPBox, approximate_fitness, allocate_evaluations
# MM2Compiler (Phase 3 §9)
export MM2Priority, MM2ExecAtom, sprint_exec, sprint_priority
export CompileCtx, BiSimObligation
export compile_sequential!, compile_conditional!, compile_node!, compile_program
export VerifyResult, BisimVerdict, verify_bisim   # Boundary #3 — bisim verifier
export schedule_static, batch_space_ops, fuse_identical_patterns   # v1 §10.6 MM2 optimizations
export sprint_mcore_to_mm2
# Integration layer (Phase 4)
export SCOptions, SC_DEFAULTS, SCResult, run!, execute, timing_report
export ProfilePhase,
    PHASE_STATS, PHASE_PLAN, PHASE_DECOMPOSE, PHASE_LOAD, PHASE_EXECUTE, PHASE_TOTAL
export SCProfile, profile, speedup_report
export explain, to_dot, diff_programs
export AdaptivePlan, should_replan, replan!, run_adaptive!, update_stats!

export MAX_PLAN_AGE, REPLAN_DRIFT
# Approx Layer (Doc 2)
export pbox_exact, pbox_point, pbox_interval, pbox_empty
export width, max_width, overlap, are_dependent, mark_dependent
export add_pbox, mul_pbox, widen_pbox, merge_overlapping, sample_from_pbox
export error_composition_bound, frechet_width_bound, hoeffding_bound, hoeffding_epsilon
export CostWeights, safety_critical, exploratory, balanced, total_cost
export EstimateCardinalityPBox, estimate_cardinality_pbox
export ApproxBranch, ApproximateSplitResult, approximate_split
export ApproxJoinNode, plan_join_order_approx
export ProofTree, UncertainFact, certain_fact
export conjunction_and, disjunction_or
export structural_similarity, match_with_uncertainty, NO_MATCH
export apply_rule, convergence_width_bound
export InferenceContext, step_deeper, derive_fact
export tournament_with_pbox, MONTE_CARLO_TRIALS, CONVERGENCE_THETA
export offspring_fitness_pbox
export population_converged, convergence_report, rank_population
export ErrorLevel, EXACT, BOUNDED, STATISTICAL
export ApproximatePathSig, is_cacheable, approx_subsumes
export SimpleBloomFilter, bloom_add!, bloom_check, bloom_false_positive_rate
export ApproxIndex, approx_index_insert!, approx_index_lookup
export register_approx_primitives!
export ApproxPhase, ApproxPipelineResult, run_approx_pipeline
# CanonicalKeys (Phase 2)
export CompactShape, shape_subsumes, extract_shape
export FixedArgMask, CanonicalKBSig, CanonicalEffectSig, CanonicalPathSig
export canonical_key, subsumes
export FoldTable, record!, lookup_fold, can_fold
# MG Framework (Doc 3)
export SemanticKind, SK_REL, SK_PROG, SK_MODEL, SK_CODEC, SK_SCHED, SK_STREAM
export SemanticType, sem_rel, sem_prog, sem_model, sem_codec, sem_sched, sem_stream
export GeomTag, GEOM_FACTOR, GEOM_DAG, GEOM_TRIE, GEOM_TENSOR_SPARSE, GEOM_TENSOR_DENSE
export HybridGeom, PresType
export MGType, MGUnit, MGVoid, MGBase, MGProd, MGSum, MGFun, MGSemType, MGPres
export Coercion, is_exact, find_coercion, REGISTERED_COERCIONS
export T_DAG_TO_FACTOR, T_FACTOR_TO_TRIE, T_TRIE_TO_TENSOR, T_TRIE_TO_CODEC
export TyLADirection, F_DIRECTION, G_DIRECTION
export PolicyFamily, LOCAL_REWRITE_POLICY, FIXED_POINT_MESSAGE_POLICY
export PREFIX_SHARD_POLICY, PATCH_LOG_SHARD_POLICY, DEME_AGENT_POLICY, default_policy
export LocalConcurrencyContract, DistributedExecContract, CacheContract
export GeometryTemplate, is_valid_template, geometry_of, all_geometries, is_hybrid,
    policy_families, make_template
export TEMPLATE_HEURISTIC_MP, TEMPLATE_EVIDENCE_CAPSULE
export SchemaRegistry, register!, lookup, search, coercion_path, GLOBAL_REGISTRY
export DSLForm, AuthoringResult, authoring_workflow
export define_factor_rule, define_trie_miner, define_codec_search
export FactorNode, FactorEdge, FactorGraph, SpecializedRegion
export specialize_exact, specialize_approximate
export stv_forward_map, stv_to_pbox
export stv_symmetric_mp, stv_negation, stv_inversion, stv_revision
export stv_deduction, stv_induction, stv_abduction
export noether_charge, conserves_evidence
export DAGNode, DAGStore, dag_intern!, Deme, DemeEvolutionResult, evolve_demes!
export PatternTrie, trie_seed!, trie_grow!, trie_score!, run_trie_miner
export AffinityLevel, HIGH, MEDIUM, LOW, NONE
export BackendProfile, BackendChoice
export affinity_analysis, select_backend, CompilationResult, mg_compile, mg_run!
export build_geodesic_bgc_composite
# Multi-Space (Layer 10)
export ENABLE_MULTI_SPACE, enable_multi_space!
export LOCAL_PEER, NamedSpaceID, SpaceRegistry
export get_registry, new_space!, get_space, common_space, list_spaces, compute_cid
export TRAVERSAL_THRESHOLD, TraversalResult, space_traverse!, process_mpi_traversals!
export process_multispace_commands!
export save_space!, load_space!, checkpoint_all!
# MPI transport (Stage 2)
export mpi_init!, mpi_finalize!
export mpi_rank, mpi_nranks, mpi_active
export mpi_send_traverse!, mpi_poll_traverse!, mpi_broadcast_traverse!, mpi_barrier!
export mpi_allreduce_sum, mpi_allgatherv_strings, mpi_bcast_bytes!
export TRAVERSE_TAG, RESULT_TAG
# ShardedSpace — Topology 2
export ShardedSpace, new_sharded_space
export sharded_add!, sharded_flush!, sharded_query, sharded_val_count
export shard_owner, SHARD_ATOM_TAG
# MORK Space re-exports (for test convenience)
export Space, new_space
export space_add_all_sexpr!, space_metta_calculus!, space_dump_all_sexpr, space_val_count

end # module
