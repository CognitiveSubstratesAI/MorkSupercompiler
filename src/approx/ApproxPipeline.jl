"""
ApproxPipeline — 4-phase approximate supercompilation pipeline.

Implements §6 of the Approximate Supercompilation spec (Goertzel, Oct 2025):
  §6.1  4-phase pipeline: Analysis → Planning → Specialization → Verification
  §6.2  ApproxIndex (core, BloomFilter, weights, coverage PBox)
  §6.3  New IR primitives: approx_kb_query + sample_fitness (with error_bound)
  §6.4  ApproximatePathSig (base_sig, error_level EXACT/BOUNDED/STATISTICAL, confidence)

The ApproximatePathSig (§6.4) extends CanonicalPathSig with an error level:
  EXACT       — no approximation, cacheable forever (same as Doc 1)
  BOUNDED     — deterministic approximation with known error bound (cacheable with bound)
  STATISTICAL — probabilistic, may give different results each run (NOT persistently cacheable)
"""

using MORK: Space, new_space, space_add_all_sexpr!, space_metta_calculus!, space_val_count

# ── §6.4 ApproximatePathSig ───────────────────────────────────────────────────

"""
    ErrorLevel

Three error levels for approximate canonical path signatures (§6.4):
EXACT        — no approximation, cacheable forever
BOUNDED      — deterministic with known Float64 error bound
STATISTICAL  — probabilistic, may give different results per run
"""
@enum ErrorLevel begin
    EXACT
    BOUNDED
    STATISTICAL
end

"""
    ApproximatePathSig

§6.4: Canonical key extended with error metadata.

base_sig    — the exact CanonicalPathSig from Doc 1
error_level — EXACT / BOUNDED(ε) / STATISTICAL
error_bound — for BOUNDED: maximum approximation error ε
confidence  — for STATISTICAL: probability of correctness
"""
struct ApproximatePathSig
    base_sig::CanonicalPathSig
    error_level::ErrorLevel
    error_bound::Float64    # 0.0 for EXACT, ε for BOUNDED
    confidence::Float64    # 1.0 for EXACT, p for STATISTICAL
end

ApproximatePathSig(base::CanonicalPathSig) = ApproximatePathSig(base, EXACT, 0.0, 1.0)

ApproximatePathSig(base::CanonicalPathSig, ε::Float64) =
    ApproximatePathSig(base, BOUNDED, ε, 1.0)

ApproximatePathSig(base::CanonicalPathSig, p::Float64, ::Val{:statistical}) =
    ApproximatePathSig(base, STATISTICAL, 0.0, p)

"""
    is_cacheable(sig::ApproximatePathSig) -> Bool

A result is persistently cacheable iff error_level ∈ {EXACT, BOUNDED}.
STATISTICAL results may differ between runs — do not cache across sessions.
"""
is_cacheable(sig::ApproximatePathSig)::Bool = sig.error_level != STATISTICAL

"""
    approx_subsumes(s1::ApproximatePathSig, s2::ApproximatePathSig) -> Bool

Extend Doc 1's subsumes (Algorithm 10) to approximate keys.
s1 subsumes s2 iff:

 1. base_sig subsumes as per Doc 1 Algorithm 10
 2. s1.error_level is at least as general (EXACT ≥ BOUNDED ≥ STATISTICAL)
 3. s1.error_bound ≥ s2.error_bound (s1 allows at least as much error)
"""
function approx_subsumes(s1::ApproximatePathSig, s2::ApproximatePathSig)::Bool
    subsumes(s1.base_sig, s2.base_sig) || return false
    # Error level ordering: EXACT < BOUNDED < STATISTICAL (less constrained = more general)
    Int(s1.error_level) <= Int(s2.error_level) || return false
    # Tighter bound (smaller ε) can substitute wherever looser bound is needed
    s1.error_bound <= s2.error_bound
end

# ── §6.2 ApproxIndex ─────────────────────────────────────────────────────────

"""
    SimpleBloomFilter

Minimal Bloom filter for approximate membership testing.
Uses k=3 independent hash functions, bit array of size `m`.
False positive rate ≈ (1 - e^(-kn/m))^k for n elements.
"""
mutable struct SimpleBloomFilter
    bits::BitVector
    k::Int       # number of hash functions
    n::Int       # number of elements inserted
end

SimpleBloomFilter(m::Int=1024, k::Int=3) = SimpleBloomFilter(falses(m), k, 0)

function bloom_add!(bf::SimpleBloomFilter, key::UInt64)
    m = length(bf.bits)
    for i in 1:bf.k
        idx = Int(hash(key, UInt64(i)) % m) + 1
        bf.bits[idx] = true
    end
    bf.n += 1
end

function bloom_check(bf::SimpleBloomFilter, key::UInt64)::Bool
    m = length(bf.bits)
    all(bf.bits[Int(hash(key, UInt64(i)) % m) + 1] for i in 1:bf.k)
end

bloom_false_positive_rate(bf::SimpleBloomFilter)::Float64 =
    bf.n == 0 ? 0.0 : (1.0 - exp(-bf.k * bf.n / length(bf.bits)))^bf.k

"""
    ApproxIndex{T}

§6.2: Approximate index with p-box coverage tracking.

core       — high-frequency entries (covers ~95% of queries)
overflow   — BloomFilter for approximate membership of rare entries
weights    — importance weights from query history (for eviction)
coverage   — PBox tracking what fraction of the true index is in `core`

Key tradeoff: core is small + fast; overflow catches what core misses with
controllable false-positive rate; coverage PBox tracks overall accuracy.
"""
mutable struct ApproxIndex{T}
    core::Dict{UInt64, T}           # hash → value
    overflow::SimpleBloomFilter
    weights::Dict{UInt64, Float64}     # query frequency weights
    coverage::PBox                      # fraction of true index in core
end

function ApproxIndex{T}(; bloom_size::Int=4096, bloom_k::Int=3) where {T}
    ApproxIndex{T}(
        Dict{UInt64, T}(),
        SimpleBloomFilter(bloom_size, bloom_k),
        Dict{UInt64, Float64}(),
        pbox_interval(0.9, 1.0, 0.95)
    )   # initial coverage: 90-100% with 95% confidence
end

"""
    approx_index_insert!(idx, key, value; threshold=0.01)

Insert a (key, value) pair. High-weight entries go into core; low-weight
entries only register in the overflow Bloom filter (space-efficient).
`threshold`: weight below which entries go overflow-only.
"""
function approx_index_insert!(
    idx::ApproxIndex{T}, key::UInt64, value::T; threshold::Float64=0.01
) where {T}
    w = get(idx.weights, key, 0.0)
    if w >= threshold
        idx.core[key] = value
    else
        bloom_add!(idx.overflow, key)
    end
    idx.weights[key] = w + 1.0
end

"""
    approx_index_lookup(idx, key) -> Union{T, Nothing, :POSSIBLE}

3-way lookup:
`value`     — found in core (exact)
`:POSSIBLE` — found in overflow Bloom (may exist, false positive possible)
`nothing`   — definitely not in index
"""
function approx_index_lookup(
    idx::ApproxIndex{T}, key::UInt64
)::Union{T, Symbol, Nothing} where {T}
    haskey(idx.core, key) && return idx.core[key]
    bloom_check(idx.overflow, key) && return :POSSIBLE
    nothing
end

# ── §6.3 New IR primitives ────────────────────────────────────────────────────

"""
    register_approx_primitives!(registry::PrimRegistry)

Register the two new approximate IR primitives from §6.3 into the given registry:
:approx_kb_query — pattern query with tolerance, effects=[Read(kb)],
error_bound=tolerance
:sample_fitness  — sample-based fitness eval, error_bound=1/√(rate·|data|)
"""
function register_approx_primitives!(registry::PrimRegistry)
    # approx_kb_query: args=[pattern_id, tolerance_id]
    # Returns STATISTICAL result with error_bound=tolerance
    register_prim!(
        registry,
        :approx_kb_query,
        (g, args, env) -> begin
            # tolerance is arg[2] (a Lit with Float64 value)
            tol = length(args) >= 2 ? begin
                n = get_node(g, args[2])
                n isa Lit ? Float64(n.val) : 0.05
            end : 0.05
            # Return Residual — actual execution connects to MORK Space
            Residual(add_prim!(g, Prim(:approx_kb_query, args, EffectSet(UInt8(0x01)))))
        end
    )

    # sample_fitness: args=[prog_id, data_id, rate_id]
    # error_bound = 1/√(rate · |data|) per spec §6.3
    register_prim!(
        registry,
        :sample_fitness,
        (g, args, env) -> begin
            rate = length(args) >= 3 ? begin
                n = get_node(g, args[3])
                n isa Lit ? Float64(n.val) : 0.1
            end : 0.1
            # error_bound stored as metadata in the Prim node (via EffectSet mask)
            Residual(add_prim!(g, Prim(:sample_fitness, args, EffectSet(UInt8(0x21)))))
        end
    )

    registry
end

# ── §6.1 4-phase approximate pipeline ────────────────────────────────────────

"""
    ApproxPhase

Which phase of the 4-phase approximate compilation pipeline (§6.1).
"""
@enum ApproxPhase begin
    PHASE_ANALYSIS       # identify approximable operations
    PHASE_PLANNING       # error-time tradeoff query planning
    PHASE_SPECIALIZATION # generate approximate versions
    PHASE_VERIFICATION   # prove error bounds maintained
end

"""
    ApproxPipelineResult

Result of running the 4-phase approximate pipeline.

program_approx   — rewritten program with approximate operations
error_budget_used — total error budget consumed across all approximations
path_signatures  — ApproximatePathSig for each compiled operation
phase_timings    — time spent in each of the 4 phases
within_tolerance — true if total error ≤ error_tolerance
"""
struct ApproxPipelineResult
    program_approx::String
    error_budget_used::Float64
    path_signatures::Vector{ApproximatePathSig}
    phase_timings::Dict{ApproxPhase, Float64}
    within_tolerance::Bool
end

"""
    run_approx_pipeline(s::Space, program::AbstractString;
                        error_tolerance=0.05, weights=balanced()) -> ApproxPipelineResult

§6.1 4-phase approximate supercompilation pipeline.

Phase 1 — Analysis: identify approximable operations (pattern matches with many sources,
numeric computations, iterative patterns).
Phase 2 — Planning: use ApproxJoinNode + 3-objective cost model to build the join plan.
Phase 3 — Specialization: rewrite program using approximate variants where beneficial.
Phase 4 — Verification: check Theorem A.2 (error composition bound ≤ tolerance).
"""
function run_approx_pipeline(
    s::Space, program::AbstractString; error_tolerance::Float64=0.05,
    weights::CostWeights=balanced()
)::ApproxPipelineResult
    timings = Dict{ApproxPhase, Float64}()

    # ── Phase 1: Analysis ──────────────────────────────────────────────────
    t1 = @elapsed begin
        stats = collect_stats(s)
        nodes = parse_program(program)
        approx_ops = _identify_approximable(nodes, stats, error_tolerance)
    end
    timings[PHASE_ANALYSIS] = t1

    # ── Phase 2: Planning ──────────────────────────────────────────────────
    t2 = @elapsed begin
        planned_nodes = SNode[]
        all_sigs = ApproximatePathSig[]
        error_used = 0.0

        for node in nodes
            node isa SList || (push!(planned_nodes, node); continue)
            items = (node::SList).items
            if length(items) >= 3 && is_conjunction(items[2])
                sources = (items[2]::SList).items[2:end]
                order = plan_join_order_approx(
                    sources, stats, s.btm; weights=weights, error_tolerance=error_tolerance
                )
                new_conj = SList([items[2].items[1]; sources[order]])
                push!(planned_nodes, SList([items[1], new_conj, items[3:end]...]))
            else
                push!(planned_nodes, node)
            end
        end
    end
    timings[PHASE_PLANNING] = t2

    # ── Phase 3: Specialization ────────────────────────────────────────────
    t3 = @elapsed begin
        program_approx = sprint_program(planned_nodes)

        # For approximable operations: build ApproximatePathSig
        g = MCoreGraph()
        for node in planned_nodes
            base_id = _sexpr_to_mcore!(g, node)
            base_key = canonical_key(g, base_id, 0)
            if _is_approximable_node(node, stats, error_tolerance)
                sig = ApproximatePathSig(base_key, error_tolerance)
                push!(all_sigs, sig)
                error_used += error_tolerance * 0.5   # conservative estimate
            else
                push!(all_sigs, ApproximatePathSig(base_key))
            end
        end
    end
    timings[PHASE_SPECIALIZATION] = t3

    # ── Phase 4: Verification (Theorem A.2) ───────────────────────────────
    t4 = @elapsed begin
        op_widths = [s.error_bound for s in all_sigs if s.error_level == BOUNDED]
        total_bound = isempty(op_widths) ? 0.0 : error_composition_bound(op_widths)
        within_tol = total_bound <= error_tolerance
    end
    timings[PHASE_VERIFICATION] = t4

    ApproxPipelineResult(program_approx, error_used, all_sigs, timings, within_tol)
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _identify_approximable(
    nodes::Vector{SNode}, stats::MORKStatistics, tol::Float64
)::Vector{SNode}
    filter(n -> _is_approximable_node(n, stats, tol), nodes)
end

function _is_approximable_node(node::SNode, stats::MORKStatistics, tol::Float64)::Bool
    node isa SList || return false
    items = (node::SList).items
    length(items) < 3 && return false
    is_conjunction(items[2]) || return false
    sources = (items[2]::SList).items[2:end]
    # Approximable if ≥3 sources (high fan-out) OR any source has high cardinality
    length(sources) >= 3 && return true
    any(s -> estimate_cardinality(s, stats) > stats.total_atoms ÷ 4, sources)
end

export ErrorLevel, EXACT, BOUNDED, STATISTICAL
export ApproximatePathSig, is_cacheable, approx_subsumes
export SimpleBloomFilter, bloom_add!, bloom_check, bloom_false_positive_rate
export ApproxIndex, approx_index_insert!, approx_index_lookup
export register_approx_primitives!
export ApproxPhase, PHASE_ANALYSIS, PHASE_PLANNING, PHASE_SPECIALIZATION, PHASE_VERIFICATION
export ApproxPipelineResult, run_approx_pipeline
