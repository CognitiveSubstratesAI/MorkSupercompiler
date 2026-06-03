"""
EvoSpecializer — gated evolutionary algorithm specialization.

Implements §8 of the MM2 Supercompiler spec (Goertzel, Oct 2025):
  §8.1  Algorithm 12 — GatedEvolutionarySpecialization
          Decide whether to specialize a fitness function and at what level.
  §8.2  Algorithm 13 — CanReuseFitnessCache
          AST-diff-based cache validity check for offspring fitness reuse.

Also integrates with Approximate Supercompilation spec §5 (MOSES domain):
  EvolutionaryPBox — fitness p-box with rank uncertainty and heritability
  ApproximateFitness — Hoeffding-bounded sample fitness (Algorithm 5)
  TournamentWithPBox — probabilistic tournament selection (Algorithm 6)
  AllocateEvaluations — value-of-information resource allocation (Algorithm 7)

Design: EvoSpecializer is a pure computation module — it decides HOW to
specialize but does not perform Julia code generation itself (that is
MM2Compiler's job). It operates on M-Core NodeIDs and returns decisions
and rewritten graph fragments.
"""

# ── Specialization levels (§8.1) ─────────────────────────────────────────────

"""
    SpecLevel

Tiered specialization levels from Algorithm 12:
SPEC_GENERIC     — no specialization (amortization takes ≥ 50% of run)
SPEC_INCREMENTAL — incremental update specialization (10–50% of run)
SPEC_VECTORIZED  — full vectorized specialization (< 10% of run)
"""
@enum SpecLevel begin
    SPEC_GENERIC
    SPEC_INCREMENTAL
    SPEC_VECTORIZED
end

"""
    SpecDecision

Output of Algorithm 12.  Carries the chosen level plus the amortization ratio
(specialization_cost / total_evals) for diagnostics.
"""
struct SpecDecision
    level::SpecLevel
    amortization_ratio::Float64   # specialization_cost / total_evals
end

# ── Algorithm 12 — GatedEvolutionarySpecialization (§8.1) ─────────────────────

"""
    should_specialize(avg_eval_time, population_size, expected_generations,
                      specialization_cost) -> SpecDecision

Algorithm 12 from §8.1.  Decides specialization level by comparing
amortization time against total evaluation budget.

Thresholds (verbatim from spec):
amortization < 10% of total evals → SPEC_VECTORIZED
amortization < 50% of total evals → SPEC_INCREMENTAL
otherwise                         → SPEC_GENERIC
"""
function should_specialize(
    avg_eval_time::Float64, population_size::Int, expected_generations::Int,
    specialization_cost::Float64
)::SpecDecision
    total_evals = population_size * expected_generations
    amortization = avg_eval_time > 0.0 ? specialization_cost / avg_eval_time : Inf
    amortization_ratio = amortization / max(1, total_evals)

    level = if amortization_ratio < 0.10
        SPEC_VECTORIZED
    elseif amortization_ratio < 0.50
        SPEC_INCREMENTAL
    else
        SPEC_GENERIC
    end

    SpecDecision(level, amortization_ratio)
end

# ── AST diff — for cache reuse (§8.2) ────────────────────────────────────────

"""
    ChangeKind

Types of change between two M-Core trees (§8.2 CanReuseFitnessCache):
CHANGE_STRUCTURAL — structural node added/removed/reordered → invalidates cache
CHANGE_CONSTANT   — a constant leaf was changed
CHANGE_NONE       — no change (identical subtree)
"""
@enum ChangeKind begin
    CHANGE_NONE
    CHANGE_CONSTANT
    CHANGE_STRUCTURAL
end

"""
    ASTDiff

Result of diffing two M-Core expressions.
num_changes     — total number of changed nodes
changes         — list of (node_id, ChangeKind) pairs
max_changes     — threshold above which cache reuse is prohibited
"""
struct ASTDiff
    num_changes::Int
    changes::Vector{Tuple{NodeID, ChangeKind}}
end

"""
    compute_ast_diff(g, child_id, parent_id; max_changes) -> ASTDiff

Compute a structural diff between `child_id` and `parent_id` in graph `g`.
Stops early if `max_changes` is exceeded (for efficiency).
"""
function compute_ast_diff(
    g::MCoreGraph, child_id::NodeID, parent_id::NodeID; max_changes::Int=10
)::ASTDiff
    changes = Tuple{NodeID, ChangeKind}[]
    _diff_nodes!(g, child_id, parent_id, changes, max_changes)
    ASTDiff(length(changes), changes)
end

function _diff_nodes!(g, cid, pid, changes, max_changes)
    length(changes) >= max_changes && return nothing
    (!isvalid(cid) || !isvalid(pid)) && (cid != pid) &&
        begin
            push!(changes, (cid, CHANGE_STRUCTURAL));
            return nothing
        end
    (!isvalid(cid) && !isvalid(pid)) && return nothing

    cn = get_node(g, cid)
    pn = get_node(g, pid)

    if typeof(cn) != typeof(pn)
        push!(changes, (cid, CHANGE_STRUCTURAL));
        return nothing
    end

    if cn isa Sym && pn isa Sym
        (cn::Sym).name != (pn::Sym).name && push!(changes, (cid, CHANGE_STRUCTURAL))
        return nothing
    end
    if cn isa Lit && pn isa Lit
        (cn::Lit).val != (pn::Lit).val && push!(changes, (cid, CHANGE_CONSTANT))
        return nothing
    end
    if cn isa Con && pn isa Con
        cc = cn::Con;
        pc = pn::Con
        if cc.head != pc.head || length(cc.fields) != length(pc.fields)
            push!(changes, (cid, CHANGE_STRUCTURAL));
            return nothing
        end
        for (cf, pf) in zip(cc.fields, pc.fields)
            _diff_nodes!(g, cf, pf, changes, max_changes)
        end
        return nothing
    end
    if cn isa App && pn isa App
        ca = cn::App;
        pa = pn::App
        length(ca.args) != length(pa.args) &&
            (push!(changes, (cid, CHANGE_STRUCTURAL)); return nothing)
        _diff_nodes!(g, ca.fun, pa.fun, changes, max_changes)
        for (ca_, pa_) in zip(ca.args, pa.args)
            _diff_nodes!(g, ca_, pa_, changes, max_changes)
        end
        return nothing
    end
    # All other node kinds: structural change if IDs differ
    cid != pid && push!(changes, (cid, CHANGE_STRUCTURAL))
end

# ── Algorithm 13 — CanReuseFitnessCache (§8.2) ────────────────────────────────

"""
    CacheMetadata

Metadata about a cached fitness evaluation.
max_changes      — maximum AST edits that still permit reuse
sensitive_nodes  — NodeIDs of constants whose change invalidates the cache
cached_fitness   — the cached fitness value
eval_count       — how many times this individual has been evaluated
"""
struct CacheMetadata
    max_changes::Int
    sensitive_nodes::Set{NodeID}
    cached_fitness::Float64
    eval_count::Int
end
CacheMetadata(fit::Float64) = CacheMetadata(3, Set{NodeID}(), fit, 1)

"""
    can_reuse_cache(g, child_id, parent_id, meta) -> Bool

Algorithm 13 (CanReuseFitnessCache) from §8.2.

Returns true iff the offspring `child_id` can reuse the cached fitness
for parent `parent_id`, given `meta`.

Rules (verbatim from spec):

 1. diff.num_changes > meta.max_changes → false
 2. Any structural change → false
 3. Any constant change in meta.sensitive_nodes → false
 4. Otherwise → true
"""
function can_reuse_cache(
    g::MCoreGraph, child_id::NodeID, parent_id::NodeID, meta::CacheMetadata
)::Bool
    diff = compute_ast_diff(g, child_id, parent_id; max_changes=meta.max_changes + 1)

    diff.num_changes > meta.max_changes && return false

    for (nid, kind) in diff.changes
        kind == CHANGE_STRUCTURAL && return false
        kind == CHANGE_CONSTANT && nid in meta.sensitive_nodes && return false
    end

    true
end

# ── Approximate fitness from §5 of Approximate Supercompilation spec ──────────

"""
    EvolutionaryPBox

Fitness representation for approximate evolution (§5.1 of approx spec).
individual_id    — identifier for this individual
fitness_pbox     — p-box over fitness estimate (Hoeffding-bounded)
rank_pbox        — p-box over rank in population (matters for selection)
heritability     — fraction of fitness that is heritable (0=environment, 1=genetic)
evaluation_count — number of times evaluated (diminishing returns)
"""
struct EvolutionaryPBox
    individual_id::Int
    fitness_pbox::PBox
    rank_pbox::PBox
    heritability::Float64
    evaluation_count::Int
end

"""
    approximate_fitness(sample_fitness, n_samples; delta=0.05) -> PBox

Algorithm 5 (ApproximateFitness) from §5.2 of the Approximate Supercompilation spec.
Returns a Hoeffding-bounded p-box over the true fitness.

Hoeffding bound: P(|X̄ - E[X]| > t) ≤ 2·exp(-2nt²/(b-a)²)
For bounded fitness in [0,1]: ε = √(ln(2/δ) / (2·n_samples))

The 5% tail reserve covers discontinuities in fitness landscapes.
"""
function approximate_fitness(
    sample_fitness::Float64, n_samples::Int; delta::Float64=0.05
)::PBox
    n_samples <= 0 && return PBox(0.0, 1.0, 1.0)

    epsilon = sqrt(log(2.0 / delta) / (2.0 * n_samples))
    lo = clamp(sample_fitness - epsilon, 0.0, 1.0)
    hi = clamp(sample_fitness + epsilon, 0.0, 1.0)

    # 95% main interval + 5% tail reserve (§5.2)
    PBox([(lo, hi), (0.0, 1.0)], [0.95, 0.05], 1.0)
end

"""
    allocate_evaluations(population, budget) -> Vector{Tuple{Int,Float64}}

Algorithm 7 (AllocateEvaluations) from §5.5 of the Approximate Supercompilation spec.
Returns (individual_id, priority) pairs sorted descending.

Priority = uncertainty × could_be_best × novelty  (§5.5)
uncertainty  = width of fitness_pbox
could_be_best = estimated P(this individual is best)
novelty      = 1 / (1 + evaluation_count)  (diminishing returns)
"""
function allocate_evaluations(
    population::Vector{EvolutionaryPBox}, budget::Int
)::Vector{Tuple{Int, Float64}}
    isempty(population) && return Tuple{Int, Float64}[]

    best_lo = maximum(p.fitness_pbox.intervals[1][1] for p in population)

    priorities = Tuple{Int, Float64}[]
    for p in population
        lo, hi = p.fitness_pbox.intervals[1]
        uncertainty = hi - lo
        could_be_best = hi >= best_lo ? 1.0 : max(0.0, (hi - lo) / (best_lo - lo + 1e-9))
        novelty = 1.0 / (1.0 + p.evaluation_count)
        priority = uncertainty * could_be_best * novelty
        push!(priorities, (p.individual_id, priority))
    end

    sort!(priorities; by=x -> -x[2])
    priorities[1:min(budget, length(priorities))]
end

export SpecLevel, SPEC_GENERIC, SPEC_INCREMENTAL, SPEC_VECTORIZED
export SpecDecision, should_specialize
export ChangeKind, CHANGE_NONE, CHANGE_CONSTANT, CHANGE_STRUCTURAL
export ASTDiff, compute_ast_diff
export CacheMetadata, can_reuse_cache
export EvolutionaryPBox, approximate_fitness, allocate_evaluations
