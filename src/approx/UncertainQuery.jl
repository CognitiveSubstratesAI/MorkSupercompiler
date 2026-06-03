"""
UncertainQuery — error-aware query planning with p-box cardinality estimates.

Implements §3 of the Approximate Supercompilation spec (Goertzel, Oct 2025):
  §3.1  Extended cost model: Cost_total = α·Time + β·Error + γ·Variance
  §3.2  Algorithm 2 — EstimateCardinalityPBox (Hoeffding-bounded cardinality)
  §3.3  Algorithm 3 (ApproximateSplit) — error-tolerance-gated branch pruning

The key extension over exact planning (Doc 1):
  - Cardinalities become PBoxes, not scalars → track uncertainty in estimates
  - Join ordering optimizes the 3-objective cost function, not just time
  - ApproximateSplit adds an explicit error_tolerance budget per split decision
"""

# ── §3.1 Extended cost model ──────────────────────────────────────────────────

"""
    CostWeights

User-specified weights for the 3-objective cost model (§3.1):
α — time weight      (performance focus)
β — error weight     (accuracy focus)
γ — variance weight  (safety/reliability focus)

Presets:
safety_critical()   — high β, γ; exact computation preferred
exploratory()       — high α, moderate β
balanced()          — equal weights
"""
struct CostWeights
    α::Float64   # time
    β::Float64   # error
    γ::Float64   # variance
end

safety_critical()::CostWeights = CostWeights(0.1, 0.6, 0.3)
exploratory()::CostWeights = CostWeights(0.7, 0.2, 0.1)
balanced()::CostWeights = CostWeights(1/3, 1/3, 1/3)

"""
    total_cost(time, error_pbox, weights) -> Float64

§3.1: Cost_total = α·Time + β·Error + γ·Variance

time       — estimated execution time (seconds or normalized)
error_pbox — PBox representing the error distribution
weights    — CostWeights (α, β, γ)
"""
function total_cost(time::Float64, error_pbox::PBox, weights::CostWeights)::Float64
    expected_error = width(error_pbox) / 2.0
    variance_term = max_width(error_pbox)^2
    weights.α * time + weights.β * expected_error + weights.γ * variance_term
end

# ── §3.2 Algorithm 2 — EstimateCardinalityPBox ────────────────────────────────

"""
    EstimateCardinalityPBox

Result of Algorithm 2: a p-box cardinality estimate with a Hoeffding confidence
interval. Richer than a scalar — carries uncertainty about the estimate itself.
"""
struct EstimateCardinalityPBox
    point_estimate::Int       # from exact stats (EstimateMatches)
    pbox::PBox      # Hoeffding-bounded interval around the estimate
    allows_sampling::Bool      # true when sampling is feasible for this pattern
end

"""
    estimate_cardinality_pbox(src::SNode, stats::MORKStatistics,
                              btm=nothing; confidence=0.95) -> EstimateCardinalityPBox

Algorithm 2 (EstimateCardinalityPBox) from §3.2.

If `btm` is provided and the pattern allows sampling, uses Hoeffding bounds
to build a proper p-box around the cardinality estimate.
Otherwise falls back to a point estimate (exact mode).

Hoeffding bound: ε = √(ln(2/δ) / 2n) where n = sample_size = √(total_atoms).
"""
function estimate_cardinality_pbox(
    src::SNode, stats::MORKStatistics, btm=nothing; confidence::Float64=0.95
)::EstimateCardinalityPBox
    base = estimate_cardinality(src, stats)   # Algorithm 2 from Doc 1

    # Can we sample? Only if btm available and pattern allows it
    allows_sampling = btm !== nothing && (src isa SList)

    if allows_sampling
        total = max(1, stats.total_atoms)
        n_sample = max(1, isqrt(total))   # √n per spec §3.2

        # Dynamic count from btm as our "observed" count within sample
        observed = dynamic_count(btm, src)
        # Scale to full space
        est_full = round(Int, observed * (total / max(1, n_sample)))

        δ = 1.0 - confidence
        ε = hoeffding_epsilon(n_sample, δ; b=Float64(total))

        lo = max(0.0, est_full - ε)
        hi = est_full + ε
        pb = pbox_interval(lo, hi, confidence)

        EstimateCardinalityPBox(base, pb, true)
    else
        # Exact mode: point estimate
        pb = pbox_exact(Float64(base))
        EstimateCardinalityPBox(base, pb, false)
    end
end

# ── §3.3 Algorithm 3 (ApproximateSplit) ──────────────────────────────────────

"""
    ApproxBranch

One branch from an ApproximateSplit, carrying its error contribution.

branch_id      — SNode or NodeID representing this branch
probability    — P(this branch executes)
error_contrib  — error introduced if this branch is pruned (= 1 - probability)
is_selected    — true if included in the plan (false if pruned)
"""
struct ApproxBranch
    branch_id::SNode
    probability::Float64
    error_contrib::Float64
    is_selected::Bool
end

"""
    ApproximateSplitResult

Result of Algorithm 3 (ApproximateSplit).
selected       — branches included in the plan
pruned         — branches excluded (each contributes error_contrib to total)
total_error    — sum of pruned branch probabilities (= actual approximation error)
within_budget  — true if total_error ≤ error_tolerance
"""
struct ApproximateSplitResult
    selected::Vector{ApproxBranch}
    pruned::Vector{ApproxBranch}
    total_error::Float64
    within_budget::Bool
end

"""
    approximate_split(branches::Vector{Tuple{SNode,Float64}},
                      error_tolerance::Float64) -> ApproximateSplitResult

Algorithm 3 (ApproximateSplit) from §3.3.

Key insight: if a branch has probability p < error_tolerance, ignoring it
introduces at most error_tolerance error. We can provably bound the total error
introduced by pruning as exactly the sum of pruned branch probabilities.

Steps:

 1. Sort branches by probability descending
 2. Select until cumulative_prob ≥ 1.0 - error_tolerance
 3. Prune the rest — they contribute ≤ error_tolerance total error

Returns a sound approximation: the pruned mass IS the error budget consumed.
"""
function approximate_split(
    branches::AbstractVector, error_tolerance::Float64
)::ApproximateSplitResult
    isempty(branches) &&
        return ApproximateSplitResult(ApproxBranch[], ApproxBranch[], 0.0, true)

    # Sort by probability descending
    sorted = sort(branches; by=x -> -x[2])

    selected = ApproxBranch[]
    pruned = ApproxBranch[]
    cumulative = 0.0

    for (node, prob) in sorted
        if cumulative >= 1.0 - error_tolerance
            # Pruning is sound: remaining mass < error_tolerance
            push!(pruned, ApproxBranch(node, prob, prob, false))
        else
            push!(selected, ApproxBranch(node, prob, 0.0, true))
            cumulative += prob
        end
    end

    total_error = sum(b.error_contrib for b in pruned; init=0.0)
    within_budget = total_error <= error_tolerance

    ApproximateSplitResult(selected, pruned, total_error, within_budget)
end

# ── Approximate join ordering with cost model ─────────────────────────────────

"""
    ApproxJoinNode

Like `JoinNode` from QueryPlanner but with a PBox cardinality estimate.
"""
struct ApproxJoinNode
    source::SNode
    card_pbox::EstimateCardinalityPBox
    vars_out::Set{String}
    vars_in::Set{String}
end

"""
    plan_join_order_approx(sources, stats, btm=nothing;
                           weights=balanced(), error_tolerance=0.05) -> Vector{Int}

Approximate join ordering using the 3-objective cost model (§3.1).

Uses PBox cardinality estimates rather than scalars. Optimizes:
Cost = α·Time + β·Error + γ·Variance

For each candidate next source, estimates the cost contribution and picks
the lowest-cost option (greedy, like Doc 1, but with uncertainty tracking).
"""
function plan_join_order_approx(
    sources::Vector{SNode},
    stats::MORKStatistics,
    btm=nothing;
    weights::CostWeights=balanced(),
    error_tolerance::Float64=0.05
)::Vector{Int}
    n = length(sources)
    n <= 1 && return collect(1:n)

    # Build ApproxJoinNodes
    seen_vars = Set{String}()
    nodes = ApproxJoinNode[]
    for src in sources
        all_vars = collect_var_names(src)
        new_vars = setdiff(all_vars, seen_vars)
        used_vars = intersect(all_vars, seen_vars)
        union!(seen_vars, new_vars)
        epbox = estimate_cardinality_pbox(src, stats, btm)
        push!(nodes, ApproxJoinNode(src, epbox, new_vars, used_vars))
    end

    # Greedy selection by total_cost
    remaining = collect(1:n)
    order = Int[]
    bound = Set{String}()

    while !isempty(remaining)
        best_pos = 0
        best_cost = Inf

        for (pos, i) in enumerate(remaining)
            node = nodes[i]
            epbox = node.card_pbox

            n_unbound = length(setdiff(node.vars_in, bound))
            # Time estimate: point_estimate scaled by unbound penalty
            t_est = Float64(epbox.point_estimate) * (4.0 ^ n_unbound)

            cost = total_cost(t_est, epbox.pbox, weights)
            if cost < best_cost
                best_cost = cost
                best_pos = pos
            end
        end

        chosen_i = remaining[best_pos]
        push!(order, chosen_i)
        union!(bound, nodes[chosen_i].vars_out)
        deleteat!(remaining, best_pos)
    end

    order
end

export CostWeights, safety_critical, exploratory, balanced, total_cost
export EstimateCardinalityPBox, estimate_cardinality_pbox
export ApproxBranch, ApproximateSplitResult, approximate_split
export ApproxJoinNode, plan_join_order_approx
