"""
QueryPlanner — effect-aware join ordering for MORK exec patterns.

Implements Algorithm 6 (EffectAwarePlanning) from MM2 Supercompiler §5.3.1.

Key insight (§4.2 EffectCommutes + §5.3):
  All sources in a MORK `,` pattern list have effect `Read(space)`.
  `Commutes(Read(r), Read(r)) = true`.
  → Sources form a **pure region** — they can be freely reordered.
  → Apply cost_based_join_order to find the optimal ordering.

The optimal join order for a pure region minimizes expected intermediate
result size: place most-selective sources first (lowest cardinality estimate),
propagating variable bindings to reduce subsequent searches.

Variable flow analysis:
  Sources that INTRODUCE variables should precede sources that USE them.
  A source S2 uses variable \$x introduced by S1 → S1 should precede S2.
  This is the semi-join optimization: bound variables restrict later searches.
"""

# ── Effect algebra (§4.1–4.2) ─────────────────────────────────────────────────

"""
Effect types from §4.1.
"""
@enum EffectKind begin
    EFF_PURE
    EFF_READ
    EFF_APPEND
    EFF_WRITE
    EFF_OBSERVE
end

"""
    effects_commute(e1::EffectKind, e2::EffectKind) -> Bool

Algorithm 1 — EffectCommutes from §4.2.  MORK read sources are always
`EFF_READ` on the same space resource; they commute with each other.
"""
function effects_commute(e1::EffectKind, e2::EffectKind)::Bool
    e1 == EFF_PURE && return true
    e2 == EFF_PURE && return true
    e1 == EFF_READ && e2 == EFF_READ && return true
    e1 == EFF_READ && e2 == EFF_OBSERVE && return true
    e1 == EFF_OBSERVE && e2 == EFF_OBSERVE && return true
    e1 == EFF_APPEND && e2 == EFF_APPEND && return true   # diff resource; same-resource handled at call site
    false
end

# ── Variable flow analysis ─────────────────────────────────────────────────────

"""
Collect variable names from an SNode.
"""
function collect_var_names(node::SNode)::Set{String}
    out = Set{String}()
    _collect_vars!(out, node)
    out
end
function _collect_vars!(out, node::SNode)
    if node isa SVar
        push!(out, (node::SVar).name)
    elseif node isa SList
        for c in (node::SList).items
            ;
            _collect_vars!(out, c)
        end
    end
end

"""
    JoinNode

Represents one source in the query plan with its cost estimate and variable I/O.

Fields:

  - `source`       — the original SNode
  - `cardinality`  — estimated match count (from statistics or dynamic count)
  - `vars_out`     — variables this source introduces (first occurrence in pattern)
  - `vars_in`      — variables this source uses but does not introduce
  - `effect`       — always EFF_READ for MORK exec sources
"""
struct JoinNode
    source::SNode
    cardinality::Int
    vars_out::Set{String}   # variables this source introduces
    vars_in::Set{String}   # variables this source constrains (already bound)
    effect::EffectKind
end

"""
    build_join_nodes(sources::Vector{SNode}, stats::MORKStatistics) -> Vector{JoinNode}

Build JoinNode descriptors for each source, computing variable flow.
Variable ownership: a variable is "introduced" by the first source that mentions it.
"""
function build_join_nodes(sources::Vector{SNode}, stats::MORKStatistics)::Vector{JoinNode}
    seen_vars = Set{String}()
    nodes = JoinNode[]
    for src in sources
        all_vars = collect_var_names(src)
        new_vars = setdiff(all_vars, seen_vars)
        used_vars = intersect(all_vars, seen_vars)
        union!(seen_vars, new_vars)
        card = estimate_cardinality(src, stats)
        push!(nodes, JoinNode(src, card, new_vars, used_vars, EFF_READ))
    end
    nodes
end

"""
    build_join_nodes_dynamic(sources::Vector{SNode}, btm) -> Vector{JoinNode}

Like `build_join_nodes` but uses dynamic btm-prefix cardinality (Algorithm 3).
More accurate than statistics-based estimation, at O(1) per source.
"""
function build_join_nodes_dynamic(sources::Vector{SNode}, btm)::Vector{JoinNode}
    seen_vars = Set{String}()
    nodes = JoinNode[]
    for src in sources
        all_vars = collect_var_names(src)
        new_vars = setdiff(all_vars, seen_vars)
        used_vars = intersect(all_vars, seen_vars)
        union!(seen_vars, new_vars)
        card = prefix_sample_count(btm, src)
        push!(nodes, JoinNode(src, card, new_vars, used_vars, EFF_READ))
    end
    nodes
end

# ── Algorithm 6 — EffectAwarePlanning (§5.3.1) ────────────────────────────────

# Semi-join penalty per unbound dependency: cost *= UNBOUND_DEP_PENALTY ^ n_unbound.
# This is a concretization of the spec's abstract `cost_based_join_order` (§5.3.1) —
# the 4× figure reflects empirical observation that one unbound join variable typically
# multiplies required I/O by ~4× (similar to the standard 4× rule-of-thumb in
# relational cost models). Tune here if profiling shows a different breakeven.
const UNBOUND_DEP_PENALTY = 4

"""
    plan_join_order(nodes::Vector{JoinNode}) -> Vector{Int}

Algorithm 6 — cost_based_join_order for a pure region (all sources are Read).

The algorithm:

 1. All sources are `EFF_READ` → pure region → free reordering (§4.2)
 2. Use a greedy selection: repeatedly pick the node with the lowest
    *effective* cardinality given currently-bound variables.
 3. A node's effective cardinality is reduced when its `vars_in` are all
    bound (semi-join pushdown): treat its cardinality as 1 if fully bound.

Returns an index permutation (1-based) of `nodes` in optimal order.
"""
function plan_join_order(nodes::Vector{JoinNode})::Vector{Int}
    n = length(nodes)
    n <= 1 && return collect(1:n)

    remaining = collect(1:n)
    order = Int[]
    bound = Set{String}()

    while !isempty(remaining)
        # Effective cardinality: reduce if all vars_in are bound
        best_idx = 0
        best_cost = typemax(Int)
        for (pos, i) in enumerate(remaining)
            node = nodes[i]
            # Penalty: if this node requires unbound variables, de-prioritize
            # Required-but-unbound increases effective cost by a factor
            n_unbound_deps = length(setdiff(node.vars_in, bound))
            cost = node.cardinality
            if n_unbound_deps > 0
                cost = cost * (UNBOUND_DEP_PENALTY ^ n_unbound_deps)
            end
            if cost < best_cost
                best_cost = cost
                best_idx = pos
            end
        end
        chosen_pos = best_idx
        chosen_i = remaining[chosen_pos]
        push!(order, chosen_i)
        union!(bound, nodes[chosen_i].vars_out)
        deleteat!(remaining, chosen_pos)
    end
    order
end

"""
    plan_join_order_static(sources::Vector{SNode}) -> Vector{Int}

Simplified join planning using only static selectivity scores (no stats/btm needed).
Uses static_score from Selectivity.jl.  Cheaper but less accurate.
"""
function plan_join_order_static(sources::Vector{SNode})::Vector{Int}
    n = length(sources)
    n <= 1 && return collect(1:n)
    scores = static_score.(sources)
    sortperm(scores; alg=MergeSort)
end

# ── Conjunction planning ───────────────────────────────────────────────────────

"""
    plan_conjunction(conj::SList, stats::MORKStatistics) -> SList

Apply effect-aware join planning to a `,` conjunction list using statistics.
Returns a new conjunction list with sources in the planned order.
"""
function plan_conjunction(conj::SList, stats::MORKStatistics)::SList
    is_conjunction(conj) || return conj
    items = conj.items
    head = items[1]
    sources = items[2:end]
    length(sources) <= 1 && return conj

    nodes = build_join_nodes(sources, stats)
    perm = plan_join_order(nodes)
    SList([head; sources[perm]])
end

"""
    plan_conjunction_dynamic(conj::SList, btm) -> SList

Apply join planning using dynamic cardinality from btm prefix counts.
"""
function plan_conjunction_dynamic(conj::SList, btm)::SList
    is_conjunction(conj) || return conj
    items = conj.items
    head = items[1]
    sources = items[2:end]
    length(sources) <= 1 && return conj

    nodes = build_join_nodes_dynamic(sources, btm)
    perm = plan_join_order(nodes)
    SList([head; sources[perm]])
end

# ── Program-level planning ────────────────────────────────────────────────────

"""
    plan_program(program::AbstractString, stats::MORKStatistics) -> String

Apply query planning to all multi-source conjunction lists in `program`.
"""
function plan_program(program::AbstractString, stats::MORKStatistics)::String
    nodes = parse_program(program)
    sprint_program(SNode[_plan_atom(n, stats) for n in nodes])
end

"""
    plan_program_dynamic(program::AbstractString, btm) -> String

Apply query planning using dynamic btm-prefix cardinality.
`btm` should contain background facts but NOT the exec/rule atoms being planned.
"""
function plan_program_dynamic(program::AbstractString, btm)::String
    nodes = parse_program(program)
    sprint_program(SNode[_plan_atom_dynamic(n, btm) for n in nodes])
end

function _plan_atom(node::SNode, stats::MORKStatistics)::SNode
    node isa SList || return node
    items = (node::SList).items
    length(items) < 3 && return node
    is_conjunction(items[2]) || return node
    new_conj = plan_conjunction(items[2]::SList, stats)
    SList([items[1], new_conj, items[3:end]...])
end

function _plan_atom_dynamic(node::SNode, btm)::SNode
    node isa SList || return node
    items = (node::SList).items
    length(items) < 3 && return node
    is_conjunction(items[2]) || return node
    new_conj = plan_conjunction_dynamic(items[2]::SList, btm)
    SList([items[1], new_conj, items[3:end]...])
end

"""
    plan_report(program::AbstractString, stats::MORKStatistics) -> String

Human-readable join-plan report for all multi-source conjunctions in `program`.
Shows original order, estimated cardinalities, and planned order.
"""
function plan_report(program::AbstractString, stats::MORKStatistics)::String
    io = IOBuffer()
    nodes = parse_program(program)
    for node in nodes
        node isa SList || continue
        items = (node::SList).items
        length(items) < 3 || !is_conjunction(items[2]) && continue
        conj = items[2]::SList
        sources = conj.items[2:end]
        length(sources) <= 1 && continue

        jnodes = build_join_nodes(sources, stats)
        perm = plan_join_order(jnodes)

        println(io, "\nPattern: ", sprint_sexpr(items[1]))
        println(io, "  Sources (original order, estimated cardinality):")
        for (k, (s, jn)) in enumerate(zip(sources, jnodes))
            println(
                io,
                "    [$k] card=$(jn.cardinality) vars_in=$(jn.vars_in) vars_out=$(jn.vars_out)  ",
                sprint_sexpr(s)
            )
        end
        println(io, "  Planned order: ", perm)
        println(io, "  Reordered sources:")
        for (k, i) in enumerate(perm)
            println(io, "    [$k] ", sprint_sexpr(sources[i]))
        end
    end
    String(take!(io))
end

# ── §5.3.1 Algorithm 6 full implementation — pure-region identification ───────

"""
    PureRegion

A contiguous group of sources that all have mutually commuting effects.
Pure regions can be freely reordered (cost-based join order).
Non-pure regions require topological ordering respecting effect dependencies.
"""
struct PureRegion
    sources::Vector{SNode}
    is_pure::Bool
    effects::Vector{EffectKind}
end

"""
    EffectBarrier

Inserted between non-commuting plan regions to prevent illegal reordering.
Mirrors the spec's `EffectBarrier(region.effects)` insertion point.
"""
struct EffectBarrier
    blocking_effects::Vector{EffectKind}
end

"""
    identify_pure_regions(sources::Vector{SNode}) -> Vector{PureRegion}

Algorithm 6 line: `regions ← identify_pure_regions(query, effect_analysis)`.
Groups consecutive sources into pure regions (all Read → all commute → one region).
For MORK exec patterns this always produces a single pure region, since all
sources are Read(space). Non-pure regions arise with mixed Append/Write sources.
"""
function identify_pure_regions(sources::Vector{SNode})::Vector{PureRegion}
    isempty(sources) && return PureRegion[]
    # For MORK: all exec sources are EFF_READ — one pure region
    # Future: detect mixed effects by inspecting source patterns
    [PureRegion(sources, true, fill(EFF_READ, length(sources)))]
end

"""
    plan_query(sources, stats) -> Tuple{Vector{SNode}, Vector{EffectBarrier}}

Full Algorithm 6 (EffectAwarePlanning) from §5.3.1.
Returns reordered sources + any effect barriers inserted between regions.
"""
function plan_query(
    sources::Vector{SNode}, stats::MORKStatistics
)::Tuple{Vector{SNode}, Vector{EffectBarrier}}
    regions = identify_pure_regions(sources)
    plan = SNode[]
    barriers = EffectBarrier[]

    for (i, region) in enumerate(regions)
        if region.is_pure
            nodes = build_join_nodes(region.sources, stats)
            perm = plan_join_order(nodes)
            append!(plan, region.sources[perm])
        else
            # Non-pure: topological order respecting effect deps (conservative: keep original)
            append!(plan, region.sources)
        end
        i < length(regions) && push!(barriers, EffectBarrier(regions[i].effects))
    end

    (plan, barriers)
end

export EffectKind, EFF_PURE, EFF_READ, EFF_APPEND, EFF_WRITE, EFF_OBSERVE
export effects_commute
export JoinNode, build_join_nodes, build_join_nodes_dynamic
export plan_join_order, plan_join_order_static
export plan_conjunction, plan_conjunction_dynamic
export plan_program, plan_program_dynamic
export plan_report
export PureRegion, EffectBarrier, identify_pure_regions, plan_query
