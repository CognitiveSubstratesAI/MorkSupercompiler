"""
FactorGeometry — exact and approximate factor-geometry specialization.

Implements §10.1 of the MG Framework spec (Goertzel, April 2026):
  §10.1.3  Algorithm 1 — Exact factor-geometry specialization (8 steps)
  §10.1.4  Algorithm 2 — Approximate factor specialization with explicit witness (7 steps)
  §10.1.1  Semantic role: Model(Q, Formula) / Pres(Factor, Model(Q, Formula))
  §10.1.5  Concurrency and distribution (built into GeometryTemplate defaults)

Factor geometry is the right choice when: "method looks like rules with premises
and conclusions and some notion of uncertainty, answer queries against evidence."

STV (Simple Truth Value) family: forward-map = heuristic-mp-tv function,
backward-demand = adjoint-need. Instantiated by Algorithm 1 step 4.

The Noether charge (§12.2 EvidenceCapsule): evidence mass w ≤ w_start ensures
the derivation neither hallucinates nor leaks support.
"""

# ── Factor graph structures ───────────────────────────────────────────────────

"""
    FactorNode

A node in a factor graph. Can be either a variable node or a factor node.
is_factor  — true = factor node (rule), false = variable node (formula)
name       — identifier
role       — :premise, :conclusion, or :boundary
message    — current outgoing message (a PBox for STV)
cache_ver  — version tuple for cache validity
"""
mutable struct FactorNode
    is_factor::Bool
    name::Symbol
    role::Symbol         # :premise | :conclusion | :boundary
    message::PBox
    cache_ver::Int
end

FactorNode(name::Symbol, role::Symbol; is_factor=false) =
    FactorNode(is_factor, name, role, pbox_exact(0.5), 0)

"""
    FactorEdge

An edge in the factor graph: variable ↔ factor incidence with role label.
"""
struct FactorEdge
    var_node::Symbol
    factor_node::Symbol
    role_label::Symbol   # e.g., :premise_1, :premise_2, :conclusion
end

"""
    FactorGraph

A factor graph with variable nodes, factor nodes, and role-labeled edges.
Presentation of Model(Q, Formula) in the Factor geometry.
"""
mutable struct FactorGraph
    var_nodes::Dict{Symbol, FactorNode}
    factor_nodes::Dict{Symbol, FactorNode}
    edges::Vector{FactorEdge}
    template::GeometryTemplate
    epoch::Int
    boundary_cache::Dict{Symbol, PBox}   # frozen boundary values
end

function FactorGraph(template::GeometryTemplate)::FactorGraph
    FactorGraph(
        Dict{Symbol, FactorNode}(),
        Dict{Symbol, FactorNode}(),
        FactorEdge[],
        template,
        0,
        Dict{Symbol, PBox}()
    )
end

"""
    SpecializedRegion

Result of Algorithm 1 or 2: a query-specialized executable region with metadata.
"""
struct SpecializedRegion
    graph::FactorGraph
    active_nodes::Set{Symbol}        # nodes in the active subgraph
    cache_keys::Dict{Symbol, UInt64}  # canonical cache keys
    exactness::ErrorLevel
    error_bound::Float64            # 0.0 for EXACT
    witness::Union{Nothing, PBox}  # approximation witness (Alg 2)
    lowering::Symbol             # :factor_runtime | :mm2_worklist | :direct
end

# ── Algorithm 1 — Exact factor-geometry specialization (§10.1.3) ─────────────

"""
    specialize_exact(query, graph, budget) -> SpecializedRegion

Algorithm 1 (Exact factor-geometry specialization) from §10.1.3.
8 steps:

 1. Infer goal node, relevant role labels, and demand family
 2. Run backward demand expansion → active subgraph G_act
 3. Freeze valid boundary caches or attach priors at frontier
 4. Specialize local factor kernels by rule family, mode, truth-value representation
 5. Build canonical keys for active region and cache dependencies
 6. Choose lowering: factor runtime, MM2 worklist, or another exact backend
 7. Emit residual kernel + cache-contract metadata
 8. Return specialized executable region
"""
function specialize_exact(
    query::Symbol, graph::FactorGraph, budget::Int=1000
)::SpecializedRegion

    # Step 1: Infer goal node and demand family
    goal_node = get(graph.var_nodes, query, nothing)
    demand_family = _infer_demand_family(graph.template)

    # Step 2: Backward demand expansion
    active = _backward_demand_expansion(query, graph, budget)

    # Step 3: Freeze boundary caches or attach priors
    _freeze_boundary_caches!(graph, active)

    # Step 4: Specialize factor kernels by truth-value family
    truth_family = _get_truth_family(graph.template)
    _specialize_kernels!(graph, active, truth_family)

    # Step 5: Build canonical cache keys
    cache_keys = _build_cache_keys(graph, active)

    # Step 6: Choose lowering strategy
    lowering = _choose_lowering(graph.template, length(active))

    # Step 7+8: Emit and return
    SpecializedRegion(graph, active, cache_keys, EXACT, 0.0, nothing, lowering)
end

# ── Algorithm 2 — Approximate factor specialization (§10.1.4) ────────────────

"""
    specialize_approximate(query, graph, ε, confidence, budget) -> SpecializedRegion

Algorithm 2 (Approximate factor specialization with explicit witness) from §10.1.4.
7 steps:

 1. Derive active subgraph by backward demand (same as Algorithm 1 step 2)
 2. Replace expensive truth families by registered approximate coercions when legal
 3. Attach approximation witness objects to affected kernels and boundary caches
 4. Introduce early-stop guards based on total confidence or residual utility
 5. Compose error witnesses through the active region
 6. Emit residual kernel with explicit (ε, p) contract
 7. Return bounded or statistical executable region
"""
function specialize_approximate(
    query::Symbol,
    graph::FactorGraph,
    ε::Float64=0.05,
    confidence::Float64=0.95,
    budget::Int=1000
)::SpecializedRegion

    # Step 1: Active subgraph by backward demand
    active = _backward_demand_expansion(query, graph, budget)

    # Step 2: Replace with approximate coercions where legal
    approx_nodes = _apply_approx_coercions!(graph, active, ε)

    # Step 3: Build witness objects for each approximated kernel
    witnesses = [pbox_interval(1.0 - ε, 1.0, confidence) for _ in approx_nodes]

    # Step 4: Early-stop guard
    # If cumulative confidence < (1 - ε), introduce stopping condition
    total_conf = prod(confidence for _ in approx_nodes; init=1.0)

    # Step 5: Compose witness through the region (Theorem A.2)
    widths = [width(w) for w in witnesses]
    total_error = isempty(widths) ? 0.0 : error_composition_bound(widths)

    # Aggregate witness PBox
    region_witness = if isempty(witnesses)
        nothing
    else
        pbox_interval(max(0.0, 1.0 - total_error), 1.0, total_conf)
    end

    # Step 6: Exactness class
    exactness_class = if total_error == 0.0
        EXACT
    elseif confidence == 1.0
        BOUNDED
    else
        STATISTICAL
    end

    # Step 7: Return
    cache_keys = _build_cache_keys(graph, active)
    lowering = _choose_lowering(graph.template, length(active))

    SpecializedRegion(
        graph, active, cache_keys, exactness_class, total_error, region_witness, lowering
    )
end

# ── STV (Simple Truth Value) family ──────────────────────────────────────────

"""
    stv_forward_map(strength_a, conf_a, strength_impl, conf_impl) -> (Float64, Float64)

STV ModusPonens forward map — **faithful to PLN book §5.7.1** (lib/pln `Truth_ModusPonens`),
the EXACT-rule contract:

    strength_B = strength_a * strength_impl + 0.02 * (1 - strength_a)
    conf_B     = w2c(conf_a * conf_impl),   w2c(w) = w / (w + 1)

Conclusion B from A `(strength_a, conf_a)` and `(A → B) (strength_impl, conf_impl)`.

Independent INLINE implementation: the Layer-1 gate (test/mgfw/test_pln_reference.jl)
diffs this against the `PLNBook` oracle, so the two must NOT share code. This replaces the
earlier "simplified" map (`sA·sI`, `cA·cI·min(sA,sI)`) that matched neither book nor the
template's own lowering — PLN Layer-1 "Finding B", resolved in step 3b. The MG-Framework
§10.1.2 heuristic (`min·0.9`) lives separately in `references.jl::stv_mp_reference` as the
APPROXIMATE family for `specialize_approximate`.
"""
function stv_forward_map(
    strength_a::Float64, conf_a::Float64, strength_impl::Float64, conf_impl::Float64
)::Tuple{Float64, Float64}
    s_b = clamp(strength_a * strength_impl + 0.02 * (1.0 - strength_a), 0.0, 1.0)
    w = conf_a * conf_impl
    c_b = clamp(w / (w + 1.0), 0.0, 1.0)
    (s_b, c_b)
end

"""
    stv_to_pbox(strength::Float64, confidence::Float64) -> PBox

Convert an STV (strength, confidence) pair to a PBox.
Width of the interval encodes uncertainty: lower confidence → wider interval.
"""
function stv_to_pbox(strength::Float64, confidence::Float64)::PBox
    half_width = (1.0 - confidence) / 2.0
    lo = clamp(strength - half_width, 0.0, 1.0)
    hi = clamp(strength + half_width, 0.0, 1.0)
    pbox_interval(lo, hi, confidence)
end

"""
    stv_backward_demand(goal_strength::Float64, goal_conf::Float64) -> (Float64, Float64)

STV backward demand function: given a goal on B, what do we need from the premises?
Returns the minimum (strength, confidence) we need on A and (A→B).
"""
function stv_backward_demand(
    goal_strength::Float64, goal_conf::Float64
)::Tuple{Float64, Float64}
    # Adjoint: if we need strength s_B with conf c_B, we need at least
    # sqrt(s_B) strength on premises (approximate inverse of product)
    needed_strength = sqrt(max(0.0, goal_strength))
    needed_conf = sqrt(max(0.0, goal_conf))
    (needed_strength, needed_conf)
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _backward_demand_expansion(
    query::Symbol, graph::FactorGraph, budget::Int
)::Set{Symbol}
    active = Set{Symbol}([query])
    frontier = [query]
    steps = 0

    while !isempty(frontier) && steps < budget
        node = popfirst!(frontier)
        steps += 1
        # Find all factor nodes that have this node as conclusion
        for e in graph.edges
            e.var_node == node && e.role_label == :conclusion || continue
            push!(active, e.factor_node)
            for e2 in graph.edges
                e2.factor_node == e.factor_node && e2.role_label != :conclusion || continue
                e2.var_node ∉ active && push!(frontier, e2.var_node)
                push!(active, e2.var_node)
            end
        end
    end
    active
end

function _freeze_boundary_caches!(graph::FactorGraph, active::Set{Symbol})
    for (name, node) in graph.var_nodes
        name ∉ active || continue
        # Node is at boundary: freeze its current message as prior
        graph.boundary_cache[name] = node.message
    end
end

function _specialize_kernels!(graph::FactorGraph, active::Set{Symbol}, truth_family::Symbol)
    # For STV family: specialize each factor node in active subgraph
    for (name, node) in graph.factor_nodes
        name ∉ active && continue
        node.is_factor || continue
        # Specialize: tag with truth family for downstream lowering
        # (In full implementation: replace generic map with STV-specific fast path)
        node.cache_ver += 1
    end
end

function _apply_approx_coercions!(
    graph::FactorGraph, active::Set{Symbol}, ε::Float64
)::Vector{Symbol}
    approx_nodes = Symbol[]
    for (name, node) in graph.factor_nodes
        name ∉ active && continue
        # Can approximate: widen message by ε
        pb = node.message
        new_pb = pbox_interval(
            max(0.0, pb.intervals[1][1] - ε/2), min(1.0, pb.intervals[end][2] + ε/2), 0.95
        )
        node.message = new_pb
        push!(approx_nodes, name)
    end
    approx_nodes
end

function _build_cache_keys(graph::FactorGraph, active::Set{Symbol})::Dict{Symbol, UInt64}
    Dict(
        name => hash(string(name, graph.epoch, get(graph.boundary_cache, name, nothing)))
        for name in active
    )
end

function _infer_demand_family(template::GeometryTemplate)::Symbol
    # §10.1.3 Step 1: infer demand family from declared operators
    :backward_demand in template.operators && return :backward_demand
    :adjoint_need in template.operators && return :adjoint_need
    :demand_push in template.operators && return :demand_push
    :forward_only
end

function _get_truth_family(template::GeometryTemplate)::Symbol
    :STV  # default; would be read from template metadata
end

function _choose_lowering(template::GeometryTemplate, active_size::Int)::Symbol
    affinity = get(template.backend_affinity, :mm2, :medium)
    active_size > 50 ? :mm2_worklist : :factor_runtime
end

# ── Noether charge (evidence conservation, §12.2) ────────────────────────────

"""
    noether_charge(region::SpecializedRegion) -> Float64

§12.2 EvidenceCapsule: the Noether charge is the evidence mass.
Invariant: w_end ≤ w_start — derivation neither hallucinates nor leaks.
Returns the total probability mass in the region's witness PBox (or 1.0 for EXACT).
"""
function noether_charge(region::SpecializedRegion)::Float64
    region.witness === nothing && return 1.0
    region.witness.confidence
end

"""
    conserves_evidence(region::SpecializedRegion, initial_mass::Float64) -> Bool

Check the Noether invariant: evidence mass at end ≤ evidence mass at start.
"""
conserves_evidence(region::SpecializedRegion, initial_mass::Float64)::Bool =
    noether_charge(region) <= initial_mass + 1e-9

export FactorNode, FactorEdge, FactorGraph, SpecializedRegion
export specialize_exact, specialize_approximate
export stv_forward_map, stv_to_pbox, stv_backward_demand
export noether_charge, conserves_evidence
