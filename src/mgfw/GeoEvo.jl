"""
GeoEvo — GEO-EVO geodesic evolutionary control (Geo-Evo.pdf §3–6).

Couples the MORK-MOSES *forward* engine (demes/EDA — `TrieDAGGeometry.jl`) to the PLN *backward*
demand field (`PLNDemand.jl`) by the §3.4 two-ends-of-the-path co-adaptation, ranked by the §3.5
control law. Fills the operators declared by `build_geodesic_bgc_composite`
(`:scheduler_pop`/`:guidance_update`/`:evidence_update`/`:splice_check`).

╔═ CONTRACT — the anti-hardcode invariant (mirrors the WorldModel Space-as-data lesson) ═════════╗
║ This file holds ZERO rules, subgoals, or control parameters as Julia constants. ALL structure  ║
║ is read from MORK atoms/capsules at runtime, so the system can tune and self-evolve them:      ║
║   (geo-param <name> <value>)                       control-law params  (λ μ γ τ α1 α2 α3 budget)║
║   (subgoal <id> ...)                               backward-waypoint library (SubRep emits)     ║
║   (factor <f> <rule> (premises ...) (conclusion ...))   rule/factor set (the lib/pln pattern)   ║
║   program / deme root atoms                        the evolving population                      ║
║ GeoEvo provides ONLY grounded kernels (comp/gap/sinkhorn/score/effort) + the scheduler loop.    ║
║ Mutating an atom in the space changes behaviour with no code edit — asserted by test_geoevo.jl. ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝

Staging (honest, no stubs-called-done):
  v0 (this commit): data-read layer + pure grounded kernels + the data-driven invariant test.
  v1 (next): backward g via `compute_demand_field`, forward via `evolve_demes!`, the 4 composite
             operators, bidirectional two-ends co-adaptation (DecompEffort+η·minGap AND Ω_align),
             splicing + deme bandit. Depth-limits documented as each lands.
"""

# Space ops are MORK-qualified to avoid clashing with MorkSupercompiler's own `new_space`/`Space`.

# ── structure read FROM the space (NOT hardcoded) ─────────────────────────────────────────────

"Cheap line-split of the space dump (the geo capsules are flat s-exprs)."
function _geo_dump_lines(s::MORK.Space)::Vector{SubString{String}}
    filter!(l -> !isempty(l), strip.(split(MORK.space_dump_all_sexpr(s), "\n"; keepempty=false)))
end

"""
    geo_params(space) -> Dict{Symbol,Float64}

Read `(geo-param <name> <value>)` capsules. The returned defaults are FALLBACKS for absent keys
only (documented §3.4/§3.5 values); the operative numbers live in the space and are tunable at
runtime. Nothing here is a hardcoded policy — only safe fallbacks so an empty space still runs.
"""
function geo_params(s::MORK.Space)::Dict{Symbol,Float64}
    p = Dict{Symbol,Float64}(:lambda => 1.0, :mu => 1.0, :gamma => 0.5, :tau => 1.0,
        :alpha1 => 1.0, :alpha2 => 1.0, :alpha3 => 1.0, :budget => 1000.0)
    for a in _geo_dump_lines(s)
        m = match(r"^\(geo-param\s+([A-Za-z_][\w-]*)\s+(-?[\d.][\d.eE+-]*)\s*\)$", a)
        m === nothing && continue
        p[Symbol(m.captures[1])] = parse(Float64, m.captures[2])
    end
    return p
end

"Read `(subgoal <id> ...)` atoms → the backward-waypoint library (ids, in dump order)."
function geo_subgoals(s::MORK.Space)::Vector{String}
    out = String[]
    for a in _geo_dump_lines(s)
        m = match(r"^\(subgoal\s+(\S+?)[\s)]", a)
        m === nothing || push!(out, String(m.captures[1]))
    end
    return out
end

# ── grounded kernels (the fixed engine — pure functions over data) ────────────────────────────

geo_sigma(x::Float64) = 1.0 / (1.0 + exp(-x))

"§3.4  Comp = σ(α1·Cover + α2·Reach − α3·Gap).  All weights read from `p` (the space)."
geo_comp(cover::Float64, reach::Float64, gap::Float64, p::Dict{Symbol,Float64})::Float64 =
    geo_sigma(p[:alpha1] * cover + p[:alpha2] * reach - p[:alpha3] * gap)

"§3.5  Score(U) = Δ(φ+ψ) − λ·ΔCost + μ·ΔAlign.  λ,μ read from `p`."
geo_score(dphi::Float64, dpsi::Float64, dcost::Float64, dalign::Float64, p::Dict{Symbol,Float64})::Float64 =
    (dphi + dpsi) - p[:lambda] * dcost + p[:mu] * dalign

"""
    geo_sinkhorn(C, p; iters) -> Matrix

§3.4/§5.1.4 soft pairing π_{m,k}: Sinkhorn balancing of exp(τ·C) toward uniform marginals
(entropic-OT style). Returns rows that sum to 1, so π_{m,·} is a distribution over analytical
manifolds for each deme m. τ read from `p`. Pure & deterministic.
"""
function geo_sinkhorn(C::Matrix{Float64}, p::Dict{Symbol,Float64}; iters::Int=20)::Matrix{Float64}
    K = exp.(p[:tau] .* C)
    for _ in 1:iters
        rs = sum(K, dims=2); rs[rs .== 0] .= 1.0; K = K ./ rs
        cs = sum(K, dims=1); cs[cs .== 0] .= 1.0; K = K ./ cs
    end
    rs = sum(K, dims=2); rs[rs .== 0] .= 1.0
    return K ./ rs
end

"§3.1/§3.10  Effective fitness F_eff = F − γ·W (weakness regularizer).  γ read from `p`."
geo_feff(fitness::Float64, weakness::Float64, p::Dict{Symbol,Float64})::Float64 =
    fitness - p[:gamma] * weakness

"""
    geo_effort(neg_log_edit_lik, kl_to_prior, diversity) -> Float64

§5.1.1 per-step cost (the ΔCost term): −log edit-likelihood + KL-to-prior + diversity smoothing.
Caller supplies the three measured components; this is the fixed combiner.
"""
geo_effort(neg_log_edit_lik::Float64, kl_to_prior::Float64, diversity::Float64)::Float64 =
    neg_log_edit_lik + kl_to_prior + diversity

# ── backward potential g: the PLN demand field over factors read FROM the space ───────────────
# The rule/factor set is DATA, stored as FLAT MORK atoms (atomic + prefix-queryable — the
# MORK-native form of lib/pln's `(factor f rule (premises …) (conclusion …))` schema):
#   (factor <f> <rule>)          — a factor with its PLN rule tag (hmp/deduction/conjunction/…)
#   (premise <f> <var> <role>)   — premise edge; role ∈ {premise_1, premise_2, …} (positional)
#   (conclusion <f> <var>)       — conclusion edge
#   (stv <var> <s> <c>)          — optional premise STV (default 0.5/0.5)
# Nothing here is hardcoded: change a factor atom → the backward field changes (test asserts it).

_geo_ensure_var!(g::FactorGraph, name::Symbol) =
    (haskey(g.var_nodes, name) || (g.var_nodes[name] = FactorNode(name, :premise; is_factor=false)); name)

"""
    geo_factor_graph(space) -> FactorGraph

Build a `FactorGraph` from the factor/premise/conclusion/stv atoms in `space`. Pure read — the
inference rules live in the space as data, never as Julia constants.
"""
function geo_factor_graph(s::MORK.Space)::FactorGraph
    g = FactorGraph(TEMPLATE_PLN_STV_MP)
    stvs = Dict{Symbol, Tuple{Float64, Float64}}()
    for a in _geo_dump_lines(s)
        m = match(r"^\(stv\s+(\S+)\s+(-?[\d.][\d.eE+-]*)\s+(-?[\d.][\d.eE+-]*)\s*\)$", a)
        m === nothing && continue
        stvs[Symbol(m.captures[1])] = (parse(Float64, m.captures[2]), parse(Float64, m.captures[3]))
    end
    for a in _geo_dump_lines(s)
        if (m = match(r"^\(factor\s+(\S+)\s+(\S+?)\s*\)$", a)) !== nothing
            f = Symbol(m.captures[1])
            g.factor_nodes[f] = FactorNode(f, :boundary; is_factor=true, rule=Symbol(m.captures[2]))
        elseif (m = match(r"^\(premise\s+(\S+)\s+(\S+)\s+(\S+?)\s*\)$", a)) !== nothing
            f = Symbol(m.captures[1]); v = _geo_ensure_var!(g, Symbol(m.captures[2]))
            push!(g.edges, FactorEdge(v, f, Symbol(m.captures[3])))
        elseif (m = match(r"^\(conclusion\s+(\S+)\s+(\S+?)\s*\)$", a)) !== nothing
            f = Symbol(m.captures[1]); v = _geo_ensure_var!(g, Symbol(m.captures[2]))
            push!(g.edges, FactorEdge(v, f, :conclusion))
        end
    end
    for (name, node) in g.var_nodes
        sc = get(stvs, name, (0.5, 0.5))
        node.message = stv_to_pbox(sc[1], sc[2])
    end
    return g
end

"""
    geo_backward_g(space, goal; budget=1000) -> Dict{Symbol,Float64}

§3.3.1 backward potential `g`: the PLN demand field seeded at `goal`, propagated conclusion→premise
over the factors read from `space` (via `compute_demand_field`). Higher demand = a subgoal/premise
the goal more needs evidence for. The geodesic backward signal that v1's two-ends coupling consumes.
"""
function geo_backward_g(s::MORK.Space, goal::Symbol; budget::Int=1000)::Dict{Symbol, Float64}
    _, dem = compute_demand_field(goal, geo_factor_graph(s), budget)
    return dem
end
