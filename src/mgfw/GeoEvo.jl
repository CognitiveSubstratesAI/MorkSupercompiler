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
#   (stv <var> <s> <c>)          — optional premise STV; unknown = neutral prior (½,0) per §2.4
#                                  (ignorance is zero-confidence ⇒ need=1−c=1; never (0,0)=falsity)
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
        sc = get(stvs, name, (0.5, 0.0))   # §2.4 neutral prior: ignorance = zero-confidence (need=1), never (0,0)
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

# ── v1b: two-ends-of-the-path co-adaptation (Geo-Evo §3.4) ─────────────────────────────────────
# Couples the forward demes (evolve_demes!) to the backward subgoals. The forward loop is REUSED
# (not rebuilt); v1b adds the coupling. Subgoal motifs are DATA: (subgoal-motif <id> <op>) atoms.
# Deme manifold M_m ≈ its operator profile (eda_model / store heads). Analytical manifold S_k ≈ a
# subgoal's motif operator set. Compatibility/pairing computed over operator-set Cover/Reach/Gap.

"Operator profile of a deme (its `eda_model` favoured ops; else the heads present in its store)."
function geo_deme_ops(d::Deme)::Set{Symbol}
    isempty(d.eda_model) || return Set(keys(d.eda_model))
    return Set(n.head for n in values(d.store.nodes))
end

"§9.4 weakness W = Occam node-count: the number of unique DAG nodes reachable from `root`."
function geo_weakness(store::DAGStore, root::UInt64)::Int
    seen = Set{UInt64}()
    stack = UInt64[root]
    while !isempty(stack)
        id = pop!(stack)
        (id in seen || !haskey(store.nodes, id)) && continue
        push!(seen, id)
        append!(stack, store.nodes[id].children)
    end
    return length(seen)
end

"Read `(subgoal-motif <id> <op>)` atoms → Dict(subgoal-id ⇒ Set of motif operators). Pure data read."
function geo_subgoal_motifs(s::MORK.Space)::Dict{String, Set{Symbol}}
    out = Dict{String, Set{Symbol}}()
    for a in _geo_dump_lines(s)
        m = match(r"^\(subgoal-motif\s+(\S+)\s+(\S+?)\s*\)$", a)
        m === nothing && continue
        push!(get!(out, String(m.captures[1]), Set{Symbol}()), Symbol(m.captures[2]))
    end
    return out
end

"§3.4 Cover: fraction of the subgoal's motif the deme already expresses (population satisfies subgoal)."
geo_cover(deme_ops::Set{Symbol}, motif::Set{Symbol})::Float64 =
    isempty(motif) ? 0.0 : length(intersect(deme_ops, motif)) / length(motif)

"§3.4 Gap: projection mismatch = Jaccard distance between motif space and deme knob-span ∈ [0,1]."
function geo_gap(deme_ops::Set{Symbol}, motif::Set{Symbol})::Float64
    u = length(union(deme_ops, motif))
    return u == 0 ? 0.0 : length(symdiff(deme_ops, motif)) / u
end

"§3.4 Reach (editability within the deme). MVP proxy: variation capacity ~ operator richness.
§5.1 sanctions a simple surrogate here; deferred = short-rollout / learned editability predictor."
geo_reach(deme_ops::Set{Symbol})::Float64 = 1.0 - 1.0 / (1.0 + length(deme_ops))

"""
    geo_pairing(demes, motifs, p) -> (sgids, Comp, π, omega_align, subgoal_deme_gap)

§3.4 two-ends co-adaptation, bidirectional:
  • `Comp[m,k]` = `geo_comp(Cover, Reach, Gap)` over operator sets (demes × subgoals);
  • `π` = `geo_sinkhorn(Comp)` — the soft deme↔subgoal pairing;
  • `omega_align[m]` = minₖ Gap(m,k)  — **analysis-proximal-demes** regularizer (how far each deme is
    from its nearest subgoal; the variation bias the scheduler would reduce);
  • `subgoal_deme_gap[k]` = minₘ Gap(m,k) — **deme-proximal-analysis** signal (prefer decompositions
    near existing demes, Geo-Evo `DecompEffort + η·minGap`).
All weights (α1,α2,α3,τ) read from `p` (data). Pure — produces the coupling STATE; the scheduler
(v1c) consumes it. NOTE/depth-limit: applying `omega_align` as a variation bias is gated on the
forward engine's `_sample_candidates` consuming `eda_model` (it currently samples randomly — a
documented forward-completeness gap, NOT faked here).
"""
function geo_pairing(demes::Vector{Deme}, motifs::Dict{String, Set{Symbol}}, p::Dict{Symbol, Float64})
    sgids = sort(collect(keys(motifs)))
    M, K = length(demes), length(sgids)
    C = zeros(Float64, M, K)
    G = zeros(Float64, M, K)
    for (m, d) in enumerate(demes)
        ops = geo_deme_ops(d); r = geo_reach(ops)
        for (k, sg) in enumerate(sgids)
            motif = motifs[sg]
            C[m, k] = geo_comp(geo_cover(ops, motif), r, geo_gap(ops, motif), p)
            G[m, k] = geo_gap(ops, motif)
        end
    end
    π = (M > 0 && K > 0) ? geo_sinkhorn(C, p) : zeros(Float64, M, K)
    omega_align = [K > 0 ? minimum(@view G[m, :]) : 0.0 for m in 1:M]
    subgoal_deme_gap = [M > 0 ? minimum(@view G[:, k]) : 0.0 for k in 1:K]
    return (sgids, C, π, omega_align, subgoal_deme_gap)
end

# ── v1c: the scheduler — one GEO-EVO step = forward + backward + coupling + splice + bandit ─────
# Geo-Evo §4.5 operator-oriented outline / §5.1 minimal-viable. Fills the composite operators
# (scheduler_pop/guidance_update/evidence_update/splice_check). Reuses the forward engine
# (evolve_demes!); all structure read from the space.

"§3.6/§5.1 forward attainability f(deme): MVP proxy = best fitness in the deme (more fit ⇒ more
attainable), else operator-richness reach. Deferred (§5.1): short rollouts / learned predictor."
function geo_forward_f(d::Deme)::Float64
    isempty(d.fitnesses) && return geo_reach(geo_deme_ops(d))
    return clamp(maximum(values(d.fitnesses)), 0.0, 1.0)
end

"""
    geo_splice_check(demes, motifs, backward_g, p) -> Vector

§5.1.6 splicing: forward↔backward meet-points by high f·g and high Comp. For each (deme m,
subgoal k): splice = f(m)·g(k)·Comp[m,k], where g(k) = backward demand at subgoal k's node.
Returns `(deme, subgoal, splice, gap)` rows sorted by descending splice (lowest action first).
"""
function geo_splice_check(demes::Vector{Deme}, motifs::Dict{String, Set{Symbol}},
        backward_g::Dict{Symbol, Float64}, p::Dict{Symbol, Float64})
    sgids, C, _, _, _ = geo_pairing(demes, motifs, p)
    out = NamedTuple[]
    for (m, d) in enumerate(demes)
        f = geo_forward_f(d); ops = geo_deme_ops(d)
        for (k, sg) in enumerate(sgids)
            g = get(backward_g, Symbol(sg), 0.0)
            push!(out, (deme=m, subgoal=sg, splice=f * g * C[m, k], gap=geo_gap(ops, motifs[sg])))
        end
    end
    sort!(out; by = x -> -x.splice)
    return out
end

"§5.1.5/§3.7 deme bandit: compute allocation ∝ softmax of the sliding-window Score-trend, with a
bonus for high maxₖ Comp. Returns normalized per-deme weights. τ read from `p`."
function geo_bandit(score_trend::Vector{Float64}, comp_max::Vector{Float64}, p::Dict{Symbol, Float64})::Vector{Float64}
    isempty(score_trend) && return Float64[]
    w = exp.(p[:tau] .* score_trend) .* (1.0 .+ comp_max)
    s = sum(w)
    return s == 0 ? fill(1.0 / length(w), length(w)) : w ./ s
end

"""
    geo_step!(demes, space, goal, p; fitness_fn) -> NamedTuple

One GEO-EVO scheduler iteration (§4.5 minimal): advance the forward demes one round
(`evolve_demes!`), recompute the two-ends coupling + backward field, find splices, and the bandit
allocation. Mutates `demes` (forward evolution). All structure (subgoal motifs, factors, params)
read from `space` — nothing hardcoded.
"""
function geo_step!(demes::Vector{Deme}, s::MORK.Space, goal::Symbol, p::Dict{Symbol, Float64};
        fitness_fn::Function)
    evolve_demes!(demes, fitness_fn)                          # forward round (reused engine)
    motifs = geo_subgoal_motifs(s)
    bwd = geo_backward_g(s, goal)
    sgids, C, π, omega, sgap = geo_pairing(demes, motifs, p)
    splices = geo_splice_check(demes, motifs, bwd, p)
    comp_max = isempty(sgids) ? zeros(Float64, length(demes)) : [maximum(@view C[m, :]) for m in 1:length(demes)]
    trend = [geo_forward_f(d) for d in demes]                 # MVP Score-trend proxy = forward f
    alloc = geo_bandit(trend, comp_max, p)
    return (subgoals=sgids, comp=C, pairing=π, omega_align=omega, subgoal_gap=sgap,
        backward=bwd, splices=splices, allocation=alloc,
        generation=isempty(demes) ? 0 : demes[1].generation)
end
