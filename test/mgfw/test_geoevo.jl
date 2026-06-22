using Test
using MorkSupercompiler
import MORK

@testset "GeoEvo v0 — data-driven (no-hardcode) contract" begin
    # Structure lives in the space; the engine reads it. Two spaces with different param atoms
    # must yield different reads — proving the values are SOURCED FROM DATA, not hardcoded.
    s = MORK.new_space()
    MORK.space_add_all_sexpr!(s, join([
        "(geo-param lambda 0.25)",
        "(geo-param mu 2.0)",
        "(geo-param gamma 0.1)",
        "(subgoal reach-door)",
        "(subgoal pick-key)",
    ], "\n"))

    p = geo_params(s)
    @test p[:lambda] ≈ 0.25
    @test p[:mu] ≈ 2.0
    @test p[:gamma] ≈ 0.1
    @test p[:alpha1] ≈ 1.0          # absent key → documented fallback (not a hardcoded policy)

    sg = geo_subgoals(s)
    @test "reach-door" in sg
    @test "pick-key" in sg
    @test length(sg) == 2

    # THE INVARIANT: a different space → different param, with NO code change.
    s2 = MORK.new_space()
    MORK.space_add_all_sexpr!(s2, "(geo-param lambda 0.9)")
    @test geo_params(s2)[:lambda] ≈ 0.9
    @test geo_params(s)[:lambda] ≈ 0.25    # original space unaffected

    # empty space still runs on documented fallbacks
    @test geo_params(MORK.new_space())[:lambda] ≈ 1.0
end

@testset "GeoEvo v0 — grounded kernels (the fixed engine)" begin
    p = Dict{Symbol,Float64}(:lambda => 1.0, :mu => 1.0, :gamma => 0.5, :tau => 1.0,
        :alpha1 => 1.0, :alpha2 => 1.0, :alpha3 => 1.0, :budget => 1000.0)

    # Comp ∈ (0,1); ↑ in cover/reach, ↓ in gap (§3.4)
    @test 0.0 < geo_comp(1.0, 1.0, 0.0, p) < 1.0
    @test geo_comp(2.0, 0.0, 0.0, p) > geo_comp(0.0, 0.0, 0.0, p)
    @test geo_comp(0.0, 0.0, 2.0, p) < geo_comp(0.0, 0.0, 0.0, p)

    # Score = Δ(φ+ψ) − λΔcost + μΔalign (§3.5)
    @test geo_score(1.0, 0.5, 0.0, 0.0, p) ≈ 1.5
    @test geo_score(0.0, 0.0, 1.0, 0.0, p) ≈ -1.0
    @test geo_score(0.0, 0.0, 0.0, 1.0, p) ≈ 1.0

    # F_eff = F − γW (§3.1/§3.10)
    @test geo_feff(1.0, 1.0, p) ≈ 0.5

    # Sinkhorn pairing: rows are distributions; higher Comp ⇒ more mass (§3.4/§5.1.4)
    C = [1.0 0.0; 0.0 1.0; 0.5 0.5]
    P = geo_sinkhorn(C, p)
    @test all(P .>= 0.0)
    @test all(abs.(sum(P, dims=2) .- 1.0) .< 1e-9)
    @test P[1, 1] > P[1, 2]
end

@testset "GeoEvo v1a — backward g (PLN demand) over factors read from the space" begin
    # The rule set is DATA: an hmp factor concluding B from premises A, AB — stored as atoms.
    s = MORK.new_space()
    MORK.space_add_all_sexpr!(s, join([
        "(factor fmp hmp)",
        "(conclusion fmp B)",
        "(premise fmp A premise_1)",
        "(premise fmp AB premise_2)",
        "(stv A 0.8 0.9)",
        "(stv AB 0.7 0.85)",
    ], "\n"))

    g = geo_backward_g(s, :B)
    @test g[:B] ≈ 1.0                       # demand seeded at the goal
    @test haskey(g, :A) && g[:A] > 0.0      # propagated conclusion→premise
    @test haskey(g, :AB) && g[:AB] > 0.0

    # DATA-DRIVEN: a space with NO factor for B → only the seed, nothing to propagate to.
    g2 = geo_backward_g(MORK.new_space(), :B)
    @test g2[:B] ≈ 1.0
    @test !haskey(g2, :A)

    # the factor graph reflects the atoms (factor rule-tagged, conclusion edge present)
    fg = geo_factor_graph(s)
    @test haskey(fg.factor_nodes, :fmp)
    @test fg.factor_nodes[:fmp].rule === :hmp
    @test any(e -> e.role_label === :conclusion && e.var_node === :B, fg.edges)

    # §2.4: an unknown premise (no stv atom) gets the NEUTRAL prior (½,0) — confidence 0, need=1.
    # NOT (0.5,0.5); the spec rejects high-confidence/(0,0) defaults for ignorance.
    su = MORK.new_space()
    MORK.space_add_all_sexpr!(su, join([
        "(factor fu negation)", "(conclusion fu Q)", "(premise fu Punk premise_1)",
    ], "\n"))
    fgu = geo_factor_graph(su)
    @test fgu.var_nodes[:Punk].message.confidence ≈ 0.0   # ignorance = zero confidence (§2.4)
end

@testset "GeoEvo v1b — two-ends coupling (Comp/Gap/π) over demes × subgoal motifs (data)" begin
    p = geo_params(MORK.new_space())   # documented fallback params

    # subgoal motifs are DATA: (subgoal-motif id op) atoms
    sm = MORK.new_space()
    MORK.space_add_all_sexpr!(sm, join([
        "(subgoal-motif reach-door and)",
        "(subgoal-motif reach-door move)",
        "(subgoal-motif pick-key grasp)",
    ], "\n"))
    motifs = geo_subgoal_motifs(sm)
    @test motifs["reach-door"] == Set([:and, :move])
    @test motifs["pick-key"] == Set([:grasp])

    # operator-set kernels (§3.4)
    @test geo_cover(Set([:a, :b]), Set([:a, :b, :c])) ≈ 2 / 3
    @test geo_gap(Set([:a]), Set([:a])) ≈ 0.0
    @test geo_gap(Set([:a]), Set([:b])) ≈ 1.0

    # two demes with distinct operator profiles (set via eda_model — what evolve_demes! would learn)
    d1 = Deme(1); d1.eda_model[:and] = 0.5; d1.eda_model[:move] = 0.5   # ~ reach-door
    d2 = Deme(2); d2.eda_model[:grasp] = 1.0                            # = pick-key
    sgids, C, P, omega, sgap = geo_pairing([d1, d2], motifs, p)

    @test sgids == ["pick-key", "reach-door"]     # sorted
    rd = findfirst(==("reach-door"), sgids); pk = findfirst(==("pick-key"), sgids)
    @test C[1, rd] > C[1, pk]                      # deme1 covers reach-door better
    @test C[2, pk] > C[2, rd]                      # deme2 covers pick-key better
    @test all(abs.(sum(P, dims=2) .- 1.0) .< 1e-9)  # π rows are distributions
    @test P[1, rd] > P[1, pk]                      # pairing follows Comp
    @test omega[2] ≈ 0.0                           # deme2 exactly matches a motif ⇒ min Gap = 0
    @test sgap[pk] ≈ 0.0                           # pick-key has a deme that matches it exactly

    # §9.4 weakness = unique DAG node count
    st = DAGStore()
    leaf = dag_intern!(st, :x)
    root = dag_intern!(st, :f, UInt64[leaf, leaf])   # f over a shared leaf ⇒ 2 unique nodes
    @test geo_weakness(st, root) == 2

    # DATA-DRIVEN: no subgoal motifs in the space → no coupling
    sgids0, _, _, _, _ = geo_pairing([d1], Dict{String, Set{Symbol}}(), p)
    @test isempty(sgids0)
end

@testset "GeoEvo v1c — scheduler step (forward + coupling + splice + bandit)" begin
    p = geo_params(MORK.new_space())

    # one space carries BOTH the backward factors and the subgoal motifs — all data
    s = MORK.new_space()
    MORK.space_add_all_sexpr!(s, join([
        "(factor fmp hmp)", "(conclusion fmp G)",
        "(premise fmp A premise_1)", "(premise fmp AB premise_2)",
        "(stv A 0.8 0.9)", "(stv AB 0.7 0.85)",
        "(subgoal-motif G and)", "(subgoal-motif G move)",
    ], "\n"))

    d = Deme(1); d.eda_model[:and] = 0.5; d.eda_model[:move] = 0.5
    fit(store, id) = 0.5
    res = geo_step!([d], s, :G, p; fitness_fn=fit)

    @test "G" in res.subgoals
    @test haskey(res.backward, :G) && res.backward[:G] ≈ 1.0        # demand seeded at goal G
    @test length(res.allocation) == 1 && res.allocation[1] ≈ 1.0    # single deme → all compute
    @test !isempty(res.splices) && res.splices[1].subgoal == "G"
    @test res.generation ≥ 1                                        # forward round advanced

    # forward f proxy
    df = Deme(2); df.fitnesses[UInt64(1)] = 0.9
    @test geo_forward_f(df) ≈ 0.9
    @test geo_forward_f(Deme(3)) ≈ geo_reach(Set{Symbol}())         # empty deme → reach proxy

    # bandit: higher score-trend ⇒ more compute, weights normalized
    w = geo_bandit([2.0, 0.0], [0.0, 0.0], p)
    @test w[1] > w[2]
    @test sum(w) ≈ 1.0
end
