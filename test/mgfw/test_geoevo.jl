using Test
using MorkSupercompiler
import MORK
using Random: MersenneTwister

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

@testset "GeoEvo (a) — closed forward loop: steered geo_step! drives Ω_align down" begin
    p = geo_params(MORK.new_space())
    s = MORK.new_space()
    MORK.space_add_all_sexpr!(s, "(subgoal-motif G and)\n(subgoal-motif G move)")
    motif = Set([:and, :move])
    fit(store, id) = (haskey(store.nodes, id) && store.nodes[id].head in motif) ? 1.0 : 0.0

    # EDA-guided sampler CONSUMES eda_model (acquires the biased operator)
    d3 = Deme(3); geo_align_bias!(d3, Set([:grab]))
    geo_eda_sample!(d3, 5; rng=MersenneTwister(1))
    @test :grab in Set(n.head for n in values(d3.store.nodes))

    # STEERED: deme starts far from the subgoal (only :x); the coupling should pull it in.
    d = Deme(1); d.eda_model[:x] = 1.0
    _, _, _, om_init, _ = geo_pairing([d], geo_subgoal_motifs(s), p)   # initial Ω_align (≈1)
    rng = MersenneTwister(7)
    omegas = Float64[om_init[1]]
    for _ in 1:8
        push!(omegas, geo_step!([d], s, :G, p; fitness_fn=fit, steer=true, rng=rng).omega_align[1])
    end
    @test omegas[end] < omegas[1]                       # the coupling STEERED evolution (Ω_align↓)
    @test omegas[end] ≤ 0.1                             # FULLY CLOSED: Ω_align→0 (EDA-guided round, no random junk)
    @test geo_cover(geo_deme_ops(d), motif) ≈ 1.0       # CONVERGED: deme acquired the FULL subgoal motif

    # CONTROL: unsteered, the random forward variation cannot reach the subgoal ops
    d2 = Deme(2); d2.eda_model[:x] = 1.0
    rng2 = MersenneTwister(7); om2 = 1.0
    for _ in 1:8
        om2 = geo_step!([d2], s, :G, p; fitness_fn=fit, steer=false, rng=rng2).omega_align[1]
    end
    @test om2 ≥ omegas[end]            # steering helps: unsteered stays no closer than steered
end

@testset "GeoEvo §7 — quantale crossover/mutation + building-block recombination" begin
    a = Set([:x, :y]); b = Set([:y, :z])
    @test geo_xover_join(a, b) == Set([:x, :y, :z])                 # ⊕ = union (permissive)
    @test geo_xover_product(a, b) == Set([:y])                      # ⊗ = intersection (common)
    @test geo_xover_mask(Set([:x, :y]), Set([:p, :q]), Set([:x])) == Set([:x, :p, :q])  # x from a, p,q from b
    @test geo_mutate_add(Set([:x]), :y) == Set([:x, :y])           # m ⊕ δ
    @test geo_mutate_restrict(Set([:x, :y, :z]), Set([:x, :y])) == Set([:x, :y])        # m ⊗ δ

    # BUILDING-BLOCK MIXING: two HALF-covering parents → a FULL-covering child (§7's whole point)
    motif = Set([:a, :b, :c, :d])
    parents = [Set([:a, :b]), Set([:c, :d])]   # neither covers the motif alone
    @test geo_cover(parents[1], motif) ≈ 0.5 && geo_cover(parents[2], motif) ≈ 0.5
    kids = geo_recombine(parents, motif; rng=MersenneTwister(2), n=4)
    @test geo_cover(kids[1], motif) ≈ 1.0      # join recombined the building blocks → full coverage
end

@testset "GeoEvo §3.9 — success metrics + §4.1/4.2 guidance capsule persistence" begin
    # §3.9 metrics over a convergent Ω_align trajectory
    omega = [1.0, 0.5, 0.2, 0.0, 0.0]
    pi1 = [0.5 0.5; 0.5 0.5]; pi2 = [0.9 0.1; 0.1 0.9]
    mt = geo_metrics(omega, [pi1, pi2])
    @test mt.coupling_gain ≈ 1.0               # Ω_align converged 1.0 → 0.0
    @test mt.action_length ≈ 1.0               # Σ|ΔΩ| = 0.5+0.3+0.2+0
    @test mt.evenness ≥ 0.0
    @test mt.pi_stability ≈ 1.6                # L1 change of π in the final step

    # §4.1/4.2 guidance capsules persisted to the MORK space + readable back
    p = geo_params(MORK.new_space())
    s = MORK.new_space()
    MORK.space_add_all_sexpr!(s, "(subgoal-motif G and)")
    d = Deme(1); d.eda_model[:and] = 1.0
    res = geo_step!([d], s, :G, p; fitness_fn=(st, id) -> 1.0)
    nwrote = geo_guidance_capsules!(s, res)
    @test nwrote > 0
    dump = MORK.space_dump_all_sexpr(s)
    @test occursin("geo-guidance", dump)       # state persisted as queryable MORK atoms (not in-memory only)
end

@testset "GeoEvo §8 — factor-graph EDA: n-ary co-occurrence mining + dependency-aware sampling" begin
    # population: a,b co-occur in high-fitness programs; c,d co-occur; the blocks are never mixed
    programs = [Set([:a, :b]), Set([:a, :b]), Set([:c, :d]), Set([:c, :d])]
    fits = [1.0, 1.0, 1.0, 1.0]
    marg, fac = geo_mine_factors(programs, fits)
    @test fac[(:a, :b)] ≈ 2.0                  # §8.2 fitness-weighted co-occurrence
    @test fac[(:c, :d)] ≈ 2.0
    @test !haskey(fac, (:a, :c))               # a,c never co-occur → no factor
    @test marg[:a] ≈ 2.0

    # §8.3/8.4 dependency-aware sampling PRESERVES building blocks — never glues {a,c}
    kids = geo_fg_sample(marg, fac; rng=MersenneTwister(3), n=6, size=2)
    @test !isempty(kids)
    for k in kids
        @test (k ⊆ Set([:a, :b])) || (k ⊆ Set([:c, :d]))   # coherent block, vs independent EDA mixing
    end
end

@testset "GeoEvo §7+§8 in-loop — block evolution accumulates building blocks into a full coverer" begin
    motif = Set([:a, :b, :c])
    fit(s::Set{Symbol}) = geo_cover(s, motif)                  # reward subgoal-motif coverage
    pop = [Set([:a]), Set([:b]), Set([:c]), Set([:x])]          # building blocks SCATTERED across programs
    best0 = maximum(fit, pop)
    @test best0 ≈ 1 / 3                                         # no single program covers >1 op of the motif
    for _ in 1:5
        pop = geo_evolve_blocks!(pop, motif, fit; rng=MersenneTwister(4))
    end
    bestN = maximum(fit, pop)
    @test bestN > best0                                        # §7 recombine + §8 EDA combined the blocks
    @test bestN ≈ 1.0                                          # CONVERGED: a program covers the FULL motif {a,b,c}
end

@testset "GeoEvo §3.7/§4.5 — corridor tracking + scheduler orchestration" begin
    p = geo_params(MORK.new_space())
    # §4.5 side-choice: expand the side whose cost deviates most from c*
    @test geo_side_choice(0.9, 0.5, 0.5) == :forward          # fwd cost 0.9 deviates more from c*=0.5
    @test geo_side_choice(0.5, 0.1, 0.5) == :backward         # bwd deviates more
    # §3.7.1 progress: needs φ+ψ+μ·align>0 AND cost within the band
    @test geo_progress(0.3, 0.2, 0.1, 0.5, p; cmin=0.1, cmax=1.0) == true
    @test geo_progress(-0.5, 0.0, 0.0, 0.5, p; cmin=0.1, cmax=1.0) == false   # degraded → off-corridor
    @test geo_progress(0.3, 0.2, 0.0, 2.0, p; cmin=0.1, cmax=1.0) == false    # cost out of band
    # §3.7.3 diversity: a unique deme is more novel than a duplicated one
    pop = [Set([:a, :b]), Set([:a, :b]), Set([:c, :d])]
    @test geo_diversity_bonus(Set([:c, :d]), pop) > geo_diversity_bonus(Set([:a, :b]), pop)
    # §3.7.4 corridor maintenance: retire low-yield arms, spawn near high-Comp
    retire, spawn = geo_corridor_maintain([0.5, -0.2, 0.8], [0.1, 0.9, 0.3]; retire_below=0.0)
    @test retire == [2]                                       # deme 2 (negative trend) retired
    @test spawn[1] == 2                                       # spawn near deme 2 (highest Comp 0.9)
end
