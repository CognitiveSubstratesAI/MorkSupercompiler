using Test
using MorkSupercompiler

@testset "EvoSpecializer — Algorithm 12 GatedSpecialization" begin
    # < 10% → SPEC_VECTORIZED
    d1 = should_specialize(1.0, 100, 100, 500.0)
    @test d1.level == SPEC_VECTORIZED
    @test d1.amortization_ratio < 0.10

    # 10–50% → SPEC_INCREMENTAL
    d2 = should_specialize(1.0, 100, 100, 3000.0)
    @test d2.level == SPEC_INCREMENTAL

    # ≥ 50% → SPEC_GENERIC
    d3 = should_specialize(1.0, 100, 100, 8000.0)
    @test d3.level == SPEC_GENERIC
end

@testset "EvoSpecializer — Algorithm 13 CanReuseFitnessCache" begin
    g = MCoreGraph()
    # Parent: (f (lit 1))
    id_f = add_sym!(g, Sym(:f))
    id_l1 = add_lit!(g, Lit(1))
    id_p = add_app!(g, App(id_f, [id_l1]))
    # Child: (f (lit 2)) — only constant changed
    id_l2 = add_lit!(g, Lit(2))
    id_c = add_app!(g, App(id_f, [id_l2]))

    meta = CacheMetadata(1.0)
    @test can_reuse_cache(g, id_c, id_p, meta)   # 1 constant change ≤ max_changes=3

    # Child: (g (lit 1)) — structural change (different head)
    id_g = add_sym!(g, Sym(:g))
    id_c2 = add_app!(g, App(id_g, [id_l1]))
    @test !can_reuse_cache(g, id_c2, id_p, meta)  # structural change
end

@testset "EvoSpecializer — Algorithm 5 ApproximateFitness (Hoeffding bound)" begin
    pb = approximate_fitness(0.7, 100)
    lo, hi = pb.intervals[1]
    @test lo < 0.7 < hi         # interval straddles sample fitness
    @test pb.probabilities[1] ≈ 0.95
    @test pb.probabilities[2] ≈ 0.05   # 5% tail reserve
    @test pb.confidence ≈ 1.0

    # More samples → narrower interval
    pb2 = approximate_fitness(0.7, 10_000)
    lo2, hi2 = pb2.intervals[1]
    @test (hi2 - lo2) < (hi - lo)   # narrower
end

@testset "EvoSpecializer — Algorithm 7 AllocateEvaluations" begin
    pop = [
        EvolutionaryPBox(1, PBox(0.5, 0.9, 1.0), PBox(0.0, 1.0, 1.0), 0.8, 1),
        EvolutionaryPBox(2, PBox(0.1, 0.3, 1.0), PBox(0.0, 1.0, 1.0), 0.5, 5),
        EvolutionaryPBox(3, PBox(0.8, 0.95, 1.0), PBox(0.0, 1.0, 1.0), 0.9, 2)
    ]
    alloc = allocate_evaluations(pop, 2)
    @test length(alloc) == 2
    # All returned ids should be valid individual ids
    @test all(x -> x[1] in 1:3, alloc)
    # Priorities should be non-negative
    @test all(x -> x[2] >= 0.0, alloc)
end
