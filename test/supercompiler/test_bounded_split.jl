using Test
using MorkSupercompiler

@testset "BoundedSplit — non-splittable node passes through" begin
    g = MCoreGraph()
    id_l = add_lit!(g, Lit(42))
    stats = MORKStatistics()
    sr = bounded_split(g, id_l, Env(), stats)
    @test length(sr.branches) == 1
    @test sr.branches[1].id == id_l
    @test sr.total_prob ≈ 1.0
    @test !sr.catchall_added
end

@testset "BoundedSplit — Choice: selects top branches by probability" begin
    g = MCoreGraph()
    ids = [add_sym!(g, Sym(Symbol("alt$i"))) for i in 1:5]
    alts = ChoiceAlt.(ids)
    id_c = add_choice!(g, Choice(alts))

    stats = MORKStatistics(
        Dict("alt1"=>100, "alt2"=>200, "alt3"=>50, "alt4"=>10, "alt5"=>5), 365
    )

    sr = bounded_split(g, id_c, Env(), stats; budget=3)
    # Should have selected up to 3 branches + maybe a catchall
    n_non_catchall = count(b -> !b.is_catchall, sr.branches)
    @test n_non_catchall <= 3
    # Total prob of selected branches should be in [0, 1]
    @test 0.0 <= sr.total_prob <= 1.0
    # If not all branches covered, catchall added for soundness
    if sr.total_prob < 1.0
        @test sr.catchall_added
    end
end

@testset "BoundedSplit — budget=1 selects exactly 1 main branch" begin
    g = MCoreGraph()
    ids = [add_sym!(g, Sym(Symbol("x$i"))) for i in 1:4]
    alts = ChoiceAlt.(ids)
    id_c = add_choice!(g, Choice(alts))
    # Use realistic stats so prob per guard < 1.0
    stats = MORKStatistics(Dict("x1"=>25, "x2"=>25, "x3"=>25, "x4"=>25), 100)

    sr = bounded_split(g, id_c, Env(), stats; budget=1)
    non_ca = count(b -> !b.is_catchall, sr.branches)
    @test non_ca == 1           # exactly 1 real branch selected
    @test length(sr.branches) >= 1
end

@testset "BoundedSplit — SPLIT_PROB_THRESHOLD constant is 0.95" begin
    @test SPLIT_PROB_THRESHOLD == 0.95
    @test SPLIT_DEFAULT_BUDGET == 16
end
