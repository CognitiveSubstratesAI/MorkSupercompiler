using Test
using MorkSupercompiler

@testset "drive! — Sym/Lit reach Value in zero steps" begin
    g  = MCoreGraph()
    s_id = add_sym!(g, Sym(:foo))
    l_id = add_lit!(g, Lit(42))

    rs = drive!(g, s_id)
    @test rs.terminated == :value
    @test rs.final_id == s_id
    @test rs.steps == 0

    rl = drive!(g, l_id)
    @test rl.terminated == :value
    @test rl.steps == 0
end

@testset "drive! — Con with all-value fields reaches Value" begin
    g  = MCoreGraph()
    s1 = add_sym!(g, Sym(:hello))
    s2 = add_sym!(g, Sym(:world))
    c  = add_con!(g, Con(:pair, [s1, s2]))

    rs = drive!(g, c)
    @test rs.terminated == :value
    # Stepper.Con rebuilds the Con with stepped fields and returns Value.
    # The returned id may equal c (no change) or a freshly-added Con.
    @test isvalid(rs.final_id)
end

@testset "drive! — FoldTable records canonical keys" begin
    g  = MCoreGraph()
    ft = FoldTable()
    s_id = add_sym!(g, Sym(:x))
    drive!(g, s_id; ft=ft)
    # One record before terminating with Value.
    @test length(ft.entries) >= 1
end

@testset "drive! — fold-back fires when same key recurs" begin
    # Two separate Cons with identical canonical keys (same head + shape)
    # share the same FoldTable. The second call's first key lookup should
    # hit the entry recorded by the first call.
    g  = MCoreGraph()
    ft = FoldTable()
    s1 = add_sym!(g, Sym(:p))
    s2 = add_sym!(g, Sym(:q))
    c1 = add_con!(g, Con(:wrap, [s1]))
    c2 = add_con!(g, Con(:wrap, [s2]))   # same head :wrap → same canonical key

    r1 = drive!(g, c1; ft=ft)
    r2 = drive!(g, c2; ft=ft)
    # r2 should have folded back to r1's recorded id.
    @test r2.terminated == :fold
    @test r2.n_folds >= 1
end

@testset "drive! — Choice triggers bounded_split when whistle blows" begin
    # Build a Choice with two alternatives. Stepper.Choice → Blocked → drive!
    # must invoke bounded_split. With no stats, split picks all alternatives
    # via the catch-all path.
    g  = MCoreGraph()
    s1 = add_sym!(g, Sym(:a))
    s2 = add_sym!(g, Sym(:b))
    alt1 = ChoiceAlt(NULL_NODE, s1)   # no guard
    alt2 = ChoiceAlt(NULL_NODE, s2)
    ch = add_choice!(g, Choice([alt1, alt2]))

    rs = drive!(g, ch)
    @test rs.terminated == :blocked
    @test rs.n_splits >= 1
end
