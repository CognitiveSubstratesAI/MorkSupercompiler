using Test
using MorkSupercompiler

@testset "MCore — 11 node types (§3.1)" begin
    g = MCoreGraph()

    id_foo = add_sym!(g, Sym(:foo))
    id_x = add_var!(g, Var(0))
    id_42 = add_lit!(g, Lit(42))
    id_con = add_con!(g, Con(:kb_fact, [id_foo]))
    id_app = add_app!(g, App(id_foo, [id_x]))
    id_abs = add_abs!(g, Abs([0], id_app))
    id_let = add_let!(g, LetNode([(0, id_42)], id_foo))
    id_match = add_match!(g, MatchNode(id_x, [MatchClause(id_foo, id_app)]))
    id_choice = add_choice!(g, Choice([ChoiceAlt(id_foo)]))
    id_prim = add_prim!(g, Prim(:kb_query, [id_foo]))
    id_ref = add_mref!(g, MCoreRef(:my_def))

    @test (get_node(g, id_foo)::Sym).name == :foo
    @test (get_node(g, id_x)::Var).ix == 0
    @test (get_node(g, id_42)::Lit).val == 42
    @test (get_node(g, id_con)::Con).head == :kb_fact
    @test (get_node(g, id_app)::App).fun == id_foo
    @test (get_node(g, id_abs)::Abs).body == id_app
    @test (get_node(g, id_let)::LetNode).body == id_foo
    @test length((get_node(g, id_match)::MatchNode).clauses) == 1
    @test length((get_node(g, id_choice)::Choice).alts) == 1
    @test (get_node(g, id_prim)::Prim).op == :kb_query
    @test (get_node(g, id_ref)::MCoreRef).def_id == :my_def
    @test g.next_id == UInt32(12)
end

@testset "MCore — domain compilation (§3.2)" begin
    g = MCoreGraph()
    pat = add_sym!(g, Sym(:pattern))
    kid = compile_kb_query(g, pat)
    @test isvalid(kid)
    n = get_node(g, kid)::Prim
    @test n.op == :kb_query && n.args == [pat]
end

@testset "UncertainNode (§2.4 approx spec)" begin
    vp = PBox(0.8, 1.0, 0.95)
    cp = PBox(0.0, 10.0, 1.0)
    g = MCoreGraph()
    bid = add_sym!(g, Sym(:base))
    un = UncertainNode(bid, vp, cp, 0.05)
    @test un.base == bid && un.error_bound == 0.05
    @test un.value_pbox.confidence == 0.95
end
