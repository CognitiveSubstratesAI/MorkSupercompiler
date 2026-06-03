using Test
using MorkSupercompiler

@testset "MM2Compiler — sprint_mcore_to_mm2" begin
    g = MCoreGraph()
    id_s = add_sym!(g, Sym(:foo))
    id_l = add_lit!(g, Lit(42))
    id_v = add_var!(g, Var(0))
    id_c = add_con!(g, Con(:pair, [id_s, id_v]))

    @test sprint_mcore_to_mm2(g, id_s) == "foo"
    @test sprint_mcore_to_mm2(g, id_l) == "42"
    @test sprint_mcore_to_mm2(g, id_v) == "\$x0"
    @test sprint_mcore_to_mm2(g, id_c) == "(pair foo \$x0)"
end

@testset "MM2Compiler — priority encoding (§9.3)" begin
    @test sprint_priority(MM2Priority(1)) == "(1 0)"
    @test sprint_priority(MM2Priority(3, 2)) == "(3 2)"
    @test MM2Priority(1) < MM2Priority(2)
    @test MM2Priority(2, 0) < MM2Priority(2, 1)
end

@testset "MM2Compiler — compile_conditional! produces 2 atoms" begin
    g = MCoreGraph()
    ctx = CompileCtx(g)
    id_cond = add_sym!(g, Sym(:cond))
    id_then = add_sym!(g, Sym(:then_val))
    id_else = add_sym!(g, Sym(:else_val))

    atoms = compile_conditional!(ctx, id_cond, id_then, id_else)
    @test length(atoms) == 2
    # First atom: pattern contains cond
    @test occursin("cond", atoms[1].pattern)
    # Second atom: pattern contains "not"
    @test occursin("not", atoms[2].pattern)
    # Second atom has higher priority
    @test atoms[1].priority < atoms[2].priority
end

@testset "MM2Compiler — compile_program produces loadable s-exprs" begin
    g = MCoreGraph()
    id_pat = add_con!(g, Con(:edge, [add_var!(g, Var(0)), add_var!(g, Var(1))]))
    id_tmpl = add_con!(g, Con(:node, [add_var!(g, Var(0))]))
    id_exec = add_prim!(
        g, Prim(:mm2_exec, [NULL_NODE, id_pat, id_tmpl], EffectSet(UInt8(0x05)))
    )

    prog, obligs = compile_program(g, [id_exec])
    @test !isempty(prog)
    @test occursin("exec", prog)
    @test !isempty(obligs)
    # All three bisimulation obligations recorded
    kinds = Set(o.kind for o in obligs)
    @test :forward_sim in kinds
    @test :backward_sim in kinds
    @test :fairness in kinds
end

@testset "MM2Compiler — sequential compilation preserves order" begin
    g = MCoreGraph()
    ctx = CompileCtx(g)
    ids = [add_prim!(g, Prim(:mm2_exec, NodeID[], EffectSet(UInt8(0x05)))) for _ in 1:3]
    atoms = compile_sequential!(ctx, ids)
    @test length(atoms) == 3
    # Priorities must be strictly increasing
    @test atoms[1].priority < atoms[2].priority < atoms[3].priority
end
