using Test
using MorkSupercompiler

# =====================================================================
# register_space_primitives! — live Space wiring tests
# =====================================================================

@testset "register_space_primitives! — kb_query reads atom count" begin
    s = new_space()
    space_add_all_sexpr!(s, "(edge a b) (edge b c) (edge c d)")

    g = MCoreGraph()
    reg = copy(DEFAULT_PRIM_REGISTRY)
    register_space_primitives!(reg, s)

    # Build: kb_query(pattern=(edge $x $y))
    var_x = add_var!(g, Var(0))
    var_y = add_var!(g, Var(1))
    pat_id = add_con!(g, Con(:edge, [var_x, var_y]))
    prim_id = add_prim!(g, Prim(:kb_query, [pat_id], EffectSet(UInt8(0x01))))

    env = Env()
    result = rewrite_once(g, prim_id, env, DepSet(), reg)

    @test result isa Value
    node = get_node(g, result.id)
    @test node isa Lit
    @test node.val == 3
end

@testset "register_space_primitives! — kb_query no args → count 0" begin
    s = new_space()
    g = MCoreGraph()
    reg = copy(DEFAULT_PRIM_REGISTRY)
    register_space_primitives!(reg, s)

    prim_id = add_prim!(g, Prim(:kb_query, NodeID[], EffectSet(UInt8(0x01))))
    env = Env()
    result = rewrite_once(g, prim_id, env, DepSet(), reg)

    @test result isa Value
    node = get_node(g, result.id)
    @test node isa Lit
    @test node.val == 0
end

@testset "register_space_primitives! — mm2_exec returns Value" begin
    s = new_space()
    space_add_all_sexpr!(s, "(node a) (node b)")

    g = MCoreGraph()
    reg = copy(DEFAULT_PRIM_REGISTRY)
    register_space_primitives!(reg, s)

    # :mm2_exec handler receives the (already-evaluated) arg node ids.
    # Pass a Sym as arg (handler calls sprint_mcore_to_mm2 on it).
    arg_id = add_sym!(g, Sym(Symbol("exec_placeholder")))
    prim_id = add_prim!(g, Prim(:mm2_exec, [arg_id], EffectSet(UInt8(0x05))))
    env = Env()
    result = rewrite_once(g, prim_id, env, DepSet(), reg)

    # mm2_exec always returns Value wrapping the arg
    @test result isa Value
end

@testset "register_space_primitives! — load_defs no crash on (= ...) atoms" begin
    s = new_space()
    space_add_all_sexpr!(s, "(= double-val (foo bar))")

    g = MCoreGraph()
    reg = copy(DEFAULT_PRIM_REGISTRY)
    register_space_primitives!(reg, s)

    prim_id = add_prim!(g, Prim(:load_defs, NodeID[], EffectSet(UInt8(0x00))))
    env = Env()
    result = rewrite_once(g, prim_id, env, DepSet(), reg)

    @test result isa Value
end

@testset "register_space_primitives! — reregister replaces default kb_query" begin
    s = new_space()
    space_add_all_sexpr!(s, "(x 1) (x 2)")

    g = MCoreGraph()
    reg = copy(DEFAULT_PRIM_REGISTRY)

    # Before register: DEFAULT kb_query returns Residual (no space wired)
    var_x = add_var!(g, Var(0))
    pat_id = add_con!(g, Con(:x, [var_x]))
    prim_id = add_prim!(g, Prim(:kb_query, [pat_id], EffectSet(UInt8(0x01))))
    r_before = rewrite_once(g, prim_id, Env(), DepSet(), reg)
    # Default: Residual (no Space wired)
    @test r_before isa Residual

    # After register: returns correct count
    register_space_primitives!(reg, s)
    g2 = MCoreGraph()
    var_x2 = add_var!(g2, Var(0))
    pat_id2 = add_con!(g2, Con(:x, [var_x2]))
    prim_id2 = add_prim!(g2, Prim(:kb_query, [pat_id2], EffectSet(UInt8(0x01))))
    r_after = rewrite_once(g2, prim_id2, Env(), DepSet(), reg)
    @test r_after isa Value
    @test get_node(g2, r_after.id) isa Lit
    @test (get_node(g2, r_after.id)::Lit).val == 2
end
