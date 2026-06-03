using Test
using MorkSupercompiler

@testset "KBSaturation — base facts + VersionedIndex" begin
    g = MCoreGraph()
    kb = KBState(g)

    id1 = add_con!(g, Con(:parent, [add_sym!(g, Sym(:alice)), add_sym!(g, Sym(:bob))]))
    id2 = add_con!(g, Con(:parent, [add_sym!(g, Sym(:bob)), add_sym!(g, Sym(:carol))]))
    kb_add_fact!(kb, id1)
    kb_add_fact!(kb, id2)

    @test length(index_lookup(kb.index, :parent)) == 2
    @test length(all_facts(kb)) == 2
    @test length(kb.delta) == 2   # both in delta before saturation
end

@testset "KBSaturation — saturate! reaches fixed point" begin
    g = MCoreGraph()
    kb = KBState(g)

    # Fact: (parent alice bob)
    id_alice = add_sym!(g, Sym(:alice))
    id_bob = add_sym!(g, Sym(:bob))
    id_carol = add_sym!(g, Sym(:carol))
    id_p1 = add_con!(g, Con(:parent, [id_alice, id_bob]))
    id_p2 = add_con!(g, Con(:parent, [id_bob, id_carol]))
    kb_add_fact!(kb, id_p1)
    kb_add_fact!(kb, id_p2)

    # Rule: (parent X Y) ∧ (parent Y Z) → (ancestor X Z)
    id_vx = add_var!(g, Var(0))
    id_vy = add_var!(g, Var(1))
    id_vz = add_var!(g, Var(2))
    id_body1 = add_con!(g, Con(:parent, [id_vx, id_vy]))
    id_body2 = add_con!(g, Con(:parent, [id_vy, id_vz]))
    id_head = add_con!(g, Con(:ancestor, [id_vx, id_vz]))
    rule_id = add_sym!(g, Sym(:ancestor_rule))
    rule = Rule(id_head, [id_body1, id_body2], rule_id)
    kb_add_rule!(kb, rule)

    n_new = saturate!(kb; max_rounds=10)
    @test n_new >= 1   # at least (ancestor alice carol) derived

    anc_facts = index_lookup(kb.index, :ancestor)
    @test length(anc_facts) >= 1
end

@testset "KBSaturation — idempotent (second saturate! adds nothing)" begin
    g = MCoreGraph()
    kb = KBState(g)

    id_a = add_con!(g, Con(:fact, [add_sym!(g, Sym(:a))]))
    kb_add_fact!(kb, id_a)
    saturate!(kb; max_rounds=5)

    n2 = saturate!(kb; max_rounds=5)
    @test n2 == 0   # no rules → fixed point immediately
end
