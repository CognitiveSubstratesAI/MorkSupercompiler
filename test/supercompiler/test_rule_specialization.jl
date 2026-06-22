using Test
using MorkSupercompiler

# ── Helpers — build the parent/ancestor canonical example ────────────────────
function _build_parent_ancestor(g::MCoreGraph)
    # Sym atoms
    s_alice = add_sym!(g, Sym(:alice))
    s_bob = add_sym!(g, Sym(:bob))
    s_carol = add_sym!(g, Sym(:carol))

    # Variables (Var(0)=X, Var(1)=Y, Var(2)=Z)
    vx = add_var!(g, Var(0))
    vy = add_var!(g, Var(1))
    vz = add_var!(g, Var(2))

    # Base facts
    f1 = add_con!(g, Con(:parent, [s_alice, s_bob]))      # (parent alice bob)
    f2 = add_con!(g, Con(:parent, [s_bob, s_carol]))      # (parent bob carol)

    # Rule: (ancestor X Y) :- (parent X Z), (ancestor Z Y)
    rule_body1 = add_con!(g, Con(:parent, [vx, vz]))
    rule_body2 = add_con!(g, Con(:ancestor, [vz, vy]))
    rule_head = add_con!(g, Con(:ancestor, [vx, vy]))
    rule_id = add_sym!(g, Sym(:ancestor_recursive))
    rule = Rule(rule_head, [rule_body1, rule_body2], rule_id)

    # Base rule: (ancestor X Y) :- (parent X Y) — needed so the recursive
    # rule has a base case to build on
    base_body = add_con!(g, Con(:parent, [vx, vy]))
    base_head_id = add_con!(g, Con(:ancestor, [vx, vy]))
    base_rule_id = add_sym!(g, Sym(:ancestor_base))
    base_rule = Rule(base_head_id, [base_body], base_rule_id)

    (s_alice, s_bob, s_carol, [f1, f2], [base_rule, rule])
end

# ── Partial Instantiation ────────────────────────────────────────────────────
@testset "specialize_rules — emits specialized variants per matching fact" begin
    g = MCoreGraph()
    s_alice, s_bob, s_carol, facts, rules = _build_parent_ancestor(g)

    # specialize the recursive rule against the 2 parent facts → 2 variants
    rec_rule = rules[2]   # the recursive ancestor rule
    result = specialize_rules(g, [rec_rule], facts)
    @test result isa SpecializationResult
    # 2 parent facts, each instantiates the X & Z in the rule head → 2 specialized rules
    @test length(result.specialized_rules) == 2

    # Each specialized rule should have ONE LESS body premise (the (parent X Z)
    # premise is consumed by the substitution).
    for sr in result.specialized_rules
        @test length(sr.body_ids) == length(rec_rule.body_ids) - 1
    end
end

@testset "specialize_rules — base rule with single premise yields ground facts" begin
    g = MCoreGraph()
    s_alice, s_bob, s_carol, facts, rules = _build_parent_ancestor(g)

    base_rule = rules[1]   # (ancestor X Y) :- (parent X Y)
    result = specialize_rules(g, [base_rule], facts)
    # Both premises consumed → derived ground facts, not new rules
    @test length(result.specialized_rules) == 0
    @test length(result.derived_facts) == 2     # ancestor(alice,bob) + ancestor(bob,carol)

    # Verify the head symbols
    for fid in result.derived_facts
        n = get_node(g, fid)
        @test n isa Con
        @test (n::Con).head === :ancestor
    end
end

@testset "specialize_rules — max_per_rule caps the per-rule output" begin
    g = MCoreGraph()
    _, _, _, facts, rules = _build_parent_ancestor(g)
    result = specialize_rules(g, [rules[1]], facts; max_per_rule=1)
    # max_per_rule=1 → at most 1 derived fact (or specialized rule) per rule
    @test length(result.derived_facts) + length(result.specialized_rules) <= 1
end

@testset "specialize_rules — semantic equivalence via differential saturation" begin
    # Boundary #3 / bisim shape: saturating with [original + specialized] rules
    # should produce a strict superset of atoms vs [original] alone. The extras
    # are duplicates eliminated by saturate!'s uniqueness gate, so the FINAL
    # atom set is the same.

    # Baseline: original rules only
    g1 = MCoreGraph()
    _, _, _, facts1, rules1 = _build_parent_ancestor(g1)
    kb1 = KBState(g1)
    for f in facts1
        kb_add_fact!(kb1, f)
    end
    for r in rules1
        kb_add_rule!(kb1, r)
    end
    saturate!(kb1; max_rounds=10)
    baseline_ancestor_count = length(index_lookup(kb1.index, :ancestor))
    @test baseline_ancestor_count >= 1    # at least 1 derived

    # Specialized: original + partial-instantiation variants
    g2 = MCoreGraph()
    _, _, _, facts2, rules2 = _build_parent_ancestor(g2)
    spec = specialize_rules(g2, rules2, facts2)
    kb2 = KBState(g2)
    for f in facts2
        kb_add_fact!(kb2, f)
    end
    for f in spec.derived_facts
        kb_add_fact!(kb2, f)
    end
    for r in rules2
        kb_add_rule!(kb2, r)
    end
    for r in spec.specialized_rules
        kb_add_rule!(kb2, r)
    end
    saturate!(kb2; max_rounds=10)
    specialized_ancestor_count = length(index_lookup(kb2.index, :ancestor))

    # Equivalence: specialized run produces same number of ancestor atoms
    # as the baseline run (specialization preserves semantics, dedup
    # eliminates duplicates).
    @test specialized_ancestor_count == baseline_ancestor_count
end

# ── Magic Sets ───────────────────────────────────────────────────────────────
@testset "magic_sets_transform — seeds a magic fact for the bound query value" begin
    g = MCoreGraph()
    s_alice, _, _, _, rules = _build_parent_ancestor(g)
    vy = add_var!(g, Var(1))
    query = add_con!(g, Con(:ancestor, [s_alice, vy]))    # (ancestor alice $Y)

    result = magic_sets_transform(g, rules, query; bound_position=0)
    @test result isa MagicSetsResult
    @test result.magic_pred === :magic_ancestor
    @test length(result.magic_seeds) == 1

    # Seed should be (magic_ancestor alice)
    seed = get_node(g, result.magic_seeds[1])
    @test seed isa Con
    @test (seed::Con).head === :magic_ancestor
    seed_arg = get_node(g, (seed::Con).fields[1])
    @test seed_arg isa Sym
    @test (seed_arg::Sym).name === :alice
end

@testset "magic_sets_transform — rewrites rules to add magic guard premise" begin
    g = MCoreGraph()
    s_alice, _, _, _, rules = _build_parent_ancestor(g)
    vy = add_var!(g, Var(1))
    query = add_con!(g, Con(:ancestor, [s_alice, vy]))

    result = magic_sets_transform(g, rules, query; bound_position=0)
    # Two original rules + extra magic-propagation rules
    @test length(result.rewritten_rules) >= length(rules)

    # Each rewritten rule with head :ancestor should now have a leading
    # magic_ancestor guard premise.
    for r in result.rewritten_rules
        head = get_node(g, r.head_id)
        head isa Con || continue
        if (head::Con).head === :ancestor
            @test !isempty(r.body_ids)
            first_premise = get_node(g, r.body_ids[1])
            @test first_premise isa Con
            @test (first_premise::Con).head === :magic_ancestor
        end
    end
end

@testset "magic_sets_transform — non-Con queries pass through unchanged" begin
    g = MCoreGraph()
    _, _, _, _, rules = _build_parent_ancestor(g)
    # Sym query — not a Con, so magic-sets can't extract an adornment.
    sq = add_sym!(g, Sym(:bogus))
    result = magic_sets_transform(g, rules, sq)
    @test result.magic_pred === :_no_magic
    @test isempty(result.magic_seeds)
    @test length(result.rewritten_rules) == length(rules)
end

@testset "magic_sets_transform — unbound (Var) at bound_position pass-through" begin
    g = MCoreGraph()
    _, _, _, _, rules = _build_parent_ancestor(g)
    vx = add_var!(g, Var(0))
    vy = add_var!(g, Var(1))
    fully_unbound = add_con!(g, Con(:ancestor, [vx, vy]))   # (ancestor $X $Y)
    result = magic_sets_transform(g, rules, fully_unbound; bound_position=0)
    # With a Var at the "bound" position, there's no concrete value to seed → no_magic
    @test result.magic_pred === :_no_magic
    @test isempty(result.magic_seeds)
end

@testset "magic_sets_transform — propagation rule has correct shape" begin
    g = MCoreGraph()
    s_alice, _, _, _, rules = _build_parent_ancestor(g)
    vy = add_var!(g, Var(1))
    query = add_con!(g, Con(:ancestor, [s_alice, vy]))

    result = magic_sets_transform(g, rules, query; bound_position=0)
    # At least one propagation rule should head with :magic_ancestor
    n_prop = 0
    for r in result.rewritten_rules
        head = get_node(g, r.head_id)
        if head isa Con && (head::Con).head === :magic_ancestor
            n_prop += 1
        end
    end
    @test n_prop >= 1
end
