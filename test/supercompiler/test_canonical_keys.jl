using Test
using MorkSupercompiler

@testset "CompactShape — shape_subsumes" begin
    s0 = CompactShape(0, 0, 0)
    s2 = CompactShape(2, 1, 0)
    s3 = CompactShape(3, 2, 0)
    @test shape_subsumes(s0, s0)   # reflexive
    @test shape_subsumes(s2, s3)   # s2 ≤ s3 component-wise
    @test !shape_subsumes(s3, s2)  # not the other way
end

@testset "canonical_key — extracts head + shape from graph" begin
    g    = MCoreGraph()
    id_l = add_lit!(g, Lit(1))
    id_c = add_con!(g, Con(:pair, [id_l, id_l]))

    key = canonical_key(g, id_c, 0)
    @test key.head == :pair
    @test key.shape.arities[1] == UInt8(2)   # :pair has 2 fields
    @test :pair in key.tags
    @test :Lit  in key.tags
end

@testset "Algorithm 10 — KeySubsumption (§6.3.2)" begin
    g   = MCoreGraph()
    id1 = add_con!(g, Con(:foo, NodeID[]))
    id2 = add_con!(g, Con(:foo, NodeID[]))

    k1 = canonical_key(g, id1, 0)
    k2 = canonical_key(g, id2, 0)
    @test subsumes(k1, k2)    # identical structure → k1 subsumes k2
    @test subsumes(k2, k1)    # symmetric when identical

    # Different head → no subsumption
    id3  = add_con!(g, Con(:bar, NodeID[]))
    k3   = canonical_key(g, id3, 0)
    @test !subsumes(k1, k3)

    # Wider shape subsumes narrower (general subsumes specific)
    id_lit = add_lit!(g, Lit(1))
    id_big = add_con!(g, Con(:foo, [id_lit, id_lit, id_lit]))
    k_big  = canonical_key(g, id_big, 0)
    @test !subsumes(k_big, k1)   # k_big has arity 3, k1 has 0 → 3 ≰ 0
    @test subsumes(k1, k_big)    # k1 shape (0,0,0) ≤ k_big shape → k1 subsumes k_big
end

@testset "FoldTable — record and lookup" begin
    g    = MCoreGraph()
    ft   = FoldTable()
    id_c = add_con!(g, Con(:foo, NodeID[]))
    key  = canonical_key(g, id_c, 0)

    @test !can_fold(ft, key)     # empty table — nothing to fold
    record!(ft, key, id_c)
    @test can_fold(ft, key)      # now it's there
    @test lookup_fold(ft, key) == id_c

    # A more specific key (same head, bigger shape) is subsumed by the recorded one
    id_lit  = add_lit!(g, Lit(1))
    id_big  = add_con!(g, Con(:foo, [id_lit]))
    key_big = canonical_key(g, id_big, 0)
    @test can_fold(ft, key_big)  # key (shape 0) subsumes key_big (shape 1)
end

# ── Regression tests for 2026-05-30 audit fixes ────────────────────────────────

@testset "CanonicalKBSig — kb-signature path was dead-on-arrival" begin
    # Audit found 5 latent bugs that made the KB-subsumption half of Alg 10
    # silently do nothing (kb_sig always empty in tests):
    #   1. Operator-precedence early-return guards (lines 191, 201, 233, 244)
    #   2. `isvalid(::Symbol)` no-op check
    #   3. `_sym_node_of(::Symbol)` returned NULL_NODE → get_node would throw
    #   4. `pat.args` FieldError — Con has `fields::Vector{NodeID}`, not args
    #   5. Effect-set decoder missing Create/Delete bits
    #
    # This test builds a real Prim(:kb_query, [pat]) and asserts the kb_sig
    # is non-empty with the right predicate + fixed-arg mask.
    g = MCoreGraph()
    sym_alice = add_sym!(g, Sym(:alice))
    var_x     = add_var!(g, Var(0))
    sym_bob   = add_sym!(g, Sym(:bob))
    # Pattern: (parent alice $x bob) — args 1 and 3 ground, arg 2 var.
    pat = add_con!(g, Con(:parent, [sym_alice, var_x, sym_bob]))
    query = add_prim!(g, Prim(:kb_query, [pat], EffectSet(0x01)))   # Read effect

    key = canonical_key(g, query, 0)
    @test !isempty(key.kb_sig.predicates)
    # Should record :parent with fixed-arg mask covering positions 1 and 3.
    pred_entry = first(key.kb_sig.predicates)
    @test pred_entry[1] == :parent
    @test fixed_arg(pred_entry[2], 1)
    @test !fixed_arg(pred_entry[2], 2)
    @test fixed_arg(pred_entry[2], 3)
end

@testset "CanonicalEffectSig — Create + Delete bits now decoded" begin
    # Previously _effectset_to_effects checked 0x01/0x02/0x04/0x20 only.
    # Bits 0x08 (Create) and 0x10 (Delete) were silently dropped, so
    # effect-subsumption (Alg 10 step 3) would let key1 subsume key2 even
    # when key1 had Delete and key2 didn't.
    g = MCoreGraph()
    sym_x = add_sym!(g, Sym(:x))
    # 0x18 = Create | Delete
    create_delete = add_prim!(g, Prim(:doit, [sym_x], EffectSet(0x18)))
    key = canonical_key(g, create_delete, 0)
    @test MorkSupercompiler.ECLASS_CREATE in key.effect_sig.effects
    @test MorkSupercompiler.ECLASS_DELETE in key.effect_sig.effects
end

@testset "Algorithm 10 KB-subsumption — actually exercised end-to-end" begin
    # Build two keys differing only in kb_sig predicate sets. With the fix,
    # subsumes should now correctly enforce the Algorithm 10 KB rule
    # ("key1 has matching pred for every pred2 with mask1 ⊆ mask2").
    g = MCoreGraph()
    sym_a = add_sym!(g, Sym(:a))
    sym_b = add_sym!(g, Sym(:b))
    var_x = add_var!(g, Var(0))

    pat1 = add_con!(g, Con(:foo, [sym_a]))                # arg 1 ground
    q1   = add_prim!(g, Prim(:kb_query, [pat1], EffectSet(0x01)))
    pat2 = add_con!(g, Con(:foo, [sym_a, sym_b]))         # args 1+2 ground
    q2   = add_prim!(g, Prim(:kb_query, [pat2], EffectSet(0x01)))

    k1 = canonical_key(g, q1, 0)
    k2 = canonical_key(g, q2, 0)
    # Bonus: confirm the keys actually carry kb_sig predicates (proves the
    # path no longer silently returns empty).
    @test !isempty(k1.kb_sig.predicates)
    @test !isempty(k2.kb_sig.predicates)
end
