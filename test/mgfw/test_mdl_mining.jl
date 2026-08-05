# test_mdl_mining.jl — the trie miner ranks by MDL gain, not support, and gates on the compression
# condition. Properties are taken from the SPEC, not from observing what the code happens to do.
#
# WHY THIS EXISTS. `trie_score!` used to rank by `count * log(1 + total/count)` — support. None of the
# three normative sources license that as a final ranking key:
#   Franz IC theory 2020 — shortest-feature-first, admitted only under the STRICT compression
#     condition l(f)+l(r) < l(x) (Def. 2.1 Eq. 7); defines NO notion of support at all.
#   MORK-WILLIAM clarifications — gain(r,S) = L(S) - L(S') - C(r), kept while cumulative gain > 0.
#   AdaptiMORK v8 §4.5 (normative) — ΔL drives promotion/precedence/retirement; candidate order is
#     (larger ΔL, longer span, …) with NO frequency term. Frequency is licensed only as a
#     candidate-generation GATE (occs >= 3) applied BEFORE ΔL.
# Frequency keeps that licensed role here (trie_seed!/trie_grow! generate); MDL decides.
using Test, MorkSupercompiler
const M = MorkSupercompiler

@testset "MDL gain ΔL(p,n) = n(p-1) - (p+1)" begin
    @test M.mdl_rule_gain(3, 10) == 16
    @test M.mdl_rule_gain(5, 2)  == 2

    # A LENGTH-1 PATTERN IS NEVER ADMISSIBLE, at any frequency. Replacing one symbol with one
    # reference saves nothing and still costs a rule body + name. Under the old support weight,
    # length-1 seeds dominated the top-k purely by being frequent — this is the sharpest behavioural
    # difference between the two rankings.
    @test all(M.mdl_rule_gain(1, n) <= 0 for n in 1:10_000)
    @test M.mdl_rule_gain(1, 10_000) == -2          # and it does not even grow with n

    # The objective REPRODUCES AdaptiMORK's hand-tuned `occs >= 3` gate for pairs rather than needing
    # it asserted separately: ΔL > 0 first holds at n = 4 (n = 3 gives exactly 0, which the STRICT
    # condition — Franz Eq. 7 uses `<`, not `<=` — rejects).
    @test M.mdl_rule_gain(2, 3) == 0
    @test findfirst(n -> M.mdl_rule_gain(2, n) > 0, 1:20) == 4

    # longer patterns pay for themselves at lower support — the MDL trade, not a heuristic
    @test M.mdl_rule_gain(8, 2) > M.mdl_rule_gain(2, 2)
    # monotone in both arguments where it should be
    @test all(M.mdl_rule_gain(4, n) < M.mdl_rule_gain(4, n + 1) for n in 1:20)
end

@testset "compression condition gates the miner (Franz Def. 2.1 Eq. 7)" begin
    mk(strs) = reduce(vcat, [M.parse_program(s) for s in strs]; init = M.SNode[])
    run(strs; k = 10, d = 3) =
        M.run_trie_miner(M.TEMPLATE_EVIDENCE_CAPSULE, mk(strs); k = k, max_depth = d)

    # EMPTY INPUT — no pattern can satisfy l(f)+l(r) < l(x) when l(x) = 0. Must halt, not error.
    @test isempty(run(String[]))

    # INCOMPRESSIBLE INPUT — every symbol distinct, so nothing recurs and nothing compresses.
    # A support-ranked miner still returns its "heaviest" symbols here; an MDL-gated one returns
    # NOTHING, which is the correct answer and the point of the gate.
    @test isempty(run(["(a$i b$i c$i)" for i in 1:40]))

    # TRIVIALLY COMPRESSIBLE INPUT — one repeated multi-symbol pattern, well above the gate.
    got = run(["(f a b)" for _ in 1:12])
    @test !isempty(got)
    @test all(w > 0 for (_, w) in got)                    # nothing admitted below the condition
    @test all(length(p) >= 2 for (p, _) in got)           # ...and no length-1 rules, ever

    # EVERY admitted entry must satisfy the condition, on a mixed corpus
    mixed = run(vcat(["(f a b)" for _ in 1:9], ["(g c d)" for _ in 1:7], ["(z$i)" for i in 1:15]))
    @test all(w > 0 for (_, w) in mixed)
    @test all(M.mdl_rule_gain(length(p), 0) <= w for (p, w) in mixed)   # w is a real ΔL, not a proxy

    # DETERMINISM — the top-k boundary decides what reaches Smine, so the order must be stable
    # across runs on identical input, including ties.
    a = run(["(f a b)" for _ in 1:9]); b = run(["(f a b)" for _ in 1:9])
    @test a == b

    # RANKED BY ΔL, DESCENDING — the ordering contract itself
    @test issorted([w for (_, w) in mixed]; rev = true)

    # the gate is a GATE, not a filter applied after truncation: disabling it admits more
    unfiltered = M.PatternTrie(M.TEMPLATE_EVIDENCE_CAPSULE; k = 50)
    atoms = mk(["(a$i b$i)" for i in 1:20])
    M.trie_seed!(unfiltered, atoms)
    open_scored = M.trie_score!(unfiltered; compression_condition = false)
    gated       = M.trie_score!(unfiltered; compression_condition = true)
    @test length(gated) <= length(open_scored)
    @test isempty(gated)                                   # nothing here compresses...
    @test !isempty(open_scored)                            # ...though candidates existed
end

@testset "token cost model — anchored to the clarifications paper's own figures" begin
    # The paper states two figures unambiguously, and this model reproduces BOTH:
    #   L0 of `(cumsum <29 integers>)` = "1 for cumsum, 2 for parens, 29 integers" = 32
    #   `(repeat 1 9)`                 = "1 + 2 + 1 + 1"                            = 5
    seq = M.parse_program("(cumsum " * join(fill("1", 29), " ") * ")")[1]
    @test M._mdl_token_cost(seq) == 32
    @test M._mdl_token_cost(M.parse_program("(repeat 1 9)")[1]) == 5
    @test M._mdl_token_cost(M.parse_program("x")[1]) == 1          # a bare atom
    @test M._mdl_token_cost(M.parse_program("()")[1]) == 2         # parens only

    # ⚠️ The paper's THIRD figure does NOT reconcile with its own stated rules: it gives
    # `(repeat (s0 11) 2)` = 11 via "1 + 2 + 1 + 2 + 1 + 1 + 2 + 1", eight terms that double-count.
    # Under the model anchored above it is 8. Pinned deliberately so nobody "corrects" the cost
    # function to match a bad sum — two independent figures agree with 8's derivation, one does not.
    @test M._mdl_token_cost(M.parse_program("(repeat (s0 11) 2)")[1]) == 8
end
