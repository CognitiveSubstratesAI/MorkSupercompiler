using Test
using MorkSupercompiler

# §4.1 UncertainFact
@testset "UncertainFact (§4.1)" begin
    f = certain_fact(:parent, ["alice", "bob"])
    @test f.predicate == :parent
    @test f.arguments == ["alice", "bob"]
    @test f.truth_pbox.intervals[1] == (1.0, 1.0)
    @test f.confidence == 1.0
    @test f.derivation isa ProofTree

    # With explicit p-box
    pb = pbox_interval(0.7, 0.9, 0.9)
    f2 = UncertainFact(:edge, ["a", "b"], pb)
    @test f2.truth_pbox === pb
    @test f2.confidence == 0.9
end

# §4.2.1 Conjunction AND — three cases
@testset "conjunction_and — independent (product rule)" begin
    T_A = pbox_interval(0.8, 0.9, 1.0)
    T_B = pbox_interval(0.7, 0.8, 1.0)
    result = conjunction_and(T_A, T_B)
    lo, hi = result.intervals[1]
    @test lo ≈ 0.8 * 0.7 atol=0.01   # product of lower bounds
    @test hi ≈ 0.9 * 0.8 atol=0.01   # product of upper bounds
end

@testset "conjunction_and — perfectly correlated (Łukasiewicz)" begin
    T_A = pbox_interval(0.8, 0.9, 1.0)
    T_B = pbox_interval(0.7, 0.8, 1.0)
    # Mark as perfectly correlated (same bit)
    T_A2, T_B2 = mark_dependent(T_A, T_B, 1)
    T_A3 = PBox(T_A2.intervals, T_A2.probabilities, T_A2.confidence, T_A2.correlation_sig)
    T_B3 = PBox(T_B2.intervals, T_B2.probabilities, T_B2.confidence, T_A2.correlation_sig)  # same sig
    result = conjunction_and(T_A3, T_B3)
    lo, hi = result.intervals[1]
    @test lo ≈ max(0.8 + 0.7 - 1.0, 0.0) atol=0.01   # Łukasiewicz
    @test hi ≈ max(0.9 + 0.8 - 1.0, 0.0) atol=0.01
end

@testset "conjunction_and — Fréchet (partially correlated)" begin
    T_A = pbox_interval(0.8, 0.9, 1.0)
    T_B = pbox_interval(0.7, 0.8, 1.0)
    T_A2, T_B2 = mark_dependent(T_A, T_B, 1)  # dependent but different sigs → Fréchet
    result = conjunction_and(T_A2, T_B2)
    # Fréchet result should be wider than independent
    indep = conjunction_and(T_A, T_B)
    @test width(result) >= width(indep) - 1e-9
end

# §4.2.2 Algorithm 3 — MatchWithUncertainty
@testset "MatchWithUncertainty (Algorithm 3, §4.2.2)" begin
    # Use ground atoms (no variables) for clear exact/similar/different tests
    f_exact = parse_sexpr("(parent alice bob)")
    f_sim   = parse_sexpr("(parent alice carol)")   # similar: same head + first arg
    f_diff  = parse_sexpr("(ancestor alice carol)")  # different head

    # Exact match → PBox.exact(1.0)
    result_exact = match_with_uncertainty(f_exact, f_exact, 0.1)
    @test result_exact !== NO_MATCH
    @test (result_exact::PBox).intervals[1] == (1.0, 1.0)

    # Structurally similar (same arity, same head) → uncertain match with tolerance=0.5
    result_sim = match_with_uncertainty(f_exact, f_sim, 0.5)
    @test result_sim !== NO_MATCH
    lo, hi = (result_sim::PBox).intervals[1]
    @test 0.0 <= lo <= hi <= 1.0

    # Different head, tight tolerance → NO_MATCH
    result_diff = match_with_uncertainty(f_exact, f_diff, 0.01)
    @test result_diff === NO_MATCH
end

@testset "structural_similarity" begin
    a = parse_sexpr("(foo \$x \$y)")
    b = parse_sexpr("(foo \$x \$y)")
    @test structural_similarity(a, b) ≈ 1.0

    c = parse_sexpr("(bar \$x \$y)")
    @test structural_similarity(a, c) < 1.0   # different head

    d = parse_sexpr("(foo a b)")
    @test 0.0 < structural_similarity(a, d) < 1.0   # partial match
end

# §4.2.3 Algorithm 4 — ApplyRule (UncertainModusPonens)
@testset "ApplyRule — UncertainModusPonens (Algorithm 4, §4.2.3)" begin
    premise = pbox_interval(0.8, 0.9, 0.95)
    rule_str = pbox_interval(0.9, 1.0, 0.9)

    conc = apply_rule(premise, rule_str, 0)   # depth=0
    @test !isempty(conc.intervals)
    lo, hi = conc.intervals[1]
    @test lo > 0.0 && hi <= 1.0 + 0.1   # plausible conclusion strength

    # Deeper inference → wider (more uncertain) conclusion
    conc_deep = apply_rule(premise, rule_str, 5)
    @test width(conc_deep) >= width(conc) - 1e-9

    # Correlation sig merges
    p2 = PBox(premise.intervals, premise.probabilities, premise.confidence,
              BitVector([true, false]))
    r2 = PBox(rule_str.intervals, rule_str.probabilities, rule_str.confidence,
              BitVector([false, true]))
    conc2 = apply_rule(p2, r2, 0)
    @test length(conc2.correlation_sig) >= 2
    @test any(conc2.correlation_sig)   # merged sigs
end

# §4.4 Convergence theorem
@testset "convergence_width_bound (§4.4)" begin
    w1 = convergence_width_bound(100, 0.1)
    w2 = convergence_width_bound(400, 0.1)
    @test w2 < w1   # more iterations → narrower
    @test convergence_width_bound(100, 0.5) < convergence_width_bound(100, 0.1)  # higher rate → narrower
end

# InferenceContext + derive_fact
@testset "derive_fact" begin
    premise = certain_fact(:parent, ["alice", "bob"])
    rule_str = pbox_interval(0.9, 1.0, 1.0)
    ctx = InferenceContext()

    derived = derive_fact(premise, rule_str, :ancestor, ["alice", "carol"], ctx)
    @test derived.predicate == :ancestor
    @test derived.arguments == ["alice", "carol"]
    @test !isempty(derived.truth_pbox.intervals)
    @test derived.derivation.depth == 0

    # Deeper context widens uncertainty
    ctx_deep = InferenceContext(5, 0.05, balanced())
    derived_deep = derive_fact(premise, rule_str, :ancestor, ["alice", "carol"], ctx_deep)
    @test width(derived_deep.truth_pbox) >= width(derived.truth_pbox) - 1e-9
end

# ── Regression tests for 2026-05-30 audit fixes ────────────────────────────────

@testset "disjunction_or — inclusion-exclusion upper bound" begin
    # Previously: upper bound was hard-clamped to `min(hi_ab, 1.0)` without
    # subtracting `lo_and`. So `or(0.5, 0.5)` returned [0.75, 1.0] instead of
    # the correct [0.75, 0.75] from P(A∨B) = P(A) + P(B) − P(A∧B).
    a = pbox_exact(0.5)
    b = pbox_exact(0.5)
    o = disjunction_or(a, b)
    @test !isempty(o.intervals)
    lo, hi = o.intervals[1]
    # For independent point-mass 0.5 each:
    #   ab  = 1.0;   and = 0.25
    #   or  = ab − and = [0.75, 0.75]
    @test isapprox(lo, 0.75; atol=1e-9)
    @test isapprox(hi, 0.75; atol=1e-9)
end

@testset "_and_frechet — uses conjunction Fréchet bounds not additive" begin
    # Previously delegated to add_pbox (additive Fréchet for SUM). Spec §2.3
    # + §4.2.1 prescribe joint-mass bounds [max(p_A + p_B − 1, 0), min(p_A, p_B)].
    # Need PARTIAL correlation (dependent but unequal sigs) to route to
    # _and_frechet, not _and_lukasiewicz (which fires on identical sigs).
    a0 = pbox_exact(0.6)
    b0 = pbox_exact(0.7)
    a1, b1 = mark_dependent(a0, b0, 1)   # share bit 1
    # Add bit 2 to b only — now a.sig = [1], b.sig = [1,1]; overlap but unequal.
    _, b = mark_dependent(b1, b1, 2)
    a = a1

    z = conjunction_and(a, b)   # dispatches to _and_frechet since dependent + unequal
    @test !isempty(z.intervals)
    lo, hi = z.intervals[1]
    # Fréchet conjunction of 0.6 and 0.7: [max(0.3, 0), min(0.6, 0.7)] = [0.3, 0.6]
    @test isapprox(lo, 0.3; atol=1e-9)
    @test isapprox(hi, 0.6; atol=1e-9)
    # Crucially: upper bound MUST be ≤ min(p_A, p_B) = 0.6, NOT > 1.0 as
    # additive Fréchet would yield (0.6 + 0.7 = 1.3 widened).
    @test hi <= 0.6 + 1e-9
end

@testset "convergence_width_bound — guard fires on either invalid arg" begin
    # Previously: `n <= 0 || r <= 0 && return Inf` parses as
    # `n <= 0 || (r <= 0 && return Inf)`. So invalid n with valid r fell through
    # and tried sqrt(non-positive) → NaN. Parens fix forces early return.
    @test convergence_width_bound(0, 0.5) == Inf
    @test convergence_width_bound(-1, 0.5) == Inf
    @test convergence_width_bound(100, 0.0) == Inf
    @test convergence_width_bound(100, -0.1) == Inf
end
