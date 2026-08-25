using Test
using MorkSupercompiler

# §2.2 PBox struct — all 4 spec fields
@testset "PBox struct (§2.2)" begin
    pb = pbox_exact(0.5)
    @test pb.intervals == [(0.5, 0.5)]
    @test pb.probabilities ≈ [1.0]
    @test pb.confidence ≈ 1.0
    @test pb.correlation_sig isa BitVector   # 4th field from spec

    pb2 = pbox_interval(0.2, 0.8, 0.95)
    @test pb2.intervals == [(0.2, 0.8)]
    @test pb2.probabilities ≈ [0.95]

    pb3 = pbox_empty()
    @test isempty(pb3.intervals)
    @test width(pb3) == 0.0
end

# §2.3 width / max_width / overlap
@testset "PBox properties" begin
    pb = pbox_interval(0.3, 0.7, 1.0)
    @test width(pb) ≈ 0.4
    @test max_width(pb) ≈ 0.4

    @test overlap(pbox_exact(0.5), pbox_exact(0.5)) ≈ 1.0
    @test overlap(pbox_interval(0.0, 0.3, 1.0), pbox_interval(0.7, 1.0, 1.0)) ≈ 0.0
    # Fully overlapping intervals (both are single intervals, they share [0.4,0.6]) → overlap = 1.0
    @test overlap(pbox_interval(0.0, 0.6, 1.0), pbox_interval(0.4, 1.0, 1.0)) == 1.0
    # Non-overlapping intervals → 0.0
    @test overlap(pbox_interval(0.0, 0.3, 1.0), pbox_interval(0.7, 1.0, 1.0)) == 0.0
end

# §2.2 correlation_sig — dependency detection
@testset "correlation_sig — are_dependent" begin
    a = pbox_interval(0.0, 1.0, 1.0)
    b = pbox_interval(0.0, 1.0, 1.0)
    @test !are_dependent(a, b)   # no shared bits → independent

    a2, b2 = mark_dependent(a, b, 1)
    @test are_dependent(a2, b2)   # bit 1 shared → dependent
    @test !are_dependent(a, b2)   # a unchanged
end

# §2.3 Algorithm 1 — AddPBox independent case
@testset "AddPBox — independent (Algorithm 1)" begin
    X = pbox_interval(1.0, 2.0, 1.0)
    Y = pbox_interval(3.0, 4.0, 1.0)
    Z = add_pbox(X, Y)
    @test !isempty(Z.intervals)
    lo, hi = Z.intervals[1]
    @test lo ≈ 4.0   # 1+3
    @test hi ≈ 6.0   # 2+4
    @test sum(Z.probabilities) ≈ 1.0   # prob sums to 1

    # Point masses: sum of two exact values
    Za = add_pbox(pbox_exact(2.0), pbox_exact(3.0))
    @test Za.intervals[1][1] ≈ 5.0
    @test Za.intervals[1][2] ≈ 5.0
end

# §2.3 Fréchet dependent case — Lemma A.4
@testset "AddPBox — Fréchet dependent (Lemma A.4)" begin
    X = pbox_interval(0.0, 0.5, 1.0)
    Y = pbox_interval(0.0, 0.5, 1.0)
    X2, Y2 = mark_dependent(X, Y, 1)
    Z = add_pbox(X2, Y2)

    wX = width(X)
    wY = width(Y)
    bound = frechet_width_bound(wX, wY)
    @test width(Z) <= bound + 1e-9   # Lemma A.4 satisfied

    # Fréchet sum must be wider than independent sum
    Z_indep = add_pbox(X, Y)
    @test width(Z) >= width(Z_indep) - 1e-9
end

# MulPBox
@testset "MulPBox" begin
    X = pbox_interval(0.5, 1.0, 1.0)
    Y = pbox_interval(0.8, 1.0, 1.0)
    Z = mul_pbox(X, Y)
    lo, hi = Z.intervals[1]
    @test lo ≈ 0.4   # 0.5 * 0.8
    @test hi ≈ 1.0   # 1.0 * 1.0
end

# WidenPBox
@testset "WidenPBox" begin
    pb = pbox_interval(0.5, 0.5, 1.0)   # point mass
    wide = widen_pbox(pb, 1.1)
    lo, hi = wide.intervals[1]
    @test lo < 0.5
    @test hi > 0.5
    @test widen_pbox(pb, 1.0) == pb   # no-op when factor=1
end

# merge_overlapping
@testset "merge_overlapping" begin
    pb = PBox([(0.0, 0.3), (0.2, 0.5), (0.8, 1.0)], [0.4, 0.3, 0.3], 1.0, BitVector())
    merged = merge_overlapping(pb)
    @test length(merged.intervals) == 2   # first two overlap
    @test merged.intervals[1] == (0.0, 0.5)
    @test merged.probabilities[1] ≈ 0.7
end

# sample_from_pbox
@testset "sample_from_pbox" begin
    pb = pbox_interval(0.0, 1.0, 1.0)
    for _ in 1:100
        s = sample_from_pbox(pb)
        @test 0.0 <= s <= 1.0
    end
    # Point mass must always return same value
    pb2 = pbox_exact(0.42)
    @test sample_from_pbox(pb2) ≈ 0.42
end

# §7 Theoretical guarantees
@testset "Theorem A.2 — error_composition_bound" begin
    widths = [0.01, 0.02, 0.01]
    bound = error_composition_bound(widths)
    @test bound >= sum(widths)   # always ≥ linear term
    @test bound > 0.0
end

@testset "Lemma A.5 — hoeffding_bound + hoeffding_epsilon" begin
    # Bound decreases as n increases (all with same t)
    @test hoeffding_bound(10_000, 0.1) < hoeffding_bound(100, 0.1)
    # Bound ≤ 2.0 for any n ≥ 1 (since 2·exp(x) ≤ 2 for x ≤ 0)
    @test hoeffding_bound(10, 0.01) <= 2.0
    # Large t → very small bound
    @test hoeffding_bound(100, 0.5) < 0.01

    # Epsilon inversely scales with √n (Lemma A.5: ε = √(ln(2/δ)/2n))
    ε1 = hoeffding_epsilon(100, 0.05)
    ε2 = hoeffding_epsilon(400, 0.05)
    @test ε2 < ε1            # more samples → smaller epsilon
    @test ε2 ≈ ε1 / 2 atol=0.01   # ε ∝ 1/√n: 4× samples → ε/2
end

# ── MASS CONSERVATION ON DEPENDENT COMBINATIONS (regression, 2026-08-25) ─────────────────────
#
# The dependent branches assign each pair its pointwise Fréchet upper bound `min(px,py)`, which is
# attainable pointwise but NOT simultaneously — so the masses summed ABOVE 1. Measured 1.6 on two
# multi-interval p-boxes sharing a correlation bit. §2.3 requires "the total mass must sum
# correctly". INVISIBLE on single-interval inputs (`min(1,1) == 1*1`), which is why it survived.
@testset "dependent combinations conserve mass (was summing to 1.6)" begin
    sig = falses(64); sig[7] = true
    X  = PBox([(0.1,0.3),(0.5,0.7)], [0.4,0.6], 1.0, copy(sig))
    Y  = PBox([(0.2,0.4),(0.6,0.8)], [0.3,0.7], 1.0, copy(sig))
    Xi = PBox(X.intervals, X.probabilities, X.confidence, BitVector())
    Yi = PBox(Y.intervals, Y.probabilities, Y.confidence, BitVector())

    @test are_dependent(X, Y)
    for (label, r) in (("mul dependent", mul_pbox(X, Y)), ("add dependent", add_pbox(X, Y)))
        @test sum(r.probabilities) ≈ 1.0 atol=1e-9      # label kept for failure output
        @test r.confidence ≈ 1.0 atol=1e-9
    end

    # the INDEPENDENT branch must be untouched, and must NOT be rescaled:
    @test sum(mul_pbox(Xi, Yi).probabilities) ≈ 1.0 atol=1e-9

    # 🔴 PARTIAL MASS MUST SURVIVE. Rescaling the independent branch would inflate a p-box with
    # confidence 0.8 up to 1.0 — manufacturing mass. This asserts we did not do that.
    Xp = PBox([(0.1,0.3)], [0.8], 0.8, BitVector())
    Yp = PBox([(0.2,0.4)], [1.0], 1.0, BitVector())
    @test sum(mul_pbox(Xp, Yp).probabilities) ≈ 0.8 atol=1e-9

    # dependent_masses is a no-op when already normalised
    @test dependent_masses([0.4, 0.6]) == [0.4, 0.6]
    @test sum(dependent_masses([0.8, 0.8])) ≈ 1.0 atol=1e-12
end
