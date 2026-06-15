# §3.2-§3.4 STV demand sensitivity gate (PLN step 4, part 1).
#
# Verifies the closed-form block sensitivities (sens_*) against the OPERATOR ∞-NORM of the
# finite-difference Jacobian of the spec's §3.4 sensitivity-model forward maps — NOT PLNBook
# (the §3.4 confidence bookkeeping is deliberately simplified; this table is the demand
# CONTROLLER, distinct from the inference oracle). So the gate is self-checking: a transcription
# error in sens_* shows up as a mismatch with the FD Jacobian of the formula it claims to be.

using Test
using MorkSupercompiler

# ── §3.4 sensitivity-model forward maps (the maps the closed-form sens are the Jacobian of). ──
# Premises as (s,c) pairs; output (s_out, c_out).
fwd_hmp(sA, cA, sAB, cAB; pi_b=0.02) = (sA * sAB + pi_b * (1.0 - sA), cA * cAB)            # eq 2
fwd_conjunction(s1, c1, s2, c2) = (s1 * s2, 1.0 - (1.0 - c1) * (1.0 - c2))                  # eq 21
fwd_disjunction(s1, c1, s2, c2) = (s1 + s2 - s1 * s2, c1 + c2 - c1 * c2)                    # eq 24
fwd_negation(s, c) = (1.0 - s, c)

# Operator ∞-norm of the 2×2 block ∂(s_out,c_out)/∂(s_i,c_i), by central finite difference.
# `point` is the flat premise vector (s1,c1,s2,c2,…); premise `i` is 1-based.
function fd_block_infnorm(fwd, point::Vector{Float64}, i::Int; h=1e-6)
    si, ci = 2i - 1, 2i
    bump(idx, d) = (q=copy(point); q[idx] += d; collect(fwd(q...)))
    ds = (bump(si, h) .- bump(si, -h)) ./ (2h)   # (∂s_out/∂s_i, ∂c_out/∂s_i)
    dc = (bump(ci, h) .- bump(ci, -h)) ./ (2h)   # (∂s_out/∂c_i, ∂c_out/∂c_i)
    return max(abs(ds[1]) + abs(dc[1]), abs(ds[2]) + abs(dc[2]))   # max row-abs-sum
end

@testset "§3.4 closed-form sensitivities vs FD ∞-norm of the §3.4 maps" begin
    # HMP at (sA,cA,sAB,cAB) = (0.8,0.9,0.7,0.85): sens = (max(|0.7−0.02|,0.85), max(0.8,0.9))
    let p = [0.8, 0.9, 0.7, 0.85], s = sens_hmp(0.8, 0.9, 0.7, 0.85)
        @test isapprox(s[1], fd_block_infnorm((a...) -> fwd_hmp(a...), p, 1); atol=1e-4)
        @test isapprox(s[2], fd_block_infnorm((a...) -> fwd_hmp(a...), p, 2); atol=1e-4)
        @test isapprox(collect(s), [0.85, 0.9]; atol=1e-9)   # the literal closed form
    end
    # Conjunction → (max(s2,1−c2), max(s1,1−c1)) = (0.7, 0.8)
    let p = [0.8, 0.9, 0.7, 0.85], s = sens_conjunction(0.8, 0.9, 0.7, 0.85)
        @test isapprox(
            s[1], fd_block_infnorm((a...) -> fwd_conjunction(a...), p, 1); atol=1e-4
        )
        @test isapprox(
            s[2], fd_block_infnorm((a...) -> fwd_conjunction(a...), p, 2); atol=1e-4
        )
        @test isapprox(collect(s), [0.7, 0.8]; atol=1e-9)
    end
    # Disjunction → (max(1−s2,1−c2), max(1−s1,1−c1)) = (0.3, 0.2)
    let p = [0.8, 0.9, 0.7, 0.85], s = sens_disjunction(0.8, 0.9, 0.7, 0.85)
        @test isapprox(
            s[1], fd_block_infnorm((a...) -> fwd_disjunction(a...), p, 1); atol=1e-4
        )
        @test isapprox(
            s[2], fd_block_infnorm((a...) -> fwd_disjunction(a...), p, 2); atol=1e-4
        )
        @test isapprox(collect(s), [0.3, 0.2]; atol=1e-9)
    end
    # Negation → sens = 1 (Jacobian [[-1,0],[0,1]], ∞-norm = 1)
    @test isapprox(
        sens_negation()[1],
        fd_block_infnorm((a...) -> fwd_negation(a...), [0.7, 0.85], 1);
        atol=1e-4
    )
    @test sens_negation() == (1.0,)
end

@testset "§3.2-§3.3 framework: need, normalization, Eq (1) demand adjoint" begin
    @test need_stv(0.85) ≈ 0.15                                   # confidence deficit
    @test all(isapprox.(normalize_sens((0.85, 0.9)), (0.85 / 0.9, 1.0)))   # divide by S_f = max
    @test normalize_sens((0.0, 0.0)) == (0.0, 0.0)                # S_f = 0 ⇒ zeros

    # Eq (1): Ψ_i = clip(d_v · sens̄_i · (1−c_i)). HMP sens (0.85,0.9), c (0.9,0.85), d_v=1.
    let psi = demand_adjoint(1.0, (0.85, 0.9), (0.9, 0.85))
        @test isapprox(psi[1], (0.85 / 0.9) * 0.1; atol=1e-9)
        @test isapprox(psi[2], 1.0 * 0.15; atol=1e-9)
        @test all(0.0 .<= collect(psi) .<= 1.0)                   # clipped to [0,1]
    end

    # Negation passes demand through, scaled only by the premise's own need (§3.4 remark).
    @test isapprox(demand_adjoint(0.8, (1.0,), (0.7,))[1], 0.8 * 0.3; atol=1e-9)

    # Proposition 1: demand to premise i is NON-INCREASING in c_i (need = 1−c_i shrinks).
    let lo = demand_adjoint(1.0, (0.85, 0.9), (0.5, 0.85))[1],
        hi = demand_adjoint(1.0, (0.85, 0.9), (0.95, 0.85))[1]

        @test hi <= lo                                           # higher c_1 ⇒ lower demand
    end
end
