# §3.2-§3.4 STV demand sensitivity gate (PLN step 4, part 1).
#
# Verifies the closed-form block sensitivities (sens_*) against the OPERATOR ∞-NORM of the
# finite-difference Jacobian of the spec's §3.4 sensitivity-model forward maps — NOT PLNBook
# (the §3.4 confidence bookkeeping is deliberately simplified; this table is the demand
# CONTROLLER, distinct from the inference oracle). So the gate is self-checking: a transcription
# error in sens_* shows up as a mismatch with the FD Jacobian of the formula it claims to be.

using Test
using MorkSupercompiler

# The §3.4 forward maps (fwd_hmp/conjunction/disjunction/negation/inversion/deduction/induction/
# abduction) are now in src (PLNDemand) — the §4.5 forward-supply family — and reused here as the
# maps the closed-form sensitivities are the Jacobian of (single source, no duplicate).

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

@testset "§3.4 rows-only sensitivities — INTERIOR FD-agreement" begin
    # SCOPE: this validates transcription of the SPEC CONTROLLER table against the FD Jacobian
    # of the §3.4 (simplified-confidence) map. It is NOT a cross-check against PLNBook — the
    # §3.4 confidence bookkeeping deliberately differs from book, so there is no inference-oracle
    # agreement here (unlike 3b's forward maps). FD is only meaningful in the SMOOTH interior;
    # the boundary is asserted separately below.
    let p = [0.5, 0.9, 0.4, 0.8, 0.7, 0.9], s = sens_inversion(0.5, 0.9, 0.4, 0.8, 0.7, 0.9)
        @test isapprox(
            s[1], fd_block_infnorm((a...) -> fwd_inversion(a...), p, 1); atol=1e-3
        )
        @test isapprox(
            s[2], fd_block_infnorm((a...) -> fwd_inversion(a...), p, 2); atol=1e-3
        )
        @test isapprox(
            s[3], fd_block_infnorm((a...) -> fwd_inversion(a...), p, 3); atol=1e-3
        )
    end
    let p = [0.4, 0.8, 0.6, 0.85, 0.7, 0.9, 0.5, 0.85],
        s = sens_deduction(0.4, 0.8, 0.6, 0.85, 0.7, 0.9, 0.5, 0.85)

        for i in 1:4
            @test isapprox(
                s[i], fd_block_infnorm((a...) -> fwd_deduction(a...), p, i); atol=1e-3
            )
        end
    end
end

@testset "§3.4 rows-only sensitivities — BOUNDARY clamp (FD-meaningless; asserts chosen value)" begin
    # At the singularity FD and the analytic partial diverge TOGETHER, so FD proves nothing
    # here — these assertions pin the CHOSEN clamp behavior directly (bounded cap, NOT empty).
    # Inversion sA→0: raw ∂s/∂sA = sBA·sB/sA² → ∞ ⇒ capped, and demand CONCENTRATES (not killed).
    let s = sens_inversion(1.0e-5, 0.9, 0.4, 0.8, 0.7, 0.9)
        @test s[1] == SENS_CAP                                   # capped to a finite value, no Inf/NaN
        let psi = demand_adjoint(1.0, s, (0.9, 0.8, 0.9))
            @test psi[1] > 0.0                                   # demand NOT killed at the singularity
            @test isapprox(psi[1], 1.0 - 0.9; atol=1e-6)         # sens̄_A→1 ⇒ demand ≈ need_A (high)
        end
    end
    # Deduction sB→1: raw ∂s/∂sB ∝ 1/(1−sB)² → ∞ ⇒ same cap decision.
    let s = sens_deduction(1.0 - 1.0e-5, 0.8, 0.6, 0.85, 0.7, 0.9, 0.5, 0.85)
        @test s[1] == SENS_CAP
    end
end

@testset "§3.4 induction/abduction — INTERIOR FD-agreement (chain rule vs closed-form)" begin
    # Strength-dominated interior point so the FD check actually exercises the CHAIN RULE
    # (not just the product confidence). Composition (sens_*) vs FD of the closed eq 14/15.
    let p = [0.6, 0.8, 0.4, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8],
        s = sens_induction(0.6, 0.8, 0.4, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8)

        for i in 1:5
            @test isapprox(
                s[i], fd_block_infnorm((a...) -> fwd_induction(a...), p, i); atol=1e-3
            )
        end
    end
    let p = [0.6, 0.8, 0.4, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8],
        s = sens_abduction(0.6, 0.8, 0.4, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8)

        # FD-exercise ALL 5 premises INCLUDING A: the i=1 check PERTURBS sA (and cA) and confirms
        # the forward map's ∂/∂A block is genuinely zero — TESTING sens_A=0, not just asserting it
        # (a dropped-sA transcription error would otherwise be invisible: sens=0, no complaint).
        # The genuine sA-independence is corroborated three ways: eq 15, the inversion∘deduction
        # composition, and lib/pln Truth_Abduction all lack any sA term.
        for i in 1:5
            @test isapprox(
                s[i], fd_block_infnorm((a...) -> fwd_abduction(a...), p, i); atol=1e-3
            )
        end
        @test s[1] == 0.0   # the expected value: premise A drops out of eq 15
    end
end

@testset "§3.4 induction/abduction — BOUNDARY clamp (chosen value, FD-meaningless)" begin
    # Induction sA→0 (inversion intermediate sAB=sBA·sB/sA): premise A's strength sens → ∞ ⇒ capped.
    let s = sens_induction(1.0e-5, 0.8, 0.4, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8)
        @test s[1] == SENS_CAP
    end
    # Abduction has TWO singular boundaries in sB (inversion sBC=sCB·sC/sB at sB→0; deduction
    # 1/(1−sB) at sB→1). sens_B caps at both.
    let s0 = sens_abduction(1.0e-5, 0.8, 1.0e-5, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8),
        s1 = sens_abduction(0.6, 0.8, 1.0 - 1.0e-5, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8)

        @test s0[2] == SENS_CAP   # sB → 0
        @test s1[2] == SENS_CAP   # sB → 1
    end
end

@testset "per-factor rule tagging: FactorNode.rule + rule_sensitivity dispatch" begin
    # Backward-compat: the new `rule` field defaults to :none, so existing constructors are unbroken.
    @test FactorNode(:x, :premise).rule === :none                       # var node, untagged
    @test FactorNode(:f, :factor; is_factor=true).rule === :none        # factor, no rule given
    @test FactorNode(:f, :factor; is_factor=true, rule=:deduction).rule === :deduction

    # rule_sensitivity dispatches a factor's tag to its §3.4 sens_*, unpacking premises in role
    # order. Verified == the direct sens_* call → all 8 verified sensitivities reachable via a tag.
    @test rule_sensitivity(:hmp, [(0.8, 0.9), (0.7, 0.85)]) == sens_hmp(0.8, 0.9, 0.7, 0.85)
    @test rule_sensitivity(:conjunction, [(0.8, 0.9), (0.7, 0.85)]) ==
        sens_conjunction(0.8, 0.9, 0.7, 0.85)
    @test rule_sensitivity(:disjunction, [(0.8, 0.9), (0.7, 0.85)]) ==
        sens_disjunction(0.8, 0.9, 0.7, 0.85)
    @test rule_sensitivity(:negation, [(0.7, 0.85)]) == sens_negation()
    @test rule_sensitivity(:inversion, [(0.5, 0.9), (0.4, 0.8), (0.7, 0.9)]) ==
        sens_inversion(0.5, 0.9, 0.4, 0.8, 0.7, 0.9)
    @test rule_sensitivity(
        :deduction, [(0.4, 0.8), (0.6, 0.85), (0.7, 0.9), (0.5, 0.85)]
    ) ==
        sens_deduction(0.4, 0.8, 0.6, 0.85, 0.7, 0.9, 0.5, 0.85)
    let prem = [(0.6, 0.8), (0.4, 0.85), (0.6, 0.9), (0.5, 0.85), (0.5, 0.8)]
        @test rule_sensitivity(:induction, prem) ==
            sens_induction(0.6, 0.8, 0.4, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8)
        @test rule_sensitivity(:abduction, prem) ==
            sens_abduction(0.6, 0.8, 0.4, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8)
    end

    # A factor whose demand is computed MUST name a rule — :none / unknown errors loudly.
    @test_throws ErrorException rule_sensitivity(:none, [(0.8, 0.9), (0.7, 0.85)])
    @test_throws ErrorException rule_sensitivity(:bogus, [(0.8, 0.9)])
end

@testset "pbox_to_stv round-trips stv_to_pbox" begin
    let (s, c) = pbox_to_stv(stv_to_pbox(0.7, 0.85))
        @test isapprox(s, 0.7; atol=1e-9)
        @test isapprox(c, 0.85; atol=1e-9)
    end
end

@testset "compute_demand_field — edge-label-order NON-CIRCULAR validation (deduction)" begin
    # The premise-order convention (rule_sensitivity unpacks [premise_1, premise_2, …]) is only
    # validated if premises arrive in GRAPH EDGE-LABEL order and the result matches a sensitivity
    # hand-computed with SEMANTIC assignment (B in the sB slot, AB in the sAB slot, …). Deduction
    # is asymmetric, so a premise transposition changes the answer — it can't hide.
    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    g = FactorGraph(t)
    B, C, AB, BC = (0.4, 0.8), (0.6, 0.85), (0.7, 0.9), (0.5, 0.85)   # distinct ⇒ asymmetric
    for (n, stv) in ((:B, B), (:C, C), (:AB, AB), (:BC, BC))
        g.var_nodes[n] = FactorNode(n, :premise)
        g.var_nodes[n].message = stv_to_pbox(stv...)
    end
    g.var_nodes[:AC] = FactorNode(:AC, :conclusion)
    g.factor_nodes[:ded] = FactorNode(:ded, :factor; is_factor=true, rule=:deduction)
    # Edges added in SCRAMBLED order — extraction must sort by role_label, not insertion order.
    push!(g.edges, FactorEdge(:AB, :ded, :premise_3))
    push!(g.edges, FactorEdge(:AC, :ded, :conclusion))
    push!(g.edges, FactorEdge(:BC, :ded, :premise_4))
    push!(g.edges, FactorEdge(:B, :ded, :premise_1))
    push!(g.edges, FactorEdge(:C, :ded, :premise_2))

    active, dem = compute_demand_field(:AC, g)

    # HAND-COMPUTED expected: sens with the SEMANTIC assignment, demand from d_v=1 at :AC.
    sens = sens_deduction(B..., C..., AB..., BC...)   # (sB,cB, sC,cC, sAB,cAB, sBC,cBC)
    psi = demand_adjoint(1.0, sens, (B[2], C[2], AB[2], BC[2]))
    @test isapprox(dem[:B], psi[1]; atol=1e-9)
    @test isapprox(dem[:C], psi[2]; atol=1e-9)
    @test isapprox(dem[:AB], psi[3]; atol=1e-9)
    @test isapprox(dem[:BC], psi[4]; atol=1e-9)
    @test dem[:AC] == 1.0                              # seed

    # NON-CIRCULAR BITE: link (AB) vs term (B) demands differ under deduction → a transposition
    # in extraction would land demand on the wrong node and break the asserts above.
    @test !isapprox(dem[:AB], dem[:B]; atol=1e-6)
    # (a) contract: the support set is IDENTICAL to the unweighted BFS — compute-and-attach only.
    @test active == MorkSupercompiler._backward_demand_expansion(:AC, g, 1000)
end

@testset "compute_demand_field — HMP smoke + untagged factor errors" begin
    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    g = FactorGraph(t)
    g.var_nodes[:A] = FactorNode(:A, :premise);
    g.var_nodes[:A].message = stv_to_pbox(0.8, 0.9)
    g.var_nodes[:AB] = FactorNode(:AB, :premise);
    g.var_nodes[:AB].message = stv_to_pbox(0.7, 0.85)
    g.var_nodes[:B] = FactorNode(:B, :conclusion)
    g.factor_nodes[:mp] = FactorNode(:mp, :factor; is_factor=true, rule=:hmp)
    push!(g.edges, FactorEdge(:A, :mp, :premise_1))
    push!(g.edges, FactorEdge(:AB, :mp, :premise_2))
    push!(g.edges, FactorEdge(:B, :mp, :conclusion))
    let (_, dem) = compute_demand_field(:B, g)
        psi = demand_adjoint(1.0, sens_hmp(0.8, 0.9, 0.7, 0.85), (0.9, 0.85))
        @test isapprox(dem[:A], psi[1]; atol=1e-9)
        @test isapprox(dem[:AB], psi[2]; atol=1e-9)
    end
    # An UNTAGGED factor (:none) on the demand path errors loudly (no silent wrong answer).
    g.factor_nodes[:mp].rule = :none
    @test_throws ErrorException compute_demand_field(:B, g)
end

@testset "forward_supply — Mammal/Lassie worked example (§6.1) → marginal (0.8935, 0.456)" begin
    # The chainer COMPUTING: a 2-step zero-default HMP chain Collie(Lassie) → Dog(Lassie) →
    # Mammal(Lassie). Forward supply over the activated subgraph must reproduce the spec's headline.
    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    g = FactorGraph(t)
    g.var_nodes[:A] = FactorNode(:A, :premise)     # Collie(Lassie)
    g.var_nodes[:A].message = stv_to_pbox(0.95, 0.95)
    g.var_nodes[:AB] = FactorNode(:AB, :premise)   # Collie(x)→Dog(x)  — high-s/low-c: exercises the pbox round-trip fix
    g.var_nodes[:AB].message = stv_to_pbox(0.99, 0.80)
    g.var_nodes[:BC] = FactorNode(:BC, :premise)   # Dog(x)→Mammal(x)
    g.var_nodes[:BC].message = stv_to_pbox(0.95, 0.60)
    g.var_nodes[:B] = FactorNode(:B, :conclusion)  # Dog(Lassie) — unknown
    g.var_nodes[:B].message = stv_to_pbox(0.5, 0.0)
    g.var_nodes[:C] = FactorNode(:C, :conclusion)  # Mammal(Lassie) — query
    g.var_nodes[:C].message = stv_to_pbox(0.5, 0.0)
    g.factor_nodes[:f1] = FactorNode(:f1, :factor; is_factor=true, rule=:hmp)
    g.factor_nodes[:f2] = FactorNode(:f2, :factor; is_factor=true, rule=:hmp)
    push!(g.edges, FactorEdge(:A, :f1, :premise_1))
    push!(g.edges, FactorEdge(:AB, :f1, :premise_2))
    push!(g.edges, FactorEdge(:B, :f1, :conclusion))
    push!(g.edges, FactorEdge(:B, :f2, :premise_1))
    push!(g.edges, FactorEdge(:BC, :f2, :premise_2))
    push!(g.edges, FactorEdge(:C, :f2, :conclusion))

    _, marg = forward_supply(:C, g)   # pi_b = 0 (zero-default HMP, §6.1)
    @test all(isapprox.(marg[:B], (0.9405, 0.76); atol=1e-4))    # intermediate Dog(Lassie)
    @test all(isapprox.(marg[:C], (0.8935, 0.456); atol=1e-4))   # the headline marginal Mammal(Lassie)
end

@testset "§3.4 forward maps — direct VALUE check (the maps, not just their Jacobians)" begin
    # The FD sensitivity tests cross-check each map's DERIVATIVE; the HMP/deduction/inversion graph
    # tests check three maps' VALUES. This pins the VALUE of ALL EIGHT directly — closing the gap
    # that conjunction/disjunction/negation/induction/abduction had NO value check at all (a
    # value-preserving-derivative error in fwd_disjunction would otherwise slip every green).
    @test all(isapprox.(fwd_hmp(0.8, 0.9, 0.7, 0.85; pi_b=0.0), (0.56, 0.765); atol=1e-9))
    @test all(isapprox.(fwd_conjunction(0.8, 0.9, 0.7, 0.85), (0.56, 0.985); atol=1e-9))      # c=1−(.1)(.15)
    @test all(isapprox.(fwd_disjunction(0.8, 0.9, 0.7, 0.85), (0.94, 0.985); atol=1e-9))      # s=.8+.7−.56
    @test all(isapprox.(fwd_negation(0.7, 0.85), (0.3, 0.85); atol=1e-9))
    @test all(
        isapprox.(
            fwd_inversion(0.8, 0.9, 0.5, 0.85, 0.99, 0.80), (0.61875, 0.612); atol=1e-6
        )
    )
    @test all(
        isapprox.(
            fwd_deduction(0.4, 0.8, 0.6, 0.85, 0.99, 0.80, 0.5, 0.85), (0.501667, 0.4624);
            atol=1e-5
        )
    )
    @test all(
        isapprox.(
            fwd_induction(0.6, 0.8, 0.4, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8),
            (0.61111, 0.5202);
            atol=1e-4
        )
    )
    @test all(
        isapprox.(
            fwd_abduction(0.6, 0.8, 0.4, 0.85, 0.6, 0.9, 0.5, 0.85, 0.5, 0.8),
            (0.625, 0.5202);
            atol=1e-4
        )
    )
end

# ── Forward-inference CORRECTNESS close-out: the non-HMP rules through forward_supply on a real
#    graph, with ADVERSARIAL inputs (clamp/transpose) + INTERMEDIATE assertions. Before this, only
#    HMP (Mammal/Lassie) had an end-to-end forward test; the other 7 were dispatch-reachable but
#    path-unverified — and forward_supply reads 4-5 premises through pbox_to_stv (HMP reads 2), so
#    a 5-input round-trip/clamp bug would hide. See feedback_adversarial_test_inputs.

@testset "forward_supply — deduction→HMP chain (clamp + scramble + intermediate)" begin
    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    g = FactorGraph(t)
    B, C, AB, BC = (0.4, 0.8), (0.6, 0.85), (0.99, 0.80), (0.5, 0.85)   # AB CLAMPS (s=0.99,c=0.80)
    XY = (0.9, 0.7)
    for (n, v) in ((:B, B), (:C, C), (:AB, AB), (:BC, BC), (:XY, XY))
        g.var_nodes[n] = FactorNode(n, :premise)
        g.var_nodes[n].message = stv_to_pbox(v...)
    end
    g.var_nodes[:X] = FactorNode(:X, :conclusion)   # deduction conclusion = HMP premise (intermediate)
    g.var_nodes[:Y] = FactorNode(:Y, :conclusion)   # query
    g.factor_nodes[:f1] = FactorNode(:f1, :factor; is_factor=true, rule=:deduction)
    g.factor_nodes[:f2] = FactorNode(:f2, :factor; is_factor=true, rule=:hmp)
    # f1 deduction premise edges added SCRAMBLED — extraction must sort by role_label.
    push!(g.edges, FactorEdge(:AB, :f1, :premise_3))
    push!(g.edges, FactorEdge(:B, :f1, :premise_1))
    push!(g.edges, FactorEdge(:X, :f1, :conclusion))
    push!(g.edges, FactorEdge(:BC, :f1, :premise_4))
    push!(g.edges, FactorEdge(:C, :f1, :premise_2))
    push!(g.edges, FactorEdge(:X, :f2, :premise_1))
    push!(g.edges, FactorEdge(:XY, :f2, :premise_2))
    push!(g.edges, FactorEdge(:Y, :f2, :conclusion))

    _, marg = forward_supply(:Y, g)

    # HAND-COMPUTED with SEMANTIC assignment (B in sB slot, AB in sAB slot, …):
    X = fwd_deduction(B..., C..., AB..., BC...)   # (0.501667, 0.4624)
    Y = fwd_hmp(X..., XY...; pi_b=0.0)            # (0.45150, 0.32368)
    @test all(isapprox.(marg[:X], X; atol=1e-6))   # INTERMEDIATE pinned (not just terminal)
    @test all(isapprox.(marg[:Y], Y; atol=1e-6))   # final marginal
    @test isapprox(marg[:X][1], 0.501667; atol=1e-5)   # explicit value (AB clamp recovered → 0.99)
    # ASYMMETRY: deduction's term (B) and link (AB) slots are NOT interchangeable — swapping
    # premise_1↔premise_3 gives a different conclusion, so a mis-extraction would break marg[:X].
    @test !all(
        isapprox.(
            fwd_deduction(B..., C..., AB..., BC...), fwd_deduction(AB..., C..., B..., BC...)
        )
    )
end

@testset "forward_supply — inversion factor (clamp + scramble + asymmetric denominator)" begin
    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    g = FactorGraph(t)
    A, Bv, BA = (0.8, 0.9), (0.5, 0.85), (0.99, 0.80)   # BA CLAMPS; sA in the denominator ⇒ asymmetric
    for (n, v) in ((:A, A), (:Bv, Bv), (:BA, BA))
        g.var_nodes[n] = FactorNode(n, :premise)
        g.var_nodes[n].message = stv_to_pbox(v...)
    end
    g.var_nodes[:AB] = FactorNode(:AB, :conclusion)   # query
    g.factor_nodes[:inv] = FactorNode(:inv, :factor; is_factor=true, rule=:inversion)
    push!(g.edges, FactorEdge(:BA, :inv, :premise_3))   # scrambled
    push!(g.edges, FactorEdge(:A, :inv, :premise_1))
    push!(g.edges, FactorEdge(:AB, :inv, :conclusion))
    push!(g.edges, FactorEdge(:Bv, :inv, :premise_2))

    _, marg = forward_supply(:AB, g)
    expected = fwd_inversion(A..., Bv..., BA...)   # (0.99·0.5/0.8, 0.9·0.85·0.80) = (0.61875, 0.612)
    @test all(isapprox.(marg[:AB], expected; atol=1e-6))
    @test isapprox(marg[:AB][1], 0.61875; atol=1e-6)   # BA clamp recovered → 0.99 in the numerator
    # ASYMMETRY: sA is the denominator, sBA the numerator — swapping A↔BA changes the answer.
    @test !all(
        isapprox.(fwd_inversion(A..., Bv..., BA...), fwd_inversion(BA..., Bv..., A...))
    )
end

# ── §4.7 demand-gated walk (gated_demand_expansion) — the THREE-graph acceptance gate. Sharpened
#    so a pass MEANS the scheduler prunes CORRECTLY (selects the query-relevant subgraph), not
#    merely harmlessly. Small verified graphs are FRIENDLY INPUTS for the scheduler (nothing prunes),
#    so each test forces the behavior under test.

function _mammal_lassie_graph()
    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    g = FactorGraph(t)
    g.var_nodes[:A] = FactorNode(:A, :premise)
    g.var_nodes[:A].message = stv_to_pbox(0.95, 0.95)
    g.var_nodes[:AB] = FactorNode(:AB, :premise)
    g.var_nodes[:AB].message = stv_to_pbox(0.99, 0.80)
    g.var_nodes[:BC] = FactorNode(:BC, :premise)
    g.var_nodes[:BC].message = stv_to_pbox(0.95, 0.60)
    g.var_nodes[:B] = FactorNode(:B, :conclusion)
    g.var_nodes[:B].message = stv_to_pbox(0.5, 0.0)
    g.var_nodes[:C] = FactorNode(:C, :conclusion)
    g.var_nodes[:C].message = stv_to_pbox(0.5, 0.0)
    g.factor_nodes[:f1] = FactorNode(:f1, :factor; is_factor=true, rule=:hmp)
    g.factor_nodes[:f2] = FactorNode(:f2, :factor; is_factor=true, rule=:hmp)
    push!(g.edges, FactorEdge(:A, :f1, :premise_1))
    push!(g.edges, FactorEdge(:AB, :f1, :premise_2))
    push!(g.edges, FactorEdge(:B, :f1, :conclusion))
    push!(g.edges, FactorEdge(:B, :f2, :premise_1))
    push!(g.edges, FactorEdge(:BC, :f2, :premise_2))
    push!(g.edges, FactorEdge(:C, :f2, :conclusion))
    return g
end

@testset "§4.7 gated walk (1) — NO-OP where it shouldn't prune (gated_active == ungated_active)" begin
    # Sibling ⇒ the ungated path can't regress (untouched). The real claim is that the GATED walk
    # returns the SAME support set when nothing should prune (τ=0). Marginal match is a corollary.
    g = _mammal_lassie_graph()
    ungated = MorkSupercompiler._backward_demand_expansion(:C, g, 1000)
    gated, _ = gated_demand_expansion(:C, g; tau_expand=0.0)
    @test gated == ungated
    _, marg = forward_supply(:C, g; active=gated)
    @test all(isapprox.(marg[:C], (0.8935, 0.456); atol=1e-4))   # corollary of equal active sets
end

@testset "§4.7 gated walk (2) — pruned the RIGHT thing (low dropped, high kept, marginal holds)" begin
    # Q ← conjunction(X,Y); X ← neg(Xn), Y ← neg(Yn). X high-conf ⇒ LOW demand; Y low-conf ⇒ HIGH
    # demand. The negation derivations are REDUNDANT (Xn=neg(X)) so pruning fx leaves X — and the
    # marginal — unchanged. Asserts the LOW branch was pruned and the HIGH branch kept, not just
    # "something pruned + nothing broke" (which would pass even if a query-relevant node were dropped).
    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    g = FactorGraph(t)
    g.var_nodes[:X] = FactorNode(:X, :premise)
    g.var_nodes[:X].message = stv_to_pbox(0.9, 0.95)     # high-conf ⇒ low demand
    g.var_nodes[:Y] = FactorNode(:Y, :premise)
    g.var_nodes[:Y].message = stv_to_pbox(0.6, 0.3)      # low-conf ⇒ high demand
    g.var_nodes[:Xn] = FactorNode(:Xn, :premise)
    g.var_nodes[:Xn].message = stv_to_pbox(0.1, 0.95)    # neg(X) ⇒ fx is redundant
    g.var_nodes[:Yn] = FactorNode(:Yn, :premise)
    g.var_nodes[:Yn].message = stv_to_pbox(0.4, 0.3)
    g.var_nodes[:Q] = FactorNode(:Q, :conclusion)
    g.var_nodes[:Q].message = stv_to_pbox(0.5, 0.0)
    g.factor_nodes[:f1] = FactorNode(:f1, :factor; is_factor=true, rule=:conjunction)
    g.factor_nodes[:fx] = FactorNode(:fx, :factor; is_factor=true, rule=:negation)
    g.factor_nodes[:fy] = FactorNode(:fy, :factor; is_factor=true, rule=:negation)
    push!(g.edges, FactorEdge(:X, :f1, :premise_1))
    push!(g.edges, FactorEdge(:Y, :f1, :premise_2))
    push!(g.edges, FactorEdge(:Q, :f1, :conclusion))
    push!(g.edges, FactorEdge(:Xn, :fx, :premise_1))
    push!(g.edges, FactorEdge(:X, :fx, :conclusion))
    push!(g.edges, FactorEdge(:Yn, :fy, :premise_1))
    push!(g.edges, FactorEdge(:Y, :fy, :conclusion))

    ungated = MorkSupercompiler._backward_demand_expansion(:Q, g, 1000)
    gated, dem = gated_demand_expansion(:Q, g; tau_expand=0.20)
    @test dem[:X] < 0.20 && dem[:Y] >= 0.20      # X correctly deprioritized, Y kept
    @test :Xn in ungated                          # the pruning is REAL (Xn is reachable)
    @test !(:Xn in gated)                         # LOW branch pruned (fx + Xn dropped)
    @test :Yn in gated                            # HIGH branch retained (fy walked to Yn)
    _, mg = forward_supply(:Q, g; active=gated)
    _, mu = forward_supply(:Q, g; active=ungated)
    @test all(isapprox.(mg[:Q], mu[:Q]; atol=1e-9))          # marginal unchanged by pruning
    @test all(isapprox.(mg[:Q], (0.54, 0.965); atol=1e-6))   # = conjunction(X, Y)
end

@testset "§4.7 gated walk (3) — DC#2 boundary retention is LOAD-BEARING (delete-it-fails-this)" begin
    # Mammal/Lassie at τ=0.20: §6.1 — dem(A)=0.05, dem(A→B)=0.192 both < 0.20 ⇒ expansion HALTS at
    # A and A→B. A=(0.95,0.95) high-conf ⇒ tiny need ⇒ deprioritized — BUT its strength SEEDS the
    # whole forward chain, so DC#2 must retain its edge or forward supply starves.
    g = _mammal_lassie_graph()
    gated, dem = gated_demand_expansion(:C, g; tau_expand=0.20)
    @test isapprox(dem[:A], 0.05; atol=1e-3)     # A halted (demand < τ)
    @test :A in gated && :AB in gated            # DC#2 RETAINED the halted boundaries
    _, marg = forward_supply(:C, g; active=gated)
    @test all(isapprox.(marg[:C], (0.8935, 0.456); atol=1e-4))   # marginal survives the halt

    # DC#2 OFF (retain_boundary=false): the halted boundaries are DROPPED → f1 loses A/AB → forward
    # supply can't fire f1 → C is never computed. Retention is the load-bearing term: remove it and
    # THIS marginal breaks (and only this — tests 1/2 don't halt-then-need a dropped node the same way).
    g2 = _mammal_lassie_graph()
    gated_off, _ = gated_demand_expansion(:C, g2; tau_expand=0.20, retain_boundary=false)
    @test !(:A in gated_off)                     # A dropped without DC#2
    _, marg_off = forward_supply(:C, g2; active=gated_off)
    @test !haskey(marg_off, :C)                  # marginal unreachable without boundary retention
end
