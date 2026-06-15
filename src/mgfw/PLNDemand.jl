# PLNDemand.jl — §3.2-§3.4 STV demand sensitivity (PLN backward-chaining factor-graph spec).
#
# The principled §4.4 demand-control-field input — the CANONICAL SENSITIVITY TABLE
# (spec §3.4 / line 589: "adopt as canonical sensitivity table"). It supersedes the orphan
# `stv_backward_demand` (a sqrt heuristic never wired into the activation) as the per-premise
# `need` term that step 4 part 2 feeds to `_backward_demand_expansion` (Finding A: that BFS
# is a structural support set with no demand weights — this is the weight layer).
#
# DELIBERATE confidence divergence from PLNBook: §3.4 preserves lib/pln STRENGTHS verbatim but
# uses "simplified monotone confidence bookkeeping" for the SENSITIVITY (e.g. HMP c_out=cA·cAB,
# not book w2c(cA·cI)). So the §3.4 sensitivity forward maps are a DISTINCT family from the
# PLNBook inference oracle — same strength, simplified confidence. Inference uses PLNBook; the
# demand CONTROLLER uses this table. (Verified against the §3.4 maps, NOT PLNBook.)
#
# Part 1 (this file): §3.2-§3.3 framework + the 4 CLOSED-FORM sensitivities (HMP eqs 4-5,
# conjunction 22-23, disjunction 25-26, negation=1) + the Eq-1 demand adjoint. The 4 rows-only
# rules (deduction/inversion/induction/abduction — ∞-norm assembled from strength partials, eqs
# 8-20) and the demand field over `_backward_demand_expansion` are part 2/3.

# §3.2 — information need (confidence deficit) of a premise.
need_stv(c::Real) = 1.0 - c

# §3.4 closed-form block sensitivities  sens_f(i) = ‖J_{f,i}‖_∞  (operator ∞-norm of the 2×2
# block ∂(s_out,c_out)/∂(s_i,c_i)). Each returns the per-premise sensitivity tuple, in order.

"Heuristic modus ponens (premise A, rule A→B), eqs (4),(5). `pi_b` = contextual default rate."
sens_hmp(sA, cA, sAB, cAB; pi_b::Real=0.02) = (max(abs(sAB - pi_b), cAB), max(sA, cA))

"Conjunction (noisy-confidence), eqs (22),(23)."
sens_conjunction(s1, c1, s2, c2) = (max(s2, 1.0 - c2), max(s1, 1.0 - c1))

"Disjunction (noisy-OR), eqs (25),(26)."
sens_disjunction(s1, c1, s2, c2) = (max(1.0 - s2, 1.0 - c2), max(1.0 - s1, 1.0 - c1))

"Negation: Jacobian [[-1,0],[0,1]] ⇒ sens = 1 (§3.4)."
sens_negation() = (1.0,)

# §3.2 — normalized sensitivity: sens / S_f, S_f = max_j sens_f(j); 0 if S_f = 0.
function normalize_sens(sens::NTuple{N, Float64}) where {N}
    S = maximum(sens)
    return S > 0.0 ? sens ./ S : ntuple(_ -> 0.0, N)
end

"""
    demand_adjoint(d_v, sens, confidences) -> NTuple

§3.3 Eq (1): backward demand from a factor to each premise i —
    Ψ_i = clip_[0,1]( d_v · sens̄_f(i) · need_STV(α_i) ),   need_STV = 1 − c_i.
`sens` = per-premise raw sensitivities (pre-normalization); `confidences` = per-premise c_i.
"""
function demand_adjoint(
    d_v::Real, sens::NTuple{N, Float64}, confidences::NTuple{N, Float64}
) where {N}
    ns = normalize_sens(sens)
    return ntuple(i -> clamp(d_v * ns[i] * (1.0 - confidences[i]), 0.0, 1.0), N)
end

# ── §3.4 rows-only sensitivities (deduction, inversion) — singular, capped. ───────────────
# For these rules s_out depends only on strengths and (simplified) c_out only on confidences,
# so the 2×2 block J_{f,i} is DIAGONAL and ‖J_{f,i}‖_∞ = max(|∂s_out/∂s_i|, ∂c_out/∂c_i).
#
# SINGULARITY CLAMP (decided, NOT the forward-side /safe→empty analogue): the raw ∞-norm
# DIVERGES at the boundary (inversion sA→0: ∝ sBA·sB/sA²; deduction sB→1: ∝ 1/(1−sB)²). We
# CAP it at SENS_CAP — a large FINITE value — rather than returning 0/empty. Rationale: the
# spec (§3.4 inversion remark) says demand GROWS as sA→0 ("matching the original PLN
# error-sensitivity"), and §3.2 normalization (÷ S_f=max_j) turns a capped-maximal sensitivity
# into sens̄→1, so the singular premise absorbs the demand — which is exactly what the spec
# wants. Empty/zero would KILL demand where it should be maximal. The cap also keeps
# sens/S_f finite (no Inf/Inf=NaN). The cap's exact magnitude is normalized away; it only
# needs to exceed any interior sensitivity.
const SENS_CAP = 1.0e6
cap_sens(x::Real) = isfinite(x) ? min(abs(x), SENS_CAP) : SENS_CAP

"""
    sens_inversion(sA, cA, sB, cB, sBA, cBA; w_inv=1.0) -> (sens_A, sens_B, sens_BA)

Inversion (Bayes) block sensitivities, eqs (12)-(13). Strength partials ∂(sBA·sB/sA)/∂·;
simplified confidence c_AB = cA·cB·cBA·w_inv. **Singular at sA→0 → capped (see SENS_CAP).**
"""
function sens_inversion(sA, cA, sB, cB, sBA, cBA; w_inv::Real=1.0)
    sens_A = max(cap_sens(sBA * sB / sA^2), cap_sens(cB * cBA * w_inv))    # ∂s/∂sA = −sBA·sB/sA²
    sens_B = max(cap_sens(sBA / sA), cap_sens(cA * cBA * w_inv))          # ∂s/∂sB = sBA/sA
    sens_BA = max(cap_sens(sB / sA), cap_sens(cA * cB * w_inv))           # ∂s/∂sBA = sB/sA
    return (sens_A, sens_B, sens_BA)
end

"""
    sens_deduction(sB, cB, sC, cC, sAB, cAB, sBC, cBC; w_ded=1.0) -> (sens_B, sens_C, sens_AB, sens_BC)

Deduction block sensitivities, eqs (6)-(11). Strength partials eqs (8)-(11); simplified
confidence c_AC = cB·cC·cAB·cBC·w_ded. **Singular at sB→1 → capped (see SENS_CAP).**
"""
function sens_deduction(sB, cB, sC, cC, sAB, cAB, sBC, cBC; w_ded::Real=1.0)
    d = 1.0 - sB
    ps_B = -(1.0 - sAB) * sBC / d + (1.0 - sAB) * (sC - sB * sBC) / d^2     # eq 8
    ps_C = (1.0 - sAB) / d                                                 # eq 9
    ps_AB = sBC - (sC - sB * sBC) / d                                      # eq 10
    ps_BC = sAB - (1.0 - sAB) * sB / d                                     # eq 11
    pc_B = cC * cAB * cBC * w_ded                                          # ∂c_AC/∂cB
    pc_C = cB * cAB * cBC * w_ded
    pc_AB = cB * cC * cBC * w_ded
    pc_BC = cB * cC * cAB * w_ded
    return (
        max(cap_sens(ps_B), cap_sens(pc_B)),
        max(cap_sens(ps_C), cap_sens(pc_C)),
        max(cap_sens(ps_AB), cap_sens(pc_AB)),
        max(cap_sens(ps_BC), cap_sens(pc_BC))
    )
end

export need_stv, sens_hmp, sens_conjunction, sens_disjunction, sens_negation
export normalize_sens, demand_adjoint
export SENS_CAP, sens_inversion, sens_deduction
