# PLNDemand.jl — §3.2-§3.4 STV demand sensitivity (PLN backward-chaining factor-graph spec).
#
# The principled §4.4 demand-control-field input — the CANONICAL SENSITIVITY TABLE
# (spec §3.4 / line 589: "adopt as canonical sensitivity table"). It supersedes the orphan
# `stv_backward_demand` (a sqrt heuristic never wired into the activation) as the per-premise
# `need` term that step 4 part 2 feeds to `_backward_demand_expansion` (Finding A: that BFS
# is a structural support set with no demand weights — this is the weight layer).
#
# SYSTEM OF RECORD (decided 2026-06-15): the §3.4 simplified-confidence maps are the runtime
# family for BOTH forward inference (forward_supply, §4.5) AND demand sensitivity — one coherent
# model (§3.4 preserves lib/pln STRENGTHS verbatim, simplifies CONFIDENCE, e.g. HMP c_out=cA·cAB
# not book w2c(cA·cI)). The lib/pln-book maps (FactorGeometry stv_* + PLNBook) are the lib/pln
# FAITHFULNESS REFERENCE — they validate that §3.4's strengths match lib/pln — NOT the runtime
# inference family. (Earlier "inference uses PLNBook" was imprecise — corrected.)
#
# This file: §3.2-§3.3 framework + Eq-1 demand adjoint; the 4 CLOSED-FORM sensitivities (HMP
# eqs 4-5, conjunction 22-23, disjunction 25-26, negation=1); all 4 ROWS-ONLY ∞-norms —
# deduction/inversion (eqs 6-13) and induction/abduction (eqs 14-16, chain-rule composition);
# per-factor `rule_sensitivity` dispatch; and `compute_demand_field` — the §3.3 demand field
# over `_backward_demand_expansion` (COMPUTE-AND-ATTACH: seed d_v=1, Eq-1 backward, max-join;
# the support set is unchanged — activation/gating is §4.4 (b)/§4.7, deferred). The orphan
# `stv_backward_demand` has been retired in favor of this.

# §3.2 — information need (confidence deficit) of a premise.
need_stv(c::Real) = 1.0 - c

# §3.4 closed-form block sensitivities  sens_f(i) = ‖J_{f,i}‖_∞  (operator ∞-norm of the 2×2
# block ∂(s_out,c_out)/∂(s_i,c_i)). Each returns the per-premise sensitivity tuple, in order.

"HMP sensitivity (premise A, rule A→B), eqs (4),(5). `pi_b` = contextual default rate (0 = the
§6 zero-default; MUST match `fwd_hmp`'s π_b for the same factor — §6 sens_{f2}(B)=0.95 needs π_b=0)."
sens_hmp(sA, cA, sAB, cAB; pi_b::Real=0.0) = (max(abs(sAB - pi_b), cAB), max(sA, cA))

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

# ── §3.4 rows-only sensitivities (induction, abduction) — inversion∘deduction. ────────────
# eqs (14),(15) give forward STRENGTH only ("same recipe applies", no partials). Both ARE
# exact inversion-then-deduction compositions (verified: eq 14 = deduction with sAB=sBA·sB/sA;
# eq 15 = deduction with sBC=sCB·sC/sB), so the strength partials come from the CHAIN RULE
# through the inversion intermediate — and are FD-verified against the closed strength formula
# (a real composition-vs-closed-form cross-check, not FD-vs-FD). Capped at the singularities.

# Deduction strength partials at (sB,sC,sAB,sBC): (∂D/∂sB, ∂D/∂sC, ∂D/∂sAB, ∂D/∂sBC), eqs 8-11.
function _ded_strength_partials(sB, sC, sAB, sBC)
    d = 1.0 - sB
    dB = -(1.0 - sAB) * sBC / d + (1.0 - sAB) * (sC - sB * sBC) / d^2     # eq 8
    dC = (1.0 - sAB) / d                                                 # eq 9
    dAB = sBC - (sC - sB * sBC) / d                                      # eq 10
    dBC = sAB - (1.0 - sAB) * sB / d                                     # eq 11
    return (dB, dC, dAB, dBC)
end

"""
    sens_induction(sA,cA,sB,cB,sC,cC,sBA,cBA,sBC,cBC; w=1.0) -> (sens_A,sens_B,sens_C,sens_BA,sens_BC)

Induction = inversion(→ sAB = sBA·sB/sA) then deduction (eq 14). Strength partials via chain
rule (sB enters via sAB AND directly). **SPEC GAP**: eq 14 gives no induction confidence;
c_AC = cBA·cBC·cB·cC·w mirrors abduction eq 16 (flagged assumption). Singular at sA→0, sB→1.
"""
function sens_induction(sA, cA, sB, cB, sC, cC, sBA, cBA, sBC, cBC; w::Real=1.0)
    sAB = sBA * sB / sA                                   # inversion intermediate
    dB, dC, dAB, dBC = _ded_strength_partials(sB, sC, sAB, sBC)
    iA, iB, iBA = -sBA * sB / sA^2, sBA / sA, sB / sA     # ∂sAB/∂(sA,sB,sBA), eq 13
    psA, psBA = dAB * iA, dAB * iBA                       # A, BA enter only via sAB
    psB = dAB * iB + dB                                   # B enters via sAB AND directly
    psC, psBC = dC, dBC
    pcB, pcC = cBA * cBC * cC * w, cBA * cBC * cB * w     # c = cBA·cBC·cB·cC·w (cA absent)
    pcBA, pcBC = cBC * cB * cC * w, cBA * cB * cC * w
    return (
        cap_sens(psA),                                   # cA absent ⇒ block = |psA|
        max(cap_sens(psB), cap_sens(pcB)),
        max(cap_sens(psC), cap_sens(pcC)),
        max(cap_sens(psBA), cap_sens(pcBA)),
        max(cap_sens(psBC), cap_sens(pcBC))
    )
end

"""
    sens_abduction(sA,cA,sB,cB,sC,cC,sAB,cAB,sCB,cCB; w=1.0) -> (sens_A,sens_B,sens_C,sens_AB,sens_CB)

Abduction = inversion(→ sBC = sCB·sC/sB) then deduction (eq 15). sA does NOT appear (sens_A=0).
Confidence c_AC = cAB·cCB·cB·cC·w (spec eq 16). Singular at sB→0 (inversion) AND sB→1 (deduction).
"""
function sens_abduction(sA, cA, sB, cB, sC, cC, sAB, cAB, sCB, cCB; w::Real=1.0)
    sBC = sCB * sC / sB                                   # inversion intermediate
    dB, dC, dAB, dBC = _ded_strength_partials(sB, sC, sAB, sBC)
    jB, jC, jCB = -sCB * sC / sB^2, sCB / sB, sC / sB     # ∂sBC/∂(sB,sC,sCB)
    psB = dB + dBC * jB                                   # B enters directly AND via sBC
    psC = dC + dBC * jC                                   # C enters directly AND via sBC
    psAB, psCB = dAB, dBC * jCB
    pcB, pcC = cAB * cCB * cC * w, cAB * cCB * cB * w     # c = cAB·cCB·cB·cC·w (cA absent)
    pcAB, pcCB = cCB * cB * cC * w, cAB * cB * cC * w
    return (
        0.0,                                             # sA, cA both absent ⇒ premise A irrelevant
        max(cap_sens(psB), cap_sens(pcB)),
        max(cap_sens(psC), cap_sens(pcC)),
        max(cap_sens(psAB), cap_sens(pcAB)),
        max(cap_sens(psCB), cap_sens(pcCB))
    )
end

# ── Per-factor rule dispatch ──────────────────────────────────────────────────────────────
"""
    rule_sensitivity(rule::Symbol, premises; pi_b=0.02, w=1.0) -> NTuple

Dispatch a factor's PLN `rule` tag (FactorNode.rule) to its §3.4 sensitivity function,
unpacking `premises` — the per-premise STVs `(s,c)` in role order (:premise_1, :premise_2, …)
— into that function's positional args. This is what makes all 8 verified sensitivities
reachable from a graph (vs the template-level single rule). Premise-order CONVENTION per rule
(matches the sens_* signatures exactly):

    :hmp         → [A, A→B]
    :conjunction → [1, 2]            :disjunction → [1, 2]
    :negation    → [premise]         (sens = 1 regardless of the premise value)
    :inversion   → [A, B, B→A]
    :deduction   → [B, C, A→B, B→C]
    :induction   → [A, B, C, B→A, B→C]
    :abduction   → [A, B, C, A→B, C→B]

Errors on `:none`/unknown — a factor whose demand is computed MUST name a PLN rule.
"""
function rule_sensitivity(
    rule::Symbol, premises::AbstractVector{<:Tuple{Real, Real}}; pi_b::Real=0.0, w::Real=1.0
)
    # pi_b default MUST be 0.0 to match rule_forward / sens_hmp / §6.1 — the demand controller
    # and forward supply for the SAME HMP factor must share π_b (else gating computes with a
    # different rate than inference; §6.1's dem(A→B)=0.192 + sens_{f1}(A)=0.99 both need π_b=0).
    if rule === :hmp
        (sA, cA), (sAB, cAB) = premises
        return sens_hmp(sA, cA, sAB, cAB; pi_b=pi_b)
    elseif rule === :conjunction
        (s1, c1), (s2, c2) = premises
        return sens_conjunction(s1, c1, s2, c2)
    elseif rule === :disjunction
        (s1, c1), (s2, c2) = premises
        return sens_disjunction(s1, c1, s2, c2)
    elseif rule === :negation
        return sens_negation()
    elseif rule === :inversion
        (sA, cA), (sB, cB), (sBA, cBA) = premises
        return sens_inversion(sA, cA, sB, cB, sBA, cBA; w_inv=w)
    elseif rule === :deduction
        (sB, cB), (sC, cC), (sAB, cAB), (sBC, cBC) = premises
        return sens_deduction(sB, cB, sC, cC, sAB, cAB, sBC, cBC; w_ded=w)
    elseif rule === :induction
        (sA, cA), (sB, cB), (sC, cC), (sBA, cBA), (sBC, cBC) = premises
        return sens_induction(sA, cA, sB, cB, sC, cC, sBA, cBA, sBC, cBC; w=w)
    elseif rule === :abduction
        (sA, cA), (sB, cB), (sC, cC), (sAB, cAB), (sCB, cCB) = premises
        return sens_abduction(sA, cA, sB, cB, sC, cC, sAB, cAB, sCB, cCB; w=w)
    else
        error("rule_sensitivity: factor must name a PLN rule, got :$(rule)")
    end
end

# ── §3.3 demand field over the BFS support set (part 3 — COMPUTE-AND-ATTACH) ────────────────

"""
    pbox_to_stv(pb::PBox) -> (s, c)

Recover an STV `(strength, confidence)` from a PBox — the inverse of `stv_to_pbox`, which sets
`[s−hw, s+hw]` with `hw=(1−c)/2` and CLAMPS to [0,1]. The midpoint is NOT a faithful inverse
when an endpoint was clamped (e.g. s=0.99,c=0.80 → hi clamped to 1 → midpoint 0.945 ≠ 0.99). So
recover `s` from the UNCLAMPED bound plus the half-width: `s = lo + hw` if lo>0, else `hi − hw`
if hi<1, else the midpoint (both clamped ⇒ c→0, s≈0.5). Confidence is stored exactly.
"""
function pbox_to_stv(pb::PBox)
    c = pb.confidence
    hw = (1.0 - c) / 2.0
    lo, hi = pb.intervals[1][1], pb.intervals[end][2]
    s = lo > 0.0 ? lo + hw : (hi < 1.0 ? hi - hw : (lo + hi) / 2.0)
    return (clamp(s, 0.0, 1.0), c)
end

"""
    compute_demand_field(query, graph, budget=1000) -> (active, dem::Dict{Symbol,Float64})

§3.3 backward demand over the EXISTING BFS support set (COMPUTE-AND-ATTACH, not gating —
the activation payoff is §4.4 (b)/§4.7). `active` is IDENTICAL to `_backward_demand_expansion`;
this only ATTACHES per-node demand weights for a (deferred) scheduler to read. Seed `d_v=1` at
`query`; walk conclusion→premise; for each factor apply Eq (1) via its tagged rule
(`rule_sensitivity`) to push demand to its premises, accumulated by MAX-JOIN over consumers.

Premises are taken in role_label order (:premise_1, :premise_2, …) → the `rule_sensitivity`
positional convention. Single backward pass: EXACT on trees; reconvergent/cyclic accumulation
needs the §4.4 damped relaxation (deferred with the full control field). Factors must be tagged
(`FactorNode.rule`); an untagged (:none) factor errors loudly via `rule_sensitivity`.
"""
function compute_demand_field(query::Symbol, graph::FactorGraph, budget::Int=1000)
    active = _backward_demand_expansion(query, graph, budget)
    dem = Dict{Symbol, Float64}(query => 1.0)
    frontier = [query]
    steps = 0
    while !isempty(frontier) && steps < budget
        v = popfirst!(frontier)
        steps += 1
        d_v = get(dem, v, 0.0)
        for e in graph.edges
            (e.var_node == v && e.role_label === :conclusion) || continue
            fnode = get(graph.factor_nodes, e.factor_node, nothing)
            fnode === nothing && continue
            prem_edges = sort(
                [
                    pe for pe in graph.edges if
                    pe.factor_node === e.factor_node && pe.role_label !== :conclusion
                ];
                by=pe -> string(pe.role_label)
            )
            isempty(prem_edges) && continue
            prem_stvs = [
                pbox_to_stv(graph.var_nodes[pe.var_node].message) for pe in prem_edges
            ]
            sens = rule_sensitivity(fnode.rule, prem_stvs)
            confs = ntuple(i -> prem_stvs[i][2], length(prem_stvs))
            psi = demand_adjoint(d_v, sens, confs)
            for (i, pe) in enumerate(prem_edges)
                dem[pe.var_node] = max(get(dem, pe.var_node, 0.0), psi[i])
                push!(frontier, pe.var_node)
            end
        end
    end
    return (active, dem)
end

# ── §3.4 / §4.5 forward supply — the §3.4 forward maps + dispatch + Julia-direct supply ──────
# DECISION (2026-06-15): forward INFERENCE runs on the §3.4 simplified-confidence maps (per the
# spec — §4.5 + the §6 worked example use these; one coherent model with the §3.4 demand
# controller). The lib/pln-book maps (FactorGeometry stv_* + PLNBook) are the lib/pln FAITHFULNESS
# REFERENCE, NOT the runtime forward family. Each map computes a conclusion STV from premise STVs.
# (Same maps the sensitivity gate FD-verifies against — single source.)

fwd_hmp(sA, cA, sAB, cAB; pi_b=0.0) = (sA * sAB + pi_b * (1.0 - sA), cA * cAB)       # eq 2 (π_b=0 §6)
fwd_conjunction(s1, c1, s2, c2) = (s1 * s2, 1.0 - (1.0 - c1) * (1.0 - c2))            # eq 21
fwd_disjunction(s1, c1, s2, c2) = (s1 + s2 - s1 * s2, c1 + c2 - c1 * c2)              # eq 24
fwd_negation(s, c) = (1.0 - s, c)                                                     # §3.4
fwd_inversion(sA, cA, sB, cB, sBA, cBA; w=1.0) = (sBA * sB / sA, cA * cB * cBA * w)   # eq 6/12
function fwd_deduction(sB, cB, sC, cC, sAB, cAB, sBC, cBC; w=1.0)                      # eq 6/7
    return (sAB * sBC + (1.0 - sAB) * (sC - sB * sBC) / (1.0 - sB), cB * cC * cAB * cBC * w)
end
function fwd_induction(sA, cA, sB, cB, sC, cC, sBA, cBA, sBC, cBC; w=1.0)              # eq 14
    s = sBA * sBC * sB / sA + (1.0 - sBA * sB / sA) * (sC - sB * sBC) / (1.0 - sB)
    return (s, cBA * cBC * cB * cC * w)
end
function fwd_abduction(sA, cA, sB, cB, sC, cC, sAB, cAB, sCB, cCB; w=1.0)              # eq 15
    s = sAB * sCB * sC / sB + sC * (1.0 - sAB) * (1.0 - sCB) / (1.0 - sB)
    return (s, cAB * cCB * cB * cC * w)
end

"""
    rule_forward(rule, premises; pi_b=0.0, w=1.0) -> (s, c)

Dispatch a factor's tagged `rule` to its §3.4 forward map, unpacking `premises` (per-premise
STVs in role order, same convention as `rule_sensitivity`) → the conclusion STV. `pi_b` defaults
to 0 (the §6 "zero-default heuristic modus ponens"). Errors on :none/unknown.
"""
function rule_forward(
    rule::Symbol, premises::AbstractVector{<:Tuple{Real, Real}}; pi_b::Real=0.0, w::Real=1.0
)
    if rule === :hmp
        (sA, cA), (sAB, cAB) = premises
        return fwd_hmp(sA, cA, sAB, cAB; pi_b=pi_b)
    elseif rule === :conjunction
        (s1, c1), (s2, c2) = premises
        return fwd_conjunction(s1, c1, s2, c2)
    elseif rule === :disjunction
        (s1, c1), (s2, c2) = premises
        return fwd_disjunction(s1, c1, s2, c2)
    elseif rule === :negation
        (s, c) = premises[1]
        return fwd_negation(s, c)
    elseif rule === :inversion
        (sA, cA), (sB, cB), (sBA, cBA) = premises
        return fwd_inversion(sA, cA, sB, cB, sBA, cBA; w=w)
    elseif rule === :deduction
        (sB, cB), (sC, cC), (sAB, cAB), (sBC, cBC) = premises
        return fwd_deduction(sB, cB, sC, cC, sAB, cAB, sBC, cBC; w=w)
    elseif rule === :induction
        (sA, cA), (sB, cB), (sC, cC), (sBA, cBA), (sBC, cBC) = premises
        return fwd_induction(sA, cA, sB, cB, sC, cC, sBA, cBA, sBC, cBC; w=w)
    elseif rule === :abduction
        (sA, cA), (sB, cB), (sC, cC), (sAB, cAB), (sCB, cCB) = premises
        return fwd_abduction(sA, cA, sB, cB, sC, cC, sAB, cAB, sCB, cCB; w=w)
    else
        error("rule_forward: factor must name a PLN rule, got :$(rule)")
    end
end

"""
    forward_supply(query, graph, budget=1000; pi_b=0.0) -> (active, marginals::Dict{Symbol,(s,c)})

§4.5 Algorithm 2 — forward supply on the active subgraph (Julia-direct, NOT routed through MORK's
calculus). Restricts to `active = _backward_demand_expansion`, then fires factors premises→conclusion
in topological order: a factor fires when all its premises are settled (a given leaf, or an
already-computed conclusion), computing its conclusion STV via `rule_forward` and writing it back
as the node's PBox message. Returns the per-node marginals; `marginals[query]` is the answer.
Single forward pass over a DAG; cyclic graphs need §4.4 relaxation (deferred).
"""
function forward_supply(
    query::Symbol, graph::FactorGraph, budget::Int=1000; pi_b::Real=0.0, active=nothing
)
    active = active === nothing ? _backward_demand_expansion(query, graph, budget) : active
    concl = Dict{Symbol, Symbol}()
    prem = Dict{Symbol, Vector{FactorEdge}}()
    for e in graph.edges
        e.factor_node in active || continue
        if e.role_label === :conclusion
            concl[e.factor_node] = e.var_node
        else
            push!(get!(() -> FactorEdge[], prem, e.factor_node), e)
        end
    end
    concluded = Set(values(concl))
    settled = Set(v for v in active if haskey(graph.var_nodes, v) && v ∉ concluded)
    marginals = Dict{Symbol, Tuple{Float64, Float64}}(
        v => pbox_to_stv(graph.var_nodes[v].message) for v in settled
    )
    fired = Set{Symbol}()
    progress = true
    while progress && query ∉ settled
        progress = false
        for (f, cvar) in concl
            (f in fired) && continue
            pes = sort(get(prem, f, FactorEdge[]); by=pe -> string(pe.role_label))
            (isempty(pes) || !all(pe.var_node in settled for pe in pes)) && continue
            prem_stvs = [marginals[pe.var_node] for pe in pes]
            cstv = rule_forward(graph.factor_nodes[f].rule, prem_stvs; pi_b=pi_b)
            marginals[cvar] = (clamp(cstv[1], 0.0, 1.0), clamp(cstv[2], 0.0, 1.0))
            graph.var_nodes[cvar].message = stv_to_pbox(marginals[cvar]...)
            push!(settled, cvar)
            push!(fired, f)
            progress = true
        end
    end
    return (active, marginals)
end

"""
    gated_demand_expansion(query, graph, budget=1000; tau_expand=0.0, retain_boundary=true)
        -> (active, dem)

§4.7 — the DEMAND-GATED backward walk. A SIBLING to the ungated `_backward_demand_expansion`
+ `compute_demand_field` (both left intact — the verified baseline can't regress). Computes
`dem(v)` DURING the walk (Eq-1) and HALTS expansion at a premise whose demand < `tau_expand`
(expand=dem·need below threshold) — but RETAINS that premise + its edge in the active subgraph
(§4.4 Design Commitment #2), because a high-confidence boundary still SEEDS forward supply even
when backward recursion stops there. `tau_expand=0` ⇒ nothing halts ⇒ active == the ungated
support set. `retain_boundary=false` turns DC#2 OFF (drops halted boundaries) — for testing that
retention is load-bearing. Reuses `pbox_to_stv` (no second copy of the STV read).
"""
function gated_demand_expansion(
    query::Symbol,
    graph::FactorGraph,
    budget::Int=1000;
    tau_expand::Real=0.0,
    retain_boundary::Bool=true
)
    active = Set{Symbol}([query])
    dem = Dict{Symbol, Float64}(query => 1.0)
    frontier = [query]
    done = Set{Symbol}()
    steps = 0
    while !isempty(frontier) && steps < budget
        v = popfirst!(frontier)
        push!(done, v)
        steps += 1
        d_v = get(dem, v, 0.0)
        for e in graph.edges
            (e.var_node == v && e.role_label === :conclusion) || continue
            fnode = get(graph.factor_nodes, e.factor_node, nothing)
            fnode === nothing && continue
            prem_edges = sort(
                [
                    pe for pe in graph.edges if
                    pe.factor_node === e.factor_node && pe.role_label !== :conclusion
                ];
                by=pe -> string(pe.role_label)
            )
            isempty(prem_edges) && continue
            prem_stvs = [
                pbox_to_stv(graph.var_nodes[pe.var_node].message) for pe in prem_edges
            ]
            sens = rule_sensitivity(fnode.rule, prem_stvs)
            confs = ntuple(i -> prem_stvs[i][2], length(prem_stvs))
            psi = demand_adjoint(d_v, sens, confs)
            push!(active, e.factor_node)
            for (i, pe) in enumerate(prem_edges)
                u = pe.var_node
                dem[u] = max(get(dem, u, 0.0), psi[i])
                meets = psi[i] >= tau_expand
                # DC#2: retain the premise + its edge even when expansion HALTS here (unless toggled).
                if retain_boundary || meets
                    push!(active, u)
                end
                # HALT: recurse PAST u only if its demand meets the expand threshold.
                if meets && u ∉ done && u ∉ frontier
                    push!(frontier, u)
                end
            end
        end
    end
    return (active, dem)
end

export need_stv, sens_hmp, sens_conjunction, sens_disjunction, sens_negation
export normalize_sens, demand_adjoint
export SENS_CAP, sens_inversion, sens_deduction, sens_induction, sens_abduction
export rule_sensitivity, pbox_to_stv, compute_demand_field
export fwd_hmp, fwd_conjunction, fwd_disjunction, fwd_negation
export fwd_inversion, fwd_deduction, fwd_induction, fwd_abduction
export rule_forward, forward_supply, gated_demand_expansion
