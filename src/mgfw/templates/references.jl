"""
Reference implementations for §15.4 MVP acceptance criteria.

Spec §15.4 demo 2: "STV factor path returns same result as reference interpreter"
Spec §15.4 demo 3: "Trie miner returns same top-k motifs as naive reference on toy dataset"

Without a separate canonical reference, the registered template lowerings
have no oracle to diff against. This module provides pure-Julia references
that callers can use as the "truth" side of the MVP acceptance comparison.

The references are deliberately simple — no MORK Space, no metta_calculus,
no caching. Their job is to encode the spec's formulas explicitly so that
tests can run both `template.lowering(t, region)` (which emits MeTTa
rewrite rules to be executed by MORK) AND the reference (pure Julia) on
the same input, then compare.

  - `stv_mp_reference(s_a, c_a, s_imp, c_imp) → (s_b, c_b)` — spec §10.1
    HeuristicModusPonens forward map for STV
  - `naive_top_k_motifs(atoms, k) → Vector{Tuple{String,Int}}` — spec §10.3
    motif mining via direct enumeration + sort
"""

# ── §10.1 PLN STV HeuristicModusPonens reference ───────────────────────────────

"""
    stv_mp_reference(s_a, c_a, s_imp, c_imp) → (s_b, c_b)

**Family: APPROXIMATE — MG-Framework §10.1.2 HeuristicModusPonens.** This is the
*heuristic* MP that `specialize_approximate` uses (strength multiplies; confidence is
the weaker of the two scaled by the §10.1.2 standard 0.9 `adjoint-need` decay). It is
deliberately NOT the book PLN ModusPonens — for the faithful family (the contract for
the EXACT rules) see the `PLNBook` submodule below.

It matches the formula emitted by `pln_stv_lowering` exactly, so a test can diff this
against the lowered rule's *executed* output. (NB: as of 2026-06-15 that executed diff
has never run — the `:where` lowering is inert on a bare MORK space; see
test/mgfw/test_pln_reference.jl OPEN ITEM 1 / PLN step 3a.)

Inputs: `s_a, c_a` (premise A); `s_imp, c_imp` (`(implies A B)`). Output: `(s_b, c_b)` for B.
"""
function stv_mp_reference(s_a::Real, c_a::Real, s_imp::Real, c_imp::Real)
    s_b = s_a * s_imp
    c_b = min(c_a, c_imp) * 0.9
    (s_b, c_b)
end

# ── §10.1 PLN BOOK truth-value family (the EXACT-rule contract) ────────────────
"""
    PLNBook — faithful analytic transcription of lib/pln/pln_core_logic.metta.

The second of two reference families in this module (the first being the APPROXIMATE
`stv_mp_reference` above). `PLNBook` is the **book PLN** contract that the mgfw forward
maps (FactorGeometry) are made faithful against — analytic (not a call into Core) so it
is immune to Core's live `Truth_w2c` Channel leak (CognitiveSubstratesAI/Core issue #1).
Each fn cites its lib/pln source line. `/safe` returns `nothing` (the MeTTa `(empty)`) at
the singular boundary — NOT a clamped number — so callers see "no truth value" exactly
where lib/pln would. Pinned to the lib/pln doctest goldens by test_pln_reference.jl.

Induction/abduction are intentionally absent here — they ship in lib/pln with no doctest
`→`, so their goldens must come from live execution (or hand-derivation if issue #1's leak
is live); added in PLN step 3c.
"""
module PLNBook

# pln_core_logic.metta:40-43 — (/safe A B) = (if (> B 0) (/ A B) (empty))
safe_div(a, b) = b > 0.0 ? a / b : nothing
# pln_core_logic.metta:46-47 — (negate x) = (- 1 x)
negate(x) = 1.0 - x
# pln_core_logic.metta:176-177 — Truth_c2w(c) = /safe c (1-c)
c2w(c) = safe_div(c, 1.0 - c)
# pln_core_logic.metta:180-181 — Truth_w2c(w) = /safe w (w+1)   [⚠ Core issue #1 leak root]
w2c(w) = safe_div(w, w + 1.0)
# pln_core_logic.metta:70-71 — Truth_or(a,b) = 1 - (1-a)(1-b)
t_or(a, b) = 1.0 - (1.0 - a) * (1.0 - b)

# pln_core_logic.metta:249-251 — Truth_ModusPonens (book §5.7.1)
modus_ponens(sP, cP, sPQ, cPQ) = (sP * sPQ + 0.02 * (1.0 - sP), w2c(cP * cPQ))

# pln_core_logic.metta:259-263 — Truth_SymmetricModusPonens (snotAB=0.2)
function symmetric_modus_ponens(sA, cA, sAB, cAB)
    snotAB = 0.2
    s = sA * sAB + snotAB * negate(sA) * (1.0 + sAB)
    c = cA * cAB * t_or(sA, sAB)
    (s, c)
end

# pln_core_logic.metta:283-284 — Truth_Negation: s = 1−s; c unchanged
negation(s, c) = (1.0 - s, c)

# pln_core_logic.metta:292-295 — Truth_inversion (B, AB): s = ABs; c = Bc·ABc·0.6
inversion(sB, cB, sAB, cAB) = (sAB, cB * cAB * 0.6)

# pln_core_logic.metta:272-279 — Truth_Revision
function revision(f1, c1, f2, c2)
    w1 = c2w(c1)
    w2 = c2w(c2)
    w = w1 + w2
    f = safe_div(w1 * f1 + w2 * f2, w)
    f === nothing && return (nothing, nothing)
    (min(1.0, f), min(1.0, max(c1, c2, w2c(w))))
end

# pln_core_logic.metta:192-213 — Truth_Deduction (book §1.4), 5-input.
# SINGULAR at Qs→1 (guarded: Qs>0.9999 ⇒ Rs); biting boundary is the consistency
# precondition → (1 0) fallback. s = Qs>0.9999 ? Rs : PQs·QRs + /safe((1−PQs)(Rs−Qs·QRs),1−Qs)
function deduction(sP, cP, sQ, cQ, sR, cR, sPQ, cPQ, sQR, cQR)
    cons(as, bs, abs_) =
        (0 < as) && (clamp((as + bs - 1) / as, 0, 1) <= abs_ <= clamp(bs / as, 0, 1))
    (cons(sP, sQ, sPQ) && cons(sQ, sR, sQR)) || return (1.0, 0.0)
    s = sQ > 0.9999 ? sR : begin
        d = safe_div((1.0 - sPQ) * (sR - sQ * sQR), 1.0 - sQ)
        d === nothing ? nothing : sPQ * sQR + d
    end
    (s, sPQ * sQR * cP * cQR)
end

# pln_core_logic.metta:216-224 — Truth_Induction (book App. A), 5-input.
# s = /safe(sBA·sBC·sB, sA) + (1 − /safe(sBA·sB, sA))·/safe(sC−sB·sBC, 1−sB)
# c = w2c(sBC·cBC·cBA).  SINGULAR at sA→0 (and sB→1). Goldens have NO lib/pln doctest →
# they are HAND-DERIVED independently in test_pln_reference.jl (not from this eval).
function induction(sA, cA, sB, cB, sC, cC, sBA, cBA, sBC, cBC)
    t1 = safe_div(sBA * sBC * sB, sA)
    t2a = safe_div(sBA * sB, sA)
    t2b = safe_div(sC - sB * sBC, 1.0 - sB)
    (t1 === nothing || t2a === nothing || t2b === nothing) && return (nothing, nothing)
    (t1 + (1.0 - t2a) * t2b, w2c(sBC * cBC * cBA))
end

# pln_core_logic.metta:227-236 — Truth_Abduction (book App. A), 5-input.
# s = /safe(sAB·sCB·sC, sB) + /safe(sC·(1−sAB)·(1−sCB), 1−sB);  c = w2c(sAB·cAB·cCB).
# SINGULAR at sB→0 and sB→1.  (sA, cA, cB, cC are unused by the book formula.)
function abduction(sA, cA, sB, cB, sC, cC, sAB, cAB, sCB, cCB)
    t1 = safe_div(sAB * sCB * sC, sB)
    t2 = safe_div(sC * (1.0 - sAB) * (1.0 - sCB), 1.0 - sB)
    (t1 === nothing || t2 === nothing) && return (nothing, nothing)
    (t1 + t2, w2c(sAB * cAB * cCB))
end

end  # module PLNBook

# ── §10.3 Trie motif miner reference ───────────────────────────────────────────

"""
    naive_top_k_motifs(atoms::AbstractVector{<:AbstractString}, k::Int)
        → Vector{Pair{String,Int}}

Direct, non-MORK reference for spec §10.3 trie miner. For each atom in
`atoms`, take the first token (whitespace-separated) as the "motif" symbol;
count how many times each motif appears; return the top-k as a
descending-sorted list of `motif => count` Pairs.

This is what `motif_miner_lowering`'s emitted MeTTa rewrite rules should
produce in aggregate, after MORK executes the seed → grow → score cascade.
For MVP acceptance: a test fixture supplies a toy dataset (e.g. ["foo a",
"foo b", "bar c", "foo d", "bar e"]) and asserts that:

  - This reference returns [("foo", 3), ("bar", 2)] when k=2.
  - The lowering's emitted MeTTa contains all three pattern-matching stages
    (motif-stage 1 / 2 / 3) so a MORK execution WOULD reproduce these counts.

A direct end-to-end MORK execution comparison is queued for a future
session once the trie-geometry runtime (§15.2 deliverable 6) is fully
wired through MGFW `mg_run!`.
"""
function naive_top_k_motifs(atoms::AbstractVector{<:AbstractString}, k::Int)
    counts = Dict{String, Int}()
    for atom in atoms
        # First whitespace-separated token = "motif symbol"
        tokens = split(strip(atom))
        isempty(tokens) && continue
        motif = String(tokens[1])
        counts[motif] = get(counts, motif, 0) + 1
    end
    # Sort descending by count, take top-k
    pairs = collect(counts)
    sort!(pairs; by=p -> -p[2])
    pairs[1:min(k, length(pairs))]
end

# ── §4.1 GeodesicBGC priority function reference (workload #2) ────────────────

"""
    bgc_forward_f(adj, start, target_depth) → Dict{Node, Float64}

Forward reachability per spec §4.1: f(x, t) = "how easy is x to reach from
`start`". Concrete instantiation on a directed graph: f(x) = number of
distinct paths from `start` to `x` of length ≤ `target_depth`. Reachable
nodes get f > 0; unreachable get 0 (intentionally NOT in the dict so
log(f) ill-defined cases are caught at the priority step).

`adj`: Dict{Node, Vector{Node}} adjacency list (outgoing edges).
"""
function bgc_forward_f(adj::Dict{N, Vector{N}}, start::N, target_depth::Int) where {N}
    f = Dict{N, Float64}(start => 1.0)   # 1 trivial path of length 0
    frontier = [start]
    for _ in 1:target_depth
        next_frontier = N[]
        for u in frontier
            outs = get(adj, u, N[])
            for v in outs
                f[v] = get(f, v, 0.0) + f[u]   # accumulate path counts
                push!(next_frontier, v)
            end
        end
        isempty(next_frontier) && break
        frontier = next_frontier
    end
    f
end

"""
    bgc_backward_g(adj, goal, target_depth) → Dict{Node, Float64}

Backward usefulness per spec §4.1: g(x, t) = "how easy is x to continue
to `goal`". Operates on the reversed graph — for each node x, count paths
from x to `goal` of length ≤ `target_depth`.
"""
function bgc_backward_g(adj::Dict{N, Vector{N}}, goal::N, target_depth::Int) where {N}
    # Reverse the adjacency
    rev = Dict{N, Vector{N}}()
    for (u, vs) in adj, v in vs
        push!(get!(() -> N[], rev, v), u)
    end
    bgc_forward_f(rev, goal, target_depth)
end

"""
    bgc_priority(f, g, x; step_cost=1.0, prev_x=nothing) → Float64

Spec §4.1 priority function used by GeodesicBGC-Worklist:
priority(x) = Δ(log f(x) + log g(x)) / step_cost

`Δ` = change vs the previous frontier element. If `prev_x === nothing`,
returns the raw log f + log g (initial frontier item). Returns -Inf for
nodes with f=0 or g=0 (unreachable from start or to goal — should not be
on the frontier).

The spec's GeodesicBGC-Worklist (§12.2) consumes this priority: pop highest
first, expand. The function is the heart of the composite's
"least-effort path" semantic.
"""
function bgc_priority(
    f::AbstractDict, g::AbstractDict, x; step_cost::Real=1.0, prev_x=nothing
)::Float64
    fx = get(f, x, 0.0)
    gx = get(g, x, 0.0)
    (fx <= 0.0 || gx <= 0.0) && return -Inf
    curr = log(fx) + log(gx)
    prev_x === nothing && return curr / step_cost
    fp = get(f, prev_x, 0.0);
    gp = get(g, prev_x, 0.0)
    (fp <= 0.0 || gp <= 0.0) && return curr / step_cost
    prev = log(fp) + log(gp)
    (curr - prev) / step_cost
end

export stv_mp_reference, naive_top_k_motifs, PLNBook
export bgc_forward_f, bgc_backward_g, bgc_priority
