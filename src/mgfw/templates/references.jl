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

Apply the spec §10.1 HeuristicModusPonens forward map under STV (Simple
Truth Value) — strength multiplies, confidence is the weaker of the two
inputs scaled by the spec's standard 0.9 confidence-decay constant.

Inputs:
  s_a, c_a       — strength and confidence of premise A
  s_imp, c_imp   — strength and confidence of `(implies A B)`

Output:
  (s_b, c_b)     — strength and confidence of derived conclusion B

This matches the formula emitted by `pln_stv_lowering` exactly, so the
test in test_mgfw.jl can diff the reference output against the lowered
rule's evaluation. The 0.9 decay is the spec §10.1.2 default for
`HeuristicModusPonens` / `adjoint-need` backward demand.
"""
function stv_mp_reference(s_a::Real, c_a::Real, s_imp::Real, c_imp::Real)
    s_b = s_a * s_imp
    c_b = min(c_a, c_imp) * 0.9
    (s_b, c_b)
end

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
    sort!(pairs; by = p -> -p[2])
    pairs[1:min(k, length(pairs))]
end

export stv_mp_reference, naive_top_k_motifs
