"""
Selectivity — estimate how many atoms in a Space match a given source pattern.

Two strategies:

  static_score(src)       — pure static heuristic (no Space needed).
                            Returns a Float64 in [0,1]: 0 = most selective.
                            Based on variable fraction: ground atoms score 0,
                            fully-variable atoms score 1.

  dynamic_count(btm, src) — count atoms in `btm` whose encoded prefix matches
                            the head symbol + arity of `src`.  O(1) PathMap
                            lookup.  Returns an Int; lower = more selective.
"""

using PathMaps: read_zipper_at_path, zipper_val_count
using MORK: ExprArity, ExprSymbol, item_byte

# ── Static ────────────────────────────────────────────────────────────────────

"""
    static_score(src::SNode) -> Float64

Heuristic selectivity in [0.0, 1.0].  Lower = more selective.

  - Ground atom (0 vars):      0.0  (always ≤1 match)
  - Partially variable:        n_vars / (n_vars + n_syms)
  - Fully variable (0 syms):   1.0
"""
function static_score(src::SNode)::Float64
    nv = count_vars(src)
    nv == 0 && return 0.0
    na = count_atoms(src)
    na == 0 && return 1.0
    nv / (nv + na)
end

# ── Dynamic ───────────────────────────────────────────────────────────────────

"""
    dynamic_count(btm::PathMap{UnitVal}, src::SNode) -> Int

Count atoms in `btm` whose head arity + head symbol match `src`.

For `(parity \$i \$p)` (arity=3, head="parity"):
prefix = [arity_byte(3), sym_size_byte(6), 'p','a','r','i','t','y']

Returns `typemax(Int)` if the head cannot be encoded (too long, nested head, etc.)
so that unencodable sources sort last (least selective).
"""
function dynamic_count(btm, src::SNode)::Int
    src isa SList || return 1          # bare atom/var: treat as 1 match
    isempty((src::SList).items) && return 0
    items = (src::SList).items
    arity = length(items)
    arity > 63 && return typemax(Int)

    head = items[1]

    if head isa SAtom
        sym = (head::SAtom).name
        nb = length(sym)
        nb > 63 && return typemax(Int)
        prefix = Vector{UInt8}(undef, 2 + nb)
        prefix[1] = item_byte(ExprArity(UInt8(arity)))
        prefix[2] = item_byte(ExprSymbol(UInt8(nb)))
        copyto!(prefix, 3, codeunits(sym), 1, nb)

    elseif head isa SList
        # compound head, e.g. `((step \$k) \$p0 \$t0)`
        # encode only [outer_arity, inner_arity] as prefix (rough but O(1))
        h_arity = length((head::SList).items)
        h_arity > 63 && return typemax(Int)
        prefix = [item_byte(ExprArity(UInt8(arity))), item_byte(ExprArity(UInt8(h_arity)))]
    else
        return typemax(Int)   # variable head — unencodable
    end

    # \U0001f534 EXTEND THE PREFIX THROUGH LEADING *GROUND* ARGUMENTS.
    #
    # Spec Algorithm 3 (§5.1.3) says `prefix <- pattern.to_prefix(depth=2)` — arity byte + head
    # symbol, stop. That cap is NOT arbitrary and NOT a defect: Algorithm 3 is PrefixSampling. It
    # draws `min(sample_size, sqrt(space.size))` samples and scales, so a deeper prefix means FEWER
    # samples and worse variance. Depth 2 is a SAMPLING BUDGET.
    #
    # PathMap removes the reason for it. `read_zipper_at_path` + `zipper_val_count` is an EXACT
    # subtrie count at ANY depth — this function already relies on that (no bootstrap_variance, no
    # scaling). The depth bound was a consequence of sampling, so it goes with the sampling.
    #
    # WHY IT MATTERS, MEASURED 2026-08-24: at depth 2, `(ccr CY \$i)` and `(fcr PPL \$i)` BOTH count
    # their whole relation — 57,686 each. The planner sees them as equally selective, has nothing to
    # reorder, and stats collection buys an estimate carrying ZERO information. Extended, they read
    # 26 and 32,310, and the join order becomes obvious.
    #
    # ⚠️ THIS ONLY HELPS A WELL-ENCODED QUERY, and the two levers compose. `(cc \$i CY)` has the
    # VARIABLE first, so there are no leading ground arguments to extend through and the prefix stops
    # at the head regardless — the constant is a suffix and neither the planner nor this change can
    # see it. Argument order is a schema decision (see workflows/encoding_lint.jl); conjunct order is
    # this function's. Good encoding is what gives the planner something to work with.
    #
    # Also strictly closer to Algorithm 2 (§5.1.2), which explicitly multiplies by
    # `argument_selectivity[(pred, i)]` per constrained position — the spec knows ground arguments
    # affect cardinality. This gets it exactly instead of from a histogram with a 0.5 default.
    for arg in items[2:end]
        enc = _encode_ground(arg)
        enc === nothing && break        # first non-ground argument ends the seekable prefix
        append!(prefix, enc)
    end

    rz = read_zipper_at_path(btm, prefix)
    zipper_val_count(rz)
end

"""
    _encode_ground(n::SNode) -> Union{Vector{UInt8}, Nothing}

MORK byte encoding of a FULLY GROUND s-expression, or `nothing` if it contains any variable.

A ground compound is one CONTIGUOUS byte run (verified against the encoder 2026-08-24:
`(count YES_NO)` appears byte-identical inside `(r a (count YES_NO))`), so a nested ground term
extends a seekable prefix exactly like a symbol does.
"""
# 🔴 EXACTLY THREE METHODS. A FOURTH SILENTLY REINTRODUCES RUNTIME DISPATCH.
#
# THE MECHANISM, VERIFIED NOT ASSUMED (`code_typed optimize=true` on `_encode_ground(::SList)`:
# 16 `:invoke`, ZERO dynamic calls to `_encode_ground`):
#   At the recursive call site `it` is statically `SNode`, which matches all three methods below.
#   Julia's `max_methods` limit is 3, so inference enumerates the MATCHES, emits a branch chain,
#   and every arm gets a static call. That is METHOD-MATCH SPLITTING.
#
# ⚠️ IT IS *NOT* "SNode has three concrete subtypes so inference enumerates them" — an earlier
# version of this comment said exactly that and it is WRONG. Abstract types are OPEN; a new subtype
# can be defined at any time, so the compiler can never close over them. Nothing here depends on how
# many SUBTYPES SNode has. It depends on how many METHODS match.
#
# ⇒ Adding `_encode_ground(::SVar)` — the obvious tidy-up, since the fallback reads like an
#   omission — takes the match count to 4, splitting fails, and the dispatch returns. NO TEST WILL
#   BREAK. AllocCheck catches it only while `dynamic_count` stays on the checked list.
#
# ⇒ The `::SNode` fallback is therefore STRUCTURAL, not an SVar convenience: it is what makes the
#   split total so no residual dynamic arm is emitted. Do not split it into per-type methods.
#
# The before-state was ONE method branching on `n isa SAtom` / `n isa SList` — a hand-rolled tag
# test, not dispatch at all, which AllocCheck flagged as "Dynamic dispatch to function
# _encode_ground". Multiple dispatch (the language FEATURE) is what was added here; dynamic dispatch
# (the compilation OUTCOME) is what went away. Those are different things and the report is about
# the second.
#
# 🔑 THE FRAGILITY-FREE FIX, not done here because it is a breaking AST change: make SNode a
# CLOSED UNION — `const SNode = Union{SAtom, SList, SVar}` — which does not depend on a method-count
# limit, and additionally gives `SList.items::Vector{SNode}` a union eltype instead of the abstract
# one it has today (measured). All three structs declare `<: SNode`, so that touches the whole
# frontend. Filed as FRAGILITY, not polish.
_encode_ground(::SNode)::Union{Vector{UInt8}, Nothing} = nothing

function _encode_ground(n::SAtom)::Union{Vector{UInt8}, Nothing}
    sym = n.name
    nb = length(sym)
    nb > 63 && return nothing
    out = Vector{UInt8}(undef, 1 + nb)
    out[1] = item_byte(ExprSymbol(UInt8(nb)))
    copyto!(out, 2, codeunits(sym), 1, nb)
    out
end

function _encode_ground(n::SList)::Union{Vector{UInt8}, Nothing}
    its = n.items
    # ⚠️ TWO DELIBERATE DIVERGENCES FROM UPSTREAM, both verified against
    # `frontend/src/bytestring_parser.rs` on 2026-08-24 — neither is a port gap:
    #
    # 1. `()` IS LEGAL UPSTREAM and we decline it. Its `b'('` branch does `write_arity(0)` then
    #    increments per element, so for `()` the element loop never runs and it emits the single
    #    byte `[0]` — ground, encodable, and maximally discriminating as a prefix. Our `isempty`
    #    guard is CONSERVATIVE. It costs precision only on `()` conjuncts, which no corpus program
    #    has. Recorded as a DECISION, not an oversight.
    #
    # 2. `> 63` IS STRICTER THAN UPSTREAM, and upstream is the one that is wrong. It grows the arity
    #    byte via `item_byte(Tag::Arity(a + 1))` and the only bound is a `debug_assert!` — A NO-OP
    #    IN RELEASE. A 64-element list writes `0b0100_0000`, a RESERVED discriminant `byte_item`
    #    cannot decode, and the panic surfaces LATER and ELSEWHERE on a byte already in the trie.
    #    REPRODUCED against the shipped release binary: 63 elements loads, 64 panics `reserved 64`
    #    at `expr/src/lib.rs:137`. Written up in
    #    `docs/upstream-issues/MORK-arity-64-parser-overflow.md`. Declining here is CORRECT.
    (isempty(its) || length(its) > 63) && return nothing
    out = UInt8[item_byte(ExprArity(UInt8(length(its))))]
    for it in its
        e = _encode_ground(it)
        e === nothing && return nothing      # any variable anywhere makes it non-ground
        append!(out, e)
    end
    out
end

export static_score, dynamic_count
