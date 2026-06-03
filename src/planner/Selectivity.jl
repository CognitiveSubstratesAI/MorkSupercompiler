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

using PathMap: read_zipper_at_path, zipper_val_count
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

    rz = read_zipper_at_path(btm, prefix)
    zipper_val_count(rz)
end

export static_score, dynamic_count
