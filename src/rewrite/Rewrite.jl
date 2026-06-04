"""
Rewrite — reorder source patterns inside MORK exec/rule conjunction lists.

A MORK program atom takes one of two forms:

  (exec id (, src1 ... srcN) (, tpl1 ... tplM))
  ((rule-head ...) (, src1 ... srcN) (O ...))

In both cases, position 2 (1-indexed) is a `,` conjunction list whose items
are the source patterns fed to ProductZipper.  By sorting those sources from
most- to least-selective, we reduce the effective Cartesian-product fan-out
that ProductZipper must traverse.

Two rewrite modes:

  reorder_static(node)     — uses static_score (no Space required).
  reorder_dynamic(btm, node) — uses dynamic_count (O(1) per source).

Both preserve the `,` head token and leave templates untouched.
"""

# ── Conjunction reordering ────────────────────────────────────────────────────

"""
    reorder_conjunction_static(conj::SList) -> SList

Return a new `(, ...)` list whose source children are sorted by static_score.
"""
function reorder_conjunction_static(conj::SList)::SList
    items = conj.items
    head = items[1]           # the `,` SAtom
    sources = items[2:end]
    length(sources) <= 1 && return conj

    scores = static_score.(sources)
    perm = sortperm(scores; alg=MergeSort)   # stable — preserve ties
    SList([head; sources[perm]])
end

# ── Atom-level rewriting ──────────────────────────────────────────────────────

"""
    reorder_atom_static(node::SNode) -> SNode

If `node` is a compound whose second item is a `,` list, reorder that list.
Otherwise return `node` unchanged.
"""
function reorder_atom_static(node::SNode)::SNode
    node isa SList || return node
    items = (node::SList).items
    length(items) < 3 && return node
    is_conjunction(items[2]) || return node
    new_conj = reorder_conjunction_static(items[2]::SList)
    SList([items[1], new_conj, items[3:end]...])
end

# ── Program-level rewriting ───────────────────────────────────────────────────

"""
    reorder_program_static(program::AbstractString) -> String

Reorder all conjunction lists in `program` using the static heuristic.
"""
function reorder_program_static(program::AbstractString)::String
    nodes = parse_program(program)
    sprint_program(SNode[reorder_atom_static(n) for n in nodes])
end

"""
    source_order_report(program::AbstractString) -> String

Human-readable report showing original and reordered source lists for all
multi-source conjunction lists in `program` (static mode).
"""
function source_order_report(program::AbstractString)::String
    io = IOBuffer()
    nodes = parse_program(program)
    for node in nodes
        node isa SList || continue
        items = (node::SList).items
        length(items) < 3 || !is_conjunction(items[2]) && continue
        conj = items[2]::SList
        sources = conj.items[2:end]
        length(sources) <= 1 && continue

        scores = static_score.(sources)
        perm = sortperm(scores; alg=MergeSort)

        println(io, "atom: ", sprint_sexpr(items[1]))
        for (k, s) in enumerate(sources)
            println(io, "  src[$k] score=$(round(scores[k]; digits=3))  ", sprint_sexpr(s))
        end
        println(io, "  → reordered: ", join(sprint_sexpr.(sources[perm]), "  "))
    end
    String(take!(io))
end

export reorder_conjunction_static
export reorder_atom_static
export reorder_program_static
export source_order_report
