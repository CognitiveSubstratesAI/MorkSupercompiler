"""
SExpr — minimal recursive-descent parser and serializer for MORK s-expressions.

Handles:
  - Atoms (symbols):   foo, bar-baz, !=, +, 42, _x
  - Variables:         \$x, \$ts, \$p0
  - Lists:             (item1 item2 ...)
  - Line comments:     ; ... to end of line

Does NOT handle strings (not needed for MORK exec/rule patterns).
"""

# ── AST ──────────────────────────────────────────────────────────────────────

abstract type SNode end

struct SAtom <: SNode
    name::String
end

struct SVar <: SNode
    name::String   # includes the leading $
end

struct SList <: SNode
    items::Vector{SNode}
end

# ── Helpers ───────────────────────────────────────────────────────────────────

# Exclusion-based per the canonical MeTTa EBNF (docs/specs/metta_grammar.ebnf):
# a WORD body is any non-whitespace char except the structural delimiters
# '(', ')', ';'.  This admits ':' (e.g. `(: foo (-> Number Number))`),
# `[`/`]`/`{`/`}`, and other punctuation the prior whitelist silently rejected.
# `"` and `#` remain symbol-chars to preserve existing behavior (this parser
# does not split strings as a separate token — see the module docstring).
_is_symbol_char(c::Char) =
    !isspace(c) && c ∉ ('(', ')', ';')

# ── Parser ────────────────────────────────────────────────────────────────────

"""
    parse_program(src::AbstractString) -> Vector{SNode}

Parse all top-level s-expressions from `src`.  Comments (`;...`) are skipped.
"""
function parse_program(src::AbstractString) :: Vector{SNode}
    nodes = SNode[]
    i = 1
    n = length(src)
    while i <= n
        i = _skip_ws(src, i, n)
        i > n && break
        node, i = _parse_at(src, i, n)
        push!(nodes, node)
    end
    nodes
end

"""
    parse_sexpr(src::AbstractString) -> SNode

Parse exactly one s-expression from the beginning of `src`.
"""
function parse_sexpr(src::AbstractString) :: SNode
    i = _skip_ws(src, 1, length(src))
    node, _ = _parse_at(src, i, length(src))
    node
end

function _skip_ws(src, i, n)
    while i <= n
        c = src[i]
        if c == ';'   # line comment
            while i <= n && src[i] != '\n'; i += 1 end
        elseif isspace(c)
            i += 1
        else
            break
        end
    end
    i
end

function _parse_at(src::AbstractString, i::Int, n::Int) :: Tuple{SNode, Int}
    i > n && error("unexpected EOF at position $i")
    c = src[i]

    if c == '('
        return _parse_list(src, i, n)
    elseif c == '$'
        return _parse_var(src, i, n)
    elseif _is_symbol_char(c)
        return _parse_atom(src, i, n)
    else
        error("unexpected character $(repr(c)) at position $i")
    end
end

function _parse_list(src, i, n)
    # consume '('
    i += 1
    items = SNode[]
    while true
        i = _skip_ws(src, i, n)
        i > n && error("unterminated list: reached EOF")
        src[i] == ')' && return SList(items), i + 1
        node, i = _parse_at(src, i, n)
        push!(items, node)
    end
end

function _parse_var(src, i, n)
    start = i
    i += 1   # skip '$'
    while i <= n && _is_symbol_char(src[i])
        i += 1
    end
    SVar(src[start:i-1]), i
end

function _parse_atom(src, i, n)
    start = i
    while i <= n && _is_symbol_char(src[i])
        i += 1
    end
    SAtom(src[start:i-1]), i
end

# ── Serializer ────────────────────────────────────────────────────────────────

"""
    sprint_sexpr(node::SNode) -> String
"""
sprint_sexpr(node::SAtom) = node.name
sprint_sexpr(node::SVar)  = node.name
function sprint_sexpr(node::SList)
    isempty(node.items) && return "()"
    io = IOBuffer()
    print(io, '(')
    for (k, item) in enumerate(node.items)
        k > 1 && print(io, ' ')
        print(io, sprint_sexpr(item))
    end
    print(io, ')')
    String(take!(io))
end

"""
    sprint_program(nodes::Vector{SNode}) -> String

Serialize a vector of top-level nodes, one per line.
"""
sprint_program(nodes::Vector{SNode}) = join(sprint_sexpr.(nodes), "\n")

# ── Utilities ─────────────────────────────────────────────────────────────────

"""Count the number of variable nodes (SVar) in a subtree."""
function count_vars(node::SNode) :: Int
    node isa SVar  && return 1
    node isa SAtom && return 0
    sum(count_vars(c) for c in (node::SList).items; init=0)
end

"""Count the number of atom nodes (SAtom) in a subtree."""
function count_atoms(node::SNode) :: Int
    node isa SAtom && return 1
    node isa SVar  && return 0
    sum(count_atoms(c) for c in (node::SList).items; init=0)
end

"""
Return true iff `node` is a `,` conjunction list (the pattern list in exec/rule atoms).
"""
is_conjunction(node::SNode) =
    node isa SList &&
    !isempty(node.items) &&
    node.items[1] isa SAtom &&
    (node.items[1]::SAtom).name == ","

"""
Return true iff `node` contains no variables (is fully ground).
"""
is_ground(node::SNode) = count_vars(node) == 0

# Structural equality (enables == in tests)
Base.:(==)(a::SAtom, b::SAtom) = a.name == b.name
Base.:(==)(a::SVar,  b::SVar)  = a.name == b.name
Base.:(==)(a::SList, b::SList) = a.items == b.items
Base.:(==)(::SAtom, ::SVar)    = false
Base.:(==)(::SVar,  ::SAtom)   = false
Base.:(==)(::SList, ::SAtom)   = false
Base.:(==)(::SAtom, ::SList)   = false
Base.:(==)(::SList, ::SVar)    = false
Base.:(==)(::SVar,  ::SList)   = false

export SNode, SAtom, SVar, SList
export parse_program, parse_sexpr, sprint_sexpr, sprint_program
export count_vars, count_atoms, is_conjunction, is_ground
