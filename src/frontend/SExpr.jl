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
# Token boundaries follow the canonical MeTTa EBNF (docs/specs/metta_grammar.ebnf): a WORD body is
# any byte that is NOT whitespace and NOT a structural delimiter '(' ')' ';'.  We parse at the BYTE
# level — exactly like MORK's canonical `sexpr_parse!` (frontend/Frontend.jl), which runs over a
# `Vector{UInt8}` and captures each symbol as a byte-range as-is.  Every delimiter and ASCII
# whitespace byte is < 0x80, so all bytes of a multi-byte UTF-8 char (→, 𝜑, …) are non-delimiter
# non-whitespace and flow through verbatim as symbol bytes — there is NO String char-boundary
# indexing, so nothing can land mid-codepoint.  (The prior String/byte-step parser crashed on `→`.)
@inline _ws_byte(b::UInt8)::Bool  = b < 0x80 && isspace(Char(b))     # ASCII whitespace only (multi-byte ⇒ symbol byte)
@inline _sym_byte(b::UInt8)::Bool = !_ws_byte(b) && b != UInt8('(') && b != UInt8(')') && b != UInt8(';')

# ── Parser ────────────────────────────────────────────────────────────────────

"""
    parse_program(src::AbstractString) -> Vector{SNode}

Parse all top-level s-expressions from `src`.  Comments (`;...`) are skipped.  UTF-8 safe (byte-level).
"""
function parse_program(src::AbstractString)::Vector{SNode}
    b = Vector{UInt8}(src)            # parse over bytes (mirrors MORK sexpr_to_expr: Vector{UInt8}(src))
    nodes = SNode[]
    i = 1
    n = length(b)
    while i <= n
        i = _skip_ws(b, i, n)
        i > n && break
        node, i = _parse_at(b, i, n)
        push!(nodes, node)
    end
    nodes
end

"""
    parse_sexpr(src::AbstractString) -> SNode

Parse exactly one s-expression from the beginning of `src`.
"""
function parse_sexpr(src::AbstractString)::SNode
    b = Vector{UInt8}(src)
    n = length(b)
    i = _skip_ws(b, 1, n)
    node, _ = _parse_at(b, i, n)
    node
end

function _skip_ws(b::Vector{UInt8}, i::Int, n::Int)::Int
    while i <= n
        c = b[i]
        if c == UInt8(';')            # line comment: skip to end of line
            while i <= n && b[i] != UInt8('\n')
                i += 1
            end
        elseif _ws_byte(c)
            i += 1
        else
            break
        end
    end
    i
end

function _parse_at(b::Vector{UInt8}, i::Int, n::Int)::Tuple{SNode, Int}
    i > n && error("unexpected EOF at position $i")
    c = b[i]

    if c == UInt8('(')
        return _parse_list(b, i, n)
    elseif c == UInt8('$')
        return _parse_var(b, i, n)
    elseif _sym_byte(c)
        return _parse_atom(b, i, n)
    else
        error("unexpected byte 0x$(string(c, base=16, pad=2)) at position $i")
    end
end

function _parse_list(b::Vector{UInt8}, i::Int, n::Int)::Tuple{SNode, Int}
    i += 1                            # consume '('
    items = SNode[]
    while true
        i = _skip_ws(b, i, n)
        i > n && error("unterminated list: reached EOF")
        b[i] == UInt8(')') && return SList(items), i + 1
        node, i = _parse_at(b, i, n)
        push!(items, node)
    end
end

function _parse_var(b::Vector{UInt8}, i::Int, n::Int)::Tuple{SNode, Int}
    start = i
    i += 1                            # skip '$' (start retained so the name includes the leading $)
    while i <= n && _sym_byte(b[i])
        i += 1
    end
    SVar(String(b[start:(i - 1)])), i
end

function _parse_atom(b::Vector{UInt8}, i::Int, n::Int)::Tuple{SNode, Int}
    start = i
    while i <= n && _sym_byte(b[i])
        i += 1
    end
    SAtom(String(b[start:(i - 1)])), i
end

# ── Serializer ────────────────────────────────────────────────────────────────

"""
    sprint_sexpr(node::SNode) -> String
"""
sprint_sexpr(node::SAtom) = node.name
sprint_sexpr(node::SVar) = node.name
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

"""
Count the number of variable nodes (SVar) in a subtree.
"""
function count_vars(node::SNode)::Int
    node isa SVar && return 1
    node isa SAtom && return 0
    sum(count_vars(c) for c in (node::SList).items; init=0)
end

"""
Count the number of atom nodes (SAtom) in a subtree.
"""
function count_atoms(node::SNode)::Int
    node isa SAtom && return 1
    node isa SVar && return 0
    sum(count_atoms(c) for c in (node::SList).items; init=0)
end

"""
Return true iff `node` is a `,` conjunction list (the pattern list in exec/rule atoms).
"""
is_conjunction(node::SNode) =
    node isa SList && !isempty(node.items) && node.items[1] isa SAtom &&
    (node.items[1]::SAtom).name == ","

"""
Return true iff `node` contains no variables (is fully ground).
"""
is_ground(node::SNode) = count_vars(node) == 0

# Structural equality (enables == in tests)
Base.:(==)(a::SAtom, b::SAtom) = a.name == b.name
Base.:(==)(a::SVar, b::SVar) = a.name == b.name
Base.:(==)(a::SList, b::SList) = a.items == b.items
Base.:(==)(::SAtom, ::SVar) = false
Base.:(==)(::SVar, ::SAtom) = false
Base.:(==)(::SList, ::SAtom) = false
Base.:(==)(::SAtom, ::SList) = false
Base.:(==)(::SList, ::SVar) = false
Base.:(==)(::SVar, ::SList) = false

export SNode, SAtom, SVar, SList
export parse_program, parse_sexpr, sprint_sexpr, sprint_program
export count_vars, count_atoms, is_conjunction, is_ground
