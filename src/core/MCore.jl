"""
MCore — Minimal Core IR for the MeTTa+MM2 supercompiler.

Implements §3 of the MM2 Supercompiler spec (Goertzel, Oct 2025):
  §3.1  11 core node types (all carry effect metadata as a field, not nodes)
  §3.2  Domain-specific compilation rules (KB-IR, Evo-IR, MM2-IR → M-Core)

Extended by §2.4 of the Approximate Supercompilation spec:
  UncertainNode wraps any NodeID with a value p-box, cost p-box, and error bound.

Design constraints (from Grok feedback + session rules):
  - No Vector{Any} anywhere — every container is typed
  - All node types are immutable structs
  - NodeID is a typed alias (not Any)
  - Effects are attached as metadata, NOT as nodes themselves (§3.1)
"""

# ── NodeID — typed reference ───────────────────────────────────────────────────

"""
    NodeID

Opaque reference to an M-Core node in a `MCoreGraph`.
Wraps a `UInt32` index; invalid = 0.
"""
struct NodeID
    idx::UInt32
end
NodeID(i::Integer) = NodeID(UInt32(i))
const NULL_NODE = NodeID(0)
Base.isvalid(n::NodeID) = n.idx != 0
Base.:(==)(a::NodeID, b::NodeID) = a.idx == b.idx
Base.hash(n::NodeID, h::UInt) = hash(n.idx, h)
Base.show(io::IO, n::NodeID) = print(io, "NodeID(", n.idx, ")")

# ── Effect metadata (opaque at this layer — Effects.jl provides the algebra) ──

"""
    EffectSet

Compact set of effects attached to a node.  Represented as a bitmask so
that node structs stay small and allocation-free.
Bit meanings defined in Effects.jl.
"""
struct EffectSet
    mask::UInt8
end
EffectSet() = EffectSet(UInt8(0))
Base.isempty(e::EffectSet) = e.mask == 0

# ── SpaceID — resource identity for effects ──────────────────────────────────

struct SpaceID
    name::Symbol
end
const DEFAULT_SPACE = SpaceID(:default)

# ── 11 Core Node Types (§3.1) ─────────────────────────────────────────────────

abstract type MCoreNode end

"""
    Sym(name, effects)

Interned symbol.  Maps to `Sym(name)` in the spec.
"""
struct Sym <: MCoreNode
    name::Symbol
    effects::EffectSet
end
Sym(name::Symbol) = Sym(name, EffectSet())
Sym(name::AbstractString) = Sym(Symbol(name))

"""
    Var(ix, effects)

Variable reference (de Bruijn or SSA index).  Maps to `Var(ix)`.
"""
struct Var <: MCoreNode
    ix::Int
    effects::EffectSet
end
Var(ix::Int) = Var(ix, EffectSet())

"""
    Lit(val, effects)

Literal value (Int, Float64, Bool, String, etc.).  Maps to `Lit(val)`.
`val` is untyped because literals are inherently heterogeneous; this is the
one unavoidable `Any` field in M-Core (tagged box, not a container).
"""
struct Lit <: MCoreNode
    val::Any     # tagged box — inevitable for heterogeneous literals
    effects::EffectSet
end
Lit(v) = Lit(v, EffectSet())

"""
    Con(head, fields, effects)

Constructor application.  Maps to `Con(head:Sym, fields:[NodeID])`.
"""
struct Con <: MCoreNode
    head::Symbol
    fields::Vector{NodeID}
    effects::EffectSet
end
Con(head::Symbol, fields::Vector{NodeID}) = Con(head, fields, EffectSet())
Con(head::Symbol) = Con(head, NodeID[], EffectSet())

"""
    App(fun, args, effects)

Function application.  Maps to `App(fun:NodeID, args:[NodeID])`.
"""
struct App <: MCoreNode
    fun::NodeID
    args::Vector{NodeID}
    effects::EffectSet
end
App(fun::NodeID, args::Vector{NodeID}) = App(fun, args, EffectSet())

"""
    Abs(params, body, effects)

Abstraction (lambda).  Maps to `Abs(params:[Var], body:NodeID)`.
"""
struct Abs <: MCoreNode
    params::Vector{Int}   # de Bruijn indices
    body::NodeID
    effects::EffectSet
end
Abs(params::Vector{Int}, body::NodeID) = Abs(params, body, EffectSet())

"""
    LetNode(bindings, body, effects)

Let binding.  Maps to `Let(bindings:[(Var,NodeID)], body:NodeID)`.
(Named LetNode to avoid shadowing Base.Let)
"""
struct LetNode <: MCoreNode
    bindings::Vector{Tuple{Int, NodeID}}   # (var_ix, value_id)
    body::NodeID
    effects::EffectSet
end
LetNode(bindings::Vector{Tuple{Int, NodeID}}, body::NodeID) =
    LetNode(bindings, body, EffectSet())

"""
    MatchClause

One clause of a Match node: `(pattern_id, guard_id, body_id)`.
guard_id = NULL_NODE means no guard.
"""
struct MatchClause
    pattern::NodeID
    guard::NodeID   # NULL_NODE = unconditional
    body::NodeID
end
MatchClause(p::NodeID, b::NodeID) = MatchClause(p, NULL_NODE, b)

"""
    MatchNode(scrut, clauses, effects)

Pattern match.  Maps to `Match(scrut:NodeID, clauses:[(Pat,guard,body)])`.
"""
struct MatchNode <: MCoreNode
    scrut::NodeID
    clauses::Vector{MatchClause}
    effects::EffectSet
end
MatchNode(scrut::NodeID, clauses::Vector{MatchClause}) =
    MatchNode(scrut, clauses, EffectSet())

"""
    ChoiceAlt

One alternative in a Choice: `(guard_id, expr_id)`.
guard_id = NULL_NODE means always-eligible.
"""
struct ChoiceAlt
    guard::NodeID   # NULL_NODE = unconditional
    expr::NodeID
end
ChoiceAlt(e::NodeID) = ChoiceAlt(NULL_NODE, e)

"""
    Choice(alts, effects)

Nondeterministic choice.  Requires splitting (Algorithm 9).
Maps to `Choice(alts:[(guard, expr)])`.
"""
struct Choice <: MCoreNode
    alts::Vector{ChoiceAlt}
    effects::EffectSet
end
Choice(alts::Vector{ChoiceAlt}) = Choice(alts, EffectSet())

"""
    Prim(op, args, effects)

Primitive operation with domain-specific semantics.
Maps to `Prim(op:Sym, args:[NodeID])`.

Standard ops (§3.2):
:kb_query     — KB pattern query  → effects include Read(space)
:fitness_eval — fitness evaluation → effects include Read(data), Observe(prog)
:mm2_exec     — MM2 exec           → effects include Read(space), Append(space)
"""
struct Prim <: MCoreNode
    op::Symbol
    args::Vector{NodeID}
    effects::EffectSet
end
Prim(op::Symbol, args::Vector{NodeID}) = Prim(op, args, EffectSet())
Prim(op::Symbol) = Prim(op, NodeID[], EffectSet())

"""
    MCoreRef(def_id, effects)

Reference to a named definition.  Maps to `MCoreRef(def_id:Sym)`.
"""
struct MCoreRef <: MCoreNode
    def_id::Symbol
    effects::EffectSet
end
MCoreRef(def_id::Symbol) = MCoreRef(def_id, EffectSet())

# ── Uncertain Node (§2.4 of Approximate Supercompilation spec) ───────────────

"""
    PBox

Probability box: tracks uncertainty in a computed value.

All 4 fields from §2.2 of the Approximate Supercompilation spec:
intervals       — disjoint [lo, hi] intervals
probabilities   — P(x in intervals[i])
confidence      — total probability mass tracked
correlation_sig — BitVector: shared bits with another PBox → statistically
dependent → use Fréchet bounds instead of independence.
Prevents overconfidence when combining related evidence.

Full algebra (AddPBox, MulPBox, WidenPBox, Fréchet bounds) in approx/PBoxAlgebra.jl.
"""
struct PBox
    intervals::Vector{Tuple{Float64, Float64}}
    probabilities::Vector{Float64}
    confidence::Float64
    correlation_sig::BitVector
end
PBox(lo::Float64, hi::Float64, p::Float64=1.0) = PBox([(lo, hi)], [p], p, BitVector())
PBox(v::Float64) = PBox(v, v, 1.0, BitVector())
PBox(intervals::Vector{Tuple{Float64, Float64}}, probs::Vector{Float64}, conf::Float64) =
    PBox(intervals, probs, conf, BitVector())

"""
    UncertainNode(base, value_pbox, cost_pbox, error_bound)

Extension from §2.4 of the Approximate Supercompilation spec.
Wraps any `NodeID` with uncertainty metadata.

base        — original M-Core node
value_pbox  — uncertainty in computed value
cost_pbox   — uncertainty in execution cost
error_bound — maximum acceptable error for this node
"""
struct UncertainNode <: MCoreNode
    base::NodeID
    value_pbox::PBox
    cost_pbox::PBox
    error_bound::Float64
    effects::EffectSet
end
UncertainNode(base::NodeID, vp::PBox, cp::PBox, ε::Float64) =
    UncertainNode(base, vp, cp, ε, EffectSet())

# ── MCoreGraph — typed node store ─────────────────────────────────────────────

"""
    MCoreGraph

Typed arena of M-Core nodes, indexed by NodeID.
Stores each concrete type in its own typed vector for type-stable dispatch.
"""
mutable struct MCoreGraph
    syms::Vector{Sym}
    vars::Vector{Var}
    lits::Vector{Lit}          # abstract Lit — heterogeneous literal values
    cons::Vector{Con}
    apps::Vector{App}
    abss::Vector{Abs}
    lets::Vector{LetNode}
    matches::Vector{MatchNode}
    choices::Vector{Choice}
    prims::Vector{Prim}
    refs::Vector{MCoreRef}
    # Map from NodeID.idx → (type_tag::UInt8, type_index::Int)
    index::Vector{Tuple{UInt8, Int}}
    next_id::UInt32
    # Definition store: symbol name → body NodeID (for MCoreRef unfold)
    # Populated by def_add! or by register_space_primitives! from MORK atoms.
    defs::Dict{Symbol, NodeID}
end

MCoreGraph() = MCoreGraph(
    Sym[],
    Var[],
    Lit[],
    Con[],
    App[],
    Abs[],
    LetNode[],
    MatchNode[],
    Choice[],
    Prim[],
    MCoreRef[],
    Tuple{UInt8, Int}[],
    UInt32(1),
    Dict{Symbol, NodeID}()
)

"""
    def_add!(g, name, body_id) → NodeID

Register a named definition in the MCoreGraph so MCoreRef nodes can unfold it.
`name` is the definition symbol; `body_id` is the NodeID of the definition body.
"""
function def_add!(g::MCoreGraph, name::Symbol, body_id::NodeID)::NodeID
    g.defs[name] = body_id
    body_id
end

"""
    def_lookup(g, name) → Union{NodeID, Nothing}

Look up a named definition. Returns the body NodeID or nothing if undefined.
"""
def_lookup(g::MCoreGraph, name::Symbol)::Union{NodeID, Nothing} = get(g.defs, name, nothing)

# Type tags
const TAG_SYM = UInt8(1);
const TAG_VAR = UInt8(2);
const TAG_LIT = UInt8(3)
const TAG_CON = UInt8(4);
const TAG_APP = UInt8(5);
const TAG_ABS = UInt8(6)
const TAG_LET = UInt8(7);
const TAG_MATCH = UInt8(8);
const TAG_CHOICE = UInt8(9)
const TAG_PRIM = UInt8(10);
const TAG_MREF = UInt8(11)

function _add_node!(g::MCoreGraph, tag::UInt8, vec::Vector, node)::NodeID
    push!(vec, node)
    push!(g.index, (tag, length(vec)))
    id = NodeID(g.next_id)
    g.next_id += UInt32(1)
    id
end

add_sym!(g::MCoreGraph, n::Sym) = _add_node!(g, TAG_SYM, g.syms, n)
add_var!(g::MCoreGraph, n::Var) = _add_node!(g, TAG_VAR, g.vars, n)
add_lit!(g::MCoreGraph, n::Lit) = _add_node!(g, TAG_LIT, g.lits, n)
add_con!(g::MCoreGraph, n::Con) = _add_node!(g, TAG_CON, g.cons, n)
add_app!(g::MCoreGraph, n::App) = _add_node!(g, TAG_APP, g.apps, n)
add_abs!(g::MCoreGraph, n::Abs) = _add_node!(g, TAG_ABS, g.abss, n)
add_let!(g::MCoreGraph, n::LetNode) = _add_node!(g, TAG_LET, g.lets, n)
add_match!(g::MCoreGraph, n::MatchNode) = _add_node!(g, TAG_MATCH, g.matches, n)
add_choice!(g::MCoreGraph, n::Choice) = _add_node!(g, TAG_CHOICE, g.choices, n)
add_prim!(g::MCoreGraph, n::Prim) = _add_node!(g, TAG_PRIM, g.prims, n)
add_mref!(g::MCoreGraph, n::MCoreRef) = _add_node!(g, TAG_MREF, g.refs, n)

"""
Retrieve the MCoreNode for a NodeID.
"""
function get_node(g::MCoreGraph, id::NodeID)::MCoreNode
    !isvalid(id) && error("NULL_NODE has no node")
    tag, idx = g.index[id.idx]
    if tag == TAG_SYM
        ;
        return g.syms[idx]
    elseif tag == TAG_VAR
        ;
        return g.vars[idx]
    elseif tag == TAG_LIT
        ;
        return g.lits[idx]
    elseif tag == TAG_CON
        ;
        return g.cons[idx]
    elseif tag == TAG_APP
        ;
        return g.apps[idx]
    elseif tag == TAG_ABS
        ;
        return g.abss[idx]
    elseif tag == TAG_LET
        ;
        return g.lets[idx]
    elseif tag == TAG_MATCH
        ;
        return g.matches[idx]
    elseif tag == TAG_CHOICE
        ;
        return g.choices[idx]
    elseif tag == TAG_PRIM
        ;
        return g.prims[idx]
    elseif tag == TAG_MREF
        ;
        return g.refs[idx]
    else
        error("unknown tag $tag")
    end
end

# ── §3.2 Domain compilation helpers ───────────────────────────────────────────

"""
    compile_kb_query(g, pattern_id) -> NodeID

§3.2: `KBQuery(pattern) -> Prim(kb_query, [pattern], effects=[Read(kb)])`.
"""
function compile_kb_query(g::MCoreGraph, pattern_id::NodeID)::NodeID
    add_prim!(g, Prim(:kb_query, [pattern_id], EffectSet(UInt8(0x01))))   # 0x01 = READ bit (defined in Effects.jl)
end

"""
    compile_mm2_exec(g, pri_id, pats_id, temps_id) -> NodeID

§3.2: `MM2Exec(pri, pats, temps) -> Prim(mm2_exec, ..., effects=[Read(space), Append(space)])`.
"""
function compile_mm2_exec(
    g::MCoreGraph, pri_id::NodeID, pats_id::NodeID, temps_id::NodeID
)::NodeID
    add_prim!(g, Prim(:mm2_exec, [pri_id, pats_id, temps_id], EffectSet(UInt8(0x05))))   # 0x05 = READ|APPEND
end

# ── Pretty-printing ───────────────────────────────────────────────────────────

function Base.show(io::IO, n::Sym)
    print(io, "Sym(:", n.name, ")")
end
function Base.show(io::IO, n::Var)
    print(io, "Var(", n.ix, ")")
end
function Base.show(io::IO, n::Lit)
    print(io, "Lit(", repr(n.val), ")")
end
function Base.show(io::IO, n::Con)
    print(io, "Con(:", n.head, ", ", length(n.fields), " fields)")
end
function Base.show(io::IO, n::App)
    print(io, "App(", n.fun, ", ", length(n.args), " args)")
end
function Base.show(io::IO, n::Abs)
    print(io, "Abs(", n.params, " → ", n.body, ")")
end
function Base.show(io::IO, n::LetNode)
    print(io, "Let(", length(n.bindings), " bindings)")
end
function Base.show(io::IO, n::MatchNode)
    print(io, "Match(", n.scrut, ", ", length(n.clauses), " clauses)")
end
function Base.show(io::IO, n::Choice)
    print(io, "Choice(", length(n.alts), " alts)")
end
function Base.show(io::IO, n::Prim)
    print(io, "Prim(:", n.op, ", ", length(n.args), " args)")
end
function Base.show(io::IO, n::MCoreRef)
    print(io, "MCoreRef(:", n.def_id, ")")
end

export NodeID, NULL_NODE, EffectSet, SpaceID, DEFAULT_SPACE
export MCoreNode, Sym, Var, Lit, Con, App, Abs, LetNode, MatchNode, MatchClause
export Choice, ChoiceAlt, Prim, MCoreRef
export PBox, UncertainNode
export MCoreGraph, def_add!, def_lookup
export add_sym!, add_var!, add_lit!, add_con!, add_app!, add_abs!
export add_let!, add_match!, add_choice!, add_prim!, add_mref!
export get_node
export compile_kb_query, compile_mm2_exec
export TAG_SYM, TAG_VAR, TAG_LIT, TAG_CON, TAG_APP, TAG_ABS
export TAG_LET, TAG_MATCH, TAG_CHOICE, TAG_PRIM, TAG_MREF
