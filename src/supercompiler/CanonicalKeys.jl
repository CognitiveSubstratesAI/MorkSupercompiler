"""
CanonicalKeys — canonical path signatures and subsumption for termination.

Implements §6.3 of the MM2 Supercompiler spec (Goertzel, Oct 2025):
  §6.3.1  CanonicalPathSig — canonical key encoding (depth-3 shape, sorted tags)
  §6.3.2  Algorithm 10 — KeySubsumption (sound folding via subsumption relation)

Purpose: prevent the supercompiler from unfolding the same expression family
forever.  When `Subsumes(seen_key, current_key)` holds, the current expression
can be folded back to the earlier one — guaranteeing termination.

Key invariants (spec §6.3.1):
  - Shape bounded at depth 3 (prevents key explosion)
  - Tags stored as SortedMultiset{Symbol} (canonical ordering)
  - KB signature uses SortedList of (predicate, FixedArgMask) pairs
  - Effect signature uses SortedMultiset of effect classes

Subsumption axioms (spec §6.3.2 Algorithm 10):
  (1) Structural: same head + key1.shape subsumes key2.shape
  (2) KB: for every predicate in key2, a matching entry exists in key1
          with mask1 ⊆ mask2 (key1 is more general)
  (3) Effect: key1.effects ⊆ key2.effects (key1 has fewer effects)
"""

# ── CompactShape — depth-3 arity vector (§6.3.1) ──────────────────────────────

"""
    CompactShape

Arity vector at depth ≤ 3.  The spec bounds shape depth at 3 to prevent
key explosion while still capturing enough structural information for folding.

Encoding: `arities[d]` = arity at depth d (d=1 is outermost).
Missing depths = 0 (leaf / not expanded).
"""
struct CompactShape
    arities :: NTuple{3, UInt8}   # depth 1, 2, 3
end
CompactShape()                                              = CompactShape((UInt8(0), UInt8(0), UInt8(0)))
# Convenience constructor: 1-3 integers → NTuple{3,UInt8}
function CompactShape(a1::Integer, a2::Integer=0, a3::Integer=0) :: CompactShape
    CompactShape((UInt8(a1), UInt8(a2), UInt8(a3)))
end

"""
    shape_subsumes(s1, s2) -> Bool

s1 subsumes s2 iff s1 is "at least as general": for each depth, s1 arity ≤ s2 arity
(s1 constrains less — more general).
"""
shape_subsumes(s1::CompactShape, s2::CompactShape) :: Bool =
    all(s1.arities[i] <= s2.arities[i] for i in 1:3)

"""Extract a CompactShape from an MCoreNode (depth-3 traversal)."""
function extract_shape(g::MCoreGraph, id::NodeID) :: CompactShape
    !isvalid(id) && return CompactShape()
    n = get_node(g, id)
    a1 = _node_arity(n)
    if a1 == 0; return CompactShape(0, 0, 0) end
    # depth 2: average arity of immediate children
    a2 = _avg_child_arity(g, n)
    CompactShape(a1, a2, 0)
end

_node_arity(n::Con)       = UInt8(min(length(n.fields), 63))
_node_arity(n::App)       = UInt8(min(length(n.args) + 1, 63))
_node_arity(n::Abs)       = UInt8(min(length(n.params), 63))
_node_arity(n::LetNode)   = UInt8(min(length(n.bindings), 63))
_node_arity(n::MatchNode) = UInt8(min(length(n.clauses), 63))
_node_arity(n::Choice)    = UInt8(min(length(n.alts), 63))
_node_arity(n::Prim)      = UInt8(min(length(n.args), 63))
_node_arity(::MCoreNode)  = UInt8(0)

function _avg_child_arity(g::MCoreGraph, n::Con) :: UInt8
    isempty(n.fields) && return UInt8(0)
    s = sum(_node_arity(get_node(g, f)) for f in n.fields; init=UInt8(0))
    UInt8(min(div(Int(s), length(n.fields)), 63))
end
_avg_child_arity(_, ::MCoreNode) = UInt8(0)

# ── FixedArgMask — argument position bitmask ──────────────────────────────────

"""
    FixedArgMask

Bitmask indicating which argument positions of a predicate are constrained
(fixed / non-variable) in a pattern.  Bit i set ↔ arg i is fixed.
"""
struct FixedArgMask
    bits :: UInt32
end
FixedArgMask() = FixedArgMask(UInt32(0))
fixed_arg(m::FixedArgMask, i::Int) = (m.bits >> (i - 1)) & UInt32(1) != 0
set_fixed(m::FixedArgMask, i::Int) = FixedArgMask(m.bits | (UInt32(1) << (i - 1)))
Base.:⊆(a::FixedArgMask, b::FixedArgMask) = (a.bits & b.bits) == a.bits   # a ⊆ b

# ── CanonicalKBSig (§6.3.1) ───────────────────────────────────────────────────

"""
    CanonicalKBSig

KB access signature: which predicates are accessed and which argument
positions are constrained.  Predicates stored sorted for canonical ordering.
"""
struct CanonicalKBSig
    predicates     :: Vector{Tuple{Symbol, FixedArgMask}}  # sorted by pred name
    access_pattern :: UInt32   # bitmask: which arg positions accessed overall
end
CanonicalKBSig() = CanonicalKBSig(Tuple{Symbol,FixedArgMask}[], UInt32(0))

# ── CanonicalEffectSig (§6.3.1) ───────────────────────────────────────────────

"""Effect class — coarser than the full Effect type, for canonical signatures."""
@enum EffectClass begin
    ECLASS_READ
    ECLASS_WRITE
    ECLASS_APPEND
    ECLASS_CREATE
    ECLASS_DELETE
    ECLASS_OBSERVE
end

effect_class(::ReadEffect)    = ECLASS_READ
effect_class(::WriteEffect)   = ECLASS_WRITE
effect_class(::AppendEffect)  = ECLASS_APPEND
effect_class(::CreateEffect)  = ECLASS_CREATE
effect_class(::DeleteEffect)  = ECLASS_DELETE
effect_class(::ObserveEffect) = ECLASS_OBSERVE

"""
    CanonicalEffectSig

Effect signature: sorted multiset of effect classes + set of resource IDs.
"""
struct CanonicalEffectSig
    effects   :: Vector{EffectClass}   # sorted
    resources :: Vector{Symbol}        # sorted SpaceID names
end
CanonicalEffectSig() = CanonicalEffectSig(EffectClass[], Symbol[])

# ── CanonicalPathSig (§6.3.1) ─────────────────────────────────────────────────

"""
    CanonicalPathSig

Canonical key for fold/termination checking.  All fields canonically encoded:
  head       — symbol name of the outermost node
  shape      — depth-3 arity vector (bounded to prevent key explosion)
  tags       — sorted multiset of constructor/symbol tags
  depth      — current unfolding depth
  kb_sig     — KB access signature
  effect_sig — effect signature
"""
struct CanonicalPathSig
    head       :: Symbol
    shape      :: CompactShape
    tags       :: Vector{Symbol}          # sorted
    depth      :: Int
    kb_sig     :: CanonicalKBSig
    effect_sig :: CanonicalEffectSig
end

"""Build a CanonicalPathSig from a node in the graph.

Implements Algorithm 10 §6.3 — populates kb_sig and effect_sig by
traversing the MCoreGraph subtree (bounded to depth 4):
  kb_sig     — predicates accessed via :kb_query Prim nodes
  effect_sig — effect classes from all Prim nodes in the subtree
"""
function canonical_key(g::MCoreGraph, id::NodeID, depth::Int=0) :: CanonicalPathSig
    !isvalid(id) && return CanonicalPathSig(:null, CompactShape(), Symbol[], depth,
                                            CanonicalKBSig(), CanonicalEffectSig())
    node     = get_node(g, id)
    head     = _node_head(node)
    shape    = extract_shape(g, id)
    tags     = sort!(_collect_tags(g, id, 3))
    kb_sig   = _collect_kb_sig(g, id, 4)
    eff_sig  = _collect_effect_sig(g, id, 4)
    CanonicalPathSig(head, shape, tags, depth, kb_sig, eff_sig)
end

# ── Algorithm 10 helpers — KB and Effect signature extraction ──────────────────

"""
Traverse the subtree (bounded depth) collecting :kb_query predicate accesses.
For each Prim(:kb_query, [pattern_id, ...]):
  - extract the predicate head from pattern_id (first Con/Sym child)
  - build a FixedArgMask for which arg positions are ground (Sym/Lit) vs free (Var)
"""
function _collect_kb_sig(g::MCoreGraph, id::NodeID, max_depth::Int) :: CanonicalKBSig
    (max_depth <= 0 || !isvalid(id)) && return CanonicalKBSig()
    preds = Dict{Symbol, FixedArgMask}()
    _traverse_kb_sig!(g, id, max_depth, preds)
    isempty(preds) && return CanonicalKBSig()
    pred_list = sort!([(p, m) for (p, m) in preds]; by=first)
    access = reduce(|, (m.bits for (_, m) in pred_list); init=UInt32(0))
    CanonicalKBSig(pred_list, access)
end

function _traverse_kb_sig!(g, id, depth, preds)
    (!isvalid(id) || depth <= 0) && return
    node = get_node(g, id)
    if node isa Prim && node.op == :kb_query && !isempty(node.args)
        pat_id = node.args[1]
        isvalid(pat_id) || return
        pat = get_node(g, pat_id)
        if pat isa Con
            # Previously: `head_node = get_node(g, _sym_node_of(g, pat.head))`
            # — `_sym_node_of(::Symbol)` always returned NULL_NODE, so this
            # call would throw `error("NULL_NODE has no node")`. The whole
            # KB-signature path was dead-on-arrival. Also `pat.args` was a
            # FieldError (Con stores `fields`, not `args`). Fixed both.
            pred = pat.head
            mask = FixedArgMask()
            for (i, arg_id) in enumerate(pat.fields)
                isvalid(arg_id) || continue
                arg_node = get_node(g, arg_id)
                if arg_node isa Sym || arg_node isa Lit
                    mask = set_fixed(mask, i)
                end
            end
            existing = get(preds, pred, FixedArgMask())
            preds[pred] = FixedArgMask(existing.bits | mask.bits)
        end
    end
    for cid in _node_children(node)
        _traverse_kb_sig!(g, cid, depth - 1, preds)
    end
end

"""
Traverse the subtree collecting effect classes from all Prim nodes.
"""
function _collect_effect_sig(g::MCoreGraph, id::NodeID, max_depth::Int) :: CanonicalEffectSig
    (max_depth <= 0 || !isvalid(id)) && return CanonicalEffectSig()
    effects   = EffectClass[]
    resources = Symbol[]
    _traverse_effect_sig!(g, id, max_depth, effects, resources)
    isempty(effects) && return CanonicalEffectSig()
    sort!(effects); unique!(effects)
    sort!(resources); unique!(resources)
    CanonicalEffectSig(effects, resources)
end

function _traverse_effect_sig!(g, id, depth, effects, resources)
    (!isvalid(id) || depth <= 0) && return
    node = get_node(g, id)
    if node isa Prim
        eset = node.effects
        for eff in _effectset_to_effects(eset)
            push!(effects, effect_class(eff))
        end
    end
    for cid in _node_children(node)
        _traverse_effect_sig!(g, cid, depth - 1, effects, resources)
    end
end

function _effectset_to_effects(eset::EffectSet) :: Vector{<:Effect}
    effs = Effect[]
    mask = eset.mask
    # Bit layout per Stepper.jl `_node_effects`:
    #   0x01 = Read, 0x02 = Write, 0x04 = Append, 0x08 = Create,
    #   0x10 = Delete, 0x20 = Observe
    # Create + Delete were previously missing — effect signatures silently
    # dropped them, breaking Algorithm 10's effect-subsumption check.
    mask & 0x01 != 0 && push!(effs, ReadEffect(DEFAULT_SPACE))
    mask & 0x02 != 0 && push!(effs, WriteEffect(DEFAULT_SPACE))
    mask & 0x04 != 0 && push!(effs, AppendEffect(DEFAULT_SPACE))
    mask & 0x08 != 0 && push!(effs, CreateEffect(DEFAULT_SPACE))
    mask & 0x10 != 0 && push!(effs, DeleteEffect(DEFAULT_SPACE))
    mask & 0x20 != 0 && push!(effs, ObserveEffect(DEFAULT_SPACE))
    effs
end

_node_head(n::Sym)       = n.name
_node_head(n::Var)       = :Var
_node_head(n::Lit)       = :Lit
_node_head(n::Con)       = n.head
_node_head(n::App)       = :App
_node_head(n::Abs)       = :Abs
_node_head(n::LetNode)   = :Let
_node_head(n::MatchNode) = :Match
_node_head(n::Choice)    = :Choice
_node_head(n::Prim)      = n.op
_node_head(n::MCoreRef)  = n.def_id
_node_head(::MCoreNode)  = :unknown
_node_head(::UncertainNode) = :UncertainNode

function _collect_tags(g::MCoreGraph, id::NodeID, max_depth::Int) :: Vector{Symbol}
    max_depth <= 0 || !isvalid(id) && return Symbol[]
    node = get_node(g, id)
    tags = [_node_head(node)]
    children = _node_children(node)
    for cid in children
        append!(tags, _collect_tags(g, cid, max_depth - 1))
    end
    tags
end

_node_children(n::Con)       = n.fields
_node_children(n::App)       = [n.fun; n.args]
_node_children(n::Abs)       = [n.body]
_node_children(n::LetNode)   = [v for (_, v) in n.bindings]
_node_children(n::MatchNode) = [n.scrut]
_node_children(n::Choice)    = [a.expr for a in n.alts]
_node_children(n::Prim)      = n.args
_node_children(::MCoreNode)  = NodeID[]

# ── Algorithm 10 — KeySubsumption (§6.3.2) ────────────────────────────────────

"""
    subsumes(key1::CanonicalPathSig, key2::CanonicalPathSig) -> Bool

Algorithm 10 (KeySubsumption) from §6.3.2.

Returns true iff `key1` subsumes `key2` — meaning the expression represented
by `key2` can be safely folded back to the one represented by `key1`.

Three-part check (verbatim from spec):
  (1) Structural: same head + key1.shape subsumes key2.shape
  (2) KB: for every predicate in key2, key1 has a more-general entry
  (3) Effect: key1.effects ⊆ key2.effects (key1 has fewer effects)
"""
function subsumes(key1::CanonicalPathSig, key2::CanonicalPathSig) :: Bool
    # (1) Structural subsumption
    key1.head != key2.head                    && return false
    !shape_subsumes(key1.shape, key2.shape)   && return false

    # (2) KB subsumption: every predicate in key2 must have a match in key1
    for (pred2, mask2) in key2.kb_sig.predicates
        found = false
        for (pred1, mask1) in key1.kb_sig.predicates
            if pred1 == pred2 && mask1 ⊆ mask2
                found = true
                break
            end
        end
        found || return false
    end

    # (3) Effect subsumption: key1 must have a subset of key2's effects
    for ec in key1.effect_sig.effects
        ec in key2.effect_sig.effects || return false
    end

    true
end

# ── Fold table — memoization of seen canonical keys ───────────────────────────

"""
    FoldTable

Maps CanonicalPathSig → NodeID of the previously-seen expression.
When a new expression has a key subsumed by a key in the table, it can be
folded back to the earlier result.

The fold table is the mechanism that guarantees termination: the supercompiler
can only introduce finitely many distinct canonical keys.
"""
mutable struct FoldTable
    entries :: Vector{Tuple{CanonicalPathSig, NodeID}}
end
FoldTable() = FoldTable(Tuple{CanonicalPathSig, NodeID}[])

"""Record a new expression in the fold table."""
function record!(ft::FoldTable, key::CanonicalPathSig, id::NodeID)
    push!(ft.entries, (key, id))
end

"""
    lookup_fold(ft, key) -> Union{NodeID, Nothing}

Return the NodeID of a previously-seen expression subsumed by `key`,
or `nothing` if no such expression exists in the table.
"""
function lookup_fold(ft::FoldTable, key::CanonicalPathSig) :: Union{NodeID, Nothing}
    for (seen_key, seen_id) in ft.entries
        subsumes(seen_key, key) && return seen_id
    end
    nothing
end

"""Return true if the fold table already has a subsuming entry for `key`."""
can_fold(ft::FoldTable, key::CanonicalPathSig) :: Bool =
    lookup_fold(ft, key) !== nothing

export CompactShape, shape_subsumes, extract_shape
export FixedArgMask, fixed_arg, set_fixed
export CanonicalKBSig, EffectClass
export CanonicalEffectSig, CanonicalPathSig
export canonical_key, subsumes
export FoldTable, record!, lookup_fold, can_fold
