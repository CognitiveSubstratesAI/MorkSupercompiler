"""
KBSaturation — incremental KB saturation under monotonic growth.

Implements §7 of the MM2 Supercompiler spec (Goertzel, Oct 2025):
  §7.1  Algorithm 11 — IncrementalSaturation (semi-naive evaluation)
  §7.2  VersionedIndex — versioned persistent indices for efficient lookup

Semi-naive invariant (§7.1): at least one premise of each new derivation
must come from the delta (new facts added since the last iteration).
This avoids re-deriving facts that have already been derived, giving
O(Δ) update cost rather than O(total²) for each new batch of facts.

VersionedIndex (§7.2): each index carries a version number and delta tracking.
Queries at `min_version > current_version` trigger a replan rather than
using stale data.

Both structures exploit sink-free semantics (§4.3): facts are only ever
added, never deleted, so indices never need invalidation — only extension.
"""

# ── Fact representation ───────────────────────────────────────────────────────

"""
    Fact

A derived fact: a NodeID in the M-Core graph + provenance.

id         — NodeID of the fact expression
rule_id    — NodeID of the rule that derived this fact (NULL_NODE = base fact)
premises   — NodeIDs of the premises used in the derivation
version    — saturation round when this fact was derived
"""
struct Fact
    id::NodeID
    rule_id::NodeID
    premises::Vector{NodeID}
    version::Int
end
Fact(id::NodeID) = Fact(id, NULL_NODE, NodeID[], 0)  # base fact

is_base_fact(f::Fact) = !isvalid(f.rule_id)

# ── Rule representation ───────────────────────────────────────────────────────

"""
    Rule

A KB rule: a head pattern and a body (list of premise patterns).
When all premises are matched in the current fact set, the head is derived.

head_id    — NodeID of the head pattern (what gets derived)
body_ids   — NodeIDs of premise patterns (must ALL match)
rule_id    — unique identifier for this rule
"""
struct Rule
    head_id::NodeID
    body_ids::Vector{NodeID}
    rule_id::NodeID
end

# ── VersionedIndex (§7.2) ─────────────────────────────────────────────────────

"""
    IndexStats

Per-index statistics for replanning (§5.2.2 ShouldReplan).
"""
struct IndexStats
    fact_count::Int
    last_update::Int   # version when last updated
end
IndexStats() = IndexStats(0, 0)

"""
    VersionedIndex

Versioned persistent index mapping pattern shape → set of matching Fact IDs.
Supports delta tracking for incremental updates.

version         — current version (increments with each saturation round)
index           — pattern-head → Vec{fact_id} lookup
delta_since     — version → fact_ids added since that version
stats           — per-pattern statistics for replanning
last_replan_ver — version of last full replan
"""
mutable struct VersionedIndex
    version::Int
    index::Dict{Symbol, Vector{NodeID}}   # pred_head → [fact_ids]
    delta_since::Dict{Int, Vector{NodeID}}      # ver → [new fact_ids]
    stats::Dict{Symbol, IndexStats}
    last_replan_ver::Int
end

VersionedIndex() =
    VersionedIndex(
        0,
        Dict{Symbol, Vector{NodeID}}(),
        Dict{Int, Vector{NodeID}}(),
        Dict{Symbol, IndexStats}(),
        0
    )

"""
Insert a fact into the index under its head predicate.
"""
function index_insert!(vi::VersionedIndex, g::MCoreGraph, f::Fact)
    head = _fact_head(g, f.id)
    bucket = get!(vi.index, head, NodeID[])
    push!(bucket, f.id)
    delta = get!(vi.delta_since, vi.version, NodeID[])
    push!(delta, f.id)
    old = get(vi.stats, head, IndexStats())
    vi.stats[head] = IndexStats(old.fact_count + 1, vi.version)
end

"""
Lookup all fact NodeIDs with a given predicate head.
"""
function index_lookup(vi::VersionedIndex, head::Symbol)::Vector{NodeID}
    get(vi.index, head, NodeID[])
end

"""
Facts added since `min_version` (for semi-naive filtering).
"""
function index_delta_since(vi::VersionedIndex, min_version::Int)::Vector{NodeID}
    out = NodeID[]
    for (ver, ids) in vi.delta_since
        ver >= min_version && append!(out, ids)
    end
    out
end

"""
Advance the index to the next version.
"""
bump_version!(vi::VersionedIndex) = (vi.version += 1; vi)

function _fact_head(g::MCoreGraph, id::NodeID)::Symbol
    !isvalid(id) && return :nil
    n = get_node(g, id)
    n isa Con && return n.head
    n isa Sym && return n.name
    n isa Prim && return n.op
    :unknown
end

# ── KBState — full KB for saturation ─────────────────────────────────────────

"""
    KBState

Complete KB state for incremental saturation:
facts   — all known facts (base + derived), keyed by NodeID
rules   — rewrite rules
index   — versioned index for fast lookup
delta   — facts added in the current round (for semi-naive invariant)
"""
mutable struct KBState
    g::MCoreGraph
    facts::Dict{UInt32, Fact}    # NodeID.idx → Fact
    rules::Vector{Rule}
    index::VersionedIndex
    delta::Vector{Fact}          # current-round delta
    version::Int
end

function KBState(g::MCoreGraph)
    KBState(g, Dict{UInt32, Fact}(), Rule[], VersionedIndex(), Fact[], 0)
end

"""
Add a base fact to the KB.
"""
function kb_add_fact!(kb::KBState, f::Fact)
    haskey(kb.facts, f.id.idx) && return nothing   # idempotent
    kb.facts[f.id.idx] = f
    index_insert!(kb.index, kb.g, f)
    push!(kb.delta, f)
end

kb_add_fact!(kb::KBState, id::NodeID) = kb_add_fact!(kb, Fact(id))

"""
Add a rule to the KB.
"""
kb_add_rule!(kb::KBState, r::Rule) = push!(kb.rules, r)

"""
All fact NodeIDs in the KB.
"""
all_facts(kb::KBState)::Vector{NodeID} = [f.id for f in values(kb.facts)]

# ── Algorithm 11 — IncrementalSaturation (§7.1) ───────────────────────────────

"""
    saturate!(kb; max_rounds) -> Int

Algorithm 11 (IncrementalSaturation) from §7.1.  Semi-naive evaluation:

  - Processes rules against delta (new facts only) in each round
  - A derivation is new only if at least one premise comes from delta_old
  - Continues until no new facts are derived (fixed point)
  - Returns the total number of new facts derived

Semi-naive invariant prevents quadratic re-derivation cost.

!!! note "Live-path integration"
    `saturate!` is pure: it operates on an in-memory `KBState` and does NOT touch
    the MORK Space directly. The integration layer wires the write-back —
    [`SCPipeline.execute!`](@ref) Stage 4 serializes each newly-derived fact
    (those failing `is_base_fact`) via `sprint_mcore` and adds it to the live
    Space via `space_add_all_sexpr!`. End-to-end test asserting derived facts
    appear in the live Space dump: `test/integration/test_pipeline.jl :: "opts.saturate_kb derived facts persist back to MORK space (Boundary #1)"`.
"""
function saturate!(kb::KBState; max_rounds::Int=1000)::Int
    total_new = 0

    for round in 1:max_rounds
        delta_old = copy(kb.delta)
        isempty(delta_old) && break   # fixed point reached

        kb.delta = Fact[]
        bump_version!(kb.index)
        kb.version += 1

        new_this_round = 0
        for rule in kb.rules
            new_this_round += _apply_rule_semi_naive!(kb, rule, delta_old)
        end

        total_new += new_this_round
        new_this_round == 0 && break
    end

    total_new
end

"""
Apply one rule using semi-naive strategy: at least one premise from delta_old.
"""
function _apply_rule_semi_naive!(kb::KBState, rule::Rule, delta_old::Vector{Fact})::Int
    n_new = 0
    delta_ids = Set{UInt32}(f.id.idx for f in delta_old)

    for (bindings, used_fact_ids) in _match_body_with_facts(kb, rule.body_ids)
        # Semi-naive invariant: at least one matched FACT must be from delta_old
        any(id -> id.idx in delta_ids, used_fact_ids) || continue

        head_id = _instantiate(kb.g, rule.head_id, bindings)
        isvalid(head_id) || continue
        haskey(kb.facts, head_id.idx) && continue
        # VALUE-based uniqueness gate: `_instantiate` mints a fresh NodeID per derivation, so the
        # NodeID check above never catches a re-derived value. Without this, cyclic data (e.g.
        # edge 0↔1) re-derives the same path forever. Reuse the head index + structural equality.
        any(fid -> _node_equal(kb.g, fid, head_id),
            index_lookup(kb.index, _fact_head(kb.g, head_id))) && continue

        f = Fact(head_id, rule.rule_id, collect(values(bindings)), kb.version)
        kb_add_fact!(kb, f)
        n_new += 1
    end

    n_new
end

# ── Body matching — multi-premise conjunctive query ───────────────────────────

# ── Guard premises — EVALUATED, not looked up (comparison filters) ─────────────
# A premise whose head is a comparison op (`< > <= >= == !=`) is a GUARD: it is
# evaluated against the current bindings and FILTERS the result set, rather than
# being matched against stored facts (a `(< $x 3)` premise has no `(< …)` fact to
# find, so without this it silently blocks the rule). Semantics MIRROR Core's
# `GROUNDED_REGISTRY` (`Primitives.jl _register_comparison!`) so the saturation lane
# bisimulates the MM2 calculus lane: parse both operands as Float64 and compare;
# `==`/`!=` fall back to textual (structural-symbol) (in)equality when non-numeric.
# `!=` has no GROUNDED_REGISTRY counterpart — it is the saturation-IR guard mirroring
# CeTTa's MM2-IL `ir_guard_neq` (kept off the surface, as in every reference runtime).
# Guards bind no variables and add no facts ⇒ monotone, always-terminating filters.
const _GUARD_OPS = Set{Symbol}(Symbol.(["<", ">", "<=", ">=", "==", "!="]))

_is_guard_premise(g::MCoreGraph, id::NodeID)::Bool =
    (n = get_node(g, id); n isa Con && (n::Con).head in _GUARD_OPS)

_atom_text(n)::Union{String, Nothing} =
    n isa Sym ? string((n::Sym).name) :
    n isa Lit ? string((n::Lit).val)  : nothing

# Evaluate a guard premise under `bindings`: true/false when both operands are GROUND
# and comparable, or `nothing` when an operand is still unbound/structured (the
# partial-evaluation "defer" case — an undecidable guard drops the tuple, never
# deriving on an unconfirmed guard).
function _eval_guard_premise(g::MCoreGraph, pid::NodeID,
                             bindings::Dict{Int, NodeID})::Union{Bool, Nothing}
    gcon = get_node(g, _instantiate(g, pid, bindings))
    (gcon isa Con && length((gcon::Con).fields) >= 2) || return nothing
    op = (gcon::Con).head
    sa = _atom_text(get_node(g, (gcon::Con).fields[1]))
    sb = _atom_text(get_node(g, (gcon::Con).fields[2]))
    (sa === nothing || sb === nothing) && return nothing
    fa = tryparse(Float64, sa); fb = tryparse(Float64, sb)
    num = fa !== nothing && fb !== nothing
    op === Symbol("<")  && return num ? (fa <  fb) : nothing
    op === Symbol(">")  && return num ? (fa >  fb) : nothing
    op === Symbol("<=") && return num ? (fa <= fb) : nothing
    op === Symbol(">=") && return num ? (fa >= fb) : nothing
    op === Symbol("==") && return num ? (fa == fb) : (sa == sb)
    op === Symbol("!=") && return num ? (fa != fb) : (sa != sb)
    nothing
end

"""
    _match_body_with_facts(kb, body_ids) -> Vector{Tuple{Dict{Int,NodeID}, Vector{NodeID}}}

Enumerate all complete bindings satisfying all premises in `body_ids`.
Returns (bindings, used_fact_ids) pairs — used_fact_ids tracks which
fact NodeIDs were matched (needed for the semi-naive delta check).

GUARD premises (`_GUARD_OPS`) are partitioned out and applied LAST — after the
relation premises have bound the variables (the partial-evaluation discipline:
evaluate when bound). This also fixes the ordering hazard where `_premise_cardinality`
gives an unindexed guard cardinality 0 and would otherwise sort it first.
"""
function _match_body_with_facts(
    kb::KBState, body_ids::Vector{NodeID}
)::Vector{Tuple{Dict{Int, NodeID}, Vector{NodeID}}}
    isempty(body_ids) && return [(Dict{Int, NodeID}(), NodeID[])]

    guards    = NodeID[id for id in body_ids if _is_guard_premise(kb.g, id)]
    relations = NodeID[id for id in body_ids if !_is_guard_premise(kb.g, id)]
    ordered   = sort(relations; by=id -> _premise_cardinality(kb, id))

    results = Tuple{Dict{Int, NodeID}, Vector{NodeID}}[(Dict{Int, NodeID}(), NodeID[])]

    for pid in ordered
        new_results = Tuple{Dict{Int, NodeID}, Vector{NodeID}}[]
        for (bindings, used) in results
            ground_id = _instantiate(kb.g, pid, bindings)
            for match_id in _match_fact(kb, ground_id)
                merged = _merge_bindings(kb.g, pid, match_id, bindings)
                merged !== nothing && push!(new_results, (merged, [used; match_id]))
            end
        end
        results = new_results
        isempty(results) && return results
    end

    # Guards last: evaluate against the now-bound variables; keep a tuple iff every
    # guard decides true (an undecidable/unbound guard returns nothing → dropped).
    isempty(guards) && return results
    filter(results) do (bindings, _used)
        all(gid -> _eval_guard_premise(kb.g, gid, bindings) === true, guards)
    end
end

# Keep old name for any external callers
_match_body(kb, body_ids) = [b for (b, _) in _match_body_with_facts(kb, body_ids)]

function _premise_cardinality(kb::KBState, pid::NodeID)::Int
    head = _fact_head(kb.g, pid)
    length(index_lookup(kb.index, head))
end

function _match_fact(kb::KBState, ground_id::NodeID)::Vector{NodeID}
    !isvalid(ground_id) && return NodeID[]
    head = _fact_head(kb.g, ground_id)
    filter(fid -> _facts_unify(kb.g, ground_id, fid), index_lookup(kb.index, head))
end

function _facts_unify(g::MCoreGraph, pat_id::NodeID, fact_id::NodeID)::Bool
    !isvalid(pat_id) || !isvalid(fact_id) && return false
    pn = get_node(g, pat_id)
    fn = get_node(g, fact_id)
    pn isa Var && return true   # variable matches anything
    pn isa Sym && fn isa Sym && return (pn::Sym).name == (fn::Sym).name
    pn isa Lit && fn isa Lit && return (pn::Lit).val == (fn::Lit).val
    if pn isa Con && fn isa Con
        pc = pn::Con;
        fc = fn::Con
        pc.head != fc.head && return false
        length(pc.fields) != length(fc.fields) && return false
        return all(_facts_unify(g, pf, ff) for (pf, ff) in zip(pc.fields, fc.fields))
    end
    false
end

# Structural VALUE equality of two M-Core nodes. Two facts can hold the same value (`1`) as DISTINCT
# graph nodes (different NodeIDs), so a shared variable's consistency check must compare by value, not
# by NodeID identity — otherwise a join on a shared var (e.g. `(path $x $y),(edge $y $z)`) never binds.
function _node_equal(g::MCoreGraph, a::NodeID, b::NodeID)::Bool
    a == b && return true
    na = get_node(g, a); nb = get_node(g, b)
    na isa Sym && nb isa Sym && return (na::Sym).name == (nb::Sym).name
    na isa Lit && nb isa Lit && return (na::Lit).val == (nb::Lit).val
    if na isa Con && nb isa Con
        nca = na::Con; ncb = nb::Con
        (nca.head == ncb.head && length(nca.fields) == length(ncb.fields)) || return false
        return all(_node_equal(g, x, y) for (x, y) in zip(nca.fields, ncb.fields))
    end
    return false
end

function _merge_bindings(
    g::MCoreGraph, pat_id::NodeID, fact_id::NodeID, existing::Dict{Int, NodeID}
)::Union{Dict{Int, NodeID}, Nothing}
    pn = get_node(g, pat_id)
    if pn isa Var
        ix = (pn::Var).ix
        if haskey(existing, ix)
            _node_equal(g, existing[ix], fact_id) || return nothing   # conflict (compare by VALUE)
        else
            out = copy(existing)
            out[ix] = fact_id
            return out
        end
        return existing
    end
    if pn isa Con && get_node(g, fact_id) isa Con
        pc = pn::Con;
        fc = get_node(g, fact_id)::Con
        cur = existing
        for (pf, ff) in zip(pc.fields, fc.fields)
            cur = _merge_bindings(g, pf, ff, cur)
            cur === nothing && return nothing
        end
        return cur
    end
    existing   # ground: already checked by _facts_unify
end

function _instantiate(g::MCoreGraph, tmpl_id::NodeID, bindings::Dict{Int, NodeID})::NodeID
    !isvalid(tmpl_id) && return NULL_NODE
    n = get_node(g, tmpl_id)
    if n isa Var
        return get(bindings, (n::Var).ix, tmpl_id)
    end
    if n isa Con
        c = n::Con
        new_fields = NodeID[_instantiate(g, f, bindings) for f in c.fields]
        new_fields == c.fields && return tmpl_id   # unchanged
        return add_con!(g, Con(c.head, new_fields, c.effects))
    end
    tmpl_id   # Sym, Lit, etc. — no variables to substitute
end

export Fact, is_base_fact, Rule
export VersionedIndex, index_insert!, index_lookup, index_delta_since, bump_version!
export KBState, kb_add_fact!, kb_add_rule!, all_facts
export saturate!
