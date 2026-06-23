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
    converged = false

    for round in 1:max_rounds
        delta_old = copy(kb.delta)
        if isempty(delta_old)
            converged = true; break   # fixed point reached
        end

        kb.delta = Fact[]
        bump_version!(kb.index)
        kb.version += 1

        new_this_round = 0
        for rule in kb.rules
            new_this_round += _apply_rule_semi_naive!(kb, rule, delta_old)
        end

        total_new += new_this_round
        if new_this_round == 0
            converged = true; break
        end
    end

    # No silent cap: a value-GENERATING rule with no bounding guard never reaches a fixpoint
    # (the dedup gate collapses identical values, not freshly-minted ones) and gets truncated
    # at the round cap. Surface it — the pragmatic stand-in for the supercompilation
    # homeomorphic-embedding whistle — rather than returning a partial result as if complete.
    converged || @warn "saturate!: reached max_rounds=$max_rounds without a fixpoint — likely an " *
                       "unbounded value-generating rule; add a bounding comparison guard (e.g. (< \$x K)). " *
                       "Result is truncated, not a fixpoint."

    total_new
end

# ── Stratified negation: signed dependency graph → strata → per-stratum saturation ──

# Collect the Var indices appearing anywhere in a premise pattern (for NAF safety).
function _premise_vars!(g::MCoreGraph, id::NodeID, acc::Set{Int})::Set{Int}
    n = get_node(g, id)
    if n isa Var
        push!(acc, (n::Var).ix)
    elseif n isa Con
        for f in (n::Con).fields
            _premise_vars!(g, f, acc)
        end
    end
    acc
end

"""
    _stratify(kb) -> Vector{Vector{Rule}} | nothing

Partition `kb.rules` into strata for stratified negation. Builds the signed predicate
dependency graph — a POSITIVE edge `head → pred` for a positive premise, a NEGATIVE edge
for a `(not (pred …))` premise — and assigns stratum numbers (positive dependency ≥,
negative dependency STRICTLY >). Returns the rules grouped by head-stratum in increasing
order, or `nothing` if the program is NON-stratifiable (a cycle through negation → needs
well-founded semantics, not supported). Warns on an unsafe rule (a negated literal with a
variable not bound by any positive premise).
"""
function _stratify(kb::KBState)::Union{Vector{Vector{Rule}}, Nothing}
    g, rules = kb.g, kb.rules
    derived = Set{Symbol}(_fact_head(g, r.head_id) for r in rules)
    stratum = Dict{Symbol, Int}(p => 0 for p in derived)
    edges = Tuple{Symbol, Symbol, Bool}[]                 # (head, body_pred, is_negative)
    for r in rules
        h = _fact_head(g, r.head_id)
        posvars = Set{Int}()
        for bid in r.body_ids
            if _is_negated_premise(g, bid)
                q = _fact_head(g, (get_node(g, bid)::Con).fields[1])
                q in derived && push!(edges, (h, q, true))
            elseif _is_guard_premise(g, bid)
                # guards bind nothing — no edge, no vars
            elseif _is_arith_premise(g, bid)
                _premise_vars!(g, bid, posvars)           # arith binds its output, reads inputs
            else
                q = _fact_head(g, bid)
                q in derived && push!(edges, (h, q, false))
                _premise_vars!(g, bid, posvars)
            end
        end
        for bid in r.body_ids                              # NAF safety: negated vars positively bound
            _is_negated_premise(g, bid) || continue
            nv = _premise_vars!(g, (get_node(g, bid)::Con).fields[1], Set{Int}())
            unbound = setdiff(nv, posvars)
            isempty(unbound) || @warn "stratified NAF: unsafe rule — negated premise variable(s) " *
                "$(collect(unbound)) not bound by a positive premise; result may be unsound."
        end
    end
    n = length(derived)
    for _ in 1:(n + 1)                                     # relax stratum numbers to a fixed point
        changed = false
        for (h, q, neg) in edges
            want = neg ? stratum[q] + 1 : stratum[q]
            stratum[h] < want && (stratum[h] = want; changed = true)
        end
        changed || break
    end
    for (h, q, neg) in edges                               # constraint check: neg ⇒ strictly higher
        neg && stratum[q] >= stratum[h] && return nothing  # cycle through negation → non-stratifiable
    end
    maxs = maximum(values(stratum); init = 0)
    strata = [Rule[] for _ in 0:maxs]
    for r in rules
        push!(strata[stratum[_fact_head(g, r.head_id)] + 1], r)
    end
    filter(!isempty, strata)
end

"""
    saturate_stratified!(kb; max_rounds) -> Int

Stratified saturation for programs with negated premises: saturate one stratum at a time
(lower strata frozen-complete before higher), so a `(not (p …))` premise is evaluated only
once `p`'s relation is fully derived — the closed-world reading that makes NAF sound. Falls
back to flat `saturate!` (with a warning) when the program is non-stratifiable.
"""
function saturate_stratified!(kb::KBState; max_rounds::Int=1000)::Int
    strata = _stratify(kb)
    if strata === nothing
        @warn "stratified NAF: program is NON-stratifiable (recursion through negation) — needs " *
              "well-founded semantics (tnot), not supported; running flat saturation (negation unsound here)."
        return saturate!(kb; max_rounds)
    end
    saved, total = kb.rules, 0
    for stratum_rules in strata
        kb.rules = stratum_rules
        kb.delta = collect(values(kb.facts))   # all prior-stratum facts are this stratum's frontier
        total += saturate!(kb; max_rounds)
    end
    kb.rules = saved
    total
end

# Does the program use negation (any rule with a `(not …)` premise)?
_program_has_negation(kb::KBState)::Bool =
    any(r -> any(bid -> _is_negated_premise(kb.g, bid), r.body_ids), kb.rules)

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

# ── Arithmetic premises — 3-arg MODED functions `(op a b c)` that BIND the output c ──
# `(+ $x $y $z)` is the moded lowering of `(= $z (+ $x $y))` — PeTTa compiles `+`→3-arg
# `is/2` identically and CeTTa keeps the moded form in its MM2-IL. Inputs bound + output
# unbound ⇒ COMPUTE and bind output; all bound ⇒ CHECK (filter). Float64-internal then
# Int-narrowed string — identical to GROUNDED_REGISTRY (`_register_arithmetic!`). ⚠️ A
# value-GENERATING rule (output feeds the head and recurses) must carry a bounding
# comparison guard or saturation diverges — the dedup gate collapses identical values,
# not freshly-minted ones; `saturate!` warns if it hits the round cap still-deriving
# (the pragmatic stand-in for the homeomorphic-embedding whistle, per the theory digest).
const _ARITH_OPS = Dict{Symbol, Function}(
    Symbol("+") => (+), Symbol("-") => (-), Symbol("*") => (*),
    Symbol("/") => (/), Symbol("%") => rem)

_is_arith_premise(g::MCoreGraph, id::NodeID)::Bool =
    (n = get_node(g, id); n isa Con && (n::Con).head in keys(_ARITH_OPS) &&
                          length((n::Con).fields) == 3)

# Resolve a field NodeID through bindings to a numeric value (or `nothing` if unbound/non-numeric).
function _num_arg(g::MCoreGraph, id::NodeID, bindings::Dict{Int, NodeID})
    n = get_node(g, id)
    if n isa Var
        haskey(bindings, (n::Var).ix) || return nothing
        n = get_node(g, bindings[(n::Var).ix])
    end
    t = _atom_text(n); t === nothing && return nothing
    tryparse(Float64, t)
end

# Apply `(op a b c)` under bindings: returns updated bindings (output bound / check passed),
# `nothing` to DROP (failed check), or `:defer` when an input isn't bound yet.
function _apply_arith_premise(g::MCoreGraph, pid::NodeID, bindings::Dict{Int, NodeID})
    con = get_node(g, pid)::Con
    fa = _num_arg(g, con.fields[1], bindings)
    fb = _num_arg(g, con.fields[2], bindings)
    (fa === nothing || fb === nothing) && return :defer
    r    = _ARITH_OPS[con.head](fa, fb)
    rstr = isinteger(r) ? string(Int(r)) : string(r)
    cn = get_node(g, con.fields[3])
    if cn isa Var
        ix = (cn::Var).ix
        haskey(bindings, ix) || begin
            nb = copy(bindings); nb[ix] = add_sym!(g, Sym(Symbol(rstr))); return nb
        end
        return _atom_text(get_node(g, bindings[ix])) == rstr ? bindings : nothing
    end
    return _atom_text(cn) == rstr ? bindings : nothing
end

# ── Negated premises — `(not (p …))` stratified negation-as-failure ────────────
# A `(not inner)` premise SUCCEEDS for a binding iff NO fact matches `inner` under it
# (anti-join / closed-world). Sound only under STRATIFIED evaluation: `inner`'s predicate
# must live in a strictly-lower, already-saturated stratum, so "no match" is final, not
# transient (see `saturate_stratified!` / `_stratify`). Safety: `inner`'s variables must be
# bound by positive premises (warned in `_stratify`). Detected by head, like guards/arith.
_is_negated_premise(g::MCoreGraph, id::NodeID)::Bool =
    (n = get_node(g, id); n isa Con && (n::Con).head === :not && length((n::Con).fields) == 1)

# NAF holds (keep the tuple) iff the instantiated inner pattern has NO matching fact.
_eval_negated_premise(kb::KBState, pid::NodeID, bindings::Dict{Int, NodeID})::Bool =
    isempty(_match_fact(kb, _instantiate(kb.g, (get_node(kb.g, pid)::Con).fields[1], bindings)))

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
    arith     = NodeID[id for id in body_ids if _is_arith_premise(kb.g, id)]
    negated   = NodeID[id for id in body_ids if _is_negated_premise(kb.g, id)]
    relations = NodeID[id for id in body_ids
                       if !_is_guard_premise(kb.g, id) && !_is_arith_premise(kb.g, id) &&
                          !_is_negated_premise(kb.g, id)]
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

    # Arithmetic premises, applied in READINESS order — each `(op a b c)` binds its output
    # once its inputs are bound (partial evaluation); chained arith resolves over iterations.
    # Binding structure is uniform across results (same relations bind the same vars), so a
    # premise is ready/deferred for all results together — test on the first.
    remaining = arith
    while !isempty(remaining) && !isempty(results)
        ready = NodeID[]; deferred = NodeID[]
        for ap in remaining
            (_apply_arith_premise(kb.g, ap, results[1][1]) === :defer) ?
                push!(deferred, ap) : push!(ready, ap)
        end
        isempty(ready) && break    # no progress: remaining inputs never bind
        for ap in ready
            nr = Tuple{Dict{Int, NodeID}, Vector{NodeID}}[]
            for (b, u) in results
                nb = _apply_arith_premise(kb.g, ap, b)
                (nb !== nothing && nb !== :defer) && push!(nr, (nb::Dict{Int, NodeID}, u))
            end
            results = nr
        end
        remaining = deferred
    end
    isempty(remaining) || return Tuple{Dict{Int, NodeID}, Vector{NodeID}}[]  # unsatisfiable arith

    # Guards + negation last: evaluate against the now-bound variables. Keep a tuple iff every
    # guard decides true AND every negated premise holds (its inner pattern has no matching fact).
    (isempty(guards) && isempty(negated)) && return results
    filter(results) do (bindings, _used)
        all(gid -> _eval_guard_premise(kb.g, gid, bindings) === true, guards) &&
            all(nid -> _eval_negated_premise(kb, nid, bindings), negated)
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
