"""
BoundedSplit — statistics-guided bounded splitting for the supercompiler.

Implements §6.2 of the MM2 Supercompiler spec (Goertzel, Oct 2025):
  Algorithm 9 — BoundedSplit (Split + SplitChoiceBounded)

Purpose: prevent combinatorial explosion when the structural stepper hits
a `Choice` node or an undetermined pattern match.  Instead of exploring
ALL branches, select only the most probable ones up to `budget` AND until
cumulative probability ≥ 0.95.  A catch-all residual is added for soundness.

This is the algorithmic answer to Rule-of-64:
  Without BoundedSplit: 5-source patterns → O(K^5) branches explored
  With BoundedSplit:    selects top-p branches (cumulative ≥ 0.95) → tractable

Key constants from spec §6.2:
  SPLIT_PROB_THRESHOLD  = 0.95  — stop adding branches once P ≥ this
  SPLIT_DEFAULT_BUDGET  = 16    — max branches regardless of probability

The probability of each branch comes from MORKStatistics:
  - For Choice nodes: guard patterns estimated via predicate_counts
  - For symbolic KB splits: pattern shape histogram
  - For MORK exec sources: dynamic_count / total_atoms
"""

# ── Branch descriptor ─────────────────────────────────────────────────────────

"""
    Branch

One branch of a split.  `probability` is the estimated fraction of
executions that take this branch.

id          — NodeID of the expression in this branch
env         — variable bindings in scope for this branch
probability — estimated probability in [0, 1]
is_catchall — true for the catch-all residual branch (added for soundness)
"""
struct Branch
    id::NodeID
    env::Env
    probability::Float64
    is_catchall::Bool
end
Branch(id, env, p) = Branch(id, env, p, false)
Branch_catchall(id, env) = Branch(id, env, 0.0, true)

# ── Split result ──────────────────────────────────────────────────────────────

"""
    SplitResult

Outcome of a bounded split:
branches        — selected branches (sorted by descending probability)
total_prob      — sum of selected branch probabilities (< 1 if catchall added)
budget_hit      — true if the budget limit stopped selection
catchall_added  — true if a catch-all residual was appended for soundness
"""
struct SplitResult
    branches::Vector{Branch}
    total_prob::Float64
    budget_hit::Bool
    catchall_added::Bool
end

# Constants from spec §6.2
const SPLIT_PROB_THRESHOLD = 0.95
const SPLIT_DEFAULT_BUDGET = 16

# ── Algorithm 9 — BoundedSplit (§6.2) ──────────────────────────────────────────

"""
    bounded_split(g, id, env, stats; budget) -> SplitResult

Algorithm 9 (BoundedSplit) from §6.2.  Dispatches on node kind:

Choice(alts)            → split_choice_bounded
Match(scrut, _) if scrut not yet a value → split_match_symbolic
Prim(:kb_query, [pat])  → split_kb_symbolic
_                       → single-branch split (no splitting needed)
"""
function bounded_split(
    g::MCoreGraph, id::NodeID, env::Env, stats::MORKStatistics;
    budget::Int=SPLIT_DEFAULT_BUDGET
)::SplitResult
    !isvalid(id) && return SplitResult([Branch(id, env, 1.0)], 1.0, false, false)

    node = get_node(g, id)

    if node isa Choice
        return _split_choice_bounded(g, node::Choice, id, env, stats, budget)
    elseif node isa MatchNode
        return _split_match_symbolic(g, node::MatchNode, id, env, stats, budget)
    elseif node isa Prim && (node::Prim).op == :kb_query
        return _split_kb_symbolic(g, node::Prim, id, env, stats, budget)
    else
        return SplitResult([Branch(id, env, 1.0)], 1.0, false, false)
    end
end

# ── SplitChoiceBounded (§6.2 Algorithm 9) ─────────────────────────────────────

function _split_choice_bounded(g, node::Choice, id, env, stats, budget)::SplitResult
    alts = node.alts

    # Estimate probability for each alternative
    probs = [_estimate_guard_prob(g, alt.guard, stats) for alt in alts]

    # Sort descending by probability
    perm = sortperm(probs; rev=true)
    sorted = [(alts[i], probs[i]) for i in perm]

    selected = Branch[]
    cumulative = 0.0
    budget_hit = false

    for (alt, prob) in sorted
        if length(selected) >= budget
            budget_hit = true
            break
        end
        cumulative >= SPLIT_PROB_THRESHOLD && break

        push!(selected, Branch(alt.expr, env, prob))
        cumulative += prob
    end

    # Add catch-all for soundness if we didn't cover all branches
    catchall_added = false
    if cumulative < 1.0
        # Catch-all: a residual Choice covering the remaining alts
        remaining = [alts[perm[i]] for i in (length(selected) + 1):length(sorted)]
        if !isempty(remaining)
            catchall_id = add_choice!(g, Choice(remaining, node.effects))
            push!(selected, Branch_catchall(catchall_id, env))
            catchall_added = true
        end
    end

    SplitResult(selected, cumulative, budget_hit, catchall_added)
end

# ── SplitMatchSymbolic ─────────────────────────────────────────────────────────

function _split_match_symbolic(g, node::MatchNode, id, env, stats, budget)::SplitResult
    # For each clause, estimate probability via guard
    clauses = node.clauses
    probs = [_estimate_clause_prob(g, c, stats) for c in clauses]
    perm = sortperm(probs; rev=true)

    selected = Branch[]
    cumulative = 0.0
    budget_hit = false

    for i in perm
        length(selected) >= budget && (budget_hit=true; break)
        cumulative >= SPLIT_PROB_THRESHOLD && break

        c = clauses[i]
        prob = probs[i]
        # Build a single-clause match for this branch
        branch_id = add_match!(g, MatchNode(node.scrut, [c], node.effects))
        push!(selected, Branch(branch_id, env, prob))
        cumulative += prob
    end

    catchall_added = false
    if cumulative < 1.0
        remaining_cs = [clauses[perm[i]] for i in (length(selected) + 1):length(perm)]
        if !isempty(remaining_cs)
            ca_id = add_match!(g, MatchNode(node.scrut, remaining_cs, node.effects))
            push!(selected, Branch_catchall(ca_id, env))
            catchall_added = true
        end
    end

    SplitResult(selected, cumulative, budget_hit, catchall_added)
end

# ── SplitKBSymbolic ───────────────────────────────────────────────────────────

"""
    _split_kb_symbolic

For a KB query primitive, split by predicate: each distinct predicate head
that could match gets its own branch, weighted by cardinality.

This implements the "symbolic split by predicate/mode" described in §2.2
("Bounded Splitting uses symbolic splits, not fact enumeration").
"""
function _split_kb_symbolic(g, node::Prim, id, env, stats, budget)::SplitResult
    # Pattern is args[1]
    isempty(node.args) && return SplitResult([Branch(id, env, 1.0)], 1.0, false, false)

    pat_id = node.args[1]
    total = max(1, stats.total_atoms)
    branches = Branch[]

    # Each predicate in stats is a potential match branch
    sorted_preds = sort(collect(predicate_counts(stats)); by=x->-x[2])
    cumulative = 0.0
    budget_hit = false

    for (pred, count) in sorted_preds
        length(branches) >= budget && (budget_hit=true; break)
        cumulative >= SPLIT_PROB_THRESHOLD && break

        prob = count / total
        # Build specialized query: same pattern but with predicate hint
        hint_id = add_sym!(g, Sym(pred))
        spec_id = add_prim!(g, Prim(:kb_query_pred, [pat_id, hint_id], node.effects))
        push!(branches, Branch(spec_id, env, prob))
        cumulative += prob
    end

    if isempty(branches)
        return SplitResult([Branch(id, env, 1.0)], 1.0, false, false)
    end

    catchall_added = false
    if cumulative < 1.0
        push!(branches, Branch_catchall(id, env))
        catchall_added = true
    end

    SplitResult(branches, cumulative, budget_hit, catchall_added)
end

# ── Probability estimation helpers ───────────────────────────────────────────

function _estimate_guard_prob(
    g::MCoreGraph, guard_id::NodeID, stats::MORKStatistics
)::Float64
    !isvalid(guard_id) && return 1.0 / max(1, stats.total_atoms ÷ 4)

    node = get_node(g, guard_id)
    node isa Sym && return 0.5   # unknown guard: uniform prior

    if node isa Prim && node.op == :kb_query
        isempty(node.args) && return 0.1
        pat = node.args[1]
        snode = parse_sexpr(sprint_mcore(g, pat))
        card = estimate_cardinality(snode, stats)
        return clamp(card / max(1, stats.total_atoms), 0.0, 1.0)
    end

    0.5   # fallback: uniform
end

function _estimate_clause_prob(
    g::MCoreGraph, clause::MatchClause, stats::MORKStatistics
)::Float64
    _estimate_guard_prob(g, clause.guard, stats)
end

# ── Minimal M-Core → sexpr serializer (for probability estimation) ────────────

"""
Sprint a NodeID as a rough sexpr string (used for cardinality lookup only).
"""
function sprint_mcore(g::MCoreGraph, id::NodeID)::String
    !isvalid(id) && return "nil"
    node = get_node(g, id)
    if node isa Sym
        ;
        return string(node.name)
    elseif node isa Lit
        ;
        return string(node.val)
    elseif node isa Var
        ;
        return "\$x$(node.ix)"
    elseif node isa Con
        parts = join([sprint_mcore(g, f) for f in node.fields], " ")
        return "($(node.head) $parts)"
    elseif node isa Prim
        parts = join([sprint_mcore(g, a) for a in node.args], " ")
        return "($(node.op) $parts)"
    end
    "?"
end

export Branch, Branch_catchall, SplitResult
export bounded_split
export SPLIT_PROB_THRESHOLD, SPLIT_DEFAULT_BUDGET
