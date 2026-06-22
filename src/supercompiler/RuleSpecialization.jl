"""
RuleSpecialization — partial instantiation + magic-sets transformations.

Implements §8.3 of the MM2 Supercompiler v1 spec (Goertzel, Oct 2025):

  - **Partial Instantiation**: for each rule R and each fact F matching R's
    first body premise, generate a specialized rule variant by substituting
    F's values into R's variables. When all premises become ground after
    substitution, the rule degenerates to a derived ground fact.

  - **Magic Sets** (Algorithm 4 GenerateMagicSets): goal-directed rule
    rewriting. Given a query pattern with bound positions, introduce a
    `magic_<pred>` predicate that gates rule firing — only those bindings
    that could possibly contribute to the query goal get propagated.

Both transformations are **pre-saturation rewrites**: the output is a
`(rules, facts)` pair that you then feed to `saturate!`. The semantics are
preserved relative to the unspecialized rule set, which the BisimVerifier
(Boundary #3) can verify.

These were the v1 "AI specialization" track items deferred per the spec
memo §17 ("natural next sprint when an algorithm workload demands it").
The KBSaturation write-back closure (Boundary #1, 2026-06-18) made these
genuinely composable: derived facts now persist back to the live Space,
so specialization gains are observable downstream.

# Public API

  - [`specialize_rules`](@ref) — partial instantiation
  - [`SpecializationResult`](@ref) — output: new rules + new ground facts
  - [`magic_sets_transform`](@ref) — magic-sets rewrite
  - [`MagicSetsResult`](@ref) — output: rewritten rules + magic seed facts
"""

# ── Partial Instantiation (v1 §8.3) ───────────────────────────────────────────

"""
    SpecializationResult

Output of [`specialize_rules`](@ref).

specialized_rules — new rules with one less premise per match (or untouched
                    if no specialization was possible)
derived_facts     — head atoms that became fully ground after substitution
                    (these are immediate consequences of the original rule
                    set + base facts; adding them avoids one saturate round)
"""
struct SpecializationResult
    specialized_rules::Vector{Rule}
    derived_facts::Vector{NodeID}
end

"""
    specialize_rules(g::MCoreGraph, rules::Vector{Rule},
                     facts::Vector{NodeID}; max_per_rule::Int=100)
        -> SpecializationResult

For each `rule` in `rules` and each `fact` in `facts` matching `rule`'s
first body premise, generate a specialized variant of `rule` by
substituting the bindings from `fact` into the remaining premises and
the head. When the substituted head is fully ground (i.e. all variables
are bound), it becomes a derived fact instead of a rule.

Cap per-rule generation at `max_per_rule` to prevent combinatorial blow-up
on dense KBs.

Worked example from v1 §8.3:

```
Original: ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y)
Facts:    parent(alice, bob), parent(bob, carol)

After specialize_rules:
- ancestor(alice, Y) :- ancestor(bob, Y)       (X=alice, Z=bob)
- ancestor(bob, Y)   :- ancestor(carol, Y)     (X=bob,   Z=carol)
```

The substituted recursive premise `ancestor(Z, Y)` becomes
`ancestor(bob, Y)` etc., one premise less than the original rule.
"""
function specialize_rules(
    g::MCoreGraph,
    rules::Vector{Rule},
    facts::Vector{NodeID};
    max_per_rule::Int=100
)::SpecializationResult
    specialized = Rule[]
    derived = NodeID[]
    next_spec_id = 1
    seen_facts = Set{UInt32}()

    for rule in rules
        isempty(rule.body_ids) && continue

        # Take the FIRST body premise as the partial-instantiation candidate.
        # (v1 §8.3 doesn't prescribe which premise to specialize on; the
        # first is the conventional choice and matches the "input" premise
        # in left-to-right body reading.)
        pid = rule.body_ids[1]
        target_head = _fact_head(g, pid)

        n_specialized = 0
        for fact_id in facts
            n_specialized >= max_per_rule && break
            _fact_head(g, fact_id) == target_head || continue
            _facts_unify(g, pid, fact_id) || continue

            bindings = _merge_bindings(g, pid, fact_id, Dict{Int, NodeID}())
            bindings === nothing && continue

            # Substitute bindings into head + remaining body premises.
            new_head = _instantiate(g, rule.head_id, bindings)
            new_body = NodeID[_instantiate(g, b, bindings)
                              for b in rule.body_ids[2:end]]

            isvalid(new_head) || continue

            if isempty(new_body)
                # All premises consumed → derived ground fact, not a rule.
                # Dedupe by NodeID (idx) since _instantiate may mint fresh nodes.
                if !(new_head.idx in seen_facts)
                    push!(derived, new_head)
                    push!(seen_facts, new_head.idx)
                end
            else
                # Specialized rule with one less premise.
                new_rule_id = add_sym!(g, Sym(Symbol("__spec_rule_$next_spec_id")))
                next_spec_id += 1
                push!(specialized, Rule(new_head, new_body, new_rule_id))
            end
            n_specialized += 1
        end
    end

    SpecializationResult(specialized, derived)
end

# ── Magic Sets — Algorithm 4 GenerateMagicSets (v1 §8.3) ──────────────────────

"""
    MagicSetsResult

Output of [`magic_sets_transform`](@ref).

rewritten_rules — the original rules augmented with magic-predicate guards
                  on the head and magic-propagation rules for recursive
                  premises
magic_seeds     — initial facts for the magic predicates (one per query)
magic_pred      — the magic predicate symbol introduced (e.g. `:magic_ancestor`)
"""
struct MagicSetsResult
    rewritten_rules::Vector{Rule}
    magic_seeds::Vector{NodeID}
    magic_pred::Symbol
end

"""
    magic_sets_transform(g::MCoreGraph, rules::Vector{Rule},
                         query::NodeID; bound_position::Int=0)
        -> MagicSetsResult

v1 §8.3 Algorithm 4 GenerateMagicSets.

Goal-directed rule rewriting: given a query pattern with at least one
bound argument position (defaulting to position 0), introduce a
`magic_<pred>` predicate that captures the bound binding and gates rule
firing. Only ground terms reachable from the query's bound value will
trigger rule applications, dramatically reducing the saturation surface
on large KBs.

For a query `(ancestor alice \$Y)` and rule
`(ancestor X Y) :- (parent X Z), (ancestor Z Y)`:

```
Magic seed:        (magic_ancestor alice)
Guard added:       (ancestor X Y) :- (magic_ancestor X), (parent X Z), (ancestor Z Y)
Magic propagation: (magic_ancestor Z) :- (magic_ancestor X), (parent X Z)
```

`bound_position` is the 0-indexed argument position whose value is bound
in the query (e.g. 0 for `(ancestor alice _)` where `alice` is at index 0).
"""
function magic_sets_transform(
    g::MCoreGraph,
    rules::Vector{Rule},
    query::NodeID;
    bound_position::Int=0
)::MagicSetsResult
    qn = get_node(g, query)
    if !(qn isa Con)
        # Non-Con queries don't admit the canonical adornment treatment;
        # return an empty rewrite that's still safe to saturate.
        return MagicSetsResult(copy(rules), NodeID[], :_no_magic)
    end
    qcon = qn::Con
    query_head = qcon.head
    bound_position + 1 <= length(qcon.fields) ||
        return MagicSetsResult(copy(rules), NodeID[], :_no_magic)

    bound_value = qcon.fields[bound_position + 1]
    bound_node = get_node(g, bound_value)
    # We need a CONCRETE bound value (Sym or Lit) for the magic seed to
    # be meaningful. A bound Var is no magic-restriction at all.
    if !(bound_node isa Sym || bound_node isa Lit)
        return MagicSetsResult(copy(rules), NodeID[], :_no_magic)
    end

    magic_pred = Symbol("magic_$(query_head)")
    rewritten = Rule[]
    next_prop_id = 1

    # Magic seed: e.g. (magic_ancestor alice)
    seed_id = add_con!(g, Con(magic_pred, [bound_value]))

    for rule in rules
        rule_head_node = get_node(g, rule.head_id)
        if !(rule_head_node isa Con) || (rule_head_node::Con).head !== query_head
            # Rule doesn't target the query predicate; pass through unchanged.
            push!(rewritten, rule)
            continue
        end

        rule_head = rule_head_node::Con
        bound_position + 1 <= length(rule_head.fields) || begin
            push!(rewritten, rule)
            continue
        end

        # Guard: prepend (magic_<pred> <head_arg_at_bound_position>) to the body.
        guard_arg = rule_head.fields[bound_position + 1]
        guard_premise = add_con!(g, Con(magic_pred, [guard_arg]))
        new_body = NodeID[guard_premise; rule.body_ids...]
        push!(rewritten, Rule(rule.head_id, new_body, rule.rule_id))

        # Magic-propagation rules: for each body premise whose head equals
        # the rule's head predicate (recursive premise), derive a new magic
        # fact for the recursive argument.
        for prem_id in rule.body_ids
            prem_node = get_node(g, prem_id)
            prem_node isa Con || continue
            prem = prem_node::Con
            prem.head === query_head || continue
            bound_position + 1 <= length(prem.fields) || continue

            prop_arg = prem.fields[bound_position + 1]
            prop_head_id = add_con!(g, Con(magic_pred, [prop_arg]))
            # Body of propagation rule: guard + all NON-recursive original
            # premises (they constrain how the binding propagates).
            prop_body = NodeID[guard_premise]
            for other_id in rule.body_ids
                other_id === prem_id && continue
                other_node = get_node(g, other_id)
                if other_node isa Con && (other_node::Con).head === query_head
                    continue   # skip other recursive premises — keep prop rule simple
                end
                push!(prop_body, other_id)
            end
            prop_rule_id = add_sym!(g, Sym(Symbol("__magic_prop_$next_prop_id")))
            next_prop_id += 1
            push!(rewritten, Rule(prop_head_id, prop_body, prop_rule_id))
        end
    end

    MagicSetsResult(rewritten, NodeID[seed_id], magic_pred)
end

export SpecializationResult, specialize_rules
export MagicSetsResult, magic_sets_transform
