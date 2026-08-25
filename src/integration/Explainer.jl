"""
Explainer — debugging and visualization for the supercompiler.

Implements §10.4 debugging tools from the spec.

  explain(s, program)  → human-readable plan report (join order, cardinalities,
                             canonical keys, split decisions)
  to_dot(program)         → Graphviz DOT source for the source-dependency graph
  diff_programs(before, after)  → diff two programs showing what changed after planning
"""

using MORK: Space

# ── explain ────────────────────────────────────────────────────────────────

"""
    explain(s::Space, program::AbstractString) -> String

Full explanation of what the supercompiler does to `program`:

  - Join order for each multi-source conjunction (with cardinality estimates)
  - Which sources were reordered and why
  - Canonical key preview for termination analysis
  - BoundedSplit threshold note for any Choice/multi-source patterns
"""
function explain(s::Space, program::AbstractString)::String
    io = IOBuffer()
    # (no `collect_stats` — the join nodes below come from `s.btm` directly. Calling it here would
    # walk the trie to build statistics nothing reads.)
    nodes = parse_program(program)

    println(io, "═══════════════════════════════════════════════════════")
    println(io, "  MorkSupercompiler — Plan Explanation")
    println(
        io,
        "  Space: $(space_val_count(s)) atoms  |  Program: $(length(nodes)) top-level atoms"
    )
    println(io, "═══════════════════════════════════════════════════════")

    any_multi = false
    for (atom_idx, node) in enumerate(nodes)
        node isa SList || continue
        items = (node::SList).items
        length(items) < 3 && continue
        ci = conjunction_index(items)
        ci === nothing && continue
        conj = items[ci]::SList
        sources = conj.items[2:end]
        length(sources) <= 1 && continue
        any_multi = true

        head_str = sprint_sexpr(items[1])
        println(io, "\n[$atom_idx] $(head_str)")
        println(io, "  Sources ($(length(sources)) — Rule-of-64 risk if ≥ 5):")

        # 🔴 DYNAMIC, not stats (2026-08-24). `explain` HAS the Space, so `s.btm` gives EXACT
        # per-source subtrie counts. The stats estimator cannot be used honestly here — see N3 in
        # docs/AUDIT_DOC1.md: `collect_stats` samples a TRIE-ORDER PREFIX and scales it as if
        # uniform, so on a 4020-atom space it reported ("a",3) => 4020 (truth 2020) and never saw
        # predicate "b" at all. An explanation printing "card≈" from that is confidently wrong.
        jnodes = build_join_nodes_dynamic(sources, s.btm)
        perm = plan_join_order(jnodes)
        scores = static_score.(sources)
        already_sorted = issorted([jnodes[i].cardinality for i in perm])

        for (k, (src, jn)) in enumerate(zip(sources, jnodes))
            marker = k in perm[1:1] ? "→ first (most selective)" : ""
            println(
                io,
                "    [orig $k] card=$(jn.cardinality) (exact)  static=$(round(scores[k]; digits=2))  $(sprint_sexpr(src))  $marker"
            )
        end

        println(io, "  Planned order: $(join(["[$i]" for i in perm], " → "))")
        if perm == collect(1:length(sources))
            println(io, "  ✓ Already optimal — no reordering needed")
        else
            println(io, "  ↑ Reordered — estimated cardinality improvement")
            _explain_variable_flow(io, sources, jnodes, perm)
        end

        if length(sources) >= 5
            println(
                io,
                "  ⚠ Rule-of-64: $(length(sources)) sources → BoundedSplit will prune at cumulative ≥ $(SPLIT_PROB_THRESHOLD)"
            )
        end
    end

    !any_multi && println(io, "\n  (no multi-source conjunctions found — nothing to plan)")

    println(io, "\n═══════════════════════════════════════════════════════")
    String(take!(io))
end

function _explain_variable_flow(io, sources, jnodes, perm)
    println(io, "  Variable flow in planned order:")
    bound = Set{String}()
    for (rank, i) in enumerate(perm)
        jn = jnodes[i]
        src = sources[i]
        new_binds = setdiff(jn.vars_out, bound)
        used_bound = intersect(jn.vars_in, bound)
        semi = isempty(used_bound) ? "" : "  semi-join on $(join(collect(used_bound), ","))"
        println(io, "    step $rank: $(sprint_sexpr(src))")
        !isempty(new_binds) &&
            println(io, "             introduces: $(join(collect(new_binds), ", "))")
        !isempty(semi) && println(io, "            $semi")
        union!(bound, jn.vars_out)
    end
end

# ── to_dot ────────────────────────────────────────────────────────────────────

"""
    to_dot(program::AbstractString; stats=nothing) -> String

Generate a Graphviz DOT source string visualizing the source-dependency graph
for all multi-source conjunctions in `program`.

Each source is a node; directed edges show variable bindings flowing between
sources in the planned order.  Node color encodes estimated cardinality
(green = low, yellow = medium, red = high).

Paste the output into https://dreampuf.github.io/GraphvizOnline/ to visualize.
"""
# ⚠️ `to_dot` takes NO Space, so unlike `explain` it CANNOT use the exact trie counts and is stuck
# with the stats estimator. Its cardinalities inherit N3 (docs/AUDIT_DOC1.md): `collect_stats`
# samples a trie-order prefix and scales it as if uniform, so a predicate sorting late in the trie
# is invisible and one sorting early is credited with the whole space. Read the numbers it draws as
# SHAPE-LEVEL and possibly wrong, not as measurements. Routing it would mean adding a Space
# parameter — an API change, deliberately not made here.
function to_dot(
    program::AbstractString; stats::Union{MORKStatistics, Nothing}=nothing
)::String
    io = IOBuffer()
    nodes = parse_program(program)
    stats = stats !== nothing ? stats : MORKStatistics()

    println(io, "digraph SCPlan {")
    println(io, "  rankdir=LR;")
    println(io, "  node [shape=box, fontname=\"Courier\", fontsize=10];")
    println(io, "  edge [fontname=\"Courier\", fontsize=9];")

    atom_idx = 0
    for node in nodes
        node isa SList || continue
        items = (node::SList).items
        (length(items) < 3 || !is_conjunction(items[2])) && continue
        conj = items[2]::SList
        sources = conj.items[2:end]
        length(sources) <= 1 && continue

        atom_idx += 1
        head_str = _dot_label(sprint_sexpr(items[1]))
        jnodes = build_join_nodes(sources, stats)
        perm = plan_join_order(jnodes)

        println(io, "\n  subgraph cluster_$atom_idx {")
        println(io, "    label=\"$(head_str)\";")
        println(io, "    style=rounded; color=gray;")

        # Source nodes
        for (k, (src, jn)) in enumerate(zip(sources, jnodes))
            color = _cardinality_color(jn.cardinality, stats.total_atoms)
            pos = findfirst(==(k), perm)
            label = "$(pos !== nothing ? "step $pos" : "orig $k")\\n$(sprint_sexpr(src))\\ncard≈$(jn.cardinality)"
            println(
                io,
                "    n$(atom_idx)_$k [label=\"$(_dot_label(label))\", fillcolor=$color, style=filled];"
            )
        end

        # Edges showing planned flow
        for (rank, i) in enumerate(perm[1:(end - 1)])
            j = perm[rank + 1]
            shared = intersect(jnodes[i].vars_out, jnodes[j].vars_in)
            edge_label = isempty(shared) ? "" : join(collect(shared), ",")
            println(
                io,
                "    n$(atom_idx)_$i -> n$(atom_idx)_$j [label=\"$(_dot_label(edge_label))\"];"
            )
        end

        println(io, "  }")
    end

    println(io, "}")
    String(take!(io))
end

_dot_label(s::String) = replace(s, "\"" => "'", "\n" => "\\n", "<" => "\\<", ">" => "\\>")

function _cardinality_color(card::Int, total::Int)::String
    total <= 0 && return "white"
    frac = card / total
    if frac < 0.1
        "\"#90EE90\""   # green
    elseif frac < 0.3
        "\"#FFFF99\""   # yellow
    else   # yellow
        "\"#FFB6C1\""      # red
    end      # red
end

# ── diff_programs ───────────────────────────────────────────────────────────────────

"""
    diff_programs(original::AbstractString, planned::AbstractString) -> String

Show what changed between the original program and its planned version.
Each changed atom is printed as:

  - original source order

  - planned source order
"""
function diff_programs(original::AbstractString, planned::AbstractString)::String
    io = IOBuffer()
    orig_n = parse_program(original)
    plan_n = parse_program(planned)

    any_diff = false
    for (i, (on, pn)) in enumerate(zip(orig_n, plan_n))
        os = sprint_sexpr(on)
        ps = sprint_sexpr(pn)
        os == ps && continue
        any_diff = true
        println(io, "atom $i:")
        println(io, "- $os")
        println(io, "+ $ps")
    end

    !any_diff && println(io, "(no changes — program was already optimal)")
    String(take!(io))
end

export explain, to_dot, diff_programs
