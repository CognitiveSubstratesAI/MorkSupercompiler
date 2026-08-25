"""
PipelineDecompose — split N-source conjunction patterns into chained exec stages.

This is the MORK integration for the Rule-of-64 fix.

Problem: MORK's ProductZipper runs O(K^N) for N sources each matching K atoms.
Solution: Decompose `(exec id (, src1...srcN) (, tpl))` into a chain of
smaller execs that MORK handles efficiently:

  Original (N=5):
    (exec id (, src1 src2 src3 src4 src5) (, tpl))
    → ProductZipper: O(K^5) = intractable

  Decomposed (2-2-2 chain):
    (exec _sc_0 (, src1 src2) (, (_sc_tmp0 \$shared_vars01)))
    (exec _sc_1 (, (_sc_tmp0 \$shared_vars01) src3 src4) (, (_sc_tmp1 \$shared_vars14)))
    (exec _sc_2 (, (_sc_tmp1 \$shared_vars14) src5) (, tpl))
    → Each stage: ProductZipper O(K^2) — tractable

The intermediate `_sc_tmp0`, `_sc_tmp1` atoms store partial bindings in the
MORK space between stages. This uses MORK's existing mechanism (atom storage)
to implement semi-join pushdown without any MORK code changes.

Variable flow analysis:
  Each stage passes the UNION of variables needed by all subsequent stages
  through the intermediate atom. This ensures:
  1. Stage 1 binds vars used in stages 2+
  2. Stage 2 receives those bindings + binds new vars for stage 3+
  3. No variable is lost between stages

Decomposition strategy (spec §6.2 BoundedSplit analogue):
  - Sources pre-ordered by plan_query (most selective first)
  - Split: first ⌊N/2⌋ sources in stage 1, remainder in stage 2
  - Recursive: if stage 2 still has >STAGE_MAX_SOURCES, split again
"""

const STAGE_MAX_SOURCES = 2   # max sources per stage (2 = O(K^2) per stage)
const SC_TMP_PREFIX = "_sc_tmp"   # prefix for intermediate atom heads

# ── Variable flow analysis ────────────────────────────────────────────────────

"""
    flow_vars(sources, from_idx, to_idx; final_template=nothing) -> Vector{String}

Variables introduced in `sources[1:from_idx]` AND needed anywhere downstream
(sources `from_idx+1:to_idx` OR in `final_template`).

The `final_template` argument is critical: without it, variables only needed
in the output (not in any remaining source) would be silently dropped.
"""
function flow_vars(
    sources::AbstractVector{<:SNode}, from_idx::Int, to_idx::Int;
    final_template::Union{SNode, Nothing}=nothing
)::Vector{String}
    introduced = Set{String}()
    for src in sources[1:from_idx]
        union!(introduced, collect_var_names(src))
    end

    needed_later = Set{String}()
    for src in sources[(from_idx + 1):to_idx]
        union!(needed_later, collect_var_names(src))
    end
    # Also carry vars needed by the final output template
    if final_template !== nothing
        union!(needed_later, collect_var_names(final_template))
    end

    sort!(collect(intersect(introduced, needed_later)))
end

# ── Source ordering — CONNECTIVITY IS A CONSTRAINT, static_score only breaks ties ─────────────

"""
    _connectivity_order(sources) -> Vector{Int}

Order sources so that each one after the first SHARES A VARIABLE with those already chosen,
breaking ties by `static_score`, and on equal scores by original position — so wherever
connectivity does not bind, the previous stable `sortperm(static_score)` behaviour is preserved.

🔴 WHY THIS EXISTS — MEASURED 2026-08-24. This was `sortperm(static_score.(sources))`, full stop.
`static_score` is variable-fraction only: no stats, no btm, and NO CONNECTIVITY. It therefore
prefers ground-heavy patterns, and on a corpus where that disagrees with cardinality it grouped two
sources SHARING NO VARIABLE into the first stage:

    (c \$i \$j)  static 0.667  card  20      <- best by cardinality, worst by static_score
    (a K \$i)   static 0.333  card 300
    (b K \$j)   static 0.333  card 300      <- shares NO variable with (a K \$i)

    was:  (, (a K \$i) (b K \$j)) -> _sc_tmp0     CARTESIAN PRODUCT, 90000 atoms
    now:  (, (a K \$i) (c \$i \$j)) -> _sc_tmp0    connected on \$i,        20 atoms   (4500x)

i.e. the pre-sort of the module whose whole purpose is the Rule-of-64 fix (O(K^n) -> O(K^2)/stage)
could itself produce the O(K^n) blowup. Connectivity is the property that decides whether a stage
IS a join; cardinality proxies are what drift.

⚠️ THIS IS A CONSTRAINT, NOT A REPLACEMENT, and two other designs were rejected (see
`docs/AUDIT_DOC1.md` §N4): consuming QueryPlanner's order would make this transform depend on a
stage that is OPTIONAL (`plan=false` is valid) and need a fallback anyway; "refuse to group
disconnected sources" is not implementable as stated, since the chain has to emit something.

⚠️ WHAT THIS DOES **NOT** FIX (N3/N4 interact — do not read this as closure): among CONNECTED
candidates the choice is still `static_score`, the same variable-fraction heuristic that is wrong in
the same direction as `estimate_cardinality`. This removes the CATASTROPHIC case and leaves the
ordinary one.
"""
function _connectivity_order(sources::AbstractVector{<:SNode})::Vector{Int}
    n = length(sources)
    n <= 1 && return collect(1:n)

    scores = static_score.(sources)
    vars = Set{String}[collect_var_names(src) for src in sources]

    remaining = collect(1:n)
    order = Int[]
    bound = Set{String}()

    while !isempty(remaining)
        # Candidates joined to what is already chosen. Empty on the FIRST pick (nothing is bound
        # yet) and for a genuinely disconnected query — in both cases fall back to the full pool
        # rather than failing, because the chain must still emit something.
        connected = filter(i -> !isdisjoint(vars[i], bound), remaining)
        pool = isempty(connected) ? remaining : connected

        # `argmin` on a Vector returns the FIRST minimum, and `pool` preserves original order,
        # so ties break by position exactly as the previous MergeSort did.
        best = pool[argmin(Float64[scores[i] for i in pool])]

        push!(order, best)
        union!(bound, vars[best])
        deleteat!(remaining, findfirst(==(best), remaining))
    end
    order
end

"""
    _promote_connected(srcs, bound) -> Vector{SNode}

Move the first source sharing a variable with `bound` to the front, preserving relative order
otherwise. No-op when the head already connects, or when nothing does.

Needed because `_connectivity_order` establishes connectivity against the union of ALL previously
chosen sources, while each later stage actually joins `_sc_tmpN`, which carries only the FLOW VARS.
A source connected via a variable that did not flow into the intermediate would otherwise head a
disconnected stage deeper in the chain.
"""
function _promote_connected(srcs::Vector{SNode}, bound::Set{String})::Vector{SNode}
    isempty(srcs) && return srcs
    isdisjoint(collect_var_names(srcs[1]), bound) || return srcs
    k = findfirst(sr -> !isdisjoint(collect_var_names(sr), bound), srcs)
    k === nothing && return srcs
    SNode[srcs[k]; srcs[1:(k - 1)]; srcs[(k + 1):end]]
end

# ── Decomposition ─────────────────────────────────────────────────────────────

"""
    DecomposedProgram

Result of decomposing a multi-source exec atom.
stages           — the decomposed exec atoms as SNode lists (ready to serialize)
n_intermediate   — number of intermediate `_sc_tmp*` atoms introduced
original_sources — source count before decomposition
"""
struct DecomposedProgram
    stages::Vector{SNode}
    n_intermediate::Int
    original_sources::Int
end

"""
    _is_unsafe_decompose_source(node::SNode) -> Bool

True when a source pattern can match atoms that decomposition ITSELF creates — in which case the
enclosing exec must not be decomposed. Two shapes qualify:

1. **head is literally `exec`** — e.g. `(exec (clocked \$ts) \$p1 \$t1)`. Decomposition emits new
   `(exec …)` atoms, so such a source matches the very stages just produced.
2. **head is NOT a ground symbol** — a variable head (`(\$f a b)`) or a compound head
   (`((step \$k \$ts) \$p0 \$t0)`). These match ANYTHING of the right arity, the emitted stages
   included. counter_machine_5's driver uses exactly this shape, and it does NOT trip test 1 —
   its head is the list `(step \$k \$ts)`, not the symbol `exec`.

Conservative by construction: a source that would in fact be safe costs the Rule-of-64 win on that
atom and costs no correctness. Narrow it only with a measurement.
"""
function _is_unsafe_decompose_source(node::SNode)::Bool
    node isa SList || return false
    its = (node::SList).items
    isempty(its) && return false
    h = its[1]
    h isa SAtom || return true                    # variable or compound head — matches anything
    (h::SAtom).name == "exec"
end

"""
    decompose_exec(atom::SNode; counter=Ref(0)) -> DecomposedProgram

Decompose one exec/rule atom with a multi-source conjunction into a chain
of smaller execs, each with at most STAGE_MAX_SOURCES sources.

Input:  `(id (, src1 src2 src3 src4 src5) (, tpl1 tpl2))`
Output: chain of exec atoms that together produce the same result.
"""
function decompose_exec(atom::SNode; counter::Base.RefValue{Int}=Ref(0))::DecomposedProgram
    atom isa SList || return DecomposedProgram([atom], 0, 0)
    items = (atom::SList).items

    # Only decompose direct exec atoms: (exec <id> (, sources) (, template))
    # Rule definitions like ((phase $p) (, ...) (O ...)) are NOT decomposed —
    # MORK's space_metta_calculus! only picks up top-level `exec` atoms and
    # the rule invocation mechanism handles them differently.
    isempty(items) && return DecomposedProgram([atom], 0, 0)
    !(items[1] isa SAtom && (items[1]::SAtom).name == "exec") &&
        return DecomposedProgram([atom], 0, 0)

    # Find the conjunction (, ...) — in exec form it's at index 3:
    # (exec <id> (, sources) (, template))
    conj_idx = findfirst(i -> is_conjunction(items[i]), 1:length(items))
    conj_idx === nothing && return DecomposedProgram([atom], 0, 0)

    conj = items[conj_idx]::SList
    sources = conj.items[2:end]   # skip the leading ","
    n_src = length(sources)
    n_src <= STAGE_MAX_SOURCES && return DecomposedProgram([atom], 0, n_src)

    # 🔴🔴 REFLECTIVE EXECS ARE NOT DECOMPOSABLE — MEASURED 2026-08-25, and this is a CORRECTNESS
    # guard, not a heuristic. If any SOURCE pattern can match atoms decomposition ITSELF creates,
    # the rewrite is unsound, because IT CHANGES ITS OWN MATCH SET. That covers a literal `exec`
    # head AND a non-ground head (variable or compound), which matches anything of the right arity
    # — see `_is_unsafe_decompose_source`. ⚠️ The first version of this guard tested ONLY for a
    # literal `exec` head and MISSED the driver's other reflective source `((step \$k \$ts) \$p0 \$t0)`,
    # whose head is a LIST.
    #
    # counter_machine_5's driver is the case that found it (the .mm2 comments it "(reflective!)"):
    #
    #   (exec (clocked Z) (, (exec (clocked $ts) $p1 $t1) (state $ts (IC $_)) ((step $k $ts) $p0 $t0))
    #                     (, (exec ($k $ts) $p0 $t0) (exec (clocked (S $ts)) $p1 $t1)))
    #
    # decomposes to TWO atoms, BOTH of shape `(exec (clocked Z) <pattern> <template>)` — and stage
    # 1's source `(exec (clocked $ts) $p1 $t1)` then MATCHES STAGE 2. The rewrite changed the match
    # set. Combined with `_sc_tmp*` living until `_cleanup_sc_tmp!` (which runs ONCE, after
    # execution — SCPipeline.jl:498), stale partials from tick 1 keep firing at tick 30.
    #
    # MEASURED on counter_machine_5, ceiling 20,000 steps:
    #     decompose ON  : 20,000 steps (CEILING, never halts), 546 atoms, 96 `_sc_tmp` residual
    #     decompose OFF :    241 steps,                        325 atoms,  0 `_sc_tmp`
    #     plain MORK    :    241 steps,                        325 atoms
    # i.e. with this guard the pipeline agrees with plain MORK; without it, it does not terminate.
    #
    # ⚠️ THE MODULE DOCSTRING'S SOUNDNESS ARGUMENT DOES NOT COVER THIS. It reasons entirely about
    # VARIABLE FLOW ("No variable is lost between stages"), which is the right analysis for a
    # one-shot evaluation and says nothing about (a) the LIFETIME of intermediates across steps or
    # (b) a source pattern that can match the stages themselves. Both are needed.
    #
    # Conservative on purpose: ANY `(exec …)` source disables decomposition for that atom. A
    # reflective exec that would in fact be safe is left un-decomposed, which costs the Rule-of-64
    # win on that atom and costs no correctness. Narrow it only with a measurement.
    if any(_is_unsafe_decompose_source, sources)
        return DecomposedProgram([atom], 0, n_src)
    end

    # Prefix: all items before the conjunction (e.g. ["exec", "0"] or ["(phase $p)"])
    prefix = collect(items[1:(conj_idx - 1)])
    # Template: everything after the conjunction
    suffix = collect(items[(conj_idx + 1):end])
    final_template = length(suffix) == 1 ? suffix[1] : SList([SAtom(","); suffix])

    # Order sources: connectivity constrains, static_score breaks ties (see _connectivity_order).
    ordered_sources = sources[_connectivity_order(sources)]

    stages = SNode[]
    _build_chain!(stages, prefix, ordered_sources, final_template, counter)

    DecomposedProgram(stages, counter[], n_src)
end

function _build_chain!(
    stages::Vector{SNode},
    prefix::Vector{SNode},
    sources::Vector{SNode},
    final_template::SNode,
    counter::Base.RefValue{Int}
)
    n = length(sources)

    if n <= STAGE_MAX_SOURCES
        # Base case: emit final stage
        conj = SList([SAtom(","); sources])
        push!(stages, SList([prefix..., conj, final_template]))
        return nothing
    end

    # Split: first STAGE_MAX_SOURCES sources in this stage
    split_at = STAGE_MAX_SOURCES
    first_srcs = sources[1:split_at]
    rest_srcs = sources[(split_at + 1):end]

    # Flow variables: introduced in first_srcs AND needed in rest_srcs OR final template
    all_vars = flow_vars(sources, split_at, n; final_template=final_template)

    # The next stage joins `tmp_source` (carrying exactly `all_vars`) with rest_srcs[1], so make
    # sure that head actually connects to it — connectivity at EVERY level, not just the first.
    rest_srcs = _promote_connected(rest_srcs, Set(all_vars))

    # Intermediate atom: _sc_tmp0, _sc_tmp1, ...
    tmp_id = counter[]
    counter[] += 1
    tmp_head = SAtom("$(SC_TMP_PREFIX)$(tmp_id)")
    tmp_args = [SVar(v) for v in all_vars]

    # This stage template: (, (_sc_tmpN $vars...))
    tmp_template = SList([SAtom(","), SList([tmp_head; tmp_args])])

    first_conj = SList([SAtom(","); first_srcs])
    push!(stages, SList([prefix..., first_conj, tmp_template]))

    # Feed intermediate atom as first source of the next stage
    tmp_source = SList([tmp_head; tmp_args])
    next_sources = [tmp_source; rest_srcs]

    _build_chain!(stages, prefix, next_sources, final_template, counter)
end

# ── Program-level decomposition ───────────────────────────────────────────────

"""
    decompose_program(program::AbstractString; max_sources=STAGE_MAX_SOURCES) -> String

Decompose all multi-source exec/rule atoms in `program` into chained exec stages.
Atoms with ≤ max_sources sources are left unchanged.

This is the main entry point for the MORK integration — call instead of
`plan_static` to get the full Rule-of-64 fix:

# Before (Rule-of-64 territory):

(exec 0 (, src1 src2 src3 src4 src5) (, tpl))

# After (O(K^2) per stage):

(exec 0 (, src1 src2) (, (_sc_tmp0 \$v1 \$v2)))
(exec 0 (, (_sc_tmp0 \$v1 \$v2) src3 src4) (, (_sc_tmp1 \$v1 \$v2 \$v3 \$v4)))
(exec 0 (, (_sc_tmp1 \$v1 \$v2 \$v3 \$v4) src5) (, tpl))
"""
function decompose_program(
    program::AbstractString; max_sources::Int=STAGE_MAX_SOURCES
)::String
    nodes = parse_program(program)
    counter = Ref(0)
    all_stages = SNode[]
    for node in nodes
        result = decompose_exec(node; counter=counter)
        append!(all_stages, result.stages)
    end
    sprint_program(all_stages)
end

"""
    decompose_report(program::AbstractString) -> String

Human-readable report showing which atoms were decomposed and why.
"""
function decompose_report(program::AbstractString)::String
    io = IOBuffer()
    nodes = parse_program(program)
    counter = Ref(0)

    for node in nodes
        node isa SList || continue
        items = (node::SList).items
        conj_idx = findfirst(i -> is_conjunction(items[i]), 1:length(items))
        conj_idx === nothing && continue
        sources = (items[conj_idx]::SList).items[2:end]
        n = length(sources)
        n <= STAGE_MAX_SOURCES && continue

        result = decompose_exec(node; counter=Ref(counter[]))
        counter[] += result.n_intermediate

        label = sprint_sexpr(items[1])
        println(io, "Decomposed: $label ($n sources → $(length(result.stages)) stages)")
        for (k, stage) in enumerate(result.stages)
            println(io, "  Stage $k: $(sprint_sexpr(stage))")
        end
    end

    isempty(String(take!(copy(io)))) && println(io, "(no multi-source atoms to decompose)")
    String(take!(io))
end

export STAGE_MAX_SOURCES, SC_TMP_PREFIX
export DecomposedProgram, decompose_exec
export decompose_program, decompose_report
export flow_vars
