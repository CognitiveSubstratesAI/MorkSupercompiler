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

    # Prefix: all items before the conjunction (e.g. ["exec", "0"] or ["(phase $p)"])
    prefix = collect(items[1:(conj_idx - 1)])
    # Template: everything after the conjunction
    suffix = collect(items[(conj_idx + 1):end])
    final_template = length(suffix) == 1 ? suffix[1] : SList([SAtom(","); suffix])

    # Pre-order sources by static selectivity
    scores = static_score.(sources)
    perm = sortperm(scores; alg=MergeSort)
    ordered_sources = sources[perm]

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
