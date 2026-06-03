"""
MM2Compiler — compile M-Core IR to MM2 exec atoms with formal semantics.

Implements §9 of the MM2 Supercompiler spec (Goertzel, Oct 2025):
  §9.1  Trace semantics for MeTTa and MM2
  §9.2  Algorithm 14 — BisimulationProof obligations
  §9.3  Priority encoding for sequential / conditional MeTTa statements

The compiler lowers M-Core IR to MORK exec s-expressions that can be
loaded directly into a MORK Space via `space_add_all_sexpr!`.

Priority encoding (§9.3):
  Sequential:  `stmt1; stmt2; stmt3`
               → `(exec (1 0) ...) (exec (2 0) ...) (exec (3 0) ...)`
  Conditional: `if cond then T else E`
               → `(exec (1 0) cond T') (exec (2 0) (not cond) E')`

Bisimulation obligations (Algorithm 14):
  (1) Forward simulation:  every MeTTa trace has a corresponding MM2 trace
  (2) Backward simulation: every MM2 trace projects to a MeTTa trace
  (3) Fairness:            MM2 priority ordering preserves MeTTa fairness

The compiler records proof obligations (not proofs) so they can be
checked by a future verifier or discharged manually.
"""

# ── MM2 exec atom representation ─────────────────────────────────────────────

"""
    MM2Priority

A priority pair `(p, q)` where p=sequence position, q=0 for simple statements.
Priorities are lexicographically ordered: (1,0) < (2,0) < (3,0).
"""
struct MM2Priority
    p::Int   # sequence position
    q::Int   # sub-priority (0 for simple stmts)
end
MM2Priority(p::Int) = MM2Priority(p, 0)

Base.isless(a::MM2Priority, b::MM2Priority) = a.p < b.p || (a.p == b.p && a.q < b.q)

"""
Render a priority as an s-expression string.
"""
sprint_priority(pr::MM2Priority) = "($(pr.p) $(pr.q))"

"""
    MM2ExecAtom

One compiled exec atom: `(exec priority pattern_sexpr template_sexpr)`.
priority    — MM2Priority for ordering
pattern     — source pattern s-expression string (the `,` list)
template    — output template s-expression string
source_node — M-Core NodeID this was compiled from (for traceability)
proof_obligs— bisimulation obligations attached to this atom
"""
struct MM2ExecAtom
    priority::MM2Priority
    pattern::String
    template::String
    source_node::NodeID
    proof_obligs::Vector{Symbol}   # :forward_sim, :backward_sim, :fairness
end

"""
Serialize to a MORK-loadable exec s-expression.
"""
function sprint_exec(e::MM2ExecAtom)::String
    "(exec $(sprint_priority(e.priority)) $(e.pattern) $(e.template))"
end

# ── Compilation context ───────────────────────────────────────────────────────

"""
    CompileCtx

Compilation context: graph + symbol table + priority counter + output buffer.
"""
mutable struct CompileCtx
    g::MCoreGraph
    symbols::Dict{Symbol, String}   # M-Core Symbol → MM2 string name
    next_prio::Int
    next_tmpvar::Int                    # fresh temporary variable counter
    output::Vector{MM2ExecAtom}
    obligations::Vector{Tuple{Symbol, NodeID, NodeID}}  # (kind, metta_id, mm2_id)
end

CompileCtx(g::MCoreGraph) = CompileCtx(
    g, Dict{Symbol, String}(), 1, 0, MM2ExecAtom[], Tuple{Symbol, NodeID, NodeID}[]
)

next_priority!(ctx::CompileCtx)::MM2Priority = MM2Priority(ctx.next_prio += 1, 0)

fresh_var!(ctx::CompileCtx)::String = "\$_t$(ctx.next_tmpvar += 1)"

# ── Algorithm 14 — BisimulationProof obligations (§9.2) ──────────────────────

"""
    BiSimObligation

One bisimulation proof obligation.  Three kinds from Algorithm 14:
:forward_sim  — if MeTTa trace T_M exists, MM2 produces projection T_E
:backward_sim — if MM2 trace T_E exists, MeTTa trace T_M where project(T_E)=T_M
:fairness     — MM2 priority ordering preserves MeTTa fairness

Obligations are recorded for external verification, not discharged here.
"""
struct BiSimObligation
    kind::Symbol    # :forward_sim | :backward_sim | :fairness
    metta_id::NodeID    # M-Core source node
    mm2_id::NodeID    # compiled MM2 node (in same MCoreGraph, as Prim :mm2_exec)
end

"""
    record_bisim!(ctx, kind, metta_id, mm2_id)

Record a bisimulation proof obligation.
"""
function record_bisim!(ctx::CompileCtx, kind::Symbol, metta_id::NodeID, mm2_id::NodeID)
    push!(ctx.obligations, (kind, metta_id, mm2_id))
end

# ── §9.3 Priority encoding ────────────────────────────────────────────────────

"""
    compile_sequential!(ctx, node_ids) -> Vector{MM2ExecAtom}

§9.3 Theorem (Priority-Control Equivalence):
`stmt1; stmt2; stmt3` → `(exec (1 0) ...) (exec (2 0) ...) (exec (3 0) ...)`

Compiles a sequence of M-Core nodes to MM2 exec atoms with ascending priority.
"""
function compile_sequential!(ctx::CompileCtx, node_ids::Vector{NodeID})::Vector{MM2ExecAtom}
    atoms = MM2ExecAtom[]
    for nid in node_ids
        atom = compile_node!(ctx, nid)
        atom !== nothing && push!(atoms, atom)
    end
    atoms
end

"""
    compile_conditional!(ctx, cond_id, then_id, else_id) -> Vector{MM2ExecAtom}

§9.3 Theorem (Conditional Equivalence):
`if cond then T else E`
→ `(exec (p 0) cond T') (exec (p+1 0) (not cond) E')`

Mutual exclusion of patterns ensures exactly one branch executes.
"""
function compile_conditional!(
    ctx::CompileCtx, cond_id::NodeID, then_id::NodeID, else_id::NodeID
)::Vector{MM2ExecAtom}
    cond_str = sprint_mcore_to_mm2(ctx.g, cond_id)
    then_str = sprint_mcore_to_mm2(ctx.g, then_id)
    else_str = sprint_mcore_to_mm2(ctx.g, else_id)

    p1 = next_priority!(ctx)
    p2 = next_priority!(ctx)

    then_atom = MM2ExecAtom(
        p1, "(, $cond_str)", then_str, then_id, [:forward_sim, :backward_sim, :fairness]
    )

    else_atom = MM2ExecAtom(
        p2,
        "(, (not $cond_str))",
        else_str,
        else_id,
        [:forward_sim, :backward_sim, :fairness]
    )

    [then_atom, else_atom]
end

# ── Main compilation entry point ──────────────────────────────────────────────

"""
    compile_node!(ctx, id) -> Union{MM2ExecAtom, Nothing}

Compile one M-Core node to an MM2 exec atom.
Returns `nothing` for nodes that do not produce exec atoms (values, vars, etc.)
"""
function compile_node!(ctx::CompileCtx, id::NodeID)::Union{MM2ExecAtom, Nothing}
    !isvalid(id) && return nothing
    node = get_node(ctx.g, id)

    if node isa Prim && (node::Prim).op == :mm2_exec
        return _compile_mm2_exec!(ctx, node::Prim, id)
    end
    if node isa Prim && (node::Prim).op == :kb_query
        return _compile_kb_query!(ctx, node::Prim, id)
    end
    if node isa MatchNode
        return _compile_match!(ctx, node::MatchNode, id)
    end
    if node isa Choice
        return _compile_choice!(ctx, node::Choice, id)
    end
    nothing
end

function _compile_mm2_exec!(ctx::CompileCtx, node::Prim, id::NodeID)::MM2ExecAtom
    # args: [priority_id, patterns_id, templates_id]
    pr = next_priority!(ctx)
    pats = length(node.args) >= 2 ? sprint_mcore_to_mm2(ctx.g, node.args[2]) : "(, )"
    tmpl = length(node.args) >= 3 ? sprint_mcore_to_mm2(ctx.g, node.args[3]) : "(, )"

    atom = MM2ExecAtom(pr, pats, tmpl, id, [:forward_sim, :backward_sim, :fairness])

    # Record bisimulation obligations
    mm2_node_id = add_prim!(ctx.g, Prim(:mm2_exec, node.args, node.effects))
    record_bisim!(ctx, :forward_sim, id, mm2_node_id)
    record_bisim!(ctx, :backward_sim, id, mm2_node_id)
    record_bisim!(ctx, :fairness, id, mm2_node_id)

    atom
end

function _compile_kb_query!(ctx::CompileCtx, node::Prim, id::NodeID)::MM2ExecAtom
    pr = next_priority!(ctx)
    pat = isempty(node.args) ? "\$_" : sprint_mcore_to_mm2(ctx.g, node.args[1])
    tv = fresh_var!(ctx)

    MM2ExecAtom(pr, "(, $pat)", "(, ($tv $pat))", id, [:forward_sim])
end

function _compile_match!(ctx::CompileCtx, node::MatchNode, id::NodeID)::MM2ExecAtom
    scrut = sprint_mcore_to_mm2(ctx.g, node.scrut)
    pr = next_priority!(ctx)
    # Compile first clause as the primary exec atom (others via BoundedSplit)
    if isempty(node.clauses)
        return MM2ExecAtom(pr, "(, $scrut)", "(, )", id, Symbol[])
    end
    clause = node.clauses[1]
    pat = sprint_mcore_to_mm2(ctx.g, clause.pattern)
    body = sprint_mcore_to_mm2(ctx.g, clause.body)
    MM2ExecAtom(pr, "(, $scrut $pat)", body, id, [:forward_sim, :backward_sim])
end

function _compile_choice!(ctx::CompileCtx, node::Choice, id::NodeID)::MM2ExecAtom
    # Each alt becomes its own exec atom; return the first here.
    pr = next_priority!(ctx)
    expr = isempty(node.alts) ? "\$_" : sprint_mcore_to_mm2(ctx.g, node.alts[1].expr)
    MM2ExecAtom(pr, "(, )", expr, id, [:forward_sim, :backward_sim, :fairness])
end

# ── M-Core → MM2 s-expression serializer ─────────────────────────────────────

"""
    sprint_mcore_to_mm2(g, id) -> String

Convert an M-Core node to its MORK s-expression representation.
Variables become `\$xN` (de Bruijn index N), Syms become their name,
Cons become `(head args...)`, Prims become `(op args...)`.
"""
function sprint_mcore_to_mm2(g::MCoreGraph, id::NodeID)::String
    !isvalid(id) && return "\$_"
    n = get_node(g, id)
    if n isa Sym
        ;
        return string((n::Sym).name)
    elseif n isa Lit
        ;
        return string((n::Lit).val)
    elseif n isa Var
        ;
        return "\$x$((n::Var).ix)"
    elseif n isa Con
        c = n::Con
        isempty(c.fields) && return string(c.head)
        parts = join([sprint_mcore_to_mm2(g, f) for f in c.fields], " ")
        return "($(c.head) $parts)"
    elseif n isa Prim
        p = n::Prim
        isempty(p.args) && return string(p.op)
        parts = join([sprint_mcore_to_mm2(g, a) for a in p.args], " ")
        return "($(p.op) $parts)"
    elseif n isa App
        a = n::App
        fun_str = sprint_mcore_to_mm2(g, a.fun)
        arg_strs = join([sprint_mcore_to_mm2(g, a_) for a_ in a.args], " ")
        return "($fun_str $arg_strs)"
    end
    "\$_"
end

# ── Full program compilation ───────────────────────────────────────────────────

"""
    compile_program(g, root_ids) -> Tuple{String, Vector{BiSimObligation}}

Compile a list of M-Core root NodeIDs to a MORK s-expression program string
plus the list of bisimulation proof obligations.

The output string can be passed directly to `space_add_all_sexpr!`.
"""
function compile_program(
    g::MCoreGraph, root_ids::Vector{NodeID}
)::Tuple{String, Vector{BiSimObligation}}
    ctx = CompileCtx(g)
    atoms = compile_sequential!(ctx, root_ids)

    program = join([sprint_exec(a) for a in atoms], "\n")

    obligs = [BiSimObligation(k, mid, mm2id) for (k, mid, mm2id) in ctx.obligations]

    (program, obligs)
end

export MM2Priority, sprint_priority
export MM2ExecAtom, sprint_exec
export CompileCtx, next_priority!, fresh_var!
export BiSimObligation, record_bisim!
export compile_sequential!, compile_conditional!
export compile_node!, compile_program
export sprint_mcore_to_mm2
