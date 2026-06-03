"""
Stepper — purely structural single-step evaluator for M-Core IR.

Implements §6.1 of the MM2 Supercompiler spec (Goertzel, Oct 2025):
  Algorithm 7 — RewriteOnce  (structural dispatch over MCoreNode kinds)
  Algorithm 8 — CallPrimitive (domain-specific primitive dispatch)

The Stepper is "purely structural" — it calls domain logic through primitives
only (CallPrimitive dispatch), keeping the core stepping logic free of
domain knowledge.  This separation is the key design principle of §6.1.

Step result variants (spec §6.1):
  Value(id)     — fully reduced; no further stepping needed
  Blocked(id)   — blocked on unresolved effect dependency
  Residual(id)  — partially evaluated; some sub-terms still need stepping

The Env maps de Bruijn variable indices to NodeIDs (values already reduced).
DepSet tracks which effect resources must be resolved before a step can proceed.
"""

# ── Step result ───────────────────────────────────────────────────────────────

abstract type StepResult end

"""
Fully reduced — this node is a value, no further stepping needed.
"""
struct Value <: StepResult
    id::NodeID
end

"""
Blocked — the step cannot proceed because `blocking_effect` is unresolved.
"""
struct Blocked <: StepResult
    id::NodeID
    blocking_effect::Effect
end

"""
Partial step — the node was partially evaluated; `id` points to the
rewritten node in the graph.  Caller should step again.
"""
struct Residual <: StepResult
    id::NodeID
end

# ── Environment ───────────────────────────────────────────────────────────────

"""
    Env

Variable environment for the stepper.  Maps de Bruijn index → NodeID.
Index 0 = innermost binding (as in standard de Bruijn convention).
"""
struct Env
    bindings::Vector{NodeID}   # index i → NodeID (0-based: bindings[i+1])
end
Env() = Env(NodeID[])

env_lookup(env::Env, ix::Int)::NodeID =
    1 <= ix + 1 <= length(env.bindings) ? env.bindings[ix + 1] : NULL_NODE

env_extend(env::Env, ids::Vector{NodeID})::Env = Env([ids; env.bindings])

env_extend(env::Env, id::NodeID)::Env = Env([id; env.bindings])

# ── DepSet — effect dependency tracking ──────────────────────────────────────

"""
    DepSet

Set of effects that must be resolved before a computation can proceed.
Built from analyzing a region's effect footprint (§5.3.1).
For MORK exec sources: always empty (all Read → no deps → always proceed).
"""
struct DepSet
    blocking::Vector{Effect}
end
DepSet() = DepSet(Effect[])
is_empty(d::DepSet) = isempty(d.blocking)
can_proceed(d::DepSet) = is_empty(d)
add_dep(d::DepSet, e::Effect) = DepSet([d.blocking; e])

# ── Primitive registry ────────────────────────────────────────────────────────

"""
    PrimHandler

A callable that implements a primitive operation.
`args` are the already-evaluated argument NodeIDs.
Returns a `StepResult`.
"""
const PrimHandler = Function   # (g::MCoreGraph, args::Vector{NodeID}, env::Env) -> StepResult

"""
    PrimRegistry

Maps op::Symbol → PrimHandler.  Pre-populated with standard MM2 primitives;
user can extend for domain-specific ops.
"""
mutable struct PrimRegistry
    handlers::Dict{Symbol, PrimHandler}
end
PrimRegistry() = PrimRegistry(Dict{Symbol, PrimHandler}())
Base.copy(r::PrimRegistry) = PrimRegistry(copy(r.handlers))

register_prim!(r::PrimRegistry, op::Symbol, h::PrimHandler) = (r.handlers[op]=h; r)

function lookup_prim(r::PrimRegistry, op::Symbol)::Union{PrimHandler, Nothing}
    get(r.handlers, op, nothing)
end

# ── Global default registry (populated below) ─────────────────────────────────

const DEFAULT_PRIM_REGISTRY = PrimRegistry()

# ── Algorithm 7 — RewriteOnce (§6.1) ──────────────────────────────────────────

"""
    rewrite_once(g, id, env, deps, registry) -> StepResult

Algorithm 7 (RewriteOnce) from §6.1.  Purely structural — no domain logic here.

Steps:

 1. Analyze effects of node `id`
 2. If deps not satisfied → return Blocked
 3. Dispatch on node kind → step or return Value
"""
function rewrite_once(
    g::MCoreGraph, id::NodeID, env::Env, deps::DepSet=DepSet(),
    registry::PrimRegistry=DEFAULT_PRIM_REGISTRY
)::StepResult
    !isvalid(id) && return Value(id)   # NULL_NODE is trivially a value

    # Effect check: can we proceed?
    node = get_node(g, id)
    node_effects = _node_effects(node)
    for eff in node_effects
        for dep in deps.blocking
            commutes(eff, dep) || return Blocked(id, dep)
        end
    end

    _step_node(g, id, node, env, deps, registry)
end

# ── Dispatch on node kind (Algorithm 7 match arms) ────────────────────────────

function _step_node(g, id, node::Sym, env, deps, reg)::StepResult
    Value(id)   # Sym is already a value
end

function _step_node(g, id, node::Lit, env, deps, reg)::StepResult
    Value(id)   # Lit is already a value
end

function _step_node(g, id, node::Abs, env, deps, reg)::StepResult
    Value(id)   # Abs (lambda) is a value — not applied yet
end

function _step_node(g, id, node::Var, env, deps, reg)::StepResult
    bound = env_lookup(env, node.ix)
    isvalid(bound) ? Value(bound) : Value(id)   # unbound var = Value(itself)
end

function _step_node(g, id, node::MCoreRef, env, deps, reg)::StepResult
    # Algorithm 7 §6.1: Ref(def) -> unfold_definition(def)
    # Look up def_id in g.defs. If found, return Value(body_id).
    # If not found, return Residual — definition may be loaded later.
    body_id = def_lookup(g, node.def_id)
    body_id !== nothing ? Value(body_id) : Residual(id)
end

function _step_node(g, id, node::Con, env, deps, reg)::StepResult
    # Step each field; if all values → Con is a value
    all_values = true
    new_fields = NodeID[]
    for fid in node.fields
        r = rewrite_once(g, fid, env, deps, reg)
        if r isa Value
            push!(new_fields, r.id)
        elseif r isa Blocked
            return r   # propagate block
        else
            push!(new_fields, (r::Residual).id)
            all_values = false
        end
    end
    if all_values && new_fields == node.fields
        return Value(id)
    end
    new_id = add_con!(g, Con(node.head, new_fields, node.effects))
    all_values ? Value(new_id) : Residual(new_id)
end

function _step_node(g, id, node::App, env, deps, reg)::StepResult
    # Step function position first
    f_result = rewrite_once(g, node.fun, env, deps, reg)
    f_result isa Blocked && return f_result

    f_id = f_result isa Value ? f_result.id : (f_result::Residual).id
    f_node = get_node(g, f_id)

    if f_node isa Abs && f_result isa Value
        # Beta reduction: extend env with args, step body
        arg_ids = _eval_args(g, node.args, env, deps, reg)
        arg_ids === nothing && return Blocked(id, ReadEffect(DEFAULT_SPACE))   # arg blocked
        new_env = env_extend(env, arg_ids)
        return rewrite_once(g, f_node.body, new_env, deps, reg)
    end

    # Function not yet a value — return Residual
    Residual(add_app!(g, App(f_id, node.args, node.effects)))
end

function _step_node(g, id, node::LetNode, env, deps, reg)::StepResult
    # Evaluate bindings left-to-right, then step body with extended env
    new_ids = NodeID[]
    for (_, val_id) in node.bindings
        r = rewrite_once(g, val_id, env, deps, reg)
        r isa Blocked && return r
        push!(new_ids, r isa Value ? r.id : (r::Residual).id)
    end
    new_env = env_extend(env, new_ids)
    rewrite_once(g, node.body, new_env, deps, reg)
end

function _step_node(g, id, node::MatchNode, env, deps, reg)::StepResult
    # Step scrutinee first
    s_result = rewrite_once(g, node.scrut, env, deps, reg)
    s_result isa Blocked && return s_result
    !(s_result isa Value) && return Residual(id)   # scrutinee not yet a value

    scrut_id = s_result.id
    scrut_node = get_node(g, scrut_id)

    # Try clauses in order
    for clause in node.clauses
        m = _try_match(g, scrut_node, scrut_id, clause.pattern, env)
        m === nothing && continue   # pattern does not match
        # Guard check (NULL_NODE = no guard)
        if isvalid(clause.guard)
            gr = rewrite_once(g, clause.guard, m, deps, reg)
            gr isa Value || continue
            g_node = get_node(g, gr.id)
            (g_node isa Sym && g_node.name == :false) && continue
        end
        return rewrite_once(g, clause.body, m, deps, reg)
    end
    Residual(id)   # no clause matched
end

function _step_node(g, id, node::Choice, env, deps, reg)::StepResult
    Blocked(id, ReadEffect(DEFAULT_SPACE))
    # Choice requires BoundedSplit — not handled in Stepper
end

function _step_node(g, id, node::Prim, env, deps, reg)::StepResult
    _call_primitive(g, node, env, deps, reg)
end

# ── Algorithm 8 — CallPrimitive (§6.1) ────────────────────────────────────────

"""
    _call_primitive(g, node::Prim, env, deps, reg) -> StepResult

Algorithm 8 from §6.1.  Dispatches to registered primitive handlers.
Standard ops wired at module load:
:kb_query    → query_kb_with_stats (reads space, uses QueryPlanner stats)
:fitness_eval → evaluate_fitness   (reads data, observes prog)
:mm2_exec    → execute_mm2_pattern (reads + appends space)
:identity    → returns first arg unchanged
"""
function _call_primitive(
    g::MCoreGraph, node::Prim, env::Env, deps::DepSet, reg::PrimRegistry
)::StepResult

    # Evaluate all args first (eager — standard for primitives)
    eval_args = _eval_args(g, node.args, env, deps, reg)
    eval_args === nothing && return Blocked(node.args[1], ReadEffect(DEFAULT_SPACE))

    handler = lookup_prim(reg, node.op)
    if handler !== nothing
        return handler(g, eval_args, env)
    end

    # Unknown primitive — return as Residual (may be defined later)
    new_id = add_prim!(g, Prim(node.op, eval_args, node.effects))
    Residual(new_id)
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _eval_args(
    g, arg_ids::Vector{NodeID}, env, deps, reg
)::Union{Vector{NodeID}, Nothing}
    out = NodeID[]
    for aid in arg_ids
        r = rewrite_once(g, aid, env, deps, reg)
        r isa Blocked && return nothing
        push!(out, r isa Value ? r.id : (r::Residual).id)
    end
    out
end

function _node_effects(node::MCoreNode)::Vector{Effect}
    mask = node.effects.mask
    mask == 0 && return Effect[]
    out = Effect[]
    mask & 0x01 != 0 && push!(out, ReadEffect(DEFAULT_SPACE))
    mask & 0x02 != 0 && push!(out, WriteEffect(DEFAULT_SPACE))
    mask & 0x04 != 0 && push!(out, AppendEffect(DEFAULT_SPACE))
    mask & 0x08 != 0 && push!(out, CreateEffect(DEFAULT_SPACE))   # was omitted — diverged
    # from CanonicalKeys._effectset_to_effects (6-bit). Latent (no node emits Create
    # today), aligned defensively so commutativity gating sees Create like fold-subsumption.
    mask & 0x10 != 0 && push!(out, DeleteEffect(DEFAULT_SPACE))
    mask & 0x20 != 0 && push!(out, ObserveEffect(DEFAULT_SPACE))
    out
end

function _try_match(
    g, scrut_node::MCoreNode, scrut_id::NodeID, pat_id::NodeID, env::Env
)::Union{Env, Nothing}
    !isvalid(pat_id) && return env   # null pattern = wildcard

    pat = get_node(g, pat_id)

    if pat isa Var
        # Variable pattern: bind scrut_id to this var index
        ix = pat.ix
        new_bindings = copy(env.bindings)
        # Extend bindings to cover index ix
        while length(new_bindings) <= ix
            push!(new_bindings, NULL_NODE)
        end
        new_bindings[ix + 1] = scrut_id
        return Env(new_bindings)
    end

    if pat isa Con && scrut_node isa Con
        pat_con = pat::Con;
        scrut_con = scrut_node::Con
        pat_con.head != scrut_con.head && return nothing
        length(pat_con.fields) != length(scrut_con.fields) && return nothing
        cur_env = env
        for (pf, sf) in zip(pat_con.fields, scrut_con.fields)
            sf_node = get_node(g, sf)
            cur_env = _try_match(g, sf_node, sf, pf, cur_env)
            cur_env === nothing && return nothing
        end
        return cur_env
    end

    if pat isa Sym && scrut_node isa Sym
        return (pat::Sym).name == (scrut_node::Sym).name ? env : nothing
    end

    if pat isa Lit && scrut_node isa Lit
        return (pat::Lit).val == (scrut_node::Lit).val ? env : nothing
    end

    nothing   # no match
end

# ── Standard primitive handlers (Algorithm 8) ─────────────────────────────────

# :identity — returns first arg unchanged
register_prim!(
    DEFAULT_PRIM_REGISTRY,
    :identity,
    (g, args, env) -> isempty(args) ? Value(NULL_NODE) : Value(args[1])
)

# :kb_query — Algorithm 8 §6.1: query_kb_with_stats(pattern)
# Default registry: no Space available → returns Residual.
# Space-aware registry: call register_space_primitives!(reg, space) to wire live Space.
register_prim!(
    DEFAULT_PRIM_REGISTRY,
    :kb_query,
    (g, args, env) -> Residual(add_prim!(g, Prim(:kb_query, args, EffectSet(UInt8(0x01)))))
)

# :mm2_exec — Algorithm 8 §6.1: execute_mm2_pattern(priority, patterns, templates)
register_prim!(
    DEFAULT_PRIM_REGISTRY,
    :mm2_exec,
    (g, args, env) -> Residual(add_prim!(g, Prim(:mm2_exec, args, EffectSet(UInt8(0x05)))))
)

# :fitness_eval — Algorithm 8 §6.1: evaluate_fitness(program, data)
register_prim!(
    DEFAULT_PRIM_REGISTRY,
    :fitness_eval,
    (g, args, env) ->
        Residual(add_prim!(g, Prim(:fitness_eval, args, EffectSet(UInt8(0x21)))))
)

"""
    register_space_primitives!(reg, space) → PrimRegistry

Wire live MORK Space into a PrimRegistry so :kb_query and :mm2_exec
can interact with the Space during M-Core evaluation (Algorithm 8 §6.1).

:kb_query  — query_kb_with_stats: run space_query_multi on pattern arg,
return match count as a Lit node (cardinality estimation).
:mm2_exec  — execute_mm2_pattern: add exec atom to space and run
space_metta_calculus! for one step, return Value.
"""
function register_space_primitives!(reg::PrimRegistry, space::Space)::PrimRegistry
    # :kb_query — reads space, returns cardinality as Lit
    register_prim!(
        reg,
        :kb_query,
        (g, args, env) -> begin
            isempty(args) && return Value(add_lit!(g, Lit(0)))
            pat_node = get_node(g, args[1])
            pat_str = sprint_mcore_to_mm2(g, args[1])
            count = 0
            try
                nodes = parse_program(pat_str)
                if !isempty(nodes)
                    count = dynamic_count(space.btm, only(nodes))
                    count == typemax(Int) && (count = 0)
                end
            catch
            end
            Value(add_lit!(g, Lit(count)))
        end
    )

    # :mm2_exec — appends exec atom to space and steps one round
    register_prim!(
        reg, :mm2_exec, (g, args, env) -> begin
            isempty(args) && return Value(NULL_NODE)
            exec_str = sprint_mcore_to_mm2(g, args[1])
            try
                space_add_all_sexpr!(space, exec_str)
                space_metta_calculus!(space, 1)
            catch
            end
            Value(args[1])
        end
    )

    # Populate g.defs from MORK equality atoms: (= name body) or (= (name $x) body)
    # Called lazily — MCoreGraph is passed per-call so we register a hook that
    # loads definitions on first MCoreRef unfold miss.
    register_prim!(
        reg,
        :load_defs,
        (g, args, env) -> begin
            dump = space_dump_all_sexpr(space)
            for line in split(dump, "\n"; keepempty=false)
                line = strip(line)
                startswith(line, "(= ") || continue
                try
                    nodes = parse_program(line)
                    isempty(nodes) && continue
                    node = only(nodes)
                    # (= name body) or (= (name ...) body)
                    node isa SList && length(node.items) == 3 || continue
                    eq_head = node.items[1]
                    lhs = node.items[2]
                    rhs_str = sprint_sexpr(node.items[3])
                    # Extract definition name
                    def_name = if lhs isa SAtom
                        Symbol(lhs.name)
                    elseif lhs isa SList && !isempty(lhs.items) && lhs.items[1] isa SAtom
                        Symbol((lhs.items[1]::SAtom).name)
                    else
                        nothing
                    end
                    def_name === nothing && continue
                    # Parse rhs into MCoreGraph and register
                    rhs_nodes = parse_program(rhs_str)
                    isempty(rhs_nodes) && continue
                    body_id = compile_kb_query(g, only(rhs_nodes))
                    isvalid(body_id) && def_add!(g, def_name, body_id)
                catch
                end
            end
            Value(NULL_NODE)
        end
    )

    reg
end

# ── Multi-step driver ──────────────────────────────────────────────────────────

"""
    step_to_value(g, id, env; max_steps, registry) -> StepResult

Drive `rewrite_once` until Value or Blocked, up to `max_steps`.
Returns the last result.
"""
function step_to_value(
    g::MCoreGraph, id::NodeID, env::Env=Env(); max_steps::Int=1000,
    registry::PrimRegistry=DEFAULT_PRIM_REGISTRY
)::StepResult
    deps = DepSet()
    result = rewrite_once(g, id, env, deps, registry)
    steps = 1
    while result isa Residual && steps < max_steps
        result = rewrite_once(g, result.id, env, deps, registry)
        steps += 1
    end
    result
end

export StepResult, Value, Blocked, Residual
export Env, env_lookup, env_extend
export DepSet, can_proceed, add_dep
export PrimRegistry, PrimHandler, register_prim!, lookup_prim
export DEFAULT_PRIM_REGISTRY, register_space_primitives!
export rewrite_once, step_to_value
