"""
SCPipeline — end-to-end supercompiler pipeline.

Closes the loop from spec §10.4 (Production Hardening):
  stats → plan → [approx rewrite] → decompose → (optional KB saturation) → compile → execute

A single `execute!` call replaces the manual sequence of:
  collect_stats → plan_program → decompose_program →
  space_add_all_sexpr! → space_metta_calculus!

and adds bisimulation obligation recording, timing, and replanning support.

Pipeline stages (all optional, controlled via SCOptions):
  1. STATS     — collect MORKStatistics from the space (or use cached)
  2. PLAN      — QueryPlanner join-order optimization (Algorithm 6)
  2b. APPROX   — ApproxPipeline 4-phase rewrite with error bounds (§6, Goertzel Oct 2025)
  3. DECOMPOSE — PipelineDecompose: split N-source conjunctions → chained
                 2-source stages (Rule-of-64 fix, O(K^N)→O(K^2) per stage)
  4. SATURATE  — KBSaturation incremental saturation on background facts
  5. COMPILE   — MM2Compiler lowers M-Core frags to exec s-expressions
  6. EXECUTE   — space_add_all_sexpr! + space_metta_calculus!
"""

using MORK: Space, new_space, space_add_all_sexpr!, space_metta_calculus!, space_val_count,
    space_remove_all_sexpr!, space_dump_all_sexpr   # used in _cleanup_sc_tmp!/export (A-1)

# ── Pipeline options ──────────────────────────────────────────────────────────

"""
    SCOptions

Controls which pipeline stages are active and their parameters.
"""
struct SCOptions
    collect_stats::Bool     # Stage 1: collect MORKStatistics
    plan_join_order::Bool     # Stage 2: QueryPlanner reordering
    use_approx_pipeline::Bool     # Stage 2b: ApproxPipeline error-bounded rewrite
    error_tolerance::Float64  # Stage 2b: max acceptable approximation error
    decompose_multi_source::Bool     # Stage 3: PipelineDecompose (Rule-of-64 fix)
    saturate_kb::Bool     # Stage 4: KBSaturation on background
    use_mm2_compiler::Bool     # Stage 5: lower through MM2Compiler
    supercompile::Bool     # Stage 4c: §6 driver (Stepper + CanonicalKeys + BoundedSplit)
    max_steps::Int      # Stage 6: space_metta_calculus! limit
    stats_sample_frac::Float64  # fraction of space to sample for stats
    split_budget::Int      # BoundedSplit branch budget
    sc_max_drive_steps::Int      # §6 driver: per-region rewrite_once iteration cap
    cleanup_intermediates::Bool     # Post: remove _sc_tmp* atoms from space
    use_driven_program::Bool     # Stage 4c+: replace program_planned with the driver's residual
    use_magic_sets::Bool     # Stage 3.5: magic-sets goal-direction of the saturation rules
    magic_query::String      # Stage 3.5: goal s-expr seeding magic-sets, e.g. "(path 0 \$y)"
    magic_bound::Int         # Stage 3.5: which query argument is bound (0-based)
    sat_max_rounds::Int      # Stage 4: KBSaturation round backstop; 0 = auto-scale to closure depth
end

SCOptions(;
    max_steps=typemax(Int),
    plan=true,
    stats=true,
    use_approx=false,
    error_tol=0.05,
    decompose=true,
    saturate=false,
    mm2_compile=false,
    supercompile=false,
    sample_frac=1.0,
    budget=SPLIT_DEFAULT_BUDGET,
    drive_steps=1000,
    cleanup=true,
    use_driven=false,
    use_magic_sets=false,
    magic_query="",
    magic_bound=0,
    sat_max_rounds=0          # 0 = auto-scale the saturation backstop to the closure depth
) = SCOptions(
    stats,
    plan,
    use_approx,
    error_tol,
    decompose,
    saturate,
    mm2_compile,
    supercompile,
    max_steps,
    sample_frac,
    budget,
    drive_steps,
    cleanup,
    use_driven,
    use_magic_sets,
    magic_query,
    magic_bound,
    sat_max_rounds
)

const SC_DEFAULTS = SCOptions()

# ── Pipeline result ───────────────────────────────────────────────────────────

"""
    SCResult

Output of the supercompiler pipeline.

steps_executed  — number of metta_calculus! steps taken
stats           — MORKStatistics used for planning
plan_report_str — human-readable join-plan report (if plan_join_order=true)
obligs          — bisimulation obligations from MM2Compiler (if active)
timings         — Dict of stage → elapsed seconds
program_planned — the reordered program string actually loaded
"""
struct SCResult
    steps_executed::Int
    stats::MORKStatistics
    plan_report_str::String
    obligs::Vector{BiSimObligation}
    timings::Dict{Symbol, Float64}
    program_planned::String
    n_atoms_original::Int   # atom count before decomposition
    n_atoms_decomposed::Int   # atom count after decomposition (≥ original)
    drive_results::Vector{DriveResult}    # Stage 4c: §6 driver output per region
    approx_result::Union{Nothing, ApproxPipelineResult}  # Stage 2b output (nothing if skipped)
    n_facts_derived::Int   # Stage 4: derived facts from KBSaturation (0 if saturate_kb=false)
    n_kb_facts::Int   # Stage 4: total facts (base + derived) in the saturation KB
    program_driven::String   # Stage 4c: residual program from drive!() (empty if supercompile=false)
end

# ── Main pipeline entry point ─────────────────────────────────────────────────

"""
    execute!(s::Space, program::AbstractString; opts=SC_DEFAULTS) -> SCResult

Run the full supercompiler pipeline on `program`, adding the result to `s`
and executing up to `opts.max_steps` metta_calculus! steps.

`program` should contain the exec/rule atoms NOT yet loaded into `s`.
Background facts should already be in `s` before calling.
"""
# §10.3 MeTTa→MM2 lowering (MeTTa-MM2-Supercompiler_v1_spec.md:502, "Basic pattern matching").
# The supercompiler executes `(exec …)` programs; the whole downstream (plan/decompose/space_add/
# calculus) only FIRES `exec` atoms — a `match` atom added verbatim is inert. So we rewrite a
# top-level `(match SPACE PAT TMPL)` query into `(exec 0 (, PAT) (, TMPL))` at entry, before any
# stage. This is the missing bridge: the algorithm library's `match` joins can now feed the pipeline
# (previously only hand-written `exec` worked). SPACE is dropped (exec runs against the one trie
# sc_execute! targets). A single pattern → 1-source `(, PAT)`; a `(, s₁ … sₙ)` pattern passes through
# as the multi-source join (where reorder + Rule-of-64 decomposition pay off).
function _lower_match_snode(n::SNode)::SNode
    n isa SList || return n
    items = (n::SList).items
    if length(items) == 4 && items[1] isa SAtom && (items[1]::SAtom).name == "match"
        pat = items[3]
        sources = (pat isa SList && !isempty((pat::SList).items) &&
                   (pat::SList).items[1] isa SAtom && ((pat::SList).items[1]::SAtom).name == ",") ?
                  pat : SList(SNode[SAtom(","), pat])
        template = SList(SNode[SAtom(","), items[4]])
        return SList(SNode[SAtom("exec"), SAtom("0"), sources, template])
    end
    return n
end

_is_match_node(n::SNode) = n isa SList && length((n::SList).items) == 4 &&
    (n::SList).items[1] isa SAtom && ((n::SList).items[1]::SAtom).name == "match"

function _lower_match_program(program::AbstractString)::String
    nodes = parse_program(program)
    any(_is_match_node, nodes) || return String(program)          # no match → untouched (fast path)
    return sprint_program(SNode[_lower_match_snode(n) for n in nodes])
end

# ── (=)→MM2 lowering (MeTTa_MM2_merge_design_2026-07-01.md; Phase-0 Rust-kernel-verified) ──────────────
# A MeTTa function-rule `(= LHS RHS)` is a REDUCTION, so — unlike `(match)` (additive) and `(~>)` (accumulate)
# — it must ADD the reduct AND DELETE the matched redex. Phase-0 verified shape (sinks.rs:1247-1252,
# space.rs:1695-1726): `(exec 0 (I LHS) (O (+ RHS) (- LHS)))` — `I` sources the redex from the space, `O`
# dispatches `+`→AddSink (add during the query loop) and `-`→RemoveSink (delete at finalize; remove-wins on a
# path collision, and RHS≠LHS so none for a genuine reduction). Variables stay `SVar` here; MORK's serializer
# applies the native NewVar/VarRef ordinal encoding by L→R occurrence across the whole exec (no de-Bruijn).
#
# OPT-IN / standalone (NOT wired into execute! unconditionally): `(=)` is ambiguous — a rule to lower vs a data
# fact / type-eq — unlike an unambiguous top-level `(match)`. Call `_lower_eq_program` explicitly (or via an
# opt-in flag) so the default 234-conformance path is untouched. SCOPE (Phase 1): body-form-FREE rules only —
# RHS containing `if`/arithmetic/recursive calls needs the Phase-2 body-form lowering (MM2 has no built-in
# `if`/arith); those RHS terms would be added VERBATIM, not evaluated.
_is_eq_node(n::SNode) = n isa SList && length((n::SList).items) == 3 &&
    (n::SList).items[1] isa SAtom && ((n::SList).items[1]::SAtom).name == "="

function _lower_eq_snode(n::SNode)::SNode
    n isa SList || return n
    items = (n::SList).items
    if length(items) == 3 && items[1] isa SAtom && (items[1]::SAtom).name == "="
        lhs = items[2]; rhs = items[3]
        pattern  = SList(SNode[SAtom("I"), lhs])                                    # source the redex
        template = SList(SNode[SAtom("O"), SList(SNode[SAtom("+"), rhs]),           # add reduct
                                           SList(SNode[SAtom("-"), lhs])])          # remove redex
        return SList(SNode[SAtom("exec"), SAtom("0"), pattern, template])
    end
    return n
end

function _lower_eq_program(program::AbstractString)::String
    nodes = parse_program(program)
    any(_is_eq_node, nodes) || return String(program)             # no (=) → untouched (fast path)
    return sprint_program(SNode[_lower_eq_snode(n) for n in nodes])
end

# Body of a saturation rule `(==> BODY HEAD)`: a `(, p₁ … pₙ)` conjunction → all premises;
# a single pattern → one premise.
function _sat_body_ids!(g::MCoreGraph, body::SNode, vm::Dict{String,Int})::Vector{NodeID}
    if body isa SList && !isempty((body::SList).items) &&
       (body::SList).items[1] isa SAtom && ((body::SList).items[1]::SAtom).name == ","
        its = (body::SList).items
        return NodeID[_sexpr_to_mcore!(g, its[i], vm) for i in 2:length(its)]
    end
    return NodeID[_sexpr_to_mcore!(g, body, vm)]
end

function execute!(s::Space, program::AbstractString; opts::SCOptions=SC_DEFAULTS)::SCResult
    program = _lower_match_program(program)        # §10.3: top-level (match …) → (exec …) before any stage
    timings = Dict{Symbol, Float64}()

    # Stage 1 — collect statistics
    stats = if opts.collect_stats
        t = @elapsed st = collect_stats(s; sample_frac=opts.stats_sample_frac)
        timings[:stats] = t
        st
    else
        MORKStatistics()
    end

    # Stage 2 — plan join order
    program_planned, plan_str = if opts.plan_join_order
        t = @elapsed begin
            planned = plan_program(program, stats)
            pstr = plan_report(program, stats)
        end
        timings[:plan] = t
        (planned, pstr)
    else
        (String(program), "")
    end

    # Stage 2b — approximate pipeline rewrite (§6, Goertzel Oct 2025)
    approx_res = nothing
    if opts.use_approx_pipeline
        t = @elapsed begin
            approx_res = run_approx_pipeline(
                s, program_planned; error_tolerance=opts.error_tolerance
            )
            program_planned = approx_res.program_approx
        end
        timings[:approx] = t
    end

    # Stage 3 — pipeline decomposition (Rule-of-64 fix)
    n_atoms_original = length(parse_program(program_planned))
    n_atoms_decomposed = n_atoms_original
    if opts.decompose_multi_source
        t = @elapsed begin
            program_planned = decompose_program(program_planned)
            n_atoms_decomposed = length(parse_program(program_planned))
        end
        timings[:decompose] = t
    end

    # Stage 4 — KB saturation on the LIVE Space (Algorithm 11 §7.1, IncrementalSaturation).
    # Enumerate the Space: a `(==> BODY HEAD)` atom becomes a forward RULE, everything else a base
    # FACT; run semi-naive saturation to a fixed point; then SERIALIZE the DERIVED facts back to the
    # Space (sprint_mcore → s-expr → space_add_all_sexpr!).
    #
    # This closes the long-standing write-back gap (saturation was observability-only). MORK's
    # KBSaturation already IS the seminaive forward-saturation engine (the PFC/Datalog fixpoint), so
    # nothing is ported — we only wire it onto the live path. The uniqueness gate in saturate! gives
    # termination, so a recursive rule like `(==> (, (path $x $y) (edge $y $z)) (path $x $z))` that
    # would loop the tree-walker converges here and materializes its transitive closure.
    # (Two genuine Prolog deltas — TMS retraction cascade + negation-under-maintenance — are NOT
    # added: saturation here is monotone add-only, sufficient until a non-monotonic workload needs them.)
    n_facts_derived = 0
    n_kb_facts = 0
    if opts.saturate_kb
        t = @elapsed begin
            g = MCoreGraph()
            kb = KBState(g)
            rule_ctr = 0
            _is_rule(sn) = sn isa SList && length((sn::SList).items) == 3 &&
                (sn::SList).items[1] isa SAtom && ((sn::SList).items[1]::SAtom).name == "==>"
            # RULES come from the `program` text — variables are PRESERVED there. (The space DUMP
            # renders each atom's vars positionally/anonymously, which destroys the cross-pattern
            # var sharing a join needs, so rules can't be recovered from the dump.) One fresh varmap
            # per rule so its body+head share variables; numeric/named distinct via _sexpr_to_mcore!.
            sat_rules = Rule[]
            for sn in parse_program(program)
                _is_rule(sn) || continue
                try
                    items = (sn::SList).items                        # (==> BODY HEAD)
                    vm = Dict{String,Int}()
                    body_ids = _sat_body_ids!(g, items[2], vm)
                    head_id  = _sexpr_to_mcore!(g, items[3], vm)
                    rule_ctr += 1
                    rid = add_sym!(g, Sym(Symbol("__sat_rule_$rule_ctr")))
                    push!(sat_rules, Rule(head_id, body_ids, rid))
                catch
                end
            end
            # Stage 3.5 — magic-sets goal-direction (opt-in): rewrite the rules toward `magic_query` so
            # the (otherwise full-relation) bottom-up saturation tables only goal-relevant facts — the
            # bottom-up equivalent of top-down SLG tabling. Lane-agnostic: any caller (Direct via
            # supercompile=true, or the MeTTa-IL saturate lane) reaches it through SCOptions.
            if opts.use_magic_sets && !isempty(opts.magic_query)
                try
                    qn = parse_program(opts.magic_query)
                    if !isempty(qn)
                        qid = _sexpr_to_mcore!(g, only(qn))
                        ms = magic_sets_transform(g, sat_rules, qid; bound_position = opts.magic_bound)
                        sat_rules = ms.rewritten_rules
                        for seed in ms.magic_seeds
                            isvalid(seed) && kb_add_fact!(kb, seed)
                        end
                    end
                catch
                end
            end
            for r in sat_rules
                kb_add_rule!(kb, r)
            end
            # Base FACTS come from the live Space (ground atoms — no vars to mangle). Skip any `==>`
            # stored in the space (its vars are mangled by the dump; rules are taken from `program`).
            for line in split(space_dump_all_sexpr(s), "\n"; keepempty=false)
                line = strip(line)
                isempty(line) && continue
                try
                    nodes = parse_program(line)
                    isempty(nodes) && continue
                    sn = only(nodes)
                    _is_rule(sn) && continue
                    fid = _sexpr_to_mcore!(g, sn)
                    isvalid(fid) && kb_add_fact!(kb, fid)
                catch
                    # Skip atoms that don't parse to valid M-Core
                end
            end
            # Stratified saturation when the program uses negation (`(not …)` premises) so each
            # negated premise sees a completed lower stratum; flat saturation otherwise (identical path).
            #
            # Spec (v2 §7.1 Algorithm 11 / v1 §8.2 Algorithm 3): saturation computes the deductive
            # CLOSURE — run to fixpoint (`while delta_old ≠ {}`), NO round cap. The cap here is purely
            # a non-termination backstop for the out-of-spec value-generating case (warned, not silently
            # truncated, in saturate!). Scale it to the problem so any monotone Horn closure — whose
            # round count ≤ derivation-chain depth ≤ #base-facts — always converges FIRST; a fixed low
            # cap (the old 100) truncated legitimate closures (e.g. an N>100 transitive chain) and
            # violated the spec's closure definition. Callers may override via `opts.sat_max_rounds`.
            # TODO(resource-budget): the spec puts resource bounds on splitting/allocation (§6.2
            # BoundedSplit, approx §5.5), NOT on closure rounds. When a workload risks OOM, replace this
            # round backstop with a Sys.free_memory()-derived memory/time budget that aborts-with-warning.
            sat_backstop = opts.sat_max_rounds > 0 ? opts.sat_max_rounds : max(1000, 4 * length(kb.facts))
            n_facts_derived = (_program_has_negation(kb) ? saturate_stratified! :
                                                           saturate!)(kb; max_rounds=sat_backstop)
            n_kb_facts = length(kb.facts)
            # write-back: serialize each DERIVED fact (not base) and add it to the live Space
            if n_facts_derived > 0
                derived = String[]
                for f in values(kb.facts)
                    is_base_fact(f) && continue
                    push!(derived, sprint_mcore(kb.g, f.id))
                end
                isempty(derived) || space_add_all_sexpr!(s, join(derived, "\n"))
            end
        end
        timings[:saturate] = t
    end

    # Stage 4c — optional §6 supercompiler core (Path C: closes TyLA G functor gap).
    # Drive each top-level node through Stepper + CanonicalKeys + BoundedSplit.
    # Default off — when enabled, observes folding/splitting on the M-Core view
    # of the planned program. The driver's output is recorded for inspection
    # (drive_results); execution still proceeds via Stage 5 metta_calculus!.
    #
    # This is the §6 unit-tested machinery composed end-to-end. Without this
    # block, the supercompiler is effectively a planner + decomposer; with it
    # the TyLA G functor (Appendix D) is load-bearing in code.
    # Boundary #2 (audit 2026-06-18): drive! now produces an OBSERVABLE residual program
    # (`program_driven`). When opts.use_driven_program=true, that residual REPLACES
    # program_planned for Stage 5 — i.e., drive! becomes load-bearing on the live path.
    # Default false because semantic equivalence to the source program is gated on
    # Boundary #3 (bisimulation verifier); enabling without verification is a
    # caller-acknowledged choice (consistent with v2 §12 "Differential testing").
    drive_results = DriveResult[]
    program_driven = ""
    if opts.supercompile
        t = @elapsed begin
            g_drive = MCoreGraph()
            space_reg = copy(DEFAULT_PRIM_REGISTRY)
            register_space_primitives!(space_reg, s)
            opts.use_approx_pipeline && register_approx_primitives!(space_reg)
            ft = FoldTable()
            driven_atoms = String[]
            for node in parse_program(program_planned)
                root_id = try
                    _sexpr_to_mcore!(g_drive, node)
                catch
                    continue
                end
                isvalid(root_id) || continue
                dr = drive!(
                    g_drive,
                    root_id;
                    ft=ft,
                    max_steps=opts.sc_max_drive_steps,
                    stats=stats,
                    split_budget=opts.split_budget,
                    registry=space_reg
                )
                push!(drive_results, dr)
                # Build residual atom: prefer the driven final_id when terminated cleanly
                # (:value or :fold). For :blocked / :max_steps the driver couldn't decide,
                # so keep the original source atom in the residual program.
                atom_str = if (dr.terminated === :value || dr.terminated === :fold) &&
                              isvalid(dr.final_id)
                    sprint_mcore(g_drive, dr.final_id)
                else
                    sprint_program(SNode[node])
                end
                push!(driven_atoms, atom_str)
            end
            program_driven = join(driven_atoms, "\n")
        end
        timings[:supercompile] = t
        # Optionally make drive! load-bearing: route the driven residual into Stage 5.
        if opts.use_driven_program && !isempty(program_driven)
            program_planned = program_driven
        end
    end

    # Stage 4b — optional MM2Compiler lowering with space-aware primitive registry
    obligs = BiSimObligation[]
    if opts.use_mm2_compiler
        t = @elapsed begin
            g = MCoreGraph()
            # Build a space-aware registry so :kb_query/:mm2_exec touch live Space
            # Also wire approx primitives (:approx_kb_query, :sample_fitness) if active
            space_reg = copy(DEFAULT_PRIM_REGISTRY)
            register_space_primitives!(space_reg, s)
            opts.use_approx_pipeline && register_approx_primitives!(space_reg)
            nodes = parse_program(program_planned)
            root_ids = _sexpr_nodes_to_mcore(g, nodes)
            program_planned, obligs = compile_program(g, root_ids)
        end
        timings[:compile] = t
    end

    # Stage 5 — load and execute
    t_exec = @elapsed begin
        space_add_all_sexpr!(s, program_planned)
        steps = space_metta_calculus!(s, opts.max_steps)
    end
    timings[:execute] = t_exec

    # Post — remove _sc_tmp* intermediate atoms left by pipeline decomposition
    if opts.cleanup_intermediates && n_atoms_decomposed > n_atoms_original
        t_cleanup = @elapsed _cleanup_sc_tmp!(s)
        timings[:cleanup] = t_cleanup
    end

    SCResult(
        steps,
        stats,
        plan_str,
        obligs,
        timings,
        program_planned,
        n_atoms_original,
        n_atoms_decomposed,
        drive_results,
        approx_res,
        n_facts_derived,
        n_kb_facts,
        program_driven
    )
end

"""
    execute(facts::AbstractString, program::AbstractString; opts, steps) -> Tuple{Space, SCResult}

Convenience wrapper: build a fresh space from `facts`, run the pipeline,
return (space, result).
"""
function execute(
    facts::AbstractString, program::AbstractString; opts::SCOptions=SC_DEFAULTS,
    steps::Int=typemax(Int)
)::Tuple{Space, SCResult}
    s = new_space()
    space_add_all_sexpr!(s, facts)
    opts2 = SCOptions(
        opts.collect_stats,
        opts.plan_join_order,
        opts.use_approx_pipeline,
        opts.error_tolerance,
        opts.decompose_multi_source,
        opts.saturate_kb,
        opts.use_mm2_compiler,
        opts.supercompile,
        steps,
        opts.stats_sample_frac,
        opts.split_budget,
        opts.sc_max_drive_steps,
        opts.cleanup_intermediates,
        opts.use_driven_program,
        opts.use_magic_sets,
        opts.magic_query,
        opts.magic_bound
    )
    result = execute!(s, program; opts=opts2)
    (s, result)
end

# ── SExpr → M-Core conversion (for MM2Compiler integration) ──────────────────

"""
Convert a vector of SNodes to M-Core NodeIDs (shallow; Prim for compound atoms).
"""
function _sexpr_nodes_to_mcore(g::MCoreGraph, nodes::Vector{SNode})::Vector{NodeID}
    NodeID[_sexpr_to_mcore!(g, n) for n in nodes]
end

# `varmap` maps a NAMED variable (`$x`) to a distinct, SHARED M-Core Var id within one scope (one
# atom/rule), so `(path $x $y)` + `(edge $y $z)` correctly share `$y` for the join. Without this,
# every named var collapsed to `Var(0)` and joins/saturation over the M-Core could never bind. Pass
# ONE varmap across a rule's body+head; a fresh map per independent atom. Numeric vars (`$1`) keep
# their literal id; named vars get ids offset by NAMED_VAR_BASE to avoid clashing with numeric ones.
const NAMED_VAR_BASE = 100_000
function _sexpr_to_mcore!(g::MCoreGraph, n::SNode,
                          varmap::Dict{String,Int}=Dict{String,Int}())::NodeID
    if n isa SAtom
        return add_sym!(g, Sym(Symbol((n::SAtom).name)))
    elseif n isa SVar
        name = (n::SVar).name[2:end]  # strip leading $
        num = tryparse(Int, name)
        ix = num !== nothing ? num : get!(varmap, name, NAMED_VAR_BASE + length(varmap))
        return add_var!(g, Var(ix))
    else
        items = (n::SList).items
        isempty(items) && return add_con!(g, Con(:nil))
        head = items[1]

        if head isa SAtom && (head::SAtom).name == "exec"
            # Compile exec atom as mm2_exec primitive (share the varmap across its sources+template)
            arg_ids = NodeID[_sexpr_to_mcore!(g, items[i], varmap) for i in 2:length(items)]
            return add_prim!(g, Prim(:mm2_exec, arg_ids, EffectSet(UInt8(0x05))))
        end

        head_id = _sexpr_to_mcore!(g, head, varmap)
        field_ids = NodeID[_sexpr_to_mcore!(g, items[i], varmap) for i in 2:length(items)]
        if head isa SAtom
            return add_con!(g, Con(Symbol((head::SAtom).name), field_ids))
        end
        return add_app!(g, App(head_id, field_ids))
    end
end

# ── Intermediate atom cleanup ─────────────────────────────────────────────────

"""
    _cleanup_sc_tmp!(s::Space)

Remove all `_sc_tmp*` intermediate atoms left in the space by pipeline
decomposition.  These are written by Stage N and read by Stage N+1;
after execution they are no longer needed and would otherwise accumulate.
"""
function _cleanup_sc_tmp!(s::Space)
    out = space_dump_all_sexpr(s)
    for line in split(out, '\n')
        sline = strip(line)
        isempty(sline) && continue
        startswith(sline, "(_sc_tmp") || continue
        space_remove_all_sexpr!(s, sline)
    end
end

# ── Timing report ─────────────────────────────────────────────────────────────

"""
Human-readable timing summary for an SCResult.
"""
function timing_report(r::SCResult)::String
    io = IOBuffer()
    total = sum(values(r.timings))
    println(io, "SCPipeline timings:")
    for (stage, t) in sort(collect(r.timings); by=first)
        pct = round(100 * t / max(total, 1e-9); digits=1)
        println(io, "  $(rpad(stage, 12)) $(rpad(round(t*1000; digits=2), 10)) ms  ($pct%)")
    end
    println(io, "  $(rpad(:total, 12)) $(round(total*1000; digits=2)) ms")
    println(io, "  steps_executed: $(r.steps_executed)")
    !isempty(r.obligs) && println(io, "  bisim_obligs: $(length(r.obligs))")
    r.n_atoms_decomposed > r.n_atoms_original &&
        println(io, "  decomposed:   $(r.n_atoms_original) → $(r.n_atoms_decomposed) atoms")
    if r.approx_result !== nothing
        ar = r.approx_result
        println(
            io,
            "  approx:       error_used=$(round(ar.error_budget_used; digits=4))  within_tol=$(ar.within_tolerance)"
        )
    end
    String(take!(io))
end

export SCOptions, SC_DEFAULTS, SCResult
export execute!, execute
export timing_report
export _cleanup_sc_tmp!
