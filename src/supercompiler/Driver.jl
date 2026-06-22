"""
Driver.jl — the §6 supercompiler driver (Path C close on the TyLA `G` functor gap).

Composes Stepper + CanonicalKeys + BoundedSplit into a single function that
drives a configuration graph forward per spec §6.1 Algorithm 7:

  loop:
    1. compute canonical key of the current focus
    2. if the table already has a more-general key → fold (terminate)
    3. otherwise record the key, then rewrite_once
    4. on Value: terminate with the value
    5. on Blocked (Choice): bounded_split, drive each branch
    6. on Residual: advance focus, increment step counter

Previously each of Stepper / CanonicalKeys / BoundedSplit was unit-tested in
isolation but no caller wired them together. SCPipeline.execute! would
build an MCoreGraph then throw it away. That left the spec §6 supercompiler
core as "shelf-ware sitting next to the actually-wired PipelineDecompose"
(audit 2026-05-30). This module closes that gap.

The driver is invoked from SCPipeline.execute! when `opts.supercompile=true`.
Default is false so the existing planner+decompose flow continues to be
the cheap path.
"""

# ── DriveResult — what the driver returns ─────────────────────────────────────

"""
    DriveResult

Records what happened during a `drive!` invocation:

  - `final_id`: NodeID the driver settled on
  - `steps`: how many rewrite_once iterations fired
  - `n_folds`: how many times the whistle blew (fold-back hit)
  - `n_splits`: how many Blocked → bounded_split events
  - `terminated`: :value | :fold | :blocked | :max_steps
"""
struct DriveResult
    final_id::NodeID
    steps::Int
    n_folds::Int
    n_splits::Int
    terminated::Symbol
end

# ── The driver ────────────────────────────────────────────────────────────────

"""
    drive!(g, id; env=Env(), deps=DepSet(),
            ft::FoldTable=FoldTable(),
            max_steps::Int=1000,
            stats::MORKStatistics=MORKStatistics(),
            split_budget::Int=SPLIT_DEFAULT_BUDGET,
            registry=DEFAULT_PRIM_REGISTRY) → DriveResult

Drive the M-Core node `id` to a fixed point via the §6 supercompiler core.

Invariant: every iteration either advances the focus, folds back to a
prior canonical key, or terminates. Bounded by `max_steps` to guard
against runaway recursion (which would itself indicate a Stepper bug —
the whistle should always fire eventually under finite-state termination).

!!! note "Live-path integration (Boundary #2 closure 2026-06-18)"
    Two routes are now available:
    1. **Observation (default)**: `drive!`'s `DriveResult` is recorded in
       `SCResult.drive_results`; a serialized residual program lives in
       `SCResult.program_driven`. `SCPipeline.execute!` still loads
       `program_planned` (the non-driven form) into the Space for Stage 5
       `space_metta_calculus!` — same behaviour as before.
    2. **Load-bearing (opt-in)**: set `SCOptions(use_driven=true)`. The driver's
       residual atoms — built by `sprint_mcore` on each `DriveResult.final_id`
       when terminated cleanly (`:value` or `:fold`) — replace `program_planned`
       before Stage 5. The caller acknowledges that semantic equivalence to the
       original program is gated on Boundary #3 (the bisimulation verifier),
       which checks the recorded obligations against trace-level behaviour.
    End-to-end test asserting both routes:
    `test/integration/test_pipeline.jl :: "drive! produces program_driven (Boundary #2)"`.
"""
function drive!(
    g::MCoreGraph,
    id::NodeID;
    env::Env=Env(),
    deps::DepSet=DepSet(),
    ft::FoldTable=FoldTable(),
    max_steps::Int=1000,
    stats::MORKStatistics=MORKStatistics(),
    split_budget::Int=SPLIT_DEFAULT_BUDGET,
    registry::PrimRegistry=DEFAULT_PRIM_REGISTRY
)::DriveResult
    steps = 0
    n_folds = 0
    n_splits = 0

    while steps < max_steps
        # Whistle: have we already seen a more general canonical key?
        key = canonical_key(g, id, 0)
        existing = lookup_fold(ft, key)
        if existing !== nothing
            return DriveResult(existing, steps, n_folds + 1, n_splits, :fold)
        end
        record!(ft, key, id)

        result = rewrite_once(g, id, env, deps, registry)

        if result isa Value
            return DriveResult(result.id, steps, n_folds, n_splits, :value)

        elseif result isa Blocked
            # Choice → bounded_split. Drive each branch with the same fold
            # table so fold-backs across branches are visible.
            node = get_node(g, id)
            if node isa Choice
                split = bounded_split(g, id, env, stats; budget=split_budget)
                n_splits += 1
                isempty(split.branches) &&
                    return DriveResult(id, steps, n_folds, n_splits, :blocked)
                # Drive each branch — share the fold table so common
                # sub-derivations get cached across branches.
                for branch in split.branches
                    drive!(
                        g,
                        branch.id;
                        env=branch.env,
                        deps=deps,
                        ft=ft,
                        max_steps=max_steps - steps,
                        stats=stats,
                        split_budget=split_budget,
                        registry=registry
                    )
                end
                # Return the split point itself as the residual marker —
                # the actual sub-derivations are recorded in the fold table.
                return DriveResult(id, steps, n_folds, n_splits, :blocked)
            end
            # Other Blocked kinds (effect-dep conflict) just terminate.
            return DriveResult(id, steps, n_folds, n_splits, :blocked)

        else
            # Residual — advance focus to the next form.
            new_id = (result::Residual).id
            new_id == id && return DriveResult(id, steps, n_folds, n_splits, :value)
            id = new_id
            steps += 1
        end
    end

    DriveResult(id, steps, n_folds, n_splits, :max_steps)
end

export DriveResult, drive!
