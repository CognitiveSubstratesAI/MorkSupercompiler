"""
AdaptivePlanner — drift-based adaptive replanning.

Implements §5.2.2 Algorithm 5 (ShouldReplan) from the MM2 Supercompiler spec,
adapted for the incremental MORK execution model.

The adaptive planner caches a join plan and invalidates it when:
  1. Cardinality drift exceeds the adaptive threshold (Algorithm 5)
  2. Plan age exceeds MAX_PLAN_AGE steps
  3. Space growth rate changes significantly

Under sink-free semantics (§4.3), facts are only appended — so plans can
only become stale in one direction (higher actual cardinalities).  This lets
us use a simpler threshold than a bidirectional drift check.

Usage:
  ap = AdaptivePlan(s, program)        # build initial plan
  run_adaptive!(s, ap, new_facts)   # update + maybe replan
"""

using MORK: Space, space_val_count, space_add_all_sexpr!, space_metta_calculus!

# ── Constants (§5.2.2) ────────────────────────────────────────────────────────

const MAX_PLAN_AGE = 50     # replan after this many metta_calculus! calls
const REPLAN_DRIFT = 0.20   # replan if cardinality drifts by > 20%

# ── AdaptivePlan ──────────────────────────────────────────────────────────────

"""
    AdaptivePlan

Cached join plan with version tracking and drift detection.

program_planned  — the last planned version of the program
stats            — MORKStatistics at planning time
atom_count_base  — space atom count when this plan was created
plan_version     — monotone counter incremented on each replan
calls_since_plan — metta_calculus! calls since last plan
cardinality_cache — per-predicate cardinality at planning time (for drift)
"""
mutable struct AdaptivePlan
    program_original::String
    program_planned::String
    stats::MORKStatistics
    atom_count_base::Int
    plan_version::Int
    calls_since_plan::Int
    cardinality_cache::Dict{String, Int}   # predicate → count at plan time
end

"""
    AdaptivePlan(s::Space, program::AbstractString) -> AdaptivePlan

Build the initial plan for `program` given the current state of `s`.
"""
function AdaptivePlan(s::Space, program::AbstractString)::AdaptivePlan
    stats = collect_stats(s)
    prog = plan_program(program, stats)
    AdaptivePlan(
        String(program),
        prog,
        stats,
        space_val_count(s),
        1,
        0,
        copy(predicate_counts(stats))
    )
end

# ── Algorithm 5 — ShouldReplan (§5.2.2) ──────────────────────────────────────

"""
    should_replan(ap::AdaptivePlan, s::Space) -> Bool

Algorithm 5 (ShouldReplan) adapted for MORK.

Triggers a replan if:

 1. Plan age (calls_since_plan) exceeds MAX_PLAN_AGE, OR
 2. Any predicate cardinality drifts by more than REPLAN_DRIFT × original, OR
 3. Total atom count has doubled since last plan

Under sink-free semantics, cardinalities can only increase.
"""
function should_replan(ap::AdaptivePlan, s::Space)::Bool
    # Condition 1: plan age
    ap.calls_since_plan >= MAX_PLAN_AGE && return true

    # Condition 3: atom count doubled
    current_total = space_val_count(s)
    current_total >= 2 * max(1, ap.atom_count_base) && return true

    # Condition 2: per-predicate drift
    new_total = current_total
    for (pred, old_card) in ap.cardinality_cache
        new_card = get(predicate_counts(ap.stats), pred, old_card)
        old_card == 0 && continue
        drift = abs(new_card - old_card) / old_card
        drift > REPLAN_DRIFT && return true
    end

    false
end

"""
    replan!(ap::AdaptivePlan, s::Space) -> Bool

Rebuild the plan for `ap.program_original` against the current `s`.
Returns true if the plan actually changed.
"""
function replan!(ap::AdaptivePlan, s::Space)::Bool
    new_stats = collect_stats(s)
    new_prog = plan_program(ap.program_original, new_stats)

    changed = new_prog != ap.program_planned
    ap.program_planned = new_prog
    ap.stats = new_stats
    ap.atom_count_base = space_val_count(s)
    ap.plan_version += 1
    ap.calls_since_plan = 0
    ap.cardinality_cache = copy(predicate_counts(new_stats))
    changed
end

# ── run_adaptive! ──────────────────────────────────────────────────────────

"""
    run_adaptive!(s, ap, new_facts; steps, force_replan) -> NamedTuple

Execute one adaptive planning cycle:

 1. Load `new_facts` into `s`
 2. Check if replanning is needed (Algorithm 5)
 3. Replan if needed (or forced)
 4. Load the planned program and run metta_calculus!

Returns `(steps=N, replanned=Bool, plan_version=Int)`.
"""
function run_adaptive!(
    s::Space, ap::AdaptivePlan, new_facts::AbstractString=""; steps::Int=typemax(Int),
    force_replan::Bool=false
)::NamedTuple
    !isempty(new_facts) && space_add_all_sexpr!(s, new_facts)

    replanned = false
    if force_replan || should_replan(ap, s)
        replanned = replan!(ap, s)
    end

    space_add_all_sexpr!(s, ap.program_planned)
    n = space_metta_calculus!(s, steps)
    ap.calls_since_plan += 1

    (steps=n, replanned=replanned, plan_version=ap.plan_version)
end

# ── IncrementalStats update (§5.2.1 UpdateIncrementalStats) ──────────────────

"""
    update_stats!(is::IncrementalStats, s::Space) -> IncrementalStats

Algorithm 4 (UpdateIncrementalStats) adapted for MORK: re-scan the space
incrementally and merge into base when delta exceeds 10% of base.

Returns the updated `is` (mutates in place).
"""
function update_stats!(is::IncrementalStats, s::Space)::IncrementalStats
    new_delta = collect_stats(s)   # full scan; future: scan only new atoms
    is.delta = _merge_mork_stats(is.delta, new_delta)

    current_atoms = new_delta.total_atoms
    α = 0.3   # EMA smoothing factor
    is.growth_rate = α * (current_atoms - is.base.total_atoms) + (1.0 - α) * is.growth_rate

    if _should_merge(is)
        is.base = _merge_mork_stats(is.base, is.delta)
        is.delta = MORKStatistics()
        is.last_merge_total = current_atoms
    end

    is
end

export AdaptivePlan, should_replan, replan!
export run_adaptive!
export update_stats!
export MAX_PLAN_AGE, REPLAN_DRIFT
