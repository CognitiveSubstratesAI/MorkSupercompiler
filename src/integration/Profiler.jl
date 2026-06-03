"""
Profiler — per-phase timing and baseline-vs-planned speedup measurement.

Implements §10.4 performance profiling from the MM2 Supercompiler spec.

Usage:
  profile = profile(facts, program; steps=5)
  println(speedup_report(profile))

The profiler runs both the baseline (no planning) and the planned version
on identical fresh spaces, measuring wall-clock time for each pipeline stage.
Reports speedup = baseline_exec_time / planned_exec_time.
"""

using MORK: Space, new_space, space_add_all_sexpr!, space_metta_calculus!

# ── Phase enumeration ─────────────────────────────────────────────────────────

"""
Pipeline phases tracked by the profiler.
"""
@enum ProfilePhase begin
    PHASE_STATS
    PHASE_PLAN
    PHASE_DECOMPOSE
    PHASE_LOAD
    PHASE_EXECUTE
    PHASE_TOTAL
end

# ── SCProfile ─────────────────────────────────────────────────────────────────

"""
    SCProfile

Comparison between baseline (no planning) and planned execution.

baseline_times  — phase → seconds for baseline run
planned_times   — phase → seconds for planned run
baseline_steps  — steps executed without planning
planned_steps   — steps executed with planning
atom_count_before   — space size before loading program
atom_count_after    — space size after execution
n_sources_reordered — number of source lists that were reordered
n_atoms_decomposed  — extra exec atoms added by pipeline decomposition
"""
struct SCProfile
    baseline_times::Dict{ProfilePhase, Float64}
    planned_times::Dict{ProfilePhase, Float64}
    baseline_steps::Int
    planned_steps::Int
    atom_count_before::Int
    atom_count_after::Int
    n_sources_reordered::Int
    n_atoms_decomposed::Int
end

# ── profile ────────────────────────────────────────────────────────────────

"""
    profile(facts, program; steps, trials, sample_frac) -> SCProfile

Measure baseline vs. planned execution on `program` loaded into a space
pre-populated with `facts`.  Runs `trials` repetitions and returns the
median timing.

`steps` limits `space_metta_calculus!` so the measurement completes quickly.
"""
function profile(
    facts::AbstractString, program::AbstractString; steps::Int=10, trials::Int=3,
    sample_frac::Float64=1.0
)::SCProfile

    # Measure atom count before loading program
    s_ref = new_space()
    space_add_all_sexpr!(s_ref, facts)
    n_before = space_val_count(s_ref)

    # Count sources reordered (static check, no space needed)
    n_reordered = _count_reordered_sources(program)

    # Count extra atoms added by decomposition
    orig_prog = plan_static(program)
    decomposed_prog = decompose_program(orig_prog)
    n_decomposed = length(parse_program(decomposed_prog)) - length(parse_program(orig_prog))

    # Baseline: load unmodified program, time execution
    baseline_times, baseline_steps_count = _run_trial_with_steps(
        facts, program, steps, trials, false, sample_frac
    )

    # Planned: reorder + decompose sources, time all phases
    planned_times, planned_steps_count = _run_trial_with_steps(
        facts, program, steps, trials, true, sample_frac
    )

    # Atom count after one planned run
    s2 = new_space()
    space_add_all_sexpr!(s2, facts)
    space_add_all_sexpr!(s2, decomposed_prog)
    space_metta_calculus!(s2, steps)
    n_after = space_val_count(s2)

    SCProfile(
        baseline_times,
        planned_times,
        baseline_steps_count,
        planned_steps_count,
        n_before,
        n_after,
        n_reordered,
        n_decomposed
    )
end

function _run_trial_with_steps(
    facts, program, steps, trials, do_plan, sample_frac
)::Tuple{Dict{ProfilePhase, Float64}, Int}
    all_times = [Dict{ProfilePhase, Float64}() for _ in 1:trials]
    last_steps = 0

    for i in 1:trials
        t = Dict{ProfilePhase, Float64}()
        stats_time = 0.0
        if do_plan
            s_tmp = new_space();
            space_add_all_sexpr!(s_tmp, facts)
            stats_time = @elapsed collect_stats(s_tmp; sample_frac=sample_frac)
        end
        t[PHASE_STATS] = stats_time
        plan_time = 0.0;
        prog_to_use = program
        if do_plan
            plan_time = @elapsed (prog_to_use = plan_static(program))
        end
        t[PHASE_PLAN] = plan_time
        decompose_time = 0.0
        if do_plan
            decompose_time = @elapsed (prog_to_use = decompose_program(prog_to_use))
        end
        t[PHASE_DECOMPOSE] = decompose_time
        s = new_space();
        space_add_all_sexpr!(s, facts)
        t[PHASE_LOAD] = @elapsed space_add_all_sexpr!(s, prog_to_use)
        steps_done = 0
        t[PHASE_EXECUTE] = @elapsed (steps_done = space_metta_calculus!(s, steps))
        last_steps = steps_done
        t[PHASE_TOTAL] =
            t[PHASE_STATS] + t[PHASE_PLAN] + t[PHASE_DECOMPOSE] + t[PHASE_LOAD] +
            t[PHASE_EXECUTE]
        all_times[i] = t
    end
    (_median_times(all_times), last_steps)
end

function _median_times(
    all::Vector{Dict{ProfilePhase, Float64}}
)::Dict{ProfilePhase, Float64}
    isempty(all) && return Dict{ProfilePhase, Float64}()
    phases = keys(all[1])
    Dict(ph => _median([t[ph] for t in all]) for ph in phases)
end

_median(v::Vector{Float64}) = sort(v)[div(length(v), 2) + 1]
_extract_steps(t::Dict) = Int(get(t, :_steps, 0.0))   # read step count stashed by _run_trial

function _count_reordered_sources(program::AbstractString)::Int
    nodes = parse_program(program)
    count = 0
    for node in nodes
        node isa SList || continue
        items = (node::SList).items
        length(items) < 3 || !is_conjunction(items[2]) && continue
        sources = (items[2]::SList).items[2:end]
        length(sources) <= 1 && continue
        scores = static_score.(sources)
        # Count as reordered if the scores are not already sorted
        !issorted(scores) && (count += 1)
    end
    count
end

# ── speedup_report ────────────────────────────────────────────────────────────

"""
    speedup_report(p::SCProfile) -> String

Human-readable speedup report.
"""
function speedup_report(p::SCProfile)::String
    io = IOBuffer()
    println(io, "╔══════════════════════════════════════════╗")
    println(io, "║  MorkSupercompiler — Profiler Report     ║")
    println(io, "╚══════════════════════════════════════════╝")
    println(io)

    # Phase table
    phases = [
        PHASE_STATS, PHASE_PLAN, PHASE_DECOMPOSE, PHASE_LOAD, PHASE_EXECUTE, PHASE_TOTAL
    ]
    names = Dict(
        PHASE_STATS => "stats",
        PHASE_PLAN => "plan",
        PHASE_DECOMPOSE => "decompose",
        PHASE_LOAD => "load",
        PHASE_EXECUTE => "execute",
        PHASE_TOTAL => "TOTAL"
    )

    println(io, "  Phase        Baseline      Planned       Speedup")
    println(io, "  ─────────────────────────────────────────────────")
    for ph in phases
        bt = get(p.baseline_times, ph, 0.0)
        pt = get(p.planned_times, ph, 0.0)
        sp = bt > 0 ? round(bt / max(pt, 1e-9); sigdigits=3) : 1.0
        name = get(names, ph, "?")
        println(
            io, "  $(rpad(name, 12)) $(rpad(_fmt_ms(bt), 14))$(rpad(_fmt_ms(pt), 14))$(sp)×"
        )
    end

    println(io)
    bt_exec = get(p.baseline_times, PHASE_EXECUTE, 0.0)
    pt_exec = get(p.planned_times, PHASE_EXECUTE, 0.0)
    exec_speedup = bt_exec > 0 ? round(bt_exec / max(pt_exec, 1e-9); sigdigits=3) : 1.0
    println(io, "  Execution speedup:   $(exec_speedup)×")
    println(io, "  Sources reordered:   $(p.n_sources_reordered)")
    p.n_atoms_decomposed > 0 &&
        println(io, "  Extra stages added:  +$(p.n_atoms_decomposed) (decomposition)")
    println(io, "  Atoms before:        $(p.atom_count_before)")
    println(io, "  Atoms after:         $(p.atom_count_after)")

    String(take!(io))
end

_fmt_ms(t::Float64) = string(round(t * 1000; digits=2), " ms")

export ProfilePhase,
    PHASE_STATS, PHASE_PLAN, PHASE_DECOMPOSE, PHASE_LOAD, PHASE_EXECUTE, PHASE_TOTAL
export SCProfile, profile, speedup_report
