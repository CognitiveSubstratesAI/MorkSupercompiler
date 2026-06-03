"""
Supercompiler benchmark — baseline vs query-planned on canonical MORK patterns.

Measures:
  1. Planning overhead: time to reorder sources (should be microseconds)
  2. Execution speedup: steps/second with vs without reordering on 3 canonical cases

The three Rule-of-64 cases from QUALITY_REPORT D2:
  - odd_even_sort   — 5-source phase rule (5 sources, ~5-10 atoms each)
  - counter_machine — 5-source JZ/INC/DEC Peano rules (state atoms accumulate)
  - transitive      — 3-source detect (classic 3-join, good for validation)

Run: julia --project=. benchmark/supercompiler_bench.jl
"""

using MorkSupercompiler
using MORK
using BenchmarkTools

# ── Helpers ───────────────────────────────────────────────────────────────────

function measure_planning(program::String, label::String)
    t_static = @elapsed for _ in 1:1000
        ;
        plan_static(program);
    end
    println("  [static]  $label: $(round(t_static*1000/1000, sigdigits=3)) ms/plan")
end

function bench_execution(facts::String, program::String, steps::Int, label::String)
    # Baseline: facts + unmodified program
    t_base = @elapsed begin
        s = new_space()
        space_add_all_sexpr!(s, facts)
        space_add_all_sexpr!(s, program)
        space_metta_calculus!(s, steps)
    end

    # Planned: facts + reordered program (static planning)
    prog_planned = plan_static(program)
    t_plan = @elapsed begin
        s2 = new_space()
        space_add_all_sexpr!(s2, facts)
        space_add_all_sexpr!(s2, prog_planned)
        space_metta_calculus!(s2, steps)
    end

    speedup = t_base / max(t_plan, 1e-9)
    println(
        "  [$label]  baseline: $(round(t_base*1000, sigdigits=4)) ms  " *
        "planned: $(round(t_plan*1000, sigdigits=4)) ms  " *
        "speedup: $(round(speedup, sigdigits=3))×"
    )

    # Report
    report = plan_report(
        begin
            s3=new_space();
            space_add_all_sexpr!(s3, facts);
            s3
        end,
        program
    )
    isempty(strip(report)) || print(report)
end

# ── 1. Odd-even sort (5-source phase rule) ────────────────────────────────────

const ODD_EVEN_FACTS = raw"""
(lt A B) (lt A C) (lt A D) (lt A E) (lt B C) (lt B D) (lt B E) (lt C D) (lt C E) (lt D E)
(succ 0 1) (succ 1 2) (succ 2 3) (succ 3 4) (succ 4 5)
(parity 0 even) (parity 1 odd) (parity 2 even) (parity 3 odd) (parity 4 even)
(A 0 B) (A 1 A) (A 2 E) (A 3 C) (A 4 D)
(phase 0 odd) (phase 1 even)
"""

const ODD_EVEN_PROG = raw"""
((phase $p)  (, (parity $i $p) (succ $i $si) (A $i $e) (A $si $se) (lt $se $e))
             (O (- (A $i $e)) (- (A $si $se)) (+ (A $i $se)) (+ (A $si $e))))
(exec repeat (, (A $k $_) (phase $kp $phase) ((phase $phase) $p0 $t0))
             (, (exec ($k $kp) $p0 $t0)))
"""

# ── 2. Transitive closure (3-source detect) ──────────────────────────────────

function transitive_facts(nnodes, nedges)
    rng_state = UInt64(0x12345678ABCDEF01)
    edges = Set{String}()
    while length(edges) < nedges
        rng_state = rng_state * 6364136223846793005 + 1442695040888963407
        i = Int(rng_state >> 33) % nnodes
        j = Int(rng_state >> 33) % nnodes
        i == j && continue
        i, j = minmax(i, j)
        push!(edges, "(edge $i $j)")
    end
    join(edges, "\n")
end

const TRANS_DETECT_PROG = raw"""
(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $y $z $w)))
"""

# ── 3. Source order report for phase rule ────────────────────────────────────

function show_plan_report()
    println("\n== Join plan report: odd-even sort phase rule ==")
    s = new_space()
    space_add_all_sexpr!(s, ODD_EVEN_FACTS)
    println(plan_report(s, ODD_EVEN_PROG))

    println("== Statically planned program: ==")
    planned = plan_static(ODD_EVEN_PROG)
    println(planned)
end

# ── Runner ────────────────────────────────────────────────────────────────────

function main()
    println("\n═══════════════════════════════════════════════════════")
    println("  MorkSupercompiler — Query Planner Benchmark")
    println("═══════════════════════════════════════════════════════")

    println("\n[1] Planning overhead (1000 plans each)")
    measure_planning(ODD_EVEN_PROG, "odd-even sort")
    measure_planning(TRANS_DETECT_PROG, "transitive detect")

    println("\n[2] Odd-even sort — 5-source phase rule (steps=3)")
    bench_execution(ODD_EVEN_FACTS, ODD_EVEN_PROG, 3, "odd_even_sort")

    println("\n[3] Transitive detect — 3-source (steps=1)")
    facts = transitive_facts(50, 150)
    bench_execution(facts, TRANS_DETECT_PROG, 1, "transitive_detect")

    println("\n[4] Join plan reports")
    show_plan_report()

    println("\n═══════════════════════════════════════════════════════")
    println("  Done.")
    println("═══════════════════════════════════════════════════════")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
