"""
Rule-of-64 benchmark: baseline vs. plan_static on the 3 canonical cases.

Cases:
  1. odd_even_sort      — 5-source phase rule
  2. counter_machine    — 5-source JZ/INC/DEC Peano rules
  3. transitive_detect  — 3-source edge chain (warmup / control)

Each case is measured with profile(facts, prog; steps, trials=3) to get
median timing across 3 runs. Speedup = baseline_exec / planned_exec.

NOTE: plan_static does join-order reordering only (static variable-count
heuristic). The nested-loop join (semi-join pushdown) is not yet wired in,
so speedup reflects source ordering benefit only.
"""

using MorkSupercompiler
using MORK

# ── helpers ──────────────────────────────────────────────────────────────────

function peano(n)
    n == 0 ? "Z" : "(S $(peano(n-1)))"
end

function seeded_rand(state, n)
    state = state * 6364136223846793005 + 1442695040888963407
    (Int(state >> 33) % n, state)
end

function rand_edges(nnodes, nedges)
    state = UInt64(0x12345678ABCDEF01)
    edges = Set{String}()
    while length(edges) < nedges
        i, state = seeded_rand(state, nnodes)
        j, state = seeded_rand(state, nnodes)
        i == j && continue
        i, j = minmax(i, j)
        push!(edges, "(edge $i $j)")
    end
    join(edges, "\n")
end

# ── Case 1: odd_even_sort (5-source) ─────────────────────────────────────────

const ODD_EVEN_FACTS = raw"""
(lt A B) (lt A C) (lt A D) (lt A E) (lt B C) (lt B D) (lt B E)
(lt C D) (lt C E) (lt D E)
(succ 0 1) (succ 1 2) (succ 2 3) (succ 3 4) (succ 4 5)
(parity 0 even) (parity 1 odd) (parity 2 even) (parity 3 odd) (parity 4 even)
(A 0 B) (A 1 A) (A 2 E) (A 3 C) (A 4 D)
(phase 0 odd) (phase 1 even)
"""

# The 5-source conjunction:
#   (, (parity $i $p) (succ $i $si) (A $i $e) (A $si $se) (lt $se $e))
# Static scores: parity(2/3) = succ(2/3) = A(2/3) = lt(2/3) — all equal
# So plan_static will not reorder (all same score) → minimal speedup expected.
const ODD_EVEN_PROG = raw"""
((phase $p)
 (, (parity $i $p) (succ $i $si) (A $i $e) (A $si $se) (lt $se $e))
 (O (- (A $i $e)) (- (A $si $se)) (+ (A $i $se)) (+ (A $si $e))))
(exec repeat (, (A $k $_) (phase $kp $phase) ((phase $phase) $p0 $t0))
             (, (exec ($k $kp) $p0 $t0)))
"""

# ── Case 2: counter_machine (5-source JZ rule) ───────────────────────────────

# Only the JZ step rule — 5 sources:
#   (state $ts (IC $i))  (program $i (JZ $r $j))
#   (state $ts (REG $r $v))  (if $v (S $i) $j $ni)
#   (state $ts (REG $k $kv))
# Static scores: all 5 have 2-3 vars → mixed → some reordering possible.
const CM_FACTS = """
(program Z (JZ 2 (S (S (S (S (S Z)))))))
(program (S Z) (DEC 2))
(program (S (S Z)) (INC 3))
(program (S (S (S Z))) (INC 1))
(program (S (S (S (S Z)))) (JZ 0 Z))
(program (S (S (S (S (S Z))))) (JZ 1 (S (S (S (S (S (S (S (S (S Z)))))))))))
(program (S (S (S (S (S (S Z)))))) (DEC 1))
(program (S (S (S (S (S (S (S Z))))))) (INC 2))
(program (S (S (S (S (S (S (S (S Z)))))))) (JZ 0 (S (S (S (S (S Z)))))))
(program (S (S (S (S (S (S (S (S (S Z))))))))) H)
(state Z (REG 0 Z))
(state Z (REG 1 Z))
(state Z (REG 2 $(peano(2))))
(state Z (REG 3 Z))
(state Z (REG 4 Z))
(state Z (IC Z))
(if (S \$n) \$x \$y \$x)
(if Z \$x \$y \$y)
(0 != 1) (0 != 2) (0 != 3) (0 != 4)
(1 != 0) (1 != 2) (1 != 3) (1 != 4)
(2 != 1) (2 != 0) (2 != 3) (2 != 4)
(3 != 1) (3 != 2) (3 != 0) (3 != 4)
(4 != 1) (4 != 2) (4 != 0) (4 != 3)
"""

# Use 2-step eval to avoid full counter machine run (too slow for benchmarking)
# Isolate the 5-source JZ step pattern:
const CM_PROG = raw"""
((step JZ $ts)
 (, (state $ts (IC $i)) (program $i (JZ $r $j)) (state $ts (REG $r $v))
    (if $v (S $i) $j $ni) (state $ts (REG $k $kv)))
 (, (state (S $ts) (IC $ni)) (state (S $ts) (REG $k $kv))))
(exec (clocked Z) (, (exec (clocked $ts) $p1 $t1) (state $ts (IC $_))
                     ((step JZ $ts) $p0 $t0))
                  (, (exec (JZ $ts) $p0 $t0) (exec (clocked (S $ts)) $p1 $t1)))
"""

# ── Case 3: transitive_detect (3-source — control) ────────────────────────────

# 3-source chain: (, (edge $x $y) (edge $y $z) (edge $z $w))
# All have 2 vars each. Static scores equal → minimal reordering.
# This is the control: if we see no speedup here, it confirms plan_static
# has limited effect when all sources have equal selectivity.
const TRANS_FACTS = rand_edges(50, 150)
const TRANS_PROG = raw"""
(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w))
         (, (dtrans $x $y $z $w)))
"""

# ── Benchmark runner ──────────────────────────────────────────────────────────

function run_bench(label, facts, prog, steps; trials=3)
    println("\n  [$label]  steps=$steps  trials=$trials")

    # Show source order info
    s_tmp = new_space()
    space_add_all_sexpr!(s_tmp, facts)
    exp_txt = explain(s_tmp, prog)
    # Extract just the "Planned order" lines
    for line in split(exp_txt, '\n')
        (
            occursin("Sources (", line) ||
            occursin("Planned order", line) ||
            occursin("Already optimal", line) ||
            occursin("Reordered", line)
        ) && println("    ", strip(line))
    end

    # Measure
    p = profile(facts, prog; steps=steps, trials=trials, sample_frac=1.0)

    bt = get(p.baseline_times, PHASE_EXECUTE, 0.0)
    pt = get(p.planned_times, PHASE_EXECUTE, 0.0)
    plan_overhead = get(p.planned_times, PHASE_PLAN, 0.0)
    speedup = bt > 0 ? bt / max(pt, 1e-9) : 1.0

    println("    baseline exec:  $(round(bt*1000; digits=2)) ms")
    println("    planned  exec:  $(round(pt*1000; digits=2)) ms")
    println("    plan overhead:  $(round(plan_overhead*1000; digits=3)) ms")
    println("    speedup:        $(round(speedup; sigdigits=3))×")
    println("    sources reord:  $(p.n_sources_reordered)")
    (
        label=label,
        speedup=speedup,
        overhead_ms=plan_overhead*1000,
        baseline_ms=bt*1000,
        planned_ms=pt*1000
    )
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    println("╔═══════════════════════════════════════════════════════╗")
    println("║  MorkSupercompiler — Rule-of-64 Benchmark             ║")
    println("║  Measuring: plan_static (join-order reordering only)  ║")
    println("╚═══════════════════════════════════════════════════════╝")

    results = []

    # Case 1: odd_even_sort — 5-source phase rule (steps=1: one exec application)
    push!(results, run_bench("odd_even_sort", ODD_EVEN_FACTS, ODD_EVEN_PROG, 1; trials=2))

    # Case 2: counter_machine JZ step — 5-source rule (steps=1)
    push!(results, run_bench("counter_machine", CM_FACTS, CM_PROG, 1; trials=2))

    # Case 3: transitive_detect — 3-source control (steps=1)
    push!(results, run_bench("trans_detect", TRANS_FACTS, TRANS_PROG, 1; trials=2))

    println("\n╔═══════════════════════════════════════════════════════╗")
    println("║  Summary                                              ║")
    println("║  Algorithm: static variable-count heuristic          ║")
    println("╠═══════════════════════════════════════════════════════╣")
    for r in results
        speedup_str = "$(round(r.speedup; sigdigits=3))×"
        label_padded = rpad(r.label, 20)
        overhead_str = "$(round(r.overhead_ms; digits=3)) ms overhead"
        println("║  $(label_padded)  speedup=$(rpad(speedup_str, 8))  $(overhead_str)")
    end
    println("╠═══════════════════════════════════════════════════════╣")
    println("║  Note: plan_static = source reordering only.         ║")
    println("║  Nested-loop join (10-1000× target) = next milestone ║")
    println("╚═══════════════════════════════════════════════════════╝")

    results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
