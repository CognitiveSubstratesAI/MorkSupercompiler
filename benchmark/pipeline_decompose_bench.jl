"""
Pipeline Decomposition Benchmark — Rule-of-64 real speedup measurement.

Compares:
  Baseline  — original multi-source exec, ProductZipper O(K^N)
  Decomposed — chained 2-source stages via decompose_program, O(K^2) per stage

Cases:
  1. trans_detect      — 3-source edge chain (K=150 edges)
  2. counter_machine   — 5-source JZ step rule
  3. odd_even_sort     — 5-source phase rule

NOTE: steps=typemax(Int) (run to completion) is used for both baseline and
decomposed. A decomposed N-source program needs N-1 steps minimum to complete
all pipeline stages; using steps=1 only runs the first stage and is incorrect.
"""

using MorkSupercompiler
using MORK

# ── Benchmark helpers ────────────────────────────────────────────────────────

function time_exec(
    facts::AbstractString, prog::AbstractString; trials=3, steps::Int=typemax(Int)
)::Float64
    times = Float64[]
    for _ in 1:trials
        s = new_space()
        space_add_all_sexpr!(s, facts)
        space_add_all_sexpr!(s, prog)
        t0 = time_ns()
        space_metta_calculus!(s, steps)
        push!(times, (time_ns() - t0) / 1e6)   # → ms
    end
    sort!(times)
    times[max(1, div(length(times), 2))]   # median
end

function run_case(label, facts, baseline_prog; trials=3)
    decomposed_prog = decompose_program(baseline_prog)
    n_stages = length(parse_program(decomposed_prog))
    n_orig = length(parse_program(baseline_prog))

    # Show what the decomposition produced
    println("\n  [$label]")
    println("    Original:  $n_orig atom(s)")
    println(
        "    Decomposed: $n_stages atom(s) ($(n_stages - n_orig) intermediate stages added)"
    )

    report = decompose_report(baseline_prog)
    for line in split(report, '\n')
        isempty(strip(line)) && continue
        println("    ", strip(line))
    end

    # Correctness check: decomposed output must match baseline
    s_b = new_space();
    space_add_all_sexpr!(s_b, facts)
    space_add_all_sexpr!(s_b, baseline_prog);
    space_metta_calculus!(s_b, typemax(Int))
    out_b = space_dump_all_sexpr(s_b)

    s_d = new_space();
    space_add_all_sexpr!(s_d, facts)
    space_add_all_sexpr!(s_d, decomposed_prog);
    space_metta_calculus!(s_d, typemax(Int))
    out_d = space_dump_all_sexpr(s_d)

    # Compare non-fact, non-intermediate atoms
    filter_out(s) =
        sort(
            filter(
                l -> !isempty(strip(l)) && !occursin("edge ", l) && !occursin("_sc_tmp", l),
                split(s, '\n')
            )
        )
    correct = filter_out(out_b) == filter_out(out_d)
    println("    Correctness:     $(correct ? "✓ outputs match" : "✗ MISMATCH")")

    # Time with full execution (steps=typemax to complete all stages)
    bt = time_exec(facts, baseline_prog; trials=trials, steps=typemax(Int))
    dt = time_exec(facts, decomposed_prog; trials=trials, steps=typemax(Int))

    speedup = bt / max(dt, 1e-9)

    println("    Baseline exec:   $(round(bt; digits=1)) ms")
    println("    Decomposed exec: $(round(dt; digits=1)) ms")
    println("    Speedup:         $(round(speedup; sigdigits=3))×")

    (
        label=label,
        baseline_ms=bt,
        decomposed_ms=dt,
        speedup=speedup,
        n_orig=n_orig,
        n_stages=n_stages,
        correct=correct
    )
end

# ── Case 1: trans_detect (3-source, K=150 edges) ────────────────────────────

function rand_edges(nnodes, nedges)
    state = UInt64(0x12345678ABCDEF01)
    edges = Set{String}()
    while length(edges) < nedges
        state = state * 6364136223846793005 + 1442695040888963407
        i = Int(state >> 33) % nnodes
        state = state * 6364136223846793005 + 1442695040888963407
        j = Int(state >> 33) % nnodes
        i == j && continue
        i, j = minmax(i, j)
        push!(edges, "(edge $i $j)")
    end
    join(edges, "\n")
end

const TRANS_FACTS = rand_edges(50, 150)
const TRANS_PROG = raw"""
(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w))
         (, (dtrans $x $y $z $w)))
"""

# ── Case 2: counter_machine JZ step (5-source) ──────────────────────────────

function peano(n)
    n == 0 ? "Z" : "(S $(peano(n-1)))"
end

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

const CM_PROG = raw"""
((step JZ $ts)
 (, (state $ts (IC $i)) (program $i (JZ $r $j)) (state $ts (REG $r $v))
    (if $v (S $i) $j $ni) (state $ts (REG $k $kv)))
 (, (state (S $ts) (IC $ni)) (state (S $ts) (REG $k $kv))))
"""

# ── Case 3: odd_even_sort (5-source) ────────────────────────────────────────

const ODD_EVEN_FACTS = raw"""
(lt A B) (lt A C) (lt A D) (lt A E) (lt B C) (lt B D) (lt B E)
(lt C D) (lt C E) (lt D E)
(succ 0 1) (succ 1 2) (succ 2 3) (succ 3 4) (succ 4 5)
(parity 0 even) (parity 1 odd) (parity 2 even) (parity 3 odd) (parity 4 even)
(A 0 B) (A 1 A) (A 2 E) (A 3 C) (A 4 D)
(phase 0 odd) (phase 1 even)
"""

const ODD_EVEN_PROG = raw"""
((phase $p)
 (, (parity $i $p) (succ $i $si) (A $i $e) (A $si $se) (lt $se $e))
 (O (- (A $i $e)) (- (A $si $se)) (+ (A $i $se)) (+ (A $si $e))))
"""

# ── Main ────────────────────────────────────────────────────────────────────

function main(; trials=3)
    println("╔═══════════════════════════════════════════════════════╗")
    println("║  MorkSupercompiler — Pipeline Decomposition Benchmark ║")
    println("║  Measuring: decompose_program speedup (Rule-of-64)    ║")
    println("╚═══════════════════════════════════════════════════════╝")

    results = []

    push!(
        results,
        run_case("trans_detect (3-src K=150)", TRANS_FACTS, TRANS_PROG; trials=trials)
    )

    push!(results, run_case("counter_machine (5-src)", CM_FACTS, CM_PROG; trials=trials))

    push!(
        results,
        run_case("odd_even_sort (5-src)", ODD_EVEN_FACTS, ODD_EVEN_PROG; trials=trials)
    )

    println()
    println("╔═══════════════════════════════════════════════════════╗")
    println("║  Summary — decompose_program vs baseline              ║")
    println("╠═══════════════════════════════════════════════════════╣")
    for r in results
        lbl = rpad(r.label, 28)
        su = "$(round(r.speedup; sigdigits=3))×"
        bms = "$(round(r.baseline_ms; digits=0)) ms"
        dms = "$(round(r.decomposed_ms; digits=0)) ms"
        stages = "$(r.n_orig)→$(r.n_stages) atoms"
        println("║  $lbl  $(rpad(su, 8))  $(rpad(bms,10)) → $(rpad(dms,10))  $stages")
    end
    println("╠═══════════════════════════════════════════════════════╣")
    println("║  Expected: trans_detect 10-100×, 5-src cases 50-1000× ║")
    println("╚═══════════════════════════════════════════════════════╝")

    results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
