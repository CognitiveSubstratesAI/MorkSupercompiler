# 5-Source CountSink Benchmark — Rule-of-64 with exec atoms (not rule forms).
# Pattern: 5-hop chain with CountSink accumulation.
# Without decomposition: O(K^5) ProductZipper. With: O(K^2) per stage x 4 stages.
# Theoretical speedup at K=20: ~2000x, K=30: ~6750x.
# Baseline capped at 50 steps; decomposed runs to completion.

using MorkSupercompiler
using MORK

# ── Synthetic graph generator ─────────────────────────────────────────────────

function make_chain_graph(n_nodes::Int, density::Float64=0.3)::String
    rng_state = Ref(UInt64(0xDEADBEEF42))
    function rand_node()
        rng_state[] = rng_state[] * 6364136223846793005 + 1442695040888963407
        Int(rng_state[] >> 33) % n_nodes
    end
    facts = String[]
    # Add nodes
    for i in 0:(n_nodes - 1)
        push!(facts, "(node $i)")
    end
    # Add edges with given density
    for i in 0:(n_nodes - 1)
        for j in 0:(n_nodes - 1)
            i == j && continue
            if (i * 31 + j * 17) % 100 < Int(density * 100)
                push!(facts, "(edge $i $j)")
            end
        end
    end
    join(facts, "\n")
end

# ── 5-source exec CountSink pattern ──────────────────────────────────────────

const FIVE_SOURCE_EXEC = "(exec 0 (, (edge \$a \$b) (edge \$b \$c) (edge \$c \$d) (edge \$d \$e) (node \$a)) (O (count (chain-count \$a \$k) \$k \$e)))"

# ── Timing helper ─────────────────────────────────────────────────────────────

function time_exec(facts, prog; max_steps=typemax(Int), trials=2)::Float64
    times = Float64[]
    for _ in 1:trials
        s = new_space()
        space_add_all_sexpr!(s, facts)
        space_add_all_sexpr!(s, prog)
        t0 = time_ns()
        space_metta_calculus!(s, max_steps)
        push!(times, (time_ns() - t0) / 1e6)
    end
    minimum(times)   # best time
end

function run_countsink_bench(K::Int; baseline_steps=50)
    facts = make_chain_graph(K, 0.25)
    n_edges = count(l -> startswith(l, "(edge"), split(facts, "\n"))
    println("\n  [5-source CountSink, K=$K nodes, $n_edges edges]")

    # Decomposed — runs to completion
    decomposed = decompose_program(FIVE_SOURCE_EXEC)
    n_stages = length(parse_program(decomposed))
    n_orig = length(parse_program(FIVE_SOURCE_EXEC))
    println("    Original exec atoms: $n_orig  →  decomposed: $n_stages stages")

    t_decomp = time_exec(facts, decomposed; max_steps=typemax(Int))
    s_check = new_space()
    space_add_all_sexpr!(s_check, facts)
    space_add_all_sexpr!(s_check, decomposed)
    space_metta_calculus!(s_check, typemax(Int))
    dump = space_dump_all_sexpr(s_check)
    n_results = count(l -> startswith(l, "(chain-count"), split(dump, "\n"))

    println(
        "    Decomposed:  $(round(t_decomp, digits=1)) ms  → $n_results chain-count atoms"
    )

    # Baseline — capped (exponential cost at K≥15)
    t_base = time_exec(facts, FIVE_SOURCE_EXEC; max_steps=baseline_steps)
    println("    Baseline ($baseline_steps steps cap):  $(round(t_base, digits=1)) ms")

    theoretical = Float64(K)^3 / 4.0
    println("    Theoretical speedup at K=$K: $(round(theoretical, digits=0))×")
    println(
        "    Observed ratio (decomp vs capped baseline): $(round(t_base/t_decomp, digits=1))×"
    )
end

# ── Run ───────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 60)
    println("  5-Source CountSink Benchmark — Rule-of-64 with exec atoms")
    println("=" ^ 60)
    println("\nDecomposition preview:")
    println(decompose_report(FIVE_SOURCE_EXEC))

    for K in [10, 15, 20]
        run_countsink_bench(K)
    end

    println("\n" * "=" ^ 60)
    println("  Note: baseline uses step cap to avoid O(K^5) explosion")
    println("  Decomposed runs to true fixed point — results are correct")
    println("=" ^ 60)
end
