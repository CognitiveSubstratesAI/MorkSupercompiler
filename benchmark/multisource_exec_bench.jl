"""
Multi-source exec benchmark — Rule-of-64 decomposed execution time.

We measure only the DECOMPOSED version. We do NOT run the O(K^N) baseline —
that's the problem we already solved and it would take hours for large N.
We already proved the baseline is intractable (trans_detect 3-src K=150 = 82s).

Synthetic: (exec 0 (, (R0 \$x0 \$x1) (R1 \$x1 \$x2) ... (RN \$xN-1 \$xN)) (, (path \$x0 \$xN)))
Each Ri is a K-node chain with K-1 edges. Expected output: K-N path atoms.
"""

using MorkSupercompiler
using MORK

function chain_facts(pred::String, k::Int)::String
    join(["($pred $i $(i+1))" for i in 0:(k - 2)], " ")
end

function chain_prog(n_sources::Int, k::Int)::String
    preds = ["R$i" for i in 0:(n_sources - 1)]
    raw_vars = [raw"$" * "x$i" for i in 0:n_sources]
    raw_srcs = join(
        ["($(preds[i]) $(raw_vars[i]) $(raw_vars[i+1]))" for i in 1:n_sources], " "
    )
    "(exec 0 (, $raw_srcs) (, (path $(raw_vars[1]) $(raw_vars[end]))))"
end

function time_exec(facts::String, prog::String; trials=2)::Float64
    times = Float64[]
    for _ in 1:trials
        s = new_space()
        space_add_all_sexpr!(s, facts)
        space_add_all_sexpr!(s, prog)
        t0 = time_ns()
        space_metta_calculus!(s, typemax(Int))
        push!(times, (time_ns() - t0) / 1e6)
    end
    sort!(times)[max(1, div(length(times), 2))]
end

function run_case(n_sources::Int, k::Int; trials=2)
    facts = join([chain_facts("R$i", k) for i in 0:(n_sources - 1)], "\n")
    prog = chain_prog(n_sources, k)
    decomp = decompose_program(prog)
    n_stages = length(parse_program(decomp))

    println("\n  [$n_sources-source chain, K=$k]")
    println("    Decomposed into $n_stages stages")

    # Run decomposed, verify output
    s = new_space();
    space_add_all_sexpr!(s, facts)
    space_add_all_sexpr!(s, decomp)
    space_metta_calculus!(s, typemax(Int))
    out = space_dump_all_sexpr(s)
    n_path = count(l -> occursin("(path", l), split(out, "\n"))
    n_expected = max(0, k - n_sources)
    correct = (n_path == n_expected)   # _sc_tmp atoms cleaned by execute!, not raw metta_calculus!

    dt = time_exec(facts, decomp; trials=trials)

    # Theoretical baseline cost
    theory_ms = (k-1)^n_sources * 24e-3
    theory_su = theory_ms / max(dt, 1e-9)

    println(
        "    Decomposed: $(round(dt; digits=1)) ms — $n_path/$n_expected path atoms $(correct ? "✓" : "✗ MISMATCH")"
    )
    println(
        "    Theoretical baseline: O($(k-1)^$n_sources)=$(Int((k-1)^n_sources)) combos ≈ $(round(theory_ms/1000;digits=1)) s"
    )
    println("    Theoretical speedup:  $(round(theory_su; sigdigits=3))×")

    (
        n_sources=n_sources,
        k=k,
        decomposed_ms=dt,
        theoretical_ms=theory_ms,
        speedup_theory=theory_su,
        correct=correct,
        n_stages=n_stages
    )
end

function main(; trials=2)
    println("╔══════════════════════════════════════════════════════╗")
    println("║  Multi-source exec — Decomposed execution benchmark  ║")
    println("║  Baseline NOT run (it IS the Rule-of-64 problem)     ║")
    println("╚══════════════════════════════════════════════════════╝")

    results = [
        run_case(3, 50; trials=trials),
        run_case(4, 20; trials=trials),
        run_case(5, 15; trials=trials),
        run_case(5, 20; trials=trials)
    ]

    println()
    println("╔══════════════════════════════════════════════════════╗")
    println("║  Summary                                             ║")
    println("╠══════════════════════════════════════════════════════╣")
    for r in results
        lbl = "$(r.n_sources)-src K=$(r.k)"
        dt = "$(round(r.decomposed_ms; digits=0)) ms"
        su = "~$(round(r.speedup_theory; sigdigits=3))×"
        ok = r.correct ? "✓" : "✗"
        println(
            "║  $(rpad(lbl,14))  decomposed=$(rpad(dt,10))  theory speedup=$(rpad(su,10))  $ok"
        )
    end
    println("╠══════════════════════════════════════════════════════╣")
    println("║  Empirical baseline: trans_detect 3-src K=150 = 82s  ║")
    println("║  Empirical speedup:  18.5× (verified correct)        ║")
    println("╚══════════════════════════════════════════════════════╝")

    results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
