# ── DIFFERENTIAL AGAINST PLAIN MORK, OVER THE REAL .mm2 CORPUS ───────────────────────────────
#
# 🔴 WHY THIS FILE EXISTS. Every pipeline defect found on 2026-08-25 was invisible to the existing
# suite because the fixtures could not fail:
#   · `batch_space_ops` shipped unsound for ten weeks — its bisim fixture had ALL patterns
#     satisfiable, so merged and unmerged agreed. One removed fact flipped it.
#   · `PipelineDecompose` emitted a NON-TERMINATING program on the default path — the tests here
#     asserted stage SHAPE, which cannot observe a change in which atoms a pattern matches.
# In both cases the missing ingredient was the same: A REAL WORKLOAD. A workload is what supplies
# inputs where NOT everything matches. No unit fixture we wrote by hand did that.
#
# THE CONTRACT UNDER TEST: for any program, `run!` must compute what plain `space_metta_calculus!`
# computes. It may reorder, decompose, or plan however it likes — the DERIVED ATOMS must agree and
# the step count must match.
#
# ⚠️ PROGRAM ATOMS ARE EXCLUDED FROM THE COMPARISON, DELIBERATELY AND NARROWLY. The planner
# reorders conjuncts inside stored rule atoms (`((step INC $a) (, ($b != $c) …))` vs
# `(, (state …) … ($c != $e) …)`), which is semantically identical and textually different —
# measured as exactly 3 atoms each way on counter_machine_5. DERIVED atoms are compared EXACTLY.
# If a future change makes the pipeline drop or invent a derived atom, this fails.
@testset "corpus differential — run! agrees with plain MORK" begin
    # Resolved, not hardcoded: dev-zone is a sibling checkout whose location varies per machine.
    # Override with MORK_CORPUS_DIR when it lives elsewhere.
    CORPUS = get(ENV, "MORK_CORPUS_DIR",
                 joinpath(homedir(), "JuliaAGI", "dev-zone", "MORK", "kernel", "resources"))
    # (file, plain steps from the .mm2's own @expect-steps header, run! steps MEASURED 2026-08-25)
    #
    # 🔴 run! LEGITIMATELY TAKES MORE STEPS THAN PLAIN, and an earlier version of this file asserted
    # equality and failed on 4 of 8. That assertion was WRONG, not the pipeline: decomposition
    # splits one exec into a CHAIN and each stage costs a step. decision_tree's 34 -> 38 exec forms
    # is exactly its 71 -> 75 steps. Every other assertion passed on all eight programs, including
    # exact agreement on derived atoms — so `run!` computes what plain computes, in more steps.
    #
    # The run! column is therefore a PIN, not a bound: it is what the pipeline does today. A guessed
    # tolerance (`<= 2x plain`) would be the same unmeasured-constant mistake as UNBOUND_DEP_PENALTY.
    # If a pin moves, decomposition changed — say why in the commit, then move it.
    PROGRAMS = [
        ("transitive.mm2",                              3,   4),
        ("ancestor.mm2",                                6,   9),
        ("grounding.mm2",                               7,   7),
        ("string_convert.mm2",                          1,   1),
        ("odd_even_sort.mm2",                          11,  11),
        ("ip_sudoku.mm2",                              34,  35),
        ("decision_tree_learning_without_min_sink.mm2", 71,  75),
        ("counter_machine_5.mm2",                     241, 241),
    ]

    if !isdir(CORPUS)
        @warn "corpus differential SKIPPED — dev-zone resources not present at $CORPUS. \
               Set MORK_CORPUS_DIR to point at them. This suite cannot verify \
               pipeline/substrate agreement without this corpus."
        @test_skip false
    else
        # split a .mm2 into (background facts, exec program) by TOP-LEVEL FORM, not by line —
        # rules span many lines and a line split silently corrupts them.
        function split_mm2(path)
            raw = read(path, String)
            nocmt = join([replace(l, r";.*$" => "") for l in split(raw, "\n")], "\n")
            forms = String[]; depth = 0; buf = IOBuffer()
            for c in nocmt
                c == '(' && (depth += 1)
                depth > 0 && print(buf, c)
                if c == ')'
                    depth -= 1
                    if depth == 0
                        f = strip(String(take!(buf))); !isempty(f) && push!(forms, f)
                    end
                end
            end
            isprog(f) = startswith(f, "(exec")
            (join([f for f in forms if !isprog(f)], "\n"),
             join([f for f in forms if  isprog(f)], "\n"))
        end
        # a DERIVED atom is anything that is not a program/rule form
        isderived(l) = !startswith(l, "(exec") && !startswith(l, "((")
        derived(dump) = Set(l for l in map(strip, split(strip(dump), "\n"))
                            if !isempty(l) && isderived(l))

        for (name, expected_steps, expected_run_steps) in PROGRAMS
            path = joinpath(CORPUS, name)
            isfile(path) || continue
            @testset "$name" begin
                facts, prog = split_mm2(path)
                isempty(prog) && continue

                sA = new_space()
                space_add_all_sexpr!(sA, facts); space_add_all_sexpr!(sA, prog)
                stepsA = space_metta_calculus!(sA, 20_000)
                dA = derived(space_dump_all_sexpr(sA))

                sB = new_space()
                space_add_all_sexpr!(sB, facts)
                r = run!(sB, prog, 20_000)
                dB = derived(space_dump_all_sexpr(sB))

                # 1. plain itself must match the .mm2's own recorded step count — if this fails the
                #    SUBSTRATE moved, and the pipeline comparison below means nothing.
                @test stepsA == expected_steps
                # 2. the pipeline must not run away. `decompose` emitted a non-halting program on
                #    2026-08-25 and burned the full 20,000-step ceiling; this is that tripwire.
                @test r.steps_executed < 20_000
                # 3. the pipeline never does LESS work than plain (it cannot: same answers),
                #    and its step count matches the measured pin above.
                @test r.steps_executed >= stepsA
                @test r.steps_executed == expected_run_steps
                # 4. and derive EXACTLY the same atoms.
                @test dA == dB
                # 5. no intermediate residue survives the pipeline.
                @test !any(x -> occursin("_sc_tmp", x), dB)
            end
        end
    end
end
