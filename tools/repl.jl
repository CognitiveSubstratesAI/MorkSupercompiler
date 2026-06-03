#!/usr/bin/env julia
# tools/repl.jl — MorkSupercompiler development REPL
#
# Interactive (warm, hot-reload, recommended):
#   julia --project=. -i tools/repl.jl
#
# Scripted (pipe expressions):
#   echo 'include("/tmp/test.jl")' | julia --project=. tools/repl.jl
#
# NEVER cold-start julia for iteration — each restart costs 80s JIT.
# Use this REPL + Revise for all development.

try
    ;
    using Revise;
catch
    ;
end

using MorkSupercompiler
using MORK

# ── Shortcuts ─────────────────────────────────────────────────────────────────

# Test runner
t(path=joinpath(@__DIR__, "..", "test", "runtests.jl")) = include(path)

# Parse + reorder a program string
plan(prog) = plan_static(prog)

# Analyse join order: show report for prog given a space with facts
report(facts_prog, prog) = begin
    s = new_space()
    space_add_all_sexpr!(s, facts_prog)
    print(plan_report(s, prog))
end

# Quick space builder
ns(src) = begin
    s = new_space()
    space_add_all_sexpr!(s, src)
    s
end

# Plan + run (reorder + decompose + execute — full pipeline)
run(facts, prog, steps=typemax(Int)) = begin
    s = new_space()
    space_add_all_sexpr!(s, facts)
    run!(s, prog, steps)
    s
end

if isinteractive()
    println("MorkSupercompiler loaded.")
    println("  t()               — run full test suite (warm, no restart needed)")
    println("  plan(prog)        — show statically-reordered program")
    println("  report(facts, p)  — show join-plan report")
    println("  run(facts, p, n)  — plan + execute n steps, return space")
    println("  ns(src)           — new Space from s-expr string")
    println()
    println("Revise is active — edit src/*.jl and changes hot-reload instantly.")
else
    local failed = false
    for line in eachline(stdin)
        isempty(strip(line)) && continue
        try
            result = eval(Meta.parse(line))
            # Only auto-print scalars and strings — suppress Space/large structs
            # that would flood stdout with raw binary trie data.
            if result !== nothing &&
                (
                result isa Number || result isa AbstractString || result isa Symbol ||
                result isa Bool
            )
                println(result)
            end
        catch e
            println("ERROR: ", e)
            failed = true
        end
    end
    exit(failed ? 1 : 0)
end
