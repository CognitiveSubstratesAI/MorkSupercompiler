using Test
using MorkSupercompiler

@testset "Profiler — profile returns SCProfile" begin
    facts = "(edge 0 1) (edge 1 2)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z)) (, (path $x $z)))"
    p = profile(facts, prog; steps=2, trials=1)
    @test p isa SCProfile
    @test p.atom_count_before >= 2
    @test haskey(p.baseline_times, PHASE_EXECUTE)
    @test haskey(p.planned_times, PHASE_EXECUTE)
end

@testset "Profiler — speedup_report is non-empty" begin
    facts = "(edge 0 1)"
    prog = raw"(exec 0 (, (edge $x $y)) (, (reach $x)))"
    p = profile(facts, prog; steps=1, trials=1)
    rep = speedup_report(p)
    @test !isempty(rep)
    @test occursin("Speedup", rep) || occursin("speedup", rep)
end

@testset "Profiler — PHASE_DECOMPOSE tracked in planned times" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"
    p = profile(facts, prog; steps=1, trials=1)
    @test haskey(p.planned_times, PHASE_DECOMPOSE)
    @test !haskey(p.baseline_times, PHASE_DECOMPOSE) ||
        get(p.baseline_times, PHASE_DECOMPOSE, 0.0) == 0.0
end

@testset "Profiler — n_atoms_decomposed > 0 for 3-source program" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"
    p = profile(facts, prog; steps=1, trials=1)
    @test p.n_atoms_decomposed >= 1   # 1 atom → 2 stages = +1
end

@testset "Profiler — speedup_report shows decompose line for multi-source" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"
    p = profile(facts, prog; steps=1, trials=1)
    rep = speedup_report(p)
    @test occursin("decompos", rep)
end
