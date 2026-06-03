using Test
using MorkSupercompiler

@testset "AdaptivePlanner — build initial plan" begin
    s = new_space()
    space_add_all_sexpr!(s, "(edge 0 1) (edge 1 2)")
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z)) (, (path $x $z)))"
    ap = AdaptivePlan(s, prog)
    @test ap.plan_version == 1
    @test ap.calls_since_plan == 0
    @test !isempty(ap.program_planned)
end

@testset "AdaptivePlanner — should_replan triggers on age" begin
    s = new_space()
    space_add_all_sexpr!(s, "(foo 1)")
    prog = raw"(exec 0 (, (foo $x)) (, (bar $x)))"
    ap = AdaptivePlan(s, prog)
    ap.calls_since_plan = MAX_PLAN_AGE   # force age threshold
    @test should_replan(ap, s)
end

@testset "AdaptivePlanner — should_replan false when fresh" begin
    s = new_space()
    space_add_all_sexpr!(s, "(foo 1)")
    prog = raw"(exec 0 (, (foo $x)) (, (bar $x)))"
    ap = AdaptivePlan(s, prog)
    @test !should_replan(ap, s)   # fresh plan → no replan needed
end

@testset "AdaptivePlanner — replan! increments version" begin
    s = new_space()
    space_add_all_sexpr!(s, "(foo 1) (foo 2) (foo 3)")
    prog = raw"(exec 0 (, (foo $x)) (, (bar $x)))"
    ap = AdaptivePlan(s, prog)
    v0 = ap.plan_version
    replan!(ap, s)
    @test ap.plan_version == v0 + 1
    @test ap.calls_since_plan == 0
end

@testset "AdaptivePlanner — run_adaptive! executes" begin
    s = new_space()
    space_add_all_sexpr!(s, "(foo 1)")
    prog = raw"(exec 0 (, (foo $x)) (, (bar $x)))"
    ap = AdaptivePlan(s, prog)
    res = run_adaptive!(s, ap; steps=1)
    @test res.plan_version >= 1
    @test space_val_count(s) > 1   # some new atoms added
end
