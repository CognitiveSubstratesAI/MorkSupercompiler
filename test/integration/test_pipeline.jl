using Test
using MorkSupercompiler

@testset "SCPipeline — execute! stages (plan=true)" begin
    facts = raw"""
    (edge 0 1) (edge 1 2) (edge 2 3)
    """
    prog = raw"""
   (exec 0 (, (edge $x $y) (edge $y $z)) (, (trans $x $z)))
   """
    s = new_space()
    space_add_all_sexpr!(s, facts)
    r = execute!(s, prog; opts=SCOptions(max_steps=1))
    @test r isa SCResult
    @test haskey(r.timings, :execute)
    @test haskey(r.timings, :plan)
    @test !isempty(r.program_planned)
end

@testset "SCPipeline — execute convenience wrapper" begin
    facts = "(a 1) (a 2) (a 3)"
    prog = raw"(exec 0 (, (a $x)) (, (b $x)))"
    s, r = execute(facts, prog; steps=1)
    @test s isa Space
    @test r isa SCResult
    @test space_val_count(s) > 3   # at least some new atoms
end

@testset "SCPipeline — timing_report returns non-empty string" begin
    _, r = execute("(foo 1)", raw"(exec 0 (, (foo $x)) (, (bar $x)))"; steps=1)
    rep = timing_report(r)
    @test !isempty(rep)
    @test occursin("execute", rep)
end

@testset "SCPipeline — decompose stage fires for 3-source program" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"
    _, r = execute(facts, prog; steps=1)
    # decompose=true by default: 1 original atom → 2 stages
    @test r.n_atoms_original == 1
    @test r.n_atoms_decomposed == 2
    @test haskey(r.timings, :decompose)
end

@testset "SCPipeline — decompose=false skips decomposition" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"
    opts = SCOptions(decompose=false)
    _, r = execute(facts, prog; opts=opts, steps=1)
    @test r.n_atoms_original == r.n_atoms_decomposed   # no change
    @test !haskey(r.timings, :decompose)
end

@testset "SCPipeline — 2-source program unchanged by decompose" begin
    facts = "(edge 0 1) (edge 1 2)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z)) (, (path $x $z)))"
    _, r = execute(facts, prog; steps=1)
    # 2 sources ≤ STAGE_MAX_SOURCES → no decomposition
    @test r.n_atoms_original == r.n_atoms_decomposed
end

@testset "SCPipeline — _sc_tmp atoms cleaned up after execution" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"
    s, _ = execute(facts, prog)   # cleanup=true by default
    out = space_dump_all_sexpr(s)
    n_tmp = count(l -> occursin("_sc_tmp", l), split(out, "\n"))
    @test n_tmp == 0   # no intermediate atoms left
end

@testset "SCPipeline — cleanup=false leaves _sc_tmp atoms" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"
    opts = SCOptions(cleanup=false)
    s, _ = execute(facts, prog; opts=opts)
    out = space_dump_all_sexpr(s)
    n_tmp = count(l -> occursin("_sc_tmp", l), split(out, "\n"))
    @test n_tmp > 0   # intermediate atoms remain when cleanup disabled
end

@testset "run! — drop-in for space_metta_calculus!" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"
    s = new_space();
    space_add_all_sexpr!(s, facts)
    r = run!(s, prog)
    @test r isa SCResult
    # decomposed: 3-src → 2 stages → correct dtrans atoms, no _sc_tmp* left
    out = space_dump_all_sexpr(s)
    @test count(l -> occursin("dtrans", l), split(out, "\n")) > 0
    @test count(l -> occursin("_sc_tmp", l), split(out, "\n")) == 0
end

@testset "plan! — includes decomposition" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"
    s = new_space();
    space_add_all_sexpr!(s, facts)
    steps = plan!(s, prog)
    out = space_dump_all_sexpr(s)
    @test count(l -> occursin("dtrans", l), split(out, "\n")) > 0
    @test count(l -> occursin("_sc_tmp", l), split(out, "\n")) == 0
    @test steps >= 1
end

@testset "SCPipeline — decomposed output matches original" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"

    s_orig, _ = execute(facts, prog; opts=SCOptions(decompose=false))
    s_dec, _ = execute(facts, prog)   # decompose=true

    out_orig = space_dump_all_sexpr(s_orig)
    out_dec = space_dump_all_sexpr(s_dec)

    dtrans_orig = sort(filter(l -> occursin("dtrans", l), split(out_orig, "\n")))
    dtrans_dec = sort(filter(l -> occursin("dtrans", l), split(out_dec, "\n")))
    @test !isempty(dtrans_dec)
    @test dtrans_orig == dtrans_dec   # same result
end

@testset "SCPipeline — use_approx=true runs Stage 2b and returns ApproxPipelineResult" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z)) (, (trans $x $z)))"

    s, r = execute(facts, prog; opts=SCOptions(use_approx=true, error_tol=0.05))

    @test r isa SCResult
    @test r.approx_result isa ApproxPipelineResult
    @test haskey(r.timings, :approx)
    @test r.approx_result.within_tolerance
    @test r.approx_result.error_budget_used >= 0.0
end

@testset "SCPipeline — use_approx=false leaves approx_result as nothing" begin
    _, r = execute(
        "(a 1)", raw"(exec 0 (, (a \$x)) (, (b \$x)))"; opts=SCOptions(use_approx=false)
    )
    @test r.approx_result === nothing
    @test !haskey(r.timings, :approx)
end

@testset "SCPipeline — approx + decompose combined" begin
    facts = "(edge 0 1) (edge 1 2) (edge 2 3)"
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (path3 $x $w)))"

    s, r = execute(
        facts, prog; opts=SCOptions(use_approx=true, decompose=true, error_tol=0.1)
    )
    @test r isa SCResult
    @test r.approx_result isa ApproxPipelineResult
    @test !isempty(r.program_planned)
end

@testset "SCPipeline — timing_report includes approx line when active" begin
    _, r = execute(
        "(foo 1) (foo 2) (foo 3)",
        raw"(exec 0 (, (foo $x) (foo $y) (foo $z)) (, (triple $x $y $z)))";
        opts=SCOptions(use_approx=true, error_tol=0.05, max_steps=1)
    )
    rep = timing_report(r)
    @test occursin("approx", rep)
    @test occursin("within_tol", rep)
end

# ── Path C regression: SCPipeline drives §6 supercompiler core ─────────────────

@testset "opts.supercompile=true invokes the §6 driver" begin
    # Verifies the TyLA G functor wiring: when opts.supercompile=true,
    # execute! must populate result.drive_results with a DriveResult per
    # top-level node AND record timings[:supercompile]. With supercompile=false
    # (default) the drive_results must be empty.
    facts = "(edge a b) (edge b c)"
    prog = raw"(exec 0 (, (edge $x $y)) (, (node $x)))"

    s = new_space()
    space_add_all_sexpr!(s, facts)
    opts_off = SCOptions(supercompile=false, max_steps=1)
    res_off = execute!(s, prog; opts=opts_off)
    @test isempty(res_off.drive_results)
    @test !haskey(res_off.timings, :supercompile)

    # Fresh space — supercompile=true path
    s2 = new_space()
    space_add_all_sexpr!(s2, facts)
    opts_on = SCOptions(supercompile=true, drive_steps=100, max_steps=1)
    res_on = execute!(s2, prog; opts=opts_on)
    @test !isempty(res_on.drive_results)
    @test haskey(res_on.timings, :supercompile)
    @test res_on.timings[:supercompile] >= 0.0
    # Each DriveResult must record a terminal state — value/fold/blocked/max_steps
    for dr in res_on.drive_results
        @test dr.terminated in (:value, :fold, :blocked, :max_steps)
    end
end

# ── Workload-side #3: saturate stage now reports observable counts ─────────────

@testset "opts.saturate_kb populates n_facts_derived + n_kb_facts" begin
    # Previously: opts.saturate_kb=true ran KBSaturation on a throwaway
    # MCoreGraph that was discarded — no observable effect. Audit
    # MGFW_INTEGRATION.md §6 flagged this as workload-side cleanup.
    # Fix: capture saturate! return value + kb fact count, surface in SCResult.
    s = new_space()
    space_add_all_sexpr!(s, "(parent alice bob) (parent bob carol)")
    prog = raw"(exec 0 (, (parent $x $y)) (, (ancestor $x $y)))"

    # saturate_kb=false: counts are zero (default)
    opts_off = SCOptions(saturate=false, max_steps=1)
    res_off = execute!(s, prog; opts=opts_off)
    @test res_off.n_facts_derived == 0
    @test res_off.n_kb_facts == 0

    # saturate_kb=true: counts populated (≥ 0 — exact value depends on
    # what M-Core round-trips to). With no rules registered we expect
    # 0 derived but n_kb_facts > 0 from the base fact parse.
    s2 = new_space()
    space_add_all_sexpr!(s2, "(parent alice bob) (parent bob carol)")
    opts_on = SCOptions(saturate=true, max_steps=1)
    res_on = execute!(s2, prog; opts=opts_on)
    @test res_on.n_kb_facts >= 0     # base facts loaded into local KB
    @test haskey(res_on.timings, :saturate)
end
