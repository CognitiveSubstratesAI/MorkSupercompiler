using Test
using MorkSupercompiler

@testset "BisimVerifier — atom-set extraction" begin
    dump = """
    (parent alice bob)
    (parent bob carol)

    (ancestor alice carol)
    """
    # _atom_set_from_dump is internal; verify via verify_bisim's behaviour
    # on an empty obligation list
    verdict = verify_bisim(dump, dump, BiSimObligation[])
    @test verdict.all_discharged
    @test verdict.forward_ok
    @test verdict.backward_ok
    @test verdict.fairness_ok
    # source_atoms should contain the 3 non-blank lines from the dump,
    # post-loading and metta_calculus! (which may add scaffolding)
    @test length(verdict.source_atoms) >= 3
end

@testset "BisimVerifier — identical programs discharge cleanly" begin
    # Equivalent programs: same exec atom → same Space dump after execution
    prog = raw"(exec 0 (, (a $x)) (, (b $x)))"
    facts = "(a 1) (a 2) (a 3)"

    o = BiSimObligation(:forward_sim, NodeID(1), NodeID(2))
    verdict = verify_bisim(prog, prog, [o]; facts=facts, max_steps=20)
    @test verdict.all_discharged
    @test verdict.forward_ok
    @test verdict.backward_ok
    @test length(verdict.results) == 1
    @test verdict.results[1].discharged
end

@testset "BisimVerifier — divergent programs fail forward_sim" begin
    # Two different exec patterns → different Space states after execution.
    # source: a→b; compiled: a→c. Then source has (b 1) etc.; compiled has (c 1) etc.
    source = raw"(exec 0 (, (a $x)) (, (b $x)))"
    compiled = raw"(exec 0 (, (a $x)) (, (c $x)))"
    facts = "(a 1)"

    o_fwd = BiSimObligation(:forward_sim, NodeID(1), NodeID(2))
    o_bwd = BiSimObligation(:backward_sim, NodeID(1), NodeID(2))
    verdict = verify_bisim(source, compiled, [o_fwd, o_bwd]; facts=facts, max_steps=20)

    # source derives (b 1); compiled derives (c 1). Neither is a subset of the other.
    @test !verdict.all_discharged
    @test !verdict.forward_ok
    @test !verdict.backward_ok
    @test !verdict.results[1].discharged
    @test !verdict.results[2].discharged
end

@testset "BisimVerifier — fairness discharges when both halt within budget" begin
    prog = raw"(exec 0 (, (a $x)) (, (b $x)))"
    facts = "(a 1)"

    o_fair = BiSimObligation(:fairness, NodeID(1), NodeID(2))
    verdict = verify_bisim(prog, prog, [o_fair]; facts=facts, max_steps=50)

    @test verdict.fairness_ok
    @test verdict.results[1].discharged
    @test occursin("halted", verdict.results[1].reason)
end

@testset "BisimVerifier — end-to-end with MM2Compiler" begin
    # Compile a real M-Core program, then verify the compiled MM2 against the source.
    g = MCoreGraph()
    pat = add_sym!(g, Sym(:hello))
    tmpl = add_sym!(g, Sym(:world))
    pri = add_lit!(g, Lit(0))
    pats_con = add_con!(g, Con(:_comma, [pat]))
    tmpls_con = add_con!(g, Con(:_comma, [tmpl]))
    exec_id = add_prim!(g, Prim(:mm2_exec, [pri, pats_con, tmpls_con]))

    compiled_str, obligs = compile_program(g, [exec_id])
    @test !isempty(obligs)

    # The compiled program is just `(exec (1 0) (, hello) (, world))` —
    # the source we use for the diff IS the same string, since this is a
    # trivial case (no MeTTa→MM2 semantic gap to exercise).
    verdict = verify_bisim(compiled_str, compiled_str, obligs;
                           facts="hello", max_steps=10)
    # Self-verification of an identical pair MUST pass — sanity gate.
    @test verdict.forward_ok
    @test verdict.backward_ok
end
