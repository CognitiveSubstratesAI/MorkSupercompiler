using Test
using MorkSupercompiler

# A-4 (audit's #1 test): the MM2Compiler → space_interpret! seam was never validated —
# every other integration test starts from hand-written exec s-expressions. This drives
# an M-Core program through compile_program and feeds the emitted exec atom into the REAL
# runtime (space_add_all_sexpr! + space_metta_calculus!), asserting the expected derived
# atom. It proves the compiler emits runnable exec text in the shape space_interpret!
# accepts — including the `(exec (p q) …)` (p,q)-priority encoding the §9.3 Priority-
# Control Equivalence theorem rests on (every working example elsewhere uses a plain `0`).
@testset "MM2 round-trip: compile_program output runs through space_metta_calculus!" begin
    # M-Core for the rewrite (foo $x) → (bar $x), as an mm2_exec Prim.
    g = MCoreGraph()
    vx = add_var!(g, Var(0))
    foo_x = add_con!(g, Con(:foo, [vx]))
    bar_x = add_con!(g, Con(:bar, [vx]))
    comma = Symbol(",")
    pats = add_con!(g, Con(comma, [foo_x]))   # (, (foo $x0))
    tmpl = add_con!(g, Con(comma, [bar_x]))   # (, (bar $x0))
    prio = add_sym!(g, Sym(:p0))              # placeholder; compiler assigns next_priority!
    exec = add_prim!(g, Prim(:mm2_exec, [prio, pats, tmpl]))

    program, _obligs = compile_program(g, [exec])
    @test occursin("(exec (", program)       # emits the (p q) priority-pair form
    @test occursin("(, (foo", program) && occursin("(, (bar", program)

    # Feed the COMPILER OUTPUT (not a hand-written exec) into the real runtime.
    s = new_space()
    space_add_all_sexpr!(s, program * "\n(foo a)")
    space_metta_calculus!(s)
    out = space_dump_all_sexpr(s)

    @test occursin("(bar a)", out)   # the compiled rewrite actually fired
    @test occursin("(foo a)", out)   # source retained

    # Control: hand-written canonical form derives the same — isolates that the seam,
    # not the rewrite logic, is what's under test.
    s2 = new_space()
    space_add_all_sexpr!(s2, raw"(exec 0 (, (foo $x)) (, (bar $x)))" * "\n(foo a)")
    space_metta_calculus!(s2)
    @test occursin("(bar a)", space_dump_all_sexpr(s2))
end
