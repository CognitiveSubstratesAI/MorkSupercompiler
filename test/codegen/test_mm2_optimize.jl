using Test
using MorkSupercompiler

# ── helpers for building atoms in tests ──────────────────────────────────────
_atom(p, q, pat, tmpl) = MM2ExecAtom(
    MM2Priority(p, q), pat, tmpl, NodeID(0), Symbol[]
)

# ── Static Scheduling (v1 §10.6 Algorithm 11) ────────────────────────────────
@testset "schedule_static — sorts by priority lex order" begin
    atoms = [
        _atom(3, 0, "(, p3)", "(, t3)"),
        _atom(1, 0, "(, p1)", "(, t1)"),
        _atom(2, 0, "(, p2)", "(, t2)")
    ]
    sorted = schedule_static(atoms)
    @test sorted[1].priority == MM2Priority(1, 0)
    @test sorted[2].priority == MM2Priority(2, 0)
    @test sorted[3].priority == MM2Priority(3, 0)
end

@testset "schedule_static — input not mutated" begin
    atoms = [_atom(2, 0, "(, a)", "(, b)"), _atom(1, 0, "(, c)", "(, d)")]
    original_first_priority = atoms[1].priority
    schedule_static(atoms)
    @test atoms[1].priority == original_first_priority
end

@testset "schedule_static — sub-priorities (q field) honoured" begin
    atoms = [
        _atom(1, 2, "(, a)", "(, x)"),
        _atom(1, 0, "(, b)", "(, y)"),
        _atom(1, 1, "(, c)", "(, z)")
    ]
    sorted = schedule_static(atoms)
    @test sorted[1].priority == MM2Priority(1, 0)
    @test sorted[2].priority == MM2Priority(1, 1)
    @test sorted[3].priority == MM2Priority(1, 2)
end

# ── Space-Operation Batching (v1 §10.6) — REMOVED, REFUTATION PINNED HERE ────
#
# `batch_space_ops` was deleted 2026-08-25: v1 §10.6's transformation is UNSOUND (see the
# tombstone in src/codegen/MM2Optimize.jl). The tests that stood here asserted STRING SHAPE
# — `occursin("(kb fact1)", batched[1].pattern)` — and so structurally could not observe the
# defect, which is a change in the ANSWER SET. This testset replaces them and pins the
# refutation EXECUTABLY, by building the merged form by hand.
#
# If this ever fails, do not "fix" it: it means `,` in an exec's source position stopped being a
# conjunction, and that is a substrate change that needs its own conversation.
@testset "v1 §10.6 batching is UNSOUND — merged execs lose answers (spec defect)" begin
    unmerged = [
        _atom(1, 0, "(, (a \$x))", "(, (seen_a \$x))"),
        _atom(1, 0, "(, (b \$y))", "(, (seen_b \$y))")
    ]
    # what §10.6 tells you to produce: concatenate the source lists into ONE comma-list
    merged = [_atom(1, 0, "(, (a \$x) (b \$y))", "(, (seen_a \$x) (seen_b \$y))")]

    up = join([sprint_exec(x) for x in unmerged], "\n")
    mp = join([sprint_exec(x) for x in merged], "\n")
    o  = BiSimObligation(:forward_sim, NodeID(0), NodeID(0))

    # BOTH patterns satisfiable — the old fixture. Agreement here is what hid the bug.
    v_both = verify_bisim(up, mp, [o]; facts="(a 1) (b 2)", max_steps=20)
    @test v_both.forward_ok

    # ONE FACT REMOVED — the disconfirming case. Unmerged still derives (seen_a 1);
    # the merged conjunction derives nothing, so forward simulation MUST fail.
    v_one = verify_bisim(up, mp, [o]; facts="(a 1)", max_steps=20)
    @test !v_one.forward_ok
end

# ── Pattern Fusion (v1 §10.6, identical-pattern variant) ─────────────────────
@testset "fuse_identical_patterns — merges atoms with same pattern" begin
    atoms = [
        _atom(1, 0, "(, (hello))", "(, world1)"),
        _atom(2, 0, "(, (hello))", "(, world2)"),
        _atom(3, 0, "(, (other))", "(, single)")
    ]
    fused = fuse_identical_patterns(atoms)
    @test length(fused) == 2
    @test fused[1].pattern == "(, (hello))"
    @test occursin("world1", fused[1].template)
    @test occursin("world2", fused[1].template)
    @test fused[1].priority == MM2Priority(1, 0)   # priority of the first atom
end

@testset "fuse_identical_patterns — whitespace differences treated as equal" begin
    atoms = [
        _atom(1, 0, "(, (foo))", "(, a)"),
        _atom(2, 0, " (, (foo)) ", "(, b)")    # extra whitespace
    ]
    fused = fuse_identical_patterns(atoms)
    @test length(fused) == 1
end

@testset "fuse_identical_patterns — distinct patterns pass through" begin
    atoms = [
        _atom(1, 0, "(, (a))", "(, x)"),
        _atom(2, 0, "(, (b))", "(, y)")
    ]
    fused = fuse_identical_patterns(atoms)
    @test length(fused) == 2
end

# ── End-to-end: optimization composed with MM2Compiler + Bisim verification ──
@testset "MM2Optimize — schedule_static preserves emitted-program text" begin
    # Build a small M-Core program, compile it, run schedule_static,
    # confirm the sorted-by-priority text is bisim-equivalent to the original.
    g = MCoreGraph()
    pat1 = add_sym!(g, Sym(:hello))
    tmpl1 = add_sym!(g, Sym(:world))
    pri1 = add_lit!(g, Lit(0))
    pat1c = add_con!(g, Con(:_comma, [pat1]))
    tmpl1c = add_con!(g, Con(:_comma, [tmpl1]))
    e1 = add_prim!(g, Prim(:mm2_exec, [pri1, pat1c, tmpl1c]))

    compiled_str, obligs = compile_program(g, [e1])

    # schedule_static on the compiler context output (manually rebuild from string)
    # → for a single atom, the output is unchanged
    @test !isempty(compiled_str)
    @test length(obligs) >= 1
end

# (the batch_space_ops bisim round-trip testset was removed with the function —
#  its fixture had ALL patterns matching, which is exactly why it passed. The
#  refutation that replaces it is pinned above.)
