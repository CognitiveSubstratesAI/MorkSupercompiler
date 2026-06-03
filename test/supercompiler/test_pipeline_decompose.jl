using Test
using MorkSupercompiler

# ── flow_vars ─────────────────────────────────────────────────────────────────

@testset "flow_vars — basic variable flow" begin
    # 3 sources: (edge $x $y), (edge $y $z), (edge $z $w)
    srcs = parse_program("(edge \$x \$y) (edge \$y \$z) (edge \$z \$w)")
    # After first 1 source, $x and $y introduced; only $y needed in srcs 2-3
    fv = flow_vars(srcs, 1, 3)
    @test "\$y" in fv
    @test !("\$x" in fv)   # $x not needed in srcs 2-3 (no final_template)
end

@testset "flow_vars — with final_template includes output vars" begin
    srcs = parse_program("(edge \$x \$y) (edge \$y \$z) (edge \$z \$w)")
    tpl = only(parse_program("(dtrans \$x \$y \$z \$w)"))
    # $x is introduced in src 1 and needed in template
    fv = flow_vars(srcs, 1, 3; final_template=tpl)
    @test "\$y" in fv
    @test "\$x" in fv   # now carried because template needs it
end

@testset "flow_vars — all sources, full flow" begin
    srcs = parse_program("(state \$ts \$ic) (program \$i \$op) (state \$ts \$reg)")
    # After first 2 sources: $ts, $i, $op, $ic introduced; $ts needed in src 3
    fv = flow_vars(srcs, 2, 3)
    @test "\$ts" in fv
end

@testset "flow_vars — no overlap returns empty" begin
    # Sources share no variables between stages
    srcs = parse_program("(a \$x) (b \$y) (c \$z)")
    fv = flow_vars(srcs, 1, 3)
    @test isempty(fv)
end

@testset "flow_vars — deterministic sort" begin
    srcs = parse_program("(p \$b \$a) (q \$a \$b \$c)")
    fv1 = flow_vars(srcs, 1, 2)
    fv2 = flow_vars(srcs, 1, 2)
    @test fv1 == fv2
    @test issorted(fv1)
end

# ── decompose_exec — pass-through for small source counts ─────────────────────

@testset "decompose_exec — 1-source passes through" begin
    atom = only(parse_program("(exec 0 (, (edge \$x \$y)) (, (r \$x \$y)))"))
    dp = decompose_exec(atom)
    @test length(dp.stages) == 1
    @test dp.n_intermediate == 0
end

@testset "decompose_exec — 2-source passes through (≤ STAGE_MAX_SOURCES)" begin
    atom = only(parse_program("(exec 0 (, (edge \$x \$y) (edge \$y \$z)) (, (p \$x \$z)))"))
    dp = decompose_exec(atom)
    @test length(dp.stages) == 1
    @test dp.n_intermediate == 0
end

@testset "decompose_exec — non-list passes through" begin
    atom = SAtom("not-a-list")
    dp = decompose_exec(atom)
    @test length(dp.stages) == 1
    @test dp.n_intermediate == 0
end

# ── decompose_exec — decomposition for N > STAGE_MAX_SOURCES ─────────────────

@testset "decompose_exec — 3-source decomposes into 2 stages" begin
    prog = "(exec 0 (, (edge \$x \$y) (edge \$y \$z) (edge \$z \$w)) (, (dtrans \$x \$w)))"
    atom = only(parse_program(prog))
    dp = decompose_exec(atom)
    @test length(dp.stages) == 2
    @test dp.n_intermediate >= 1
    # Each stage must be a list (exec ...) form
    for s in dp.stages
        @test s isa SList
    end
end

@testset "decompose_exec — 5-source decomposes into multiple stages" begin
    prog = "(exec 0 (, (a \$x \$y) (b \$y \$z) (c \$z \$u) (d \$u \$v) (e \$v \$w)) (, (r \$x \$w)))"
    atom = only(parse_program(prog))
    dp = decompose_exec(atom)
    # 5 sources with STAGE_MAX_SOURCES=2 → should require at least 3 stages
    @test length(dp.stages) >= 3
    @test dp.original_sources == 5
end

@testset "decompose_exec — intermediate atoms have _sc_tmp prefix" begin
    prog = "(exec 0 (, (a \$x \$y) (b \$y \$z) (c \$z \$w)) (, (r \$x \$w)))"
    atom = only(parse_program(prog))
    dp = decompose_exec(atom)
    # Intermediate atoms in non-final stages should contain _sc_tmp
    all_text = join(sprint_sexpr.(dp.stages), " ")
    @test occursin(SC_TMP_PREFIX, all_text)
end

@testset "decompose_exec — counter monotonically increases" begin
    prog = "(exec 0 (, (a \$x) (b \$x \$y) (c \$y \$z) (d \$z)) (, (r \$x \$z)))"
    atom = only(parse_program(prog))
    c1 = Ref(0)
    c2 = Ref(5)
    dp1 = decompose_exec(atom; counter=c1)
    dp2 = decompose_exec(atom; counter=c2)
    # Same structural output regardless of starting counter
    @test length(dp1.stages) == length(dp2.stages)
    # Second run should use tmp5, tmp6, etc. — prefix differs
    txt2 = join(sprint_sexpr.(dp2.stages), " ")
    @test occursin("$(SC_TMP_PREFIX)5", txt2)
end

# ── decompose_program — full program transformation ───────────────────────────

@testset "decompose_program — small program unchanged" begin
    prog = "(exec 0 (, (a \$x) (b \$x)) (, (r \$x)))"
    out = decompose_program(prog)
    # 2-source: no decomposition needed
    @test length(parse_program(out)) == 1
end

@testset "decompose_program — trans_detect 3-source" begin
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $y $z $w)))"
    out = decompose_program(prog)
    nodes = parse_program(out)
    @test length(nodes) == 2   # 3-source → 2 stages
    # Each node is an exec form starting with "exec"
    for n in nodes
        @test n isa SList
        items = (n::SList).items
        @test items[1] isa SAtom
    end
end

@testset "decompose_program — multiple atoms in program" begin
    prog = """
    (exec 0 (, (a \$x) (b \$x \$y)) (, (p \$x \$y)))
    (exec 1 (, (c \$u) (d \$u \$v) (e \$v \$w)) (, (q \$u \$w)))
    """
    out = decompose_program(prog)
    nodes = parse_program(out)
    # First atom: 2-source (pass-through), second: 3-source → 2 stages
    @test length(nodes) == 3
end

@testset "decompose_program — counter unique across atoms" begin
    # Two 3-source atoms → each should get unique _sc_tmp indices
    prog = """
    (exec 0 (, (a \$x \$y) (b \$y \$z) (c \$z \$w)) (, (p \$x \$w)))
    (exec 1 (, (d \$u \$v) (e \$v \$r) (f \$r \$s)) (, (q \$u \$s)))
    """
    out = decompose_program(prog)
    @test occursin("_sc_tmp0", out)
    @test occursin("_sc_tmp1", out)   # second atom gets _sc_tmp1 (counter carries over)
end

# ── decompose_report ──────────────────────────────────────────────────────────

@testset "decompose_report — no multi-source atoms" begin
    prog = "(exec 0 (, (a \$x) (b \$x)) (, (r \$x)))"
    report = decompose_report(prog)
    @test occursin("no multi-source atoms", report)
end

@testset "decompose_report — shows decomposed atoms" begin
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $w)) (, (dtrans $x $w)))"
    report = decompose_report(prog)
    @test occursin("Decomposed", report)
    @test occursin("Stage 1", report)
    @test occursin("Stage 2", report)
end

@testset "decompose_report — source count in output" begin
    prog = raw"(exec 0 (, (a $x $y) (b $y $z) (c $z $u) (d $u $v)) (, (r $x $v)))"
    report = decompose_report(prog)
    # Should mention source count
    @test occursin("4 sources", report)
end

# ── STAGE_MAX_SOURCES constant ────────────────────────────────────────────────

@testset "STAGE_MAX_SOURCES constant" begin
    @test STAGE_MAX_SOURCES == 2
    @test SC_TMP_PREFIX == "_sc_tmp"
end

# ── Canonical case: odd_even_sort (5-source) ──────────────────────────────────

@testset "decompose_program — odd_even_sort rule form NOT decomposed" begin
    # Rule definitions like ((phase $p) (, ...) (O ...)) are NOT exec atoms.
    # MORK only executes top-level (exec ...) atoms via space_metta_calculus!.
    # Decomposing rule heads breaks their invocation semantics — pass through unchanged.
    prog = raw"""
    ((phase $p)
     (, (parity $i $p) (succ $i $si) (A $i $e) (A $si $se) (lt $se $e))
     (O (- (A $i $e)) (- (A $si $se)) (+ (A $i $se)) (+ (A $si $e))))
    """
    out = decompose_program(prog)
    nodes = parse_program(out)
    @test length(nodes) == 1   # unchanged — not an exec atom
    @test !occursin(SC_TMP_PREFIX, out)
end

@testset "decompose_program — counter_machine rule form NOT decomposed" begin
    prog = raw"""
    ((step JZ $ts)
     (, (state $ts (IC $i)) (program $i (JZ $r $j)) (state $ts (REG $r $v))
        (if $v (S $i) $j $ni) (state $ts (REG $k $kv)))
     (, (state (S $ts) (IC $ni)) (state (S $ts) (REG $k $kv))))
    """
    out = decompose_program(prog)
    nodes = parse_program(out)
    @test length(nodes) == 1   # unchanged — not an exec atom
    @test !occursin(SC_TMP_PREFIX, out)
end

@testset "decompose_program — exec 5-source IS decomposed" begin
    # An actual exec atom with 5 sources does get decomposed
    prog = raw"(exec 0 (, (a $x $y) (b $y $z) (c $z $u) (d $u $v) (e $v $w)) (, (r $x $w)))"
    out = decompose_program(prog)
    @test length(parse_program(out)) >= 3
    @test occursin(SC_TMP_PREFIX, out)
end
