# Lossless projection regression for PipelineDecompose.
#
# Property under test: decomposition's flow_vars carries every variable the
# final template needs through every _sc_tmp* intermediate, transitively.
# A variable can only be dropped from an intermediate if it appears in
# neither a downstream source nor the final template — in which case it
# provably can't distinguish final results, so MORK dedup of identical
# intermediates is semantically equivalent to the direct engine.
#
# Two cases:
#   Test 1 (3-source, single hop):  one _sc_tmp0 must carry $x and $z.
#   Test 2 (4-source, chained):     _sc_tmp0 → _sc_tmp1; $x must survive both.
#
# Both run direct (`space_metta_calculus!`) and decomposed (`plan!`) and
# diff the final-atom SET.  Mismatch = regression in flow_vars's
# final_template projection (line 64 of PipelineDecompose.jl).

using Test
using MORK: new_space, space_add_all_sexpr!, space_dump_all_sexpr,
            space_metta_calculus!
using MorkSupercompiler: plan!

function _result_set(dump::String) :: Set{String}
    Set(filter(l -> startswith(strip(l), "(Result"), strip.(split(dump, '\n'))))
end

function _run_both(facts::String, program::String)
    s_direct = new_space()
    space_add_all_sexpr!(s_direct, facts)
    space_add_all_sexpr!(s_direct, program)
    space_metta_calculus!(s_direct, typemax(Int))
    direct = _result_set(space_dump_all_sexpr(s_direct))

    s_decomp = new_space()
    space_add_all_sexpr!(s_decomp, facts)
    plan!(s_decomp, program, typemax(Int))
    decomp = _result_set(space_dump_all_sexpr(s_decomp))

    (direct, decomp)
end

@testset "PipelineDecompose lossless projection" begin

    @testset "3-source single-hop (x is final-only)" begin
        # (A $x $y)(B $y $z)(C $z $w) → (Result $x $w)
        # Two $x values share $y=shared.  Without final-template threading,
        # _sc_tmp0 would carry only $z, dedup would collapse, only 1 result.
        facts = """
        (A val1 shared)
        (A val2 shared)
        (B shared z_val)
        (C z_val w_val)
        """
        program = raw"""
        (exec 0 (, (A $x $y) (B $y $z) (C $z $w)) (, (Result $x $w)))
        """
        direct, decomp = _run_both(facts, program)
        @test direct == decomp
        @test length(direct) == 2
        @test "(Result val1 w_val)" in direct
        @test "(Result val2 w_val)" in direct
    end

    @testset "4-source chained (x must survive 2 intermediate hops)" begin
        # (A $x $y1)(B $y1 $y2)(C $y2 $y3)(D $y3 $w) → (Result $x $w)
        # Decomposes to _sc_tmp0($x,$y2) → _sc_tmp1($x,$y3) → final.
        # The transitive case: every recursive _build_chain! call must
        # re-thread final_template through flow_vars to keep $x alive.
        facts = """
        (A val1 a)
        (A val2 a)
        (B a b)
        (C b c)
        (D c d)
        """
        program = raw"""
        (exec 0 (, (A $x $y1) (B $y1 $y2) (C $y2 $y3) (D $y3 $w)) (, (Result $x $w)))
        """
        direct, decomp = _run_both(facts, program)
        @test direct == decomp
        @test length(direct) == 2
        @test "(Result val1 d)" in direct
        @test "(Result val2 d)" in direct
    end
end
