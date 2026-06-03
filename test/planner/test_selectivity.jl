using Test
using MorkSupercompiler

@testset "Selectivity" begin
    ground = parse_sexpr("(parity 0 even)")
    partial = parse_sexpr("(parity \$i even)")
    all_var = parse_sexpr("(parity \$i \$p)")
    bare_var = parse_sexpr("\$x")

    @testset "static_score" begin
        @test static_score(ground) == 0.0
        @test static_score(partial) < static_score(all_var)
        @test static_score(bare_var) == 1.0
    end

    @testset "count_vars / count_atoms" begin
        @test count_vars(ground) == 0
        @test count_atoms(ground) == 3
        @test count_vars(all_var) == 2
        @test count_atoms(all_var) == 1
    end

    @testset "is_ground / is_conjunction" begin
        @test is_ground(ground)
        @test !is_ground(partial)
        conj = parse_sexpr("(, (a \$x) (b \$y))")
        @test is_conjunction(conj)
        @test !is_conjunction(ground)
    end
end
