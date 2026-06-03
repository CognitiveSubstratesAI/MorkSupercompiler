using Test
using MorkSupercompiler

@testset "Statistics" begin
    stats = MORKStatistics()
    @test stats.total_atoms == 0

    @testset "estimate_cardinality — no stats" begin
        src = parse_sexpr("(parity \$i \$p)")
        # With no stats, falls back to total_atoms / 4 = 0 → max(1,...) = 1
        card = estimate_cardinality(src, stats)
        @test card >= 1
    end

    @testset "estimate_cardinality — with predicate count" begin
        stats2 = MORKStatistics(Dict("parity" => 5, "succ" => 5, "lt" => 10), 20)
        @test estimate_cardinality(parse_sexpr("(parity \$i \$p)"), stats2) == 5
        @test estimate_cardinality(parse_sexpr("(lt \$x \$y)"), stats2) == 10
        # Ground atom: should give 1 (no variables to expand)
        @test estimate_cardinality(parse_sexpr("(parity 0 even)"), stats2) >= 1
    end
end
