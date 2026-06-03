using Test
using MorkSupercompiler

@testset "Rewrite (static)" begin
    @testset "conjunction reorder" begin
        conj = parse_sexpr("(, (parity \$i \$p) (succ 0 1) (\$x \$y \$z))")
        @test is_conjunction(conj)
        reordered = reorder_conjunction_static(conj::SList)
        items = reordered.items
        sources = items[2:end]
        scores = static_score.(sources)
        # Result must be sorted by ascending static_score
        @test issorted(scores)
        # (succ 0 1) is ground → score 0.0 → must be first
        @test static_score(sources[1]) == 0.0
        # (\$x \$y \$z) is fully variable → score 1.0 → must be last
        @test static_score(sources[end]) == 1.0
    end

    @testset "program reorder" begin
        prog = """((phase \$p) (, (parity \$i \$p) (succ 0 1) (\$x \$y \$z)) (O res))"""
        reordered = reorder_program_static(prog)
        # The program should still parse
        nodes = parse_program(reordered)
        @test length(nodes) == 1
        # First source in the conjunction should be the most selective
        conj = (nodes[1]::SList).items[2]
        @test is_conjunction(conj)
        first_src = (conj::SList).items[2]
        @test static_score(first_src) <= static_score((conj::SList).items[end])
    end
end
