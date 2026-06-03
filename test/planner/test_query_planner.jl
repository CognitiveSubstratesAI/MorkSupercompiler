using Test
using MorkSupercompiler

@testset "QueryPlanner" begin
    @testset "effects_commute" begin
        @test effects_commute(EFF_READ, EFF_READ)
        @test effects_commute(EFF_PURE, EFF_WRITE)
        @test !effects_commute(EFF_WRITE, EFF_APPEND)
    end

    @testset "variable flow" begin
        sources = parse_program("(parity \$i \$p)\n(succ \$i \$si)\n(A \$si \$se)")
        stats2 = MORKStatistics(Dict("parity" => 5, "succ" => 5, "A" => 5), 15)
        nodes = build_join_nodes(sources, stats2)
        @test length(nodes) == 3
        # \$i introduced by (parity \$i \$p) so succ and A should have \$i in vars_in
        @test "\$i" in nodes[2].vars_in
        @test "\$si" in nodes[3].vars_in
    end

    @testset "plan_join_order" begin
        # Ground source should be first (card=1 beats card=5)
        stats3 = MORKStatistics(Dict("parity" => 5, "lt" => 10, "succ" => 1), 20)
        sources = parse_program("(lt \$x \$y)\n(parity \$i \$p)\n(succ 0 1)")
        nodes = build_join_nodes(sources, stats3)
        perm = plan_join_order(nodes)
        # Lowest card wins: succ has card=1 (and is ground) → should be first
        @test perm[1] == 3   # succ 0 1 is source 3 (1-indexed)
    end

    @testset "plan_program" begin
        prog = """((phase \$p) (, (lt \$x \$y) (parity \$i \$p) (succ 0 1)) (O res))"""
        stats3 = MORKStatistics(Dict("parity" => 5, "lt" => 10, "succ" => 1), 16)
        planned = plan_program(prog, stats3)
        nodes = parse_program(planned)
        conj = ((nodes[1]::SList).items[2]::SList)
        first_src = conj.items[2]
        # Lowest-card source first: succ 0 1 (card=1)
        @test sprint_sexpr(first_src) == "(succ 0 1)"
    end
end

@testset "PureRegion identification" begin
    @testset "identify_pure_regions — all-pure conjunction" begin
        # Extract sources from a conjunction list
        conj = parse_sexpr("(, (a \$x) (b \$x) (c \$x))")
        sources = (conj::SList).items[2:end]
        regions = identify_pure_regions(sources)
        @test length(regions) == 1          # single pure region
        @test regions[1].is_pure            # all Read → pure
        @test length(regions[1].sources) == 3
    end

    @testset "plan_query — reorders by cardinality" begin
        # (lt $x $y) has 50 atoms, (succ $i $si) has 5 — succ should come first
        conj = parse_sexpr("(, (lt \$x \$y) (succ \$i \$si))")
        sources = (conj::SList).items[2:end]
        stats = MORKStatistics(Dict("lt" => 50, "succ" => 5), 55)
        planned, barriers = plan_query(sources, stats)
        @test isempty(barriers)             # all-pure → no barriers
        @test length(planned) == 2
        # succ (card=5) should precede lt (card=50)
        first_head = (planned[1]::SList).items[1]
        @test (first_head::SAtom).name == "succ"
    end
end
