using Test
using MorkSupercompiler

@testset "SExpr parser" begin
    @testset "atoms" begin
        n = parse_sexpr("foo")
        @test n isa SAtom && (n::SAtom).name == "foo"

        n = parse_sexpr("foo-bar")
        @test n isa SAtom && (n::SAtom).name == "foo-bar"

        n = parse_sexpr("!=")
        @test n isa SAtom && (n::SAtom).name == "!="

        n = parse_sexpr("42")
        @test n isa SAtom && (n::SAtom).name == "42"
    end

    @testset "variables" begin
        n = parse_sexpr("\$x")
        @test n isa SVar && (n::SVar).name == "\$x"

        n = parse_sexpr("\$ts")
        @test n isa SVar && (n::SVar).name == "\$ts"
    end

    @testset "lists" begin
        n = parse_sexpr("(foo bar baz)")
        @test n isa SList
        items = (n::SList).items
        @test length(items) == 3
        @test (items[1]::SAtom).name == "foo"
        @test (items[2]::SAtom).name == "bar"

        n = parse_sexpr("()")
        @test n isa SList && isempty((n::SList).items)

        n = parse_sexpr("(parity \$i \$p)")
        @test n isa SList
        items = (n::SList).items
        @test (items[1]::SAtom).name == "parity"
        @test (items[2]::SVar).name == "\$i"
        @test (items[3]::SVar).name == "\$p"
    end

    @testset "nested" begin
        n = parse_sexpr("((phase \$p) (, (parity \$i \$p)) (O x))")
        @test n isa SList
        items = (n::SList).items
        @test items[1] isa SList
        conj = items[2]
        @test is_conjunction(conj)
    end

    @testset "program" begin
        src = """
        (foo bar)
        ; comment
        (baz \$x \$y)
        """
        nodes = parse_program(src)
        @test length(nodes) == 2
    end

    @testset "grammar conformance (was: ':' whitelist bug)" begin
        # Regression: previously the inclusion-based whitelist rejected ':' and
        # the bracket family, breaking on stdlib atoms like
        #   (: + (-> Number Number Number))
        # Reference: docs/specs/metta_grammar.ebnf; audit:
        # docs/specs/metta_grammar_audit.md.

        n = parse_sexpr("(: foo (-> Number Number))")
        @test n isa SList
        items = (n::SList).items
        @test (items[1]::SAtom).name == ":"
        @test (items[2]::SAtom).name == "foo"

        # bare ':' as a symbol
        n = parse_sexpr(":")
        @test n isa SAtom && (n::SAtom).name == ":"

        # bracket family chars admitted as symbol body
        for s in ("[", "]", "{", "}", "[x]", "{a,b}")
            n = parse_sexpr(s)
            @test n isa SAtom && (n::SAtom).name == s
        end

        # roundtrip a stdlib-style typing atom
        e = "(: + (-> Number Number Number))"
        @test sprint_sexpr(parse_sexpr(e)) == e
    end

    @testset "roundtrip" begin
        exprs = [
            "(parity \$i \$p)",
            "(exec 0 (, (edge \$x \$y) (edge \$y \$z)) (, (trans \$x \$z)))",
            "((phase \$p) (, (parity \$i \$p) (succ \$i \$si) (A \$i \$e)) (O x))"
        ]
        for e in exprs
            n = parse_sexpr(e)
            @test sprint_sexpr(n) == e
        end
    end
end
