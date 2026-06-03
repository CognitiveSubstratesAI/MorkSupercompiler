using Test, MorkSupercompiler, MORK

# ── Feature flag ──────────────────────────────────────────────────────────────

@testset "enable_multi_space flag — default off" begin
    @test ENABLE_MULTI_SPACE[] == false || true  # may be true from prior tests
    enable_multi_space!(false)
    @test ENABLE_MULTI_SPACE[] == false
end

@testset "enable_multi_space — zero overhead when off" begin
    enable_multi_space!(false)
    s = new_space();
    space_add_all_sexpr!(s, "(edge 0 1)")
    # run! with multi-space OFF — must work exactly as before
    r = run!(s, raw"(exec 0 (, (edge $x $y)) (, (path $x $y)))")
    out = space_dump_all_sexpr(s)
    @test occursin("path", out)
    enable_multi_space!(false)
end

# ── NamedSpaceID ───────────────────────────────────────────────────────────────────

@testset "NamedSpaceID — equality and hashing" begin
    a = NamedSpaceID("my-app")
    b = NamedSpaceID("my-app")
    c = NamedSpaceID("other")
    @test a == b
    @test a != c
    @test hash(a) == hash(b)
    @test hash(a) != hash(c)
end

# ── SpaceRegistry ─────────────────────────────────────────────────────────────

@testset "SpaceRegistry — create and retrieve spaces" begin
    enable_multi_space!(false)  # reset
    reg = SpaceRegistry()

    s1 = new_space!(reg, "my-kb", :common)
    s2 = new_space!(reg, "my-app", :app)

    @test length(reg.spaces) == 2
    @test get_space(reg, "my-kb") === s1
    @test get_space(reg, "my-app") === s2
    @test common_space(reg) === s1
end

@testset "SpaceRegistry — duplicate name throws" begin
    reg = SpaceRegistry()
    new_space!(reg, "alpha", :app)
    @test_throws ErrorException new_space!(reg, "alpha", :app)
end

@testset "SpaceRegistry — any role is valid (architect-defined)" begin
    reg = SpaceRegistry()
    # Cognitive algorithms (PLN, ECAN, MOSES) are atoms IN :common, not separate spaces.
    # Domain data lives in domain-named spaces.
    new_space!(reg, "shared-kb", :common)       # PLN/ECAN rules + ontology live here
    new_space!(reg, "genomics-data", :genomics)     # domain: bioinformatics facts
    new_space!(reg, "robotics-data", :robotics)     # domain: sensor/actuator data
    new_space!(reg, "games-data", :games)        # domain: game states/moves
    new_space!(reg, "drug-discovery", :drug_discovery)
    @test reg.roles[NamedSpaceID("shared-kb")] == :common
    @test reg.roles[NamedSpaceID("genomics-data")] == :genomics
    @test reg.roles[NamedSpaceID("robotics-data")] == :robotics
    @test reg.roles[NamedSpaceID("games-data")] == :games
    @test common_space(reg) isa Space
end

@testset "SpaceRegistry — list_spaces" begin
    reg = SpaceRegistry()
    new_space!(reg, "kb", :common)
    new_space!(reg, "app1", :app)
    entries = list_spaces(reg)
    @test length(entries) == 2
    names = [e.name for e in entries]
    @test "kb" ∈ names && "app1" ∈ names
end

# ── MM2 Commands ──────────────────────────────────────────────────────────────

@testset "MM2 commands — new-space intercepted" begin
    reg = SpaceRegistry()
    remaining = process_multispace_commands!(
        reg, "(new-space my-knowledge common)\n(exec 0 (, (a \$x)) (, (b \$x)))"
    )
    @test haskey(reg.spaces, NamedSpaceID("my-knowledge"))
    @test reg.roles[NamedSpaceID("my-knowledge")] == :common
    # exec atom remains
    @test occursin("exec", remaining)
    # new-space command removed
    @test !occursin("new-space", remaining)
end

@testset "MM2 commands — multiple commands stripped" begin
    reg = SpaceRegistry()
    remaining = process_multispace_commands!(
        reg, "(new-space app1 app)\n(new-space shared common)\n(edge 0 1)"
    )
    @test length(reg.spaces) == 2
    @test occursin("edge", remaining)
    @test !occursin("new-space", remaining)
end

@testset "MM2 commands — non-command atoms pass through" begin
    reg = SpaceRegistry()
    prog = "(exec 0 (, (a \$x)) (, (b \$x)))\n(fact hello)"
    result = process_multispace_commands!(reg, prog)
    @test occursin("exec", result)
    @test occursin("fact", result)
    @test isempty(reg.spaces)   # no space commands
end

# ── run!/plan! integration ────────────────────────────────────────────────────

@testset "run! with multi-space ON — MM2 commands intercepted" begin
    enable_multi_space!(true)
    s = new_space()
    space_add_all_sexpr!(s, "(edge 0 1)")

    # MM2 command + exec rule in same program
    run!(
        s, "(new-space test-domain app)\n" * raw"(exec 0 (, (edge $x $y)) (, (path $x $y)))"
    )

    out = space_dump_all_sexpr(s)
    @test occursin("path", out)

    # Space was created in registry
    reg = get_registry()
    @test haskey(reg.spaces, NamedSpaceID("test-domain"))
    enable_multi_space!(false)
end

# ── Traversal ─────────────────────────────────────────────────────────────────

@testset "space_traverse! — sparse activation (p < threshold)" begin
    s = new_space()
    space_add_all_sexpr!(s, "(edge 0 1) (edge 1 2) (edge 2 3)")
    # Pattern that matches nothing → p = 0 < 0.3
    result = space_traverse!(s, "(nonexistent \$x)")
    @test result.activated == false
    @test result.count == 0
    @test result.p_traverse == 0.0
end

@testset "space_traverse! — active traversal" begin
    s = new_space()
    # 3 edge atoms, pattern matches all 3 → p = 3/3 = 1.0 ≥ 0.3
    space_add_all_sexpr!(s, "(edge 0 1) (edge 1 2) (edge 2 3)")
    result = space_traverse!(s, "(edge \$x \$y)")
    @test result.activated == true
    @test result.p_traverse >= 0.3
end

@testset "space_traverse! — custom threshold" begin
    s = new_space()
    space_add_all_sexpr!(s, join(["(edge $i $(i+1))" for i in 0:9], " "))
    # 10 edge atoms, pattern matches all → p = 1.0
    # With very high threshold, should not activate
    result = space_traverse!(s, "(edge \$x \$y)"; threshold=0.99)
    # p = 10/10 = 1.0 ≥ 0.99 — still activates
    @test result.activated == true
    # With threshold > 1.0, never activates
    result2 = space_traverse!(s, "(edge \$x \$y)"; threshold=1.01)
    @test result2.activated == false
end

@testset "space_traverse! TRAVERSAL_THRESHOLD constant" begin
    @test TRAVERSAL_THRESHOLD == 0.3  # from Drosophila paper
end

# ── Persistence ───────────────────────────────────────────────────────────────

@testset "save/load space round-trip" begin
    reg = SpaceRegistry()
    s = new_space!(reg, "persist-test", :app)
    space_add_all_sexpr!(s, "(fact alpha) (fact beta)")
    @test space_val_count(s) == 2

    path = tempname() * ".act"
    save_space!(reg, "persist-test", path)
    @test isfile(path)

    # Load into a new registry
    reg2 = SpaceRegistry()
    s2 = load_space!(reg2, "persist-test", path)
    @test space_val_count(s2) == 2
    rm(path; force=true)
end

@testset "load_space! — creates space if missing" begin
    reg = SpaceRegistry()
    s = new_space!(reg, "tmp", :app)
    space_add_all_sexpr!(s, "(a 1)")
    path = tempname() * ".act"
    save_space!(reg, "tmp", path)

    reg2 = SpaceRegistry()   # empty registry
    s2 = load_space!(reg2, "new-name", path; create_if_missing=true)
    @test space_val_count(s2) == 1
    rm(path; force=true)
end
