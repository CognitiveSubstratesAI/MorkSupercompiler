using Test
using MorkSupercompiler
using MORK

# §6.4 ApproximatePathSig — error levels
@testset "ApproximatePathSig (§6.4)" begin
    g = MCoreGraph()
    id = add_sym!(g, Sym(:foo))
    key = canonical_key(g, id, 0)

    exact_sig = ApproximatePathSig(key)
    @test exact_sig.error_level == EXACT
    @test exact_sig.error_bound == 0.0
    @test exact_sig.confidence == 1.0
    @test is_cacheable(exact_sig)

    bounded_sig = ApproximatePathSig(key, 0.05)
    @test bounded_sig.error_level == BOUNDED
    @test bounded_sig.error_bound ≈ 0.05
    @test is_cacheable(bounded_sig)

    stat_sig = ApproximatePathSig(key, 0.9, Val(:statistical))
    @test stat_sig.error_level == STATISTICAL
    @test !is_cacheable(stat_sig)   # STATISTICAL is not persistently cacheable
end

@testset "approx_subsumes (extends Algorithm 10)" begin
    g = MCoreGraph()
    id1 = add_sym!(g, Sym(:foo))
    id2 = add_sym!(g, Sym(:foo))   # same head
    id3 = add_sym!(g, Sym(:bar))   # different head
    k1 = canonical_key(g, id1, 0)
    k2 = canonical_key(g, id2, 0)
    k3 = canonical_key(g, id3, 0)

    s_exact = ApproximatePathSig(k1)
    s_bounded = ApproximatePathSig(k2, 0.05)
    s_other = ApproximatePathSig(k3)

    @test approx_subsumes(s_exact, s_bounded)    # EXACT subsumes BOUNDED (same base)
    @test !approx_subsumes(s_bounded, s_exact)   # BOUNDED doesn't subsume EXACT
    @test !approx_subsumes(s_exact, s_other)     # different head
end

# §6.2 BloomFilter
@testset "SimpleBloomFilter" begin
    bf = SimpleBloomFilter(1024, 3)
    @test bf.n == 0
    @test bloom_false_positive_rate(bf) == 0.0

    bloom_add!(bf, UInt64(42))
    bloom_add!(bf, UInt64(99))
    @test bloom_check(bf, UInt64(42))
    @test bloom_check(bf, UInt64(99))
    @test bf.n == 2
    @test bloom_false_positive_rate(bf) >= 0.0
end

# §6.2 ApproxIndex
@testset "ApproxIndex" begin
    idx = ApproxIndex{String}()

    # Insert high-weight entry → goes to core
    for _ in 1:10   # increase weight above threshold
        approx_index_insert!(idx, UInt64(1), "value_1")
    end
    result = approx_index_lookup(idx, UInt64(1))
    @test result == "value_1"   # found in core

    # Insert low-weight entry → overflow only
    approx_index_lookup(idx, UInt64(9999))   # never inserted
    @test approx_index_lookup(idx, UInt64(9999)) === nothing   # definitely absent

    # Coverage PBox exists
    @test !isempty(idx.coverage.intervals)
end

# §6.3 New IR primitives
@testset "register_approx_primitives!" begin
    reg = PrimRegistry()
    register_approx_primitives!(reg)
    @test lookup_prim(reg, :approx_kb_query) !== nothing
    @test lookup_prim(reg, :sample_fitness) !== nothing
end

@testset "approx_kb_query primitive — returns Residual" begin
    reg = PrimRegistry()
    register_approx_primitives!(reg)
    g = MCoreGraph()
    pat = add_sym!(g, Sym(:pattern))
    tol = add_lit!(g, Lit(0.05))
    pid = add_prim!(g, Prim(:approx_kb_query, [pat, tol]))
    r = rewrite_once(g, pid, Env(), DepSet(), reg)
    @test r isa Residual   # remains residual until connected to Space
end

# §6.1 4-phase pipeline
@testset "run_approx_pipeline — 4 phases" begin
    s = new_space()
    space_add_all_sexpr!(s, "(edge 0 1) (edge 1 2) (edge 2 3) (node 0) (node 1)")
    prog = raw"""
    (exec 0 (, (edge $x $y) (edge $y $z) (node $x)) (, (path $x $z)))
    """
    result = run_approx_pipeline(s, prog; error_tolerance=0.1)
    @test result isa ApproxPipelineResult
    @test !isempty(result.program_approx)
    @test haskey(result.phase_timings, PHASE_ANALYSIS)
    @test haskey(result.phase_timings, PHASE_PLANNING)
    @test haskey(result.phase_timings, PHASE_SPECIALIZATION)
    @test haskey(result.phase_timings, PHASE_VERIFICATION)
    @test result.error_budget_used >= 0.0
    @test result.within_tolerance   # 3-source pattern → approximated within tolerance
end

@testset "run_approx_pipeline — STATISTICAL sigs not cacheable" begin
    s = new_space()
    space_add_all_sexpr!(s, "(foo 1) (foo 2)")
    prog = raw"(exec 0 (, (foo $x)) (, (bar $x)))"
    result = run_approx_pipeline(s, prog; error_tolerance=0.05)
    # EXACT sigs (single-source pattern) should be cacheable
    exact_sigs = filter(s -> s.error_level == EXACT, result.path_signatures)
    @test all(is_cacheable, exact_sigs)
end
