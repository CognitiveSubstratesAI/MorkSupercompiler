using Test
using MorkSupercompiler

# §3.1 Cost model
@testset "CostWeights + total_cost (§3.1)" begin
    w = balanced()
    @test w.α ≈ 1/3 atol=1e-9
    @test w.β ≈ 1/3 atol=1e-9
    @test w.γ ≈ 1/3 atol=1e-9

    ws = safety_critical()
    @test ws.β > ws.α   # safety: accuracy > speed
    we = exploratory()
    @test we.α > we.β   # exploratory: speed > accuracy

    err_pb = pbox_interval(0.0, 0.1, 1.0)
    c = total_cost(0.5, err_pb, balanced())
    @test c > 0.0
    # Safety-critical penalizes error/variance; exploratory penalizes time.
    # Use a scenario where error dominates (small time, large error interval).
    err_large = pbox_interval(0.0, 0.8, 1.0)   # wide error interval
    c_safe_big = total_cost(0.1, err_large, safety_critical())
    c_expl_big = total_cost(0.1, err_large, exploratory())
    @test c_safe_big > c_expl_big   # safety_critical β=0.6 > exploratory β=0.2
end

# §3.2 Algorithm 2 — EstimateCardinalityPBox
@testset "EstimateCardinalityPBox (Algorithm 2, §3.2)" begin
    stats = MORKStatistics(Dict("edge" => 10, "node" => 5), 15)
    src = parse_sexpr("(edge \$x \$y)")

    result = estimate_cardinality_pbox(src, stats)
    @test result isa EstimateCardinalityPBox
    @test result.point_estimate >= 1
    @test !isempty(result.pbox.intervals)

    # Point estimate matches Doc 1 estimate_cardinality
    @test result.point_estimate == estimate_cardinality(src, stats)
end

@testset "EstimateCardinalityPBox — Hoeffding interval (with btm)" begin
    using MORK
    s = new_space()
    space_add_all_sexpr!(s, "(edge 0 1) (edge 1 2) (edge 2 3) (edge 3 4)")
    stats = collect_stats(s)
    src = parse_sexpr("(edge \$x \$y)")

    result = estimate_cardinality_pbox(src, stats, s.btm; confidence=0.95)
    @test result.allows_sampling
    lo, hi = result.pbox.intervals[1]
    @test lo >= 0.0       # lower bound is non-negative
    @test hi > lo         # interval has positive width
    @test result.pbox.probabilities[1] ≈ 0.95   # confidence level matches
end

# §3.3 Algorithm 3 — ApproximateSplit
@testset "ApproximateSplit (Algorithm 3, §3.3)" begin
    nodes = parse_program("(a \$x)\n(b \$y)\n(c \$z)\n(d \$w)")
    branches = [(nodes[1], 0.4), (nodes[2], 0.35), (nodes[3], 0.15), (nodes[4], 0.1)]

    result = approximate_split(branches, 0.2)
    @test result isa ApproximateSplitResult
    @test !isempty(result.selected)
    @test result.total_error <= 0.2 + 1e-9   # within tolerance
    @test result.within_budget

    # Total selected prob + pruned prob = 1
    sel_prob = sum(b.probability for b in result.selected; init=0.0)
    prune_prob = sum(b.error_contrib for b in result.pruned; init=0.0)
    @test sel_prob + prune_prob ≈ 1.0 atol=1e-9
end

@testset "ApproximateSplit — tight tolerance keeps all branches" begin
    nodes = parse_program("(a \$x)\n(b \$y)")
    branches = [(nodes[1], 0.6), (nodes[2], 0.4)]
    result = approximate_split(branches, 0.0)   # tolerance=0 → keep all
    @test length(result.selected) == 2
    @test isempty(result.pruned)
    @test result.total_error ≈ 0.0
end

# plan_join_order_approx
@testset "plan_join_order_approx — uses 3-objective cost" begin
    using MORK
    s = new_space()
    space_add_all_sexpr!(s, "(a 1) (a 2) (a 3) (b 1)")
    stats = collect_stats(s)
    sources = parse_program("(a \$x)\n(b \$x)")

    order = plan_join_order_approx(sources, stats, s.btm; weights=balanced())
    @test length(order) == 2
    @test Set(order) == Set(1:2)
    # (b \$x) has fewer atoms → should come first
    @test order[1] == 2   # b is more selective
end
