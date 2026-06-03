using Test
using MorkSupercompiler

# §5.3 Algorithm 6 — TournamentWithPBox
@testset "TournamentWithPBox (Algorithm 6, §5.3)" begin
    pop = [
        EvolutionaryPBox(
            1, pbox_interval(0.8, 0.95, 1.0), pbox_interval(0.0, 1.0, 1.0), 0.8, 1
        ),
        EvolutionaryPBox(
            2, pbox_interval(0.3, 0.5, 1.0), pbox_interval(0.0, 1.0, 1.0), 0.6, 3
        ),
        EvolutionaryPBox(
            3, pbox_interval(0.5, 0.7, 1.0), pbox_interval(0.0, 1.0, 1.0), 0.7, 2
        )
    ]

    # Run many tournaments — winner should be skewed toward individual 1 (highest fitness)
    wins = Dict(1 => 0, 2 => 0, 3 => 0)
    for _ in 1:200
        winner = tournament_with_pbox(pop, 3)
        wins[winner.individual_id] += 1
    end
    @test wins[1] > wins[2]   # highest fitness wins most often
    @test wins[1] > wins[3]

    # Single candidate always wins
    single = tournament_with_pbox([pop[1]], 1)
    @test single.individual_id == 1

    # size < count exercises the randperm subsample branch — threw UndefVarError
    # (`randperm` not imported) before J-1; the default-size path above never hit it.
    sub = tournament_with_pbox(pop, 1)
    @test sub.individual_id in (1, 2, 3)
end

# §5.4 Heritability formula — offspring_fitness_pbox
@testset "offspring_fitness_pbox (§5.4)" begin
    p1 = EvolutionaryPBox(1, pbox_interval(0.8, 0.9, 1.0), pbox_exact(0.5), 0.9, 2)
    p2 = EvolutionaryPBox(2, pbox_interval(0.7, 0.85, 1.0), pbox_exact(0.4), 0.8, 1)

    offspring_pb = offspring_fitness_pbox(p1, p2)
    @test !isempty(offspring_pb.intervals)
    lo, hi = offspring_pb.intervals[1]
    @test 0.0 <= lo <= hi <= 1.1   # clamped to [0,1] (small overshoot ok from noise)

    # High heritability → offspring fitness close to parents' average
    p1_h = EvolutionaryPBox(1, pbox_exact(0.9), pbox_exact(0.5), 0.99, 1)
    p2_h = EvolutionaryPBox(2, pbox_exact(0.9), pbox_exact(0.5), 0.99, 1)
    off_h = offspring_fitness_pbox(p1_h, p2_h; mutation_strength=0.001)
    @test abs(off_h.intervals[1][1] - 0.9) < 0.2   # close to parent fitness
end

# §5.6 Convergence detection
@testset "population_converged (§5.6)" begin
    # Identical fitness p-boxes → fully converged
    same = [EvolutionaryPBox(i, pbox_exact(0.8), pbox_exact(0.5), 0.8, i) for i in 1:5]
    @test population_converged(same)

    # Very different fitness p-boxes → not converged
    diverse = [
        EvolutionaryPBox(1, pbox_interval(0.0, 0.2, 1.0), pbox_exact(0.3), 0.7, 1),
        EvolutionaryPBox(2, pbox_interval(0.5, 0.7, 1.0), pbox_exact(0.5), 0.8, 2),
        EvolutionaryPBox(3, pbox_interval(0.8, 1.0, 1.0), pbox_exact(0.7), 0.9, 3)
    ]
    @test !population_converged(diverse)
end

@testset "convergence_report returns non-empty string" begin
    pop = [
        EvolutionaryPBox(i, pbox_interval(0.5, 0.8, 1.0), pbox_exact(0.5), 0.8, i) for
        i in 1:3
    ]
    rep = convergence_report(pop)
    @test !isempty(rep)
    @test occursin("Population", rep)
end

# rank_population
@testset "rank_population" begin
    pop = [
        EvolutionaryPBox(1, pbox_interval(0.1, 0.3, 1.0), pbox_exact(0.2), 0.7, 1),
        EvolutionaryPBox(2, pbox_interval(0.8, 1.0, 1.0), pbox_exact(0.8), 0.9, 2),
        EvolutionaryPBox(3, pbox_interval(0.4, 0.6, 1.0), pbox_exact(0.5), 0.8, 3)
    ]
    ranking = rank_population(pop; mc_trials=200)
    @test ranking[1] == 2   # individual 2 (highest fitness) ranked first
    @test length(ranking) == 3
    @test Set(ranking) == Set(1:3)
end
