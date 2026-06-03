"""
ApproxMOSES — approximate evolutionary program learning with p-box fitness.

Implements §5 of the Approximate Supercompilation spec (Goertzel, Oct 2025):
  §5.1  EvolutionaryPBox struct (individual_id, fitness_pbox, rank_pbox,
        heritability, evaluation_count)
  §5.2  Algorithm 5 — ApproximateFitness (already in EvoSpecializer.jl, extended here)
  §5.3  Algorithm 6 — TournamentWithPBox (Monte Carlo tournament selection)
  §5.4  Heritability crossover formula from quantitative genetics
  §5.5  Algorithm 7 — AllocateEvaluations (value of information, in EvoSpecializer.jl)
  §5.6  Convergence detection: overlap(Fᵢ, Fⱼ) > 0.5 threshold

Note: EvolutionaryPBox struct and Algorithms 5+7 exist in EvoSpecializer.jl.
This file adds the missing pieces: Algorithm 6 (TournamentWithPBox) + §5.4
heritability + §5.6 convergence. No duplication — each file owns its algorithms.
"""

using Random: randperm   # tournament_with_pbox subsamples candidates when size < count

const MONTE_CARLO_TRIALS = 100   # §5.3: MC trials for tournament selection
const CONVERGENCE_THETA = 0.5   # §5.6: overlap threshold → converged

# ── §5.3 Algorithm 6 — TournamentWithPBox ────────────────────────────────────

"""
    tournament_with_pbox(candidates::Vector{EvolutionaryPBox},
                         size::Int) -> EvolutionaryPBox

Algorithm 6 (TournamentWithPBox) from §5.3.

Monte Carlo tournament selection with overlapping p-boxes.

For each candidate cᵢ in the tournament:

 1. Draw MONTE_CARLO_TRIALS samples of (fᵢ, f_others)
 2. Count how many times fᵢ > max(f_others)
 3. win_probability[i] = count / MONTE_CARLO_TRIALS

Why Monte Carlo? "With overlapping p-boxes, analytical formulas become intractable.
MC gives good estimates with bounded computation." — spec §5.3.
"""
function tournament_with_pbox(
    candidates::Vector{EvolutionaryPBox}, size::Int=length(candidates)
)::EvolutionaryPBox
    isempty(candidates) && error("tournament_with_pbox: empty candidate list")
    length(candidates) == 1 && return candidates[1]

    # Take `size` candidates (randomly if more available)
    tournament = if length(candidates) <= size
        candidates
    else
        candidates[randperm(length(candidates))[1:size]]
    end

    n = length(tournament)
    win_counts = zeros(Int, n)

    for _ in 1:MONTE_CARLO_TRIALS
        scores = [sample_from_pbox(c.fitness_pbox) for c in tournament]
        winner = argmax(scores)
        win_counts[winner] += 1
    end

    win_probs = win_counts ./ MONTE_CARLO_TRIALS

    # Sample winner weighted by win_probability
    r = rand()
    cum = 0.0
    for (i, p) in enumerate(win_probs)
        cum += p
        r <= cum && return tournament[i]
    end
    tournament[end]
end

# ── §5.4 Heritability and crossover ──────────────────────────────────────────

"""
    offspring_fitness_pbox(parent1::EvolutionaryPBox,
                           parent2::EvolutionaryPBox;
                           mutation_strength=0.05) -> PBox

§5.4: Offspring fitness from quantitative genetics heritability model.

F_offspring = h · (F_p1 + F_p2)/2 + (1-h)·N(0,σ²) + M_mutation

where:
h = heritability (average of both parents' heritability values)
(1-h)·N(0,σ²) = environmental noise component
M_mutation = mutation perturbation (modeled as pbox_interval centered at 0)

Returns a PBox representing offspring fitness uncertainty.

High h (≈1): fitness mostly genetic — good parents → good offspring.
Low h (≈0): fitness mostly environmental — parent quality doesn't transfer.
"""
function offspring_fitness_pbox(
    parent1::EvolutionaryPBox, parent2::EvolutionaryPBox; mutation_strength::Float64=0.05
)::PBox
    h = (parent1.heritability + parent2.heritability) / 2.0

    # Genetic component: h · average of parent fitness p-boxes
    avg_fitness = add_pbox(parent1.fitness_pbox, parent2.fitness_pbox)
    # Scale intervals by h/2 (average + heritability weight)
    genetic_ivs = [(h * lo / 2.0, h * hi / 2.0) for (lo, hi) in avg_fitness.intervals]
    genetic_pb = PBox(
        genetic_ivs,
        avg_fitness.probabilities,
        avg_fitness.confidence * h,
        avg_fitness.correlation_sig
    )

    # Environmental noise component: (1-h) · N(0, σ²)
    # Model as symmetric interval around 0
    noise_half = (1.0 - h) * mutation_strength
    noise_pb = pbox_interval(-noise_half, noise_half, 1.0 - h)

    # Mutation perturbation: small symmetric interval
    mutation_pb = pbox_interval(-mutation_strength/2.0, mutation_strength/2.0, 1.0)

    # Combine: genetic + noise + mutation
    combined = add_pbox(add_pbox(genetic_pb, noise_pb), mutation_pb)

    # Clamp to [0, 1] fitness range
    clamped_ivs = [
        (clamp(lo, 0.0, 1.0), clamp(hi, 0.0, 1.0)) for (lo, hi) in combined.intervals
    ]
    PBox(clamped_ivs, combined.probabilities, combined.confidence, combined.correlation_sig)
end

# ── §5.6 Convergence detection ────────────────────────────────────────────────

"""
    population_converged(population::Vector{EvolutionaryPBox};
                         theta=CONVERGENCE_THETA) -> Bool

§5.6: Population convergence criterion.

Converged = |{(i,j): overlap(Fᵢ,Fⱼ) > 0.5}| / |P|² > θ

Why overlap instead of variance? "Overlapping p-boxes mean we can't confidently
distinguish individuals — population has effectively converged even if absolute
values differ." — spec §5.6.
"""
function population_converged(
    population::Vector{EvolutionaryPBox}; theta::Float64=CONVERGENCE_THETA
)::Bool
    n = length(population)
    n <= 1 && return true

    n_overlap = 0
    for i in 1:n, j in (i + 1):n
        ov = overlap(population[i].fitness_pbox, population[j].fitness_pbox)
        ov > 0.5 && (n_overlap += 2)   # count both (i,j) and (j,i)
    end
    n_overlap += n   # diagonal: each individual overlaps itself

    converge_frac = n_overlap / (n * n)
    converge_frac > theta
end

"""
    convergence_report(population::Vector{EvolutionaryPBox}) -> String

Human-readable convergence state for a population.
"""
function convergence_report(population::Vector{EvolutionaryPBox})::String
    n = length(population)
    converged = population_converged(population)

    io = IOBuffer()
    println(io, "Population: $n individuals")
    println(io, "Converged:  $converged")

    if n > 0
        mean_width = sum(width(p.fitness_pbox) for p in population) / n
        best_lo = maximum(p.fitness_pbox.intervals[1][1] for p in population)
        println(io, "Mean p-box width: $(round(mean_width; digits=4))")
        println(io, "Best lower bound: $(round(best_lo; digits=4))")
    end
    String(take!(io))
end

"""
    rank_population(population::Vector{EvolutionaryPBox};
                    mc_trials=50) -> Vector{Int}

Rank population by expected win probability in pairwise Monte Carlo tournaments.
Returns a permutation (1-indexed) sorted by descending win probability.
"""
function rank_population(
    population::Vector{EvolutionaryPBox}; mc_trials::Int=50
)::Vector{Int}
    n = length(population)
    n <= 1 && return collect(1:n)

    win_counts = zeros(Int, n)
    for _ in 1:mc_trials
        scores = [sample_from_pbox(p.fitness_pbox) for p in population]
        winner = argmax(scores)
        win_counts[winner] += 1
    end

    sortperm(win_counts; rev=true)
end

export tournament_with_pbox, MONTE_CARLO_TRIALS, CONVERGENCE_THETA
export offspring_fitness_pbox
export population_converged, convergence_report, rank_population
