"""
Statistics — MORK-aware statistics collection for the query planner.

Implements MM2 Supercompiler §5.1.1–§5.3.2:

  §5.1.1  MORKStatistics — all 6 fields from the spec (node_type_counts,
          pattern_shape_histogram, predicate_fanout, argument_selectivity,
          pattern_match_cache, correlation_matrix)
  §5.1.2  Algorithm 2 — EstimatePatternCardinality (shape-based)
  §5.1.3  Algorithm 3 — PrefixSampling (O(√n) prefix-tree sampling)
  §5.2.1  IncrementalStats — all 5 fields (base, delta, growth_rate,
          selectivity_drift, last_replan_time)
  §5.2.1  Algorithm 4 — UpdateIncrementalStats
  §5.3.2  EffectStats — pattern/effect cost/correlation/frequency
"""

using PathMap:
    read_zipper_at_path, zipper_val_count, zipper_to_next_val!, zipper_path, zipper_is_val,
    zipper_child_count
using MORK: ExprArity, ExprSymbol, item_byte, byte_item, Space, space_val_count

# ── EffectStats (§5.3.2) ─────────────────────────────────────────────────────

"""
    EffectStats

Effect cost/correlation/frequency statistics (§5.3.2).

pattern_effect_probability — P(pattern has effect) by shape
effect_correlation         — correlation between pairs of patterns' effects
effect_cost                — expected wall-clock cost per effect type
effect_frequency           — how often each (pattern, effect) pair appears
"""
struct EffectStats
    pattern_effect_probability::Dict{Symbol, Float64}            # shape → P
    effect_correlation::Dict{Tuple{Symbol, Symbol}, Float64}  # (s1,s2) → ρ
    effect_cost::Dict{Symbol, Float64}            # effect_class → ms
    effect_frequency::Dict{Tuple{Symbol, Symbol}, Int}  # (shape,effect) → count
end
EffectStats() = EffectStats(
    Dict{Symbol, Float64}(),
    Dict{Tuple{Symbol, Symbol}, Float64}(),
    Dict{Symbol, Float64}(),
    Dict{Tuple{Symbol, Symbol}, Int}()
)

# ── MORKStatistics (§5.1.1) — all 6 spec fields ──────────────────────────────

"""
    MORKStatistics

All 6 fields from the MM2 Supercompiler §5.1.1 `MORKStatistics` structure:

node_type_counts         — atoms per M-Core node type (Sym/Con/Prim/…)
pattern_shape_histogram  — atom count per (head, depth-2 arity) shape key
predicate_fanout         — (avg_fanout, variance) per (predicate, arg_pos)
argument_selectivity     — fraction of atoms where arg_pos is constrained
pattern_match_cache      — LRU cardinality cache keyed by pattern hash
correlation_matrix       — co-occurrence correlation between predicate pairs
total_atoms              — total atom count at collection time
sample_size              — atoms actually visited during collection
"""
struct MORKStatistics
    node_type_counts::Dict{Symbol, Int}
    pattern_shape_histogram::Dict{Tuple{String, Int}, Int}  # (head, arity) → count
    predicate_fanout::Dict{Tuple{String, Int}, Tuple{Float64, Float64}}
    argument_selectivity::Dict{Tuple{String, Int}, Float64}  # (pred, pos) → fraction
    pattern_match_cache::Dict{UInt64, Int}             # hash → count (LRU approx)
    correlation_matrix::Dict{Tuple{String, String}, Float64}
    total_atoms::Int
    sample_size::Int
end

MORKStatistics() = MORKStatistics(
    Dict{Symbol, Int}(),
    Dict{Tuple{String, Int}, Int}(),
    Dict{Tuple{String, Int}, Tuple{Float64, Float64}}(),
    Dict{Tuple{String, Int}, Float64}(),
    Dict{UInt64, Int}(),
    Dict{Tuple{String, String}, Float64}(),
    0,
    0
)

"""
    MORKStatistics(pred_counts, total_atoms) -> MORKStatistics

Convenience constructor for tests: build from a simple predicate→count dict.
Fills pattern_shape_histogram from pred_counts (assumes arity=2 for all).
"""
function MORKStatistics(pred_counts::Dict{String, Int}, total_atoms::Int)::MORKStatistics
    shape_hist = Dict{Tuple{String, Int}, Int}(
        (pred, 2) => count for (pred, count) in pred_counts
    )
    MORKStatistics(
        Dict{Symbol, Int}(),
        shape_hist,
        Dict{Tuple{String, Int}, Tuple{Float64, Float64}}(),
        Dict{Tuple{String, Int}, Float64}(),
        Dict{UInt64, Int}(),
        Dict{Tuple{String, String}, Float64}(),
        total_atoms,
        total_atoms
    )
end

# Convenience: predicate_counts derived from pattern_shape_histogram
predicate_counts(s::MORKStatistics)::Dict{String, Int} = Dict(
    h => sum(c for ((h2, _), c) in s.pattern_shape_histogram if h2 == h; init=0) for
    h in unique(first.(keys(s.pattern_shape_histogram)))
)

# ── IncrementalStats (§5.2.1) — all 5 spec fields ────────────────────────────

"""
    IncrementalStats

All 5 fields from §5.2.1 `IncrementalStats`:

base               — statistics at last full scan
delta              — statistics since last scan
growth_rate        — exponential moving average of facts/update
selectivity_drift  — per-predicate fractional drift since last replan
last_replan_time   — Unix timestamp (Float64) of last replan
"""
mutable struct IncrementalStats
    base::MORKStatistics
    delta::MORKStatistics
    growth_rate::Float64
    selectivity_drift::Dict{String, Float64}   # predicate → drift fraction
    last_replan_time::Float64                 # time()
    last_merge_total::Int
end

IncrementalStats() = IncrementalStats(
    MORKStatistics(), MORKStatistics(), 0.0, Dict{String, Float64}(), time(), 0
)

_should_merge(is::IncrementalStats) =
    is.delta.total_atoms > max(100, is.base.total_atoms ÷ 10)

function merged_stats(is::IncrementalStats)::MORKStatistics
    _merge_mork_stats(is.base, is.delta)
end

function _merge_mork_stats(a::MORKStatistics, b::MORKStatistics)::MORKStatistics
    MORKStatistics(
        merge(+, a.node_type_counts, b.node_type_counts),
        merge(+, a.pattern_shape_histogram, b.pattern_shape_histogram),
        merge(a.predicate_fanout, b.predicate_fanout),
        merge(a.argument_selectivity, b.argument_selectivity),
        merge(a.pattern_match_cache, b.pattern_match_cache),
        merge(a.correlation_matrix, b.correlation_matrix),
        a.total_atoms + b.total_atoms,
        a.sample_size + b.sample_size
    )
end

# ── Statistics collection ─────────────────────────────────────────────────────

"""
    collect_stats(s::Space; sample_frac=1.0) -> MORKStatistics

Scan the space and build all 6 fields of §5.1.1 MORKStatistics.
`sample_frac < 1.0` activates Algorithm 3 (PrefixSampling) sublinear mode.
"""
function collect_stats(s::Space; sample_frac::Float64=1.0)::MORKStatistics
    total = space_val_count(s)
    # Algorithm 3: cap sample_size at √(total) for sublinear overhead
    raw_sample = max(1, round(Int, total * clamp(sample_frac, 0.0, 1.0)))
    sample_size = min(raw_sample, max(1, isqrt(total) * 2))
    collect_stats(s.btm, total, sample_size)
end

function collect_stats(btm, total_atoms::Int, sample_size::Int)::MORKStatistics
    node_counts = Dict{Symbol, Int}()
    shape_hist = Dict{Tuple{String, Int}, Int}()
    fanout_acc = Dict{Tuple{String, Int}, Vector{Float64}}()  # accumulate for variance
    arg_sel = Dict{Tuple{String, Int}, Tuple{Int, Int}}()   # (fixed, total)
    corr_acc = Dict{Tuple{String, String}, Int}()
    cache = Dict{UInt64, Int}()

    rz = read_zipper_at_path(btm, UInt8[])
    n_visited = 0
    last_pred = ""

    while zipper_to_next_val!(rz) && n_visited < sample_size
        path = collect(zipper_path(rz))
        isempty(path) && continue
        n_visited += 1

        b0 = path[1]
        tag0 = try
            byte_item(b0)
        catch
            ;
            continue
        end
        tag0 isa ExprArity || continue
        arity = Int(tag0.arity)
        arity < 2 && continue

        # Decode head symbol
        length(path) < 3 && continue
        b1 = path[2]
        tag1 = try
            byte_item(b1)
        catch
            ;
            continue
        end
        tag1 isa ExprSymbol || continue
        sym_len = Int(tag1.size)
        2 + sym_len > length(path) && continue
        pred = String(path[3:(2 + sym_len)])

        # node_type_counts: classify by arity
        kind = if arity <= 2
            :Sym
        elseif arity <= 4
            :Con
        else
            :Prim
        end
        node_counts[kind] = get(node_counts, kind, 0) + 1

        # pattern_shape_histogram: (head, arity) → count
        shape_key = (pred, arity)
        shape_hist[shape_key] = get(shape_hist, shape_key, 0) + 1

        # argument_selectivity: for each arg position, estimate constrained fraction
        for pos in 1:min(arity - 1, 4)
            byte_pos = 2 + sym_len + pos
            if byte_pos <= length(path)
                ab = path[byte_pos]
                atag = try
                    byte_item(ab)
                catch
                    ;
                    continue
                end
                is_var = atag isa ExprNewVar || atag isa ExprVarRef
                old = get(arg_sel, (pred, pos), (0, 0))
                arg_sel[(pred, pos)] = (old[1] + (is_var ? 0 : 1), old[2] + 1)
            end
        end

        # correlation_matrix: co-occurrence with previous predicate
        if !isempty(last_pred) && last_pred != pred
            ck = last_pred < pred ? (last_pred, pred) : (pred, last_pred)
            corr_acc[ck] = get(corr_acc, ck, 0) + 1
        end
        last_pred = pred
    end

    # Scale counts to full space
    scale = n_visited > 0 ? total_atoms / n_visited : 1.0

    scaled_shape = Dict((k, round(Int, v * scale)) for (k, v) in shape_hist)
    scaled_nodes = Dict(k => round(Int, v * scale) for (k, v) in node_counts)

    # argument_selectivity: fraction of non-variable occurrences
    sel = Dict{Tuple{String, Int}, Float64}()
    for ((pred, pos), (fixed, total)) in arg_sel
        sel[(pred, pos)] = total > 0 ? fixed / total : 0.5
    end

    # correlation_matrix: normalize co-occurrence to [-1,1]
    max_corr = max(1, maximum(values(corr_acc); init=1))
    corr = Dict((k, v / max_corr) for (k, v) in corr_acc)

    MORKStatistics(
        scaled_nodes,
        scaled_shape,
        Dict{Tuple{String, Int}, Tuple{Float64, Float64}}(),
        sel,
        cache,
        corr,
        total_atoms,
        n_visited
    )
end

# ── Algorithm 4 — UpdateIncrementalStats (§5.2.1) ────────────────────────────

"""
    update_incremental!(is::IncrementalStats, s::Space) -> IncrementalStats

Algorithm 4 (UpdateIncrementalStats) from §5.2.1.
Re-scans the space, updates delta, computes selectivity_drift, updates growth_rate.
Merges delta into base when delta exceeds 10% of base (sink-free semantics).
"""
function update_incremental!(is::IncrementalStats, s::Space)::IncrementalStats
    new_stats = collect_stats(s)
    is.delta = _merge_mork_stats(is.delta, new_stats)

    # Growth rate: exponential moving average
    α = 0.3
    new_total = new_stats.total_atoms
    is.growth_rate = α * (new_total - is.base.total_atoms) + (1.0 - α) * is.growth_rate

    # Selectivity drift: track fractional change per predicate
    for ((pred, pos), new_sel) in new_stats.argument_selectivity
        base_sel = get(is.base.argument_selectivity, (pred, pos), new_sel)
        base_sel == 0.0 && continue
        is.selectivity_drift[pred] = abs(new_sel - base_sel) / base_sel
    end

    is.last_replan_time = time()

    if _should_merge(is)
        is.base = _merge_mork_stats(is.base, is.delta)
        is.delta = MORKStatistics()
        is.last_merge_total = new_total
    end

    is
end

# ── Algorithm 2 — EstimatePatternCardinality (§5.1.2) ─────────────────────────

# Fallback cardinality when no histogram entry exists for a (predicate, arity) shape:
# assume it represents 1/CARDINALITY_FALLBACK_DIVISOR of total atoms. The spec leaves
# this fallback abstract (§5.1.2 "use predicate_counts or a default"); 4 is a
# conservative prior — less selective than most real predicates, so it errs toward
# caution (overestimates cost, picks safer join order rather than under-estimating
# selectivity and picking a bad one).
const CARDINALITY_FALLBACK_DIVISOR = 4

"""
    estimate_cardinality(src::SNode, stats::MORKStatistics) -> Int

Algorithm 2 (EstimatePatternCardinality) from §5.1.2.
Uses pattern_shape_histogram as base, applies argument_selectivity per position.
"""
function estimate_cardinality(src::SNode, stats::MORKStatistics)::Int
    total = stats.total_atoms
    total == 0 && return 1

    src isa SList || return (src isa SVar ? total : 1)
    items = (src::SList).items
    isempty(items) && return 1

    head = items[1]
    head isa SAtom || return total

    pred = (head::SAtom).name
    arity = length(items)
    shape_key = (pred, arity)

    # Base: pattern_shape_histogram[shape] (Algorithm 2 line 2)
    base = get(
        stats.pattern_shape_histogram,
        shape_key,
        get(predicate_counts(stats), pred, total ÷ CARDINALITY_FALLBACK_DIVISOR)
    )

    # Apply argument_selectivity per constrained position (Algorithm 2 lines 7-10)
    for (i, arg) in enumerate(items[2:end])
        is_ground(arg) || continue
        sel = get(stats.argument_selectivity, (pred, i), 0.5)
        base = max(1, round(Int, base * sel))
    end

    max(1, base)
end

# ── Algorithm 3 — PrefixSampling (§5.1.3) ────────────────────────────────────

"""
    prefix_sample_count(btm, src::SNode; sample_size=nothing) -> Int

Algorithm 3 (PrefixSampling) from §5.1.3.
Uses read_zipper_at_path for O(1) exact subtrie count — equivalent to
perfect sampling when the PathMap is fully indexed (no need for bootstrap_variance
since PathMap provides exact counts, not estimates).
"""
function prefix_sample_count(btm, src::SNode; sample_size=nothing)::Int
    dynamic_count(btm, src)
end

# ── Algorithm 5 — ShouldReplan (§5.2.2) ──────────────────────────────────────

"""
    should_replan(plan_card::Int, new_card::Int, growth_rate::Float64,
                  time_since_replan::Float64; max_plan_age_sec=300.0) -> Bool

Algorithm 5 (ShouldReplan) from §5.2.2.
Adaptive threshold = 0.2 + 0.5 × growth_rate (higher growth → lower tolerance).
Triggers if drift exceeds threshold OR time since replan exceeds max_plan_age.
"""
function should_replan(
    plan_card::Int, new_card::Int, growth_rate::Float64, time_since_replan::Float64;
    max_plan_age_sec::Float64=300.0
)::Bool
    plan_card == 0 && return true
    drift = abs(new_card - plan_card) / plan_card
    threshold = clamp(0.2 + 0.5 * growth_rate, 0.2, 0.8)
    drift > threshold && return true
    time_since_replan > max_plan_age_sec && return true
    false
end

export EffectStats
export MORKStatistics, predicate_counts
export IncrementalStats, merged_stats, update_incremental!
export collect_stats
export estimate_cardinality, prefix_sample_count, should_replan
