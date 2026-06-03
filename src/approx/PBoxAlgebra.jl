"""
PBoxAlgebra — full p-box arithmetic for approximate supercompilation.

Implements §2 of the Approximate Supercompilation spec (Goertzel, Oct 2025):
  §2.2  PBox struct (4 fields including correlation_sig)
  §2.3  Algorithm 1 — AddPBox (independent + Fréchet-Hoeffding dependent case)
        MulPBox, WidenPBox, merge_overlapping
  §2.3  Fréchet-Hoeffding bounds: P(X≤x, Y≤y) ∈ [max(Fx+Fy-1,0), min(Fx,Fy)]
  §7.1  Theorem A.2 — Error Composition: width(E_total) ≤ Σ width(Eᵢ) + O(n²·w_max²)
  §7.3  Lemma A.4 — P-Box Width Under Fréchet: w_{X+Y} ≤ wX + wY + 2·min(wX,wY)
  §7.3  Lemma A.5 — Hoeffding Bound: P(|X̄_n - E[X]| > t) ≤ 2·exp(-2nt²/(b-a)²)

The correlation_sig BitVector (§2.2): when two PBoxes share set bits, they are
statistically dependent → Fréchet bounds are used instead of the independence
assumption. This is the key safety invariant: prevents overconfident combination
of correlated evidence.
"""

# ── PBox constructors (full API) ──────────────────────────────────────────────

"""
    pbox_exact(v::Float64) -> PBox

Point-mass p-box at v: confidence=1, single interval [v,v], no correlation.
Use when a value is known exactly — cacheable forever (EXACT error level).
"""
pbox_exact(v::Float64)::PBox = PBox([(v, v)], [1.0], 1.0, BitVector())

"""
    pbox_point(v::Float64) -> PBox

Alias for pbox_exact. Returns PBox.exact(v) per spec §3.2.
"""
pbox_point(v::Float64)::PBox = pbox_exact(v)

"""
    pbox_interval(lo, hi, p=1.0; sig=BitVector()) -> PBox

Single-interval p-box with probability mass p in [lo, hi].
"""
pbox_interval(lo::Float64, hi::Float64, p::Float64=1.0; sig::BitVector=BitVector())::PBox =
    PBox([(lo, hi)], [p], p, sig)

"""
    pbox_empty() -> PBox

Zero-probability p-box. Starting point for AddPBox accumulation.
"""
pbox_empty()::PBox = PBox(Tuple{Float64, Float64}[], Float64[], 0.0, BitVector())

# ── Core properties ───────────────────────────────────────────────────────────

"""
    width(pb::PBox) -> Float64

Expected interval width: Σᵢ pᵢ · (hiᵢ - loᵢ).
Lemma A.4 uses width for Fréchet bound calculations.
"""
function width(pb::PBox)::Float64
    isempty(pb.intervals) && return 0.0
    sum(
        pb.probabilities[i] * (pb.intervals[i][2] - pb.intervals[i][1]) for
        i in eachindex(pb.intervals);
        init=0.0
    )
end

"""
    max_width(pb::PBox) -> Float64

Maximum single-interval width. Used in Theorem A.2 (w_max term).
"""
max_width(pb::PBox)::Float64 =
    isempty(pb.intervals) ? 0.0 : maximum(hi - lo for (lo, hi) in pb.intervals)

"""
    overlap(a::PBox, b::PBox) -> Float64

Fraction of probability mass where intervals [lo_a, hi_a] and [lo_b, hi_b] overlap.
Used in §5.6 convergence detection:
Converged = |{(i,j): overlap(Fᵢ,Fⱼ) > 0.5}| / |P|² > θ
"""
function overlap(a::PBox, b::PBox)::Float64
    total_mass = 0.0
    overlap_mass = 0.0
    for (i, (lo_a, hi_a)) in enumerate(a.intervals)
        pa = a.probabilities[i]
        for (j, (lo_b, hi_b)) in enumerate(b.intervals)
            pb = b.probabilities[j]
            # overlap of two intervals
            lo_ov = max(lo_a, lo_b)
            hi_ov = min(hi_a, hi_b)
            joint = pa * pb
            total_mass += joint
            lo_ov <= hi_ov && (overlap_mass += joint)
        end
    end
    total_mass > 0 ? overlap_mass / total_mass : 0.0
end

"""
    are_dependent(a::PBox, b::PBox) -> Bool

Two p-boxes are statistically dependent iff their correlation_sig BitVectors
share at least one set bit (§2.2). Dependent pairs use Fréchet bounds.
"""
function are_dependent(a::PBox, b::PBox)::Bool
    isempty(a.correlation_sig) || isempty(b.correlation_sig) && return false
    n = min(length(a.correlation_sig), length(b.correlation_sig))
    any(a.correlation_sig[i] & b.correlation_sig[i] for i in 1:n)
end

"""
    mark_dependent!(a::PBox, b::PBox, bit::Int) -> (PBox, PBox)

Set `bit` in both p-boxes' correlation_sig, marking them as dependent.
Returns updated copies (PBox is immutable — returns new instances).
"""
function mark_dependent(a::PBox, b::PBox, bit::Int)::Tuple{PBox, PBox}
    sig_a = copy(a.correlation_sig)
    sig_b = copy(b.correlation_sig)
    # Extend BitVectors if needed
    while length(sig_a) < bit
        ;
        push!(sig_a, false)
    end
    while length(sig_b) < bit
        ;
        push!(sig_b, false)
    end
    sig_a[bit] = true
    sig_b[bit] = true
    (
        PBox(a.intervals, a.probabilities, a.confidence, sig_a),
        PBox(b.intervals, b.probabilities, b.confidence, sig_b)
    )
end

# ── Algorithm 1 — AddPBox (§2.3) ─────────────────────────────────────────────

"""
    add_pbox(X::PBox, Y::PBox) -> PBox

Algorithm 1 (AddPBox) from §2.3. Dispatches on dependency:

  - Independent (disjoint correlation_sig): product rule — p_ij = p_x · p_y
  - Dependent (shared sig bits): Fréchet-Hoeffding bounds

The quadratic blowup in intervals (|X|·|Y| pairs) is intentional per spec:
"necessary to avoid losing precision." merge_overlapping() then collapses nearby
intervals to keep representation compact.
"""
function add_pbox(X::PBox, Y::PBox)::PBox
    are_dependent(X, Y) && return _add_pbox_frechet(X, Y)
    _add_pbox_independent(X, Y)
end

function _add_pbox_independent(X::PBox, Y::PBox)::PBox
    isempty(X.intervals) && return Y
    isempty(Y.intervals) && return X

    new_intervals = Tuple{Float64, Float64}[]
    new_probs = Float64[]

    for (i, (xlo, xhi)) in enumerate(X.intervals)
        px = X.probabilities[i]
        for (j, (ylo, yhi)) in enumerate(Y.intervals)
            py = Y.probabilities[j]
            push!(new_intervals, (xlo + ylo, xhi + yhi))
            push!(new_probs, px * py)
        end
    end

    # Merge correlation_sig (union of both)
    sig = _union_sig(X.correlation_sig, Y.correlation_sig)
    merge_overlapping(PBox(new_intervals, new_probs, sum(new_probs), sig))
end

"""
Fréchet-Hoeffding dependent case (§2.3):
P(X ≤ x, Y ≤ y) ∈ [max(Fx(x) + Fy(y) - 1, 0), min(Fx(x), Fy(y))]

Uses worst-case (upper Fréchet) bound for the interval endpoints,
giving a conservative (wider) result that is always sound.
Lemma A.4: w_{X+Y} ≤ wX + wY + 2·min(wX, wY).
"""
function _add_pbox_frechet(X::PBox, Y::PBox)::PBox
    isempty(X.intervals) && return Y
    isempty(Y.intervals) && return X

    wX = width(X);
    wY = width(Y)
    wmin = min(wX, wY)

    new_intervals = Tuple{Float64, Float64}[]
    new_probs = Float64[]

    for (i, (xlo, xhi)) in enumerate(X.intervals)
        px = X.probabilities[i]
        for (j, (ylo, yhi)) in enumerate(Y.intervals)
            py = Y.probabilities[j]
            # Fréchet: widen each interval by 2·min(wX,wY) factor
            lo = xlo + ylo - wmin
            hi = xhi + yhi + wmin
            # probability: min of marginals (upper Fréchet bound)
            p = min(px, py)
            push!(new_intervals, (lo, hi))
            push!(new_probs, p)
        end
    end

    sig = _union_sig(X.correlation_sig, Y.correlation_sig)
    merge_overlapping(PBox(new_intervals, new_probs, sum(new_probs), sig))
end

# ── MulPBox — multiplication (for UncertainModusPonens §4.2.3) ───────────────

"""
    mul_pbox(X::PBox, Y::PBox) -> PBox

Multiply two p-boxes (used in Algorithm 4: conclusion = premise × rule_strength).
Independent case: p_ij = px·py, interval = [xlo·ylo, xhi·yhi].
Dependent: Fréchet-like conservative widening.
"""
function mul_pbox(X::PBox, Y::PBox)::PBox
    isempty(X.intervals) && return pbox_exact(0.0)
    isempty(Y.intervals) && return pbox_exact(0.0)

    new_intervals = Tuple{Float64, Float64}[]
    new_probs = Float64[]

    use_frechet = are_dependent(X, Y)

    for (i, (xlo, xhi)) in enumerate(X.intervals)
        px = X.probabilities[i]
        for (j, (ylo, yhi)) in enumerate(Y.intervals)
            py = Y.probabilities[j]
            # Interval multiplication: [min, max] of all four products
            products = [xlo*ylo, xlo*yhi, xhi*ylo, xhi*yhi]
            lo = minimum(products)
            hi = maximum(products)
            p = use_frechet ? min(px, py) : px * py
            push!(new_intervals, (lo, hi))
            push!(new_probs, p)
        end
    end

    sig = _union_sig(X.correlation_sig, Y.correlation_sig)
    merge_overlapping(PBox(new_intervals, new_probs, sum(new_probs), sig))
end

# ── WidenPBox — depth-factor widening (Algorithm 4 §4.2.3) ───────────────────

"""
    widen_pbox(pb::PBox, factor::Float64) -> PBox

Widen each interval by `factor` to account for inference-depth uncertainty.
Algorithm 4: depth_factor = 1.0 + 0.1·inference_depth.
Each interval [lo, hi] → [lo/factor, hi·factor] (multiplicative widening).
"""
function widen_pbox(pb::PBox, factor::Float64)::PBox
    factor <= 1.0 && return pb
    new_intervals = [(lo / factor, hi * factor) for (lo, hi) in pb.intervals]
    PBox(new_intervals, pb.probabilities, pb.confidence, pb.correlation_sig)
end

# ── merge_overlapping ─────────────────────────────────────────────────────────

"""
    merge_overlapping(pb::PBox; tol=1e-9) -> PBox

Collapse overlapping/adjacent intervals, summing their probabilities.
Keeps representation compact after AddPBox quadratic blowup.
Intervals [a,b] and [c,d] are merged iff b + tol >= c (sorted by lo).
"""
function merge_overlapping(pb::PBox; tol::Float64=1e-9)::PBox
    isempty(pb.intervals) && return pb
    n = length(pb.intervals)
    n == 1 && return pb

    # Sort by lower bound
    order = sortperm(pb.intervals; by=x->x[1])
    sorted_ivs = pb.intervals[order]
    sorted_ps = pb.probabilities[order]

    merged_ivs = Tuple{Float64, Float64}[]
    merged_ps = Float64[]

    cur_lo, cur_hi = sorted_ivs[1]
    cur_p = sorted_ps[1]

    for k in 2:n
        lo, hi = sorted_ivs[k]
        p = sorted_ps[k]
        if lo <= cur_hi + tol
            # Overlapping → merge
            cur_hi = max(cur_hi, hi)
            cur_p += p
        else
            push!(merged_ivs, (cur_lo, cur_hi))
            push!(merged_ps, cur_p)
            cur_lo, cur_hi, cur_p = lo, hi, p
        end
    end
    push!(merged_ivs, (cur_lo, cur_hi))
    push!(merged_ps, cur_p)

    conf = sum(merged_ps)
    PBox(merged_ivs, merged_ps, conf, pb.correlation_sig)
end

# ── sample_from_pbox (§5.3 Monte Carlo) ──────────────────────────────────────

"""
    sample_from_pbox(pb::PBox) -> Float64

Draw one sample from the p-box distribution.
Used in Algorithm 6 (TournamentWithPBox) Monte Carlo trials.
Step 1: pick interval by probability weight.
Step 2: uniform sample within chosen interval.
"""
function sample_from_pbox(pb::PBox)::Float64
    isempty(pb.intervals) && return 0.0
    total = sum(pb.probabilities)
    total <= 0 && return pb.intervals[1][1]

    r = rand() * total
    cum = 0.0
    for (i, p) in enumerate(pb.probabilities)
        cum += p
        if r <= cum
            lo, hi = pb.intervals[i]
            return lo + rand() * (hi - lo)
        end
    end
    pb.intervals[end][2]
end

# ── Theoretical guarantees (§7) ───────────────────────────────────────────────

"""
    error_composition_bound(widths::Vector{Float64}) -> Float64

Theorem A.2 (Error Composition, §7.1):
width(E_total) ≤ Σᵢ width(Eᵢ) + O(n² · w_max²)

Returns the upper bound on total error width for a sequence of n operations.
Used to verify that approximate pipeline error stays within tolerance.
"""
function error_composition_bound(widths::Vector{Float64})::Float64
    isempty(widths) && return 0.0
    n = length(widths)
    linear_term = sum(widths)
    w_max = maximum(widths)
    quadratic_term = n^2 * w_max^2   # O(n² · w_max²) from Theorem A.2
    linear_term + quadratic_term
end

"""
    frechet_width_bound(wX::Float64, wY::Float64) -> Float64

Lemma A.4 (P-Box Width Under Fréchet Bounds, §7.3):
w_{X+Y} ≤ wX + wY + 2·min(wX, wY)

The 2·min term is the additional uncertainty from unknown correlation structure.
"""
frechet_width_bound(wX::Float64, wY::Float64)::Float64 = wX + wY + 2.0 * min(wX, wY)

"""
    hoeffding_bound(n::Int, t::Float64; a=0.0, b=1.0) -> Float64

Lemma A.5 (Hoeffding Bound for P-Boxes, §7.3):
P(|X̄_n - E[X]| > t) ≤ 2·exp(-2nt²/(b-a)²)

Returns the tail probability bound for `n` independent samples from [a,b].
Used to validate Hoeffding-based cardinality estimates (Algorithm 2).
"""
hoeffding_bound(n::Int, t::Float64; a::Float64=0.0, b::Float64=1.0)::Float64 =
    2.0 * exp(-2.0 * n * t^2 / (b - a)^2)

"""
    hoeffding_epsilon(n::Int, delta::Float64; a=0.0, b=1.0) -> Float64

Inverse: given sample count n and confidence 1-δ, return ε such that
P(|X̄_n - E[X]| > ε) ≤ δ
ε = √(ln(2/δ) / (2n)) · (b-a)
"""
hoeffding_epsilon(n::Int, delta::Float64; a::Float64=0.0, b::Float64=1.0)::Float64 =
    sqrt(log(2.0 / delta) / (2.0 * n)) * (b - a)

# ── Utilities ─────────────────────────────────────────────────────────────────

function _union_sig(a::BitVector, b::BitVector)::BitVector
    isempty(a) && return copy(b)
    isempty(b) && return copy(a)
    n = max(length(a), length(b))
    out = BitVector(undef, n)
    for i in 1:n
        ai = i <= length(a) ? a[i] : false
        bi = i <= length(b) ? b[i] : false
        out[i] = ai | bi
    end
    out
end

export pbox_exact, pbox_point, pbox_interval, pbox_empty
export width, max_width, overlap, are_dependent, mark_dependent
export add_pbox, mul_pbox, widen_pbox, merge_overlapping
export sample_from_pbox
export error_composition_bound, frechet_width_bound
export hoeffding_bound, hoeffding_epsilon
