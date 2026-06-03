"""
UncertainInference — uncertain logic inference with p-box truth values.

Implements §4 of the Approximate Supercompilation spec (Goertzel, Oct 2025):
  §4.1  UncertainFact struct (predicate, arguments, truth_pbox, confidence, derivation)
  §4.2.1 Conjunction (AND) — 3 cases: independent, perfectly correlated, Fréchet
  §4.2.2 Algorithm 3 — MatchWithUncertainty (structural similarity with quadratic decay)
  §4.2.3 Algorithm 4 — ApplyRule (UncertainModusPonens with depth widening)
  §4.4   Convergence theorem: p-box width → O(1/√(nr)) under semi-naive + sampling

The `confidence` field (§4.1) captures meta-uncertainty — "how sure are we about
this p-box?" — which differs from the p-box's own probability mass. Crucial when
combining evidence from sources of varying reliability.

Connection to PLN (PRIMUS Relevance §): UncertainFact.truth_pbox directly extends
PLN's (stv strength confidence) — BOUNDED error level maps to IndefiniteTruthValue.
"""

# ── §4.1 Core structures ──────────────────────────────────────────────────────

"""
    ProofTree

Provenance record for an UncertainFact derivation.
Minimal implementation: stores the derivation chain as a list of
(rule_name, premise_fact_ids) tuples for debugging/explanation.
"""
struct ProofTree
    rule_name::Symbol
    premises::Vector{UInt64}   # hash IDs of premise facts
    depth::Int
end
ProofTree(rule::Symbol) = ProofTree(rule, UInt64[], 0)
ProofTree() = ProofTree(:base, UInt64[], 0)

"""
    UncertainFact

§4.1: An inferred fact with probabilistic truth value.

predicate  — the relation name (e.g. :parent, :edge)
arguments  — the argument terms as strings
truth_pbox — truth value in [0,1] or [-1,1] (p-box)
confidence — meta-uncertainty: how sure about the p-box itself (scalar in [0,1])
derivation — provenance ProofTree for debugging/explanation
"""
struct UncertainFact
    predicate::Symbol
    arguments::Vector{String}
    truth_pbox::PBox
    confidence::Float64
    derivation::ProofTree
end

function UncertainFact(pred::Symbol, args::Vector{String}, pb::PBox)::UncertainFact
    UncertainFact(pred, args, pb, pb.confidence, ProofTree())
end

"""
Create a ground truth UncertainFact (exact truth value 1.0).
"""
function certain_fact(pred::Symbol, args::Vector{String})::UncertainFact
    UncertainFact(pred, args, pbox_exact(1.0), 1.0, ProofTree())
end

# ── §4.2.1 Conjunction (AND) ──────────────────────────────────────────────────

"""
    conjunction_and(T_A::PBox, T_B::PBox) -> PBox

§4.2.1: Conjunction T_{A∧B} with three cases based on correlation_sig:

Independent (disjoint sig):    T_A ⊗ T_B  (product rule)
Perfectly correlated (same):   max(T_A + T_B - 1, 0)  (Łukasiewicz t-norm)
Partially correlated (shared): Fréchet bounds (add_pbox uses Fréchet internally)

The Łukasiewicz t-norm for perfect correlation prevents double-counting:
if A and B use the same evidence, P(A∧B) ≥ P(A) + P(B) - 1, not P(A)·P(B).
"""
function conjunction_and(T_A::PBox, T_B::PBox)::PBox
    # Detect correlation level via shared sig bits
    if are_dependent(T_A, T_B)
        # Check for perfect correlation: identical sig
        if T_A.correlation_sig == T_B.correlation_sig && !isempty(T_A.correlation_sig)
            # Perfectly correlated: Łukasiewicz t-norm
            return _and_lukasiewicz(T_A, T_B)
        else
            # Partially correlated: Fréchet (add_pbox handles this)
            return _and_frechet(T_A, T_B)
        end
    end
    # Independent: product rule
    mul_pbox(T_A, T_B)
end

function _and_lukasiewicz(T_A::PBox, T_B::PBox)::PBox
    # max(T_A + T_B - 1, 0) — element-wise on each interval pair
    new_intervals = Tuple{Float64, Float64}[]
    new_probs = Float64[]
    for (i, (alo, ahi)) in enumerate(T_A.intervals)
        pa = T_A.probabilities[i]
        for (j, (blo, bhi)) in enumerate(T_B.intervals)
            pb = T_B.probabilities[j]
            lo = max(alo + blo - 1.0, 0.0)
            hi = max(ahi + bhi - 1.0, 0.0)
            push!(new_intervals, (lo, hi))
            push!(new_probs, min(pa, pb))   # Fréchet probability
        end
    end
    sig = _union_sig(T_A.correlation_sig, T_B.correlation_sig)
    merge_overlapping(PBox(new_intervals, new_probs, sum(new_probs), sig))
end

function _and_frechet(T_A::PBox, T_B::PBox)::PBox
    # Fréchet conjunction bounds (spec §4.2.1):
    #   max(p_A + p_B - 1, 0) ≤ P(A ∧ B) ≤ min(p_A, p_B)
    # Previous impl delegated to `add_pbox` (additive Fréchet — for SUM not
    # INTERSECTION). add_pbox returns intervals approximating [aL+bL, aU+bU]
    # capped by widening, NOT the conjunction bounds above. Wrong operator,
    # silently passed because tests only checked "width ≥ independent width".
    new_intervals = Tuple{Float64, Float64}[]
    new_probs = Float64[]
    for (i, (alo, ahi)) in enumerate(T_A.intervals)
        pa = T_A.probabilities[i]
        for (j, (blo, bhi)) in enumerate(T_B.intervals)
            pb = T_B.probabilities[j]
            lo = max(alo + blo - 1.0, 0.0)
            hi = min(ahi, bhi)
            push!(new_intervals, (lo, hi))
            push!(new_probs, min(pa, pb))
        end
    end
    sig = _union_sig(T_A.correlation_sig, T_B.correlation_sig)
    merge_overlapping(PBox(new_intervals, new_probs, sum(new_probs), sig))
end

"""
    disjunction_or(T_A::PBox, T_B::PBox) -> PBox

Disjunction T_{A∨B} = T_A + T_B - T_{A∧B} (inclusion-exclusion).
Uses conjunction_and internally for the subtracted term.
"""
function disjunction_or(T_A::PBox, T_B::PBox)::PBox
    # T_{A∨B} intervals: [min(lo_a + lo_b, 1), min(hi_a + hi_b, 1)]
    # simplified: clamp sum to [0,1]
    and_term = conjunction_and(T_A, T_B)
    # or = A + B - A∧B; for p-boxes: add then subtract (via widened bounds)
    ab = add_pbox(T_A, T_B)
    # Inclusion-exclusion bounds on P(A ∨ B):
    #   lower = (aL + bL) − andU = lo_ab − hi_and  (worst case: most subtracted)
    #   upper = (aU + bU) − andL = hi_ab − lo_and  (best case: least subtracted)
    # Previous impl hard-clamped the upper bound to `min(hi_ab, 1.0)` — never
    # subtracted `lo_and`. So `disjunction_or(0.5, 0.5)` returned [0.75, 1.0]
    # instead of the correct [0.75, 0.75]. Untested.
    sub_ivs = [
        (max(lo_ab - hi_and, 0.0), min(max(hi_ab - lo_and, 0.0), 1.0)) for
        ((lo_ab, hi_ab), (lo_and, hi_and)) in
        zip(ab.intervals, and_term.intervals[1:min(end, length(ab.intervals))])
    ]
    isempty(sub_ivs) && return ab
    PBox(
        sub_ivs,
        ab.probabilities[1:length(sub_ivs)],
        ab.confidence,
        _union_sig(T_A.correlation_sig, T_B.correlation_sig)
    )
end

# ── §4.2.2 Algorithm 3 — MatchWithUncertainty ────────────────────────────────

const BASE_VARIANCE = 0.1   # §4.2.2: variance per unit of structural difference
const NO_MATCH = nothing

"""
    structural_similarity(pattern::SNode, fact::SNode) -> Float64

Compute structural similarity in [0,1] between a pattern and a fact.
Uses recursive tree edit distance normalized by tree size.
Exact match → 1.0. Completely different → 0.0.
"""
function structural_similarity(pattern::SNode, fact::SNode)::Float64
    pattern == fact && return 1.0
    _tree_similarity(pattern, fact)
end

function _tree_similarity(a::SNode, b::SNode)::Float64
    typeof(a) != typeof(b) && return 0.0
    if a isa SAtom && b isa SAtom
        return (a::SAtom).name == (b::SAtom).name ? 1.0 : 0.0
    end
    if a isa SVar && b isa SVar
        return (a::SVar).name == (b::SVar).name ? 1.0 : 0.8  # vars: similar even if different name
    end
    if a isa SList && b isa SList
        ai = (a::SList).items;
        bi = (b::SList).items
        isempty(ai) && isempty(bi) && return 1.0
        isempty(ai) || isempty(bi) && return 0.0
        length(ai) != length(bi) && return 0.3  # structural mismatch: low similarity
        child_sims = [_tree_similarity(ai[k], bi[k]) for k in eachindex(ai)]
        return sum(child_sims) / length(child_sims)
    end
    0.0
end

"""
    match_with_uncertainty(pattern::SNode, fact::SNode,
                           tolerance::Float64) -> Union{PBox, Nothing}

Algorithm 3 (MatchWithUncertainty) from §4.2.2.

Exact match        → PBox.exact(1.0)
similarity > 1-tol → PBox([sim-variance, sim+variance], confidence=similarity²)
otherwise          → NO_MATCH (nothing)

Quadratic decay in confidence (similarity²): empirically, match quality degrades
super-linearly with structural differences — a 90% similar fact is only 81% likely
to satisfy a query that expects an exact match.
"""
function match_with_uncertainty(
    pattern::SNode, fact::SNode, tolerance::Float64
)::Union{PBox, Nothing}
    # identity check first (same object → definitely equal); then structural ==
    (pattern === fact || pattern == fact) && return pbox_exact(1.0)

    similarity = structural_similarity(pattern, fact)
    similarity > 1.0 - tolerance || return NO_MATCH

    confidence = similarity^2                      # quadratic decay
    variance = (1.0 - similarity) * BASE_VARIANCE
    lo = clamp(similarity - variance, 0.0, 1.0)
    hi = clamp(similarity + variance, 0.0, 1.0)
    pbox_interval(lo, hi, confidence)
end

# ── §4.2.3 Algorithm 4 — ApplyRule (UncertainModusPonens) ────────────────────

const DEPTH_FACTOR_PER_STEP = 0.1   # §4.2.3: linear growth 0.1/step

"""
    apply_rule(premise_pbox::PBox, rule_strength_pbox::PBox,
               inference_depth::Int) -> PBox

Algorithm 4 (ApplyRule / UncertainModusPonens) from §4.2.3.

Steps:

 1. conclusion = premise ⊗ rule_strength  (mul_pbox handles dep/indep)
 2. Widen by depth_factor = 1.0 + 0.1·depth  (uncertainty grows with depth)
 3. Merge correlation_sig (union) — conclusion depends on all premise dependencies

The widening in step 2 is "linear growth: 0.1/step empirically reasonable" (spec).
Prevents false confidence from deep inference chains.
"""
function apply_rule(
    premise_pbox::PBox, rule_strength_pbox::PBox, inference_depth::Int=0
)::PBox
    # Step 1: multiply
    conclusion = mul_pbox(premise_pbox, rule_strength_pbox)

    # Step 2: widen by depth factor
    depth_factor = 1.0 + DEPTH_FACTOR_PER_STEP * inference_depth
    conclusion = widen_pbox(conclusion, depth_factor)

    # Step 3: merge correlation signatures
    merged_sig = _union_sig(
        premise_pbox.correlation_sig, rule_strength_pbox.correlation_sig
    )
    PBox(conclusion.intervals, conclusion.probabilities, conclusion.confidence, merged_sig)
end

# ── §4.4 Convergence theorem ──────────────────────────────────────────────────

"""
    convergence_width_bound(n_iterations::Int, sampling_rate::Float64) -> Float64

§4.4 Theorem (Inference Convergence):
Under semi-naive evaluation with sampling rate r,
p-box width → O(1/√(n·r)) as iterations n → ∞.

Returns the theoretical upper bound on p-box width at iteration n with rate r.
"""
function convergence_width_bound(n_iterations::Int, sampling_rate::Float64)::Float64
    # Parens fix Julia precedence — `&&` binds tighter than `||`, so the old
    # `cond1 || cond2 && return X` only early-returned for cond2.
    (n_iterations <= 0 || sampling_rate <= 0.0) && return Inf
    1.0 / sqrt(n_iterations * sampling_rate)
end

# ── Inference engine (combining the above) ────────────────────────────────────

"""
    InferenceContext

Tracks the current inference state for applying rules iteratively:
depth     — current inference depth (used for depth_factor widening)
tolerance — similarity tolerance for approximate matching
weights   — cost model weights (for planning decisions within inference)
"""
struct InferenceContext
    depth::Int
    tolerance::Float64
    weights::CostWeights
end
InferenceContext() = InferenceContext(0, 0.05, balanced())
step_deeper(ctx::InferenceContext) =
    InferenceContext(ctx.depth + 1, ctx.tolerance, ctx.weights)

"""
    derive_fact(premise::UncertainFact, rule_strength::PBox,
                conclusion_pred::Symbol, conclusion_args::Vector{String},
                ctx::InferenceContext) -> UncertainFact

Derive a new UncertainFact by applying a rule to a premise.
Uses Algorithm 4 (apply_rule) internally.
"""
function derive_fact(
    premise::UncertainFact, rule_strength::PBox, conc_pred::Symbol,
    conc_args::Vector{String}, ctx::InferenceContext
)::UncertainFact
    conc_pbox = apply_rule(premise.truth_pbox, rule_strength, ctx.depth)
    tree = ProofTree(
        conc_pred, [hash(string(premise.predicate, premise.arguments...))], ctx.depth
    )
    conc_conf = min(premise.confidence, conc_pbox.confidence)
    UncertainFact(conc_pred, conc_args, conc_pbox, conc_conf, tree)
end

export ProofTree, UncertainFact, certain_fact
export conjunction_and, disjunction_or
export structural_similarity, match_with_uncertainty, NO_MATCH
export apply_rule, DEPTH_FACTOR_PER_STEP, BASE_VARIANCE
export convergence_width_bound
export InferenceContext, step_deeper, derive_fact
