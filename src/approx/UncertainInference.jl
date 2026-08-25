"""
UncertainInference — uncertain logic inference with p-box truth values.

Implements §4 of the Approximate Supercompilation spec (Goertzel, Oct 2025):
  §4.1  UncertainFact struct (predicate, arguments, truth_pbox, confidence, derivation)
  §4.2.1 Conjunction (AND) — 3 cases: independent, perfectly correlated, Fréchet
  §4.2.2 Algorithm 3 — MatchWithUncertainty (structural similarity with quadratic decay)
  §4.3   Algorithm 4 — ApplyRule (UncertainModusPonens with depth widening)
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

"""
    UncertainFact(pred, args, pb) -> UncertainFact

⚠️⚠️ **THIS LINE CARRIES A MASS INTO A META-UNCERTAINTY SLOT, AND THE PAPER IS WHY.**
The spec defines `confidence` TWICE, as DIFFERENT quantities, and never reconciles them:

    §2.2  PBox.confidence          "Total probability mass tracked"
    §4.1  UncertainFact.confidence "Meta-uncertainty: how sure about the p-box"
                                   ("uncertainty about uncertainty")

`pb.confidence` below is the §2.2 quantity; the field it lands in is the §4.1 one. There is no
type error to stop it, because both are `Float64` and the paper gave them the same name. Filed
upstream: `docs/upstream-issues/Approximate-MeTTa-Supercompilation-confidence-two-definitions.md`

🔴🔴 SEPARATE AND CONFIRMED **OUR** BUG: THE MASS RULE OVER-COUNTS, AGAINST AN EXPLICIT CONSTRAINT.
§2.3 states it in as many words — "you can't have more probability mass in the joint distribution
than in either marginal, **and the total mass must sum correctly**". `mul_pbox`'s dependent branch
uses `min(px,py) >= px*py` while constructors set `confidence = sum(new_probs)`. MEASURED on two
multi-interval p-boxes sharing a correlation bit: **dependent sum(probs) = 1.6, independent 1.0**.
Single-interval p-boxes hide it entirely (`min(1,1) == 1*1`), which is why it survived.

⚠️ THIS WAS BRIEFLY MIS-FILED AS AN UPSTREAM DEFECT. Our spec extraction had DROPPED the
"total mass must sum correctly" clause, so §2.3 read as silent and a report was drafted claiming
the paper violates its own field definition. Withdrawn. The paper states the constraint; the
Fréchet result it gives bounds the joint **CDF** and never prescribes an interval-PROBABILITY
rule. `min(px,py)` is OURS, and it assigns every pair its pointwise upper bound — which is
attainable pointwise but NOT SIMULTANEOUSLY, so the collection is not a distribution.

FIX DIRECTION (non-trivial, deliberately not attempted here):
  · RENORMALISE — divides the probabilities down, NARROWING intervals' mass. Unsafe. Out.
  · CLAMP THE SCALAR only — `confidence` then contradicts `sum(probabilities)` in the same struct.
    Two fields silently disagreeing is worse than one being unusual. Out.
  · WHAT IS ACTUALLY REQUIRED: a mass allocation that respects the Fréchet bounds per pair AND
    sums to 1. Taking every upper bound is not such an allocation. This is a real piece of work,
    not a patch, and it needs an oracle — §2.3 gives the CDF inequality but no algorithm.

Reachable only since `correlation_sig` was seeded on 2026-08-25; before that nothing was ever
dependent, so every combination took the product rule and the sums were correct by construction.
"""
function UncertainFact(pred::Symbol, args::Vector{String}, pb::PBox)::UncertainFact
    UncertainFact(pred, args, pb, pb.confidence, ProofTree())
end

"""
    CORRELATION_SIG_WIDTH

Width of the `correlation_sig` BitVector seeded at base facts. 64 bits, hashed rather than one bit
per fact, so the signature stays fixed-size as the fact base grows.

🔴 A COLLISION IS SAFE BY CONSTRUCTION. Two unrelated facts hashing to the same bit are treated as
DEPENDENT, which routes their combination through Fréchet bounds instead of the product rule —
a WIDER interval. Over-reporting uncertainty is the safe direction; the unsafe direction is
claiming independence that does not hold, which is what an unseeded signature does everywhere.

⚠️⚠️ **64 IS A STARTING POINT, NOT A TUNED VALUE, AND IT IS TOO NARROW FOR A REAL KB.** Birthday
collisions at this width, COMPUTED not estimated:

      5 distinct facts -> P(>=1 collision)  14.8%
     10                                     52.3%
     20                                     96.4%
     50                                    100.0%

So beyond a few dozen facts essentially every pair collides, everything is FALSELY DEPENDENT, and
the mechanism degenerates to "always Fréchet" — safe, but barely more informative than no
dependence tracking, just uniformly wider. SAFE AND USEFUL DIVERGE FAST HERE.

Two ways out, and choosing between them needs a WORKLOAD, not an opinion:
  (a) widen the signature so collisions are rare at KB scale (cheap, still approximate);
  (b) seed from actual PROVENANCE — the ancestor set — rather than a hash of the fact's identity.
      That is arguably what §2.2 intends and it eliminates false dependence outright, at the cost
      of signature growth proportional to the fact base.
Do not tune this number without measuring which regime the workload lives in. It is deliberately
recorded as unmeasured so it does not quietly become a constant nobody questions.
"""
const CORRELATION_SIG_WIDTH = 64

"""
    _seed_correlation_sig(pred, args) -> BitVector

One set bit, chosen by hashing the fact's identity. Distinct facts almost always get distinct bits
(independent); the SAME fact reaching a conclusion by two routes gets the same bit, and
`apply_rule`'s union propagation then makes those conclusions correctly dependent.
"""
function _seed_correlation_sig(pred::Symbol, args::Vector{String})::BitVector
    sig = falses(CORRELATION_SIG_WIDTH)
    sig[mod(hash((pred, args)), CORRELATION_SIG_WIDTH) + 1] = true
    sig
end

"""
Create a ground truth UncertainFact (exact truth value 1.0).

🔴 SEEDS `correlation_sig` — WITHOUT THIS THE WHOLE DEPENDENCE MECHANISM IS INERT. Until
2026-08-25 this used `pbox_exact(1.0)`, whose signature is `BitVector()`. Every PBox constructor
defaults to an empty signature, so `are_dependent` short-circuited on
`isempty(a.correlation_sig) && return false` before testing a single bit, `_union_sig` unioned
empties forever, and EVERY addition took the independent path.

§2.2 calls the correlation field essential precisely to avoid "the overconfidence that plagues
naive probabilistic approaches" — and naive independence was exactly what production computed.
The propagation was correct all along (§4.3 step 4, `conclusion.sig ← premise ∨ rule_strength`);
only the SEED was missing. Same shape as MORK's `ground_skip`: the field was declared, threaded
and documented, and nothing ever put a value in it.
"""
function certain_fact(pred::Symbol, args::Vector{String})::UncertainFact
    pb = pbox_interval(1.0, 1.0, 1.0; sig=_seed_correlation_sig(pred, args))
    UncertainFact(pred, args, pb, 1.0, ProofTree())
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
            # ⚠️ NOT "Fréchet probability" — this is the PERFECT-CORRELATION (Łukasiewicz) branch.
            # That mislabel is how an attribution error propagated on 2026-08-25: a measured
            # widening was reported as "the Fréchet path" when it came from here.
            # 🔴 AND `min(pa,pb)` OVER-COUNTS MASS — see the register at `certain_fact`. INVISIBLE on
            # single-interval inputs (`min(1,1) == 1*1`), which is the only shape production builds
            # today; measured sum(probs) = 1.8 on multi-interval pairs.
            push!(new_probs, min(pa, pb))
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
        ai = (a::SList).items
        bi = (b::SList).items
        isempty(ai) && isempty(bi) && return 1.0
        (isempty(ai) || isempty(bi)) && return 0.0
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

# ── §4.3 Algorithm 4 — ApplyRule (UncertainModusPonens) ──────────────────────

const DEPTH_FACTOR_PER_STEP = 0.1   # §4.3: linear growth 0.1/step

"""
    apply_rule(premise_pbox::PBox, rule_strength_pbox::PBox,
               inference_depth::Int) -> PBox

Algorithm 4 (ApplyRule / UncertainModusPonens) from §4.3.

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
    # 🔴🔴 READ THIS BEFORE SEEDING RULE P-BOXES WITH CORRELATION BITS.
    # Algorithm 4 says `MultiplyPBox(premise, rule_strength)`, and `MultiplyPBox` is NEVER DEFINED
    # in the paper — it appears only as this call site. We use `mul_pbox`, which dispatches TWO
    # ways (independent product / dependent `min`). But §4.2.1 defines conjunction with THREE
    # cases, and `conjunction_and` right here in this file implements all three. If `MultiplyPBox`
    # IS §4.2.1's conjunction — plausible, since modus ponens combines premise truth AND rule
    # strength — then `mul_pbox` is the deviation and `conjunction_and` is the correct callee.
    #
    # MEASURED 2026-08-25, mul_pbox vs conjunction_and on the same inputs:
    #
    #   case                                     mul_pbox   conjunction_and
    #   PRODUCTION TODAY (rule sig EMPTY)          0.28          0.28     <- IDENTICAL
    #   identical sigs, single-interval            0.28          0.30
    #   identical sigs, MULTI-interval             0.384         0.72     (~2x)
    #   overlapping-but-different sigs (partial)   0.384         1.26     (~3.3x)
    #
    # ⚠️ DO NOT "JUST SWAP IT BECAUSE IT IS A NO-OP TODAY." It is a no-op ONLY because rule p-boxes
    # carry empty signatures, so `are_dependent` is false and both take the product branch. Swap it
    # now and the 3.3x divergence lands LATER, at the moment someone seeds rules, and gets
    # attributed to the SEEDING rather than to a change made weeks earlier under "it cannot be
    # observed yet". That is exactly the reasoning that shipped `batch_space_ops` (deleted, 7fd0cd7)
    # and the leapfrog shape gate (deleted, 712514a): a change justified by a measurement taken in
    # the regime where the change is invisible.
    #
    # TWO SAFE ROUTES, and only these: (a) swap AND seed rules in the SAME commit, so the behaviour
    # change is visible and measured where it is introduced; or (b) leave this as-is until the
    # upstream question is answered. (b) is what we chose — swapping on a guess about what
    # `MultiplyPBox` means is worse than recording the open question beside the numbers.
    #
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
    nr = n_iterations * sampling_rate
    # 🔴 BOTH TERMS OF THEOREM A.1 STEP 4, NOT THE COLLAPSED ASYMPTOTIC FORM.
    # W_total = W_sample + W_coverage = O(1/sqrt(nr)) + e^(-nr).
    # This returned ONLY `1/sqrt(nr)` until 2026-08-25 — the asymptotic result, which the proof
    # establishes ONLY inside a regime it states twice and this function never tested:
    #     Step 3: "For nr >= ln(1/delta), the coverage error becomes negligible (< delta)"
    #     Step 4: "For nr > ln(n), the second term is o(1/sqrt(nr))"
    # Outside that regime the dropped `e^(-nr)` is NOT negligible. MEASURED 2026-08-25:
    #
    #     n     r     nr    ln(n)  in regime   returned   true bound   UNDER-REPORT
    #     10   0.1   1.00    2.30      no        1.0000     1.3679        36.8%
    #     10   0.01  0.10    2.30      no        3.1623     4.0671        28.6%
    #     10   0.5   5.00    2.30     yes        0.4472     0.4540         1.5%
    #    100   0.1  10.00    4.61     yes        0.3162     0.3163         0.0%
    #
    # So it was materially wrong BELOW nr ~ ln(n) and fine above — exactly the regime split the
    # proof describes. A "bound" that under-reports by 37% is worse than no bound: it is used to
    # decide whether an approximation is within tolerance.
    #
    # ⚠️ WHY THE FULL FORM RATHER THAN A GUARD. Returning `Inf` outside the regime would be honest
    # but throws away a usable answer and cliff-edges at the boundary. `1/sqrt(nr) + exp(-nr)` is
    # valid EVERYWHERE — it is the Step 4 expression before the asymptotic collapse — and degrades
    # gracefully. Same lesson as A.3, whose headline O(T*log(1/eps)) holds only "for typical values
    # where log N ~ b": implement the FULL form, not the collapsed one, and let the docstring carry
    # the regime note instead of a control-flow branch.
    #
    # Monotonicity in both arguments is preserved (both terms decrease in nr), so the existing
    # §4.4 assertions still hold.
    1.0 / sqrt(nr) + exp(-nr)
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
export apply_rule, DEPTH_FACTOR_PER_STEP, BASE_VARIANCE, CORRELATION_SIG_WIDTH
export convergence_width_bound
export InferenceContext, step_deeper, derive_fact
