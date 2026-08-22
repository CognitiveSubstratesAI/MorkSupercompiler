"""
TrieDAGGeometry — trie geometry runtime and canonical DAG evolutionary loop.

Implements §10.2–10.3 of the MG Framework spec (Goertzel, April 2026):
  §10.2   DAG geometry: Prog(Σ,T), hash-consed canonical DAGs
  §10.2.3 Algorithm 3 — Canonical DAG evolutionary loop (8 steps)
  §10.3   Trie geometry: pattern mining and WILLIAM-style codec-search
  §10.3.2 Trie miner (define-trie-miner DSL)
  §10.3.3 WILLIAM / codec-search (define-codec-search DSL)
  §15.6   Three PathMap trie stages: seed → growth → scoring (MVP deliverable 6)

DAG geometry is for: "population of candidate programs or structured artifacts
to canonicalize, score, mutate, recombine" (evolutionary program learning).

Trie geometry is for: "count how often a structural pattern appears and grow
the frequent ones" or "find the heavy compressors inside a stream."
"""

# ── §10.2 DAG geometry ────────────────────────────────────────────────────────

"""
    DAGNode

A node in a hash-consed canonical DAG. Immutable once created.
id       — content hash (structural identity)
head     — node type/constructor name
children — ordered child node IDs
metadata — per-node annotations (fitness, normalization tags, etc.)
"""
struct DAGNode
    id::UInt64
    head::Symbol
    children::Vector{UInt64}   # child node IDs
    metadata::Dict{Symbol, Any}
end

DAGNode(head::Symbol, children::Vector{UInt64}=UInt64[]) =
    DAGNode(hash(string(head, children)), head, children, Dict{Symbol, Any}())

"""
    DAGStore

Hash-consed store of DAGNodes. Structural sharing: identical substructures
share the same UInt64 ID. This is the canonical DAG representation.
"""
mutable struct DAGStore
    nodes::Dict{UInt64, DAGNode}
    root_ids::Vector{UInt64}    # current deme roots
end
DAGStore() = DAGStore(Dict{UInt64, DAGNode}(), UInt64[])

"""
    dag_intern!(store, node) -> UInt64

Add a DAGNode to the store (or return existing ID for identical structure).
This is hash-consing: same structure → same ID.
"""
function dag_intern!(store::DAGStore, node::DAGNode)::UInt64
    haskey(store.nodes, node.id) || (store.nodes[node.id] = node)
    node.id
end

dag_intern!(store::DAGStore, head::Symbol, children::Vector{UInt64}=UInt64[]) =
    dag_intern!(store, DAGNode(head, children))

"""
    dag_normalize!(store, id) -> UInt64

Normalize a DAG node to ENF (Existential Normal Form) per §10.2.3 and §3.2.

ENF rules applied iteratively until fixed point:

 1. Canonical child ordering — sort children IDs for commutative operators
    (:and, :or, :conj) so structurally equivalent programs share the same ID.
 2. Flatten nested same-head operators — (:and (:and a b) c) → (:and a b c).
 3. Deduplicate children — remove duplicate child IDs under commutative ops.
 4. Re-intern after rewriting — hash-consing ensures structural sharing.

Returns the normalized (possibly new) node ID.
"""
function dag_normalize!(store::DAGStore, id::UInt64)::UInt64
    haskey(store.nodes, id) || return id
    node = store.nodes[id]
    isempty(node.children) && return id   # leaf: already in ENF

    # Recursively normalize children first (bottom-up)
    norm_children = map(c -> dag_normalize!(store, c), node.children)

    # Commutative operators where child order doesn't matter
    commutative = node.head ∈ (:and, :or, :conj, :disj, :union, :intersect)

    # Rule 2: flatten nested same-head (associativity)
    flat_children = UInt64[]
    for c in norm_children
        if haskey(store.nodes, c) && store.nodes[c].head == node.head && commutative
            append!(flat_children, store.nodes[c].children)
        else
            push!(flat_children, c)
        end
    end

    # Rule 3: deduplicate under commutative ops
    if commutative
        unique!(flat_children)
        sort!(flat_children)   # Rule 1: canonical ordering
    end

    # Rule 4: re-intern with normalised children
    norm_children == node.children && flat_children == node.children && return id
    dag_intern!(store, node.head, flat_children)
end

"""
    Deme

A population of candidate DAG programs (one deme in the MOSES sense).
"""
mutable struct Deme
    id::Int
    store::DAGStore
    fitnesses::Dict{UInt64, Float64}    # id → fitness score
    eda_model::Dict{Symbol, Float64}    # estimated distribution of operators
    generation::Int
end

Deme(id::Int) = Deme(id, DAGStore(), Dict{UInt64, Float64}(), Dict{Symbol, Float64}(), 0)

# ── Algorithm 3 — Canonical DAG evolutionary loop (§10.2.3) ──────────────────

"""
    DemeEvolutionResult

Result of one round of Algorithm 3.
updated_demes   — demes after mutation, scoring, EDA update
exemplars       — top-k programs to potentially migrate to other demes
shared_stats    :: Dict — subgraph statistics shared across demes
"""
struct DemeEvolutionResult
    updated_demes::Vector{Deme}
    exemplars::Vector{UInt64}    # root IDs of best programs
    shared_stats::Dict{Symbol, Int}  # subgraph frequency counts
end

"""
    evolve_demes!(demes, fitness_fn; top_k, migration_policy) -> DemeEvolutionResult

Algorithm 3 (Canonical DAG evolutionary loop) from §10.2.3.
8 steps:

 1. For all demes in parallel:
 2. Sample/mutate candidate DAG programs
 3. Normalize by ENF
 4. Evaluate fitness, record shared subgraph statistics
 5. Update local EDA model (optionally coerce to factor geometry)
 6. Rebuild candidate pool, migrate exemplars if policy allows
 7. End parallel
 8. Return updated demes and optional new sketches
"""
function evolve_demes!(
    demes::Vector{Deme},
    fitness_fn::Function;
    top_k::Int=5,
    migration_frac::Float64=0.1,
    max_candidates::Int=20
)::DemeEvolutionResult
    shared_stats = Dict{Symbol, Int}()

    # Steps 1–6: process each deme (in practice: in parallel)
    for deme in demes
        # Step 2: sample/mutate candidates
        candidates = _sample_candidates(deme, max_candidates)

        # Step 3: normalize each candidate
        normalized = [dag_normalize!(deme.store, id) for id in candidates]

        # Step 4: evaluate fitness + record subgraph statistics
        for id in normalized
            score = fitness_fn(deme.store, id)
            deme.fitnesses[id] = score
            _update_subgraph_stats!(shared_stats, deme.store, id)
        end

        # Step 5: update EDA model from top-scoring programs
        _update_eda_model!(deme, top_k)

        deme.generation += 1
    end

    # Step 6: collect exemplars (top-k across all demes)
    all_scored = [(id, score) for deme in demes for (id, score) in deme.fitnesses]
    sort!(all_scored; by=x -> -x[2])
    exemplars = [id for (id, _) in all_scored[1:min(top_k, length(all_scored))]]

    # Migration: inject top exemplars into other demes
    if migration_frac > 0
        _migrate_exemplars!(demes, exemplars, migration_frac)
    end

    DemeEvolutionResult(demes, exemplars, shared_stats)
end

function _sample_candidates(deme::Deme, n::Int)::Vector{UInt64}
    existing = collect(keys(deme.store.nodes))
    isempty(existing) && return UInt64[dag_intern!(deme.store, :leaf)]

    candidates = UInt64[]
    for _ in 1:n
        # Mutation: randomly pick an existing node and build a variant
        base = existing[rand(1:length(existing))]
        node = deme.store.nodes[base]
        # Simple mutation: change head or add/remove a child
        new_head = rand([node.head, Symbol("mut_$(node.head)"), :var])
        push!(candidates, dag_intern!(deme.store, new_head, copy(node.children)))
    end
    candidates
end

function _update_subgraph_stats!(stats::Dict{Symbol, Int}, store::DAGStore, id::UInt64)
    haskey(store.nodes, id) || return nothing
    n = store.nodes[id]
    stats[n.head] = get(stats, n.head, 0) + 1
    for child_id in n.children
        _update_subgraph_stats!(stats, store, child_id)
    end
end

function _update_eda_model!(deme::Deme, top_k::Int)
    sorted = sort(collect(deme.fitnesses); by=x -> -x[2])
    top_ids = [id for (id, _) in sorted[1:min(top_k, length(sorted))]]
    # Count operator frequencies in top programs
    op_counts = Dict{Symbol, Int}()
    for id in top_ids
        haskey(deme.store.nodes, id) || continue
        n = deme.store.nodes[id]
        op_counts[n.head] = get(op_counts, n.head, 0) + 1
    end
    total = max(1, sum(values(op_counts)))
    for (op, count) in op_counts
        deme.eda_model[op] = count / total
    end
end

function _migrate_exemplars!(demes::Vector{Deme}, exemplars::Vector{UInt64}, frac::Float64)
    n_migrate = max(1, round(Int, length(exemplars) * frac))
    migrants = exemplars[1:min(n_migrate, length(exemplars))]
    for deme in demes
        for id in migrants
            # Find the store that has this id
            src = findfirst(d -> haskey(d.store.nodes, id), demes)
            src === nothing && continue
            node = demes[src].store.nodes[id]
            dag_intern!(deme.store, node)
        end
    end
end

# ── §10.3 Trie geometry runtime (MVP §15.6) ───────────────────────────────────

"""
    TrieEntry

One entry in the prefix trie for pattern mining.
pattern   — the structural pattern (as a vector of SNode path items)
count     — how many times this pattern appears in the dataset
weight    — importance weight for ranking
children  :: Dict — sub-patterns indexed by next path step
"""
mutable struct TrieEntry
    pattern::Vector{Symbol}
    count::Int
    weight::Float64
    children::Dict{Symbol, TrieEntry}
end

TrieEntry(pattern::Vector{Symbol}) = TrieEntry(pattern, 0, 0.0, Dict{Symbol, TrieEntry}())

"""
    PatternTrie

The trie geometry runtime for §10.3.
Supports the three PathMap stages from §15.6:
Stage 1 — seed extraction by subtree scan
Stage 2 — growth by prefix proximity
Stage 3 — scoring via in-place prefix counters
"""
mutable struct PatternTrie
    root::TrieEntry
    top_k::Vector{Tuple{Vector{Symbol}, Float64}}  # (pattern, weight) top-k
    k::Int
    template::GeometryTemplate
end

PatternTrie(template::GeometryTemplate; k::Int=10) =
    PatternTrie(TrieEntry(Symbol[]), Tuple{Vector{Symbol}, Float64}[], k, template)

"""
    trie_seed!(trie, data_atoms) -> Int

§15.6 Stage 1 — Seed extraction by subtree scan.
Scans `data_atoms` (as SNode patterns), inserts all length-1 patterns as seeds.
Returns the number of seeds inserted.
"""
function trie_seed!(trie::PatternTrie, data_atoms::Vector{SNode})::Int
    n_seeds = 0
    for atom in data_atoms
        atom isa SList || continue
        for item in (atom::SList).items
            item isa SAtom || continue
            sym = Symbol((item::SAtom).name)
            child = get!(trie.root.children, sym, TrieEntry([sym]))
            child.count += 1
            n_seeds += 1
        end
    end
    _rebuild_topk!(trie)
    n_seeds
end

"""
    trie_grow!(trie, data_atoms; max_depth=3) -> Int

§15.6 Stage 2 — Growth by prefix proximity.
Extends existing patterns by one step using prefix-proximity on `data_atoms`.
Returns number of new extended patterns.
"""
function trie_grow!(trie::PatternTrie, data_atoms::Vector{SNode}; max_depth::Int=3)::Int
    n_new = 0
    # For each existing leaf in top_k, try extending with one more symbol
    for (pattern, _) in trie.top_k
        length(pattern) >= max_depth && continue
        for atom in data_atoms
            atom isa SList || continue
            items = (atom::SList).items
            length(items) < length(pattern) + 1 && continue
            # Check if atom starts with pattern
            matches = all(
                k -> items[k] isa SAtom && Symbol((items[k]::SAtom).name) == pattern[k],
                eachindex(pattern)
            )
            matches || continue
            next_item = items[length(pattern) + 1]
            next_item isa SAtom || continue
            next_sym = Symbol((next_item::SAtom).name)
            new_pattern = [pattern; next_sym]
            # Insert extended pattern
            entry = trie.root
            for sym in new_pattern
                entry = get!(entry.children, sym, TrieEntry(new_pattern))
            end
            entry.count += 1
            n_new += 1
        end
    end
    _rebuild_topk!(trie)
    n_new
end

"""
    trie_score!(trie) -> Vector{Tuple{Vector{Symbol},Float64}}

§15.6 Stage 3 — Scoring via in-place prefix counters.
Computes TF-IDF-like weights for each pattern and returns top-k sorted by weight.
"""
# ── MDL scoring (WILLIAM-conformant) ─────────────────────────────────────────────────────────────
#
# 🔴 WHY THIS REPLACED A FREQUENCY WEIGHT (2026-08-05). `trie_score!` ranked by
# `count * log(1 + total/count)` — a TF-IDF-shaped SUPPORT score. All three normative sources rank
# by DESCRIPTION LENGTH, and support-primary ranking is licensed by none of them:
#
#   * Franz IC theory (Franz/Antonenko/Soletskyi 2020, Info.Sci. 547:28-48) — shortest-feature-first,
#     primary key l(f), tie-break l(f'), admitted only under the STRICT compression condition
#     l(f) + l(r) < l(x) (Def. 2.1 Eq. 7). It defines NO notion of support anywhere; both its
#     algorithms enumerate candidates SORTED BY ASCENDING TOTAL LENGTH.
#   * MORK-WILLIAM clarifications (Goertzel, Sept 2025) — gain(r,S) = L(S) - L(S') - C(r), keeping a
#     branch while cumulative gain stays > 0.
#   * AdaptiMORK v8 §4.5 (NORMATIVE) — "we use the same ΔL for: promotion decisions, precedence
#     ordering in overlap resolution, rule retirement decisions"; its candidate order is
#     (larger ΔL, longer span, smaller start, lower type id) with NO frequency term. Frequency is
#     licensed ONLY as a candidate-generation GATE (`occs >= 3`), applied BEFORE ΔL is computed.
#
# WHICH SOURCE IS AUTHORITATIVE FOR WHAT — per CLAIM, not a ranking of documents. These are internal
# design notes by the architect, not academic submissions; counting citations or theorems would score
# their GENRE, not their correctness, and would rate the most operationally specific document lowest.
#   * the ACCEPTANCE CONDITION — Franz IC theory, ℓ(f)+ℓ(r) < ℓ(x) (Def. 2.1 Eq. 7). The formal
#     foundation every WILLIAM paper builds on; that is the gate implemented below.
#   * the ΔL ARITHMETIC — AdaptiMORK §4.5. The most SPECIFIC statement of the objective in the
#     corpus, and the only one that spells out promotion/precedence/retirement sharing one ΔL. That
#     specificity is why it is anchored to here.
#   * Theorem 1 (heavy-first IC bound) — MORK-WILLIAM §4.1, which carries the 4-part proof. The
#     clarifications note restates it near-verbatim, unattributed and unproved; cite the source.
# The three AGREE — all are description-length objectives — so nothing here rests on adjudicating
# between them.
#
# ⚠️ The one distinction that IS load-bearing is the author's own: AdaptiMORK prefaces its
# performance figures with "Based on the design and theoretical analysis, we expect:" — PROJECTIONS,
# labelled as such by the paper. Do not cite them as measurements. Separately, its §9.1
# WeightedTriemap extensions assume a substrate that does not exist: MORK-WILLIAM's own §11 lists the
# base fields as "not yet added to MORK". That is a build-order fact, not a quality judgement.
#
# So frequency keeps its licensed role — `trie_seed!`/`trie_grow!` remain the GENERATOR — and MDL
# gain becomes the acceptance/ranking key. That is exactly AdaptiMORK's §4 architecture.
#
# THE GAIN, for this miner's pattern language. A pattern is a symbol sequence of length `p`
# occurring `n` times. Promoting it to a dictionary rule replaces each occurrence with a single
# reference symbol and pays once for the definition:
#
#     before      n * p                     tokens
#     after       n * 1  +  (p + 1)         n references, plus the rule body and its name
#     ΔL(p, n) =  n*p - n - p - 1  =  n*(p-1) - (p+1)
#
# This is `gain(r,S) = L(S) - L(S') - C(r)` with L = token count and C(r) = the dictionary cost.
#
# ✅ AND IT IS AN EXACT SPECIALIZATION OF AdaptiMORK §4.5 (NORMATIVE), term by term:
#
#     ΔL(R) = Σ_{u∈uses} [ ℓ(covered_u) - ℓ(SYM(R)) - ℓ(RESIDUAL_u) ]
#                       - ℓ(DEF_PDR(R)) - λ·compute_cost(R)
#
#     ℓ(covered_u)   = p      tokens this use covers
#     ℓ(SYM(R))      = 1      the reference symbol that replaces them
#     ℓ(RESIDUAL_u)  = 0      this pattern language covers EXACTLY — a symbol-sequence match leaves
#                             no residual (unlike a template application, which can)
#     Σ over n uses  = n(p-1)
#     ℓ(DEF_PDR(R))  = p + 1  the rule body plus its name
#     λ·compute_cost = 0      not modelled (λ = 0)
#     ⇒ ΔL = n(p-1) - (p+1)
#
# ORDERING likewise follows §4.4's total order — (1) larger ΔL, (2) longer span — then lexicographic.
# Its keys (3) smaller start index and (4) lower type id do not apply to this pattern language; the
# lexicographic tie-break serves their purpose, which §4.4 states plainly: confluence, so encoder and
# decoder agree. Here it makes the top-k BOUNDARY deterministic, and that boundary decides what
# reaches Smine.
#
# ⚠️ DELIBERATELY NOT MODELLED, so nobody reads this as full §4 conformance: `λ·compute_cost(R)`;
# a non-zero `ℓ(RESIDUAL_u)` (needs a template language, not symbol sequences); §4.6's adaptive
# staged threshold `b` (tau_hi=5.0, tau_lo=0.5, W_stall=1000); and "coder-accurate codelengths under
# current adaptive counts" — ℓ here is a STATIC token count, not an adaptive-coder codelength. Those
# are the remaining distance between this miner and AdaptiMORK's §4 engine.
#
# ⚠️ TWO CONSEQUENCES THAT ARE THE POINT, not side effects:
#   * a length-1 pattern can NEVER be admitted: ΔL(1,n) = -2 for every n. Replacing one symbol with
#     one reference saves nothing and still costs a rule. Under the old weight, length-1 seeds
#     dominated the top-k purely by being frequent.
#   * the MDL arithmetic independently REPRODUCES AdaptiMORK's `occs >= 3` heuristic: for p = 2,
#     ΔL > 0 first holds at n = 4 (n=3 gives 0, which the STRICT condition rejects). The paper's
#     hand-tuned gate falls out of the objective rather than being asserted alongside it.
#
# ⚠️ THE TWO SOURCE DOCUMENTS DISAGREE ON THE COST CONVENTION, and the spec extraction says so
# outright (`docs/specs/william/William-MORK-QA_spec.md`): "this convention does NOT count
# parentheses as separate tokens — contrast with the clarifications document below."
#     QA doc              S(cumsum E) = 1 + S(E)              parens NOT counted
#     clarifications doc  L0 = 1 + 2 + 29 = 32                parens counted, 1 each
# `source_papers.md` gives the divergence concretely: the SAME `(repeat 1 9)` step scores +3 under one
# convention and +5 under the other — "each internally consistent but must not be mixed".
# We implement the CLARIFICATIONS convention because it is the one with a fully worked example and
# two independently checkable figures. Neither convention yields its third figure (11): QA gives 4,
# clarifications gives 8 — so that sum is wrong under BOTH, not just ours.
#
# 📌 THE NAMED FUTURE ORACLE, when this grows past a token surrogate: `franz_ml_gen_spec.md` supplies
# "the 5 canonical regression tests for any future WILLIAM-on-MORK implementation" — centralization,
# outlier detection, linear regression, linear classification, decision-tree classification, each
# emerging as a special case of compression with NO specialized ML machinery. That is the real
# conformance target; the tests here cover the objective's algebra, not those five behaviours.
#
# ⚠️ AND THE AUTHOR CALLS THE TOKEN MODEL PEDAGOGICAL: "In production systems the score would be a
# true MDL (optimal codes for integers, structure priors, and residuals) or an equivalent weakness
# prior. The simple token model above is only to make the bookkeeping explicit in this ASCII note."
# So this is conformant to the ILLUSTRATION, not to the production objective. Upgrading means optimal
# integer codes + structure priors + a real residual term — and §8.1's quantale weakness is the
# framework these notes illustrate but do not develop (itself flagged in the whitepaper as a research
# hypothesis "with explicit proof and ablation obligations", not settled theory).
#
# COST MODEL, from the clarifications paper's Example A: literal 1, operator 1, each paren 1, first
# use of a template +1 dictionary cost. `_mdl_token_cost` below reproduces two of the paper's three
# stated figures EXACTLY — L0 of `(cumsum <29 ints>)` = 32 ("1 for cumsum, 2 for parens, 29
# integers") and `(repeat 1 9)` = 5 ("1 + 2 + 1 + 1"). ⚠️ Its THIRD figure does not reconcile: it
# states `(repeat (s0 11) 2)` = 11 via "1 + 2 + 1 + 2 + 1 + 1 + 2 + 1", eight terms that double-count
# under its own rules; this model gives 8. Anchored to the two unambiguous figures rather than
# fudged to the third — recorded so nobody "fixes" the cost model to match a bad sum.

"Token cost L(·): an atom/var costs 1; a list costs 2 (its parens) plus its contents."
function _mdl_token_cost(n::SNode)::Int
    n isa SList || return 1
    c = 2
    for it in (n::SList).items
        c += _mdl_token_cost(it)
    end
    c
end

"""
    mdl_rule_gain(pattern_len, count) -> Int

ΔL for promoting a length-`pattern_len` pattern occurring `count` times to a dictionary rule:
`count*(pattern_len - 1) - (pattern_len + 1)`. Positive ⇔ the compression condition holds.
"""
mdl_rule_gain(plen::Int, count::Int)::Int = count * (plen - 1) - (plen + 1)

"""
    trie_score!(trie; compression_condition=true) -> Vector{Tuple{Vector{Symbol}, Float64}}

§15.6 Stage 3 — score and rank. Ranks by MDL gain ΔL (see the note above), NOT by support.

`compression_condition=true` (the default, and what Franz Def. 2.1 Eq. 7 requires) admits only
patterns with ΔL > 0 — so a corpus with no compressible structure correctly yields NOTHING rather
than the least-bad frequent symbols. Pass `false` to rank without the gate (diagnostics only).
"""
function trie_score!(
    trie::PatternTrie; compression_condition::Bool=true
)::Vector{Tuple{Vector{Symbol}, Float64}}
    _score_subtrie!(trie.root, trie.root.count + 1)
    _rebuild_topk!(trie; compression_condition=compression_condition, by=:mdl)
    trie.top_k
end

function _score_subtrie!(entry::TrieEntry, total::Int)
    entry.weight = Float64(mdl_rule_gain(length(entry.pattern), entry.count))
    for child in values(entry.children)
        _score_subtrie!(child, total)
    end
end

# 🔴 TWO DIFFERENT RANKINGS, AND CONFLATING THEM IS A REGRESSION — measured 2026-08-05.
# AdaptiMORK §4 splits the decision in two, and each half needs its OWN key:
#
#   GENERATION  (trie_seed! / trie_grow!) — ordered by SUPPORT. This is the role frequency IS
#       licensed for: "`occs >= 3`" gates which digrams are even scored (§4.3), applied BEFORE ΔL.
#       `top_k` here is the GROWTH FRONTIER: trie_grow! only extends patterns currently in it.
#   ACCEPTANCE  (trie_score!) — ordered by ΔL and gated by the compression condition (§4.4/§4.5).
#
# The first version of this change scored the frontier with ΔL too. Every length-1 seed has
# ΔL = -2, so they all TIED, the tie-break fell to lexicographic, and with a small `k` the frontier
# silently dropped high-frequency seeds: on `(chop tree wood) x2 …` with k=5, `:wood` was cut, so
# `[:chop,:tree,:wood]` — the genuinely compressing pattern, ΔL=4 — was NEVER GROWN and the miner
# returned nothing. The gate was right; the generator had been blinded. Frequency belongs here.
function _rebuild_topk!(
    trie::PatternTrie; compression_condition::Bool=false, by::Symbol=:support
)
    all_entries = Tuple{Vector{Symbol}, Float64}[]
    _collect_entries!(trie.root, all_entries)
    if by === :support
        # GENERATOR frontier: keep the highest-COUNT candidates, longer first on a tie so growth
        # prefers the more explanatory branch. Counts are read from the trie, not from `weight`.
        counts = Dict{Vector{Symbol}, Int}()
        _collect_counts!(trie.root, counts)
        sort!(all_entries; by=x -> (-get(counts, x[1], 0), -length(x[1]), string.(x[1])))
    else
        # Franz Def. 2.1 Eq. 7 — STRICT: l(f)+l(r) < l(x), so ΔL must be > 0, not >= 0.
        compression_condition && filter!(e -> e[2] > 0.0, all_entries)
        # AdaptiMORK §4.4's total order: larger ΔL, then LONGER span, then lexicographic for
        # determinism — the top-k boundary decides what reaches Smine, so it must be stable.
        sort!(all_entries; by=x -> (-x[2], -length(x[1]), string.(x[1])))
    end
    trie.top_k = all_entries[1:min(trie.k, length(all_entries))]
end

function _collect_counts!(entry::TrieEntry, out::Dict{Vector{Symbol}, Int})
    entry.count > 0 && (out[entry.pattern] = entry.count)
    for child in values(entry.children)
        _collect_counts!(child, out)
    end
end

function _collect_entries!(entry::TrieEntry, out::Vector{Tuple{Vector{Symbol}, Float64}})
    entry.count > 0 && push!(out, (entry.pattern, entry.weight))
    for child in values(entry.children)
        _collect_entries!(child, out)
    end
end

"""
    run_trie_miner(template, data_atoms; k=10, max_depth=3) -> Vector{Tuple}

Full three-stage trie mining (§15.6 MVP deliverable 6):
seed → grow → score → return top-k patterns
"""
function run_trie_miner(
    template::GeometryTemplate, data_atoms::Vector{SNode}; k::Int=10, max_depth::Int=3
)::Vector{Tuple{Vector{Symbol}, Float64}}
    trie = PatternTrie(template; k=k)
    trie_seed!(trie, data_atoms)
    for _ in 1:(max_depth - 1)
        trie_grow!(trie, data_atoms; max_depth=max_depth)
    end
    trie_score!(trie)
end

export mdl_rule_gain
export DAGNode, DAGStore, dag_intern!, dag_normalize!, Deme
export DemeEvolutionResult, evolve_demes!
export TrieEntry, PatternTrie, trie_seed!, trie_grow!, trie_score!, run_trie_miner
