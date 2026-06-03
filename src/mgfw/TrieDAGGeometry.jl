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
function trie_score!(trie::PatternTrie)::Vector{Tuple{Vector{Symbol}, Float64}}
    _score_subtrie!(trie.root, trie.root.count + 1)
    _rebuild_topk!(trie)
    trie.top_k
end

function _score_subtrie!(entry::TrieEntry, total::Int)
    total = max(1, total)
    entry.weight = entry.count * log(1.0 + total / max(1, entry.count))
    for child in values(entry.children)
        _score_subtrie!(child, total)
    end
end

function _rebuild_topk!(trie::PatternTrie)
    all_entries = Tuple{Vector{Symbol}, Float64}[]
    _collect_entries!(trie.root, all_entries)
    sort!(all_entries; by=x -> -x[2])
    trie.top_k = all_entries[1:min(trie.k, length(all_entries))]
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

export DAGNode, DAGStore, dag_intern!, dag_normalize!, Deme
export DemeEvolutionResult, evolve_demes!
export TrieEntry, PatternTrie, trie_seed!, trie_grow!, trie_score!, run_trie_miner
