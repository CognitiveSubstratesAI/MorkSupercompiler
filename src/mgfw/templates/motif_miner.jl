"""
Trie-geometry FactorGraphMotifMiner template — §15.4 MVP demonstration 3.

Implements the spec §10.3.2 / §15.4 demo: a registered trie-geometry template
for the canonical "find heavy motifs under a prefix" miner from §10.3.

Spec §15.4 acceptance criterion: "Trie miner returns same top-k motifs as
naive reference on toy dataset" — the lowering emits a MeTTa program that
walks a PathMap prefix, counts matches, and emits the top-k by support.

Spec §15.2 deliverable 6 (Trie geometry runtime): three PathMap stages
  - seed extraction by subtree scan
  - growth by prefix proximity
  - scoring via in-place prefix counters
plus one factor-graph-as-trie encoding example.
"""

const TEMPLATE_TRIE_MOTIF_MINER = make_template(
    :FactorGraphMotifMiner,
    sem_codec(:EvidenceSet),   # mining outputs are evidence over patterns
    GEOM_TRIE;
    operators = [:seed_scan,         # §10.3.2 stage 1
                 :prefix_grow,       # §10.3.2 stage 2
                 :prefix_counter,    # §10.3.2 stage 3
                 :topk_select],
    effects   = [ReadEffect(DEFAULT_SPACE), AppendEffect(DEFAULT_SPACE)],
    laws      = [:idempotent_merge, :commutative_merge, :evidence_monotone,
                 :counter_associative],
    cache     = CacheContract(
        [:prefix_root, :motif_template, :support_threshold],
        [:new_token_mint, :prefix_extension]),
    coercions = [
        Coercion(:TrieToTensor, GEOM_TRIE, GEOM_TENSOR_SPARSE,
                 sem_rel(:Motif, :Count)),
        Coercion(:TrieToCodec,  GEOM_TRIE, GEOM_TRIE,
                 sem_codec(:Motif)),
    ],
    affinity  = Dict(:mork => :high, :mm2 => :medium, :tensor => :low),
    noether   = :evidence_mass)   # §12.2: conserved evidence-mass across merges

"""
    motif_miner_lowering(t, region) → String

Spec §10.3.2 — emit a MeTTa miner over the MORK PathMap.

The mined motifs are stored under the `(motif ...)` prefix and ranked by
the `(motif-count ...)` counter. The lowering wires the three stages
(seed, grow, score) into MORK exec atoms with priorities so MM2 schedules
them in order (per spec §13.3 PrefixShardPolicy).
"""
function motif_miner_lowering(t::GeometryTemplate, region::AbstractString) :: String
    """
    ;; mgfw:lowering FactorGraphMotifMiner
    ;; §10.3 Trie geometry: 3-stage motif miner (seed → grow → score)
    ;; Stage 1: seed-scan — extract single-symbol candidates from input
    (exec (motif-stage 1)
          (, (\$any))
          (, (motif \$any) (motif-count \$any 1)))
    ;; Stage 2: prefix-grow — extend each motif by a co-occurring symbol
    (exec (motif-stage 2)
          (, (motif \$m) (\$m \$next))
          (, (motif (\$m \$next))))
    ;; Stage 3: prefix-counter — bump the support counter for each derived motif
    (exec (motif-stage 3)
          (, (motif (\$m \$n)) (motif-count (\$m \$n) \$c))
          (, (motif-count (\$m \$n) (+ \$c 1))))
    """
end

export TEMPLATE_TRIE_MOTIF_MINER, motif_miner_lowering
