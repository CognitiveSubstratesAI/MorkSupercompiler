"""
MGCompiler — geometry-aware compilation and supercompilation pipeline.

Implements §9 and §12 of the MG Framework spec (Goertzel, April 2026):
  §9    Backend affinity and late commitment — 4-stage decision process
  §12.1 Algorithm 5 — Geometry-aware compilation pipeline (9 steps)
  §12.2 Worked example: supercompiling geodesic inference control
        (GeodesicBGC-Worklist, FactorFGSurrogate, EvidenceCapsule, Composite)

The "late commitment" principle (§9): "hard decision (actual lowering) should
happen late" — soft affinity analysis first, then backend-neutral optimization,
then backend selection, then backend-specific polishing.

The supercompiler's TyLAA role (§7.5): given the "blue" layer (geometry
templates with typed contracts), find the **weakest operational semantics**
consistent with those contracts — making fewest scheduling distinctions while
preserving observational equivalence.
"""

using MORK: Space, new_space, space_add_all_sexpr!, space_metta_calculus!

# ── §9 Backend affinity types ─────────────────────────────────────────────────

"""
    BackendAffinity

How strongly a region prefers a given backend (§9).
  HIGH   — strong preference (region uses backend-specific features)
  MEDIUM — moderate preference (region works well on this backend)
  LOW    — weak preference (backend works but not optimal)
  NONE   — backend not applicable
"""
@enum AffinityLevel HIGH MEDIUM LOW NONE

"""
    BackendProfile

Affinity scores for all supported backends (§9).
"""
struct BackendProfile
    mm2         :: AffinityLevel   # MM2 worklist patterns
    mork        :: AffinityLevel   # MORK-native operations
    factor      :: AffinityLevel   # Factor graph runtime
    trie        :: AffinityLevel   # Trie/PathMap runtime
    tensor      :: AffinityLevel   # GPU tensor kernels
    petta       :: AffinityLevel   # PeTTa (Prolog-based)
end

BackendProfile(; mm2=MEDIUM, mork=MEDIUM, factor=LOW, trie=LOW, tensor=NONE, petta=LOW) =
    BackendProfile(mm2, mork, factor, trie, tensor, petta)

"""
    BackendChoice

The result of backend selection (stage 3 of §9).
  primary   — main backend to use
  fallback  — backup if primary is unavailable
  is_hybrid — true if multiple backends needed (mixed execution)
"""
struct BackendChoice
    primary   :: Symbol
    fallback  :: Symbol
    is_hybrid :: Bool
end

# ── §9 Four-stage late commitment ─────────────────────────────────────────────

"""
    affinity_analysis(templates) -> BackendProfile

§9 Stage 1: Soft backend affinity analysis. Inspects geometry templates,
effects, and laws to suggest backend preferences WITHOUT committing.
"""
function affinity_analysis(templates::Vector{GeometryTemplate}) :: BackendProfile
    mm2_score  = 0
    mork_score = 0
    factor_score = 0
    trie_score = 0

    for t in templates
        g = geometry_of(t)
        aff = t.backend_affinity

        # Geometry signals
        g == GEOM_FACTOR       && (factor_score += 2; mm2_score  += 1; mork_score += 1)
        g == GEOM_TRIE         && (trie_score   += 2; mork_score += 2)
        g == GEOM_DAG          && (mm2_score    += 2; mork_score += 1)
        g == GEOM_TENSOR_SPARSE && (mork_score  += 1)
        g == GEOM_TENSOR_DENSE  && nothing   # tensor/GPU low unless batched

        # Template-level hints
        get(aff, :mm2, :medium)  == :high && (mm2_score  += 2)
        get(aff, :mork, :medium) == :high && (mork_score += 2)
        :sink_free in t.laws             && (mm2_score   += 1)
        :monotone  in t.laws             && (mork_score  += 1)
    end

    n = max(1, length(templates))
    _score_to_affinity(x) = x ÷ n >= 3 ? HIGH : x ÷ n >= 2 ? MEDIUM : x ÷ n >= 1 ? LOW : NONE

    BackendProfile(
        mm2    = _score_to_affinity(mm2_score),
        mork   = _score_to_affinity(mork_score),
        factor = _score_to_affinity(factor_score),
        trie   = _score_to_affinity(trie_score))
end

"""
    backend_neutral_optimize(templates, stats) -> Vector{GeometryTemplate}

§9 Stage 2: Backend-aware but backend-neutral IR optimization (ADR-055).
Four passes — first three preserve prior behaviour; Pass 4 is new:

  1. Validity pruning — drop templates that fail is_valid_template.
  2. Cache contract ordering — content_hash > version_tuple > epoch > others.
  3. Semiring-geometry matching (ADR-055, replaces ad-hoc cardinality heuristic):
     infer natural semiring from geometry tag and re-rank accordingly.
     mg_framework_spec §9/§12 maps:
       GEOM_TRIE         → Boolean / MaxPlus  (reachability, best-path)
       GEOM_FACTOR       → SumProduct         (counting, PLN inference)
       GEOM_DAG          → MinPlus / Boolean  (shortest paths, DAG traversal)
       GEOM_TENSOR_SPARSE → MaxPlus           (ECAN STI spreading)
       GEOM_TENSOR_DENSE  → SumProduct        (Tucker dense einsums)
  4. Semiring cost estimation — use semiring_matmul complexity proxy
     (O(nnz) for sparse vs O(n³) for dense) to break ties within same rank.
"""
function backend_neutral_optimize(templates :: Vector{GeometryTemplate},
                                   stats    :: MORKStatistics) :: Vector{GeometryTemplate}
    isempty(templates) && return templates

    # Pass 1: validity pruning
    valid = filter(is_valid_template, templates)
    isempty(valid) && return templates   # guard: don't drop everything

    # Pass 2: cache contract ordering
    _cache_rank(t::GeometryTemplate) :: Int = begin
        keys = t.cache_contract.key
        :content_hash  ∈ keys ? 0 :
        :version_tuple ∈ keys ? 1 :
        :epoch         ∈ keys ? 2 : 3
    end
    sort!(valid; by=_cache_rank)

    # Pass 3: semiring-geometry matching (ADR-055)
    # Maps geometry → natural semiring priority and re-ranks templates.
    # Replaces the prior total_atoms > 1000 heuristic with algebraic grounding.
    #
    # Priority semantics (lower = preferred):
    #   0 = GEOM_TRIE   (Boolean/MaxPlus — prefix-indexed, O(log N) lookup)
    #   1 = GEOM_DAG    (MinPlus/Boolean — DAG shortest-path, O(E) traversal)
    #   2 = GEOM_FACTOR (SumProduct — factor graph, O(N) but exact counting)
    #   3 = GEOM_TENSOR_SPARSE (MaxPlus — ECAN, patch-log shard workflow)
    #   4 = GEOM_TENSOR_DENSE  (SumProduct — Tucker dense einsums)
    #   5 = everything else
    sort!(valid; by = t -> begin
        g = geometry_of(t)
        g == GEOM_TRIE          ? 0 :
        g == GEOM_DAG           ? 1 :
        g == GEOM_FACTOR        ? 2 :
        g == GEOM_TENSOR_SPARSE ? 3 :
        g == GEOM_TENSOR_DENSE  ? 4 : 5
    end)

    # Pass 4: semiring cost estimation within same rank
    # Use a simple nnz proxy: sparser structures (TRIE/DAG) get priority over
    # denser ones (FACTOR) when they share the same semiring rank.
    # Proxy: templates with :content_hash + GEOM_TRIE are cheapest to evaluate.
    if stats.total_atoms > 0
        n = stats.total_atoms
        sort!(valid; by = t -> begin
            g    = geometry_of(t)
            rank = g == GEOM_TRIE ? 0 : g == GEOM_DAG ? 1 :
                   g == GEOM_FACTOR ? 2 : g == GEOM_TENSOR_SPARSE ? 3 :
                   g == GEOM_TENSOR_DENSE ? 4 : 5
            base = if g == GEOM_TRIE;          log2(max(1, n))
                   elseif g == GEOM_DAG;       sqrt(Float64(n))
                   elseif g == GEOM_FACTOR;    Float64(n)
                   elseif g == GEOM_TENSOR_SPARSE; Float64(n) * 0.1
                   elseif g == GEOM_TENSOR_DENSE;  Float64(n)^1.5
                   else Float64(n)^2
                   end
            rank * 1e9 + base
        end)
    end

    valid
end

"""
    select_backend(profile, templates) -> BackendChoice

§9 Stage 3: Backend selection.  Pick the backend with highest affinity.
"""
function select_backend(profile   :: BackendProfile,
                         templates :: Vector{GeometryTemplate}) :: BackendChoice
    # Highest-affinity backend wins
    scores = [(:mm2, profile.mm2), (:mork, profile.mork),
              (:factor, profile.factor), (:trie, profile.trie)]
    # HIGH=0 < MEDIUM=1 < LOW=2 < NONE=3 — argmin picks highest affinity
    best_idx = argmin(Int(s[2]) for s in scores)
    primary  = scores[best_idx][1]

    # Fallback: second best
    rest        = [s for s in scores if s[1] != primary]
    fallback_idx = isempty(rest) ? nothing : argmin(Int(s[2]) for s in rest)
    fallback     = fallback_idx === nothing ? :direct : rest[fallback_idx][1]

    # Hybrid if factor + trie both HIGH (e.g., GeodesicBGC-Composite)
    is_hybrid = profile.factor >= MEDIUM && profile.trie >= MEDIUM

    BackendChoice(primary, fallback, is_hybrid)
end

"""
    polish(code, choice) -> String

§9 Stage 4: Backend-specific polishing.

MM2 backend: renumber exec atom priorities sequentially (0,1,2,...) so
  no two exec atoms share the same priority — required by MM2 semantics.
  Malformed (exec ...) lines without a numeric second token get priority 0.

MORK/trie/factor/tensor backends: code is already in s-expression form
  compatible with space_add_all_sexpr! — pass through unchanged.
"""
function polish(code::String, choice::BackendChoice) :: String
    choice.primary != :mm2 && return code   # only MM2 needs priority renumbering
    isempty(strip(code))  && return code

    lines = split(code, "\n"; keepempty=false)
    result = String[]
    priority = 0

    for line in lines
        stripped = strip(line)
        # Renumber (exec P ...) atoms with sequential priorities
        m = match(r"^\(exec\s+(-?\d+)\s+(.*)\)$"s, stripped)
        if m !== nothing
            push!(result, "(exec $priority $(m.captures[2]))")
            priority += 1
        else
            push!(result, String(stripped))
        end
    end

    join(result, "\n")
end

# ── §12.1 Algorithm 5 — Geometry-aware compilation pipeline ──────────────────

"""
    CompilationResult

Result of Algorithm 5 (9-step geometry-aware pipeline, §12.1).
  residual_code   — lowered executable code (MM2 exec s-expressions or other)
  backend_choice  — selected backend
  templates_used  — all geometry templates involved
  coercion_chain  — coercions applied
  proof_artifacts — bisimulation obligations + exactness witnesses
  profile         — backend affinity scores
  phase_timings   :: Dict
"""
struct CompilationResult
    residual_code   :: String
    backend_choice  :: BackendChoice
    templates_used  :: Vector{GeometryTemplate}
    coercion_chain  :: Vector{Coercion}
    proof_artifacts :: Vector{BiSimObligation}
    profile         :: BackendProfile
    phase_timings   :: Dict{Symbol, Float64}
end

"""
    mg_compile(region::AbstractString, registry::SchemaRegistry;
               stats=MORKStatistics(), error_tolerance=0.0) -> CompilationResult

Algorithm 5 (Geometry-aware compilation and supercompilation pipeline, §12.1):
9 steps:
  1.  Parse and infer semantic objects and effect regions
  2.  Attach initial presentations from the framework registry
  3.  Compute backend affinities and candidate coercion graph
  4.  Run geometry-aware planning and selective supercompilation
  5.  Choose exact or approximate kernels subject to cost and witness constraints
  6.  Choose concurrency and distribution policies
  7.  Select backend(s): PeTTa, MM2, tensor, trie, factor, or hybrid
  8.  Lower to residual executable code and runtime metadata
  9.  Return mixed executable + proof/witness artifacts
"""
function mg_compile(region          :: AbstractString,
                    registry        :: SchemaRegistry = GLOBAL_REGISTRY;
                    stats           :: MORKStatistics = MORKStatistics(),
                    error_tolerance :: Float64        = 0.0) :: CompilationResult

    timings = Dict{Symbol, Float64}()

    # Step 1: Parse region → SNodes → infer semantic types
    t1 = @elapsed begin
        nodes   = parse_program(region)
        sem_types = _infer_semantic_types(nodes)
    end
    timings[:parse] = t1

    # Step 2: Attach initial presentations from registry
    t2 = @elapsed begin
        templates = _attach_presentations(sem_types, registry)
    end
    timings[:attach] = t2

    # Step 3: Backend affinity analysis + coercion graph
    t3 = @elapsed begin
        profile  = affinity_analysis(templates)
        coercions = _build_coercion_chain(templates, registry)
    end
    timings[:affinity] = t3

    # Step 4: Geometry-aware planning (join ordering via QueryPlanner)
    t4 = @elapsed begin
        optimized_templates = backend_neutral_optimize(templates, stats)
        planned_region = plan_program(region, stats)
    end
    timings[:plan] = t4

    # Step 5: Exact or approximate kernels
    t5 = @elapsed begin
        use_approx = error_tolerance > 0.0
        region_to_compile = planned_region
    end
    timings[:kernel_choice] = t5

    # Step 6: TyLAA concurrency verification (§7.2, §15.3)
    violations = String[]
    t6 = @elapsed begin
        violations = _apply_concurrency_policies!(templates)
        isempty(violations) || @warn "TyLAA violations: $(join(violations, "; "))"
    end
    timings[:concurrency] = t6

    # Step 7: Backend selection
    t7 = @elapsed begin
        choice = select_backend(profile, templates)
    end
    timings[:backend_select] = t7

    # Step 8: Lower to residual executable code AND runtime metadata
    #
    # Spec §12.1 Algorithm 5 step 8 says lower to "residual executable code
    # and runtime metadata". The previous implementation discarded
    # optimized_templates + coercions and routed through plain MM2 compile_program
    # — geometry-blind at the lowering boundary (audit 2026-05-30 Bug 1).
    #
    # Fix: each optimized template is given the chance to emit its own residual
    # via a registered lowering function (TEMPLATE_LOWERINGS map). The pieces
    # are concatenated, and templates without a registered lowering fall back
    # to the generic MM2 codegen path. Runtime metadata (which templates fired,
    # which coercions composed) is recorded as `;; mgfw:` comment annotations
    # at the top of the residual so the supercompiler / runtime can inspect it.
    t8 = @elapsed begin
        residual_pieces  = String[]
        templates_used   = Symbol[]
        coercions_used   = Symbol[]
        used_fallback    = false
        obligs           = nothing

        for t in optimized_templates
            fn = get_lowering(t.name)
            if fn !== nothing
                piece = fn(t, region_to_compile)
                push!(residual_pieces, piece)
                push!(templates_used, t.name)
                # Record which exact-coercions the template registered with
                # (for the runtime-metadata annotation block; supercompiler
                # uses this to know what conversions were committed to).
                for c in t.coercions
                    push!(coercions_used, c.name)
                end
            else
                used_fallback = true
            end
        end

        # Fallback: any template without a registered lowering → generic MM2
        # codegen on the planned region. This keeps callers without registered
        # templates working (and preserves the test_mgfw.jl baseline behavior).
        if used_fallback || isempty(residual_pieces)
            g      = MCoreGraph()
            root_ids = _sexpr_to_mcore!(g, parse_sexpr("(dummy)")) |> x -> [x]
            for node in parse_program(region_to_compile)
                push!(root_ids, _sexpr_to_mcore!(g, node))
            end
            fallback_residual, fb_obligs = compile_program(g, root_ids)
            push!(residual_pieces, fallback_residual)
            obligs = fb_obligs
        end
        obligs === nothing && (obligs = BiSimObligation[])

        # Compose: runtime-metadata annotations + concatenated residual.
        # Templates that fired produce machine-readable comments that the
        # runtime / supercompiler can parse; this is the "runtime metadata"
        # half of Algorithm 5 step 8.
        meta_lines = String[
            "; mgfw:templates " * join(string.(templates_used), ","),
            "; mgfw:coercions " * join(string.(unique(coercions_used)), ","),
            "; mgfw:backend " * string(choice),
        ]
        residual = polish(join([join(meta_lines, "\n"); residual_pieces], "\n"), choice)
    end
    timings[:lower] = t8

    # Step 9: Return
    CompilationResult(
        residual, choice, optimized_templates, coercions,
        obligs, profile, timings)
end

# ── §12.2 Geodesic BGC composite template ─────────────────────────────────────

"""
    build_geodesic_bgc_composite(reg) -> GeometryTemplate

§12.2: Build the GeodesicBGC-Composite template — the worked example of
multi-geometry composition (DualWorklist + Factor + Trie).

Components:
  scheduler → GeodesicBGC-Worklist (DualWorklist geometry)
  guidance  → FactorFGSurrogate (Factor geometry)
  evidence  → EvidenceCapsule (Trie geometry)

Data flow:
  scheduler → guidance  via :active_subgraph_query
  guidance  → scheduler via :fg_score_update
  scheduler → evidence  via :capsule_transport
  evidence  → scheduler via :overlap_veto
"""
function build_geodesic_bgc_composite(reg::SchemaRegistry) :: GeometryTemplate
    make_template(
        :GeodesicBGC_Composite,
        sem_model(:Q, :Formula),
        GEOM_FACTOR;  # primary geometry
        operators = [:scheduler_pop, :guidance_update, :evidence_update, :splice_check],
        effects   = [ReadEffect(DEFAULT_SPACE), AppendEffect(DEFAULT_SPACE),
                     ObserveEffect(DEFAULT_SPACE)],
        laws      = [:monotone_priority, :anytime_splice, :evidence_conserved],
        cache     = CacheContract([:capsule_cid, :sketch_cid, :epoch], [:new_token_mint]),
        exactness = EXACT,
        affinity  = Dict(:mm2 => :high, :mork => :high, :tensor => :low))
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _infer_semantic_types(nodes::Vector{SNode}) :: Vector{SemanticType}
    types = SemanticType[]
    for node in nodes
        node isa SList || continue
        items = (node::SList).items
        isempty(items) && continue
        # Heuristic: exec atoms → Sched, rule atoms → Model, query atoms → Rel
        head = items[1]
        if head isa SAtom
            name = (head::SAtom).name
            if name == "exec"
                push!(types, sem_sched(:A))
            elseif length(items) >= 3 && is_conjunction(items[2])
                push!(types, sem_model(:Q, :Formula))
            else
                push!(types, sem_rel(:A, :B))
            end
        end
    end
    isempty(types) ? [sem_model(:Q, :Formula)] : types
end

function _attach_presentations(sem_types::Vector{SemanticType},
                                reg::SchemaRegistry) :: Vector{GeometryTemplate}
    templates = GeometryTemplate[]
    for st in sem_types
        # Find best matching template in registry
        candidates = search(reg; semantic_kind=st.kind)
        if !isempty(candidates)
            push!(templates, candidates[1])
        else
            # Create a default template
            geom = st.kind == SK_MODEL ? GEOM_FACTOR :
                   st.kind == SK_PROG  ? GEOM_DAG    :
                   st.kind == SK_CODEC ? GEOM_TRIE   : GEOM_FACTOR
            push!(templates, make_template(
                Symbol("auto_$(st.kind)"), st, geom;
                operators=[:read, :write],
                effects=[ReadEffect(DEFAULT_SPACE)]))
        end
    end
    isempty(templates) ? [TEMPLATE_HEURISTIC_MP] : templates
end

function _build_coercion_chain(templates::Vector{GeometryTemplate},
                                reg::SchemaRegistry) :: Vector{Coercion}
    coercions = Coercion[]
    for i in 1:length(templates)-1
        g1 = geometry_of(templates[i])
        g2 = geometry_of(templates[i+1])
        g1 == g2 && continue
        path = coercion_path(reg, g1, g2)
        append!(coercions, path)
    end
    # Also add registered minimum coercions
    append!(coercions, REGISTERED_COERCIONS)
    coercions
end

"""
    _apply_concurrency_policies!(templates) → Vector{String}

TyLAA verification (§7.2, §15.3): verify that compiled geometry templates
satisfy independence-respecting traces — the three TyLAA conditions:
  LC (Local Confluence): each pair of applicable rewrites has a common reduct
  G  (Generalisation):   the type layer is at least as general as the red theory
  SHD (Shared Derivation): parallel independent steps share a common derivation

Practical approximation using Effects.jl:
  For each pair of templates, check that their effect sets commute.
  If effects commute → LC holds for that pair (independent steps are safe).
  Returns a list of violation messages (empty = all checks pass).
"""
function _apply_concurrency_policies!(templates::Vector{GeometryTemplate}) :: Vector{String}
    violations = String[]
    for i in 1:length(templates), j in (i+1):length(templates)
        t1 = templates[i]
        t2 = templates[j]
        # Extract effect kinds from concurrency contracts
        e1 = _template_effect_kind(t1)
        e2 = _template_effect_kind(t2)
        # LC check: effects must commute for safe parallel execution
        if !effects_commute(e1, e2)
            push!(violations,
                "TyLAA LC violation: templates $(t1.name) ($(e1)) and $(t2.name) ($(e2)) " *
                "have non-commuting effects — parallel execution unsafe")
        end
    end
    violations
end

function _template_effect_kind(t::GeometryTemplate) :: EffectKind
    contract = t.local_concurrency
    # The old classifier checked for [:never, :read_only, :append_only, :always]
    # which NEVER appear in commutes_when — `default_local_concurrency` emits
    # geometry-specific tags like :disjoint_targets_or_monotone_join,
    # :regions_disjoint, :distinct_prefix_ownership, :distinct_output_patch_sets.
    # So every template fell through to EFF_READ and the TyLAA verification
    # step was dead.
    #
    # The corrected classifier maps the *actual* tags `default_local_concurrency`
    # emits to the right EffectKind:
    #
    #   Factor  → :disjoint_targets_or_monotone_join  (Append-like, commutative)
    #   DAG     → :regions_disjoint, :rewrite_confluent  (Write — needs confluence)
    #   Trie    → :distinct_prefix_ownership  (Append-like, counter-merge)
    #   Tensor  → :distinct_output_patch_sets  (Write — patch replay)
    #
    # Plus the original generic tags (kept for hand-built templates).
    cw = contract.commutes_when
    # Generic tags (override geometry inference)
    :never       ∈ cw && return EFF_WRITE
    :read_only   ∈ cw && return EFF_READ
    :append_only ∈ cw && return EFF_APPEND
    :always      ∈ cw && return EFF_PURE
    # Geometry-specific tags from default_local_concurrency
    (:disjoint_targets_or_monotone_join ∈ cw ||
     :disjoint_prefix_ownership         ∈ cw ||
     :distinct_prefix_ownership         ∈ cw)        && return EFF_APPEND
    (:regions_disjoint                  ∈ cw ||
     :distinct_output_patch_sets        ∈ cw)        && return EFF_WRITE
    EFF_READ  # safe default
end

"""
    mg_run!(s::Space, region::AbstractString; kwargs...) -> Tuple{CompilationResult, Int}

High-level entry point: compile + execute a region.
Compiles via mg_compile, then loads and runs in MORK Space.
"""
function mg_run!(s      :: Space,
                 region :: AbstractString;
                 registry  :: SchemaRegistry = GLOBAL_REGISTRY,
                 stats     :: MORKStatistics = MORKStatistics(),
                 max_steps :: Int            = typemax(Int),
                 kwargs...) :: Tuple{CompilationResult, Int}

    result = mg_compile(region, registry; stats=stats, kwargs...)
    space_add_all_sexpr!(s, result.residual_code)
    n_steps = space_metta_calculus!(s, max_steps)
    (result, n_steps)
end

export AffinityLevel, HIGH, MEDIUM, LOW, NONE
export BackendProfile, BackendChoice
export affinity_analysis, backend_neutral_optimize, select_backend, polish
export CompilationResult, mg_compile, mg_run!
export build_geodesic_bgc_composite
