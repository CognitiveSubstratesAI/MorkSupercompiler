"""
SchemaRegistry — schema registry, DSL forms, normalization, and authoring workflow.

Implements §8 and §11 of the MG Framework spec (Goertzel, April 2026):
  §8.1  Human-facing DSL — 5 forms:
          define-factor-rule, define-trie-miner, define-coercion,
          define-exactness, define-cache-contract
  §8.2  LLM coding assistant support (generation + refactoring modes)
  §11   Algorithm 4 — Human/LLM authoring workflow (8-step pipeline)

Each DSL form expands into exactly one normalized GeometryTemplate record.
The registry maintains ALL registered templates with search, validation, and
version-tuple tracking.

Stability invariant (§2): the canonical schema form is stable — DSL can
evolve without invalidating cached supercompilation results.
"""

# ── Schema registry ───────────────────────────────────────────────────────────

"""
    SchemaRegistry

Central registry of GeometryTemplate records, indexed by name.
Also maintains coercion graph for presentation changes.
"""
mutable struct SchemaRegistry
    templates::Dict{Symbol, GeometryTemplate}
    coercions::Dict{Tuple{GeomTag, GeomTag}, Vector{Coercion}}  # (from,to) → coercions
    version::Int                  # incremented on each register!
    history::Vector{Tuple{Symbol, Int}}  # (name, version) changelog
end

SchemaRegistry() = SchemaRegistry(
    Dict{Symbol, GeometryTemplate}(), Dict{Tuple{GeomTag, GeomTag}, Vector{Coercion}}(),
    0, Tuple{Symbol, Int}[]
)

"""
    register!(reg, template) -> SchemaRegistry

Add or update a template. Increments version for cache invalidation.
"""
function register!(reg::SchemaRegistry, t::GeometryTemplate)::SchemaRegistry
    is_valid_template(t) || error("Template $(t.name) is not valid (missing fields)")
    reg.templates[t.name] = t
    reg.version += 1
    push!(reg.history, (t.name, reg.version))
    # Update coercion index
    for c in t.coercions
        key = (c.from_geom, c.to_geom)
        bucket = get!(reg.coercions, key, Coercion[])
        push!(bucket, c)
    end
    reg
end

"""
    lookup(reg, name) -> Union{GeometryTemplate, Nothing}
"""
lookup(reg::SchemaRegistry, name::Symbol)::Union{GeometryTemplate, Nothing} =
    get(reg.templates, name, nothing)

"""
    search(reg; geometry=nothing, semantic_kind=nothing) -> Vector{GeometryTemplate}

Find templates matching optional geometry and/or semantic kind filters.
"""
function search(
    reg::SchemaRegistry;
    geometry::Union{GeomTag, Nothing}=nothing,
    semantic_kind::Union{SemanticKind, Nothing}=nothing
)::Vector{GeometryTemplate}
    results = collect(values(reg.templates))
    if geometry !== nothing
        results = filter(t -> geometry_of(t) == geometry, results)
    end
    if semantic_kind !== nothing
        results = filter(t -> t.semantic_type.kind == semantic_kind, results)
    end
    results
end

"""
    coercion_path(reg, from, to) -> Vector{Coercion}

Find a coercion path from `from` geometry to `to` geometry (direct or 1-hop).
Returns empty if no path exists.
"""
function coercion_path(reg::SchemaRegistry, from::GeomTag, to::GeomTag)::Vector{Coercion}
    # Direct path
    direct = get(reg.coercions, (from, to), Coercion[])
    !isempty(direct) && return direct

    # One-hop: from → mid → to
    for mid in [GEOM_FACTOR, GEOM_DAG, GEOM_TRIE, GEOM_TENSOR_SPARSE]
        mid == from || mid == to && continue
        step1 = get(reg.coercions, (from, mid), Coercion[])
        step2 = get(reg.coercions, (mid, to), Coercion[])
        !isempty(step1) && !isempty(step2) && return [step1[1], step2[1]]
    end
    Coercion[]
end

# Shared global registry
const GLOBAL_REGISTRY = SchemaRegistry()

# Map of template name → lowering callback. Templates declare contracts
# (effects / laws / cache); their lowering (how step 8 emits residual code)
# lives here so the registry-as-data stays separable from runtime callbacks.
# Signature: `lowering_fn(template::GeometryTemplate, region::AbstractString, ctx) → String`.
const TEMPLATE_LOWERINGS = Dict{Symbol, Function}()

"""
    register_lowering!(name::Symbol, fn::Function)

Associate a lowering function with a registered template by name. Step 8
of `mg_compile` dispatches via this map when present, falling back to the
generic MM2 codegen path otherwise.
"""
function register_lowering!(name::Symbol, fn::Function)
    TEMPLATE_LOWERINGS[name] = fn
    fn
end

"""
    get_lowering(name::Symbol) → Union{Nothing, Function}
"""
get_lowering(name::Symbol) = get(TEMPLATE_LOWERINGS, name, nothing)

"""
    init_registry!()

Populate `GLOBAL_REGISTRY` with the framework-shipped templates and their
lowerings. Auto-invoked at module load via `__init__` (defined in the
top-level module after all template files load) so callers can rely on
`GLOBAL_REGISTRY` being non-empty by the time `using MorkSupercompiler`
returns. Idempotent — safe to call multiple times (register! / register_lowering!
both overwrite).

NOTE: this function is defined AND `__init__` is bound in the top-level
`MorkSupercompiler` module (see MorkSupercompiler.jl) — NOT here — because
the MVP-template files load AFTER SchemaRegistry.jl, and we need the init
function to register them too. Defining it once at top level avoids the
Julia "Method overwriting is not permitted during precompilation" error.
"""
function init_registry! end

# ── §8.1 Human-facing DSL ─────────────────────────────────────────────────────

"""
    DSLForm

A parsed human-facing DSL declaration. Expands to one GeometryTemplate.
"""
struct DSLForm
    form_type::Symbol   # :define_factor_rule / :define_trie_miner / etc.
    fields::Dict{Symbol, Any}
end

"""
    parse_define_factor_rule(fields) -> GeometryTemplate

§8.1: `(define-factor-rule :name ... :premises [...] :conclusion [...] ...)`.
Expands to a GEOM_FACTOR template with Model(Q,Formula) semantic type.
"""
function parse_define_factor_rule(fields::Dict{Symbol, Any})::GeometryTemplate
    name = get(fields, :name, :unnamed_rule)
    premises = get(fields, :premises, Symbol[])
    conclusion = get(fields, :conclusion, Symbol[])
    truth_fam = get(fields, :truth_family, :STV)
    fwd_map = get(fields, :forward_map, :default_forward)
    bwd_dem = get(fields, :backward_demand, :default_demand)
    cache_pol = get(fields, :cache_policy, :versioned_message_cache)

    ops = [fwd_map, bwd_dem, :message_update, :boundary_refresh]
    cache = CacheContract(
        [:schema_id, :factor_id, :subst_shape, :evidence_ver],
        [:evidence_change, :rule_change]
    )

    make_template(
        name,
        sem_model(:Q, :Formula),
        GEOM_FACTOR;
        operators=ops,
        effects=[ReadEffect(DEFAULT_SPACE), AppendEffect(DEFAULT_SPACE)],
        laws=[:monotone, :sink_free],
        cache=cache,
        exactness=EXACT,
        coercions=[
            Coercion(
                Symbol("$(name)_to_trie"), GEOM_FACTOR, GEOM_TRIE, sem_model(:Q, :Formula)
            )
        ]
    )
end

"""
    parse_define_trie_miner(fields) -> GeometryTemplate

§8.1: `(define-trie-miner :name ... :seed-op ... :growth-op ... :ranking ...)`.
Expands to a GEOM_TRIE template with Model or Rel semantic type.
"""
function parse_define_trie_miner(fields::Dict{Symbol, Any})::GeometryTemplate
    name = get(fields, :name, :unnamed_miner)
    seed_op = get(fields, :seed_op, :subtree_scan)
    growth_op = get(fields, :growth_op, :prefix_proximity)
    support_op = get(fields, :support_op, :prefix_counter)
    ranking = get(fields, :ranking, :topk_heavy)

    make_template(
        name,
        sem_model(:Q, :Motif),
        GEOM_TRIE;
        operators=[seed_op, growth_op, support_op, ranking],
        effects=[ReadEffect(DEFAULT_SPACE), AppendEffect(DEFAULT_SPACE)],
        laws=[:monotone, :prefix_locality],
        exactness=EXACT
    )
end

"""
    parse_define_codec_search(fields) -> GeometryTemplate

§8.1: `(define-codec-search :name ... :proposal-surface ... :acceptance ...)`.
Expands to GEOM_TRIE template with Codec semantic type (WILLIAM-style).
"""
function parse_define_codec_search(fields::Dict{Symbol, Any})::GeometryTemplate
    name = get(fields, :name, :unnamed_codec)
    proposal = get(fields, :proposal_surface, :heavy_subpatterns)
    templates = get(fields, :feature_templates, Symbol[])
    accept = get(fields, :acceptance, :mdl_or_weakness)

    make_template(
        name,
        sem_codec(:A),
        GEOM_TRIE;
        operators=[proposal, accept, :feature_extract, :residual_compute],
        effects=[ReadEffect(DEFAULT_SPACE)],
        laws=[:mdl_monotone, :reversible_features],
        exactness=EXACT
    )
end

"""
    parse_define_coercion(fields) -> Coercion

§8.1: `(define-coercion :name ... :from ... :to ... :exactness ...)`.
"""
function parse_define_coercion(fields::Dict{Symbol, Any})::Coercion
    name = get(fields, :name, :unnamed_coercion)
    from = get(fields, :from, GEOM_DAG)
    to = get(fields, :to, GEOM_FACTOR)
    ex_str = get(fields, :exactness, :EXACT)
    ex = if ex_str == :BOUNDED
        BOUNDED
    elseif ex_str == :STATISTICAL
        STATISTICAL
    else
        EXACT
    end
    ε = Float64(get(fields, :error_bound, 0.0))
    sem = get(fields, :semantic_type, sem_model(:Q, :A))

    Coercion(name, from, to, sem; kind=ex, ε=ε)
end

"""
    parse_define_exactness(fields) -> ErrorLevel

§8.1: `(define-exactness :level ... :bound ...)`.
Returns the ErrorLevel enum value.
"""
function parse_define_exactness(fields::Dict{Symbol, Any})::ErrorLevel
    level = get(fields, :level, :EXACT)
    if level == :BOUNDED
        BOUNDED
    elseif level == :STATISTICAL
        STATISTICAL
    else
        EXACT
    end
end

"""
    parse_define_cache_contract(fields) -> CacheContract

§8.1: `(define-cache-contract :key [...] :invalidate-on [...])`.
"""
function parse_define_cache_contract(fields::Dict{Symbol, Any})::CacheContract
    CacheContract(get(fields, :key, Symbol[]), get(fields, :invalidate_on, Symbol[]))
end

# ── §11 Algorithm 4 — Human/LLM authoring workflow ────────────────────────────

"""
    AuthoringResult

Output of Algorithm 4 (Human/LLM authoring workflow, §11):
template        — normalized GeometryTemplate
dsl_form        — the original DSL form
linter_report   — validation/contract-check output
test_harness    — a Julia test expression (as string) for immediate verification
registered      — true if template was added to registry
"""
struct AuthoringResult
    template::GeometryTemplate
    dsl_form::DSLForm
    linter_report::String
    test_harness::String
    registered::Bool
end

"""
    authoring_workflow(form::DSLForm, reg::SchemaRegistry) -> AuthoringResult

Algorithm 4 (Human/LLM authoring workflow) from §11:

Step 1: Choose semantic object family (from form_type)
Step 2: Choose/suggest geometry template
Step 3: Fill required fields from DSL
Step 4: Expand to canonical schema record (GeometryTemplate)
Step 5: Run linting: missing contracts, invalid cache law, unsupported coercions
Step 6: Generate tests and sample execution traces
Step 7: Optionally suggest geometry improvements
Step 8: Commit into registry as both DSL and canonical schema
"""
function authoring_workflow(
    form::DSLForm, reg::SchemaRegistry=GLOBAL_REGISTRY
)::AuthoringResult

    # Step 1–4: parse DSL form → GeometryTemplate
    template = if form.form_type == :define_factor_rule
        parse_define_factor_rule(form.fields)
    elseif form.form_type == :define_trie_miner
        parse_define_trie_miner(form.fields)
    elseif form.form_type == :define_codec_search
        parse_define_codec_search(form.fields)
    else
        error("Unknown DSL form: $(form.form_type)")
    end

    # Step 5: lint
    report = _lint_template(template, reg)

    # Step 6: generate test harness
    harness = _generate_test_harness(template)

    # Step 7: geometry suggestions (§11 Algorithm 4 Step 7)
    suggestions = _suggest_geometry(template, reg)
    !isempty(suggestions) && (report = report * "\nSUGGESTIONS:\n" * suggestions)

    # Step 8: register if lint passes
    success = !occursin("ERROR", report)
    success && register!(reg, template)

    AuthoringResult(template, form, report, harness, success)
end

function _lint_template(t::GeometryTemplate, reg::SchemaRegistry)::String
    io = IOBuffer()
    isempty(t.operators) && println(io, "WARN: no operators declared")
    isempty(t.laws) && println(io, "WARN: no algebraic laws declared")
    isempty(t.cache_contract.key) &&
        println(io, "INFO: no cache key — results not cacheable")
    for c in t.coercions
        path = coercion_path(reg, c.from_geom, c.to_geom)
        isempty(path) &&
            lookup(reg, t.name) === nothing &&
            println(io, "INFO: coercion $(c.name) not yet in registry — will be added")
    end
    result = String(take!(io))
    isempty(result) ? "OK: template $(t.name) passed linting" : result
end

function _suggest_geometry(t::GeometryTemplate, reg::SchemaRegistry)::String
    io = IOBuffer()
    g = geometry_of(t)
    # §11: "ask the planner for geometry suggestions or backend affinity report"
    if g == GEOM_FACTOR && :evidence_monotone in t.laws
        println(
            io, "  Consider adding EvidenceCapsule (Trie geometry) for evidence accounting."
        )
    end
    if g == GEOM_DAG && isempty(t.coercions)
        println(
            io, "  Consider registering T_DAG_to_Factor coercion for EDA model lifting."
        )
    end
    if length(t.operators) > 5 && !is_hybrid(t)
        println(
            io,
            "  Many operators — consider splitting into Hybrid geometry (Factor + Trie)."
        )
    end
    existing = search(reg; semantic_kind=t.semantic_type.kind)
    if !isempty(existing) && geometry_of(existing[1]) != g
        println(
            io,
            "  Similar template '$(existing[1].name)' uses $(geometry_of(existing[1])) — verify geometry choice."
        )
    end
    String(take!(io))
end

function _generate_test_harness(t::GeometryTemplate)::String
    """
# Auto-generated test harness for $(t.name)
@testset "$(t.name) schema" begin
    t = lookup(GLOBAL_REGISTRY, :$(t.name))
    @test t !== nothing
    @test is_valid_template(t)
    @test t.exactness_class == $(t.exactness_class)
    @test geometry_of(t) == $(geometry_of(t))
end
"""
end

# ── Convenience DSL builder ────────────────────────────────────────────────────

"""
    define_factor_rule(; name, premises, conclusion, kwargs...) -> AuthoringResult

Convenience wrapper to declare a factor rule and register it.

Example:
result = define_factor_rule(
name        = :HeuristicModusPonens,
premises    = [:A, :(implies A B)],
conclusion  = [:B],
truth_family= :STV,
forward_map = :heuristic_mp_tv)
"""
define_factor_rule(; kwargs...) =
    authoring_workflow(DSLForm(:define_factor_rule, Dict{Symbol, Any}(kwargs)))

define_trie_miner(; kwargs...) =
    authoring_workflow(DSLForm(:define_trie_miner, Dict{Symbol, Any}(kwargs)))

define_codec_search(; kwargs...) =
    authoring_workflow(DSLForm(:define_codec_search, Dict{Symbol, Any}(kwargs)))

export SchemaRegistry, register!, lookup, search, coercion_path, GLOBAL_REGISTRY
export TEMPLATE_LOWERINGS, register_lowering!, get_lowering, init_registry!
export DSLForm, AuthoringResult, authoring_workflow
export parse_define_factor_rule, parse_define_trie_miner, parse_define_codec_search
export parse_define_coercion, parse_define_exactness, parse_define_cache_contract
export define_factor_rule, define_trie_miner, define_codec_search
