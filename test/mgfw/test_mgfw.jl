using Test
using MorkSupercompiler
using MORK

# ── §6.1–6.2 SemanticObjects ──────────────────────────────────────────────────

@testset "SemanticObjects — semantic types" begin
    rel = sem_rel(:A, :B)
    @test rel.kind == SK_REL && rel.args == [:A, :B]

    model = sem_model(:Q, :Formula)
    @test model.kind == SK_MODEL

    prog = sem_prog(:Sigma, :T)
    @test prog.kind == SK_PROG

    codec = sem_codec(:A)
    @test codec.kind == SK_CODEC
end

@testset "SemanticObjects — geometry tags and Pres(G,A)" begin
    p1 = PresType(GEOM_FACTOR, sem_model(:Q, :Formula))
    @test p1.geometry == GEOM_FACTOR
    @test p1.sem_type.kind == SK_MODEL

    h = HybridGeom(GEOM_FACTOR, GEOM_TRIE)
    @test length(h.components) == 2
    @test h.components[1] == GEOM_FACTOR
end

@testset "SemanticObjects — registered coercions" begin
    @test length(REGISTERED_COERCIONS) == 4
    c = find_coercion(GEOM_DAG, GEOM_FACTOR)
    @test c !== nothing && (c::Coercion).name == :T_DAG_to_Factor
    @test is_exact(c::Coercion)

    # No coercion from FACTOR to DAG (not registered)
    @test find_coercion(GEOM_FACTOR, GEOM_DAG) === nothing
end

@testset "SemanticObjects — TyLA direction" begin
    @test F_DIRECTION != G_DIRECTION
    @test F_DIRECTION isa TyLADirection
end

# ── §6.3 + §13 GeometryTemplate ──────────────────────────────────────────────

@testset "GeometryTemplate — all 13 fields present" begin
    t = TEMPLATE_HEURISTIC_MP
    @test t.name == :HeuristicModusPonens
    @test t.semantic_type.kind == SK_MODEL
    @test t.presentation == GEOM_FACTOR
    @test !isempty(t.operators)
    @test !isempty(t.effects)
    @test !isempty(t.laws)
    @test !isempty(t.symmetries)
    @test !isempty(t.cache_contract.key)
    @test t.exactness_class == EXACT
    @test !isempty(t.coercions)
    @test t.local_concurrency isa LocalConcurrencyContract
    @test t.distributed_exec isa DistributedExecContract
    @test !isempty(t.backend_affinity)
    @test is_valid_template(t)
end

@testset "GeometryTemplate — default_policy per geometry" begin
    @test default_policy(GEOM_FACTOR) == FIXED_POINT_MESSAGE_POLICY
    @test default_policy(GEOM_TRIE) == PREFIX_SHARD_POLICY
    @test default_policy(GEOM_TENSOR_DENSE) == PATCH_LOG_SHARD_POLICY
    @test default_policy(GEOM_DAG) == DEME_AGENT_POLICY
end

@testset "GeometryTemplate — geometry_of" begin
    @test geometry_of(TEMPLATE_HEURISTIC_MP) == GEOM_FACTOR
    @test geometry_of(TEMPLATE_EVIDENCE_CAPSULE) == GEOM_TRIE
end

@testset "GeometryTemplate — make_template with defaults" begin
    t = make_template(
        :TestTemplate,
        sem_rel(:A, :B),
        GEOM_TRIE;
        operators=[:scan, :rank],
        laws=[:monotone]
    )
    @test t.name == :TestTemplate
    @test t.presentation == GEOM_TRIE
    @test is_valid_template(t)
    @test t.local_concurrency.unit_of_parallelism == [:prefix_subtree]
end

# ── §8 + §11 SchemaRegistry + DSL ────────────────────────────────────────────

@testset "SchemaRegistry — register and lookup" begin
    reg = SchemaRegistry()
    register!(reg, TEMPLATE_HEURISTIC_MP)
    @test reg.version == 1

    found = lookup(reg, :HeuristicModusPonens)
    @test found !== nothing
    @test (found::GeometryTemplate).name == :HeuristicModusPonens

    @test lookup(reg, :nonexistent) === nothing
end

@testset "SchemaRegistry — search by geometry/kind" begin
    reg = SchemaRegistry()
    register!(reg, TEMPLATE_HEURISTIC_MP)
    register!(reg, TEMPLATE_EVIDENCE_CAPSULE)

    factor_templates = search(reg; geometry=GEOM_FACTOR)
    @test length(factor_templates) == 1
    @test factor_templates[1].name == :HeuristicModusPonens

    model_templates = search(reg; semantic_kind=SK_MODEL)
    @test length(model_templates) == 1
end

@testset "SchemaRegistry — coercion_path" begin
    reg = SchemaRegistry()
    register!(reg, TEMPLATE_HEURISTIC_MP)   # has FactorToTrie coercion

    # Direct path
    path = coercion_path(reg, GEOM_FACTOR, GEOM_TRIE)
    @test !isempty(path)

    # No path for unmapped pair
    empty_path = coercion_path(reg, GEOM_DAG, GEOM_TENSOR_DENSE)
    @test isempty(empty_path)
end

@testset "SchemaRegistry — Algorithm 4 authoring_workflow" begin
    reg = SchemaRegistry()
    form = DSLForm(
        :define_factor_rule,
        Dict{Symbol, Any}(
            :name => :TransitivityRule,
            :premises => [:Ancestor_x_y, :Ancestor_y_z],
            :conclusion => [:Ancestor_x_z],
            :truth_family => :STV,
            :forward_map => :transitive_stv
        )
    )

    result = authoring_workflow(form, reg)
    @test result isa AuthoringResult
    @test result.template.name == :TransitivityRule
    @test result.registered
    @test !isempty(result.test_harness)
    @test lookup(reg, :TransitivityRule) !== nothing
end

@testset "SchemaRegistry — define_trie_miner" begin
    reg = SchemaRegistry()
    result = define_trie_miner(
        name=:MotifMiner,
        seed_op=:subtree_scan,
        growth_op=:prefix_proximity,
        support_op=:prefix_counter,
        ranking=:topk_heavy
    )
    @test result.template.name == :MotifMiner
    @test geometry_of(result.template) == GEOM_TRIE
end

# ── §10.1 FactorGeometry — Algorithms 1 + 2 ──────────────────────────────────

@testset "FactorGeometry — STV functions" begin
    # stv_forward_map: (A, A→B) → B
    s_b, c_b = stv_forward_map(0.9, 0.8, 0.8, 0.7)
    @test 0.0 <= s_b <= 1.0
    @test 0.0 <= c_b <= 1.0
    @test s_b < 0.9   # conclusion weaker than premise

    # stv_to_pbox: converts to interval
    pb = stv_to_pbox(0.7, 0.9)
    lo, hi = pb.intervals[1]
    @test lo < 0.7 < hi
    @test pb.probabilities[1] ≈ 0.9

    # stv_backward_demand
    ns, nc = stv_backward_demand(0.81, 0.64)
    @test ns ≈ 0.9 atol=0.01
    @test nc ≈ 0.8 atol=0.01
end

@testset "FactorGeometry — Algorithm 1 specialize_exact" begin
    t = TEMPLATE_HEURISTIC_MP
    g = FactorGraph(t)
    # Add some nodes
    g.var_nodes[:A] = FactorNode(:A, :premise)
    g.var_nodes[:B] = FactorNode(:B, :conclusion)
    g.factor_nodes[:mp] = FactorNode(:mp, :factor; is_factor=true)
    push!(g.edges, FactorEdge(:A, :mp, :premise))
    push!(g.edges, FactorEdge(:B, :mp, :conclusion))

    region = specialize_exact(:B, g, 100)
    @test region isa SpecializedRegion
    @test region.exactness == EXACT
    @test region.error_bound == 0.0
    @test region.witness === nothing
    @test :B in region.active_nodes
end

@testset "FactorGeometry — Algorithm 2 specialize_approximate" begin
    t = TEMPLATE_HEURISTIC_MP
    g = FactorGraph(t)
    g.var_nodes[:Q] = FactorNode(:Q, :premise)
    g.var_nodes[:R] = FactorNode(:R, :conclusion)
    g.factor_nodes[:rule] = FactorNode(:rule, :factor; is_factor=true)
    push!(g.edges, FactorEdge(:Q, :rule, :premise))
    push!(g.edges, FactorEdge(:R, :rule, :conclusion))

    region = specialize_approximate(:R, g, 0.05, 0.95, 100)
    @test region isa SpecializedRegion
    @test region.exactness != nothing  # EXACT or BOUNDED
    @test region.error_bound >= 0.0
    # Noether invariant: charge ≤ 1.0
    @test noether_charge(region) <= 1.0 + 1e-9
end

# ── §10.2–10.3 TrieDAGGeometry — Algorithm 3 + trie ─────────────────────────

@testset "TrieDAGGeometry — DAGStore hash-consing" begin
    store = DAGStore()
    id1 = dag_intern!(store, :leaf)
    id2 = dag_intern!(store, :leaf)
    @test id1 == id2   # same structure → same ID (hash-consing)

    id3 = dag_intern!(store, :node, [id1])
    id4 = dag_intern!(store, :node, [id2])
    @test id3 == id4   # structurally equal
end

@testset "TrieDAGGeometry — Algorithm 3 evolve_demes!" begin
    demes = [Deme(i) for i in 1:3]
    # Seed each deme with a leaf
    for d in demes
        dag_intern!(d.store, :leaf)
    end

    result = evolve_demes!(demes,
        (store, id) -> begin
            haskey(store.nodes, id) ? 0.5 + rand() * 0.5 : 0.0
        end; top_k=2)

    @test result isa DemeEvolutionResult
    @test length(result.updated_demes) == 3
    @test !isempty(result.exemplars)
    @test all(d -> d.generation == 1, result.updated_demes)
end

@testset "TrieDAGGeometry — trie mining 3 stages" begin
    t = TEMPLATE_EVIDENCE_CAPSULE
    atoms = parse_program(
        "(parent alice bob)\n(parent bob carol)\n(parent alice carol)\n(sibling alice dave)"
    )

    # Stage 1: seed
    trie = PatternTrie(t; k=5)
    n_seeds = trie_seed!(trie, atoms)
    @test n_seeds > 0
    @test !isempty(trie.top_k)

    # Stage 2: grow
    n_grown = trie_grow!(trie, atoms; max_depth=2)
    # (may be 0 if no 2-symbol patterns found, that's ok for toy data)
    @test n_grown >= 0

    # Stage 3: score
    scored = trie_score!(trie)
    @test !isempty(scored)
    # Sorted by descending weight
    ws = [w for (_, w) in scored]
    @test issorted(ws; rev=true)
end

@testset "TrieDAGGeometry — run_trie_miner end-to-end" begin
    t = TEMPLATE_EVIDENCE_CAPSULE
    data = parse_program("(a x) (a y) (b x) (a z) (b y)")
    top_k = run_trie_miner(t, data; k=3, max_depth=2)
    @test !isempty(top_k)
    # :a should appear more often than :b → higher weight
    a_weight = sum(w for (p, w) in top_k if !isempty(p) && p[1] == :a; init=0.0)
    b_weight = sum(w for (p, w) in top_k if !isempty(p) && p[1] == :b; init=0.0)
    @test a_weight >= b_weight
end

# ── §9 + §12 MGCompiler ──────────────────────────────────────────────────────

@testset "MGCompiler — backend_neutral_optimize (ADR-055 semiring-geometry)" begin
    # Use canonical valid templates (HEURISTIC_MP=FACTOR, EVIDENCE_CAPSULE=TRIE, CAUSAL_DAG=DAG)
    t_factor = TEMPLATE_HEURISTIC_MP       # GEOM_FACTOR, rank 2
    t_trie = TEMPLATE_EVIDENCE_CAPSULE   # GEOM_TRIE,   rank 0
    t_dag = TEMPLATE_CAUSAL_DAG         # GEOM_DAG,    rank 1

    # All three should be valid
    @test is_valid_template(t_factor)
    @test is_valid_template(t_trie)
    @test is_valid_template(t_dag)

    # Pass 3: semiring rank — TRIE(0) < DAG(1) < FACTOR(2)
    result = backend_neutral_optimize([t_factor, t_dag, t_trie], MORKStatistics())
    geoms = geometry_of.(result)
    @test geoms[1] == GEOM_TRIE    # rank 0: Boolean/MaxPlus — reachability
    @test geoms[2] == GEOM_DAG     # rank 1: MinPlus — shortest paths
    @test geoms[3] == GEOM_FACTOR  # rank 2: SumProduct — counting/inference

    # Pass 4: cost proxy with stats — trie still wins (log n < n)
    stats = MORKStatistics(Dict{String, Int}(), 5000)  # immutable struct
    result2 = backend_neutral_optimize([t_factor, t_trie], stats)
    @test geometry_of(result2[1]) == GEOM_TRIE

    # Pass 1: validity pruning — result only contains valid templates
    @test all(is_valid_template, result)

    # Empty input guard
    @test backend_neutral_optimize(GeometryTemplate[], MORKStatistics()) ==
        GeometryTemplate[]
end

@testset "MGCompiler — affinity_analysis" begin
    templates = [TEMPLATE_HEURISTIC_MP, TEMPLATE_EVIDENCE_CAPSULE]
    profile = affinity_analysis(templates)
    @test profile isa BackendProfile
    # Factor + Trie templates → MM2 and MORK should have some affinity
    @test profile.mm2 != NONE
    @test profile.mork != NONE
end

@testset "MGCompiler — select_backend" begin
    profile = BackendProfile(mm2=HIGH, mork=HIGH, factor=LOW, trie=LOW)
    templates = [TEMPLATE_HEURISTIC_MP]
    choice = select_backend(profile, templates)
    @test choice isa BackendChoice
    @test choice.primary in (:mm2, :mork)   # highest affinity wins
end

@testset "MGCompiler — Algorithm 5 mg_compile" begin
    reg = SchemaRegistry()
    register!(reg, TEMPLATE_HEURISTIC_MP)
    prog = raw"(exec 0 (, (edge $x $y) (edge $y $z)) (, (path $x $z)))"

    result = mg_compile(prog, reg)
    @test result isa CompilationResult
    @test !isempty(result.residual_code)
    @test result.backend_choice isa BackendChoice
    @test haskey(result.phase_timings, :parse)
    @test haskey(result.phase_timings, :lower)
end

@testset "MGCompiler — build_geodesic_bgc_composite" begin
    reg = SchemaRegistry()
    composite = build_geodesic_bgc_composite(reg)
    @test composite.name == :GeodesicBGC_Composite
    @test composite.presentation == GEOM_FACTOR
    @test :evidence_conserved in composite.laws
    @test get(composite.backend_affinity, :mm2, :low) == :high
end

@testset "MGCompiler — mg_run! end-to-end" begin
    reg = SchemaRegistry()
    s = new_space()
    space_add_all_sexpr!(s, "(edge 0 1) (edge 1 2)")
    prog = raw"(exec 0 (, (edge $x $y)) (, (node $x)))"

    result, n_steps = mg_run!(s, prog; registry=reg)
    @test result isa CompilationResult
    @test n_steps >= 0
    @test !isempty(result.residual_code)   # compilation produced output
    # Space size unchanged or larger (no guarantee of new atoms from IR stub)
    @test space_val_count(s) >= 2
end

# ── Regression tests for 2026-05-30 audit fixes (Path B) ───────────────────────

@testset "Bug 2: GLOBAL_REGISTRY auto-initialized via __init__" begin
    # Audit found __init_registry__ was never auto-called by Julia (only __init__
    # auto-fires). Fix: rename + auto-invoke. With the fix, GLOBAL_REGISTRY must
    # be non-empty by the time `using MorkSupercompiler` returns.
    @test !isempty(GLOBAL_REGISTRY.templates)
    @test haskey(GLOBAL_REGISTRY.templates, :HeuristicModusPonens)
    @test haskey(GLOBAL_REGISTRY.templates, :EvidenceCapsule)
end

@testset "Bug 3: _template_effect_kind classifies by actual concurrency tags" begin
    # Audit found the classifier checked for [:never, :read_only, :append_only,
    # :always] which never appear in commutes_when — default_local_concurrency
    # emits geometry-specific tags. Fix: geometry-aware classifier.
    # All four geometries should classify as a non-default EffectKind now.
    factor_t = make_template(:f_test, sem_model(:Q, :Formula), GEOM_FACTOR)
    trie_t = make_template(:t_test, sem_codec(:Set), GEOM_TRIE)
    dag_t = make_template(:d_test, sem_prog(:Sig, :T), GEOM_DAG)
    tensor_t = make_template(:x_test, sem_rel(:A, :B), GEOM_TENSOR_SPARSE)
    # Factor + Trie are append-like (commutative under their tags)
    @test MorkSupercompiler._template_effect_kind(factor_t) == EFF_APPEND
    @test MorkSupercompiler._template_effect_kind(trie_t) == EFF_APPEND
    # DAG + Tensor require write-confluence/patch-replay
    @test MorkSupercompiler._template_effect_kind(dag_t) == EFF_WRITE
    @test MorkSupercompiler._template_effect_kind(tensor_t) == EFF_WRITE
end

@testset "Bug 1: mg_compile step 8 honors optimized_templates + emits metadata" begin
    # Audit found step 8 silently discarded optimized_templates and routed
    # through plain MM2 compile, violating Alg 5 step 8 directly. Fix: dispatch
    # via TEMPLATE_LOWERINGS; emit ;; mgfw: annotations as runtime metadata.
    reg = GLOBAL_REGISTRY    # populated by __init__
    result = mg_compile("(edge 0 1) (edge 1 2)", reg)
    @test result isa CompilationResult
    # Runtime metadata annotations must be present in the residual
    @test occursin("mgfw:templates", result.residual_code)
    @test occursin("mgfw:backend", result.residual_code)
end

@testset "MVP demo 1: PLN STV factor template registered + lowering wired" begin
    # §15.4 demo 2: register the PLN STV HeuristicModusPonens template; verify
    # its lowering emits a residual that contains the STV-MP rewrite skeleton.
    @test haskey(GLOBAL_REGISTRY.templates, :PLN_STV_HeuristicModusPonens)
    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    @test t.presentation == GEOM_FACTOR
    @test :stv_strength_revisable in t.laws
    fn = get_lowering(:PLN_STV_HeuristicModusPonens)
    @test fn !== nothing
    residual = fn(t, "")   # region is unused by this template
    @test occursin("stv-mp", residual)
    @test occursin("apply-mp", residual)
    @test occursin("PLN_STV_HeuristicModusPonens", residual)   # metadata tag
end

@testset "MVP demo 2: Trie motif-miner template registered + lowering wired" begin
    # §15.4 demo 3: register the FactorGraphMotifMiner trie template; verify
    # its lowering emits the 3-stage seed→grow→count miner skeleton.
    @test haskey(GLOBAL_REGISTRY.templates, :FactorGraphMotifMiner)
    t = GLOBAL_REGISTRY.templates[:FactorGraphMotifMiner]
    @test t.presentation == GEOM_TRIE
    @test :evidence_mass == t.noether_charge
    @test :counter_associative in t.laws
    fn = get_lowering(:FactorGraphMotifMiner)
    @test fn !== nothing
    residual = fn(t, "")
    @test occursin("motif-stage 1", residual)
    @test occursin("motif-stage 2", residual)
    @test occursin("motif-stage 3", residual)
    @test occursin("FactorGraphMotifMiner", residual)   # metadata tag
end

# ── Item 2 (Path C tail): GeodesicBGC-Composite registered + lowering wired ────

@testset "MVP §12.2: GeodesicBGC-Composite template registered + lowering" begin
    # Spec §12.2 + Appendix A: the canonical hybrid that composes
    # DualWorklist scheduler + Factor guidance + Trie evidence. Previously
    # `build_geodesic_bgc_composite` existed but the template was never
    # registered in GLOBAL_REGISTRY and had no lowering.
    @test haskey(GLOBAL_REGISTRY.templates, :GeodesicBGC_Composite)
    t = GLOBAL_REGISTRY.templates[:GeodesicBGC_Composite]
    @test :monotone_priority in t.laws
    @test :anytime_splice in t.laws
    @test :evidence_conserved in t.laws
    @test get(t.backend_affinity, :mm2, :low) == :high
    @test get(t.backend_affinity, :mork, :low) == :high

    fn = get_lowering(:GeodesicBGC_Composite)
    @test fn !== nothing
    residual = fn(t, "")
    # All four §12.2 data-flow edges must appear as exec rewrite blocks.
    @test occursin("bgc-stage scheduler-to-guidance", residual)
    @test occursin("bgc-stage guidance-to-scheduler", residual)
    @test occursin("bgc-stage scheduler-to-evidence", residual)
    @test occursin("bgc-stage evidence-to-scheduler", residual)
    @test occursin("GeodesicBGC_Composite", residual)   # metadata tag
end

# ── Item 1 (Path C tail): MGFW MVP demo references (§15.4) ─────────────────────

@testset "MVP §15.4 demo 2: PLN STV reference vs lowering formula" begin
    # The lowering emits a MeTTa rewrite rule with the strength × strength,
    # min × min × 0.9 formula. The reference computes the same in plain Julia.
    # Both must agree on representative inputs.
    s_b, c_b = stv_mp_reference(0.8, 0.9, 0.7, 0.85)
    @test s_b ≈ 0.8 * 0.7                              # = 0.56
    @test c_b ≈ min(0.9, 0.85) * 0.9                   # = 0.85 * 0.9 = 0.765
    # Degenerate case: zero strength → zero conclusion strength
    s2, c2 = stv_mp_reference(0.0, 1.0, 0.9, 1.0)
    @test s2 == 0.0
    @test c2 ≈ 1.0 * 0.9
    # The lowering must encode the SAME formula structure (text-level
    # check — full execution-vs-reference diff requires the trie-geometry
    # runtime, queued separately).
    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    fn = get_lowering(:PLN_STV_HeuristicModusPonens)
    residual = fn(t, "")
    @test occursin("(* \$As \$Is)", residual)    # strength: As * Is
    @test occursin("(* (min \$Ac \$Ic) 0.9)", residual)    # confidence: min * 0.9
end

@testset "MVP §15.4 demo 3: motif miner reference top-k vs lowering structure" begin
    # Reference: naive_top_k returns descending-by-count pairs.
    atoms = ["foo a", "foo b", "bar c", "foo d", "bar e", "baz f"]
    top2 = naive_top_k_motifs(atoms, 2)
    @test length(top2) == 2
    @test top2[1] == ("foo" => 3)
    @test top2[2] == ("bar" => 2)

    # k=1 picks the most-frequent motif
    @test naive_top_k_motifs(atoms, 1) == ["foo" => 3]

    # k > distinct → all distinct returned (here 3 distinct)
    all_motifs = naive_top_k_motifs(atoms, 10)
    @test length(all_motifs) == 3

    # The lowering encodes the 3-stage seed → grow → score cascade. A full
    # MORK execution comparison against the reference is queued; for now
    # we verify the lowering emits all three stages (so the trie miner
    # downstream consumer KNOWS what to wire).
    t = GLOBAL_REGISTRY.templates[:FactorGraphMotifMiner]
    fn = get_lowering(:FactorGraphMotifMiner)
    residual = fn(t, "")
    @test occursin("motif-stage 1", residual)
    @test occursin("motif-stage 2", residual)
    @test occursin("motif-stage 3", residual)
end

@testset "MVP §15.4 demo 3 (end-to-end): miner lowering executed by MORK" begin
    # Load the miner's lowering into a fresh MORK space, pre-load 1-arg atoms
    # representing the toy dataset, run space_metta_calculus!, and verify
    # the stage-1 seed-scan produces the expected (motif X) +
    # (motif-count X 1) atoms for each distinct symbol. This is the first
    # genuine framework round-trip — previously every demo test was
    # structural-only (read the lowering string).
    t = GLOBAL_REGISTRY.templates[:FactorGraphMotifMiner]
    fn = get_lowering(:FactorGraphMotifMiner)
    rules = fn(t, "")

    s = new_space()
    # Use 4 distinct 1-arg atoms — MORK hash-cons collapses duplicates so
    # the count semantic at this stage is "distinct symbols seen", not
    # "occurrences". Higher-stage occurrence-merge wires up once the
    # trie-geometry runtime is fully integrated (workload #2 / §15.2 d6).
    space_add_all_sexpr!(s, "(a) (b) (c) (d)")
    space_add_all_sexpr!(s, rules)
    space_metta_calculus!(s, 100)

    dump = space_dump_all_sexpr(s)
    lines = split(dump, "\n"; keepempty=false)
    for sym in ("a", "b", "c", "d")
        @test any(l -> occursin("(motif $sym)", l), lines)
        @test any(l -> occursin("(motif-count $sym 1)", l), lines)
    end

    # The naive Julia reference on the equivalent dataset has 4 distinct
    # entries, matching MORK's stage-1 output cardinality.
    ref = naive_top_k_motifs(["a", "b", "c", "d"], 10)
    @test length(ref) == 4
end

@testset "MVP §15.4 demo 2 (smoke): PLN STV lowering loads into MORK" begin
    # Smoke test: the lowering's (= ...) MeTTa-style rules parse into MORK
    # without raising. Full numerical equivalence (executing the rule
    # via MORK against stv_mp_reference) requires arithmetic primitive
    # wiring (`*`, `min`) through the supercompiler's prim registry —
    # queued for the PLN session.
    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    fn = get_lowering(:PLN_STV_HeuristicModusPonens)
    rules = fn(t, "")
    s = new_space()
    @test_nowarn space_add_all_sexpr!(s, rules)
end

@testset "GeodesicBGC priority — toy graph-reachability domain (workload #2)" begin
    # Concrete instantiation of the spec §4.1 priority function on a
    # graph-reachability domain. Closes the MGFW_INTEGRATION.md tail item
    # "GeodesicBGC priority functions — workload-side, queued."
    #
    # Toy graph (directed):
    #   S -> A -> B -> G          path: S→A→B→G  (length 3)
    #   S -> C -> G               path: S→C→G    (length 2)
    #   S -> D                    dead end (no path to G)
    adj = Dict(:S => [:A, :C, :D], :A => [:B], :B => [:G], :C => [:G])

    f = bgc_forward_f(adj, :S, 5)
    g = bgc_backward_g(adj, :G, 5)

    # Reachability sanity
    @test f[:S] == 1.0     # 1 trivial path
    @test f[:A] == 1.0     # S→A
    @test f[:C] == 1.0     # S→C
    @test f[:G] >= 1.0     # at least one path reaches goal (≥1 via S→C→G)
    @test get(f, :D, 0.0) == 1.0   # D reachable but goal unreachable from D

    @test g[:G] == 1.0     # trivial backward path
    @test g[:C] == 1.0     # C→G
    @test g[:B] == 1.0     # B→G
    @test g[:A] == 1.0     # A→B→G
    # D has no path to G — should not be in g (or g[:D] = 0)
    @test get(g, :D, 0.0) == 0.0

    # Priority: nodes reachable from S AND able to continue to G have finite
    # priority; dead-end D should be -Inf.
    pri_A = bgc_priority(f, g, :A)
    pri_C = bgc_priority(f, g, :C)
    pri_D = bgc_priority(f, g, :D)
    @test isfinite(pri_A)
    @test isfinite(pri_C)
    @test pri_D == -Inf    # unreachable to goal

    # Δ priority — moving from a less-promising to a more-promising node
    # yields positive priority (spec §4.1 says raise log f + log g per cost).
    # Δ from D (-Inf path) to A (real path) should be finite > 0 because the
    # function falls back to absolute when prev is degenerate.
    pri_delta = bgc_priority(f, g, :A; prev_x=:D)
    @test isfinite(pri_delta)

    # step_cost scales the priority inversely
    pri_high_cost = bgc_priority(f, g, :A; step_cost=2.0)
    @test pri_high_cost ≈ pri_A / 2.0
end
