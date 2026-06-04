using Test
using MorkSupercompiler

# Aqua is a test-only [extras] dep: present under Pkg.test/CI, but NOT in a warm
# `--project=.` REPL (the sanctioned iteration path — `julia --project=. tools/repl.jl`).
# Load it optionally so the suite runs BOTH ways; the Aqua quality testset runs only
# when Aqua is loadable. (A hard `using Aqua` here silently broke warm-REPL runs.)
const _HAS_AQUA = try
    @eval using Aqua
    true
catch
    false
end

@testset "MorkSupercompiler" begin

    if _HAS_AQUA
        @testset "Aqua quality" begin
            # deps_compat check_extras=false: [extras] are dev/test tools; runtime [deps]
            # carry [compat] (MORK/PathMap/HPC/MORKTensorNetworks dev-linked via Manifest).
            # unbound_args=false / piracies=false: dynamic MeTTa-style dispatch + Base-type
            # method extensions on substrate types flagged here are intentional.
            Aqua.test_all(
                MorkSupercompiler;
                deps_compat=(check_extras=false,),
                unbound_args=false,
                piracies=false
            )
        end
    else
        @info "Aqua not loadable (warm REPL --project=.) — Aqua quality runs under Pkg.test/CI"
    end

    # ── Frontend ──────────────────────────────────────────────────────────────
    include("frontend/test_sexpr.jl")

    # ── Planner ───────────────────────────────────────────────────────────────
    include("planner/test_selectivity.jl")
    include("planner/test_statistics.jl")
    include("planner/test_query_planner.jl")

    # ── Rewrite ───────────────────────────────────────────────────────────────
    include("rewrite/test_rewrite.jl")

    # ── Core IR & Effects ─────────────────────────────────────────────────────
    include("core/test_mcore.jl")
    include("core/test_effects.jl")

    # ── Supercompiler ─────────────────────────────────────────────────────────
    include("supercompiler/test_stepper.jl")
    include("supercompiler/test_canonical_keys.jl")
    include("supercompiler/test_bounded_split.jl")
    include("supercompiler/test_driver.jl")
    include("supercompiler/test_kb_saturation.jl")
    include("supercompiler/test_evo_specializer.jl")
    include("supercompiler/test_pipeline_decompose.jl")
    include("supercompiler/test_projection_lossless.jl")

    # ── Code Generation ───────────────────────────────────────────────────────
    include("codegen/test_mm2_compiler.jl")
    include("codegen/test_space_primitives.jl")

    # ── Integration ───────────────────────────────────────────────────────────
    include("integration/test_pipeline.jl")
    include("integration/test_profiler.jl")
    include("integration/test_explainer.jl")
    include("integration/test_adaptive_planner.jl")
    include("integration/test_mm2_roundtrip.jl")   # A-4: compiler → runtime seam

    # ── Multi-Geometry Framework (Doc 3) ─────────────────────────────────────
    include("mgfw/test_mgfw.jl")

    # ── Multi-Space (Stage 1 + Stage 2) ─────────────────────────────────────
    include("multispace/test_multispace.jl")
    include("multispace/test_mpi_transport.jl")
    include("multispace/test_sharded_space.jl")

    # ── Approximate Supercompilation (Doc 2) ──────────────────────────────────
    include("approx/test_pbox_algebra.jl")
    include("approx/test_uncertain_query.jl")
    include("approx/test_uncertain_inference.jl")
    include("approx/test_approx_moses.jl")
    include("approx/test_approx_pipeline.jl")
end

println("All tests passed ✓")
