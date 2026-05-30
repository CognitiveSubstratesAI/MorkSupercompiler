using Test
using MorkSupercompiler

@testset "MorkSupercompiler" begin

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
