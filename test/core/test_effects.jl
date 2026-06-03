using Test
using MorkSupercompiler

@testset "Effects — Algorithm 1 EffectCommutes (§4.2)" begin
    sp = DEFAULT_SPACE
    sp2 = SpaceID(:other)

    @test commutes(PURE, ReadEffect(sp))
    @test commutes(ReadEffect(sp), PURE)
    @test commutes(ReadEffect(sp), ReadEffect(sp))
    @test commutes(ReadEffect(sp), ReadEffect(sp2))
    @test commutes(ReadEffect(sp), ObserveEffect(sp))
    @test commutes(ObserveEffect(sp), ObserveEffect(sp))
    @test commutes(AppendEffect(sp), AppendEffect(sp2))  # diff resource
    @test !commutes(AppendEffect(sp), AppendEffect(sp))  # same resource
    @test !commutes(AppendEffect(sp), ReadEffect(sp))    # Read sees Append
    @test commutes(AppendEffect(sp), ObserveEffect(sp))  # Observe doesn't
    @test !commutes(WriteEffect(sp), WriteEffect(sp))
    @test !commutes(WriteEffect(sp), ReadEffect(sp))
    @test commutes(PURE, WriteEffect(sp))                # Pure commutes with everything
end

@testset "Effects — sink-free checks (§4.3)" begin
    sp = DEFAULT_SPACE
    @test is_sink_free([ReadEffect(sp), AppendEffect(sp)])
    @test sink_free_check([ReadEffect(sp), AppendEffect(sp)]) === nothing
    @test !is_sink_free([ReadEffect(sp), DeleteEffect(sp)])
    @test sink_free_check([WriteEffect(sp)]) isa String
end

@testset "Effects — MORK sources all commute (free reorder justification)" begin
    e1 = mork_source_effects()
    e2 = mork_source_effects()
    @test commutes_all(e1, e2)
end
