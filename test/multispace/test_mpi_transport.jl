using Test, MorkSupercompiler, MORK

# ── Single-node MPI stubs (no mpirun needed) ──────────────────────────────────

@testset "mpi_active — false without MPI init" begin
    @test mpi_active() == false
    @test mpi_rank() == LOCAL_PEER
    @test mpi_nranks() == Int32(1)
end

@testset "LOCAL_PEER constant" begin
    @test LOCAL_PEER == Int32(0)
    @test isa(LOCAL_PEER, Int32)
end

# ── NamedSpaceID with peer_id ─────────────────────────────────────────────────

@testset "NamedSpaceID — peer_id field" begin
    a = NamedSpaceID("my-kb")
    @test a.peer_id == LOCAL_PEER
    @test a.name == "my-kb"

    b = NamedSpaceID("my-kb", UInt64(0), Int32(3))
    @test b.peer_id == Int32(3)

    # Same name + same peer = equal
    c = NamedSpaceID("my-kb", UInt64(0), LOCAL_PEER)
    @test a == c
    @test hash(a) == hash(c)

    # Same name + different peer = NOT equal (two peers can own same-named space)
    @test a != b
    @test hash(a) != hash(b)
end

# ── SpaceRegistry rank/nranks ─────────────────────────────────────────────────

@testset "SpaceRegistry — rank/nranks default" begin
    reg = SpaceRegistry()
    @test reg.rank == LOCAL_PEER
    @test reg.nranks == Int32(1)
end

# ── enable_multi_space! use_mpi=false (no MPI init) ──────────────────────────

@testset "enable_multi_space! — use_mpi=false stays single-node" begin
    enable_multi_space!(false)
    enable_multi_space!(true; use_mpi=false)
    @test ENABLE_MULTI_SPACE[] == true
    @test mpi_active() == false   # no MPI init without use_mpi=true
    enable_multi_space!(false)
end

# ── mpi_* no-ops when not initialized ────────────────────────────────────────

@testset "MPI ops are no-ops without init" begin
    # send to rank 1 when MPI not initialized — should silently do nothing
    @test mpi_send_traverse!(Int32(1), UInt8[0x01, 0x02]) === nothing
    @test mpi_poll_traverse!() === nothing
    @test mpi_broadcast_traverse!(UInt8[0x01]) === nothing
    @test mpi_barrier!() === nothing
end

# ── space_traverse! — Stage 1 unchanged with MPI off ────────────────────────

@testset "space_traverse! — MPI off, Stage 1 behavior preserved" begin
    s = new_space()
    space_add_all_sexpr!(s, "(edge 0 1) (edge 1 2) (edge 2 3)")

    result = space_traverse!(s, raw"(edge $x $y)")
    @test result.activated == true
    @test result.count == 3
    @test result.p_traverse >= 0.3

    result2 = space_traverse!(s, raw"(nonexistent $x)")
    @test result2.activated == false
    @test result2.count == 0
end

# ── process_mpi_traversals! — no-op without MPI ──────────────────────────────

@testset "process_mpi_traversals! — zero without MPI" begin
    s = new_space()
    space_add_all_sexpr!(s, "(fact a) (fact b)")
    n = process_mpi_traversals!(s)
    @test n == 0
end

# ── SPMD peer model documentation test ───────────────────────────────────────

@testset "SPMD peer model — rank semantics" begin
    reg = SpaceRegistry()
    # On a single-node run, rank 0 owns all spaces
    s = new_space!(reg, "local-0", :app)
    id = NamedSpaceID("local-0")
    @test id.peer_id == LOCAL_PEER

    # Remote space reference (peer 3 owns "remote-kb")
    remote_id = NamedSpaceID("remote-kb", UInt64(0), Int32(3))
    @test remote_id.peer_id == Int32(3)
    @test remote_id != id
end
