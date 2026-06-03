using Test, MorkSupercompiler, MORK

# All tests run single-node (MPI not active).
# Topology 2 single-node path: sharded_add! → local shard, no routing overhead.
# MPI path is validated structurally (shard_owner, SHARD_ATOM_TAG constants).

# ── Constructor ───────────────────────────────────────────────────────────────

@testset "new_sharded_space — default policy" begin
    ss = new_sharded_space("test-kb")
    @test ss.name == "test-kb"
    @test ss.policy == :hash_mod
    @test ss isa ShardedSpace
    @test sharded_val_count(ss) == 0
end

@testset "new_sharded_space — invalid policy throws" begin
    @test_throws ErrorException new_sharded_space("bad"; policy=:prefix_range)
end

# ── sharded_add! single-node path ─────────────────────────────────────────────

@testset "sharded_add! — single-node stores locally" begin
    ss = new_sharded_space("atoms")
    sharded_add!(ss, "(edge 0 1)")
    sharded_add!(ss, "(edge 1 2)")
    sharded_add!(ss, "(edge 2 3)")
    @test sharded_val_count(ss) == 3
end

@testset "sharded_add! — multiple atoms" begin
    ss = new_sharded_space("multi")
    for i in 1:10
        sharded_add!(ss, "(fact $i)")
    end
    @test sharded_val_count(ss) == 10
end

# ── shard_owner — single node always LOCAL_PEER ──────────────────────────────

@testset "shard_owner — LOCAL_PEER on single node" begin
    ss = new_sharded_space("routing")
    # MPI not active → always LOCAL_PEER
    @test shard_owner(ss, "(edge 0 1)") == LOCAL_PEER
    @test shard_owner(ss, "(fact hello)") == LOCAL_PEER
    @test shard_owner(ss, UInt8[0x01, 0x02]) == LOCAL_PEER
end

# ── sharded_flush! — no-op without MPI ───────────────────────────────────────

@testset "sharded_flush! — zero without MPI" begin
    ss = new_sharded_space("flush-test")
    @test sharded_flush!(ss) == 0
end

# ── sharded_query — single-node path ─────────────────────────────────────────

@testset "sharded_query — finds atoms in local shard" begin
    ss = new_sharded_space("query-test")
    sharded_add!(ss, "(edge 0 1)")
    sharded_add!(ss, "(edge 1 2)")
    sharded_add!(ss, "(node 0)")
    sharded_add!(ss, "(node 1)")

    results = sharded_query(ss, raw"(edge $x $y)")
    @test length(results) == 2

    node_results = sharded_query(ss, raw"(node $x)")
    @test length(node_results) == 2
end

@testset "sharded_query — no matches returns empty" begin
    ss = new_sharded_space("empty-query")
    sharded_add!(ss, "(fact a)")
    results = sharded_query(ss, raw"(nonexistent $x)")
    @test isempty(results)
end

# ── sharded_val_count — uses Allreduce (falls back to local on single-node) ──

@testset "sharded_val_count — single-node = local count" begin
    ss = new_sharded_space("count-test")
    sharded_add!(ss, "(a 1) (a 2) (a 3) (a 4) (a 5)")
    @test sharded_val_count(ss) == 5
end

# ── MPI collective stubs — no-ops without init ───────────────────────────────

@testset "mpi_allreduce_sum — identity without MPI" begin
    @test mpi_allreduce_sum(Int64(42)) == 42
    @test mpi_allreduce_sum(Int64(0)) == 0
end

@testset "mpi_allgatherv_strings — identity without MPI" begin
    strs = ["hello", "world"]
    @test mpi_allgatherv_strings(strs) == strs
end

@testset "mpi_bcast_bytes! — identity without MPI" begin
    buf = UInt8[0x01, 0x02, 0x03]
    result = mpi_bcast_bytes!(buf)
    @test result == UInt8[0x01, 0x02, 0x03]
end

@testset "SHARD_ATOM_TAG constant" begin
    @test SHARD_ATOM_TAG == Int32(44)
    @test SHARD_ATOM_TAG != TRAVERSE_TAG
end

# ── Topology 2 vs Topology 3 distinction ─────────────────────────────────────

@testset "Topology 2 vs 3 — ShardedSpace is one logical space" begin
    # Topology 2: one ShardedSpace across N peers
    ss = new_sharded_space("global-kb")
    sharded_add!(ss, "(entity A)")
    sharded_add!(ss, "(entity B)")
    sharded_add!(ss, "(entity C)")
    @test sharded_val_count(ss) == 3

    # All atoms are queryable as one logical space
    results = sharded_query(ss, raw"(entity $x)")
    @test length(results) == 3

    # Topology 3: separate SpaceRegistry entries per peer
    reg = SpaceRegistry()
    s1 = new_space!(reg, "peer-0-kb", :app)
    s2 = new_space!(reg, "peer-1-kb", :app)
    space_add_all_sexpr!(s1, "(entity A)")
    space_add_all_sexpr!(s2, "(entity B)")
    # Each registry space is independent — not the same logical space
    @test space_val_count(s1) == 1
    @test space_val_count(s2) == 1
end
