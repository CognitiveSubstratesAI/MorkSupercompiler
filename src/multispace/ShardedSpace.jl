"""
ShardedSpace — Topology 2: single logical MORK space distributed across MPI peers.

Contrast with Topology 3 (peer-to-peer, different knowledge per node):
  Topology 2: ONE logical space, atoms partitioned across N peers
  Topology 3: N independent spaces, each peer owns different knowledge

Sharding policy: :hash_mod (default)
  atom_rank = hash(encode(atom)) % nranks
  Uniform distribution regardless of atom content.
  Alternative policy :prefix_range is future work (arity-prefix partitioning).

Query execution (sharded_query):
  1. Initiating rank broadcasts pattern to all peers  (MPI.Bcast)
  2. Each peer queries its local shard independently
  3. All results gathered back to every rank          (MPI.Allgatherv)

Write execution (sharded_add!):
  Single-node: atoms stored locally (MPI not active — zero overhead).
  Multi-node:  atoms routed to the correct rank via MPI point-to-point.
               If dest == local rank: store directly.
               Else: MPI.Isend to owner rank.
  Owner rank must call sharded_flush! to receive and commit incoming atoms.

Atom count (sharded_val_count):
  Global count = MPI.Allreduce(SUM, local_count)

Scale: single-rank on laptop → 9216 ranks on Frontier, same binary.
"""

# ── ShardedSpace struct ────────────────────────────────────────────────────────

"""
    ShardedSpace

A logical MORK space distributed across MPI peers.

local_shard  — this rank's portion of the full space
name         — logical name of the full distributed space
policy       — sharding policy (:hash_mod only in Stage 2)
incoming_buf — atoms received from remote peers, pending commit
"""
mutable struct ShardedSpace
    local_shard::Space
    name::String
    policy::Symbol
    incoming_buf::Vector{String}   # pending atoms from remote peers
end

const SHARD_ATOM_TAG = Int32(44)     # MPI tag for atom routing messages

# ── Constructor ───────────────────────────────────────────────────────────────

"""
    new_sharded_space(name; policy=:hash_mod) → ShardedSpace

Create a new sharded space. Call on EVERY rank (SPMD).
Each rank gets an empty local shard — atoms are added via sharded_add!.
"""
function new_sharded_space(name::AbstractString; policy::Symbol=:hash_mod)::ShardedSpace
    policy == :hash_mod || error("Only :hash_mod sharding supported in Stage 2")
    ShardedSpace(new_space(), String(name), policy, String[])
end

# ── Shard routing ─────────────────────────────────────────────────────────────

"""
    shard_owner(ss, atom_bytes) → Int32

Determine which MPI rank owns `atom_bytes` under :hash_mod policy.
Returns LOCAL_PEER (0) when MPI not active.
"""
function shard_owner(ss::ShardedSpace, atom_bytes::Vector{UInt8})::Int32
    mpi_active() || return LOCAL_PEER
    Int32(hash(atom_bytes) % UInt64(mpi_nranks()))
end

function shard_owner(ss::ShardedSpace, atom_str::AbstractString)::Int32
    shard_owner(ss, collect(codeunits(atom_str)))
end

# ── Write path ────────────────────────────────────────────────────────────────

"""
    sharded_add!(ss, atom_str) → Nothing

Add an atom to the sharded space.

Single-node (MPI not active): always store locally — zero overhead.
Multi-node:

  - If this rank owns the atom: store in local_shard directly.
  - Else: route to owner rank via non-blocking MPI send.
    Owner must call sharded_flush!() to receive and commit.
"""
function sharded_add!(ss::ShardedSpace, atom_str::AbstractString)::Nothing
    if !mpi_active()
        space_add_all_sexpr!(ss.local_shard, String(atom_str))
        return nothing
    end

    owner = shard_owner(ss, atom_str)
    if owner == mpi_rank()
        space_add_all_sexpr!(ss.local_shard, String(atom_str))
    else
        # Route to owner — non-blocking send
        atom_bytes = Vector{UInt8}(atom_str)
        MPI.Isend(atom_bytes, _MPI_COMM[]; dest=Int(owner), tag=Int(SHARD_ATOM_TAG))
    end
    nothing
end

"""
    sharded_flush!(ss) → Int

Receive and commit all pending atoms sent from remote peers.
Call periodically in the peer main loop.
Returns the number of atoms committed.
"""
function sharded_flush!(ss::ShardedSpace)::Int
    mpi_active() || return 0
    count = 0
    while true
        status = MPI.Iprobe(_MPI_COMM[]; source=MPI.ANY_SOURCE, tag=Int(SHARD_ATOM_TAG))
        status === nothing && break
        src = MPI.Get_source(status)
        nbytes = MPI.Get_count(status, UInt8)
        buf = Vector{UInt8}(undef, nbytes)
        MPI.Recv!(buf, _MPI_COMM[]; source=src, tag=Int(SHARD_ATOM_TAG))
        space_add_all_sexpr!(ss.local_shard, String(buf))
        count += 1
    end
    count
end

# ── Read path ─────────────────────────────────────────────────────────────────

"""
    sharded_query(ss, pattern_str) → Vector{String}

Query the full distributed space for `pattern_str`.

Protocol:

 1. Rank 0 (or any rank) broadcasts the pattern to all peers.
 2. Each peer queries its local shard.
 3. All results gathered via MPI.Allgatherv → every rank gets full results.

Single-node: queries local shard directly, no MPI overhead.
"""
function sharded_query(ss::ShardedSpace, pattern_str::AbstractString)::Vector{String}
    if !mpi_active()
        # Single-node: query local shard directly
        return _local_query(ss.local_shard, pattern_str)
    end

    # Step 1: broadcast pattern from rank 0 to all peers
    pat_bytes = Vector{UInt8}(pattern_str)
    mpi_bcast_bytes!(pat_bytes, LOCAL_PEER)
    pat_str = String(pat_bytes)

    # Step 2: each rank queries its local shard
    local_results = _local_query(ss.local_shard, pat_str)

    # Step 3: gather results from all ranks
    mpi_allgatherv_strings(local_results)
end

"""
    sharded_val_count(ss) → Int

Total number of atoms across all shards.
Single-node: local count. Multi-node: MPI.Allreduce(SUM).
"""
function sharded_val_count(ss::ShardedSpace)::Int
    local_n = space_val_count(ss.local_shard)
    Int(mpi_allreduce_sum(Int64(local_n)))
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _local_query(s::Space, pattern_str::AbstractString)::Vector{String}
    results = String[]
    nodes = parse_program(pattern_str)
    isempty(nodes) && return results
    pat = only(nodes)
    pat_str = sprint_sexpr(pat)
    # Build a one-shot exec rule to collect matches
    prog = "(exec 0 (, $pat_str) (, (__sq_hit__ \$__v__)))"
    s_tmp = new_space()
    space_add_all_sexpr!(s_tmp, space_dump_all_sexpr(s))
    space_add_all_sexpr!(s_tmp, prog)
    space_metta_calculus!(s_tmp, typemax(Int))
    dump = space_dump_all_sexpr(s_tmp)
    for line in split(dump, "\n")
        startswith(line, "(__sq_hit__") && push!(results, line)
    end
    results
end

# ── Export ────────────────────────────────────────────────────────────────────

export ShardedSpace, new_sharded_space
export sharded_add!, sharded_flush!, sharded_query, sharded_val_count
export shard_owner, SHARD_ATOM_TAG
