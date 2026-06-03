"""
Traverse — fruit-fly-inspired traversal primitive for multi-space queries.

Inspired by information flow in the Drosophila central brain (Lappalainen et al. 2024):
  traversal_probability = incoming_synapses_from_set / total_incoming_synapses

In MORK:
  traversal_probability = cardinality(seed_pattern) / total_atoms(space)

When p < threshold (default 0.3, from the Drosophila paper), the space is
NOT traversed — sparse activation, like neurons outside the traversed set.

Stage 1: single-space traversal with probability gate.
Stage 2 (HPC): propagate to peer spaces with locality weight.
Stage 3 (Web3): propagate globally with CID-addressed routing.
"""

"""
    TRAVERSAL_THRESHOLD

Default traversal probability threshold (0.3) from the Drosophila connectome paper:
"Neurons are traversed probabilistically according to the ratio of incoming synapses
from neurons that are in the traversed set." — Lappalainen et al. 2024, Fig. 6.

Below this threshold, a space is not traversed — sparse activation.
"""
const TRAVERSAL_THRESHOLD = 0.3

"""
    TraversalResult

Result of a `space_traverse!` call.

count      — number of matches found
p_traverse — traversal probability (cardinality / total_atoms)
activated  — whether the space was actually traversed (p ≥ threshold)
rank       — traversal depth (0 = seed, 1 = first hop, etc.)
"""
struct TraversalResult
    count::Int
    p_traverse::Float64
    activated::Bool
    rank::Int
end

"""
    space_traverse!(space, seed, depth=1; threshold, on_match, dest_peer) → TraversalResult

Fruit-fly-inspired traversal of `space` starting from `seed` pattern.

Computes traversal probability = cardinality(seed) / total_atoms.
If p < threshold (0.3), returns immediately — sparse activation gate.

Stage 1 (single-node): local traversal only.
Stage 2 (MPI peers):   if mpi_active() and p ≥ threshold, propagate seed to
remote peer spaces via non-blocking MPI.Isend.
Each peer independently decides whether to activate.

`depth` controls MPI hop count:
depth=1 — local + direct peer neighbours
depth=N — N-hop propagation (each peer re-traverses with depth-1)

`dest_peer`:
LOCAL_PEER (default) — local + broadcast to all MPI peers when depth > 0
specific Int32 rank  — send only to that peer (point-to-point)
"""
function space_traverse!(
    space::Space,
    seed::SNode,
    depth::Int=1;
    threshold::Float64=TRAVERSAL_THRESHOLD,
    on_match::Function=(b, e) -> true,
    dest_peer::Int32=LOCAL_PEER
)::TraversalResult
    total = Float64(max(1, space_val_count(space)))
    n_match = dynamic_count(space.btm, seed)
    p = Float64(n_match) / total

    # Sparse activation gate — mirrors Drosophila 0.3 synapse ratio
    p < threshold && return TraversalResult(0, p, false, 0)

    # ── Stage 2: MPI peer propagation ─────────────────────────────────────────
    if mpi_active() && depth > 0
        query_bytes = Vector{UInt8}(sprint_sexpr(seed))
        if dest_peer == LOCAL_PEER
            mpi_broadcast_traverse!(query_bytes)
        else
            mpi_send_traverse!(dest_peer, query_bytes)
        end
    end

    # ── Local on_match callback ────────────────────────────────────────────────
    if on_match !== ((b, e) -> true)
        seed_str = sprint_sexpr(seed)
        prog = "(exec 0 (, $seed_str) (, (__traverse_hit__ \$__v__)))"
        s_tmp = new_space()
        space_add_all_sexpr!(s_tmp, space_dump_all_sexpr(space))
        space_add_all_sexpr!(s_tmp, prog)
        space_metta_calculus!(s_tmp, typemax(Int))
    end

    TraversalResult(n_match, p, true, 0)
end

"""
    space_traverse!(space, seed_str; ...) → TraversalResult

Convenience overload accepting a seed as an s-expression string.
Also used to process incoming MPI traverse requests: pass the string
received from `mpi_poll_traverse!()` with `depth=0` to prevent re-broadcast.
"""
function space_traverse!(
    space::Space,
    seed_str::AbstractString,
    depth::Int=1;
    threshold::Float64=TRAVERSAL_THRESHOLD,
    on_match::Function=(b, e) -> true,
    dest_peer::Int32=LOCAL_PEER
)::TraversalResult
    nodes = parse_program(seed_str)
    isempty(nodes) && return TraversalResult(0, 0.0, false, 0)
    space_traverse!(
        space,
        only(nodes),
        depth;
        threshold=threshold,
        on_match=on_match,
        dest_peer=dest_peer
    )
end

"""
    process_mpi_traversals!(space; threshold) → Int

Poll and process all pending MPI traverse requests from peer nodes.
Returns number of requests handled. Non-blocking — zero overhead when no messages.

Call from the peer main loop:
while true
space_metta_calculus!(s, 100)
process_mpi_traversals!(s)
end
"""
function process_mpi_traversals!(space::Space; threshold::Float64=TRAVERSAL_THRESHOLD)::Int
    mpi_active() || return 0
    count = 0
    while true
        msg = mpi_poll_traverse!()
        msg === nothing && break
        _, query_bytes = msg
        # depth=0 prevents infinite re-broadcast across peers
        space_traverse!(space, String(query_bytes), 0; threshold=threshold)
        count += 1
    end
    count
end

export TRAVERSAL_THRESHOLD, TraversalResult, space_traverse!, process_mpi_traversals!
