"""
MM2Optimize — MM2-specific compiler post-passes from v1 §10.6.

Three optimizations operating on `Vector{MM2ExecAtom}` (the output of
MM2Compiler.compile_program):

  - [`schedule_static`](@ref) — v1 §10.6 Algorithm 11 `StaticScheduleMM2`.
    Sort exec atoms by priority; when all priorities are compile-time
    constants (which is always true for `MM2Priority`), emit a linear
    sequence. Side-effect: makes priority ordering observable in the
    program text, not just in MORK's runtime scheduler.

  - `batch_space_ops` — REMOVED 2026-08-25: v1 §10.6 is UNSOUND, refuted by execution.
    Merge exec atoms with IDENTICAL priority by concatenating their
    pattern and template comma-lists. Safe because same-priority atoms
    are unordered in MM2 already; combining them just reduces the number
    of separate exec calls. (The v1 paper's worked example uses different
    priorities, which would require proving order-independence — we
    restrict to same-priority for soundness without verification.)

  - [`fuse_identical_patterns`](@ref) — v1 §10.6 "Pattern Fusion"
    (identical-pattern variant). When two execs share the exact same
    pattern string, merge them into one exec whose template concatenates
    both. The single pattern match drives both templates instead of two
    independent matches. The v1 paper's "shared subpatterns" generalization
    requires sub-pattern detection; we ship the strict-equality form here
    and gate the generalized form on a future workload.

All three passes preserve semantics relative to the input — verified by
the `BisimVerifier` from Boundary #3. The acceptance tests in
`test_mm2_optimize.jl` use the bisim verifier to confirm equivalence on
the v1 §10.6 worked examples.

v1 §17 originally tagged these as "natural next sprint when an algorithm
workload demands it." The bisim verifier (Boundary #3) made shipping them
without a real workload safe — equivalence to the unoptimized form is
verifiable per-test.
"""

# ── helpers — comma-list surgery on MM2 pattern/template strings ─────────────

"""
    _comma_inner(s) -> String

Strip the `(, ...)` wrapper from a MM2 comma-list and return the inner
content (whitespace-trimmed). For empty `(, )`, returns `""`.
"""
function _comma_inner(s::AbstractString)::String
    t = strip(s)
    if startswith(t, "(,")
        inner = strip(t[3:(end - 1)])    # drop "(," and ")"
        return String(inner)
    end
    return String(t)
end

"""
    _comma_wrap(s) -> String

Wrap content `s` in a MM2 comma-list `(, s)`. Empty input yields `(, )`.
"""
function _comma_wrap(s::AbstractString)::String
    t = strip(s)
    isempty(t) ? "(, )" : "(, $t)"
end

"""
    _comma_join(parts) -> String

Build a `(, p1 p2 ...)` comma-list from a list of inner content strings.
Filters out empty parts.
"""
function _comma_join(parts::Vector{String})::String
    nonempty = filter(!isempty, [strip(p) for p in parts])
    isempty(nonempty) ? "(, )" : "(, " * join(nonempty, " ") * ")"
end

# ── Algorithm 11 — StaticScheduleMM2 (v1 §10.6) ───────────────────────────────

"""
    schedule_static(atoms::Vector{MM2ExecAtom}) -> Vector{MM2ExecAtom}

Sort exec atoms by their `MM2Priority` (lex order on `(p, q)`). Because
`MM2Priority` is always compile-time constant by construction, this is
always safe to apply. Algorithm 11 from v1 §10.6.

Returns a NEW vector — input is not mutated.
"""
function schedule_static(atoms::Vector{MM2ExecAtom})::Vector{MM2ExecAtom}
    sort(atoms; by=a -> a.priority)
end

# ── Space Operation Batching (v1 §10.6) — REMOVED AS UNSOUND ──────────────────
#
# 🔴🔴 THE SPEC'S TRANSFORMATION IS UNSOUND. THIS IS A DEFECT IN THE PAPER, NOT IN THE PORT.
# `batch_space_ops` was a FAITHFUL implementation of v1 §10.6 "Space operation batching", which
# states, unconditionally:
#
#     ; Before                                  ; After
#     (exec p1 (, (kb fact1)) (, result1))      (exec p_batch
#     (exec p2 (, (kb fact2)) (, result2))        (, (kb fact1) (kb fact2) (kb fact3))
#     (exec p3 (, (kb fact3)) (, result3))        (, result1 result2 result3))
#
# `,` IN AN EXEC'S SOURCE POSITION IS A CONJUNCTION. Before the merge, `fact1` present derives
# `result1` regardless of `fact2`. After it, the merged exec fires only when ALL patterns match
# simultaneously. N independent rules become one N-way join.
#
# 🔴 REFUTED BY EXECUTION 2026-08-25, not by reading. Two same-priority execs over disjoint
# patterns, run through `verify_bisim` on the live substrate:
#
#     facts = "(a 1) (b 2)"   ->  forward_ok = true    <- the shipped fixture
#     facts = "(a 1)"         ->  forward_ok = FALSE   <- one fact removed
#
# The merged program was `(exec (1 0) (, (a $x) (b $y)) (, (seen_a $x) (seen_b $y)))` — and note
# its two conjuncts share NO variable, so the pass also EMITS the disconnected-conjunct Cartesian
# shape that MORK's own engines are slowest on.
#
# ⚠️ THE PAPER'S VERSION IS STRICTLY STRONGER THAN WHAT WAS REFUTED HERE: its example merges
# p1/p2/p3 — THREE DIFFERENT PRIORITIES — into one `p_batch`, so it additionally reorders across
# priority classes. The old implementation saw that hazard and restricted itself to same-priority
# "for soundness without verification". That guard was real but addressed the WEAKER problem, and
# supplied false confidence about the conjunction underneath it. The refutation holds either way:
# conjunction breaks it at equal priority, before reordering is even considered.
#
# STEELMAN, AND WHY IT DOES NOT RESCUE THE PASS. If `(kb fact1..3)` are all ground and all PROVEN
# present — a saturated KB — merging is sound and saves two trie descents. §10.7 immediately after
# is a logic-engine example with a pre-computed index, so a KB-saturated setting is plausibly the
# implicit context and the paper simply omits the precondition. It does not rescue anything: the
# precondition is a WHOLE-PROGRAM property this pass has no access to (it sees a Vector of exec
# atoms), it is undecidable in general, and §10.6 states the rewrite unconditionally.
#
# ✅ UPSTREAM MEANS SOMETHING ELSE ENTIRELY BY THE PHRASE. In `dev-zone/mork_ffi` (the bridge PeTTa
# actually uses), batching space operations is `queue-atom` accumulating atoms + `flush` loading
# them together — amortising LOAD cost, with the test `queued_atoms_are_loaded_together_on_flush`.
# Nothing upstream merges exec patterns. The paper reached for "batching" and applied it to an
# operation that does not support it.
#
# 🟢 §10.6's OTHER clause — PATTERN FUSION, directly below — IS SOUND, and `fuse_identical_patterns`
# implements it correctly: identical patterns match once and drive both templates. The paper is
# right in one clause and wrong in the next, which is how this survived review. Fusion is the only
# sound residue of batching, so removing `batch_space_ops` loses no capability.
#
# DO NOT REINSTATE without a simultaneous-satisfiability precondition and an oracle that can
# observe answer-set changes (the old tests asserted STRING SHAPE — `occursin("(kb fact1)", …)` —
# and structurally could not see a conjunction change).

# ── Pattern Fusion (v1 §10.6, identical-pattern variant) ──────────────────────

"""
    fuse_identical_patterns(atoms::Vector{MM2ExecAtom}) -> Vector{MM2ExecAtom}

When two exec atoms have IDENTICAL pattern strings (modulo whitespace),
merge them into one whose template is the concatenation of both
templates. The single pattern match drives both templates.

Soundness: a pattern match in MORK derives one binding; both original
templates would have used that same binding (since the patterns match).
The merged exec computes the same set of derived atoms in one pass.

Limitation: detects ONLY pattern-string equality (after whitespace
normalization). The v1 paper's "shared sub-patterns" generalization
requires sub-pattern detection that's gated on a future workload.

Returns a NEW vector. Atoms with unique patterns pass through unchanged.

Preserves the priority of the FIRST atom in each fused group.
"""
function fuse_identical_patterns(atoms::Vector{MM2ExecAtom})::Vector{MM2ExecAtom}
    # Group by normalized pattern (preserving first-occurrence order)
    groups = Dict{String, Vector{Int}}()
    order = String[]
    for (i, a) in enumerate(atoms)
        key = String(strip(a.pattern))   # whitespace-normalize
        if !haskey(groups, key)
            groups[key] = Int[]
            push!(order, key)
        end
        push!(groups[key], i)
    end

    out = MM2ExecAtom[]
    for key in order
        ixs = groups[key]
        if length(ixs) == 1
            push!(out, atoms[ixs[1]])
        else
            merged_template = _comma_join([_comma_inner(atoms[i].template) for i in ixs])
            first = atoms[ixs[1]]
            push!(
                out,
                MM2ExecAtom(
                    first.priority,
                    first.pattern,
                    merged_template,
                    first.source_node,
                    first.proof_obligs
                )
            )
        end
    end
    out
end

export schedule_static, fuse_identical_patterns
