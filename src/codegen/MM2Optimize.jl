"""
MM2Optimize — MM2-specific compiler post-passes from v1 §10.6.

Three optimizations operating on `Vector{MM2ExecAtom}` (the output of
MM2Compiler.compile_program):

  - [`schedule_static`](@ref) — v1 §10.6 Algorithm 11 `StaticScheduleMM2`.
    Sort exec atoms by priority; when all priorities are compile-time
    constants (which is always true for `MM2Priority`), emit a linear
    sequence. Side-effect: makes priority ordering observable in the
    program text, not just in MORK's runtime scheduler.

  - [`batch_space_ops`](@ref) — v1 §10.6 "Space Operation Batching".
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
        inner = strip(t[3:end - 1])    # drop "(," and ")"
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

# ── Space Operation Batching (v1 §10.6, same-priority variant) ────────────────

"""
    batch_space_ops(atoms::Vector{MM2ExecAtom}) -> Vector{MM2ExecAtom}

Merge exec atoms with IDENTICAL priority by concatenating their pattern
and template comma-lists into a single exec.

Soundness: MM2 treats same-priority atoms as unordered, so merging them
into one is observably equivalent — the matching engine still applies
the same set of patterns to derive the same set of templates. The v1
paper's example shows merging across DIFFERENT priorities, but that
requires proving order-independence; we restrict to same-priority for
soundness without verification.

Returns a NEW vector. Atoms with unique priorities pass through unchanged.

# Example
```
Before:
  (exec (1 0) (, (kb fact1)) (, result1))
  (exec (1 0) (, (kb fact2)) (, result2))
  (exec (2 0) (, foo) (, bar))

After batch_space_ops:
  (exec (1 0) (, (kb fact1) (kb fact2)) (, result1 result2))
  (exec (2 0) (, foo) (, bar))
```
"""
function batch_space_ops(atoms::Vector{MM2ExecAtom})::Vector{MM2ExecAtom}
    # Group atoms by priority (preserving first-occurrence order)
    groups = Dict{MM2Priority, Vector{Int}}()
    order = MM2Priority[]
    for (i, a) in enumerate(atoms)
        if !haskey(groups, a.priority)
            groups[a.priority] = Int[]
            push!(order, a.priority)
        end
        push!(groups[a.priority], i)
    end

    out = MM2ExecAtom[]
    for pri in order
        ixs = groups[pri]
        if length(ixs) == 1
            push!(out, atoms[ixs[1]])
        else
            merged_pattern = _comma_join([_comma_inner(atoms[i].pattern) for i in ixs])
            merged_template = _comma_join([_comma_inner(atoms[i].template) for i in ixs])
            # Inherit source_node + proof_obligs from the FIRST exec in the group;
            # downstream verification (verify_bisim) checks whole-program equivalence
            # so per-atom traceability is enough.
            first = atoms[ixs[1]]
            push!(out, MM2ExecAtom(
                pri,
                merged_pattern,
                merged_template,
                first.source_node,
                first.proof_obligs
            ))
        end
    end
    out
end

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
            push!(out, MM2ExecAtom(
                first.priority,
                first.pattern,
                merged_template,
                first.source_node,
                first.proof_obligs
            ))
        end
    end
    out
end

export schedule_static, batch_space_ops, fuse_identical_patterns
