"""
Effects — formal effect algebra for the MM2 supercompiler.

Implements §4 of the MM2 Supercompiler spec (Goertzel, Oct 2025):
  §4.1  Effect algebra: Read/Write/Append/Create/Delete/Observe/Pure
  §4.2  Algorithm 1 — EffectCommutes (pairwise commutativity)
  §4.3  Sink-free semantics: four conditions for safe speculation

Key axioms from §4.2 (verbatim from spec):
  Pure commutes with everything.
  Read / Observe are mutually commutative.
  Append(r1) commutes with Append(r2) iff r1 ≠ r2.
  Append(r) does NOT commute with Read(r)  [Read sees Appends].
  Append(r) DOES commute with Observe(r)   [Observe doesn't].
  Write never commutes with itself or anything else on same resource.

The `EffectSet` bitmask in MCore.jl uses these bit positions:
  bit 0 (0x01) = READ
  bit 1 (0x02) = WRITE
  bit 2 (0x04) = APPEND
  bit 3 (0x08) = CREATE
  bit 4 (0x10) = DELETE
  bit 5 (0x20) = OBSERVE
  bit 6 (0x40) = PURE (unused; absence of other bits = Pure)
"""

# ── §4.1 Effect types ─────────────────────────────────────────────────────────

"""
Effect algebra from §4.1.
"""
abstract type Effect end

struct ReadEffect <: Effect
    ;
    resource::SpaceID
end
struct WriteEffect <: Effect
    ;
    resource::SpaceID
end
struct AppendEffect <: Effect
    ;
    resource::SpaceID
end
struct CreateEffect <: Effect
    ;
    resource::SpaceID
end
struct DeleteEffect <: Effect
    ;
    resource::SpaceID
end
struct ObserveEffect <: Effect
    ;
    resource::SpaceID
end
struct PureEffect <: Effect end

const PURE = PureEffect()

# ── Algorithm 1 — EffectCommutes (§4.2) ──────────────────────────────────────

"""
    commutes(e1::Effect, e2::Effect) -> Bool

Algorithm 1 (EffectCommutes) from §4.2.  Returns true iff operations with
effects `e1` and `e2` can be safely reordered.

Axioms (verbatim from spec):
(Pure, _)                    → true
(_, Pure)                    → true
(Read(r1), Read(r2))         → true
(Read(r1), Observe(r2))      → true
(Observe(r1), Observe(r2))   → true
(Append(r1), Append(r2))     → r1 ≠ r2
(Append(r), Read(r))         → false  [Read sees Appends]
(Append(r), Observe(r))      → true   [Observe doesn't]
(Write(r1), Write(r2))       → false
(Write(r), _)                → false  [Writes don't commute]
_                            → false
"""
function commutes(e1::Effect, e2::Effect)::Bool
    e1 isa PureEffect && return true
    e2 isa PureEffect && return true
    e1 isa ReadEffect && e2 isa ReadEffect && return true
    e1 isa ReadEffect && e2 isa ObserveEffect && return true
    e1 isa ObserveEffect && e2 isa ReadEffect && return true
    e1 isa ObserveEffect && e2 isa ObserveEffect && return true

    if e1 isa AppendEffect && e2 isa AppendEffect
        return e1.resource != e2.resource
    end
    if e1 isa AppendEffect && e2 isa ReadEffect
        return e1.resource != e2.resource   # false if same resource
    end
    if e1 isa ReadEffect && e2 isa AppendEffect
        return e1.resource != e2.resource
    end
    if e1 isa AppendEffect && e2 isa ObserveEffect
        return true   # Observe doesn't see Appends
    end
    if e1 isa ObserveEffect && e2 isa AppendEffect
        return true
    end
    if e1 isa WriteEffect || e2 isa WriteEffect
        return false   # Write never commutes
    end
    false
end

"""
    commutes_all(effects1::Vector{Effect}, effects2::Vector{Effect}) -> Bool

Return true iff ALL pairs (e1, e2) from the two effect sets commute.
Used for reordering two groups of operations.
"""
function commutes_all(
    effects1::AbstractVector{<:Effect}, effects2::AbstractVector{<:Effect}
)::Bool
    for e1 in effects1, e2 in effects2
        commutes(e1, e2) || return false
    end
    true
end

# ── §4.3 Sink-Free Semantics ──────────────────────────────────────────────────

"""
    is_sink_free(effects::AbstractVector{<:Effect}) -> Bool

§4.3: A program is sink-free iff:

 1. No Delete effects
 2. All Write effects are idempotent (can be modeled as Append)
 3. State changes are monotonic additions to multiset-valued stores
 4. Guards are antimonotonic: if g holds in state s, holds in all s' ⊇ s

This function checks condition (1): no Delete effects.
Condition (2): use `has_only_idempotent_writes`.
Conditions (3) and (4) are semantic and checked at program level.
"""
is_sink_free(effects::AbstractVector{<:Effect})::Bool =
    !any(e -> e isa DeleteEffect, effects)

"""
Return true if all Write effects in the list are idempotent (condition 2).
"""
has_only_idempotent_writes(effects::AbstractVector{<:Effect})::Bool =
    all(e -> !(e isa WriteEffect), effects)   # simplification: treat Write as non-idempotent

"""
    sink_free_check(effects::AbstractVector{<:Effect}) -> Union{Nothing, String}

Returns nothing if effects satisfy sink-free conditions (1) + (2),
or a diagnostic string explaining the violation.
"""
function sink_free_check(effects::AbstractVector{<:Effect})::Union{Nothing, String}
    for e in effects
        e isa DeleteEffect && return "Delete effect violates sink-free (condition 1)"
        e isa WriteEffect &&
            return "Write effect may violate sink-free (condition 2): check idempotency"
    end
    nothing
end

# ── Effect set for MORK exec sources ─────────────────────────────────────────

"""
    mork_source_effects(space::SpaceID=DEFAULT_SPACE) -> Vector{Effect}

All MORK exec sources have effect `Read(space)`.
Under Algorithm 1: Read(r) and Read(r) commute → sources are freely reorderable.
This is the formal justification for the QueryPlanner's free reordering.
"""
mork_source_effects(space::SpaceID=DEFAULT_SPACE)::Vector{Effect} = [ReadEffect(space)]

export Effect, ReadEffect, WriteEffect, AppendEffect, CreateEffect
export DeleteEffect, ObserveEffect, PureEffect, PURE
export commutes, commutes_all
export is_sink_free, has_only_idempotent_writes, sink_free_check
export mork_source_effects
