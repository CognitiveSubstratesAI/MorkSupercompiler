"""
SemanticObjects — geometry-neutral semantic object types and presentation layer.

Implements §6.1–6.2 + §7.6–7.7 of the Multi-Geometry Hyperon Framework spec
(Goertzel, April 2026):

  §6.1  Six semantic object classes: Rel, Prog, Model, Codec, Sched, Stream
  §6.2  Geometry tags: Factor, DAG, Trie, TensorSparse, TensorDense, Hybrid
        Presentation type: Pres(G, A) — explicit, typed, shareable
  §7.6  Core type calculus (presentation-aware lambda calculus)
  §7.7  Coercions: exact (⇒) and approximate (⇒_ε)
        Four minimum registered coercions:
          T_DAG→Factor, T_Factor→Trie, T_Trie→Tensor, T_Trie→Codec

The key move (§6.2): presentations are NOT hidden backend encodings.
A semantic object's geometry is declared, typed, and visible to tools.
"""

# ── §6.1 Semantic object classes ─────────────────────────────────────────────

"""
Semantic object kind — the 6 geometry-neutral base types from §6.1.
"""
@enum SemanticKind begin
    SK_REL     # Relations and relational views
    SK_PROG    # Programs over a signature
    SK_MODEL   # Quantale- or probability-valued models
    SK_CODEC   # Reversible feature/residual structures
    SK_SCHED   # Scheduling and worklist objects
    SK_STREAM  # Streams and datasets
end

"""
    SemanticType

Typed semantic object. `args` are type parameters, e.g.:

  - `Rel(A, B)`:    kind=SK_REL,   args=[:A, :B]
  - `Prog(Σ, T)`:   kind=SK_PROG,  args=[:Sigma, :T]
  - `Model(Q, A)`:  kind=SK_MODEL, args=[:Q, :A]
  - `Codec(A)`:     kind=SK_CODEC, args=[:A]
  - `Sched(A)`:     kind=SK_SCHED, args=[:A]
  - `Stream(A)`:    kind=SK_STREAM,args=[:A]
"""
struct SemanticType
    kind::SemanticKind
    args::Vector{Symbol}
end

SemanticType(kind::SemanticKind) = SemanticType(kind, Symbol[])

# Convenience constructors matching spec notation
sem_rel(a::Symbol, b::Symbol) = SemanticType(SK_REL, [a, b])
sem_prog(sig::Symbol, t::Symbol) = SemanticType(SK_PROG, [sig, t])
sem_model(q::Symbol, a::Symbol) = SemanticType(SK_MODEL, [q, a])
sem_codec(a::Symbol) = SemanticType(SK_CODEC, [a])
sem_sched(a::Symbol) = SemanticType(SK_SCHED, [a])
sem_stream(a::Symbol) = SemanticType(SK_STREAM, [a])

Base.show(io::IO, s::SemanticType) = print(io, "$(s.kind)($(join(s.args, ", ")))")

# ── §6.2 Geometry tags ────────────────────────────────────────────────────────

"""
    GeomTag

Geometry tags from §6.2.  A semantic object's presentation geometry.

FACTOR        — factor graph (PLN inference, forward/backward message passing)
DAG           — hash-consed canonical DAG (MOSES programs, gCoDD/ENF/CENF)
TRIE          — prefix trie / PathMap (MORK-Miner, WILLIAM compression)
TENSOR_SPARSE — sparse semiring tensor (relation joins, restriction, projection)
TENSOR_DENSE  — dense neural tensor shard (GPU attention, embedding)
"""
@enum GeomTag begin
    GEOM_FACTOR
    GEOM_DAG
    GEOM_TRIE
    GEOM_TENSOR_SPARSE
    GEOM_TENSOR_DENSE
end

"""
    HybridGeom

Composite of multiple geometry tags (e.g., DualWorklist + Factor + Trie).
Used in GeometryTemplate :presentation field for multi-geometry composites.
"""
struct HybridGeom
    components::Vector{GeomTag}
end
HybridGeom(g1::GeomTag, g2::GeomTag) = HybridGeom([g1, g2])
HybridGeom(g1::GeomTag, g2::GeomTag, g3::GeomTag) = HybridGeom([g1, g2, g3])

# ── §6.2 Presentation type Pres(G, A) ─────────────────────────────────────────

"""
    PresType

Presentation type from §6.2: `Pres(G, A)` where G ∈ Geom, A is a SemanticType.

This is the **key innovation** of the MG framework: presentations are first-class,
typed, and shareable — not hidden backend encodings. Every method that works on
a semantic object must declare which geometry it expects.

Examples from spec §6.2:
Pres(Factor, Model(Q, Formula))     — PLN inference
Pres(DAG,    Prog(Σ, T))           — MOSES / gCoDD programs
Pres(Trie,   Set(Motif(A)))        — pattern mining / compression
Pres(TensorSparse, Rel(A, B))      — semiring tensor logic
"""
struct PresType
    geometry::GeomTag
    sem_type::SemanticType
end
PresType(g::GeomTag, kind::SemanticKind, args::Symbol...) =
    PresType(g, SemanticType(kind, collect(args)))

Base.show(io::IO, p::PresType) = print(io, "Pres($(p.geometry), $(p.sem_type))")

# ── §7.6 Core type calculus ───────────────────────────────────────────────────

"""
    MGType

Core type from the presentation-aware lambda calculus (§7.6):
A, B ::= 1 | 0 | Base(σ) | A×B | A+B | A→B |
Rel(A,B) | Prog(Σ,T) | Model(Q,A) | Codec(A) | Pres(G,A)
"""
abstract type MGType end

struct MGUnit <: MGType end                           # 1
struct MGVoid <: MGType end                           # 0
struct MGBase <: MGType
    ;
    name::Symbol
end             # Base(σ)
struct MGProd <: MGType
    ;
    a::MGType;
    b::MGType
end     # A × B
struct MGSum <: MGType
    ;
    a::MGType;
    b::MGType
end     # A + B
struct MGFun <: MGType
    ;
    dom::MGType;
    cod::MGType
end # A → B
struct MGSemType <: MGType
    ;
    sem::SemanticType
end        # Rel/Prog/Model/Codec
struct MGPres <: MGType
    ;
    pres::PresType
end           # Pres(G,A)
struct MGRewrite <: MGType
    ;
    geom::GeomTag;
    a::MGType
end # Rewrite(G,A)
struct MGCost <: MGType
    ;
    geom::GeomTag;
    a::MGType
end # Cost(G,A)

# ── §7.7 Coercions ────────────────────────────────────────────────────────────

"""
    CoercionKind

Whether a coercion is exact or approximate (§7.7).
Reuses ErrorLevel from ApproxPipeline.jl (EXACT / BOUNDED / STATISTICAL).
"""
const CoercionKind = ErrorLevel   # EXACT | BOUNDED | STATISTICAL

"""
    Coercion

A typed change of presentation (§7.7):
φ : Pres(G1, A) ⇒ Pres(G2, A)      (exact)
φ : Pres(G1, A) ⇒_ε Pres(G2, A)    (approximate, error ε)

Semantic preservation obligation (§7.8):
Exact:   [[φ(x)]]_{G2} = [[x]]_{G1}
Approx:  dist([[φ(x)]]_{G2}, [[x]]_{G1}) ≤ ε
"""
struct Coercion
    name::Symbol
    from_geom::GeomTag
    to_geom::GeomTag
    sem_type::SemanticType
    kind::CoercionKind
    error_bound::Float64        # 0.0 for EXACT
    confidence::Float64        # 1.0 for EXACT/BOUNDED
end

Coercion(name, from, to, sem; kind=EXACT, ε=0.0, conf=1.0) =
    Coercion(name, from, to, sem, kind, ε, conf)

is_exact(c::Coercion) = c.kind == EXACT

# ── §7.7 Minimum registered coercions ────────────────────────────────────────

"""
Four minimum registered coercions from §7.7.
"""
const T_DAG_TO_FACTOR = Coercion(
    :T_DAG_to_Factor, GEOM_DAG, GEOM_FACTOR, sem_prog(:Sigma, :T)
)

const T_FACTOR_TO_TRIE = Coercion(
    :T_Factor_to_Trie, GEOM_FACTOR, GEOM_TRIE, sem_model(:Q, :A)
)

const T_TRIE_TO_TENSOR = Coercion(
    :T_Trie_to_Tensor, GEOM_TRIE, GEOM_TENSOR_SPARSE, sem_rel(:A, :B)
)

const T_TRIE_TO_CODEC = Coercion(
    :T_Trie_to_Codec,
    GEOM_TRIE,
    GEOM_TRIE,    # changes semantic type, not geometry
    sem_codec(:A)
)

const REGISTERED_COERCIONS = [
    T_DAG_TO_FACTOR, T_FACTOR_TO_TRIE, T_TRIE_TO_TENSOR, T_TRIE_TO_CODEC
]

"""
    find_coercion(from::GeomTag, to::GeomTag) -> Union{Coercion, Nothing}

Look up a registered coercion for the given geometry pair.
Returns the first matching coercion, or nothing if none exists.
"""
function find_coercion(from::GeomTag, to::GeomTag)::Union{Coercion, Nothing}
    idx = findfirst(c -> c.from_geom == from && c.to_geom == to, REGISTERED_COERCIONS)
    idx === nothing ? nothing : REGISTERED_COERCIONS[idx]
end

# ── TyLA adjunction markers (§7.3) ───────────────────────────────────────────

"""
    TyLADirection

Whether a transformation follows the F (operational→typed) or G (typed→operational)
direction of the TyLA adjunction F ⊣ G (§7.3).

F_DIRECTION: programmer writes `define-factor-rule` → normalization into
canonical geometry templates WITH typed ports, effects, laws.
G_DIRECTION: supercompiler reads normalized templates → lowered execution plan
(MM2 worklist, factor-graph message schedule, MORK CapsuleNode).
"""
@enum TyLADirection F_DIRECTION G_DIRECTION

export SemanticKind, SK_REL, SK_PROG, SK_MODEL, SK_CODEC, SK_SCHED, SK_STREAM
export SemanticType, sem_rel, sem_prog, sem_model, sem_codec, sem_sched, sem_stream
export GeomTag, GEOM_FACTOR, GEOM_DAG, GEOM_TRIE, GEOM_TENSOR_SPARSE, GEOM_TENSOR_DENSE
export HybridGeom, PresType
export MGType, MGUnit, MGVoid, MGBase, MGProd, MGSum, MGFun, MGSemType, MGPres
export MGRewrite, MGCost
export CoercionKind, Coercion, is_exact
export T_DAG_TO_FACTOR, T_FACTOR_TO_TRIE, T_TRIE_TO_TENSOR, T_TRIE_TO_CODEC
export REGISTERED_COERCIONS, find_coercion
export TyLADirection, F_DIRECTION, G_DIRECTION
