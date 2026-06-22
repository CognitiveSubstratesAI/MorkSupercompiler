"""
BisimVerifier — discharges bisimulation obligations recorded by MM2Compiler.

Implements §12.2 (Testing Strategy) of the MM2 Supercompiler v2 spec —
"Differential testing against interpreter" — and discharges the three
obligation kinds from Algorithm 14 (§9.2 BisimulationProof):

  :forward_sim  — every MeTTa trace has a corresponding MM2 trace
                  (source-derived atoms ⊆ MM2-derived atoms, modulo
                  scaffolding)
  :backward_sim — every MM2 trace projects to a MeTTa trace
                  (MM2-derived atoms ⊆ source-derived atoms)
  :fairness     — MM2 priority ordering preserves MeTTa fairness
                  (both terminate within the same step budget)

Strategy: run the source MeTTa program directly via `space_metta_calculus!`
on a fresh Space, run the MM2-compiled program the same way on another
fresh Space (both seeded with identical background facts), then compare the
resulting atom sets via `space_dump_all_sexpr`. The obligations are discharged
in aggregate per kind; individual obligations of the same kind get the
aggregate verdict (since differential testing examines whole-program
behaviour, not per-atom).

This is the **Boundary #3** closure from the 2026-06-18 audit. Before this
module, MM2Compiler recorded obligations but had no discharger; the
"verifier absent" gap was documented at MM2Compiler.jl:23-24.

Limitations:
- Discharge is at the **trace-set** level, not per-obligation step-by-step.
  A passing verdict means whole-program differential equivalence; a failing
  verdict means at least one obligation kind failed but does not pinpoint
  which atom triggered it.
- Sound for monotone (sink-free) programs. For programs with retractions
  or non-monotone behaviour, set-equality of final dumps may miss
  intermediate divergence.
- Fairness check is coarse (both halt within budget); the spec's full
  fairness obligation would require comparing branch-exploration orderings.
"""

using MORK: Space, new_space, space_add_all_sexpr!, space_metta_calculus!,
    space_dump_all_sexpr

# ── VerifyResult — per-obligation discharge status ────────────────────────────

"""
    VerifyResult

Discharge status for a single `BiSimObligation`.

obligation — the original obligation (from MM2Compiler)
discharged — true iff the differential test passed for its kind
reason     — short human-readable explanation
"""
struct VerifyResult
    obligation::BiSimObligation
    discharged::Bool
    reason::String
end

# ── BisimVerdict — aggregate result ───────────────────────────────────────────

"""
    BisimVerdict

Aggregate verdict from `verify_bisim`.

all_discharged — true iff every obligation discharged
results        — per-obligation `VerifyResult`s (one per input obligation)
source_atoms   — atoms in the source-program Space dump (set of trimmed lines)
mm2_atoms      — atoms in the MM2-program Space dump (set of trimmed lines)
forward_ok     — aggregate forward-sim verdict (source ⊆ MM2 atoms)
backward_ok    — aggregate backward-sim verdict (MM2 ⊆ source atoms)
fairness_ok    — aggregate fairness verdict (both halted within step budget)
"""
struct BisimVerdict
    all_discharged::Bool
    results::Vector{VerifyResult}
    source_atoms::Set{String}
    mm2_atoms::Set{String}
    forward_ok::Bool
    backward_ok::Bool
    fairness_ok::Bool
end

# ── Atom-set extraction ───────────────────────────────────────────────────────

"""
    _atom_set_from_dump(dump) -> Set{String}

Parse a `space_dump_all_sexpr` output into a set of trimmed atom strings.
Drops blank lines; preserves duplicates by keeping one copy per unique form.
"""
function _atom_set_from_dump(dump::AbstractString)::Set{String}
    out = Set{String}()
    for line in split(dump, "\n"; keepempty=false)
        s = strip(line)
        isempty(s) && continue
        push!(out, String(s))
    end
    out
end

# ── verify_bisim — the main entry point ───────────────────────────────────────

"""
    verify_bisim(source_program, compiled_program, obligations;
                 facts="", max_steps=100) -> BisimVerdict

Discharge a list of bisimulation obligations recorded by `MM2Compiler.compile_program`
via differential testing.

Runs `source_program` and `compiled_program` on fresh Spaces (both seeded with
`facts`), then compares their dumps:

  - `:forward_sim`  passes iff every source atom appears in the MM2 dump
  - `:backward_sim` passes iff every MM2 atom appears in the source dump
  - `:fairness`     passes iff both completed within `max_steps` (no truncation)

Returns a `BisimVerdict` carrying per-obligation results plus the aggregate
flags and atom sets (for diagnostic comparison if the verdict fails).

# Example

```julia
g = MCoreGraph()
# … build M-Core program …
compiled_str, obligs = compile_program(g, root_ids)
verdict = verify_bisim(source_str, compiled_str, obligs;
                       facts="(parent alice bob)")
if verdict.all_discharged
    @info "Bisim verified for all \$(length(obligs)) obligations"
else
    @warn "Bisim failed" verdict.forward_ok verdict.backward_ok verdict.fairness_ok
end
```
"""
function verify_bisim(
    source_program::AbstractString,
    compiled_program::AbstractString,
    obligations::Vector{BiSimObligation};
    facts::AbstractString="",
    max_steps::Int=100
)::BisimVerdict
    # Run source on its own fresh space
    s_src = new_space()
    isempty(facts) || space_add_all_sexpr!(s_src, facts)
    space_add_all_sexpr!(s_src, String(source_program))
    src_steps = space_metta_calculus!(s_src, max_steps)
    src_dump = space_dump_all_sexpr(s_src)
    source_atoms = _atom_set_from_dump(src_dump)

    # Run compiled on a fresh space
    s_mm2 = new_space()
    isempty(facts) || space_add_all_sexpr!(s_mm2, facts)
    space_add_all_sexpr!(s_mm2, String(compiled_program))
    mm2_steps = space_metta_calculus!(s_mm2, max_steps)
    mm2_dump = space_dump_all_sexpr(s_mm2)
    mm2_atoms = _atom_set_from_dump(mm2_dump)

    # Aggregate verdicts per obligation kind
    forward_ok = issubset(source_atoms, mm2_atoms)
    backward_ok = issubset(mm2_atoms, source_atoms)
    # Fairness (coarse): both halted within the step budget. A run that
    # consumed exactly max_steps may have been truncated; treat strict
    # equality as failure.
    fairness_ok = src_steps < max_steps && mm2_steps < max_steps

    # Per-obligation results
    results = VerifyResult[]
    for o in obligations
        d, why = if o.kind === :forward_sim
            forward_ok,
            forward_ok ? "source atoms ⊆ MM2 atoms" :
            "source has $(length(setdiff(source_atoms, mm2_atoms))) atoms missing from MM2"
        elseif o.kind === :backward_sim
            backward_ok,
            backward_ok ? "MM2 atoms ⊆ source atoms" :
            "MM2 has $(length(setdiff(mm2_atoms, source_atoms))) atoms missing from source"
        elseif o.kind === :fairness
            fairness_ok,
            fairness_ok ? "both halted (src=$src_steps, mm2=$mm2_steps, budget=$max_steps)" :
            "step budget exhausted (src=$src_steps, mm2=$mm2_steps, budget=$max_steps)"
        else
            false, "unknown obligation kind: $(o.kind)"
        end
        push!(results, VerifyResult(o, d, why))
    end

    BisimVerdict(
        forward_ok && backward_ok && fairness_ok,
        results,
        source_atoms,
        mm2_atoms,
        forward_ok,
        backward_ok,
        fairness_ok
    )
end

export VerifyResult, BisimVerdict, verify_bisim
