# PLN forward-map reference gate (Layer-1 build, step 2).
#
# Purpose — the per-rule harness the PLN Layer-1 work transcribes against.
# It does TWO jobs (the third, the LIVE diff, is deliberately NOT done here —
# see OPEN ITEMS):
#   1. Encodes the faithful PLN truth-value formulas ANALYTICALLY (PLNRef
#      below), transcribed 1:1 from lib/pln/pln_core_logic.metta with file:line
#      provenance, and PINS the transcription to the lib/pln doctest goldens.
#   2. Diffs mgfw's Julia `stv_forward_map` against PLNRef and lays down the
#      full 8-rule reference table + gate, so the Layer-1 forward-map
#      transcription (step 3) fills each row and flips its
#      `@test_skip`/`@test_broken` to a green `@test`.
#
# ── SYSTEM OF RECORD (explicit decision) ──────────────────────────────────────
# The PLN BOOK formulas (lib/pln / the uploaded PLN spec) are the contract for
# the EXACT rules. PLNRef is analytic — NOT a call into Core — but routing
# around Core's live bug is NOT the same as resolving it:
#   - Core's live `Truth_ModusPonens` confidence path leaks a `Channel`
#     (pln_core_logic.metta:245-248, "Bug 2, still open 2026-04-08"). So live
#     lib/pln does NOT currently produce the book value — and live lib/pln is
#     what MOSES scoring / ECAN consume. Pinning to book is a CHOICE of
#     contract; the reconciling action is to FILE that Core bug so live
#     converges to book. Until fixed, mgfw-faithful-to-book ≠ live-lib/pln and
#     this gate does NOT catch that fork — it is tracked, not hidden.
#   - MG-Framework §10.1.2 `min(cA,cI)·0.9` is a legitimate APPROXIMATE rule
#     (what `specialize_approximate` uses), NOT a bug. It lives in
#     `references.jl`'s `stv_mp_reference` (+ the lowering) and STAYS — step 3
#     relabels it "HeuristicMP (approx)" and folds PLNRef's book family into
#     references.jl so there is ONE reference module, two delineated families.
#
# ── FINDING B (recorded as `@test_broken`) ────────────────────────────────────
# THREE MP-confidence formulas coexist on (0.8,0.9,0.7,0.85), cross-checked by
# nothing:
#   stv_forward_map (LIVE specialize_exact)     : cA·cI·min(sA,sI) = 0.5355
#   pln_stv_lowering + stv_mp_reference (approx): min(cA,cI)·0.9   = 0.765
#   lib/pln Truth_ModusPonens = PLNRef (book)   : w2c(cA·cI)       = 0.4334
# The sharp form: `stv_forward_map` matches NEITHER its own template's lowering
# NOR book — an internal inconsistency provable with no book/live debate. Step 3
# replaces the MP path with book. This gate collapses only the TWO Julia
# surfaces (forward_map vs book); the lowering's EXECUTED output is closed by
# OPEN ITEM 1, not here.
#
# ── OPEN ITEMS (the work this gate does NOT do; step 3) ────────────────────────
#   1. LIVE LOWERING DIFF — the actual never-run diff pln_stv.jl flags and
#      test_mgfw.jl's §15.4 smoke test calls "queued for the PLN session":
#      execute `pln_stv_lowering`'s MeTTa via `space_metta_calculus!` and diff
#      its EXECUTED B against the contract. Blocked on wiring `*`/`min`/`0.9`
#      prims through the supercompiler prim registry. The existing §15.4
#      "demo 2" acceptance (test_mgfw.jl:475-494) is a TAUTOLOGY (reference vs
#      its own formula) + an `occursin` text-match — de-vacuate it.
#   2. INDUCTION / ABDUCTION goldens — ship in lib/pln with NO doctest `→`, so
#      pinning PLNRef to its own evaluation would be circular. Their goldens
#      MUST come from LIVE lib/pln execution (recorded at a consistency-
#      satisfying point), not from PLNRef.

using Test
using MorkSupercompiler

# PLNRef = the consolidated book-PLN reference family, which now lives in the package
# as `MorkSupercompiler.PLNBook` (src/mgfw/templates/references.jl) alongside the
# APPROXIMATE `stv_mp_reference` — ONE reference module, two delineated families. This
# test pins PLNBook to the lib/pln doctest goldens; the mgfw forward maps (step 3) are
# then diffed against the same module, so test and runtime share one contract.
const PLNRef = MorkSupercompiler.PLNBook

@testset "PLNRef pins to lib/pln doctest goldens" begin
    # These goldens are the recorded `→` doctest values in pln_core_logic.metta.
    # Asserting PLNRef ≈ golden pins the analytic transcription to Core's spec.
    @test all(
        isapprox.(PLNRef.modus_ponens(0.8, 0.9, 0.7, 0.85), (0.564, 0.4334); atol=1e-3)
    )           # :243
    @test all(
        isapprox.(
            PLNRef.symmetric_modus_ponens(0.8, 0.9, 0.7, 0.85), (0.628, 0.7191); atol=1e-3
        )
    ) # :258
    @test all(isapprox.(PLNRef.inversion(0.7, 0.8, 0.6, 0.9), (0.6, 0.432); atol=1e-3))                  # :291
    @test all(isapprox.(PLNRef.revision(0.6, 0.5, 0.8, 0.7), (0.74, 0.7692); atol=1e-3))                 # :271
    @test PLNRef.negation(0.7, 0.85) == (1.0 - 0.7, 0.85)                                                # :283
    @test isapprox(PLNRef.t_or(0.6, 0.4), 0.76; atol=1e-6)                                               # :67
    # Deduction 5-input golden (0.6 0.3213). Confidence here = PQs·QRs·PQc·QRc
    # = 0.7·0.6·0.9·0.85 = 0.3213.
    let (s, c) = PLNRef.deduction(0.8, 0.9, 0.7, 0.85, 0.6, 0.8, 0.7, 0.9, 0.6, 0.85)                    # :191
        @test isapprox(s, 0.6; atol=1e-3)
        @test isapprox(c, 0.3213; atol=1e-3)
    end

    # Singular-boundary behavior (faithful: /safe → nothing, i.e. (empty)):
    @test PLNRef.c2w(1.0) === nothing                 # c→1 ⇒ 1−c=0 ⇒ empty
    @test PLNRef.safe_div(1.0, 0.0) === nothing       # denominator 0 ⇒ empty
    # Deduction at Qs→1 returns Rs (the explicit pre-/safe branch), NOT empty.
    # As Qs→1 the consistency preconditions force sPQ→1 (smallest-intersection
    # →1) and sQR→Rs/Qs, so the branch is reachable only at that corner — which
    # is exactly the corner lib/pln's `Qs>0.9999 ⇒ Rs` guard exists to protect.
    let (s, _) = PLNRef.deduction(0.8, 0.9, 0.99995, 0.85, 0.6, 0.8, 1.0, 0.9, 0.6, 0.85)
        @test s == 0.6   # == Rs, via the guard (no (1−Qs) blow-up)
    end
end

@testset "mgfw forward maps vs PLNRef — Layer-1 gate" begin
    # ── Rule 1/8: HeuristicModusPonens — mgfw map EXISTS (stv_forward_map). ──
    # FINDING B: currently disagrees with the faithful reference on BOTH coords.
    # `@test_broken` → reports an "unexpected pass" once step 3 makes mgfw
    # faithful, at which point promote these to `@test`.
    let (s_mgfw, c_mgfw) = stv_forward_map(0.8, 0.9, 0.7, 0.85),
        (s_ref, c_ref) = PLNRef.modus_ponens(0.8, 0.9, 0.7, 0.85)

        @test_broken isapprox(s_mgfw, s_ref; atol=1e-3)   # 0.56  vs 0.564
        @test_broken isapprox(c_mgfw, c_ref; atol=1e-3)   # 0.5355 vs 0.4334
    end

    # ── Rules 2/8 .. 8/8: mgfw forward maps NOT YET BUILT (step 3). ──
    # The reference is ready (PLNRef.*); step 3 adds each mgfw map and flips
    # the matching skip to a green `@test … vs PLNRef.<rule>(…)`.
    @test_skip "SymmetricModusPonens forward map — PLNRef.symmetric_modus_ponens"
    @test_skip "Deduction forward map (singular Qs→1)   — PLNRef.deduction"
    @test_skip "Inversion forward map (singular sA→0)   — PLNRef.inversion"
    @test_skip "Induction forward map (singular sA→0)   — PLNRef.induction (TODO)"
    @test_skip "Abduction forward map (singular sB→0)   — PLNRef.abduction (TODO)"
    @test_skip "Revision forward map                    — PLNRef.revision"
    @test_skip "Negation forward map                    — PLNRef.negation"
end
