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
# ── OPEN ITEMS ────────────────────────────────────────────────────────────────
#   1. LIVE LOWERING DIFF — ✅ RAN (testset "MVP §15.4 demo 2 (EXECUTED)" below).
#      FINDING: the lowering is INERT and the §15.4 "reference interpreter" MVP was
#      claimed on an UNEXECUTABLE lowering. Cause = SYNTAX/FORM: `pln_stv_lowering`
#      emits `(=`/`:where`, not the `(exec source product)` triple the calculus fires
#      (positive control proves the mechanism works; grounding `*`/`min` doesn't rescue
#      it ⇒ not siting). FIX (still open) = rewrite the lowering into the `(exec …)`
#      grounded form — and settle the GroundedSource `(I (* …))` path, which also did
#      NOT reduce in a bare MORK space. WHEN it computes 0.765, the testset's
#      `!occursin("0.765", …)` asserts flip to `== 0.765` and it becomes the real gate.
#   2. INDUCTION / ABDUCTION goldens — ship in lib/pln with NO doctest `→`, so
#      pinning PLNBook to its own evaluation would be circular. Their goldens MUST come
#      from LIVE lib/pln execution (modulo Core issue #1's Truth_w2c leak), not PLNBook.

using Test
using MorkSupercompiler
using MORK: new_space, register_grounded!

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
    # ── Rule 1/8: ModusPonens — mgfw `stv_forward_map` now book-faithful (3b). ──
    # Finding B RESOLVED: `stv_forward_map` was made book-faithful (FactorGeometry.jl),
    # so it now agrees with the `PLNBook` oracle on BOTH coords. (Independent inline impl
    # vs oracle — the diff stays discriminating.)
    let (s_mgfw, c_mgfw) = stv_forward_map(0.8, 0.9, 0.7, 0.85),
        (s_ref, c_ref) = PLNRef.modus_ponens(0.8, 0.9, 0.7, 0.85)

        @test isapprox(s_mgfw, s_ref; atol=1e-3)   # 0.564 == 0.564
        @test isapprox(c_mgfw, c_ref; atol=1e-3)   # 0.4334 == 0.4334
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

@testset "MVP §15.4 demo 2 (EXECUTED) — lowering is INERT, positive-control gated" begin
    # De-vacuates OPEN ITEM 1. The old §15.4 "STV factor path == reference interpreter"
    # acceptance (test_mgfw.jl:475-494) diffed `stv_mp_reference` against its OWN formula
    # (a tautology) and only checked the lowering PARSES. This RUNS the lowering through
    # `space_metta_calculus!` and reads the result — distinguishing the two inertness causes
    # (syntax vs siting) with an EXPECTED value pinned per branch (not "any output passes").

    # ── POSITIVE CONTROL (so "no B_TV" can't pass for the wrong reason) ──
    # A known-firing `(exec source product)` rule proves the calculus mechanism works in
    # this space. Without this, "inert" could silently mean "harness miswired".
    let sc = new_space()
        space_add_all_sexpr!(sc, "(ping a)")
        space_add_all_sexpr!(sc, "(exec (pc 1) (, (ping \$x)) (, (ponged \$x)))")
        space_metta_calculus!(sc, 100)
        @test occursin("(ponged a)", space_dump_all_sexpr(sc))   # mechanism REACHED
    end

    t = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    rules = get_lowering(:PLN_STV_HeuristicModusPonens)(t, "")
    run_lowering =
        () -> begin
            s = new_space()
            space_add_all_sexpr!(s, "(apply-mp (A (stv 0.8 0.9)) (AimpB (stv 0.7 0.85)))")
            space_add_all_sexpr!(s, rules)
            space_metta_calculus!(s, 100)
            space_dump_all_sexpr(s)
        end

    # ── BARE MORK: inert. The (=/:where) rules sit as literals, the trigger is untouched,
    #    and NO computed B_TV (0.765 = min(0.9,0.85)·0.9, 0.56 = 0.8·0.7) is produced. ──
    let bare = run_lowering()
        @test !occursin("0.765", bare)                      # no executed confidence
        @test !occursin("(B (stv 0.5", bare)                # no executed B atom
        @test occursin("(apply-mp (A (stv 0.8 0.9))", bare) # trigger untouched ⇒ rule never fired
    end

    # ── CORE-LOADED (grounded * / min): STILL inert ⇒ the defect is the FORM, not siting.
    #    (NB: global GROUNDED_REGISTRY mutation — these are correct arithmetic, benign.) ──
    for (n, o) in (("*", *), ("min", min))
        register_grounded!(
            n,
            a -> begin
                length(a) < 2 && return nothing
                x = tryparse(Float64, a[1])
                y = tryparse(Float64, a[2])
                (x === nothing || y === nothing) && return nothing
                r = o(x, y)
                isinteger(r) ? string(Int(r)) : string(r)
            end
        )
    end
    @test !occursin("0.765", run_lowering())   # grounding doesn't rescue it ⇒ SYNTAX/FORM branch

    # ── STRUCTURAL proof of the FORM defect: the lowering emits `(=` / `:where`, NOT the
    #    `(exec source product)` triple the calculus actually fires (cf. the positive control). ──
    @test occursin("(=", rules) && occursin(":where", rules) && !occursin("(exec", rules)

    # FINDING (PLN 3a, 2026-06-15): the §15.4 "reference interpreter" MVP was claimed on an
    # UNEXECUTABLE lowering. Cause = SYNTAX/FORM (`(=`/`:where`, not `(exec …)`); grounding is
    # moot (the rule never engages the rewrite engine to reach arithmetic). FIX = rewrite
    # `pln_stv_lowering` into the `(exec source product)` grounded form (like
    # `motif_miner_lowering`). ⚠ the `(I (* …))` grounded-arith path did NOT reduce in a bare
    # MORK space either — the rewrite must settle the correct GroundedSource invocation /
    # Core wiring. WHEN the rewrite computes 0.765, flip the two `!occursin("0.765", …)`
    # asserts to `== 0.765` (pinned to the APPROX `stv_mp_reference` contract) — this testset
    # then becomes the real reference-interpreter gate.
end
