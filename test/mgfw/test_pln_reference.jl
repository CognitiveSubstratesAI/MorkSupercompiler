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
#   stv_forward_map (specialize_exact, TEST-ONLY) : cA·cI·min(sA,sI) = 0.5355
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
#   2. INDUCTION / ABDUCTION goldens — ✅ DONE (3c). No lib/pln doctest `→`, and 3a found
#      no execute-in-Core mechanism (+ Core issue #1's Truth_w2c leak) ⇒ live-sourcing
#      unavailable. So the goldens are HAND-DERIVED independently (the pins below) and
#      PLNBook's induction/abduction are pinned to them — non-circular (NOT PLNBook's eval).

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

    # Induction / Abduction ship in lib/pln with NO doctest `→`, so the goldens below are
    # HAND-DERIVED independently (computed by hand from the formula at a consistency-
    # satisfying point), NOT read back from PLNBook's own evaluation — else the pin would
    # be circular. Inputs: (sA,cA, sB,cB, sC,cC, s?A/AB,c, s?C/CB,c) = (.5,.9,.4,.8,.6,.85,.7,.9,.3,.85)
    let (s, c) = PLNRef.induction(0.5, 0.9, 0.4, 0.8, 0.6, 0.85, 0.7, 0.9, 0.3, 0.85)   # :216
        @test isapprox(s, 0.52; atol=1e-3)      # hand: (.7·.3·.4)/.5 + (1−(.7·.4)/.5)·(.6−.4·.3)/.6 = .168+.44·.8
        @test isapprox(c, 0.18666; atol=1e-3)   # hand: w2c(.3·.85·.9)=0.2295/1.2295
    end
    let (s, c) = PLNRef.abduction(0.5, 0.9, 0.4, 0.8, 0.6, 0.85, 0.7, 0.9, 0.3, 0.85)   # :227
        @test isapprox(s, 0.525; atol=1e-3)     # hand: (.7·.3·.6)/.4 + .6·.3·.7/.6 = .315+.21
        @test isapprox(c, 0.34875; atol=1e-3)   # hand: w2c(.7·.9·.85)=0.5355/1.5355
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

    # ── Rules 2/8 .. 8/8 (3c): each mgfw forward map (independent inline, FactorGeometry)
    #    diffed against the PLNBook oracle at an interior point. ──
    @test all(
        isapprox.(
            stv_symmetric_mp(0.8, 0.9, 0.7, 0.85),
            PLNRef.symmetric_modus_ponens(0.8, 0.9, 0.7, 0.85);
            atol=1e-3
        )
    )
    @test all(
        isapprox.(
            stv_deduction(0.8, 0.9, 0.7, 0.85, 0.6, 0.8, 0.7, 0.9, 0.6, 0.85),
            PLNRef.deduction(0.8, 0.9, 0.7, 0.85, 0.6, 0.8, 0.7, 0.9, 0.6, 0.85);
            atol=1e-3
        )
    )
    @test all(
        isapprox.(
            stv_inversion(0.7, 0.8, 0.6, 0.9), PLNRef.inversion(0.7, 0.8, 0.6, 0.9);
            atol=1e-3
        )
    )
    @test all(
        isapprox.(
            stv_induction(0.5, 0.9, 0.4, 0.8, 0.6, 0.85, 0.7, 0.9, 0.3, 0.85),
            PLNRef.induction(0.5, 0.9, 0.4, 0.8, 0.6, 0.85, 0.7, 0.9, 0.3, 0.85);
            atol=1e-3
        )
    )
    @test all(
        isapprox.(
            stv_abduction(0.5, 0.9, 0.4, 0.8, 0.6, 0.85, 0.7, 0.9, 0.3, 0.85),
            PLNRef.abduction(0.5, 0.9, 0.4, 0.8, 0.6, 0.85, 0.7, 0.9, 0.3, 0.85);
            atol=1e-3
        )
    )
    @test all(
        isapprox.(
            stv_revision(0.6, 0.5, 0.8, 0.7), PLNRef.revision(0.6, 0.5, 0.8, 0.7); atol=1e-3
        )
    )
    @test all(isapprox.(stv_negation(0.7, 0.85), PLNRef.negation(0.7, 0.85); atol=1e-3))
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

    tpl   = GLOBAL_REGISTRY.templates[:PLN_STV_HeuristicModusPonens]
    rules = get_lowering(:PLN_STV_HeuristicModusPonens)(tpl, "")
    run_lowering =
        () -> begin
            s = new_space()
            # input shape follows the rewritten lowering's sources
            space_add_all_sexpr!(s, "(stv A 0.8 0.9)\n(imp A B 0.7 0.85)")
            space_add_all_sexpr!(s, rules)
            space_metta_calculus!(s, 100)
            space_dump_all_sexpr(s)
        end

    # ── THE INVERSION THIS TESTSET PRESCRIBED FOR ITSELF, PERFORMED 2026-08-25 ──
    # The FINDING below (PLN 3a, 2026-06-15) diagnosed the lowering as UNEXECUTABLE, named the
    # cause as SYNTAX/FORM, prescribed the fix, and left instructions: "WHEN the rewrite computes
    # 0.765, flip the two `!occursin("0.765", …)` asserts to `== 0.765` … this testset then
    # becomes the real reference-interpreter gate." The rewrite is done; this is that flip.
    out = run_lowering()
    ref_s, ref_c = stv_mp_reference(0.8, 0.9, 0.7, 0.85)          # (0.56, 0.765)
    @test occursin("0.765", out)                                   # was !occursin
    @test occursin("(stv B ", out)                                 # was !occursin("(B (stv 0.5")
    got = [l for l in split(strip(out), "\n") if startswith(strip(l), "(stv B ")]
    @test length(got) == 1
    parts = split(strip(strip(got[1]), ['(', ')']), " ")
    @test parse(Float64, parts[3]) ≈ ref_s atol=1e-12
    @test parse(Float64, parts[4]) ≈ ref_c atol=1e-12

    # ── STRUCTURAL: the FORM defect is gone. The lowering now emits `(exec … (O (pure …)))`,
    #    not `(=`/`:where`. This is the inverse of the old assertion and pins the fix. ──
    @test occursin("(exec", rules)
    @test !occursin(":where", rules)
    @test occursin("(pure", rules)

    # FINDING (PLN 3a, 2026-06-15) — RESOLVED 2026-08-25. The §15.4 "reference interpreter" MVP had
    # been claimed on an UNEXECUTABLE lowering. The June diagnosis was exactly right: cause =
    # SYNTAX/FORM (`(=`/`:where`, not `(exec …)`), and grounding was moot — this testset PROVED that
    # by registering `*`/`min` into GROUNDED_REGISTRY and finding it still inert.
    #
    # ⚠️ THE RESOLUTION CORRECTS ONE DETAIL OF THE PRESCRIPTION. The fix is NOT the
    # "(exec source product) GROUNDED form": `GROUNDED_REGISTRY` is consulted only by
    # `asource_new`, i.e. for exec SOURCE conjuncts. MM2 arithmetic lives in `PURE_OPS` reached
    # through a `(pure …)` SINK — 297 of them, including product_f64 / min_f64 / f64_from_string —
    # and needed no wiring at all. The exemplar is `decision_tree_learning_without_min_sink.mm2`,
    # already green in our corpus differential at 71 steps.
    #
    # ⚠️ AND A COMMENT ELSEWHERE POINTED THE WRONG WAY: test_mgfw.jl claimed the blocker was
    # "arithmetic primitive wiring (`*`, `min`) through the supercompiler's prim registry". That is
    # a THIRD mechanism (M-Core `Prim` evaluation) which this path never touches. Following it
    # produced 21d5e03 — real and tested, but not this unblock.
end
