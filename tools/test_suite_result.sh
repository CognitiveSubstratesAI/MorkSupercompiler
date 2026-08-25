#!/usr/bin/env bash
# test_suite_result.sh — exercise the verdict gate's BRANCH ORDER against constructed verdicts.
#
# 🔴 WHY THIS EXISTS. Four defects were found in this gate on 2026-08-25, EVERY ONE by using it
# under load and NONE by reading it: probe runs overwriting the suite verdict; no TARGET provenance;
# staleness measured from the run's END (so a mid-run edit read as "tree unchanged"); and the
# provenance checks sitting BEFORE the verdict case, so a FAILED run with no TARGET reported
# "UNKNOWN PROVENANCE" and SWALLOWED the failure.
#
# That last one is the worst of the four. A stale PASS at least still says PASS — wrong, but a human
# glancing at it reads a signal. A swallowed FAIL renders as an unfamiliar provenance message that
# may not register as "something is broken" at all.
#
# It was also the only one found by READING rather than by use, and reading has a poor record here:
# on 2026-08-25 name-based conclusions were wrong 5 times out of 5. So the branch order is now
# asserted by CONSTRUCTED CASES rather than trusted from inspection.
#
# THE INVARIANT: a positive claim needs proof, a negative one does not.
#   FAIL / KILLED / RUNNING  -> always reported honestly, even with no provenance at all
#   PASS                     -> refused unless it can prove WHAT ran (TARGET) and WHEN it started
#                               (STARTED_AT, for the staleness window)
#
# Run: bash tools/test_suite_result.sh
set -uo pipefail
cd "$(dirname "$0")/.."
R=tools/.last_suite_result
SAVE=$(cat $R 2>/dev/null || true)
fails=0
check() { # name expected_rc verdict_body
    printf '%b' "$3" > $R
    ./tools/suite_result.sh >/dev/null 2>&1; got=$?
    if [ "$got" -eq "$2" ]; then printf "  ok    %-46s rc=%s\n" "$1" "$got"
    else printf "  FAIL  %-46s rc=%s expected=%s\n" "$1" "$got" "$2"; fails=$((fails+1)); fi
}
echo "suite_result gate — branch order:"
check "FAIL with no TARGET reports FAIL, not provenance" 1 'VERDICT=FAIL\nRC=1\nWHEN=x\n'
check "KILLED with no TARGET reports KILLED"             1 'VERDICT=KILLED\nRC=143\nWHEN=x\n'
check "RUNNING reports RUNNING"                          3 'VERDICT=RUNNING\nRC=-\nWHEN=x\n'
check "PASS with no TARGET is REFUSED"                   5 'VERDICT=PASS\nRC=0\nWHEN=x\n'
check "PASS with a PARTIAL target is REFUSED"            5 'VERDICT=PASS\nRC=0\nTARGET=/tmp/probe.jl\nSTARTED_AT=2026-01-01T00:00:00Z\nWHEN=x\n'
check "PASS with no STARTED_AT is REFUSED"               5 'VERDICT=PASS\nRC=0\nTARGET=test/runtests.jl\nWHEN=x\n'
check "unparseable verdict is REFUSED"                   2 'VERDICT=WAT\nRC=0\nTARGET=test/runtests.jl\n'
[ -n "$SAVE" ] && printf '%s\n' "$SAVE" > $R || rm -f $R
echo; [ "$fails" -eq 0 ] && { echo "all branch-order cases hold"; exit 0; } || { echo "$fails case(s) FAILED"; exit 1; }
