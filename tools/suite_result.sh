#!/usr/bin/env bash
# suite_result.sh — the ONLY sanctioned answer to "did the MorkSupercompiler suite pass?"
#
# 🔴 WHY THIS EXISTS. On 2026-08-25 a background harness reported "completed (exit code 0)" for a
# suite that had been SIGTERM'd mid-run; the log's own last line was `EXIT=143`. The wrapper was
# reporting the exit status of the shell that wrapped the run, not the verdict of the run. Acting
# on it would have landed a commit on a suite that never finished. Earlier the same day a grep over
# a still-BUFFERED log found no failures and was briefly read as "clean", and a suite was run
# against a tree that had since been edited.
#
# All three are one failure: AN INSTRUMENT REPORTING SUCCESS IT DID NOT OBSERVE. So do not read
# exit codes off wrappers, and do not grep logs for the absence of the word "Fail". Run this.
#
#   PASS    -> 0     the FULL suite finished green AND the tree has not changed since
#   PARTIAL -> 5     the last run targeted ONE FILE (a probe) — not a suite verdict
#   FAIL    -> 1     the suite ran and failed
#   KILLED  -> 1     died on a signal (SIGTERM/SIGINT) — NOT a green, and not a red either
#   STALE   -> 4     green, but a source file is newer than the verdict: RE-RUN, do not commit
#   RUNNING -> 3     still going; a verdict does not exist yet
#   missing -> 2     no suite has been run in this checkout
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_FILE="$SCRIPT_DIR/.last_suite_result"

if [ ! -f "$RESULT_FILE" ]; then
  echo "suite_result: NO RESULT — tools/run_tests.sh has not been run in this checkout." >&2
  exit 2
fi
# shellcheck disable=SC1090
VERDICT=$(grep '^VERDICT=' "$RESULT_FILE" | cut -d= -f2)
RC=$(grep '^RC=' "$RESULT_FILE" | cut -d= -f2)
LANE=$(grep '^LANE=' "$RESULT_FILE" | cut -d= -f2)
WHEN=$(grep '^WHEN=' "$RESULT_FILE" | cut -d= -f2-)

TARGET=$(grep '^TARGET=' "$RESULT_FILE" | cut -d= -f2-)

# A verdict from a SINGLE-FILE run is not a suite verdict. Probes go through the same runner and
# would otherwise leave a PASS that describes two lines of scratch code (measured 2026-08-25).
if [ -z "$TARGET" ]; then
  # No TARGET field: the verdict predates the field, so we cannot tell WHAT it ran. An
  # unattributable green is not evidence — re-run rather than trust it.
  echo "suite_result: UNKNOWN PROVENANCE — verdict has no TARGET field, so what it ran is" >&2
  echo "  not recorded. Re-run tools/run_tests.sh with no argument." >&2
  exit 5
elif [ "$TARGET" != "test/runtests.jl" ]; then
  echo "suite_result: PARTIAL — last run targeted '$TARGET', not the full suite." >&2
  echo "  A single-file run is not a suite verdict. Run tools/run_tests.sh with no argument." >&2
  exit 5
fi

case "$VERDICT" in
  RUNNING) echo "suite_result: RUNNING since $WHEN — no verdict yet. Do not commit." >&2; exit 3 ;;
  KILLED)  echo "suite_result: KILLED (rc=$RC) at $WHEN — the run did not finish." >&2
           echo "  A killed run is NOT a pass. Re-run before drawing any conclusion." >&2; exit 1 ;;
  FAIL)    echo "suite_result: FAIL (rc=$RC) at $WHEN, lane=$LANE" >&2; exit 1 ;;
  PASS)    : ;;
  *)       echo "suite_result: UNPARSEABLE verdict '$VERDICT' — treat as no result." >&2; exit 2 ;;
esac

# Green — but green against WHAT? A verdict older than the source it tested is not evidence.
# ⚠️ `test/` IS INCLUDED DELIBERATELY. The first version watched only `src/`, and a test edited
# after a green run would have kept reporting PASS — a green that never saw the assertion. Caught
# 2026-08-25 by editing a test immediately after a green suite.
NEWER=$(find "$SCRIPT_DIR/../src" "$SCRIPT_DIR/../test" -name '*.jl' -newer "$RESULT_FILE" 2>/dev/null | head -5)
if [ -n "$NEWER" ]; then
  echo "suite_result: STALE — PASS recorded $WHEN, but these are NEWER than it:" >&2
  echo "$NEWER" | sed 's/^/    /' >&2
  echo "  The suite did not test the current tree. Re-run before committing." >&2
  exit 4
fi
echo "suite_result: PASS at $WHEN (lane=$LANE), tree unchanged since."
