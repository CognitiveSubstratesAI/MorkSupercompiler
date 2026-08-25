#!/usr/bin/env bash
# run_tests.sh — run this package's suite (or ONE file) with a REAL EXIT CODE.
#
#   tools/run_tests.sh                      # full suite
#   tools/run_tests.sh test/some_file.jl    # one file (or any ad-hoc .jl)
#
# ─── 🔴 WHY THIS EXISTS AND WHY IT LOOKS ODD ────────────────────────────────────────────────────
# The documented invocation
#     printf 'include("test/runtests.jl");exit()\n' | julia --project=. -i
# ALWAYS EXITS 0 — regardless of failures OR errors. `julia -i` with piped stdin is interactive, and
# interactive mode SWALLOWS exceptions: the throw prints, the REPL continues, the trailing `exit()`
# returns 0. Measured in MORK:
#     piped -i, error then exit()      -> 0
#     piped -i, FAILING @testset       -> 0        <-- a RED suite reporting success
#     non-interactive `julia file.jl`  -> 1        <-- correct
# That is how MORK's only upstream differential sat ERRORING on every run unnoticed (c543841).
# ⇒ So the driver wraps everything in try/catch and calls `exit(ok ? 0 : 1)` ITSELF. The exit code
#   comes from the driver, never from the shell's view of an interactive julia.
#
# ⚠️ `< /dev/null` IS LOAD-BEARING, not tidiness. Under a pipe, stdin is a PipeEndpoint the writer
# has already closed, and anything spawning a subprocess with explicit stdio (Aqua's
# `persistent_tasks`) fails with EINVAL against a closed handle.
#
# ⚠️ MEMORY CEILING. A runaway test OOM-kills the EDITOR, not itself — measured on this box, where a
# VSCode Julia LS holds ~6.7 GB. exit 137 means the ceiling was hit: find the unbounded query, do
# not just raise the cap.
#
# ─── PORTED FROM MORK/tools/run_tests.sh (2026-08-21) ───────────────────────────────────────────
# Generic across packages: uses tools/repl.jl only IF the package has one, and skips the docstring
# lint unless the package ships it. Everything else is identical, deliberately — the exit-code trap
# above is not package-specific and every package needs the same protection from it.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PKG="$(basename "$ROOT")"
TARGET="${1:-test/runtests.jl}"
case "$TARGET" in /*) ABS_TARGET="$TARGET" ;; *) ABS_TARGET="$ROOT/$TARGET" ;; esac

[ -f "$ABS_TARGET" ] || { echo "run_tests.sh: no such target: $ABS_TARGET" >&2; exit 2; }

DRIVER="$(mktemp "${TMPDIR:-/tmp}/${PKG}_run_tests_XXXXXX.jl")"

# ─── MACHINE-READABLE VERDICT — a WRAPPER'S EXIT CODE IS NOT AN OBSERVATION ────────────────────
# Ported from MORK/tools/run_tests.sh 2026-08-25, same day, same reason: a background harness
# reported "completed (exit code 0)" for a suite that had been SIGTERM'd, whose own last line read
# `EXIT=143`. Ask tools/suite_result.sh, never a wrapper's code, and never grep a log for the
# ABSENCE of "Fail" (a still-BUFFERED log looks clean).
RESULT_FILE="$ROOT/tools/.last_suite_result"
_write_result() {
  { echo "VERDICT=$2"; echo "RC=$1"; echo "WHEN=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$RESULT_FILE" 2>/dev/null || true
}
_write_result - RUNNING
_on_exit() {
  rc=$?
  rm -f "$DRIVER" "$DRIVER.smoke"
  if [ "${_SUITE_SIGNALLED:-0}" = "1" ]; then _write_result "$rc" KILLED
  elif [ "$rc" -eq 0 ]; then                  _write_result 0 PASS
  else                                        _write_result "$rc" FAIL
  fi
}
# Trap signals explicitly or the shell dies WITHOUT running the EXIT trap, leaving the previous
# run's PASS standing — a stale green, worse than no verdict.
trap '_SUITE_SIGNALLED=1; exit 143' TERM
trap '_SUITE_SIGNALLED=1; exit 130' INT
trap _on_exit EXIT

# ─── COLD-LOAD SMOKE CHECK — fails in seconds on a DEFINITION error, before the suite ──────────
#
# 🔴 PORTED FROM MORK 2026-08-25 AFTER THIS PACKAGE COST THE THIRD FAILURE OF THE SAME CLASS.
# Definition-level errors are INVISIBLE to the warm :7702 server, because ordering, docstring
# parsing and cross-module resolution are settled during `include`, which never re-runs — a warm
# module can only tell you about FUNCTION BODIES. Measured that day: adjacent docstrings (the first
# tries to document the second and Julia refuses) x3, docstring `$` interpolation x2, a function
# defined ABOVE the struct its signature names, an unqualified cross-module call. In one case
# `isdefined(...)` returned TRUE while a fresh process could not compile the file at all.
#
# The third of those happened in THIS package, which had no smoke check while MORK did. Ten lines.
# Skip with SC_SKIP_SMOKE=1 when iterating on test files only.
if [ "${SC_SKIP_SMOKE:-0}" != "1" ]; then
  if ! julia --project=. -e "using $PKG" >/dev/null 2>"$DRIVER.smoke"; then
    echo "run_tests.sh: COLD LOAD FAILED — a definition error, not a test failure." >&2
    echo "  (the warm server cannot see this class: ordering, docstrings, cross-module names)" >&2
    if grep -qE "cannot document|UndefVarError|MethodError|BoundsError|TypeError|syntax:" "$DRIVER.smoke"; then
      echo "  ── root cause ──" >&2
      grep -nE "cannot document|UndefVarError|MethodError|BoundsError|TypeError|syntax:" "$DRIVER.smoke" | head -5 >&2
    fi
    sed -n '1,25p' "$DRIVER.smoke" >&2
    exit 1
  fi
  rm -f "$DRIVER.smoke"
fi

# Include tools/repl.jl only when present — several packages have no REPL helper, and a hard
# include would make this runner unusable in exactly those packages that most need it.
REPL_LINE=""
[ -f "$ROOT/tools/repl.jl" ] && REPL_LINE="include(raw\"$ROOT/tools/repl.jl\")"

cat > "$DRIVER" <<EOF
ok = try
    $REPL_LINE
    include(raw"$ABS_TARGET")
    true
catch e
    showerror(stderr, e); println(stderr)
    false
end
exit(ok ? 0 : 1)
EOF

# Optional per-package docstring lint (MORK ships one). A docstring \$name INTERPOLATES and breaks
# PRECOMPILE, so the module never loads and the suite cannot report it — run it BEFORE the suite.
LINT="$ROOT/tools/lint_docstring_interp.py"
if [ -f "$LINT" ] && [ -d "$ROOT/src" ]; then
  python3 "$LINT" "$ROOT/src" || { echo "run_tests.sh: docstring lint FAILED" >&2; exit 1; }
fi

MEM_MAX="${PKG_TEST_MEM_MAX:-8G}"
HEAP_HINT="${PKG_TEST_HEAP_HINT:-6G}"
JL=(julia --project=. --threads="${JULIA_TEST_THREADS:-4}" --heap-size-hint="$HEAP_HINT" -i "$DRIVER")

if [ "$MEM_MAX" = "none" ]; then
  echo "run_tests.sh: memory ceiling DISABLED (PKG_TEST_MEM_MAX=none)" >&2
  "${JL[@]}" < /dev/null
elif command -v systemd-run >/dev/null 2>&1 && systemd-run --user --scope true >/dev/null 2>&1; then
  systemd-run --user --scope -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 --quiet "${JL[@]}" < /dev/null
  rc=$?
  [ $rc -eq 137 ] && echo "run_tests.sh: KILLED at the ${MEM_MAX} ceiling — a test allocated without bound. Find it before raising PKG_TEST_MEM_MAX." >&2
  exit $rc
else
  echo "run_tests.sh: WARNING — systemd-run --user --scope unavailable; running WITHOUT a memory ceiling. A runaway test can OOM-kill unrelated processes on this machine." >&2
  "${JL[@]}" < /dev/null
fi
# allow-cold-start: full-suite runner; a suite run is a cold run by nature
