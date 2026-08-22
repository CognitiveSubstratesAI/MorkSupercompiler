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
trap 'rm -f "$DRIVER"' EXIT

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
