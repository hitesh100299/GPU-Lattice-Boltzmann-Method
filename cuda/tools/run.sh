#!/usr/bin/env bash
# run.sh [--allow-dirty] <executable> <out dir> [args...]
# Runs a benchmark binary and writes <out dir>/log.txt with the date, the git commit,
# the command, the GPU state before and after, and the program output.
# It refuses to run when the working tree has uncommitted changes outside
# cuda/results; with --allow-dirty it runs anyway and the log says "dirty tree".
# Run from the repository root (Git Bash on Windows, bash on Linux).
set -u
if [ "${1:-}" = "--allow-dirty" ]; then export LBM_ALLOW_DIRTY=1; shift; fi
. "$(dirname "$0")/common.sh"
EXE="$1"; OUT="$2"; shift 2
GIT=$(lbm_git_state) || exit 1
mkdir -p "$OUT"
LOG="$OUT/log.txt"
{
  echo "date: $(date -Iseconds)"
  echo "git: $GIT"
  echo "command: $EXE --out $OUT $*"
  echo "gpu before: $(lbm_gpu_state)"
} > "$LOG"
"$EXE" --out "$OUT" --tag "git $GIT" "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
{
  echo "gpu after: $(lbm_gpu_state)"
  echo "exit code: $RC"
} >> "$LOG"
lbm_strip_paths "$LOG"
exit $RC
