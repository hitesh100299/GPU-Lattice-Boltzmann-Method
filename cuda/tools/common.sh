# common.sh: helpers shared by the measurement scripts (sourced, not run directly).
#
# lbm_git_state prints the short commit hash for a clean tree. If tracked or untracked
# files outside cuda/results differ from the commit, it refuses (returns 1) unless
# LBM_ALLOW_DIRTY=1, in which case it prints "<hash> (dirty tree)". Results are excluded
# because the scripts write them while they run.

lbm_git_state() {
    local h
    h=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
    if [ -n "$(git status --porcelain -- . ':(exclude)cuda/results' 2>/dev/null)" ]; then
        if [ "${LBM_ALLOW_DIRTY:-0}" != 1 ]; then
            echo "refusing to run: the working tree has uncommitted changes outside cuda/results" >&2
            echo "(commit them first, or pass --allow-dirty to record a dirty run)" >&2
            return 1
        fi
        echo "$h (dirty tree)"
    else
        echo "$h"
    fi
}

lbm_gpu_state() {
    nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo Unknown
}

# ".exe" for the Windows build, empty on Linux
lbm_exe_suffix() {
    if [ -f cuda/build/lbm_bench.exe ]; then echo .exe; else echo ""; fi
}

# replace the absolute path of the repository with "." in a log file
lbm_strip_paths() {
    local root win
    root=$(git rev-parse --show-toplevel)
    win=$(cygpath -w "$root" 2>/dev/null || echo "$root")
    python - "$1" "$root" "$win" <<'PY'
import os, sys
path, *roots = sys.argv[1:]
s = open(path, encoding="utf-8", errors="replace").read()
# the repository becomes ".", the home folder "~" (tools such as CuPy print install paths)
pairs = [(r, ".") for r in roots] + [(os.path.expanduser("~"), "~")]
for r, rep in sorted(set(pairs), key=lambda x: len(x[0]), reverse=True):
    for form in (r, r.replace("\\", "/"), r.replace("/", "\\")):
        s = s.replace(form, rep)
open(path, "w", encoding="utf-8").write(s)
PY
}
