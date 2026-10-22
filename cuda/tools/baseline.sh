#!/usr/bin/env bash
# baseline.sh: timing of the two earlier versions on their own default case (1600 x 400).
#   1. the first CUDA version, cuda/showcase/karman_showcase.cu, headless build, 60 s
#   2. the CuPy prototype, python/lbm_cupy.py, 60 s (needs cupy, pygame and a display;
#      skipped with a note when they are missing)
# Both print their own profiler lines every 30 frames. The first interval (warm-up and,
# for the prototype, CUDA graph capture) is dropped; the medians of the rest go into
# cuda/results/baseline/summary.json. For the prototype, solver MLUPS = steps/s divided
# by the simulation fraction times 640,000 cells, the way its own README computed it.
# Run from the repository root: bash cuda/tools/baseline.sh [--allow-dirty]
set -u
if [ "${1:-}" = "--allow-dirty" ]; then export LBM_ALLOW_DIRTY=1; shift; fi
. "$(dirname "$0")/common.sh"
R=${LBM_RESULTS:-cuda/results}/baseline; mkdir -p "$R"
X=$(lbm_exe_suffix)
ROOT=$(pwd)
GIT=$(lbm_git_state) || exit 1

# 1. first CUDA version; it writes PPM snapshots into its working folder, so it runs in a temp folder
TMP=$(mktemp -d)
{
  echo "date: $(date -Iseconds)"; echo "git: $GIT"
  echo "command: timeout 60 cuda/build/karman_showcase_headless$X (run in a temporary folder, it writes PPM snapshots)"
  echo "gpu before: $(lbm_gpu_state)"
  (cd "$TMP" && timeout 60 "$ROOT/cuda/build/karman_showcase_headless$X")
  echo "exit code: $? (124 = stopped by the 60 s timeout, as intended)"
  echo "gpu after: $(lbm_gpu_state)"
} > "$R/cuda_first_version_60s.log" 2>&1
lbm_strip_paths "$R/cuda_first_version_60s.log"

# 2. CuPy prototype
if python -c "import cupy, pygame" > /dev/null 2>&1; then
  {
    echo "date: $(date -Iseconds)"; echo "git: $GIT"
    echo "command: timeout 60 python -u python/lbm_cupy.py"
    echo "gpu before: $(lbm_gpu_state)"
    timeout 60 python -u python/lbm_cupy.py
    echo "exit code: $? (124 = stopped by the 60 s timeout, as intended)"
    echo "gpu after: $(lbm_gpu_state)"
  } > "$R/cupy_prototype_60s.log" 2>&1
  lbm_strip_paths "$R/cupy_prototype_60s.log"
else
  echo "cupy or pygame not installed: CuPy prototype not timed" > "$R/cupy_prototype_60s.log"
fi

python - "$R" <<'PY'
import json, re, statistics, sys, os
R = sys.argv[1]
def lines(name, key):
    p = os.path.join(R, name)
    return [l for l in open(p, encoding="utf-8", errors="replace") if l.startswith("[") and key in l] if os.path.exists(p) else []
def stats(v):
    return dict(median=round(statistics.median(v), 1), min=round(min(v), 1), max=round(max(v), 1), n=len(v)) if v else None
cu = lines("cuda_first_version_60s.log", "MLUPS")[1:]
py = lines("cupy_prototype_60s.log", "steps/s")[1:]
num = lambda pat, l: float(re.search(pat, l).group(1))
sps_py = [num(r"(\d+) steps/s", l) for l in py]
frac_py = [num(r"sim\s+([\d.]+)%", l) / 100 for l in py]
out = {
    "note": "Default case of both earlier versions: 1600 x 400 cells, tau 0.57, u0 0.20, Re about 823. 60 s each; the first profiler interval is dropped.",
    "machine": "cuda/results/MACHINE.md",
    "cuda_first_version": {"log": "cuda_first_version_60s.log", "program": "cuda/showcase/karman_showcase.cu, headless build",
                           "solver_mlups": stats([num(r"([\d.]+) MLUPS", l) for l in cu]),
                           "steps_per_s": stats([num(r"(\d+) steps/s", l) for l in cu])},
    "cupy_prototype": {"log": "cupy_prototype_60s.log", "program": "python/lbm_cupy.py",
                       "graph_path": any("CUDA graph captured" in l for l in open(os.path.join(R, "cupy_prototype_60s.log"), encoding="utf-8", errors="replace")),
                       "solver_mlups": stats([s / f * 0.64 for s, f in zip(sps_py, frac_py)]),
                       "steps_per_s": stats(sps_py), "sim_fraction": stats(frac_py),
                       "fps_120_steps_per_frame": stats([num(r"([\d.]+) FPS", l) for l in py])},
}
json.dump(out, open(os.path.join(R, "summary.json"), "w"), indent=1)
for k in ("cuda_first_version", "cupy_prototype"):
    print(k, "solver MLUPS:", out[k]["solver_mlups"])
PY
