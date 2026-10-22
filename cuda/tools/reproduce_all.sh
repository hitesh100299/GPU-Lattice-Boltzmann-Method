#!/usr/bin/env bash
# reproduce_all.sh: rebuild and rerun every result in cuda/results with one command.
#
#   bash cuda/tools/reproduce_all.sh --list            show the plan and the recorded time of every step
#   bash cuda/tools/reproduce_all.sh                   full suite into cuda/results (overwrites), then figures
#   bash cuda/tools/reproduce_all.sh --quick           short subset (about 2 minutes on an RTX 3060 Laptop GPU)
#   bash cuda/tools/reproduce_all.sh --out DIR [...]   write into DIR instead and compare every number with
#                                                      the committed results (nothing in cuda/results changes)
#   bash cuda/tools/reproduce_all.sh STAGE...          only these stages: build poiseuille 2d1 2d2 fp64
#                                                      demo_check perf baseline demo report
#   --allow-dirty                                      run even with uncommitted changes; logs say "dirty tree"
#
# Run from the repository root. Windows: Git Bash, with the CUDA toolkit and Visual Studio
# Build Tools installed (the build stage calls cuda\run.bat). Linux: nvcc, libglfw3-dev, make,
# and `python` on PATH (Ubuntu: package python-is-python3). Python packages:
# cuda/tools/requirements.txt (the baseline stage also uses python/requirements.txt).
# The 2D-1 run at 160 cells per diameter (about 1.9 hours) is not part of the suite.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1

OUT=cuda/results; QUICK=0; LIST=0; STAGES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --list) LIST=1; shift ;;
    --allow-dirty) export LBM_ALLOW_DIRTY=1; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    build|poiseuille|2d1|2d2|fp64|demo_check|perf|baseline|demo|report) STAGES+=("$1"); shift ;;
    *) echo "unknown argument $1 (see --help)"; exit 1 ;;
  esac
done
if [ ${#STAGES[@]} -eq 0 ]; then
  if [ $QUICK -eq 1 ]; then STAGES=(build poiseuille 2d1 2d2 demo_check)
  else STAGES=(build poiseuille 2d1 2d2 fp64 demo_check perf baseline demo report); fi
fi
want() { local s; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
. cuda/tools/common.sh
if [ $LIST -eq 0 ]; then lbm_git_state > /dev/null || exit 1; fi

# ---- the plan: stage, result folder, benchmark arguments
PLAN=()
add() { PLAN+=("$1|$2|$3"); }
if [ $QUICK -eq 1 ]; then
  add poiseuille poiseuille_H41_fp32  "lbm_bench --case poiseuille --H 41"
  add poiseuille poiseuille_H82_fp32  "lbm_bench --case poiseuille --H 82"
  add 2d1        2D-1_N20_fp32        "lbm_bench --case 2D-1 --nd 20"
  add 2d2        2D-2_N40_fp32        "lbm_bench --case 2D-2 --nd 40"
  add 2d2        2D-2_N80_fp32        "lbm_bench --case 2D-2 --nd 80"
else
  for H in 41 82 164; do add poiseuille poiseuille_H${H}_fp32 "lbm_bench --case poiseuille --H $H"; done
  for H in 41 82 164; do add poiseuille poiseuille_H${H}_fp64 "lbm_bench_fp64 --case poiseuille --H $H"; done
  add 2d1  2D-1_N20_fp32       "lbm_bench --case 2D-1 --nd 20"
  add 2d1  2D-1_N20_fp32_rest  "lbm_bench --case 2D-1 --nd 20 --rest"
  add 2d1  2D-1_N40_fp32       "lbm_bench --case 2D-1 --nd 40"
  add 2d1  2D-1_N80_fp32       "lbm_bench --case 2D-1 --nd 80"
  add 2d2  2D-2_N40_fp32       "lbm_bench --case 2D-2 --nd 40"
  add 2d2  2D-2_N80_fp32       "lbm_bench --case 2D-2 --nd 80 --frames"
  add 2d2  2D-2_N160_fp32      "lbm_bench --case 2D-2 --nd 160 --frames"
  add fp64 2D-1_N40_fp64       "lbm_bench_fp64 --case 2D-1 --nd 40"
  add fp64 2D-1_N80_fp64       "lbm_bench_fp64 --case 2D-1 --nd 80"
  add fp64 2D-2_N80_fp64       "lbm_bench_fp64 --case 2D-2 --nd 80"
fi

# recorded wall time of a committed run, in seconds (Unknown when absent)
recorded() {
  python - "$1" <<'PY'
import json, sys, os
p = os.path.join("cuda", "results", sys.argv[1], "summary.json")
try:
    print(f"{json.load(open(p))['total_wall_s']:.0f}")
except Exception:
    print("Unknown")
PY
}

echo "Plan (results into $OUT):"
TOTAL=0
for item in "${PLAN[@]}"; do
  IFS='|' read -r stage dir cmd <<< "$item"
  want "$stage" || continue
  t=$(recorded "$dir"); [ "$t" != "Unknown" ] && TOTAL=$((TOTAL + t))
  printf "  %-11s %-22s %-48s recorded %s s\n" "$stage" "$dir" "$cmd" "$t"
done
want build      && echo "  build       compile all executables                                  about 1 min"
want demo_check && echo "  demo_check  headless demo for the 2D-2 Nd 80 step count, compare     about 70 s"
want perf       && echo "  perf        stream copy, 5 x 20,000 steps, Nsight Compute attempt    about 1 min"
want baseline   && echo "  baseline    first CUDA version and CuPy prototype, 60 s each         about 2 min"
want demo       && echo "  demo        window frame rate, hero GIF check, showcase stability    about 3 min"
want report && [ "$OUT" = "cuda/results" ] && echo "  report      figures and tables (cuda/tools/report.py)                about 3 min"
echo "Recorded simulation time of the listed runs: $((TOTAL / 60)) min $((TOTAL % 60)) s (runs without a recorded time not counted; plus the other stages)."
[ $LIST -eq 1 ] && exit 0

FAILED=()
# ---- build
if want build; then
  echo "=== build"
  if [ "${OS:-}" = "Windows_NT" ]; then
    for m in bench headless tools baseline; do KARMAN_NOPAUSE=1 cmd //c "cuda\\run.bat $m" > /dev/null || FAILED+=("build $m"); done
    if want demo; then KARMAN_NOPAUSE=1 cmd //c "cuda\\run.bat demo --no-run" > /dev/null || FAILED+=("build demo"); fi
  else
    make -C cuda bench headless tools baseline || FAILED+=("build")
  fi
fi
X=$(lbm_exe_suffix)

# ---- simulation runs
for item in "${PLAN[@]}"; do
  IFS='|' read -r stage dir cmd <<< "$item"
  want "$stage" || continue
  exe=${cmd%% *}; args=${cmd#* }
  echo "=== $stage: $dir"
  # shellcheck disable=SC2086
  bash cuda/tools/run.sh "cuda/build/$exe$X" "$OUT/$dir" $args | grep -E "^(poiseuille H|2D-1 result|2D-2 result|DIVERGED)" || true
  [ -f "$OUT/$dir/summary.json" ] || FAILED+=("$dir")
done

# ---- demo consistency, timing, baselines, demo runs, report
if want demo_check; then echo "=== demo_check"; LBM_RESULTS="$OUT" bash cuda/tools/demo_consistency_check.sh | tail -1 || FAILED+=("demo_check"); fi
if want perf; then echo "=== perf"; LBM_RESULTS="$OUT" bash cuda/tools/perf.sh > /dev/null || FAILED+=("perf"); fi
if want baseline; then echo "=== baseline"; LBM_RESULTS="$OUT" bash cuda/tools/baseline.sh || FAILED+=("baseline"); fi
if want demo; then echo "=== demo"; LBM_RESULTS="$OUT" bash cuda/tools/demo_runs.sh || FAILED+=("demo"); fi
if want report && [ "$OUT" = "cuda/results" ]; then echo "=== report"; python cuda/tools/report.py > /dev/null || FAILED+=("report"); fi

# ---- comparison with the committed results when writing elsewhere
if [ "$OUT" != "cuda/results" ]; then
  echo "=== comparison with the committed results in cuda/results"
  python - "$OUT" <<'PY'
import json, os, sys, glob
out = sys.argv[1]
keys = ["l2_rel_error", "cd", "cl", "dp", "cd_max", "cl_max", "st", "dp_t0_half"]
worst = 0.0
for p in sorted(glob.glob(os.path.join(out, "*", "summary.json"))):
    name = os.path.basename(os.path.dirname(p))
    ref = os.path.join("cuda", "results", name, "summary.json")
    if not os.path.exists(ref):
        continue
    a, b = json.load(open(p)), json.load(open(ref))
    sa = {**a.get("results", {}), **a.get("poiseuille", {})}
    sb = {**b.get("results", {}), **b.get("poiseuille", {})}
    parts = []
    for k in keys:
        if isinstance(sa.get(k), (int, float)) and isinstance(sb.get(k), (int, float)) and sb[k]:
            d = abs(sa[k] / sb[k] - 1) * 100
            worst = max(worst, d)
            parts.append(f"{k} {sa[k]:.6g} vs {sb[k]:.6g} ({d:.2e} %)")
    if parts:
        print(f"  {name}: " + "; ".join(parts))
print(f"largest relative difference: {worst:.2e} %")
PY
fi

if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED: ${FAILED[*]}"; exit 1; fi
echo "done"
