#!/usr/bin/env bash
# demo_consistency_check.sh: the real-time demo must reproduce the benchmark.
# Runs the headless demo for exactly the step count of the 2D-2 benchmark at 80
# cells per diameter and compares cD max, cL max and St (accepted within 0.1 %;
# both programs share the kernel, the case and the sampling, so they normally agree
# to every stored digit).
# Run from the repository root: bash cuda/tools/demo_consistency_check.sh [--allow-dirty]
# LBM_RESULTS=<dir> reads and writes there instead of cuda/results.
set -u
if [ "${1:-}" = "--allow-dirty" ]; then export LBM_ALLOW_DIRTY=1; shift; fi
. "$(dirname "$0")/common.sh"
R=${LBM_RESULTS:-cuda/results}
X=$(lbm_exe_suffix)
B=$R/2D-2_N80_fp32/summary.json; O=$R/demo_consistency
# paths go to Python as arguments, never pasted into its source (Windows backslashes)
STEPS=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['run']['steps'])" "$B") || STEPS=""
if [ -z "$STEPS" ]; then echo "cannot run: no benchmark result at $B (run the 2D-2 case at 80 cells per diameter first)"; exit 1; fi
GIT=$(lbm_git_state) || exit 1
mkdir -p "$O"
{
  echo "date: $(date -Iseconds)"; echo "git: $GIT"
  echo "command: cuda/build/lbm_cuda_headless$X --nd 80 --steps $STEPS --out $O"
  echo "gpu before: $(lbm_gpu_state)"
  cuda/build/lbm_cuda_headless$X --nd 80 --steps "$STEPS" --out "$O" 2>&1
  echo "exit code: $?"
  echo "gpu after: $(lbm_gpu_state)"
} > "$O/log.txt"
lbm_strip_paths "$O/log.txt"
python - "$B" "$O" <<'PY'
import json, sys, os
bpath, odir = sys.argv[1], sys.argv[2]
b = json.load(open(bpath)); d = json.load(open(os.path.join(odir, "summary.json")))
rows = []; ok = True
for k in ("cd_max", "cl_max", "st"):
    vb, vd = b["results"][k], d["results"][k]
    dev = abs(vd / vb - 1) * 100 if isinstance(vd, (int, float)) else float("inf")
    ok &= dev <= 0.1; rows.append(dict(quantity=k, benchmark=vb, demo=vd, deviation_pct=dev))
    print(f"{k}: benchmark {vb:.10g} demo {vd:.10g} deviation {dev:.2e} %")
res = dict(steps=b["run"]["steps"], demo_steps=d["steps"], rows=rows, pass_within_0p1_percent=ok,
           benchmark=bpath.replace("\\", "/"), demo=os.path.join(odir, "summary.json").replace("\\", "/"),
           log=os.path.join(odir, "log.txt").replace("\\", "/"))
json.dump(res, open(os.path.join(odir, "consistency.json"), "w"), indent=1)
print("demo matches benchmark:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
