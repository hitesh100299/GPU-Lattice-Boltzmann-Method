#!/usr/bin/env bash
# perf.sh: timing runs on the demo case (2D-2 at 80 cells per diameter).
#   1. stream_copy: measured peak device bandwidth (best of 20, 512 MiB buffers)
#   2. lbm_bench --perf: 5 runs of 20,000 timed steps after 2,000 warm-up steps,
#      kernel only (CUDA events) and including the force and pressure diagnostics
#   3. nvidia-smi clocks, temperature and power at the start and end of each run
#   4. Nsight Compute, if the GPU performance counters are accessible: DRAM bytes
#      read and written per lbm_step launch
# Writes cuda/results/perf/ (or <dir>/perf/ with LBM_RESULTS=<dir>).
# Run from the repository root: bash cuda/tools/perf.sh [--allow-dirty]
set -u
if [ "${1:-}" = "--allow-dirty" ]; then export LBM_ALLOW_DIRTY=1; shift; fi
. "$(dirname "$0")/common.sh"
export LBM_RESULTS=${LBM_RESULTS:-cuda/results}
B=cuda/build; P=$LBM_RESULTS/perf; mkdir -p "$P"
X=$(lbm_exe_suffix)
GIT=$(lbm_git_state) || exit 1
{
  echo "date: $(date -Iseconds)"; echo "git: $GIT"; echo "command: $B/stream_copy$X"
  "$B/stream_copy$X"
} > "$P/stream_copy.log" 2>&1
for k in 1 2 3 4 5; do
  bash cuda/tools/run.sh "$B/lbm_bench$X" "$P/run$k" --case 2D-2 --nd 80 --perf --steps 20000 --warmup 2000 | grep "^perf"
done
# Nsight Compute: DRAM bytes per lbm_step launch (10 launches after the warm-up)
NCU=$(command -v ncu || ls "/c/Program Files/NVIDIA Corporation/"Nsight\ Compute*/ncu.bat 2>/dev/null | tail -1)
{
  echo "date: $(date -Iseconds)"; echo "git: $GIT"
  if [ -n "$NCU" ]; then
    echo "command: ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum --kernel-name lbm_step --launch-skip 200 --launch-count 10 lbm_bench --case 2D-2 --nd 80 --perf --steps 300 --warmup 200"
    mkdir -p "$P/ncu"
    "$NCU" --metrics dram__bytes_read.sum,dram__bytes_write.sum --kernel-name lbm_step --launch-skip 200 --launch-count 10 \
      "$B/lbm_bench$X" --case 2D-2 --nd 80 --perf --steps 300 --warmup 200 --out "$P/ncu" 2>&1
    echo "ncu exit code: $?"
  else
    echo "Nsight Compute (ncu) not found: not measured"
  fi
} > "$P/ncu.log" 2>&1
lbm_strip_paths "$P/ncu.log"
python - <<'EOF'
import json, re, statistics, glob, os
P = os.environ["LBM_RESULTS"] + "/perf"
runs = [json.load(open(d + "/summary.json")) for d in sorted(glob.glob(P + "/run*"))]
k = [r["perf"]["mlups_kernel_events"] for r in runs]
w = [r["perf"]["mlups_kernel_wall"] for r in runs]
g = [r["perf"]["mlups_with_diagnostics"] for r in runs]
sc = open(P + "/stream_copy.log").read()
gbs = float(re.search(r"RESULT_GBS ([\d.]+)", sc).group(1)) if "RESULT_GBS" in sc else None
ncu = open(P + "/ncu.log", errors="replace").read()
rd = re.findall(r"dram__bytes_read.sum\s+(\S+)\s+([\d.,]+)", ncu); wr = re.findall(r"dram__bytes_write.sum\s+(\S+)\s+([\d.,]+)", ncu)
def tobytes(unit, val):
    v = float(val.replace(",", "")); return v * {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}.get(unit, 1)
ncu_res = None
if rd and wr:
    r = statistics.median(tobytes(u, v) for u, v in rd); wv = statistics.median(tobytes(u, v) for u, v in wr)
    ncu_res = dict(dram_bytes_read_per_step=r, dram_bytes_write_per_step=wv, launches=len(rd))
reason = None
if not ncu_res:
    reason = "GPU performance counters are restricted to administrators (ERR_NVGPUCTRPERM)" if "ERR_NVGPUCTRPERM" in ncu else "see ncu.log"
c = runs[0]["case"]; N = c["Nx"] * c["Ny"]
med = statistics.median(k)
out = dict(case="2D-2 at 80 cells per diameter (demo case)", grid=f"{c['Nx']} x {c['Ny']}", cells=N, steps_per_run=runs[0]["perf"]["steps"], warmup=runs[0]["perf"]["warmup"], runs=len(runs),
           kernel_mlups=dict(median=med, min=min(k), max=max(k), values=k),
           kernel_mlups_wall=dict(median=statistics.median(w), min=min(w), max=max(w)),
           diag_mlups=dict(median=statistics.median(g), min=min(g), max=max(g), values=g),
           bytes_per_cell_update_model=73, bandwidth_model_GBs=med * 1e6 * 73 / 1e9,
           measured_peak_GBs=gbs, percent_of_measured_peak=(med * 1e6 * 73 / 1e9) / gbs * 100 if gbs else None,
           ncu=ncu_res if ncu_res else f"not measured: {reason}",
           ncu_bytes_per_cell_update=(ncu_res["dram_bytes_read_per_step"] + ncu_res["dram_bytes_write_per_step"]) / N if ncu_res else None,
           gpu_start=[r["gpu_start"] for r in runs], gpu_end=[r["perf"]["gpu_end"] for r in runs],
           sources=[os.path.relpath(d, ".").replace("\\", "/") + "/summary.json" for d in sorted(glob.glob(P + "/run*"))] + [P + "/stream_copy.log", P + "/ncu.log"])
json.dump(out, open(P + "/summary.json", "w"), indent=1)
print(json.dumps({k2: v for k2, v in out.items() if k2 not in ("gpu_start", "gpu_end", "sources")}, indent=1))
EOF
