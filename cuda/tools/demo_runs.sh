#!/usr/bin/env bash
# demo_runs.sh: runs of the real-time demo that back numbers and media in the README.
#   1. frame rate of the window build (Windows with a display only): 120,000 steps of
#      the default case, profiler medians in cuda/results/demo_gui/summary.json
#   2. hero GIF check: regenerates the frames of docs/media/hero_re100_validated.gif with
#      the headless demo into a temporary folder and compares the result with the
#      committed GIF byte for byte (the committed GIF is never overwritten)
#   3. showcase stability: the --re mode at Re 110, 120, 150, 200 and 500 for 64,000 steps
# Run from the repository root: bash cuda/tools/demo_runs.sh [--allow-dirty]
set -u
if [ "${1:-}" = "--allow-dirty" ]; then export LBM_ALLOW_DIRTY=1; shift; fi
. "$(dirname "$0")/common.sh"
RES=${LBM_RESULTS:-cuda/results}
X=$(lbm_exe_suffix)
GIT=$(lbm_git_state) || exit 1
D=$RES/demo_gui; mkdir -p "$D"

# 1. window frame rate
if [ -n "$X" ] && [ -f cuda/build/lbm_cuda.exe ]; then
  {
    echo "date: $(date -Iseconds)"; echo "git: $GIT"
    echo "command: cuda/build/lbm_cuda.exe --steps 120000 (window, default case, 80 cells per diameter)"
    echo "gpu before: $(lbm_gpu_state)"
    (cd cuda/build && ./lbm_cuda.exe --steps 120000)
    echo "exit code: $?"
    echo "gpu after: $(lbm_gpu_state)"
  } > "$D/log.txt" 2>&1
  lbm_strip_paths "$D/log.txt"
  python - "$D" <<'PY'
import json, re, statistics, sys, os
D = sys.argv[1]
L = [l for l in open(os.path.join(D, "log.txt"), encoding="utf-8") if l.startswith("[")][1:]
fps = [float(re.search(r"([\d.]+) FPS", l).group(1)) for l in L]
ml = [float(re.search(r"([\d.]+) MLUPS", l).group(1)) for l in L]
st = lambda v: dict(median=statistics.median(v), min=min(v), max=max(v))
out = dict(command="cuda/build/lbm_cuda.exe --steps 120000", log="cuda/results/demo_gui/log.txt", intervals=len(fps),
           steps_per_frame=256, fps=st(fps), kernel_mlups_while_drawing=st(ml),
           interop="CUDA/GL interop active" in open(os.path.join(D, "log.txt"), encoding="utf-8").read())
json.dump(out, open(os.path.join(D, "summary.json"), "w"), indent=1)
print("demo window FPS:", out["fps"])
PY
else
  echo "window build not available on this system: frame rate not measured"
fi

# 2. hero GIF check
TMP=$(mktemp -d)
{
  echo "date: $(date -Iseconds)"; echo "git: $GIT"
  echo "command: cuda/build/lbm_cuda_headless$X --nd 80 --steps 192000 --ppm-every 2 --out <temporary folder>"
  echo "         python cuda/tools/hero_gif.py <temporary folder> 42 <temporary folder>/hero.gif"
  cuda/build/lbm_cuda_headless$X --nd 80 --steps 192000 --ppm-every 2 --out "$TMP" | tail -2
  python cuda/tools/hero_gif.py "$TMP" 42 "$TMP/hero.gif"
  new=$(sha256sum "$TMP/hero.gif" | cut -d' ' -f1); old=$(sha256sum docs/media/hero_re100_validated.gif | cut -d' ' -f1)
  echo "regenerated GIF sha256: $new"
  echo "committed GIF sha256:   $old"
  [ "$new" = "$old" ] && echo "hero GIF reproduced: identical" || echo "hero GIF reproduced: DIFFERENT (the committed GIF is kept)"
} > "$D/hero_frames_log.txt" 2>&1
sed -i "s#$TMP#<temporary folder>#g" "$D/hero_frames_log.txt"
lbm_strip_paths "$D/hero_frames_log.txt"
grep "hero GIF reproduced" "$D/hero_frames_log.txt"

# 3. showcase stability
for RE in 110 120 150 200 500; do
  S=$RES/showcase_check/re$RE; mkdir -p "$S"
  {
    echo "date: $(date -Iseconds)"; echo "git: $GIT"
    echo "command: cuda/build/lbm_cuda_headless$X --re $RE --steps 64000 --out $S"
    cuda/build/lbm_cuda_headless$X --re $RE --steps 64000 --out "$S"
    echo "exit code: $?"
  } > "$S/log.txt" 2>&1
  lbm_strip_paths "$S/log.txt"
  echo "showcase Re $RE: $(grep -q Diverged "$S/log.txt" && echo diverged || echo 'ran 20 D/U')"
done
