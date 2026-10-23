#!/usr/bin/env bash
# demo_runs.sh: runs of the real-time demo that back numbers and media in the README.
#   1. frame rate of the window build (Windows with a display only): 120,000 steps of
#      the default case, profiler medians in cuda/results/demo_gui/summary.json
#   2. hero GIF check: regenerates the frames of docs/media/hero_re100_validated.gif with
#      the headless demo into a temporary folder and compares frame count and frame size
#      with the committed GIF. Not bytes: the committed GIF is compressed with gifsicle
#      (lossy), which a fresh run doesn't apply. The committed GIF is never overwritten.
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
  python - "$TMP/hero.gif" docs/media/hero_re100_validated.gif <<'PY'
import sys
from PIL import Image
def shape(p):
    im = Image.open(p)
    return im.n_frames, im.size
new, old = shape(sys.argv[1]), shape(sys.argv[2])
print(f"regenerated GIF: {new[0]} frames, {new[1][0]} x {new[1][1]}")
print(f"committed GIF:   {old[0]} frames, {old[1][0]} x {old[1][1]}")
print("hero GIF reproduced: same frame count and size" if new == old else
      "hero GIF reproduced: DIFFERENT frame count or size (the committed GIF is kept)")
PY
} > "$D/hero_frames_log.txt" 2>&1
# the temporary folder appears in Unix form and, from Python on Windows, in Windows form
for t in "$TMP" "$(cygpath -m "$TMP" 2>/dev/null)" "$(cygpath -w "$TMP" 2>/dev/null)"; do
  [ -n "$t" ] && python - "$D/hero_frames_log.txt" "$t" <<'PY'
import sys
p, t = sys.argv[1:]
s = open(p, encoding="utf-8", errors="replace").read()
open(p, "w", encoding="utf-8").write(s.replace(t, "<temporary folder>"))
PY
done
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
