"""Hero GIF for the README from the demo's own frames.

Step 1 (GPU, done by the caller):
    cuda/build/lbm_cuda_headless.exe --nd 80 --steps 192000 --ppm-every 2 --out <tmp dir>
  writes the demo's colourised frames (same colourise kernel, same LUT as the window)
  every second frame; after shedding is detected a frame is 256 steps, so a PPM every
  512 steps, about 21 frames per shedding period at 80 cells per diameter.
Step 2 (this script):
    python cuda/tools/hero_gif.py <tmp dir> <n frames> docs/media/hero_re100_validated.gif
  takes the last n frames, scales them to 1100 px width and writes a looping GIF.
"""
import glob
import os
import sys

from PIL import Image

src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
files = sorted(glob.glob(os.path.join(src, "frame_*.ppm")))[-n:]
frames = []
for f in files:
    im = Image.open(f).convert("RGB")
    im = im.transpose(Image.FLIP_TOP_BOTTOM)         # row 0 is the bottom wall
    w = 1100
    im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)
print(f"{out}: {len(frames)} frames from {os.path.basename(files[0])} to {os.path.basename(files[-1])}, size {frames[0].size}, {os.path.getsize(out) / 1e6:.1f} MB")
