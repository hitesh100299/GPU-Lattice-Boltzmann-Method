# Building, running and reproducing

Developed and tested on Windows; Linux build is compiled in CI but not run on a GPU.

## Prebuilt release (Windows)

The [Releases page](https://github.com/hitesh100299/GPU-Lattice-Boltzmann-Method/releases) has a zip with three programs and a launcher for each:

- `run_demo.bat` opens the validated real-time demo (case 2D-2 at 80 cells per diameter).
- `run_benchmark.bat` runs two short checks and prints them next to the reference values.
- `Run showcase (not validated).bat` opens the first CUDA version at Re about 823. Its window title starts with "Showcase, not validated".

It needs Windows 10 or 11 (64-bit) and an NVIDIA driver; no toolkit, compiler or Python. The programs are built with CUDA 13.3 Update 1, with the C and CUDA runtimes linked statically: the demo depends only on `glfw3.dll`, `OPENGL32.dll` and `KERNEL32.dll`, the benchmark only on `KERNEL32.dll` (checked with `dumpbin /dependents`).

**GPUs.** Native code for compute capability 7.5 (GeForce GTX 16, RTX 20), 8.0, 8.6 (RTX 30), 8.9 (RTX 40), 9.0 and 12.0 (RTX 50), plus PTX for 12.0 (checked with `cuobjdump`). Only the GeForce RTX 3060 Laptop GPU with driver 617.42 has been tested.

**Driver.** NVIDIA's [CUDA 13.3 Update 1 release notes](https://docs.nvidia.com/cuda/archive/13.3.1/cuda-toolkit-release-notes/index.html) list driver 580 or newer for minor-version compatibility of every CUDA 13.x toolkit, and no Windows value in the table of toolkit drivers (Linux: 610.43.02). So "driver 580 or newer" on Windows comes from the compatibility table and hasn't been tested on such a driver. A GPU newer than the RTX 50 series would have to compile the PTX at load time, which most likely needs a driver from the 610 branch or newer.

## Build from source

**Windows** needs the CUDA toolkit and the Visual Studio 2022 Build Tools; `cuda\run.bat` finds both and downloads GLFW if it isn't next to the repository. Every executable goes to `cuda\build\`.

| Command | What it does |
|---|---|
| `cuda\run.bat` | builds and starts the demo; arguments after `demo` go to the demo, for example `cuda\run.bat demo --signed` |
| `cuda\run.bat demo --no-run` | builds the demo without starting it |
| `cuda\run.bat bench` | builds `lbm_bench.exe` (FP32) and `lbm_bench_fp64.exe` |
| `cuda\run.bat headless` | builds the demo without a window |
| `cuda\run.bat tools` | builds the bandwidth probe `stream_copy.exe` |
| `cuda\run.bat baseline` | builds the first CUDA version headless, for the timing baseline |
| `cuda\run.bat test` | builds the benchmark and runs the two short regression cases in `tests/` |
| `cuda\run.bat release` | builds the demo, the benchmark and the showcase for the GPU generations above and zips them with the launchers into `dist\` |

**Linux** needs nvcc, `libglfw3-dev` and `make`. `make -C cuda` builds the demo and both benchmarks into `cuda/build/`; `make -C cuda check` builds everything, and `make -C cuda test` runs the regression cases. The CI compiles this path on every push, for sm_86, inside the `nvidia/cuda:13.3.0-devel-ubuntu24.04` container. GitHub's runners have no GPU, so nothing is run there.

## Options

Demo, `lbm_cuda.exe`:

| Option | Default | Meaning |
|---|---|---|
| `--nd <n>` | 80 | cells per cylinder diameter; 80 is the validated demo case |
| `--signed` | off | signed vorticity with a blue-red map instead of the magnitude |
| `--vmax <v>` | 10 | colour scale in units of U/D |
| `--blur <n>` | 0 | display smoothing passes, 0 to 2 |
| `--steps <n>` | none | stop after n steps; results are printed only if the flow is periodic by then (about 75 D/U, 240,000 steps at the default grid) |
| `--re <value>` | 100 | another Reynolds number on the same geometry, not validated; it runs at 110 and diverges at 120 and above |
| `--out`, `--ppm-every`, `--fast`, `--smooth` | | headless build only: output folder, image snapshots, steps per frame before and after shedding |

Benchmark, `lbm_bench.exe`:

| Option | Meaning |
|---|---|
| `--case poiseuille \| 2D-1 \| 2D-2` | the channel check or one of the two cylinder cases |
| `--nd <n>`, `--H <n>` | cells per diameter (cylinder cases) or channel height (Poiseuille) |
| `--out <dir>` | result folder, created if missing; `cuda/tools/run.sh` also logs the command, git commit and GPU state there |
| `--frames` | 2D-2: store vorticity frames of the last two periods |
| `--rest` | start from fluid at rest instead of the analytic channel flow |
| `--perf --steps <n> --warmup <n>` | timing run instead of a validation run |
| `--estimate` | print grid size, step count and memory, then exit |
| `--cap <D/U>` | override the time limit (400 for 2D-1, 160 for 2D-2) |

Exit codes: 0 finished, 1 bad arguments, 2 diverged, 3 time limit reached before the stop rule was met, 4 output folder or file can't be written.

## Reproduce every number

```
bash cuda/tools/reproduce_all.sh --list              the plan, with the recorded time of every run
bash cuda/tools/reproduce_all.sh --quick --out tmp   two-minute subset, compared with the committed results
bash cuda/tools/reproduce_all.sh                     everything, then figures and tables
```

The full run rewrites `cuda/results/` and then regenerates every figure and table with `cuda/tools/report.py`. It runs in Git Bash on Windows or bash on Linux and needs `python` with the packages in `cuda/tools/requirements.txt`; the baseline stage also runs the CuPy prototype. The scripts refuse to run on a working tree with uncommitted changes outside `cuda/results/`, so every log names a real commit; `--allow-dirty` overrides that and the logs then say "dirty tree".

The 2D-1 reference values are typed into `cuda/cases.h` from the FeatFlow page. The 2D-2 values come from `python cuda/tools/extract_reference.py`, which downloads the FeatFlow zip, checks its sha256 and writes `cuda/reference/extracted_2d2_reference.json`.

## Prototype

```
pip install -r python/requirements.txt
python python/lbm_cupy.py
```

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) has two jobs. `build` compiles every CUDA program on Linux with the same toolkit version as the development machine. `results` regenerates the tables from the committed summary.json files and fails if they differ from the committed `tables.md` and `history.csv`, so no number in the docs can drift from its source.
