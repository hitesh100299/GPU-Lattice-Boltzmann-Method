# Performance

All rows were measured on the same NVIDIA GeForce RTX 3060 Laptop GPU ([MACHINE.md](../cuda/results/MACHINE.md)), on AC power, with the GPU drawing up to about 95 W and the memory at 7001 MHz; every log records the clocks at the start and end of its run. MLUPS means million lattice cell updates per second.

| Code | Grid | MLUPS, median | min | max | Source |
|---|---|---:|---:|---:|---|
| CuPy prototype (`python/lbm_cupy.py`, AoS, CUDA graph path) | 1600 × 400 | 2,140 | 2,075 | 2,192 | [baseline](../cuda/results/baseline/summary.json) |
| First CUDA version (`cuda/showcase/karman_showcase.cu`, SoA, headless) | 1600 × 400 | 3,897 | 3,802 | 4,002 | [baseline](../cuda/results/baseline/summary.json) |
| Validated solver, step kernel only | 1761 × 330 | 3,932 | 3,896 | 4,002 | [perf](../cuda/results/perf/summary.json) |
| Validated solver with force and pressure diagnostics | 1761 × 330 | 3,908 | 3,691 | 3,939 | [perf](../cuda/results/perf/summary.json) |

<p align="center"><img src="../cuda/results/figures/performance.png" width="100%" alt="Effective memory bandwidth of each code against the measured copy peak"></p>

## Method

- **Timing.** Five runs of 20,000 timed steps after 2,000 warm-up steps on the demo case (2D-2 at 80 cells per diameter, 581,130 cells), kernel time from CUDA events. The table gives the median, with min and max as the noise floor (`cuda/tools/perf.sh`).
- **Bandwidth model.** 73 bytes per cell update: nine reads and nine writes of 4 bytes, plus the 1-byte node type. 3,932 MLUPS therefore move 287 GB/s.
- **Peak.** A device-to-device copy kernel reaches 317 GB/s on this GPU (512 MiB buffers, best of 20, [stream_copy.log](../cuda/results/perf/stream_copy.log)); the memory clock and bus width allow 336 GB/s. The solver reaches 91 % of the measured copy peak. Every bandwidth figure here was measured on this machine; none is a datasheet number.
- **Cell count.** MLUPS counts every lattice site, the usual convention. The kernel skips the solid sites (walls and cylinder, 1.5 % of the grid), so on fluid sites alone the figure is about 89 % of the peak.
- **Baselines.** The two baseline rows run the prototype's own case at 640,000 cells, timed by each code's own profiler over 60 s (`cuda/tools/baseline.sh`). The validated rows use CUDA events over 20,000 steps at 581,130 cells. Across these two protocols, the per-cell cost of the validated kernel matches the first CUDA version's, so the shifted storage and the Zou and He boundaries cost nothing measurable.
- **Validation runs.** The MLUPS column of the run-facts table in [tables.md](../cuda/results/tables.md) is informational only: those runs weren't timed under a fixed GPU state.

## Measured DRAM traffic

Nsight Compute: not measured. GPU performance counters are restricted to administrators by the driver default (ERR_NVGPUCTRPERM, [ncu.log](../cuda/results/perf/ncu.log)), and changing that is a system setting. To measure it, enable counter access in the NVIDIA Control Panel (Desktop, Enable Developer Settings; then Developer, Manage GPU Performance Counters, Allow access to all users) and rerun `bash cuda/tools/perf.sh`, which records `dram__bytes_read.sum` and `dram__bytes_write.sum` for ten `lbm_step` launches and compares them with the 73-byte model.

## Where the time goes

- **Diagnostics.** Forces are reduced on the GPU (warp shuffles, then shared memory, then one double-precision atomic per block). With the eight probe densities, 80 bytes go to the host every 64 steps. All diagnostics together cost 0.6 % of the throughput.
- **Launch overhead.** The CuPy prototype needed CUDA graphs to hide the cost of launching kernels from Python. From C++ the launches queue asynchronously: the throughput from CUDA events and from the wall clock differ by less than 0.01 %, so graphs would add complexity for nothing.
- **Drawing.** CUDA writes the coloured frame straight into the OpenGL texture. In the demo log the simulation takes about 98 % of each frame, vorticity and colouring 1.5 % and drawing 0.2 % ([demo log](../cuda/results/demo_gui/log.txt)). The window runs at 25.2 frames per second with 256 time steps per frame ([summary](../cuda/results/demo_gui/summary.json)).
- **Launch shape.** The step kernel uses a 2D grid with one block row per lattice row, so the row index needs no integer division.
