# Kármán Vortex Street on the GPU

Real time 2D Lattice Boltzmann simulation of flow past a cylinder. The solver, post processing and colour mapping all run as custom CUDA kernels driven from Python through CuPy, with live rendering in pygame.

<p align="center">
  <img src="docs/media/vortex_street.gif" width="100%" alt="Vorticity magnitude of a Kármán vortex street behind a cylinder">
</p>

<p align="center"><i>Vorticity magnitude, Re ≈ 823, 1600 × 400 lattice</i></p>

---

## Highlights

- **Fully GPU resident**: the distribution functions never leave the GPU. Only a small `uint8` RGB image is copied to the host each frame.
- **Hand written CUDA kernels** compiled at runtime with `cupy.RawKernel`: LBM step, vorticity, Gaussian blur and colour mapping.
- **CUDA graph batching**: hundreds or thousands of time steps are captured once and replayed with a single launch, removing Python launch overhead.
- **GPU side colour mapping**: a 256 entry colormap lookup table lives in `__constant__` memory, replacing per frame CPU colormapping and resampling.
- **Two speed playback**: a fast forward mode and a smooth mode, switched by a vorticity threshold.
- **Built in profiler and divergence watchdog**.

---

## Numerical method

| Component        | Implementation                                                        |
|------------------|-----------------------------------------------------------------------|
| Lattice          | D2Q9                                                                  |
| Collision        | BGK, single relaxation time                                           |
| Streaming        | Pull scheme, fused with collision in one kernel                       |
| Precision        | FP32                                                                  |
| Inlet            | Equilibrium distribution at uniform velocity `u0`                     |
| Outlet           | Zero gradient (copy of the neighbouring column)                       |
| Top and bottom   | Periodic                                                              |
| Cylinder         | Bounce back on solid nodes                                            |
| Stabilisation    | Local velocity magnitude clamp at 0.45 lattice units                  |
| Turbulence hook  | Smagorinsky eddy viscosity in the kernel; `Cs = 0` by default (pure BGK) |

### Default case

| Quantity            | Value                                  |
|---------------------|----------------------------------------|
| Grid                | 1600 × 400                             |
| Cylinder            | radius 48 cells, at x = Nx/5           |
| Relaxation time τ   | 0.57                                   |
| Kinematic viscosity | ν = (2τ − 1)/6 ≈ 0.0233                |
| Inlet velocity      | u0 = 0.20                              |
| Reynolds number     | Re = u0 · D / ν ≈ 823                  |

The cylinder is placed 2 cells above the channel centreline. This small asymmetry, together with a low amplitude random perturbation of the initial state, triggers shedding earlier than a perfectly symmetric setup.

---

## GPU pipeline

```
 ┌──────────────────────────┐
 │ lbm_step × n             │   one CUDA graph launch, ping pong buffers
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ curl_kernel              │   vorticity from central differences
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ gauss_blur × 2           │   3 × 3 display smoothing
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ colourise                │   LUT in constant memory, uint8 RGB output
 └────────────┬─────────────┘
              ▼
      single host copy  ──►  pygame surface
```

### Design notes

**CUDA graph batching.** A frame advances the simulation by `n` steps. The `n` kernel launches are captured once into a CUDA graph, cached per step count, and replayed with a single `graph.launch()`. If graph capture is unavailable, the code falls back to plain launches automatically.

**Ping pong without copies.** Two distribution buffers alternate roles each step. Kernel argument tuples for both directions are built once, so no Python objects are created per launch. Step counts per frame are kept even, which guarantees the current state always ends up back in the same buffer.

**Rendering on the GPU.** Vorticity, smoothing and colour lookup all happen in device memory. The colourise kernel writes its output transposed (x major), matching the `(width, height, 3)` layout that `pygame.surfarray` expects, so the host does no reshaping or colour work. Per frame transfer is about 1.9 MB of `uint8`.

**Adaptive playback.** The run starts at 2000 steps per frame and switches to 120 steps per frame once the peak vorticity crosses `0.5 × VMAX`. Note that the cylinder boundary layer alone exceeds this threshold, so with the default settings the switch happens at the first check (iteration 2000), before shedding has actually developed.

**Watchdog and profiler.** Every 30 frames the peak vorticity is checked for NaN or blow up, and the run stops cleanly on divergence. A timing breakdown (simulation, post processing, drawing) plus throughput in steps per second is printed at the same interval.

**Memory footprint.** Each distribution buffer is Nx × Ny × 9 × 4 bytes ≈ 23 MB, about 46 MB for both.

---

## Performance

| GPU         | Steps/s | MLUPS | FPS (smooth mode) |
|-------------|---------|-------|-------------------|
| _your GPU_  |         |       |                   |

MLUPS = steps/s × Nx × Ny / 10⁶. Steps/s is printed by the built in profiler.

---

## Getting started

### Requirements

- NVIDIA GPU with a CUDA 12 toolkit
- Python 3.9+
- `cupy`, `numpy`, `pygame`

```bash
pip install cupy-cuda12x numpy pygame
```

The script sets `CUDA_PATH` to a Windows CUDA 12.9 install at the top of the file. Edit or remove that line to match your system.

### Run

```bash
python LBM_fast_3.py
```

Press **Esc** or close the window to quit.

---

## Configuration

All parameters are at the top of the script.

| Parameter                 | Default   | Meaning                                         |
|---------------------------|-----------|-------------------------------------------------|
| `Nx`, `Ny`                | 1600, 400 | Lattice size                                    |
| `tau`                     | 0.57      | Relaxation time (sets viscosity)                |
| `u0_max`                  | 0.20      | Inlet velocity in lattice units                 |
| `STEPS_PER_FRAME_FAST`    | 2000      | Steps per frame during startup (must be even)   |
| `STEPS_PER_FRAME_SMOOTH`  | 120       | Steps per frame after shedding (must be even)   |
| `VMAX`                    | 0.05      | Vorticity mapped to the top of the colormap     |
| `BLUR_PASSES`             | 2         | Display smoothing passes (0, 1 or 2)            |
| `SCALE`                   | 1         | Window pixel scale                              |
| `CHECK_EVERY`             | 30        | Frames between divergence checks                |
| `PROFILE_EVERY`           | 30        | Frames between profiler printouts               |

---

## Gallery

| Startup transient (steps 0 to 30 000) | Developed shedding |
|:-----------------:|:------------------:|
| ![startup](docs/media/startup.gif) | ![shedding](docs/media/shedding.gif) |

Full run, 72 000 steps (one frame every 120 steps): [vortex_street.mp4](docs/media/vortex_street.mp4)
