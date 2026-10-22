GPU lattice Boltzmann solver, Windows x64 build
================================================

Flow past a cylinder computed on the GPU in real time. The solver is checked
against the Schäfer and Turek cylinder benchmark; results, method and source:
https://github.com/hitesh100299/GPU-Lattice-Boltzmann-Method

What you need
-------------
- Windows 10 or 11, 64-bit.
- An NVIDIA GeForce GTX 16 or RTX 20, 30, 40 or 50 series GPU, or a workstation
  card of those generations (compute capability 7.5, 8.0, 8.6, 8.9, 9.0, or 12.0
  and newer), with about 200 MB of free GPU memory for the default case.
- NVIDIA driver 580 or newer (NVIDIA's minimum for CUDA 13.x programs; the
  build was tested only with driver 617.42 on a GeForce RTX 3060 Laptop GPU,
  other GPUs and drivers are untested).
- No CUDA toolkit, compiler or Python is needed.

Files
-----
  run_demo.bat        double-click: opens the live demo window (ESC quits)
  run_benchmark.bat   double-click: runs two short checks and prints the
                      results next to the published reference values
  Run showcase (not validated).bat
                      double-click: the first CUDA version at Re about 823.
                      NOT VALIDATED: lattice Mach 0.35, velocity clamp 0.45,
                      periodic top and bottom; its numbers are not checked
                      against any reference
  lbm_cuda.exe        the demo; options below
  lbm_bench.exe       the benchmark driver
  karman_showcase.exe the visual showcase (not validated)
  glfw3.dll           window library used by the demo (GLFW, see GLFW_LICENSE.md)
  LICENSE.txt         MIT license of this project

Demo options (lbm_cuda.exe)
---------------------------
  --nd <n>       cells per cylinder diameter, default 80 (the validated case)
  --signed       signed vorticity with a blue-red colour map
  --vmax <v>     colour scale in units of U/D, default 10
  --blur <n>     display smoothing passes, 0 to 2, default 0
  --re <value>   other Reynolds number on the same geometry: NOT validated.
                 Tested: runs at --re 110; stops with "Diverged" at 120 and
                 above, because the simple collision model is unstable there.

The window title shows the live drag (cD), lift (cL) and shedding frequency
(St) of the cylinder. With the default settings these are the quantities that
were compared with the benchmark.

If the window does not open
---------------------------
On laptops with two GPUs, Windows may start the program on the integrated GPU.
The demo asks for the NVIDIA GPU itself; if that fails, set lbm_cuda.exe to
"High performance" in Windows Settings > System > Display > Graphics.
