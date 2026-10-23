## v1.0.0

A CUDA lattice Boltzmann solver for flow past a cylinder, validated against the DFG benchmark (Schäfer and Turek 1996), prebuilt for Windows.

**What's in the zip** (`lbm-gpu-windows-x64-1.0.0.zip`). Unzip it and double-click a launcher:

- `run_demo.bat`: the real-time demo, case 2D-2 (Re 100) at 80 cells per cylinder diameter, with drag, lift and shedding frequency in the window title.
- `run_benchmark.bat`: two short checks, plane Poiseuille flow and case 2D-2 at 40 cells per diameter, printed next to the reference values.
- `Run showcase (not validated).bat`: the first CUDA version at Re about 823, kept for its look. Its numbers aren't checked against any reference.

The zip also holds the three programs, `glfw3.dll` with its license, the MIT license and a short README.

**Requirements.** Windows 10 or 11 (64-bit) and an NVIDIA driver, 580 or newer. No CUDA toolkit, compiler or Python is needed. The programs contain native code for compute capability 7.5, 8.0, 8.6, 8.9, 9.0 and 12.0 (GeForce GTX 16 and RTX 20 to 50 series) plus PTX for 12.0.

**Tested on** an NVIDIA GeForce RTX 3060 Laptop GPU with driver 617.42. Other GPUs and drivers are untested.

**Results** for this version, the validation and the source: see the repository README.

**Checksum.** Built with `cuda\run.bat release` from commit 98ded99:

```
SHA256  1bb2a04bb1b85001bc8fda84d9fc8532e1d751eb97ebe25d35ffdefc202429f2
```

Check it on Windows with `certutil -hashfile lbm-gpu-windows-x64-1.0.0.zip SHA256`.
