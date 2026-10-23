# Machine record

Recorded 2026-10-20 from the commands listed below; driver, toolkit and Python packages checked again on 2026-10-22. Every benchmark and validation number in this repository was produced on this machine.

| Item | Value | Source command |
|---|---|---|
| Laptop | ASUS ROG Zephyrus G15 GA503QM | `Get-CimInstance Win32_ComputerSystem` |
| CPU | AMD Ryzen 9 5900HS, 8 cores, 16 threads | `Get-CimInstance Win32_Processor` |
| RAM | 23.4 GiB (23,968 MB) | `Get-CimInstance Win32_ComputerSystem`, `systeminfo` |
| OS | Microsoft Windows 11 Home Single Language, 10.0.26300 | `Get-CimInstance Win32_OperatingSystem` |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, 30 SMs, compute capability 8.6 | `nvidia-smi --query-gpu=name,memory.total`, `cudaGetDeviceProperties` |
| NVIDIA driver | 617.42 (KMD 617.42, CUDA UMD 13.4) | `nvidia-smi` |
| CUDA toolkit (Windows) | 13.3, nvcc V13.3.73, build cuda_13.3.r13.3/compiler.38244171_0 | `nvcc --version` |
| Other CUDA toolkits present | 12.9 (in PATH, used by CuPy) | PATH inspection |
| Host compiler | MSVC 14.44.35207, Visual Studio 2022 Build Tools | `vcvars64.bat` |
| GLFW | 3.4 Windows binaries, next to the repository folder (found or downloaded by cuda\run.bat) | directory listing |
| Nsight Compute | 2026.2.1, default install folder, not on PATH (cuda/tools/perf.sh finds it) | directory listing |
| Python | 3.11.4 | `python --version` |
| CuPy | 14.1.1, CUDA build 12090, CUDA runtime 12090, driver 13040 | `cupy.show_config()` |
| NumPy | 2.2.2 | `numpy.__version__` |
| pygame | 2.6.1 (SDL 2.28.4) | `pygame.__version__` |
| matplotlib | 3.10.9 | `matplotlib.__version__` |
| Pillow | 12.2.0 | `PIL.__version__` |
| Power | AC adapter connected during all runs (PowerOnline True) | `Get-CimInstance BatteryStatus -Namespace root\wmi` |
| GPU idle state | P8, 210 MHz SM, 405 MHz memory, 49 C, 17.6 W | `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate` |

The Linux build (cuda/Makefile) is compiled by the CI on GitHub's hosted runners, which have no GPU; it hasn't been run on a Linux machine with a GPU.
