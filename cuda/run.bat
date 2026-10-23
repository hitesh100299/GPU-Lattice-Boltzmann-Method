@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Lattice Boltzmann on the GPU

rem ==================================================================
rem  cuda\run.bat      (all executables go to cuda\build\)
rem    run.bat            builds lbm_cuda.exe (real-time demo, GLFW window) and runs it
rem    run.bat demo ...   same, extra arguments go to lbm_cuda.exe (see README, demo options)
rem    run.bat bench      builds lbm_bench.exe (FP32) and lbm_bench_fp64.exe (FP64)
rem    run.bat headless   builds lbm_cuda_headless.exe (demo without a window)
rem    run.bat tools      builds stream_copy.exe (bandwidth measurement)
rem    run.bat test       builds lbm_bench.exe and runs the regression test in tests\ (needs Python)
rem    run.bat baseline   builds karman_showcase_headless.exe (first CUDA version, for timing)
rem    run.bat release    builds the demo, the FP32 benchmark and the visual showcase
rem                       (showcase\karman_showcase.cu, not validated) for six GPU
rem                       generations (see ARCH_ALL below) and zips them into dist\
rem
rem  Finds MSVC (vswhere or cl.exe on PATH), nvcc and GLFW. Optional variables:
rem    FORCE_REBUILD=1   always recompile
rem    GLFW_ROOT=path    use this GLFW folder instead of searching
rem    KARMAN_NOPAUSE=1  do not wait for a key press on errors
rem ==================================================================

set "GLFW_VER=3.4"
set "VERSION=1.0.1"
set "WORKDIR=%~dp0"
cd /d "%WORKDIR%"
if not exist build mkdir build

rem native code for compute capability 7.5, 8.0, 8.6, 8.9, 9.0 and 12.0 (GeForce GTX 16, RTX 20 to 50 and
rem the workstation cards of those generations), plus PTX for 12.0 that newer GPUs compile at load time
set "ARCH_ALL=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=demo"
set "SRC_NAME="
if /i "%MODE%"=="demo"     (set "SRC_NAME=lbm_cuda.cu"          & set "EXE_NAME=build\lbm_cuda.exe"          & set "NEED_GLFW=1")
if /i "%MODE%"=="bench"    (set "SRC_NAME=lbm_bench.cu"         & set "EXE_NAME=build\lbm_bench.exe"         & set "NEED_GLFW=0")
if /i "%MODE%"=="headless" (set "SRC_NAME=lbm_cuda.cu"          & set "EXE_NAME=build\lbm_cuda_headless.exe" & set "NEED_GLFW=0")
if /i "%MODE%"=="tools"    (set "SRC_NAME=tools\stream_copy.cu" & set "EXE_NAME=build\stream_copy.exe"       & set "NEED_GLFW=0")
if /i "%MODE%"=="test"     (set "SRC_NAME=lbm_bench.cu"         & set "EXE_NAME=build\lbm_bench.exe"         & set "NEED_GLFW=0")
if /i "%MODE%"=="baseline" (set "SRC_NAME=showcase\karman_showcase.cu" & set "EXE_NAME=build\karman_showcase_headless.exe" & set "NEED_GLFW=0")
if /i "%MODE%"=="release"  (set "SRC_NAME=lbm_cuda.cu"          & set "EXE_NAME=build\release\lbm_cuda.exe"  & set "NEED_GLFW=1" & set "FORCE_REBUILD=1")
if not defined SRC_NAME goto :err_mode
echo [ok] Mode:    %MODE%  (%SRC_NAME% -^> %EXE_NAME%)

rem ---- 1. rebuild only if the exe is missing or older than any source
set "NEED_BUILD=0"
if "%FORCE_REBUILD%"=="1" set "NEED_BUILD=1"
if not exist "%EXE_NAME%" set "NEED_BUILD=1"
if "%NEED_BUILD%"=="0" powershell -NoProfile -Command "$e=(Get-Item '%EXE_NAME%').LastWriteTime; exit [int]((@('%SRC_NAME%','lbm_core.cuh','cases.h') | %%{ (Get-Item $_).LastWriteTime -gt $e }) -contains $true)"
if errorlevel 1 set "NEED_BUILD=1"
if "%NEED_BUILD%"=="0" if "%NEED_GLFW%"=="0" goto :after_build
if "%NEED_BUILD%"=="0" if exist "build\glfw3.dll" goto :after_build

rem ---- 2. locate GLFW (demo only; downloads it next to this script if not found)
set "GLFW_INC="
set "GLFW_LIBDIR="
if "%NEED_GLFW%"=="0" goto :skip_glfw
call :find_glfw
if not defined GLFW_DIR goto :err_glfw
echo [ok] GLFW:    %GLFW_DIR%
copy /y "%GLFW_LIB%\glfw3.dll" build\ >nul
set "GLFW_INC=%GLFW_DIR%\include"
set "GLFW_LIBDIR=%GLFW_LIB%"
:skip_glfw
if "%NEED_BUILD%"=="0" goto :after_build

rem ---- 3. locate the Visual Studio C++ compiler (cl.exe)
set "NVCC_EXTRA="
where cl.exe >nul 2>&1
if not errorlevel 1 goto :msvc_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :err_msvc
set "VSDIR="
set "VSTMP=%TEMP%\lbm_vswhere.txt"
"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -version "[17.0,18.0)" -property installationPath > "%VSTMP%" 2>nul
set /p VSDIR=<"%VSTMP%"
if defined VSDIR goto :have_vs
"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%VSTMP%" 2>nul
set /p VSDIR=<"%VSTMP%"
if not defined VSDIR goto :err_msvc
set "NVCC_EXTRA=-allow-unsupported-compiler"
:have_vs
del "%VSTMP%" >nul 2>&1
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" goto :err_msvc
set "VSCMD_START_DIR=%CD%"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%WORKDIR%"
where cl.exe >nul 2>&1
if errorlevel 1 goto :err_msvc
:msvc_ready
set "CL_PATH="
for /f "delims=" %%c in ('where cl.exe') do if not defined CL_PATH set "CL_PATH=%%c"
echo [ok] MSVC:    %CL_PATH%

rem ---- 4. locate nvcc: PATH, then CUDA_PATH, then the default install folder
set "NVCC="
for /f "delims=" %%n in ('where nvcc.exe 2^>nul') do if not defined NVCC set "NVCC=%%n"
if defined NVCC goto :nvcc_ready
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if defined NVCC goto :nvcc_ready
for /d %%d in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*") do if exist "%%~d\bin\nvcc.exe" set "NVCC=%%~d\bin\nvcc.exe"
if not defined NVCC goto :err_cuda
:nvcc_ready
echo [ok] nvcc:    %NVCC%

rem ---- 5. build
set "ARCH=-arch=native"
if /i "%MODE%"=="release" set "ARCH=%ARCH_ALL%"
set "DEFS="
if /i "%MODE%"=="headless" set "DEFS=-DHEADLESS"
if /i "%MODE%"=="baseline" set "DEFS=-DHEADLESS"
if /i "%MODE%"=="release" if not exist build\release mkdir build\release
echo [..] Compiling %SRC_NAME% ...
if "%NEED_GLFW%"=="1" (
    "%NVCC%" -O3 -std=c++17 %ARCH% %NVCC_EXTRA% %DEFS% "%SRC_NAME%" -o "%EXE_NAME%" -I "%GLFW_INC%" -L "%GLFW_LIBDIR%" -lglfw3dll -lopengl32
) else (
    "%NVCC%" -O3 -std=c++17 %ARCH% %NVCC_EXTRA% %DEFS% "%SRC_NAME%" -o "%EXE_NAME%"
)
if errorlevel 1 goto :err_build
if /i "%MODE%"=="bench" (
    echo [..] Compiling lbm_bench.cu with -DUSE_DOUBLE ...
    "%NVCC%" -O3 -std=c++17 %ARCH% %NVCC_EXTRA% -DUSE_DOUBLE lbm_bench.cu -o build\lbm_bench_fp64.exe
    if errorlevel 1 goto :err_build
)
if /i "%MODE%"=="release" (
    echo [..] Compiling lbm_bench.cu for all generations ...
    "%NVCC%" -O3 -std=c++17 %ARCH% %NVCC_EXTRA% lbm_bench.cu -o build\release\lbm_bench.exe
    if errorlevel 1 goto :err_build
    echo [..] Compiling showcase\karman_showcase.cu ^(visual showcase, not validated^) for all generations ...
    "%NVCC%" -O3 -std=c++17 %ARCH% %NVCC_EXTRA% showcase\karman_showcase.cu -o build\release\karman_showcase.exe -I "%GLFW_INC%" -L "%GLFW_LIBDIR%" -lglfw3dll -lopengl32
    if errorlevel 1 goto :err_build
)
del build\*.exp build\*.lib build\release\*.exp build\release\*.lib >nul 2>&1
echo [ok] Built:   %WORKDIR%%EXE_NAME%

:after_build
if /i "%MODE%"=="release" goto :package
if /i "%MODE%"=="test" goto :run_tests
if /i not "%MODE%"=="demo" (
    echo [ok] Done. Executables are in %WORKDIR%build\
    exit /b 0
)

rem ---- 6. run the demo (run.bat demo --no-run only builds it)
if /i "%~2"=="--no-run" exit /b 0
echo [..] Starting %EXE_NAME% %2 %3 %4 %5 %6 %7. Press ESC in the window to quit.
echo.
"%WORKDIR%%EXE_NAME%" %2 %3 %4 %5 %6 %7
set "RC=%errorlevel%"
if not "%RC%"=="0" goto :err_run
exit /b 0

rem ---- regression test
:run_tests
python "%WORKDIR%..\tests\run_tests.py" "%WORKDIR%build\lbm_bench.exe"
if errorlevel 1 goto :err_tests
exit /b 0

rem ---- 7. release package: dist\lbm-gpu-windows-x64-<version>.zip
:package
set "PKG=lbm-gpu-windows-x64-%VERSION%"
set "STAGE=%WORKDIR%..\dist\%PKG%"
if exist "%STAGE%" rmdir /s /q "%STAGE%"
mkdir "%STAGE%"
copy /y build\release\lbm_cuda.exe "%STAGE%\" >nul
copy /y build\release\lbm_bench.exe "%STAGE%\" >nul
copy /y build\release\karman_showcase.exe "%STAGE%\" >nul
copy /y "%GLFW_LIB%\glfw3.dll" "%STAGE%\" >nul
copy /y "%GLFW_DIR%\LICENSE.md" "%STAGE%\GLFW_LICENSE.md" >nul
copy /y "%WORKDIR%..\LICENSE" "%STAGE%\LICENSE.txt" >nul
copy /y "%WORKDIR%..\docs\release\README.txt" "%STAGE%\README.txt" >nul
copy /y "%WORKDIR%..\docs\release\run_demo.bat" "%STAGE%\" >nul
copy /y "%WORKDIR%..\docs\release\run_benchmark.bat" "%STAGE%\" >nul
copy /y "%WORKDIR%..\docs\release\Run showcase (not validated).bat" "%STAGE%\" >nul
if exist "%WORKDIR%..\dist\%PKG%.zip" del "%WORKDIR%..\dist\%PKG%.zip"
rem tar (built into Windows 10 and 11) writes standard zip paths; PowerShell 5.1 Compress-Archive
rem writes backslashes, which unzip tools other than Windows Explorer mishandle
rem full path: from Git Bash the PATH starts with GNU tar, which reads "D:" as a remote host
"%SystemRoot%\System32\tar.exe" -a -c -f "%WORKDIR%..\dist\%PKG%.zip" -C "%WORKDIR%..\dist" "%PKG%"
if errorlevel 1 goto :err_build
echo [ok] Package: %WORKDIR%..\dist\%PKG%.zip
exit /b 0

rem ==================================================================
rem  errors
rem ==================================================================
:err_tests
echo [error] The regression test failed, see the lines marked FAIL above.
goto :fail
:err_mode
echo [error] Unknown mode "%MODE%". Use: run.bat [demo ^| bench ^| headless ^| tools ^| test ^| release]
goto :fail
:err_glfw
echo [error] GLFW was not found and the automatic download failed.
echo         Get the 64-bit Windows binaries from https://www.glfw.org/download
echo         and unzip them next to this script, or set GLFW_ROOT to that folder.
goto :fail
:err_msvc
echo [error] No Visual Studio C++ compiler found.
echo         Install "Visual Studio 2022 Build Tools" with the "Desktop development with C++" workload.
goto :fail
:err_cuda
echo [error] nvcc not found. Install the CUDA Toolkit or set CUDA_PATH.
goto :fail
:err_build
echo [error] Compilation or packaging failed, see the messages above.
goto :fail
:err_run
echo [error] %EXE_NAME% exited with code %RC%.
goto :fail
:fail
echo.
if not "%KARMAN_NOPAUSE%"=="1" pause
exit /b 1

rem ==================================================================
rem  GLFW search helpers
rem ==================================================================
:find_glfw
set "GLFW_DIR="
set "GLFW_LIB="
for %%d in ("%GLFW_ROOT%" "%WORKDIR%glfw" "%WORKDIR%glfw-%GLFW_VER%.bin.WIN64" "%WORKDIR%..\..\glfw-%GLFW_VER%.bin.WIN64" "%WORKDIR%..\..\..\glfw-%GLFW_VER%.bin.WIN64" "C:\libs\glfw" "C:\glfw" "%USERPROFILE%\Downloads\glfw-%GLFW_VER%.bin.WIN64") do call :check_glfw "%%~d"
if defined GLFW_DIR exit /b 0
for /d %%d in ("%WORKDIR%glfw-*.bin.WIN64" "%USERPROFILE%\Downloads\glfw-*.bin.WIN64" "C:\libs\glfw-*.bin.WIN64") do call :check_glfw "%%~d"
if defined GLFW_DIR exit /b 0
call :download_glfw
exit /b 0

:check_glfw
if defined GLFW_DIR exit /b 0
if "%~1"=="" exit /b 0
if not exist "%~1\include\GLFW\glfw3.h" exit /b 0
set "GLFW_LIB="
for %%v in (vc2022 vc2019) do if not defined GLFW_LIB if exist "%~1\lib-%%v\glfw3dll.lib" set "GLFW_LIB=%~1\lib-%%v"
if not defined GLFW_LIB exit /b 0
set "GLFW_DIR=%~1"
exit /b 0

:download_glfw
set "ZIP=%WORKDIR%glfw-%GLFW_VER%.bin.WIN64.zip"
echo [..] GLFW not found, downloading GLFW %GLFW_VER% from GitHub ...
curl.exe -L --fail -s -o "%ZIP%" "https://github.com/glfw/glfw/releases/download/%GLFW_VER%/glfw-%GLFW_VER%.bin.WIN64.zip"
if errorlevel 1 exit /b 1
tar -xf "%ZIP%" -C "%WORKDIR%."
if errorlevel 1 powershell -NoProfile -Command "Expand-Archive -Force '%ZIP%' '%WORKDIR%.'"
del "%ZIP%" >nul 2>&1
call :check_glfw "%WORKDIR%glfw-%GLFW_VER%.bin.WIN64"
exit /b 0
