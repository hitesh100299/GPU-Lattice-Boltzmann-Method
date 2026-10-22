@echo off
rem Visual showcase, NOT VALIDATED: the first CUDA version at Re about 823
rem (lattice Mach 0.35, velocity clamp 0.45, periodic top and bottom).
rem It looks good and is fast, but its numbers are not checked against any reference.
rem The validated simulation is run_demo.bat. ESC quits.
cd /d "%~dp0"
"%~dp0karman_showcase.exe" %*
if errorlevel 1 pause
