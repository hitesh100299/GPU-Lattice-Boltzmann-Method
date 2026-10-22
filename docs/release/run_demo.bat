@echo off
rem Live demo: DFG benchmark case 2D-2 (Re 100) at 80 cells per diameter. ESC quits.
cd /d "%~dp0"
"%~dp0lbm_cuda.exe" %*
if errorlevel 1 pause
