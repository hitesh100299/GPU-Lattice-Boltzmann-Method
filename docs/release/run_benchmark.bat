@echo off
rem Two short checks, about one minute in total on a mid-range GPU:
rem   1. plane Poiseuille flow at 41 cells, compared with the exact parabola
rem   2. benchmark case 2D-2 (Re 100) at 40 cells per diameter, compared with the
rem      published reference values (coarse grid, so errors of a few percent are expected;
rem      the converged results at finer grids are on the GitHub page)
cd /d "%~dp0"
if not exist results mkdir results
if not exist results\poiseuille_H41 mkdir results\poiseuille_H41
if not exist results\2D-2_N40 mkdir results\2D-2_N40
echo.
echo === 1. Poiseuille flow, 41 cells across the channel ===
"%~dp0lbm_bench.exe" --case poiseuille --H 41 --out results\poiseuille_H41 | findstr /b /c:"poiseuille" /c:"GPU"
echo.
echo === 2. Cylinder benchmark 2D-2, 40 cells per diameter ===
"%~dp0lbm_bench.exe" --case 2D-2 --nd 40 --out results\2D-2_N40 | findstr /b /c:"2D-2" /c:"converged" /c:"wall"
echo.
echo Full output and time series are in the results folder next to this file.
pause
