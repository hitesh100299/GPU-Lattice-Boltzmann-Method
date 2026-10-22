# Design notes

The engineering decisions behind the solver in `cuda/`, and why they were made. Results and their files are in [VALIDATION.md](VALIDATION.md) and [PERFORMANCE.md](PERFORMANCE.md).

## 1. Lattice conventions

Node (i, j) sits at x = i dx and y = (j − 0.5) dx. Rows 0 and Ny − 1 are solid, so with halfway bounce-back the walls lie exactly at y = 0 and y = H = Ny − 2. The cylinder is a staircase: a node is solid if (i − xc)² + (j − yc)² ≤ (Nd/2)², with xc = 2 Nd and yc = 2 Nd + 0.5 in node coordinates.

The pressure points A1 = (0.15, 0.2) and A2 = (0.25, 0.2) lie exactly halfway between rows 2 Nd and 2 Nd + 1, so the two rows are averaged with weight 0.5 each. Along x, the density is extrapolated linearly from the two nearest fluid nodes outside the cylinder. On these rows the nearest fluid node sits exactly at x = 0.15 and 0.25 for every grid, so the extrapolation distance is zero.

Grids are refined with tau held constant (diffusive scaling): the lattice velocity halves with every refinement, and the Mach number with it, so the compressibility error shrinks along with the discretisation error (Krüger et al. 2017, ch. 7).

## 2. Boundary conditions

The inlet is a Zou and He velocity condition with the analytic parabola, and the outlet a Zou and He pressure condition with ρ = 1 and zero transverse velocity (Zou and He 1997). The benchmark itself uses a "do nothing" outflow; a fixed-pressure outlet is simpler and reflects pressure waves, which shows up in notes 8 and 9. Inlet and outlet nodes next to the walls take their wall-facing populations through the same halfway bounce-back as interior nodes.

## 3. Initial state

Every column starts at the equilibrium of the analytic channel flow: the inlet parabola, zero velocity in solid nodes, and the matching linear density drop ρ(i) = 1 + 3 G (Nx − 1 − i) with G = 12 ν U_mean / H². That removes most of the start-up transient of a channel filling from rest. One 2D-1 run at 20 cells per diameter starts from rest (`--rest`) for comparison. It takes longer and ends at the same drag to within 5e-5.

## 4. What was dropped from the prototype

The first versions (the CuPy prototype and the first CUDA port) had six features that don't belong in a validated solver:

- a velocity clamp at 0.45, which hides divergence instead of reporting it;
- a density filter, which changes mass and never actually ran in the prototype's CUDA graph path;
- a Smagorinsky term with Cs = 0, which was dead code, and a laminar benchmark has no use for a turbulence model;
- random initial noise, which isn't reproducible;
- a uniform equilibrium inlet, which isn't the benchmark's parabola and sets the density wrongly;
- periodic top and bottom boundaries, where the benchmark has walls.

## 5. Shifted population storage

The first plane Poiseuille runs in single precision didn't converge on the finest channel: at H = 164 the mid-channel velocity kept wandering by about 5e-4 of U_max over millions of steps. The viscous stress lives in the non-equilibrium part f − feq, about 1e-5 at that lattice velocity, while the populations themselves are around 0.44. Single precision resolves that difference to only about 0.3 %, and the collision step loses it through cancellation.

The fix is to store g_i = f_i − w_i, the deviation from the fluid at rest (Skordos 1993; the same idea is used in Palabos and lbmpy). Streaming, bounce-back and the Zou and He relations keep their form, because w_i = w_opp(i) and the weights in the Zou and He density sums add up to one; ρ = 1 + Σ g. No parameter changed. After the fix, single and double precision give the same errors to three digits (see VALIDATION.md).

## 6. Poiseuille verification setup

A channel without the cylinder, Nx = 4 H + 1, tau = 0.65 as in 2D-1, and U_mean = 0.05 · 41 / H in lattice units. The mid-column profile is compared with the analytic parabola. The relative L2 error therefore also contains a small compressibility effect: ρ u is constant along the channel, so u is slightly higher where ρ is lower. A run is converged when the mid-column profile changes by less than 1e-6 (FP32) or 1e-10 (FP64) of U_max between checks spaced H²/(20 ν) steps apart, with 20 diffusion times as a limit. The FP32 runs at H = 82 and 164 reach that limit, because their profile change sits at the single-precision noise floor just above 1e-6; the FP64 runs settle to exactly zero change with the same error.

## 7. Convergence criteria

- **2D-1 (steady):** converged when the band (max − min) / mean of cD over the last 20 D/U drops below 1e-4. The limit is 400 D/U.
- **2D-2 (periodic):** periodic when at least 12 consistent shedding periods exist and the last three differ by less than 0.2 %. Then two more periods run (for the GIF frames) and the run stops; the limit is 160 D/U. The extrema and St come from the last three full periods, and St = 1 / period in D/U units. The 0.2 % spread is checked once, at detection. Over the last 12 periods the spread is larger: about 1.4 %, 0.7 % and 0.3 % at 40, 80 and 160 cells per diameter.
- Forces and pressures are sampled every round(steps per D/U / 50) steps, identically in the benchmark and the demo.

## 8. 2D-1 values are 20 D/U time means

At 40 and 80 cells per diameter the drag settles early, but a small oscillation never dies out in single precision, so the 1e-4 band criterion never fires and these runs end at the 400 D/U limit. The reported steady values are therefore the time means of cD, cL and Δp over the last 20 D/U, which average the oscillation out; the last sample is stored next to each mean in summary.json. cD and Δp hardly depend on the window. The lift does, by about ±0.15 % at 80 cells per diameter, because the oscillation is large compared with the small lift value. The 20-cell run meets the criterion and its mean equals its last sample to 3e-5.

The oscillation's period doesn't scale cleanly with the channel height, so the mode isn't identified. This is an inference: it's most likely an acoustic mode reflected between the velocity inlet and the pressure outlet, kept alive by single-precision round-off (note 10).

## 9. Period detection in 2D-2

During start-up the lift crosses zero many times: the cylinder starts impulsively in a flowing field, and pressure waves bounce between inlet and outlet. Early versions counted every crossing as a period and declared periodicity too soon, which gave a cD max far above the converged value at 80 cells per diameter. Now crossings closer than 0.5 D/U to the previous one are ignored, and only the trailing run of periods within 25 % of the last one counts. Together with the 12-period rule this keeps the start-up ripple out of the results.

## 10. Single-precision ripple in 2D-1

In double precision the 2D-1 run at 40 cells per diameter does meet the 1e-4 band criterion, while single precision doesn't within 400 D/U. The time means agree with the double-precision values to within 2e-5 relative in cD. I attribute the sustained ripple to single-precision round-off; it doesn't move the means.

## 11. Demo playback

The demo advances in chunks of exactly the benchmark's force-sampling interval: 40 chunks per frame before shedding is detected, 4 after. That's why it reproduces the benchmark to every stored digit. Shedding is detected from the transverse velocity at (xc + 2 D, yc), after two full oscillations above 5 % of U_mean. The colour scale is 10 U_mean / D by default, for |ω| with the original colour table and for signed ω with a red-blue map. The original display blur is off by default (`--blur 1` or `2` brings it back).

The demo shows St, and prints final values, only once the flow meets the benchmark's periodicity rule from note 7. Before that, the extrema are start-up values and would be misleading.

## 12. The showcase mode is unstable above Re 110

With `--re`, the demo runs other Reynolds numbers on the same geometry. At 80 cells per diameter it runs at Re 110 but diverges within about 1,500 steps at Re 120 and above. With this scaling tau = 0.5 + 6/Re on every grid, so refining doesn't help. This is an inference: the analytic channel-flow start puts sharp velocity jumps next to the cylinder, and BGK this close to tau = 0.5 can't damp them. A ramped inlet or a two-relaxation-time collision would most likely fix it.
