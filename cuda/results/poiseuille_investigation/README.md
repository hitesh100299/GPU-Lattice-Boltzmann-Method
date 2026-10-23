# Poiseuille verification: the first failure and its fix

These three trace logs date from 2026-10-20 and were produced by the solver before and right after the storage change described here; absolute paths were removed from them later. Setup as in [docs/DESIGN_NOTES.md](../../../docs/DESIGN_NOTES.md): a channel without the cylinder, halfway bounce-back walls, a Zou and He inlet (parabola) and outlet (ρ = 1), tau = 0.65, diffusive scaling U_mean,lat = 0.05 · 41 / H, and the mid-column profile compared with the analytic parabola.

## First attempt: plain populations f_i in single precision

| H | relative L2 error | last profile change per check (relative to U_max) | steps |
|---|---|---|---|
| 41 | 4.418e-3 | 1.0e-5 | 673,680 (limit) |
| 82 | 1.168e-3 | 2.3e-5 | 2,695,923 (limit) |
| 164 | 4.416e-4 | 2.4e-4 | 10,784,895 (limit) |

The pairwise orders were 1.92 and 1.40, where second order was expected. At H = 164 the mid-channel velocity never settled: it wandered between 1.00008 and 1.00065 U_max over ten million steps (`unshifted_fp32_H164_trace.log`, column `u_mid/umax`), while the single-precision noise floor at H = 41 is about 5e-6. So the H = 164 result was an unconverged transient, not a discretisation error.

The cause: at U_mean,lat = 0.0125 the viscous stress is carried by the non-equilibrium part f − feq, of order 1e-5, while the populations themselves are about 0.44. Single precision has about seven significant digits, so f − feq is resolved to roughly 0.3 %, and the collision step `f + ω (feq − f)` loses the stress through cancellation. The double-precision traces and the shifted single-precision runs below confirm this.

## Fix: shifted populations g_i = f_i − w_i (Skordos 1993)

The arrays now hold the deviation from the rest-state equilibrium. Streaming, halfway bounce-back and the Zou and He relations keep their form (w_i = w_opp(i), and the weights in the Zou and He density sums add up to one); ρ = 1 + Σ g and the momentum is Σ g c. No parameter changed: same tau, same grids, same boundary conditions.

The double-precision traces (`shifted_fp64_H82_trace.log`, `shifted_fp64_H164_trace.log`) show the profile change dropping to exactly zero, so those runs are converged to machine precision. The current errors in single and double precision are in [docs/VALIDATION.md](../../../docs/VALIDATION.md) and `../poiseuille_summary.json`; they agree to three digits.

What's left at a given H is the tau-dependent wall slip of BGK with halfway bounce-back plus the small compressibility effect of the density gradient along the channel (ρ u is constant, so u is higher where ρ is lower). Both scale with H⁻² under diffusive scaling, which is what the observed order shows (Ginzburg and d'Humières 2003; Krüger et al. 2017, ch. 5 and 7).
