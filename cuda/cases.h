// cases.h
// Single source of truth for the benchmark geometry, Reynolds numbers, lattice
// conversion and reference values. Plain C++, no CUDA, so it can be included by
// any driver and by host-side tools.
//
// Physical definition (Schäfer and Turek 1996, DFG benchmark, pages 3 and 4 of
// the paper, https://doi.org/10.1007/978-3-322-89849-4_39):
//   channel 2.2 m x 0.41 m, cylinder D = 0.1 m centred at (0.2, 0.2),
//   nu = 0.001 m^2/s, rho = 1, inlet u(y) = 4 Um y (H - y) / H^2.
//   2D-1: Um = 0.3, U_mean = 0.2, Re = 20, steady.
//   2D-2: Um = 1.5, U_mean = 1.0, Re = 100, periodic shedding.
#pragma once
#include <cmath>
#include <cstring>
#include <string>

namespace dfg {

// ---------------- reference values ----------------
// 2D-1: FeatFlow DFG benchmark 2D-1 page, computed by Nabh 1998 with high-order
// spectral methods, copied from the page:
// https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
constexpr double REF_2D1_CD = 5.57953523384;
constexpr double REF_2D1_CL = 0.010618948146;
constexpr double REF_2D1_DP = 0.11752016697;
// Paper acceptance ranges, Table 3 of the paper (lower bound / upper bound rows).
constexpr double RANGE_2D1_CD[2] = {5.5700, 5.5900};
constexpr double RANGE_2D1_CL[2] = {0.0104, 0.0110};
constexpr double RANGE_2D1_DP[2] = {0.1172, 0.1176};

// 2D-2: extracted from the FeatFlow "new reference results" zip
// (20141124_dfg_bench2_official.zip, downloaded by cuda/tools/extract_reference.py, file
// logs/results_cc2d_bench2_referencenew/bdforces_q2_lv6_dt4 and
// pointvalues_q2_lv6_dt4, Q2/P1disc, level 6, k = 1/1600, t in [9, 10]).
// Extraction script and all levels: cuda/reference/extracted_2d2_reference.json,
// produced by cuda/tools/extract_reference.py. Values below are copied from the
// lv6_dt4 entry of that file.
constexpr double REF_2D2_CD_MAX = 3.22739;
constexpr double REF_2D2_CD_MIN = 3.16426;
constexpr double REF_2D2_CL_MAX = 0.98658;   // cl_max_interp (parabolic interpolation of the reference samples); raw sample max 0.98657
constexpr double REF_2D2_CL_MIN = -1.02129;
constexpr double REF_2D2_ST     = 0.30184;
constexpr double REF_2D2_DP     = 2.48520;   // Delta p at t0 + 1/(2 f), t0 = time of cL max
// Paper acceptance ranges, Table 4 of the paper (lower bound / upper bound rows).
constexpr double RANGE_2D2_CD_MAX[2] = {3.2200, 3.2400};
constexpr double RANGE_2D2_CL_MAX[2] = {0.9900, 1.0100};
constexpr double RANGE_2D2_ST[2]     = {0.2950, 0.3050};
constexpr double RANGE_2D2_DP[2]     = {2.4600, 2.5000};

// ---------------- physical constants ----------------
constexpr double L_PHYS  = 2.2;     // channel length [m]
constexpr double H_PHYS  = 0.41;    // channel height [m]
constexpr double D_PHYS  = 0.1;     // cylinder diameter [m]
constexpr double XC_PHYS = 0.2;     // cylinder centre x [m]
constexpr double YC_PHYS = 0.2;     // cylinder centre y [m]
constexpr double NU_PHYS = 0.001;   // kinematic viscosity [m^2/s]
constexpr double XA_PHYS = 0.15;    // pressure probe A1 (front)
constexpr double XE_PHYS = 0.25;    // pressure probe A2 (rear)

struct Case {
    std::string name;        // "2D-1", "2D-2", "poiseuille", "showcase"
    // physical
    double Umax_phys = 0, Umean_phys = 0, Re = 0;
    // lattice
    int    Nd = 0;           // cells per diameter (0 for poiseuille)
    int    Nx = 0, Ny = 0;   // grid including the two wall rows and the inlet/outlet columns
    double H  = 0;           // fluid height in lattice units = Ny - 2
    double dx = 0;           // [m] per lattice cell (0 for poiseuille)
    double umean_lat = 0, umax_lat = 0, nu_lat = 0, tau = 0, omega = 0;
    bool   cylinder = false;
    double xc = 0, yc = 0, r2 = 0;   // centre and squared radius in lattice units
    // derived
    double steps_per_DU = 0;         // lattice steps per convective time D / U_mean
    double dt_phys = 0;              // seconds per lattice step
    double mach = 0;                 // peak Mach number umax_lat / cs, cs = 1/sqrt(3)

    // conversions (lattice rho = 1 = physical rho)
    double force_to_coeff(double F_lat) const { return 2.0 * F_lat / (umean_lat * umean_lat * Nd); }
    double dp_lat_to_phys(double dp_lat) const { double s = Umean_phys / umean_lat; return dp_lat * s * s; }
    double steps_to_DU(double steps) const { return steps / steps_per_DU; }
    double steps_to_seconds(double steps) const { return steps * dt_phys; }
    // vorticity in units of U_mean / D from lattice vorticity
    double vort_scale() const { return Nd / umean_lat; }
    // analytic inlet profile at lattice row j (node y = j - 0.5), rows 1..Ny-2
    double inlet_u(int j) const { double y = j - 0.5; return 4.0 * umax_lat * y * (H - y) / (H * H); }
};

inline void finish(Case& c) {
    c.H = c.Ny - 2;
    c.omega = 1.0 / c.tau;
    c.umax_lat = 1.5 * c.umean_lat;
    c.mach = c.umax_lat * std::sqrt(3.0);
    if (c.Nd > 0) {
        c.steps_per_DU = c.Nd / c.umean_lat;
        c.dt_phys = c.dx * c.umean_lat / c.Umean_phys;
    }
}

// Diffusive scaling: tau fixed, U_lat shrinks with refinement (Krueger et al. 2017, ch. 7).
//   2D-1: U_mean,lat = 0.05 * 20 / Nd, nu_lat = 0.05, tau = 0.65
//   2D-2: U_mean,lat = 0.05 * 40 / Nd, nu_lat = 0.02, tau = 0.56
inline Case make_dfg(int which, int Nd) {
    Case c;
    c.name = which == 1 ? "2D-1" : "2D-2";
    c.Re = which == 1 ? 20.0 : 100.0;
    c.Umean_phys = which == 1 ? 0.2 : 1.0;
    c.Umax_phys = 1.5 * c.Umean_phys;
    c.Nd = Nd;
    c.dx = D_PHYS / Nd;
    c.Nx = (int)std::lround(L_PHYS / c.dx) + 1;
    c.Ny = (int)std::lround(H_PHYS / c.dx) + 2;
    c.umean_lat = which == 1 ? 0.05 * 20.0 / Nd : 0.05 * 40.0 / Nd;
    c.nu_lat = c.umean_lat * Nd / c.Re;
    c.tau = 3.0 * c.nu_lat + 0.5;
    c.cylinder = true;
    c.xc = XC_PHYS / c.dx;           // = 2 Nd
    c.yc = YC_PHYS / c.dx + 0.5;     // node (i, j) sits at y = (j - 0.5) dx
    c.r2 = 0.25 * Nd * (double)Nd;
    finish(c);
    return c;
}

// Showcase: 2D-2 geometry at the given Nd, Reynolds number chosen by the user,
// same U_mean,lat as the validated 2D-2 case at that Nd. Not validated.
inline Case make_showcase(int Nd, double Re) {
    Case c = make_dfg(2, Nd);
    c.name = "showcase";
    c.Re = Re;
    c.nu_lat = c.umean_lat * Nd / Re;
    c.tau = 3.0 * c.nu_lat + 0.5;
    finish(c);
    return c;
}

// Plane Poiseuille verification: no cylinder, fluid height H cells, Nx = 4 H + 1,
// tau = 0.65 as in 2D-1, diffusive scaling U_mean,lat = 0.05 * 41 / H.
inline Case make_poiseuille(int H) {
    Case c;
    c.name = "poiseuille";
    c.Nx = 4 * H + 1;
    c.Ny = H + 2;
    c.tau = 0.65;
    c.nu_lat = (c.tau - 0.5) / 3.0;
    c.umean_lat = 0.05 * 41.0 / H;
    c.Re = c.umean_lat * H / c.nu_lat;   // Reynolds number based on H, 41
    c.cylinder = false;
    finish(c);
    return c;
}

} // namespace dfg
