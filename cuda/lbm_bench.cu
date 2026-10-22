// lbm_bench.cu
// Headless benchmark driver: the Poiseuille verification case, the DFG
// cylinder cases 2D-1 and 2D-2, and timing runs.
//
//   lbm_bench --case poiseuille --H 82 --out results/poiseuille_H82_fp32
//   lbm_bench --case 2D-1 --nd 40 --out results/2D-1_N40_fp32 [--rest]
//   lbm_bench --case 2D-2 --nd 80 --out results/2D-2_N80_fp32 [--frames]
//   lbm_bench --case 2D-2 --nd 80 --perf --steps 20000 --warmup 2000 --out results/perf_run1
//   lbm_bench --case 2D-2 --nd 160 --estimate        (prints the step count and memory, exits)
// Options: --cap <D/U>  override the time limit (default 400 for 2D-1, 160 for 2D-2)
//          --tag <text> free text stored in summary.json (git hash, command line)
// Exit codes: 0 finished, 2 divergence, 3 time limit reached before convergence,
//             4 output folder or summary.json could not be written.
#include "lbm_core.cuh"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static std::string shell(const char* cmd) {
    std::string out; char buf[512];
#ifdef _WIN32
    FILE* p = _popen(cmd, "r");
#else
    FILE* p = popen(cmd, "r");
#endif
    if (!p) return "Unknown";
    while (std::fgets(buf, sizeof buf, p)) out += buf;
#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out.empty() ? "Unknown" : out;
}
static std::string gpu_query() {
    return shell("nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader");
}
static std::string json_str(const std::string& s) {
    std::string o = "\""; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if (c == '\n') o += "\\n"; else o += c; } return o + "\"";
}
static std::string jnum(double v) { char b[64]; if (std::isfinite(v)) std::snprintf(b, sizeof b, "%.10g", v); else std::snprintf(b, sizeof b, "null"); return b; }

struct Args {
    std::string cse = "2D-2", out = ".", tag;
    int nd = 40, H = 82, steps = 20000, warmup = 2000;
    bool rest = false, frames = false, perf = false, estimate = false;
    double cap = -1;
};
static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(1); } return std::string(argv[++i]); };
        if (k == "--case") a.cse = next(); else if (k == "--nd") a.nd = std::atoi(next().c_str());
        else if (k == "--H") a.H = std::atoi(next().c_str()); else if (k == "--out") a.out = next();
        else if (k == "--tag") a.tag = next(); else if (k == "--steps") a.steps = std::atoi(next().c_str());
        else if (k == "--warmup") a.warmup = std::atoi(next().c_str()); else if (k == "--cap") a.cap = std::atof(next().c_str());
        else if (k == "--rest") a.rest = true; else if (k == "--frames") a.frames = true;
        else if (k == "--perf") a.perf = true; else if (k == "--estimate") a.estimate = true;
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); std::exit(1); }
    }
    return a;
}

static void write_fields(const std::string& dir, Sim& sim) {
    std::vector<float> rho, ux, uy, vort;
    sim.fields(rho, ux, uy); sim.vorticity(vort);
    std::ofstream f(dir + "/fields.bin", std::ios::binary);
    f.write((const char*)rho.data(), sizeof(float) * rho.size());
    f.write((const char*)ux.data(), sizeof(float) * ux.size());
    f.write((const char*)uy.data(), sizeof(float) * uy.size());
    f.write((const char*)vort.data(), sizeof(float) * vort.size());
    std::ofstream j(dir + "/fields.json");
    j << "{\"Nx\": " << sim.cs.Nx << ", \"Ny\": " << sim.cs.Ny << ", \"arrays\": [\"rho\", \"ux\", \"uy\", \"vort_lat\"], \"dtype\": \"float32\", \"layout\": \"row major, idx = y * Nx + x\", \"vort_scale_to_U_over_D\": "
      << jnum(sim.cs.vort_scale()) << ", \"umean_lat\": " << jnum(sim.cs.umean_lat) << ", \"step\": " << sim.step_count << "}\n";
    // node types once, so the plotting script can draw the cylinder and walls
    std::ofstream t(dir + "/types.bin", std::ios::binary);
    t.write((const char*)sim.h_type.data(), sim.h_type.size());
}

static std::string case_json(const dfg::Case& c) {
    std::ostringstream o;
    o << "{\"name\": " << json_str(c.name) << ", \"Nd\": " << c.Nd << ", \"Nx\": " << c.Nx << ", \"Ny\": " << c.Ny
      << ", \"H_lat\": " << jnum(c.H) << ", \"dx_m\": " << jnum(c.dx) << ", \"Re\": " << jnum(c.Re)
      << ", \"Umean_phys\": " << jnum(c.Umean_phys) << ", \"umean_lat\": " << jnum(c.umean_lat) << ", \"umax_lat\": " << jnum(c.umax_lat)
      << ", \"nu_lat\": " << jnum(c.nu_lat) << ", \"tau\": " << jnum(c.tau) << ", \"mach_peak\": " << jnum(c.mach)
      << ", \"cylinder\": " << (c.cylinder ? "true" : "false") << ", \"xc_lat\": " << jnum(c.xc) << ", \"yc_lat\": " << jnum(c.yc)
      << ", \"steps_per_DU\": " << jnum(c.steps_per_DU) << ", \"dt_phys_s\": " << jnum(c.dt_phys) << "}";
    return o.str();
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    // create the result folder; without it every output file would fail silently
    std::error_code fs_err;
    std::filesystem::create_directories(a.out, fs_err);
    if (fs_err) { std::fprintf(stderr, "error: cannot create the output folder %s: %s\n", a.out.c_str(), fs_err.message().c_str()); return 4; }
    // the program is recorded by file name only, so logs carry no absolute paths
    std::string cmdline = std::filesystem::path(argv[0]).filename().string();
    for (int i = 1; i < argc; i++) { cmdline += ' '; cmdline += argv[i]; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    std::printf("GPU: %s (%d SMs), driver API %d, runtime %d, precision %s\n", prop.name, prop.multiProcessorCount, drv, rt, PRECISION_NAME);
    std::printf("command: %s\n", cmdline.c_str());

    dfg::Case cs;
    if (a.cse == "2D-1") cs = dfg::make_dfg(1, a.nd);
    else if (a.cse == "2D-2") cs = dfg::make_dfg(2, a.nd);
    else if (a.cse == "poiseuille") cs = dfg::make_poiseuille(a.H);
    else { std::fprintf(stderr, "unknown case %s\n", a.cse.c_str()); return 1; }
    const bool steady = (a.cse == "2D-1");
    double cap_DU = a.cap > 0 ? a.cap : (steady ? 400.0 : 160.0);
    std::printf("case %s: grid %d x %d, tau %.4f, U_mean,lat %.5f, nu_lat %.5f, Mach %.3f, steps per D/U %.1f\n",
                cs.name.c_str(), cs.Nx, cs.Ny, cs.tau, cs.umean_lat, cs.nu_lat, cs.mach, cs.steps_per_DU);
    if (a.estimate) {
        const double steps = cap_DU * cs.steps_per_DU;
        std::printf("estimate: cap %.0f D/U = %.3g steps, %.3g cell updates, about %.1f min at 3000 MLUPS; memory %.0f MB\n",
                    cap_DU, steps, steps * cs.Nx * cs.Ny, steps * cs.Nx * cs.Ny / 3000e6 / 60.0, 2.0 * 9 * sizeof(real) * cs.Nx * cs.Ny / 1e6);
        return 0;
    }
    const std::string gpu0 = gpu_query();
    std::printf("gpu at start: %s\n", gpu0.c_str());

    Sim sim(cs, a.rest);
    std::printf("links on the cylinder: %d, probe nodes: %d\n", sim.nlinks, sim.nprobe);
    const auto t_start = Clock::now();
    std::ostringstream res;      // results part of summary.json
    int exit_code = 0;

    // ------------------------------------------------------------ perf
    if (a.perf) {
        sim.step(a.warmup); sim.kernel_ms = 0; sim.step_count = 0;
        const auto t0 = Clock::now();
        sim.step(a.steps);
        const double wall_a = secs(t0, Clock::now());
        const double ml_kernel = sim.mlups_kernel();
        const double ml_wall_kernel_only = a.steps * (double)sim.N / wall_a / 1e6;
        // with diagnostics at the normal cadence (force + probes every ~1/50 D/U)
        const int every = std::max(1, (int)std::lround(cs.steps_per_DU / 50.0));
        Diag d; sim.kernel_ms = 0; sim.step_count = 0;
        const auto t1 = Clock::now();
        for (int s = 0; s < a.steps; s += every) { const int n = std::min(every, a.steps - s); sim.step(n); d.push(cs.steps_to_DU(sim.step_count), sim.sample()); }
        const double wall_b = secs(t1, Clock::now());
        const double ml_diag = a.steps * (double)sim.N / wall_b / 1e6;
        const std::string gpu1 = gpu_query();
        std::printf("perf: %d steps, kernel only %.1f MLUPS (events), %.1f MLUPS (wall), with diagnostics every %d steps %.1f MLUPS (wall)\n",
                    a.steps, ml_kernel, ml_wall_kernel_only, every, ml_diag);
        res << "\"perf\": {\"steps\": " << a.steps << ", \"warmup\": " << a.warmup << ", \"mlups_kernel_events\": " << jnum(ml_kernel)
            << ", \"mlups_kernel_wall\": " << jnum(ml_wall_kernel_only) << ", \"mlups_with_diagnostics\": " << jnum(ml_diag)
            << ", \"sample_every_steps\": " << every << ", \"bytes_per_cell_update\": 73, \"gpu_end\": " << json_str(gpu1) << "}";
    }
    // ------------------------------------------------------------ poiseuille
    else if (a.cse == "poiseuille") {
        const int H = a.H, Nx = cs.Nx, Ny = cs.Ny, mid = Nx / 2;
        const double tdiff = H * (double)H / cs.nu_lat;          // viscous diffusion time in steps
        const int chk = std::max(100, (int)(tdiff / 20));
        const double tol = sizeof(real) == 4 ? 1e-6 : 1e-10;
        std::vector<float> rho, ux, uy, prev(Ny, 0.f);
        double change = 1; long long cap = (long long)(20 * tdiff);
        while (sim.step_count < cap) {
            sim.step(chk);
            sim.fields(rho, ux, uy);
            change = 0;
            for (int j = 1; j < Ny - 1; j++) { change = std::max(change, (double)std::fabs(ux[j * Nx + mid] - prev[j])); prev[j] = ux[j * Nx + mid]; }
            change /= cs.umax_lat;
            std::printf("  step %10lld  change %.3e  rho_in %.6f rho_mid %.6f rho_out %.6f  u_mid/umax %.7f\n", sim.step_count, change,
                        (double)rho[(Ny / 2) * Nx + 1], (double)rho[(Ny / 2) * Nx + mid], (double)rho[(Ny / 2) * Nx + Nx - 2], (double)ux[(Ny / 2) * Nx + mid] / cs.umax_lat);
            if (change < tol) break;
        }
        // compare with the analytic parabola at the mid column
        double num = 0, den = 0, umax_sim = 0, uymax = 0;
        std::ofstream pf(a.out + "/profile.csv"); pf << "j,y_lat,u_sim,u_analytic,rho\n";
        for (int j = 1; j < Ny - 1; j++) {
            const double ua = cs.inlet_u(j), us = ux[j * Nx + mid];
            num += (us - ua) * (us - ua); den += ua * ua; umax_sim = std::max(umax_sim, us);
            uymax = std::max(uymax, (double)std::fabs(uy[j * Nx + mid]));
            pf << j << "," << (j - 0.5) << "," << jnum(us) << "," << jnum(ua) << "," << jnum(rho[j * Nx + mid]) << "\n";
        }
        const double l2 = std::sqrt(num / den);
        std::printf("poiseuille H %d: steps %lld, last change %.3g, relative L2 error %.5e, umax sim/analytic %.6f, max |uy| %.3g, rho mid %.6f\n",
                    H, sim.step_count, change, l2, umax_sim / cs.umax_lat, uymax, (double)rho[(Ny / 2) * Nx + mid]);
        res << "\"poiseuille\": {\"H\": " << H << ", \"steps\": " << sim.step_count << ", \"last_change_rel\": " << jnum(change) << ", \"tol\": " << jnum(tol)
            << ", \"l2_rel_error\": " << jnum(l2) << ", \"umax_ratio\": " << jnum(umax_sim / cs.umax_lat) << ", \"max_abs_uy\": " << jnum(uymax) << ", \"column\": " << mid << "}";
        write_fields(a.out, sim);
    }
    // ------------------------------------------------------------ DFG cases
    else {
        const int every = std::max(1, (int)std::lround(cs.steps_per_DU / 50.0));
        const long long cap_steps = (long long)(cap_DU * cs.steps_per_DU);
        std::printf("sampling every %d steps, cap %.0f D/U = %lld steps\n", every, cap_DU, cap_steps);
        const double mass0 = sim.total_mass();
        Diag d;
        std::ofstream fc(a.out + "/forces.csv"); fc << "step,t_DU,t_s,cd,cl,dp_phys,Fx_lat,Fy_lat\n";
        bool converged = false, diverged = false; double t_conv = NAN; long long steps_conv = 0;
        double next_report = 0; const auto t_loop = Clock::now();
        Sim::Sample last_ok{};
        // frames (2D-2, --frames): vorticity every period/24 steps over the 2 extra periods
        std::ofstream ff; int nframes = 0; std::vector<long long> frame_steps; long long next_frame = -1, frame_every = 0;
        long long stop_at = -1;
        while (sim.step_count < cap_steps) {
            sim.step(every);
            const Sim::Sample s = sim.sample();
            const double tDU = cs.steps_to_DU(sim.step_count);
            fc << sim.step_count << "," << jnum(tDU) << "," << jnum(cs.steps_to_seconds(sim.step_count)) << "," << jnum(s.cd) << "," << jnum(s.cl) << "," << jnum(s.dp_phys) << "," << jnum(s.Fx) << "," << jnum(s.Fy) << "\n";
            if (!s.finite) { diverged = true; break; }
            last_ok = s;
            d.push(tDU, s);
            if (stop_at >= 0 && a.frames && sim.step_count >= next_frame) {
                std::vector<float> v; sim.vorticity(v);
                ff.write((const char*)v.data(), sizeof(float) * v.size()); frame_steps.push_back(sim.step_count); nframes++;
                next_frame += frame_every;
            }
            if (tDU >= next_report) {
                const double el = secs(t_loop, Clock::now());
                std::printf("t %7.2f D/U  step %10lld  cD %.6f  cL %+.6f  dp %.6f  periods %d  band20 %.2e  %.0f MLUPS  %.0fs\n",
                            tDU, sim.step_count, s.cd, s.cl, s.dp_phys, d.n_periods(), d.cd_band(20.0), sim.mlups_kernel(), el);
                std::fflush(stdout);
                next_report += steady ? 5.0 : 2.0;
            }
            if (!converged) {
                if (steady) {
                    if (tDU >= 20.0 && d.cd_band(20.0) < 1e-4) { converged = true; t_conv = tDU; steps_conv = sim.step_count; break; }
                } else if (d.n_periods() >= 12 && d.period_spread(3) < 0.002) {
                    converged = true; t_conv = tDU; steps_conv = sim.step_count;
                    const long long period_steps = (long long)std::lround(d.period_mean(3) * cs.steps_per_DU);
                    stop_at = sim.step_count + 2 * period_steps;
                    frame_every = std::max<long long>(every, period_steps / 24); next_frame = sim.step_count;
                    if (a.frames) ff.open(a.out + "/frames.bin", std::ios::binary);
                    std::printf("converged at %.2f D/U (%d periods, period %.4f D/U, spread %.2e); 2 more periods to step %lld\n",
                                tDU, d.n_periods(), d.period_mean(3), d.period_spread(3), stop_at);
                }
            }
            // not converged by the criterion but close to the cap: still record the last 2 periods
            if (!steady && !converged && stop_at < 0 && d.n_periods() >= 3) {
                const long long period_steps = (long long)std::lround(d.period_mean(3) * cs.steps_per_DU);
                if (sim.step_count >= cap_steps - 2 * period_steps) {
                    stop_at = cap_steps;
                    frame_every = std::max<long long>(every, period_steps / 24); next_frame = sim.step_count;
                    if (a.frames) ff.open(a.out + "/frames.bin", std::ios::binary);
                    std::printf("cap approaching without convergence by the criterion at %.2f D/U (%d consistent periods, period %.4f D/U, spread %.2e); recording the last 2 periods\n",
                                tDU, d.n_periods(), d.period_mean(3), d.period_spread(3));
                }
            }
            if (stop_at >= 0 && sim.step_count >= stop_at) break;
        }
        const double wall = secs(t_loop, Clock::now());
        const double mass1 = sim.total_mass();
        const std::string gpu1 = gpu_query();
        std::printf("gpu at end: %s\n", gpu1.c_str());
        std::printf("wall %.1f s, %lld steps, kernel %.0f MLUPS, incl. diagnostics %.0f MLUPS, mass drift %.3e\n",
                    wall, sim.step_count, sim.mlups_kernel(), sim.step_count * (double)sim.N / wall / 1e6, (mass1 - mass0) / mass0);
        res << "\"run\": {\"converged\": " << (converged ? "true" : "false") << ", \"diverged\": " << (diverged ? "true" : "false")
            << ", \"t_conv_DU\": " << jnum(t_conv) << ", \"steps_conv\": " << steps_conv << ", \"steps\": " << sim.step_count
            << ", \"t_end_DU\": " << jnum(cs.steps_to_DU(sim.step_count)) << ", \"t_end_s\": " << jnum(cs.steps_to_seconds(sim.step_count))
            << ", \"cap_DU\": " << jnum(cap_DU) << ", \"sample_every_steps\": " << every << ", \"from_rest\": " << (a.rest ? "true" : "false")
            << ", \"wall_s\": " << jnum(wall) << ", \"mlups_kernel_events\": " << jnum(sim.mlups_kernel())
            << ", \"mlups_with_diagnostics\": " << jnum(sim.step_count * (double)sim.N / wall / 1e6)
            << ", \"mass_start\": " << jnum(mass0) << ", \"mass_end\": " << jnum(mass1) << ", \"mass_drift_rel\": " << jnum((mass1 - mass0) / mass0)
            << ", \"nlinks\": " << sim.nlinks << ", \"gpu_end\": " << json_str(gpu1) << "},\n";
        if (diverged) {
            std::printf("DIVERGED: non-finite force or pressure at step %lld (tau %.4f). Last valid: cD %.6f cL %.6f dp %.6f\n",
                        sim.step_count, cs.tau, last_ok.cd, last_ok.cl, last_ok.dp_phys);
            res << "\"diverged\": {\"step\": " << sim.step_count << ", \"tau\": " << jnum(cs.tau) << ", \"last_cd\": " << jnum(last_ok.cd)
                << ", \"last_cl\": " << jnum(last_ok.cl) << ", \"last_dp\": " << jnum(last_ok.dp_phys) << "},\n";
            exit_code = 2;
        } else if (!converged) exit_code = 3;
        if (steady) {
            // steady values: time mean over the last 20 D/U (a small sustained acoustic oscillation
            // can keep the band above the stop threshold, see docs/DESIGN_NOTES.md); last sample kept too
            double cd = 0, cl = 0, dp = 0; int cnt = 0;
            for (size_t i = 0; i < d.t.size(); i++) if (d.t[i] >= d.t.back() - 20.0) { cd += d.cd[i]; cl += d.cl[i]; dp += d.dp[i]; cnt++; }
            cd /= cnt; cl /= cnt; dp /= cnt;
            std::printf("2D-1 last sample: cD %.6f cL %.6f dp %.6f; means over the last 20 D/U (%d samples) follow\n", last_ok.cd, last_ok.cl, last_ok.dp_phys, cnt);
            std::printf("2D-1 result: cD %.6f (ref %.6f, err %+.3f%%)  cL %.6f (ref %.6f, err %+.2f%%)  dp %.6f (ref %.6f, err %+.3f%%)  band20 %.2e\n",
                        cd, dfg::REF_2D1_CD, 100 * (cd / dfg::REF_2D1_CD - 1), cl, dfg::REF_2D1_CL, 100 * (cl / dfg::REF_2D1_CL - 1),
                        dp, dfg::REF_2D1_DP, 100 * (dp / dfg::REF_2D1_DP - 1), d.cd_band(20.0));
            res << "\"results\": {\"cd\": " << jnum(cd) << ", \"cl\": " << jnum(cl) << ", \"dp\": " << jnum(dp) << ", \"cd_band_last20DU\": " << jnum(d.cd_band(20.0))
                << ", \"definition\": \"time mean over the last 20 D/U\", \"cd_last\": " << jnum(last_ok.cd) << ", \"cl_last\": " << jnum(last_ok.cl) << ", \"dp_last\": " << jnum(last_ok.dp_phys) << "},\n"
                << "\"reference\": {\"cd\": " << jnum(dfg::REF_2D1_CD) << ", \"cl\": " << jnum(dfg::REF_2D1_CL) << ", \"dp\": " << jnum(dfg::REF_2D1_DP)
                << ", \"source\": \"FeatFlow DFG benchmark 2D-1 page, values by Nabh 1998\", \"paper_range\": {\"cd\": [5.57, 5.59], \"cl\": [0.0104, 0.0110], \"dp\": [0.1172, 0.1176]}},\n"
                << "\"error_pct\": {\"cd\": " << jnum(100 * (cd / dfg::REF_2D1_CD - 1)) << ", \"cl\": " << jnum(100 * (cl / dfg::REF_2D1_CL - 1)) << ", \"dp\": " << jnum(100 * (dp / dfg::REF_2D1_DP - 1)) << "}";
        } else {
            const Diag::Periodic p = d.periodic(3);
            if (p.valid) {
                std::printf("2D-2 result (last 3 periods): cD max %.5f (ref %.5f, %+.3f%%) cD min %.5f  cL max %.5f (ref %.5f, %+.3f%%) cL min %.5f  St %.5f (ref %.5f, %+.3f%%)  dp(t0+T/2) %.5f (ref %.5f, %+.3f%%)  period spread %.2e\n",
                            p.cd_max, dfg::REF_2D2_CD_MAX, 100 * (p.cd_max / dfg::REF_2D2_CD_MAX - 1), p.cd_min, p.cl_max, dfg::REF_2D2_CL_MAX, 100 * (p.cl_max / dfg::REF_2D2_CL_MAX - 1), p.cl_min,
                            p.st, dfg::REF_2D2_ST, 100 * (p.st / dfg::REF_2D2_ST - 1), p.dp_t0_half, dfg::REF_2D2_DP, 100 * (p.dp_t0_half / dfg::REF_2D2_DP - 1), p.period_spread);
            } else std::printf("2D-2: fewer than 3 periods seen (%d)\n", d.n_periods());
            res << "\"results\": {\"valid\": " << (p.valid ? "true" : "false") << ", \"cd_max\": " << jnum(p.cd_max) << ", \"cd_min\": " << jnum(p.cd_min)
                << ", \"cl_max\": " << jnum(p.cl_max) << ", \"cl_min\": " << jnum(p.cl_min) << ", \"st\": " << jnum(p.st) << ", \"period_DU\": " << jnum(p.period)
                << ", \"period_spread_rel\": " << jnum(p.period_spread) << ", \"dp_t0_half\": " << jnum(p.dp_t0_half) << ", \"n_periods\": " << d.n_periods()
                << ", \"window_DU\": [" << jnum(p.t0) << ", " << jnum(p.t1) << "]},\n"
                << "\"reference\": {\"cd_max\": " << jnum(dfg::REF_2D2_CD_MAX) << ", \"cd_min\": " << jnum(dfg::REF_2D2_CD_MIN) << ", \"cl_max\": " << jnum(dfg::REF_2D2_CL_MAX)
                << ", \"cl_min\": " << jnum(dfg::REF_2D2_CL_MIN) << ", \"st\": " << jnum(dfg::REF_2D2_ST) << ", \"dp_t0_half\": " << jnum(dfg::REF_2D2_DP)
                << ", \"source\": \"FeatFlow 20141124_dfg_bench2_official.zip, Q2/P1disc level 6, k = 1/1600 (cuda/reference/extracted_2d2_reference.json)\""
                << ", \"paper_range\": {\"cd_max\": [3.22, 3.24], \"cl_max\": [0.99, 1.01], \"st\": [0.295, 0.305], \"dp\": [2.46, 2.50]}},\n"
                << "\"error_pct\": {\"cd_max\": " << jnum(100 * (p.cd_max / dfg::REF_2D2_CD_MAX - 1)) << ", \"cl_max\": " << jnum(100 * (p.cl_max / dfg::REF_2D2_CL_MAX - 1))
                << ", \"st\": " << jnum(100 * (p.st / dfg::REF_2D2_ST - 1)) << ", \"dp_t0_half\": " << jnum(100 * (p.dp_t0_half / dfg::REF_2D2_DP - 1)) << "}";
            if (nframes) {
                ff.close();
                std::ofstream fj(a.out + "/frames.json");
                fj << "{\"Nx\": " << cs.Nx << ", \"Ny\": " << cs.Ny << ", \"nframes\": " << nframes << ", \"dtype\": \"float32\", \"quantity\": \"vorticity_lat\", \"vort_scale_to_U_over_D\": " << jnum(cs.vort_scale())
                   << ", \"period_DU\": " << jnum(p.period) << ", \"steps_per_DU\": " << jnum(cs.steps_per_DU) << ", \"steps\": [";
                for (int i = 0; i < nframes; i++) fj << (i ? ", " : "") << frame_steps[i];
                fj << "]}\n";
            }
        }
        write_fields(a.out, sim);
    }

    // ------------------------------------------------------------ summary.json
    std::ofstream sj(a.out + "/summary.json");
    sj << "{\n\"command\": " << json_str(cmdline) << ",\n\"tag\": " << json_str(a.tag) << ",\n\"precision\": \"" << PRECISION_NAME << "\",\n"
       << "\"gpu\": " << json_str(prop.name) << ", \"sms\": " << prop.multiProcessorCount << ", \"cuda_driver_api\": " << drv << ", \"cuda_runtime\": " << rt
       << ", \"gpu_start\": " << json_str(gpu0) << ",\n\"machine\": \"cuda/results/MACHINE.md\",\n"
       << "\"case\": " << case_json(cs) << ",\n" << res.str() << ",\n\"exit_code\": " << exit_code << ",\n\"total_wall_s\": " << jnum(secs(t_start, Clock::now())) << "\n}\n";
    sj.close();
    if (!sj) { std::fprintf(stderr, "error: could not write %s/summary.json\n", a.out.c_str()); return 4; }
    std::printf("summary written to %s/summary.json, exit %d\n", a.out.c_str(), exit_code);
    return exit_code;
}
