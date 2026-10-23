// lbm_cuda.cu
// Real-time demo of the validated D2Q9 lattice Boltzmann solver (lbm_core.cuh).
// Display uses GLFW + CUDA/OpenGL interop: the colourised frame is copied device
// to device into a GL texture, so nothing crosses PCIe per frame.
//
// Default case: Schäfer and Turek 2D-2 (Re = 100) at 80 cells per diameter,
// the same setup and the same kernel as the benchmark run in cuda/results/.
// cD, cL and St are shown live in the title bar.
//
// Options:
//   --nd <n>       cells per diameter (default 80)
//   --re <value>   showcase at another Reynolds number, same geometry, NOT validated
//   --signed       signed vorticity with a diverging colour map (default: |omega|, LUT)
//   --vmax <v>     colour scale in units of U_mean / D (default 10)
//   --blur <n>     display only 3x3 Gaussian passes, 0 to 2 (default 0)
//   --steps <n>    run exactly n steps, print cD max, cL max, St like the benchmark, exit
//   headless build (-DHEADLESS) only:
//   --out <dir>    write summary.json there (default .)
//   --ppm-every <f> write frame_XXXXX.ppm every f frames (default 0: none)
//   --fast <n> --smooth <n>  steps per frame before and after shedding is detected
//
// Build, Linux:   nvcc -O3 -arch=native lbm_cuda.cu -o lbm_cuda -lglfw -lGL
// Build, Windows: cuda\run.bat   (finds MSVC, nvcc and GLFW)
// Headless:       nvcc -O3 -arch=native -DHEADLESS lbm_cuda.cu -o lbm_cuda_headless

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
#endif

#include "lbm_core.cuh"
#undef LBM_SRC_FILE
#define LBM_SRC_FILE "lbm_cuda.cu"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#ifndef HEADLESS
  #include <GLFW/glfw3.h>
  #include <cuda_gl_interop.h>
#endif

// On hybrid graphics laptops, ask the driver to run the GL context on the
// NVIDIA GPU so the CUDA/GL interop can register the texture.
#if defined(_WIN32) && !defined(HEADLESS)
extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 1;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

constexpr int CHECK_EVERY   = 30;     // frames between divergence checks
constexpr int PROFILE_EVERY = 30;     // frames between profiler printouts

// ================================================================== //
//  colour maps
// ================================================================== //
__constant__ unsigned char LUT[256][3] = {
    {  0,  0,  3}, {  0,  0,  4}, {  0,  0,  6}, {  1,  0,  7}, {  1,  1,  9}, {  1,  1, 11}, {  2,  2, 13}, {  2,  2, 15},
    {  3,  3, 17}, {  4,  3, 19}, {  4,  4, 21}, {  5,  4, 23}, {  6,  5, 25}, {  7,  5, 27}, {  8,  6, 29}, {  9,  7, 31},
    { 10,  7, 34}, { 11,  8, 36}, { 12,  9, 38}, { 13, 10, 40}, { 14, 10, 42}, { 15, 11, 44}, { 16, 12, 47}, { 17, 12, 49},
    { 18, 13, 51}, { 20, 13, 53}, { 21, 14, 56}, { 22, 14, 58}, { 23, 15, 60}, { 24, 15, 63}, { 26, 16, 65}, { 27, 16, 68},
    { 28, 16, 70}, { 30, 16, 73}, { 31, 17, 75}, { 32, 17, 77}, { 34, 17, 80}, { 35, 17, 82}, { 37, 17, 85}, { 38, 17, 87},
    { 40, 17, 89}, { 42, 17, 92}, { 43, 17, 94}, { 45, 16, 96}, { 47, 16, 98}, { 48, 16,101}, { 50, 16,103}, { 52, 16,104},
    { 53, 15,106}, { 55, 15,108}, { 57, 15,110}, { 59, 15,111}, { 60, 15,113}, { 62, 15,114}, { 64, 15,115}, { 66, 15,116},
    { 67, 15,117}, { 69, 15,118}, { 71, 15,119}, { 72, 16,120}, { 74, 16,121}, { 75, 16,121}, { 77, 17,122}, { 79, 17,123},
    { 80, 18,123}, { 82, 18,124}, { 83, 19,124}, { 85, 19,125}, { 87, 20,125}, { 88, 21,126}, { 90, 21,126}, { 91, 22,126},
    { 93, 23,126}, { 94, 23,127}, { 96, 24,127}, { 97, 24,127}, { 99, 25,127}, {101, 26,128}, {102, 26,128}, {104, 27,128},
    {105, 28,128}, {107, 28,128}, {108, 29,128}, {110, 30,129}, {111, 30,129}, {113, 31,129}, {115, 31,129}, {116, 32,129},
    {118, 33,129}, {119, 33,129}, {121, 34,129}, {122, 34,129}, {124, 35,129}, {126, 36,129}, {127, 36,129}, {129, 37,129},
    {130, 37,129}, {132, 38,129}, {133, 38,129}, {135, 39,129}, {137, 40,129}, {138, 40,129}, {140, 41,128}, {141, 41,128},
    {143, 42,128}, {145, 42,128}, {146, 43,128}, {148, 43,128}, {149, 44,128}, {151, 44,127}, {153, 45,127}, {154, 45,127},
    {156, 46,127}, {158, 46,126}, {159, 47,126}, {161, 47,126}, {163, 48,126}, {164, 48,125}, {166, 49,125}, {167, 49,125},
    {169, 50,124}, {171, 51,124}, {172, 51,123}, {174, 52,123}, {176, 52,123}, {177, 53,122}, {179, 53,122}, {181, 54,121},
    {182, 54,121}, {184, 55,120}, {185, 55,120}, {187, 56,119}, {189, 57,119}, {190, 57,118}, {192, 58,117}, {194, 58,117},
    {195, 59,116}, {197, 60,116}, {198, 60,115}, {200, 61,114}, {202, 62,114}, {203, 62,113}, {205, 63,112}, {206, 64,112},
    {208, 65,111}, {209, 66,110}, {211, 66,109}, {212, 67,109}, {214, 68,108}, {215, 69,107}, {217, 70,106}, {218, 71,105},
    {220, 72,105}, {221, 73,104}, {222, 74,103}, {224, 75,102}, {225, 76,102}, {226, 77,101}, {228, 78,100}, {229, 80, 99},
    {230, 81, 98}, {231, 82, 98}, {232, 84, 97}, {234, 85, 96}, {235, 86, 96}, {236, 88, 95}, {237, 89, 95}, {238, 91, 94},
    {238, 93, 93}, {239, 94, 93}, {240, 96, 93}, {241, 97, 92}, {242, 99, 92}, {243,101, 92}, {243,103, 91}, {244,104, 91},
    {245,106, 91}, {245,108, 91}, {246,110, 91}, {246,112, 91}, {247,113, 91}, {247,115, 92}, {248,117, 92}, {248,119, 92},
    {249,121, 92}, {249,123, 93}, {249,125, 93}, {250,127, 94}, {250,128, 94}, {250,130, 95}, {251,132, 96}, {251,134, 96},
    {251,136, 97}, {251,138, 98}, {252,140, 99}, {252,142, 99}, {252,144,100}, {252,146,101}, {252,147,102}, {253,149,103},
    {253,151,104}, {253,153,105}, {253,155,106}, {253,157,107}, {253,159,108}, {253,161,110}, {253,162,111}, {253,164,112},
    {254,166,113}, {254,168,115}, {254,170,116}, {254,172,117}, {254,174,118}, {254,175,120}, {254,177,121}, {254,179,123},
    {254,181,124}, {254,183,125}, {254,185,127}, {254,187,128}, {254,188,130}, {254,190,131}, {254,192,133}, {254,194,134},
    {254,196,136}, {254,198,137}, {254,199,139}, {254,201,141}, {254,203,142}, {253,205,144}, {253,207,146}, {253,209,147},
    {253,210,149}, {253,212,151}, {253,214,152}, {253,216,154}, {253,218,156}, {253,220,157}, {253,221,159}, {253,223,161},
    {253,225,163}, {252,227,165}, {252,229,166}, {252,230,168}, {252,232,170}, {252,234,172}, {252,236,174}, {252,238,176},
    {252,240,177}, {252,241,179}, {252,243,181}, {252,245,183}, {251,247,185}, {251,249,187}, {251,250,189}, {251,252,191}
};
// diverging map for signed vorticity (matplotlib RdBu_r sampled at 33 points, blue negative, red positive)
__constant__ unsigned char DIV[33][3] = {
    {5, 48, 97}, {14, 65, 121}, {23, 82, 144}, {31, 99, 168}, {42, 113, 178}, {52, 128, 185}, {63, 142, 192}, {82, 157, 200},
    {107, 172, 209}, {132, 188, 217}, {155, 201, 224}, {174, 211, 230}, {194, 221, 236}, {212, 230, 241}, {224, 236, 243}, {236, 242, 245},
    {247, 246, 246}, {249, 238, 231}, {251, 229, 216}, {253, 220, 201}, {251, 204, 180}, {248, 187, 158}, {245, 170, 137}, {238, 150, 119},
    {228, 128, 102}, {219, 107, 85}, {208, 85, 72}, {197, 62, 61}, {186, 40, 50}, {171, 22, 42}, {147, 14, 38}, {124, 7, 34}, {103, 0, 31}
};

// 3x3 Gaussian blur, display only
__global__ void gauss_blur(const float* __restrict__ src, float* __restrict__ dst, const int Nx, const int Ny)
{
    const int N = Nx * Ny;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const int x = idx % Nx, y = idx / Nx;
    if (x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1) { dst[idx] = src[idx]; return; }
    const float k[3][3] = {{0.0625f, 0.125f, 0.0625f}, {0.125f, 0.25f, 0.125f}, {0.0625f, 0.125f, 0.0625f}};
    float v = 0.f;
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++)
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) v += k[dy + 1][dx + 1] * src[(y + dy) * Nx + (x + dx)];
    dst[idx] = v;
}

// colourise: vorticity (lattice units) * scale -> U_mean / D units -> colour.
// Output row major (y * Nx + x), which is what the GL texture wants.
__global__ void colourise(const float* __restrict__ vort, const unsigned char* __restrict__ type,
                          uchar4* __restrict__ rgba, const int N, const float scale, const float vmax, const int signed_map)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (is_solid(type[idx])) { rgba[idx] = make_uchar4(38, 38, 45, 255); return; }
    const float v = vort[idx] * scale / vmax;
    if (!signed_map) {
        const float a = fminf(fabsf(v), 1.f);
        const int c = (int)(a * 255.f + 0.5f);
        rgba[idx] = make_uchar4(LUT[c][0], LUT[c][1], LUT[c][2], 255);
    } else {
        const float a = fminf(fmaxf(0.5f * (v + 1.f), 0.f), 1.f) * 32.f;
        const int i0 = min((int)a, 31); const float t = a - i0;
        rgba[idx] = make_uchar4((unsigned char)(DIV[i0][0] + t * (DIV[i0 + 1][0] - DIV[i0][0])),
                                (unsigned char)(DIV[i0][1] + t * (DIV[i0 + 1][1] - DIV[i0][1])),
                                (unsigned char)(DIV[i0][2] + t * (DIV[i0 + 1][2] - DIV[i0][2])), 255);
    }
}

// max |a| reduction via atomicMax on the float bits (non negative floats order
// like their bit patterns; NaN sorts above +inf, so it shows up on the host).
__global__ void max_abs_kernel(const float* __restrict__ a, const int N, unsigned int* __restrict__ out)
{
    unsigned int m = 0u;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        m = max(m, __float_as_uint(fabsf(a[i])));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = max(m, __shfl_down_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) atomicMax(out, m);
}

// ================================================================== //
//  host
// ================================================================== //
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

#ifdef HEADLESS
static void write_ppm(const char* path, const uchar4* px, int Nx, int Ny)
{
    FILE* fp = std::fopen(path, "wb");
    if (!fp) { std::fprintf(stderr, "Cannot write %s\n", path); return; }
    std::fprintf(fp, "P6\n%d %d\n255\n", Nx, Ny);
    std::vector<unsigned char> row(3 * Nx);
    for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) { const uchar4 p = px[y * Nx + x]; row[3 * x] = p.x; row[3 * x + 1] = p.y; row[3 * x + 2] = p.z; }
        std::fwrite(row.data(), 1, row.size(), fp);
    }
    std::fclose(fp);
}
#endif

// The benchmark's periodicity rule (lbm_bench.cu): at least 12 consistent shedding
// periods and a spread below 0.2 percent over the last 3. Before that, St and the
// force extrema are start-up values and must not be compared with the reference.
static bool periodic_by_rule(const Diag& d) { return d.n_periods() >= 12 && d.period_spread(3) < 0.002; }

int main(int argc, char** argv)
{
    int nd = 80, blur = 0, fast = 0, smooth = 0, ppm_every = 0; long long steps = -1;
    double re = -1, vmax = 10.0; bool signed_map = false; std::string out = ".";
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(1); } return std::string(argv[++i]); };
        if (k == "--nd") nd = std::atoi(next().c_str()); else if (k == "--re") re = std::atof(next().c_str());
        else if (k == "--signed") signed_map = true; else if (k == "--vmax") vmax = std::atof(next().c_str());
        else if (k == "--blur") blur = std::atoi(next().c_str()); else if (k == "--steps") steps = std::atoll(next().c_str());
        else if (k == "--out") out = next(); else if (k == "--ppm-every") ppm_every = std::atoi(next().c_str());
        else if (k == "--fast") fast = std::atoi(next().c_str()); else if (k == "--smooth") smooth = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 1; }
    }
#ifndef HEADLESS
    if (ppm_every) std::printf("--ppm-every is only used by the headless build; ignored here.\n");
#endif
#ifdef HEADLESS
    {   // the headless build writes summary.json and snapshots into --out; create it
        std::error_code fs_err;
        std::filesystem::create_directories(out, fs_err);
        if (fs_err) { std::fprintf(stderr, "error: cannot create the output folder %s: %s\n", out.c_str(), fs_err.message().c_str()); return 4; }
    }
#endif
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    const bool validated = re < 0;
    dfg::Case cs = validated ? dfg::make_dfg(2, nd) : dfg::make_showcase(nd, re);
    const int Nx = cs.Nx, Ny = cs.Ny, N = Nx * Ny;
    std::printf("%s: Re = %.0f, %d cells per diameter, grid %d x %d, tau %.4f, U_mean,lat %.4f, Mach %.3f%s\n",
                validated ? "DFG benchmark 2D-2 (validated setup)" : "Showcase (NOT validated)",
                cs.Re, nd, Nx, Ny, cs.tau, cs.umean_lat, cs.mach, validated ? "" : "; numbers shown are not checked against any reference");

    Sim sim(cs);
    Diag diag;
    const int every = std::max(1, (int)std::lround(cs.steps_per_DU / 50.0));   // force sampling cadence, as in lbm_bench
    if (fast <= 0) fast = 40 * every;
    if (smooth <= 0) smooth = 4 * every;
    fast = std::max(every, (fast / every) * every); smooth = std::max(every, (smooth / every) * every);
    const int blocks = (N + THREADS - 1) / THREADS;
    const int red_blocks = prop.multiProcessorCount * 8;

    float* d_blur = nullptr; uchar4* d_rgba = nullptr; unsigned int* d_peak = nullptr;
    CK(cudaMalloc(&d_blur, sizeof(float) * N));
    CK(cudaMalloc(&d_rgba, sizeof(uchar4) * N));
    CK(cudaMalloc(&d_peak, sizeof(unsigned int)));
    sim.ensure_fields();
    const float vscale = (float)cs.vort_scale();
    // shedding probe: transverse velocity at cylinder centre + 2 D downstream
    const int probe_idx = (int)std::lround(cs.yc - 0.5) * Nx + (int)std::lround(cs.xc + 2.0 * nd);
    std::vector<real> probe_f(9);

#ifndef HEADLESS
    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(Nx, Ny, "Lattice Boltzmann", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "Window creation failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);
    GLuint tex; glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, Nx, Ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glEnable(GL_TEXTURE_2D);
    cudaGraphicsResource* tex_res = nullptr;
    const bool interop = cudaGraphicsGLRegisterImage(&tex_res, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard) == cudaSuccess;
    uchar4* h_rgba = nullptr;
    if (interop) std::printf("CUDA/GL interop active: frames stay on the GPU.\n");
    else { cudaGetLastError(); std::printf("CUDA/GL interop unavailable; falling back to a host copy per frame.\n"); CK(cudaMallocHost(&h_rgba, sizeof(uchar4) * N)); }
    std::printf("ESC or close window to quit.\n");
#else
    uchar4* h_rgba = nullptr;
    CK(cudaMallocHost(&h_rgba, sizeof(uchar4) * N));
    if (steps < 0) steps = (long long)(160.0 * cs.steps_per_DU);
    std::printf("Headless: %lld steps%s.\n", steps, ppm_every ? ", PPM snapshots" : "");
#endif

    long long it = 0; int frame = 0;
    bool shedding = false, running = true, diverged = false;
    double fps = 0, t_sim = 0, t_post = 0, t_draw = 0, title_t = 0; long long prof_steps = 0; double prof_ms = 0;
    int probe_state = 0, transitions = 0;
    auto t_prev = Clock::now();

    while (running) {
#ifndef HEADLESS
        glfwPollEvents();
        if (glfwWindowShouldClose(win) || glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        if (steps > 0 && it >= steps) break;      // --steps also ends the window run (used for logged tests)
        const int n = shedding ? smooth : fast;
#else
        if (it >= steps) break;
        int n = shedding ? smooth : fast;
        if (it + n > steps) n = (int)(steps - it);
#endif
        // ---- physics: chunks of `every` steps with force sampling, exactly as lbm_bench ----
        const auto t0 = Clock::now();
        const double k0 = sim.kernel_ms;
        for (int s = 0; s < n; s += every) {
            const int m = std::min(every, n - s);
            sim.step(m);
            const Sim::Sample smp = sim.sample();
            if (!smp.finite) { diverged = true; break; }
            diag.push(cs.steps_to_DU(sim.step_count), smp);
        }
        const auto t1 = Clock::now();
        it = sim.step_count; prof_steps += n; prof_ms += sim.kernel_ms - k0;
        if (diverged) { std::printf("Diverged (non finite force) at step %lld.\n", it); break; }

        // ---- vorticity, blur, colourise (all on GPU) ----
        sim.compute_vorticity();
        float* sbuf = sim.d_vort; float* dbuf = d_blur;
        for (int p = 0; p < blur; p++) { gauss_blur<<<blocks, THREADS>>>(sbuf, dbuf, Nx, Ny); std::swap(sbuf, dbuf); }
        colourise<<<blocks, THREADS>>>(sbuf, sim.d_type, d_rgba, N, vscale, (float)vmax, signed_map ? 1 : 0);
        CK(cudaGetLastError());
#ifndef HEADLESS
        if (interop) {
            CK(cudaGraphicsMapResources(1, &tex_res, 0));
            cudaArray_t arr; CK(cudaGraphicsSubResourceGetMappedArray(&arr, tex_res, 0, 0));
            CK(cudaMemcpy2DToArray(arr, 0, 0, d_rgba, Nx * sizeof(uchar4), Nx * sizeof(uchar4), Ny, cudaMemcpyDeviceToDevice));
            CK(cudaGraphicsUnmapResources(1, &tex_res, 0));
        } else {
            CK(cudaMemcpy(h_rgba, d_rgba, sizeof(uchar4) * N, cudaMemcpyDeviceToHost));
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Nx, Ny, GL_RGBA, GL_UNSIGNED_BYTE, h_rgba);
        }
#endif
        CK(cudaDeviceSynchronize());
        const auto t2 = Clock::now();

        // ---- shedding detector: transverse velocity probe at centre + 2 D ----
        if (!shedding) {
            sim.node_populations(probe_idx, probe_f.data());
            double r = 1.0, vy = 0; const int cy[9] = {0, 1, 1, 0, -1, -1, -1, 0, 1};   // shifted populations: rho = 1 + sum g
            for (int q = 0; q < 9; q++) { r += probe_f[q]; vy += probe_f[q] * cy[q]; }
            vy /= r;
            const int st = vy > 0.05 * cs.umean_lat ? 1 : (vy < -0.05 * cs.umean_lat ? -1 : 0);
            if (st != 0 && st != probe_state) { if (probe_state != 0) transitions++; probe_state = st; }
            if (transitions >= 4) { shedding = true; std::printf("Shedding detected at step %lld (%.1f D/U); switching to smooth playback.\n", it, cs.steps_to_DU(it)); }
        }
        // ---- divergence watchdog ----
        if (frame % CHECK_EVERY == 0) {
            CK(cudaMemset(d_peak, 0, sizeof(unsigned int)));
            max_abs_kernel<<<red_blocks, THREADS>>>(sbuf, N, d_peak);
            unsigned int bits; CK(cudaMemcpy(&bits, d_peak, sizeof(bits), cudaMemcpyDeviceToHost));
            float peak; std::memcpy(&peak, &bits, sizeof(float));
            if (!std::isfinite(peak) || peak * vscale > 1e4f) { std::printf("Diverged at step %lld.\n", it); break; }
        }

        // ---- draw ----
#ifndef HEADLESS
        int fbw, fbh; glfwGetFramebufferSize(win, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh); glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS);                      // texture row 0 (y = 0, bottom wall) drawn at the bottom
            glTexCoord2f(0.f, 0.f); glVertex2f(-1.f, -1.f);
            glTexCoord2f(1.f, 0.f); glVertex2f( 1.f, -1.f);
            glTexCoord2f(1.f, 1.f); glVertex2f( 1.f,  1.f);
            glTexCoord2f(0.f, 1.f); glVertex2f(-1.f,  1.f);
        glEnd();
        glfwSwapBuffers(win);
#else
        if (ppm_every && frame % ppm_every == 0) {
            CK(cudaMemcpy(h_rgba, d_rgba, sizeof(uchar4) * N, cudaMemcpyDeviceToHost));
            char path[512]; std::snprintf(path, sizeof(path), "%s/frame_%06d.ppm", out.c_str(), frame);
            write_ppm(path, h_rgba, Nx, Ny);
        }
#endif
        const auto t3 = Clock::now();
        t_sim += secs(t0, t1); t_post += secs(t1, t2); t_draw += secs(t2, t3);
        const double dt = secs(t_prev, t3); t_prev = t3;
        if (dt > 0) fps = (fps == 0.0) ? 1.0 / dt : 0.9 * fps + 0.1 / dt;
        frame++;

        title_t += dt;
        if (title_t > 0.25) {
            const double st = periodic_by_rule(diag) ? diag.strouhal(3) : NAN;   // shown only once periodic
            char title[256], stbuf[16];
            if (std::isfinite(st)) std::snprintf(stbuf, sizeof(stbuf), "%.3f", st); else std::snprintf(stbuf, sizeof(stbuf), "...");
            std::snprintf(title, sizeof(title), "%s Re = %.0f | %d cells/D | cD %.3f  cL %+.3f  St %s | t = %.1f D/U | %d steps/frame | %.1f FPS",
                          validated ? "DFG benchmark 2D-2 (validated setup)" : "NOT VALIDATED showcase", cs.Re, nd,
                          diag.cd.empty() ? 0.0 : diag.cd.back(), diag.cl.empty() ? 0.0 : diag.cl.back(), stbuf,
                          cs.steps_to_DU(it), n, fps);
#ifndef HEADLESS
            glfwSetWindowTitle(win, title);
#endif
            title_t = 0;
        }
        if (frame % PROFILE_EVERY == 0) {
            const double tot = t_sim + t_post + t_draw;
            std::printf("[%5d] %5d steps/frame | sim %4.1f%%  post %4.1f%%  draw %4.1f%%  | %9.0f steps/s  %5.1f FPS  %7.0f MLUPS (kernel) | t %.1f D/U cD %.4f cL %+.4f\n",
                        frame, n, 100 * t_sim / tot, 100 * t_post / tot, 100 * t_draw / tot, prof_steps / tot, PROFILE_EVERY / tot,
                        prof_steps * (double)N / (prof_ms * 1e-3) / 1e6, cs.steps_to_DU(it), diag.cd.back(), diag.cl.back());
            t_sim = t_post = t_draw = 0.0; prof_steps = 0; prof_ms = 0;
        }
    }

    // ---- final numbers, same post-processing as lbm_bench ----
    const Diag::Periodic p = diag.periodic(3);
    const bool periodic = periodic_by_rule(diag);
    if (p.valid && periodic)
        std::printf("Result (last 3 periods): cD max %.5f  cD min %.5f  cL max %.5f  cL min %.5f  St %.5f  dp(t0+T/2) %.5f  period spread %.2e%s\n",
                    p.cd_max, p.cd_min, p.cl_max, p.cl_min, p.st, p.dp_t0_half, p.period_spread, validated ? "" : "  [NOT VALIDATED]");
    else
        std::printf("Not yet periodic by the benchmark's rule after %.1f D/U (%d consistent periods, 12 needed): "
                    "no result printed, run longer (about 75 D/U at the default grid).\n", cs.steps_to_DU(it), diag.n_periods());
#ifdef HEADLESS
    {
        // finite numbers with 10 significant digits, null otherwise (valid JSON even after a short run)
        auto J = [](double v) { char t[32]; if (std::isfinite(v)) std::snprintf(t, sizeof t, "%.10g", v); else std::snprintf(t, sizeof t, "null"); return std::string(t); };
        std::ofstream sj(out + "/summary.json");
        sj << "{\"program\": \"lbm_cuda_headless\", \"precision\": \"" << PRECISION_NAME << "\", \"gpu\": \"" << prop.name << "\", \"validated_setup\": " << (validated ? "true" : "false")
           << ", \"re\": " << J(cs.Re) << ", \"nd\": " << nd << ", \"Nx\": " << Nx << ", \"Ny\": " << Ny << ", \"steps\": " << it << ", \"t_end_DU\": " << J(cs.steps_to_DU(it))
           << ", \"sample_every_steps\": " << every << ", \"diverged\": " << (diverged ? "true" : "false") << ", \"n_periods\": " << diag.n_periods() << ", \"periodic_by_benchmark_rule\": " << (periodic ? "true" : "false")
           << ", \"results\": {\"valid\": " << (p.valid ? "true" : "false") << ", \"cd_max\": " << J(p.cd_max) << ", \"cd_min\": " << J(p.cd_min)
           << ", \"cl_max\": " << J(p.cl_max) << ", \"cl_min\": " << J(p.cl_min) << ", \"st\": " << J(p.st) << ", \"period_DU\": " << J(p.period)
           << ", \"period_spread_rel\": " << J(p.period_spread) << ", \"dp_t0_half\": " << J(p.dp_t0_half) << "}, \"mlups_kernel_events\": " << J(sim.mlups_kernel()) << "}\n";
    }
#endif

#ifndef HEADLESS
    if (interop) cudaGraphicsUnregisterResource(tex_res);
    glDeleteTextures(1, &tex); glfwDestroyWindow(win); glfwTerminate();
#endif
    if (h_rgba) cudaFreeHost(h_rgba);
    cudaFree(d_blur); cudaFree(d_rgba); cudaFree(d_peak);
    return diverged ? 2 : 0;
}
