// Visual showcase, not validated.
//
// The first CUDA version of this project, a direct port of the CuPy prototype
// (python/lbm_cupy.py), kept for its look. Its numbers aren't checked against any
// reference; the validated solver is cuda/lbm_core.cuh. It looks more dramatic than
// the validated demo because it runs far outside the range where the method is accurate:
//   Reynolds number about 823 (u0 0.20, cylinder diameter 96 cells, tau 0.57)
//   lattice Mach number 0.35 (u0 / cs with cs = 1/sqrt(3))
//   a velocity clamp at 0.45 lattice units hides instability instead of reporting it
//   periodic top and bottom: the flow wraps around instead of meeting walls
//
// D2Q9 BGK with an optional Smagorinsky term (CS = 0, so it's off). The window uses
// GLFW and CUDA/OpenGL interop: the coloured frame is copied device-to-device into a
// GL texture, so nothing crosses PCIe per frame.
//
// Build:
//   Windows window build:   cuda\run.bat release (part of the release zip)
//   headless timing build:  cuda\run.bat baseline, or make -C cuda baseline on Linux
//   Linux window build:     nvcc -O3 -arch=native karman_showcase.cu -o karman_showcase -lglfw -lGL

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <utility>
#include <algorithm>
#include <cuda_runtime.h>

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

#define CK(call) do {                                                        \
    cudaError_t e_ = (call);                                                 \
    if (e_ != cudaSuccess) {                                                 \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n  in %s\n",           \
                     __FILE__, __LINE__, cudaGetErrorString(e_), #call);     \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

// ================================================================== //
//  parameters
// ================================================================== //
constexpr int       Nx  = 1600;
constexpr int       Ny  = 400;
constexpr int       N   = Nx * Ny;
constexpr float     TAU = 0.57f;
constexpr float     U0  = 0.20f;
constexpr float     CS  = 0.0f;        // Smagorinsky constant; 0 -> plain BGK
constexpr long long NT  = 1000000;

// No even-count restriction any more: buffers are swapped by pointer.
constexpr int STEPS_PER_FRAME_FAST   = 2000;  // fast forward through laminar startup
constexpr int STEPS_PER_FRAME_SMOOTH = 120;   // playback once shedding begins

constexpr float VMAX          = 0.05f;  // curl magnitude at the top of the colormap
constexpr int   BLUR_PASSES   = 2;      // display only smoothing passes
[[maybe_unused]] constexpr int SCALE = 1;     // window = Nx*SCALE x Ny*SCALE
constexpr int   CHECK_EVERY   = 30;     // frames between divergence checks
constexpr int   PROFILE_EVERY = 30;

// Anti checkerboard filter. In the Python version this never actually ran
// (the CUDA graph only captured lbm_step), so it is off by default here to
// reproduce what you were seeing. This version is out of place and race free.
constexpr bool USE_FILTER   = false;
constexpr int  FILTER_EVERY = 3;

#ifdef HEADLESS
constexpr int SNAPSHOT_EVERY = 100;     // frames between PPM dumps
#endif

constexpr int THREADS = 256;
constexpr int BLOCKS  = (N + THREADS - 1) / THREADS;

// Structure of arrays: population i of cell idx lives at i*N + idx.
// Neighbouring threads touch neighbouring addresses -> coalesced loads/stores.
#define F(i, idx) ((i) * N + (idx))

// ================================================================== //
//  LBM step
// ================================================================== //
__global__ void lbm_step(const float* __restrict__ fin,
                         float*       __restrict__ fout,
                         const unsigned char* __restrict__ solid)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const int x = idx % Nx;
    const int y = idx / Nx;

    const int   cx[9]  = {0, 0, 1, 1,  1,  0, -1, -1, -1};
    const int   cy[9]  = {0, 1, 1, 0, -1, -1, -1,  0,  1};
    const float w[9]   = {4.f/9.f,
                          1.f/9.f, 1.f/36.f, 1.f/9.f, 1.f/36.f,
                          1.f/9.f, 1.f/36.f, 1.f/9.f, 1.f/36.f};
    const int   opp[9] = {0, 5, 6, 7, 8, 1, 2, 3, 4};

    // inlet: flat equilibrium profile at U0
    if (x == 0 && !solid[idx]) {
        const float u2 = 1.5f * U0 * U0;
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            const float cu = 3.f * cx[i] * U0;
            fout[F(i, idx)] = w[i] * (1.f + cu + 0.5f * cu * cu - u2);
        }
        return;
    }
    // outlet: zero gradient
    if (x == Nx - 1) {
        const int src = y * Nx + (Nx - 2);
        #pragma unroll
        for (int i = 0; i < 9; i++) fout[F(i, idx)] = fin[F(i, src)];
        return;
    }

    // pull streaming (periodic wrap)
    float f[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        int sx = x - cx[i]; if (sx < 0) sx += Nx; else if (sx >= Nx) sx -= Nx;
        int sy = y - cy[i]; if (sy < 0) sy += Ny; else if (sy >= Ny) sy -= Ny;
        f[i] = fin[F(i, sy * Nx + sx)];
    }

    // solid: bounce-back
    if (solid[idx]) {
        #pragma unroll
        for (int i = 0; i < 9; i++) fout[F(i, idx)] = f[opp[i]];
        return;
    }

    float rho = 0.f, ux = 0.f, uy = 0.f;
    #pragma unroll
    for (int i = 0; i < 9; i++) { rho += f[i]; ux += f[i] * cx[i]; uy += f[i] * cy[i]; }
    const float inv_rho = 1.f / rho;
    ux *= inv_rho; uy *= inv_rho;
    const float umag = sqrtf(ux * ux + uy * uy);
    if (umag > 0.45f) { const float s = 0.45f / umag; ux *= s; uy *= s; }

    const float u2 = 1.5f * (ux * ux + uy * uy);
    float feq[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        const float cu = 3.f * (cx[i] * ux + cy[i] * uy);
        feq[i] = rho * w[i] * (1.f + cu + 0.5f * cu * cu - u2);
    }

    // Smagorinsky (CS = 0 -> plain BGK)
    float Qxx = 0.f, Qxy = 0.f, Qyy = 0.f;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        const float fn = f[i] - feq[i];
        Qxx += fn * cx[i] * cx[i]; Qxy += fn * cx[i] * cy[i]; Qyy += fn * cy[i] * cy[i];
    }
    const float Q       = sqrtf(2.f * (Qxx * Qxx + 2.f * Qxy * Qxy + Qyy * Qyy));
    const float tau_eff = 0.5f * (TAU + sqrtf(TAU * TAU + 18.f * CS * CS * Q));
    const float omega   = 1.f / tau_eff;
    #pragma unroll
    for (int i = 0; i < 9; i++) fout[F(i, idx)] = f[i] + omega * (feq[i] - f[i]);
}

// ================================================================== //
//  density filter (out of place: reads fin, writes fout)
// ================================================================== //
__global__ void filter_kernel(const float* __restrict__ fin,
                              float*       __restrict__ fout,
                              const unsigned char* __restrict__ solid)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const int x = idx % Nx, y = idx / Nx;

    float scale = 1.f;
    if (!(solid[idx] || x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1)) {
        float rho_avg = 0.f;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                const int n = (y + dy) * Nx + (x + dx);
                #pragma unroll
                for (int i = 0; i < 9; i++) rho_avg += fin[F(i, n)];
            }
        rho_avg /= 9.f;
        float rho_loc = 0.f;
        #pragma unroll
        for (int i = 0; i < 9; i++) rho_loc += fin[F(i, idx)];
        if (rho_loc >= 1e-6f) scale = rho_avg / rho_loc;
    }
    #pragma unroll
    for (int i = 0; i < 9; i++) fout[F(i, idx)] = fin[F(i, idx)] * scale;
}

// ================================================================== //
//  vorticity
// ================================================================== //
__global__ void curl_kernel(const float* __restrict__ f, float* __restrict__ curl)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const int x = idx % Nx, y = idx / Nx;
    if (x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1) { curl[idx] = 0.f; return; }

    const int cx[9] = {0, 0, 1, 1,  1,  0, -1, -1, -1};
    const int cy[9] = {0, 1, 1, 0, -1, -1, -1,  0,  1};
    const int a = (y + 1) * Nx + x, b = (y - 1) * Nx + x;
    const int c = y * Nx + (x + 1), d = y * Nx + (x - 1);

    float ra = 0, rb = 0, rc = 0, rd = 0, uxa = 0, uxb = 0, uyc = 0, uyd = 0;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        const float fa = f[F(i, a)], fb = f[F(i, b)], fc = f[F(i, c)], fd = f[F(i, d)];
        ra += fa; uxa += fa * cx[i];
        rb += fb; uxb += fb * cx[i];
        rc += fc; uyc += fc * cy[i];
        rd += fd; uyd += fd * cy[i];
    }
    curl[idx] = uxa / ra - uxb / rb - (uyc / rc - uyd / rd);
}

// ================================================================== //
//  3x3 Gaussian blur (display only)
// ================================================================== //
__global__ void gauss_blur(const float* __restrict__ src, float* __restrict__ dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const int x = idx % Nx, y = idx / Nx;
    if (x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1) { dst[idx] = src[idx]; return; }
    const float k[3][3] = {{0.0625f, 0.125f, 0.0625f},
                           {0.125f,  0.25f,  0.125f },
                           {0.0625f, 0.125f, 0.0625f}};
    float v = 0.f;
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++)
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++)
            v += k[dy + 1][dx + 1] * src[(y + dy) * Nx + (x + dx)];
    dst[idx] = v;
}

// ================================================================== //
//  colourise: |curl| -> RGBA8 via a 256 entry LUT in constant memory.
//  Output is row major (y*Nx + x), which is what the GL texture wants.
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

__global__ void colourise(const float* __restrict__ curl,
                          const unsigned char* __restrict__ solid,
                          uchar4* __restrict__ rgba)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (solid[idx]) { rgba[idx] = make_uchar4(38, 38, 45, 255); return; }
    float v = fabsf(curl[idx]) / VMAX;
    v = fminf(fmaxf(v, 0.f), 1.f);
    const int c = (int)(v * 255.f + 0.5f);
    rgba[idx] = make_uchar4(LUT[c][0], LUT[c][1], LUT[c][2], 255);
}

// ================================================================== //
//  max |a| reduction. Non negative floats order the same as their bit
//  patterns, so atomicMax on the uint bits works; NaN sorts above +inf,
//  so a NaN anywhere shows up as a non finite peak on the host.
// ================================================================== //
__global__ void max_abs_kernel(const float* __restrict__ a, unsigned int* __restrict__ out)
{
    unsigned int m = 0u;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        m = max(m, __float_as_uint(fabsf(a[i])));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = max(m, __shfl_down_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) atomicMax(out, m);
}

// ================================================================== //
//  host
// ================================================================== //
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

#ifdef HEADLESS
static void write_ppm(const char* path, const uchar4* px)
{
    FILE* fp = std::fopen(path, "wb");
    if (!fp) { std::fprintf(stderr, "Cannot write %s\n", path); return; }
    std::fprintf(fp, "P6\n%d %d\n255\n", Nx, Ny);
    std::vector<unsigned char> row(3 * Nx);
    for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) {
            const uchar4 p = px[y * Nx + x];
            row[3 * x] = p.x; row[3 * x + 1] = p.y; row[3 * x + 2] = p.z;
        }
        std::fwrite(row.data(), 1, row.size(), fp);
    }
    std::fclose(fp);
}
#endif

int main()
{
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    // ---------------- geometry ----------------
    const int cyl_r = 22 * Ny / 180;
    const int cx0 = Nx / 5, cy0 = Ny / 2 + 2;     // +2 breaks symmetry -> earlier shedding
    std::vector<unsigned char> h_solid(N);
    for (int y = 0; y < Ny; y++)
        for (int x = 0; x < Nx; x++) {
            const int dx = x - cx0, dy = y - cy0;
            h_solid[y * Nx + x] = (dx * dx + dy * dy < cyl_r * cyl_r) ? 1 : 0;
        }

    // ---------------- initial state: uniform equilibrium + noise ----------------
    const int   h_cx[9] = {0, 0, 1, 1, 1, 0, -1, -1, -1};
    const float h_w[9]  = {4.f/9, 1.f/9, 1.f/36, 1.f/9, 1.f/36, 1.f/9, 1.f/36, 1.f/9, 1.f/36};
    std::vector<float> h_f((size_t)9 * N);
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < 9; i++) {
        const float cu  = 3.f * h_cx[i] * U0;
        const float feq = h_w[i] * (1.f + cu + 0.5f * cu * cu - 1.5f * U0 * U0);
        for (int idx = 0; idx < N; idx++) h_f[(size_t)i * N + idx] = feq + 0.002f * nd(rng);
    }

    // ---------------- device buffers ----------------
    float *d_fa, *d_fb, *d_curl, *d_blur;
    unsigned char* d_solid;
    uchar4* d_rgba;
    unsigned int* d_peak;
    CK(cudaMalloc(&d_fa,    sizeof(float) * 9 * N));
    CK(cudaMalloc(&d_fb,    sizeof(float) * 9 * N));
    CK(cudaMalloc(&d_curl,  sizeof(float) * N));
    CK(cudaMalloc(&d_blur,  sizeof(float) * N));
    CK(cudaMalloc(&d_solid, N));
    CK(cudaMalloc(&d_rgba,  sizeof(uchar4) * N));
    CK(cudaMalloc(&d_peak,  sizeof(unsigned int)));
    CK(cudaMemcpy(d_fa, h_f.data(), sizeof(float) * 9 * N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_solid, h_solid.data(), N, cudaMemcpyHostToDevice));
    float* d_cur = d_fa;
    float* d_nxt = d_fb;

    const double nu = (2.0 * TAU - 1.0) / 6.0;
    const double Re = U0 * (2.0 * cyl_r) / nu;
    const int red_blocks = prop.multiProcessorCount * 8;

#ifndef HEADLESS
    // ---------------- window + GL texture ----------------
    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(Nx * SCALE, Ny * SCALE, "Showcase, not validated | Karman Vortex Street", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "Window creation failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);                       // no vsync cap, like pygame.flip()

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, Nx, Ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glEnable(GL_TEXTURE_2D);

    cudaGraphicsResource* tex_res = nullptr;
    const bool interop = cudaGraphicsGLRegisterImage(&tex_res, tex, GL_TEXTURE_2D,
                             cudaGraphicsRegisterFlagsWriteDiscard) == cudaSuccess;
    uchar4* h_rgba = nullptr;
    if (interop) {
        std::printf("CUDA/GL interop active: frames stay on the GPU.\n");
    } else {
        cudaGetLastError();                    // clear the non sticky error
        std::printf("CUDA/GL interop unavailable (GL context not on the NVIDIA GPU?); "
                    "falling back to a host copy per frame.\n");
        CK(cudaMallocHost(&h_rgba, sizeof(uchar4) * N));
    }
    std::printf("ESC or close window to quit.\n");
#else
    uchar4* h_rgba = nullptr;
    CK(cudaMallocHost(&h_rgba, sizeof(uchar4) * N));
    std::printf("Headless: writing frame_XXXXX.ppm every %d frames.\n", SNAPSHOT_EVERY);
#endif

    std::printf("Re ~ %.0f   nu = %.4f   grid %dx%d   cylinder r = %d\n", Re, nu, Nx, Ny, cyl_r);

    // ---------------- main loop ----------------
    long long it = 0;
    int frame = 0;
    bool shedding_started = false, running = true;
    double fps = 0.0, t_sim = 0, t_post = 0, t_draw = 0;
#ifndef HEADLESS
    double title_t = 0;
#endif
    long long prof_steps = 0;
    auto t_prev = Clock::now();

    while (running && it < NT) {
#ifndef HEADLESS
        glfwPollEvents();
        if (glfwWindowShouldClose(win) || glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
#endif
        const int n = shedding_started ? STEPS_PER_FRAME_SMOOTH : STEPS_PER_FRAME_FAST;

        // ---- physics ----
        const auto t0 = Clock::now();
        for (int k = 0; k < n; k++) {
            lbm_step<<<BLOCKS, THREADS>>>(d_cur, d_nxt, d_solid);
            std::swap(d_cur, d_nxt);
            if (USE_FILTER && (k % FILTER_EVERY) == FILTER_EVERY - 1) {
                filter_kernel<<<BLOCKS, THREADS>>>(d_cur, d_nxt, d_solid);
                std::swap(d_cur, d_nxt);
            }
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        const auto t1 = Clock::now();
        it += n;
        prof_steps += n;

        // ---- curl, blur, colourise (all on GPU) ----
        curl_kernel<<<BLOCKS, THREADS>>>(d_cur, d_curl);
        float* sbuf = d_curl;
        float* dbuf = d_blur;
        for (int p = 0; p < BLUR_PASSES; p++) {
            gauss_blur<<<BLOCKS, THREADS>>>(sbuf, dbuf);
            std::swap(sbuf, dbuf);
        }
        colourise<<<BLOCKS, THREADS>>>(sbuf, d_solid, d_rgba);
        CK(cudaGetLastError());

#ifndef HEADLESS
        if (interop) {
            CK(cudaGraphicsMapResources(1, &tex_res, 0));
            cudaArray_t arr;
            CK(cudaGraphicsSubResourceGetMappedArray(&arr, tex_res, 0, 0));
            CK(cudaMemcpy2DToArray(arr, 0, 0, d_rgba, Nx * sizeof(uchar4),
                                   Nx * sizeof(uchar4), Ny, cudaMemcpyDeviceToDevice));
            CK(cudaGraphicsUnmapResources(1, &tex_res, 0));
        } else {
            CK(cudaMemcpy(h_rgba, d_rgba, sizeof(uchar4) * N, cudaMemcpyDeviceToHost));
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Nx, Ny, GL_RGBA, GL_UNSIGNED_BYTE, h_rgba);
        }
#endif
        CK(cudaDeviceSynchronize());
        const auto t2 = Clock::now();

        // ---- divergence + shedding checks ----
        if (frame % CHECK_EVERY == 0) {
            CK(cudaMemset(d_peak, 0, sizeof(unsigned int)));
            max_abs_kernel<<<red_blocks, THREADS>>>(sbuf, d_peak);
            unsigned int bits;
            CK(cudaMemcpy(&bits, d_peak, sizeof(bits), cudaMemcpyDeviceToHost));
            float peak;
            std::memcpy(&peak, &bits, sizeof(float));
            if (!std::isfinite(peak) || peak > 10.f) {
                std::printf("Diverged at iter %lld.\n", it);
                break;
            }
            if (!shedding_started && peak > VMAX * 0.5f) {
                shedding_started = true;
                std::printf("Shedding detected at iter %lld; switching to smooth playback.\n", it);
            }
        }

        // ---- draw ----
#ifndef HEADLESS
        int fbw, fbh;
        glfwGetFramebufferSize(win, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS);                      // texture row 0 (y = 0) at the top, as in pygame
            glTexCoord2f(0.f, 1.f); glVertex2f(-1.f, -1.f);
            glTexCoord2f(1.f, 1.f); glVertex2f( 1.f, -1.f);
            glTexCoord2f(1.f, 0.f); glVertex2f( 1.f,  1.f);
            glTexCoord2f(0.f, 0.f); glVertex2f(-1.f,  1.f);
        glEnd();
        glfwSwapBuffers(win);
#else
        if (frame % SNAPSHOT_EVERY == 0) {
            CK(cudaMemcpy(h_rgba, d_rgba, sizeof(uchar4) * N, cudaMemcpyDeviceToHost));
            char path[64];
            std::snprintf(path, sizeof(path), "frame_%05d.ppm", frame);
            write_ppm(path, h_rgba);
        }
#endif
        const auto t3 = Clock::now();

        t_sim += secs(t0, t1); t_post += secs(t1, t2); t_draw += secs(t2, t3);
        const double dt = secs(t_prev, t3);
        t_prev = t3;
        if (dt > 0) fps = (fps == 0.0) ? 1.0 / dt : 0.9 * fps + 0.1 / dt;
        frame++;

#ifndef HEADLESS
        title_t += dt;
        if (title_t > 0.25) {                   // the overlay text now lives in the title bar
            char title[160];
            std::snprintf(title, sizeof(title),
                          "Showcase, not validated | Karman Vortex Street | Re~%.0f  iter %lld  %d steps/frame  %.1f FPS",
                          Re, it, n, fps);
            glfwSetWindowTitle(win, title);
            title_t = 0;
        }
#endif

        if (frame % PROFILE_EVERY == 0) {
            const double tot = t_sim + t_post + t_draw;
            const double mlups = prof_steps * (double)N / t_sim / 1e6;
            std::printf("[%5d] %5d steps/frame | sim %4.1f%%  post %4.1f%%  draw %4.1f%%  | "
                        "%9.0f steps/s  %5.1f FPS  %7.0f MLUPS\n",
                        frame, n, 100 * t_sim / tot, 100 * t_post / tot, 100 * t_draw / tot,
                        prof_steps / tot, PROFILE_EVERY / tot, mlups);
            t_sim = t_post = t_draw = 0.0;
            prof_steps = 0;
        }
    }

    // ---------------- cleanup ----------------
#ifndef HEADLESS
    if (interop) cudaGraphicsUnregisterResource(tex_res);
    glDeleteTextures(1, &tex);
    glfwDestroyWindow(win);
    glfwTerminate();
#endif
    if (h_rgba) cudaFreeHost(h_rgba);
    cudaFree(d_fa); cudaFree(d_fb); cudaFree(d_curl); cudaFree(d_blur);
    cudaFree(d_solid); cudaFree(d_rgba); cudaFree(d_peak);
    return 0;
}
