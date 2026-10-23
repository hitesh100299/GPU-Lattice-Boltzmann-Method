// lbm_core.cuh
// The one kernel set used by the benchmark driver (lbm_bench.cu) and the
// real-time demo (lbm_cuda.cu). Header only.
//
//   D2Q9, BGK single-relaxation-time, pull streaming fused with collision.
//   SoA layout: population i of cell idx lives at F(i, idx) = i * N + idx.
//   Precision: `real` is float by default, double with -DUSE_DOUBLE.
//   Node types: FLUID, WALL (channel walls), CYL (cylinder), INLET (Zou and He
//   velocity, parabolic profile), OUTLET (Zou and He pressure, rho = 1).
//   Storage: the arrays hold the shifted populations g_i = f_i - w_i, the
//   deviation from the rest-state equilibrium (Skordos 1993). Streaming,
//   bounce-back and the Zou and He relations are unchanged in g (w_i = w_opp(i), and
//   the weights that enter the Zou and He density sum add up to 1); rho = 1 + sum g,
//   momentum = sum g c. This keeps single precision usable at low lattice
//   velocities, where the viscous stress is a tiny difference f - feq on
//   populations of order 0.4; the plane Poiseuille check failed without it.
//   Solid walls: halfway bounce-back in pull form. If the source node of
//   direction i is solid, the node takes its own post-collision population of
//   the opposite direction from the previous step: fin[F(D2Q9::opp(i), idx)].
//   Force on the cylinder: momentum exchange over the flagged links, reduced on
//   the GPU in double precision, two numbers copied to the host.
//
// Dropped compared with the first CUDA version (showcase/karman_showcase.cu), see
// docs/DESIGN_NOTES.md: the 0.45 velocity clamp, the density filter, the
// Smagorinsky term, the random initial noise, the uniform equilibrium inlet and
// periodic top and bottom.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include "cases.h"

#ifdef USE_DOUBLE
typedef double real;
#define PRECISION_NAME "fp64"
#else
typedef float real;
#define PRECISION_NAME "fp32"
#endif

// Error messages name the source file by a short name fixed at compile time. __FILE__
// would put the build machine's absolute path into every executable. The drivers
// redefine LBM_SRC_FILE after including this header, so each message names the file
// the failing call is in.
#define LBM_SRC_FILE "lbm_core.cuh"
#define CK(call) do {                                                        \
    cudaError_t e_ = (call);                                                 \
    if (e_ != cudaSuccess) {                                                 \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n  in %s\n",           \
                     LBM_SRC_FILE, __LINE__, cudaGetErrorString(e_), #call);  \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

#define F(i, idx) ((i) * N + (idx))

enum NodeType : unsigned char { FLUID = 0, WALL = 1, INLET = 2, OUTLET = 3, CYL = 4 };
__host__ __device__ inline bool is_solid(unsigned char t) { return t == WALL || t == CYL; }

// D2Q9 lattice constants, usable in host and device code:
//   i:   0  1  2  3  4  5  6  7  8
//   cx:  0  0  1  1  1  0 -1 -1 -1
//   cy:  0  1  1  0 -1 -1 -1  0  1
//   opp: 0  5  6  7  8  1  2  3  4
//   w:   4/9 for i = 0, 1/9 for odd i (straight), 1/36 for even i > 0 (diagonal)
// They are constexpr functions rather than static constexpr arrays, because CUDA
// only lets scalar constexpr values into device code. Inside the unrolled loops
// every call folds to a compile-time constant.
struct D2Q9 {
    __host__ __device__ static constexpr int cx(int i) { return (i >= 2 && i <= 4) ? 1 : (i >= 6 ? -1 : 0); }
    __host__ __device__ static constexpr int cy(int i) { return (i == 1 || i == 2 || i == 8) ? 1 : ((i >= 4 && i <= 6) ? -1 : 0); }
    __host__ __device__ static constexpr int opp(int i) { return i == 0 ? 0 : (i <= 4 ? i + 4 : i - 4); }
    __host__ __device__ static constexpr double w(int i) { return i == 0 ? 4.0 / 9.0 : ((i & 1) ? 1.0 / 9.0 : 1.0 / 36.0); }
};
static_assert(D2Q9::cx(4) == 1 && D2Q9::cx(7) == -1 && D2Q9::cy(8) == 1 && D2Q9::cy(5) == -1, "D2Q9 velocities");
static_assert(D2Q9::opp(1) == 5 && D2Q9::opp(8) == 4 && D2Q9::w(3) == 1.0 / 9.0 && D2Q9::w(6) == 1.0 / 36.0, "D2Q9 opposites and weights");

constexpr int THREADS = 256;

// ================================================================== //
//  LBM step: pull streaming + boundary reconstruction + BGK collision
// ================================================================== //
__global__ void lbm_step(const real* __restrict__ fin,
                         real*       __restrict__ fout,
                         const unsigned char* __restrict__ type,
                         const real* __restrict__ uin,     // inlet u(j), length Ny
                         const int Nx, const int Ny, const real omega)
{
    // 2D launch: blockIdx.y is the row, so no integer division is needed
    const int N = Nx * Ny;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    if (x >= Nx) return;
    const int idx = y * Nx + x;
    const unsigned char t = type[idx];
    if (is_solid(t)) return;                    // solids carry no populations

    // pull streaming with halfway bounce-back
    real f[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        const int sx = x - D2Q9::cx(i), sy = y - D2Q9::cy(i);
        if (sx < 0 || sx >= Nx) { f[i] = (real)0; continue; }   // outside: reconstructed below
        const int src = sy * Nx + sx;             // sy is always inside: rows 0 and Ny-1 are solid
        f[i] = is_solid(type[src]) ? fin[F(D2Q9::opp(i), idx)] : fin[F(i, src)];
    }

    real rho, ux, uy;
    if (t == INLET) {
        // Zou and He velocity inlet (west face): unknown 2, 3, 4 (cx = +1)
        ux = uin[y]; uy = (real)0;
        rho = ((real)1 + f[0] + f[1] + f[5] + (real)2 * (f[6] + f[7] + f[8])) / ((real)1 - ux);   // weights sum to 1
        const real d = (real)0.5 * (f[1] - f[5]);
        f[3] = f[7] + (real)(2.0 / 3.0) * rho * ux;
        f[2] = f[6] - d + (real)(1.0 / 6.0) * rho * ux + (real)0.5 * rho * uy;
        f[4] = f[8] + d + (real)(1.0 / 6.0) * rho * ux - (real)0.5 * rho * uy;
    } else if (t == OUTLET) {
        // Zou and He pressure outlet (east face): rho = 1, uy = 0, unknown 6, 7, 8 (cx = -1)
        rho = (real)1;
        ux = (real)-1 + ((real)1 + f[0] + f[1] + f[5] + (real)2 * (f[2] + f[3] + f[4])) / rho;
        uy = (real)0;
        const real d = (real)0.5 * (f[1] - f[5]);
        f[7] = f[3] - (real)(2.0 / 3.0) * rho * ux;
        f[6] = f[2] + d - (real)(1.0 / 6.0) * rho * ux;
        f[8] = f[4] - d - (real)(1.0 / 6.0) * rho * ux;
    } else {
        real drho = (real)0; ux = (real)0; uy = (real)0;
        #pragma unroll
        for (int i = 0; i < 9; i++) { drho += f[i]; ux += f[i] * D2Q9::cx(i); uy += f[i] * D2Q9::cy(i); }
        rho = (real)1 + drho;
        const real inv = (real)1 / rho;
        ux *= inv; uy *= inv;
    }

    // BGK collision on the shifted populations: geq_i = feq_i - w_i = w_i ((rho - 1) + rho (cu + cu^2/2 - u^2))
    const real drho = rho - (real)1;
    const real u2 = (real)1.5 * (ux * ux + uy * uy);
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        const real cu  = (real)3 * (D2Q9::cx(i) * ux + D2Q9::cy(i) * uy);
        const real geq = (real)D2Q9::w(i) * (drho + rho * (cu + (real)0.5 * cu * cu - u2));
        fout[F(i, idx)] = f[i] + omega * (geq - f[i]);
    }
}

// ================================================================== //
//  Momentum exchange on the cylinder links.
//  link = idx * 9 + q: fluid node idx, direction q pointing into a CYL node.
//  F = sum 2 c_q f_q*(x) with post-collision populations (bounce-back returns
//  the same population, so incoming and outgoing both contribute c_q f_q*).
//  Reduction in double: warp shuffles, then shared memory across the warps of
//  a block, then one atomicAdd per block and component.
// ================================================================== //
__global__ void force_kernel(const real* __restrict__ f, const int* __restrict__ links,
                             const int nlinks, const int N, double* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double fx = 0, fy = 0;
    if (i < nlinks) {
        const int l = links[i];
        const int idx = l / 9, q = l - 9 * idx;
        const double v = 2.0 * ((double)f[F(q, idx)] + (double)(real)D2Q9::w(q));   // f_q* = g_q* + w_q
        fx = v * D2Q9::cx(q); fy = v * D2Q9::cy(q);
    }
    // 1. reduce within each warp with register shuffles (every thread of the block takes part)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        fx += __shfl_down_sync(0xffffffffu, fx, off);
        fy += __shfl_down_sync(0xffffffffu, fy, off);
    }
    // 2. one partial sum per warp in shared memory, reduced by the first warp
    __shared__ double wx[THREADS / 32], wy[THREADS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { wx[warp] = fx; wy[warp] = fy; }
    __syncthreads();
    if (warp == 0) {
        fx = lane < (int)(blockDim.x >> 5) ? wx[lane] : 0.0;
        fy = lane < (int)(blockDim.x >> 5) ? wy[lane] : 0.0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            fx += __shfl_down_sync(0xffffffffu, fx, off);
            fy += __shfl_down_sync(0xffffffffu, fy, off);
        }
        // 3. one double-precision atomic per block and component
        if (lane == 0) { atomicAdd(&out[0], fx); atomicAdd(&out[1], fy); }
    }
}

// density at a short list of probe nodes (rho is a collision invariant, so
// post-collision populations give the same value as pre-collision ones)
__global__ void probe_kernel(const real* __restrict__ f, const int* __restrict__ nodes,
                             const int n, const int N, double* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int idx = nodes[i];
    double r = 1.0;
    #pragma unroll
    for (int q = 0; q < 9; q++) r += (double)f[F(q, idx)];
    out[i] = r;
}

// macroscopic fields (zero in solids)
__global__ void macro_kernel(const real* __restrict__ f, const unsigned char* __restrict__ type,
                             const int N, real* __restrict__ rho, real* __restrict__ ux, real* __restrict__ uy)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (is_solid(type[idx])) { rho[idx] = (real)0; ux[idx] = (real)0; uy[idx] = (real)0; return; }
    real r = (real)1, vx = 0, vy = 0;
    #pragma unroll
    for (int i = 0; i < 9; i++) { const real v = f[F(i, idx)]; r += v; vx += v * D2Q9::cx(i); vy += v * D2Q9::cy(i); }
    rho[idx] = r; ux[idx] = vx / r; uy[idx] = vy / r;
}

// vorticity omega = d(uy)/dx - d(ux)/dy, central differences divided by 2,
// velocity zero on solid neighbours (macro_kernel writes zeros there).
__global__ void vort_kernel(const real* __restrict__ ux, const real* __restrict__ uy,
                            const unsigned char* __restrict__ type,
                            const int Nx, const int Ny, float* __restrict__ vort)
{
    const int N = Nx * Ny;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const int x = idx % Nx, y = idx / Nx;
    if (is_solid(type[idx]) || x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1) { vort[idx] = 0.f; return; }
    const real duy_dx = (real)0.5 * (uy[idx + 1] - uy[idx - 1]);
    const real dux_dy = (real)0.5 * (ux[idx + Nx] - ux[idx - Nx]);
    vort[idx] = (float)(duy_dx - dux_dy);
}

// ================================================================== //
//  Host-side: simulation object shared by benchmark and demo
// ================================================================== //
struct Sim {
    dfg::Case cs;
    int N = 0, blocks = 0;
    real *d_fa = nullptr, *d_fb = nullptr, *d_cur = nullptr, *d_nxt = nullptr;
    unsigned char* d_type = nullptr;
    real* d_uin = nullptr;
    int* d_links = nullptr; int nlinks = 0;
    int* d_probe = nullptr; int nprobe = 0;
    double* d_out = nullptr;                  // [0..1] force, [2..] probe densities
    real *d_rho = nullptr, *d_ux = nullptr, *d_uy = nullptr; float* d_vort = nullptr;
    std::vector<unsigned char> h_type;
    std::vector<double> h_out;
    long long step_count = 0;
    cudaEvent_t ev0, ev1;
    double kernel_ms = 0;                     // CUDA event time of lbm_step launches only
    // pressure probe bookkeeping: the two rows around y = 0.2; for each row the two
    // nearest fluid nodes in front (i1f, i2f) and behind (i1b, i2b) the cylinder
    struct Probe { int j, i1f, i2f, i1b, i2b; double wgt; } probe[2];
    double xa_lat = 0, xe_lat = 0;

    explicit Sim(const dfg::Case& c, bool from_rest = false) : cs(c) {
        N = cs.Nx * cs.Ny;
        blocks = (N + THREADS - 1) / THREADS;
        build_types();
        build_links_and_probes();
        alloc();
        init_populations(from_rest);
        CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    }
    ~Sim() {
        cudaFree(d_fa); cudaFree(d_fb); cudaFree(d_type); cudaFree(d_uin); cudaFree(d_links);
        cudaFree(d_probe); cudaFree(d_out); cudaFree(d_rho); cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_vort);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }

    void build_types() {
        const int Nx = cs.Nx, Ny = cs.Ny;
        h_type.assign(N, FLUID);
        for (int x = 0; x < Nx; x++) { h_type[x] = WALL; h_type[(Ny - 1) * Nx + x] = WALL; }
        for (int y = 1; y < Ny - 1; y++) { h_type[y * Nx] = INLET; h_type[y * Nx + Nx - 1] = OUTLET; }
        if (cs.cylinder)
            for (int y = 1; y < Ny - 1; y++)
                for (int x = 1; x < Nx - 1; x++) {
                    const double dx = x - cs.xc, dy = y - cs.yc;
                    if (dx * dx + dy * dy <= cs.r2) h_type[y * Nx + x] = CYL;
                }
    }

    void build_links_and_probes() {
        std::vector<int> links;
        const int Nx = cs.Nx, Ny = cs.Ny;
        if (cs.cylinder) {
            for (int y = 1; y < Ny - 1; y++)
                for (int x = 1; x < Nx - 1; x++) {
                    const int idx = y * Nx + x;
                    if (h_type[idx] != FLUID) continue;
                    for (int q = 1; q < 9; q++) {
                        const int nx = x + D2Q9::cx(q), ny = y + D2Q9::cy(q);
                        if (h_type[ny * Nx + nx] == CYL) links.push_back(idx * 9 + q);
                    }
                }
            // pressure probes A1 = (0.15, 0.2), A2 = (0.25, 0.2). Node row j sits at
            // y = (j - 0.5) dx, so y = 0.2 (= 2 Nd dx) lies exactly between rows
            // j0 = 2 Nd and j1 = 2 Nd + 1 (yc = 2 Nd + 0.5 is the row coordinate of the centre).
            const int j0 = (int)std::lround(cs.yc - 0.5), j1 = j0 + 1;
            xa_lat = dfg::XA_PHYS / cs.dx; xe_lat = dfg::XE_PHYS / cs.dx;
            const int jj[2] = {j0, j1};
            for (int k = 0; k < 2; k++) {
                const int j = jj[k];
                Probe& p = probe[k];
                p.j = j;
                p.wgt = 0.5;    // y = 0.2 lies exactly between the two rows
                int i = (int)std::lround(cs.xc);
                while (h_type[j * Nx + i] == CYL) i--;
                p.i1f = i; p.i2f = i - 1;
                i = (int)std::lround(cs.xc);
                while (h_type[j * Nx + i] == CYL) i++;
                p.i1b = i; p.i2b = i + 1;
            }
        }
        nlinks = (int)links.size();
        std::vector<int> nodes;
        if (cs.cylinder)
            for (int k = 0; k < 2; k++) {
                nodes.push_back(probe[k].j * Nx + probe[k].i1f); nodes.push_back(probe[k].j * Nx + probe[k].i2f);
                nodes.push_back(probe[k].j * Nx + probe[k].i1b); nodes.push_back(probe[k].j * Nx + probe[k].i2b);
            }
        nprobe = (int)nodes.size();
        h_out.assign(2 + nprobe, 0.0);
        CK(cudaMalloc(&d_links, sizeof(int) * std::max(1, nlinks)));
        CK(cudaMalloc(&d_probe, sizeof(int) * std::max(1, nprobe)));
        CK(cudaMalloc(&d_out, sizeof(double) * (2 + nprobe)));
        if (nlinks) CK(cudaMemcpy(d_links, links.data(), sizeof(int) * nlinks, cudaMemcpyHostToDevice));
        if (nprobe) CK(cudaMemcpy(d_probe, nodes.data(), sizeof(int) * nprobe, cudaMemcpyHostToDevice));
    }

    void alloc() {
        CK(cudaMalloc(&d_fa, sizeof(real) * 9 * (size_t)N));
        CK(cudaMalloc(&d_fb, sizeof(real) * 9 * (size_t)N));
        CK(cudaMalloc(&d_type, N));
        CK(cudaMalloc(&d_uin, sizeof(real) * cs.Ny));
        CK(cudaMemcpy(d_type, h_type.data(), N, cudaMemcpyHostToDevice));
        std::vector<real> uin(cs.Ny, (real)0);
        for (int j = 1; j < cs.Ny - 1; j++) uin[j] = (real)cs.inlet_u(j);
        CK(cudaMemcpy(d_uin, uin.data(), sizeof(real) * cs.Ny, cudaMemcpyHostToDevice));
        d_cur = d_fa; d_nxt = d_fb;
    }

    // Initial state: equilibrium of the analytic channel flow (inlet parabola in
    // every column, zero velocity in solids) with the matching linear pressure
    // drop rho(i) = 1 + 3 G (Nx - 1 - i), G = 12 nu U_mean / H^2. No ramp.
    // from_rest: rho = 1, u = 0 everywhere.
    void init_populations(bool from_rest) {
        const int Nx = cs.Nx, Ny = cs.Ny;
        const double G = 12.0 * cs.nu_lat * cs.umean_lat / (cs.H * cs.H);
        std::vector<real> h_f((size_t)9 * N);
        for (int y = 0; y < Ny; y++)
            for (int x = 0; x < Nx; x++) {
                const int idx = y * Nx + x;
                double rho = 1.0, u = 0.0;
                if (!from_rest) { rho = 1.0 + 3.0 * G * (Nx - 1 - x); u = cs.inlet_u(y); }
                if (is_solid(h_type[idx]) || y == 0 || y == Ny - 1) { u = 0.0; }
                const double u2 = 1.5 * u * u;
                for (int i = 0; i < 9; i++) {
                    const double cu = 3.0 * D2Q9::cx(i) * u;
                    h_f[(size_t)i * N + idx] = (real)(rho * D2Q9::w(i) * (1.0 + cu + 0.5 * cu * cu - u2) - D2Q9::w(i));   // shifted: g = feq - w
                }
            }
        CK(cudaMemcpy(d_fa, h_f.data(), sizeof(real) * 9 * (size_t)N, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_fb, h_f.data(), sizeof(real) * 9 * (size_t)N, cudaMemcpyHostToDevice));
    }

    // advance n steps; kernel time accumulated with CUDA events
    void step(int n) {
        CK(cudaEventRecord(ev0));
        for (int k = 0; k < n; k++) {
            lbm_step<<<dim3((cs.Nx + THREADS - 1) / THREADS, cs.Ny), THREADS>>>(d_cur, d_nxt, d_type, d_uin, cs.Nx, cs.Ny, (real)cs.omega);
            std::swap(d_cur, d_nxt);
        }
        CK(cudaEventRecord(ev1));
        CK(cudaGetLastError());
        CK(cudaEventSynchronize(ev1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, ev0, ev1));
        kernel_ms += ms;
        step_count += n;
    }

    // raw shifted populations of one node (for probes in the demo)
    void node_populations(int idx, real g[9]) {
        CK(cudaMemcpy2D(g, sizeof(real), d_cur + idx, (size_t)N * sizeof(real), sizeof(real), 9, cudaMemcpyDeviceToHost));
    }
    // force and probe densities from the current post-collision populations
    struct Sample { double Fx, Fy, cd, cl, dp_lat, dp_phys; bool finite; };
    Sample sample() {
        CK(cudaMemset(d_out, 0, sizeof(double) * (2 + nprobe)));
        if (nlinks) force_kernel<<<(nlinks + THREADS - 1) / THREADS, THREADS>>>(d_cur, d_links, nlinks, N, d_out);
        if (nprobe) probe_kernel<<<1, THREADS>>>(d_cur, d_probe, nprobe, N, d_out + 2);
        CK(cudaGetLastError());
        CK(cudaMemcpy(h_out.data(), d_out, sizeof(double) * (2 + nprobe), cudaMemcpyDeviceToHost));
        Sample s;
        s.Fx = h_out[0]; s.Fy = h_out[1];
        s.cd = cs.force_to_coeff(s.Fx); s.cl = cs.force_to_coeff(s.Fy);
        s.dp_lat = 0;
        if (nprobe == 8) {
            double pa = 0, pe = 0;
            for (int k = 0; k < 2; k++) {
                const Probe& p = probe[k];
                const double r1f = h_out[2 + 4 * k], r2f = h_out[3 + 4 * k];
                const double r1b = h_out[4 + 4 * k], r2b = h_out[5 + 4 * k];
                // linear extrapolation in x from the two nearest fluid nodes to the probe x
                const double ra = r1f + (r1f - r2f) * (xa_lat - p.i1f) / (double)(p.i1f - p.i2f);
                const double re = r1b + (r1b - r2b) * (xe_lat - p.i1b) / (double)(p.i1b - p.i2b);
                pa += p.wgt * ra / 3.0; pe += p.wgt * re / 3.0;    // p = rho cs^2, cs^2 = 1/3
            }
            s.dp_lat = pa - pe;
        }
        s.dp_phys = cs.dp_lat_to_phys(s.dp_lat);
        s.finite = std::isfinite(s.Fx) && std::isfinite(s.Fy) && std::isfinite(s.dp_lat);
        return s;
    }

    void ensure_fields() {
        if (d_rho) return;
        CK(cudaMalloc(&d_rho, sizeof(real) * N)); CK(cudaMalloc(&d_ux, sizeof(real) * N));
        CK(cudaMalloc(&d_uy, sizeof(real) * N)); CK(cudaMalloc(&d_vort, sizeof(float) * N));
    }
    void compute_macro() {
        ensure_fields();
        macro_kernel<<<blocks, THREADS>>>(d_cur, d_type, N, d_rho, d_ux, d_uy);
        CK(cudaGetLastError());
    }
    void compute_vorticity() {
        compute_macro();
        vort_kernel<<<blocks, THREADS>>>(d_ux, d_uy, d_type, cs.Nx, cs.Ny, d_vort);
        CK(cudaGetLastError());
    }
    // download fields as float (rho, ux, uy)
    void fields(std::vector<float>& rho, std::vector<float>& ux, std::vector<float>& uy) {
        compute_macro();
        std::vector<real> tmp(N);
        rho.resize(N); ux.resize(N); uy.resize(N);
        CK(cudaMemcpy(tmp.data(), d_rho, sizeof(real) * N, cudaMemcpyDeviceToHost)); for (int i = 0; i < N; i++) rho[i] = (float)tmp[i];
        CK(cudaMemcpy(tmp.data(), d_ux, sizeof(real) * N, cudaMemcpyDeviceToHost));  for (int i = 0; i < N; i++) ux[i] = (float)tmp[i];
        CK(cudaMemcpy(tmp.data(), d_uy, sizeof(real) * N, cudaMemcpyDeviceToHost));  for (int i = 0; i < N; i++) uy[i] = (float)tmp[i];
    }
    void vorticity(std::vector<float>& v) {
        compute_vorticity();
        v.resize(N);
        CK(cudaMemcpy(v.data(), d_vort, sizeof(float) * N, cudaMemcpyDeviceToHost));
    }
    // total mass over non solid nodes (double on the host, used twice per run)
    double total_mass() {
        std::vector<float> rho, ux, uy; fields(rho, ux, uy);
        double m = 0; for (int i = 0; i < N; i++) if (!is_solid(h_type[i])) m += rho[i];
        return m;
    }
    double mlups_kernel() const { return step_count * (double)N / (kernel_ms * 1e-3) / 1e6; }
};

// ================================================================== //
//  Diagnostics: time series of cD, cL, dp and the derived quantities.
//  Shared by benchmark and demo so that the demo reproduces the benchmark.
// ================================================================== //
struct Diag {
    std::vector<double> t, cd, cl, dp;     // t in units of D / U_mean
    std::vector<double> zc;                // upward zero crossings of cL (times)
    void push(double tt, const Sim::Sample& s) {
        if (!t.empty() && cl.back() < 0 && s.cl >= 0) {
            const double tz = t.back() - cl.back() * (tt - t.back()) / (s.cl - cl.back());
            // debounce: ripple on a near zero lift (startup, acoustic waves) produces clusters of
            // crossings; a shedding period at Re 100 is about 3.3 D/U, so crossings closer than
            // 0.5 D/U to the previous one are not new periods
            if (zc.empty() || tz - zc.back() > 0.5) zc.push_back(tz);
        }
        t.push_back(tt); cd.push_back(s.cd); cl.push_back(s.cl); dp.push_back(s.dp_phys);
    }
    // number of trailing consecutive periods whose length is within 25 percent of the last one
    // (crossings of the irregular startup phase do not count as periods)
    int n_periods() const {
        if (zc.size() < 2) return 0;
        const double last = zc.back() - zc[zc.size() - 2];
        int n = 0;
        for (size_t i = zc.size() - 1; i >= 1; i--) {
            const double p = zc[i] - zc[i - 1];
            if (std::fabs(p - last) > 0.25 * last) break;
            n++;
        }
        return n;
    }
    // mean period over the last k periods, and their relative spread
    double period_mean(int k) const {
        if (n_periods() < k) return NAN;
        return (zc.back() - zc[zc.size() - 1 - k]) / k;
    }
    double period_spread(int k) const {
        if (n_periods() < k) return NAN;
        double mn = 1e300, mx = -1e300;
        for (size_t i = zc.size() - k; i < zc.size(); i++) { const double p = zc[i] - zc[i - 1]; mn = std::min(mn, p); mx = std::max(mx, p); }
        return (mx - mn) / period_mean(k);
    }
    double strouhal(int k = 3) const { return 1.0 / period_mean(k); }   // St = D f / U = 1 / T[D/U]
    // extrema of a series over the time window [t0, t1]
    static void extrema(const std::vector<double>& t, const std::vector<double>& v, double t0, double t1,
                        double& mn, double& mx, double& tmax) {
        mn = 1e300; mx = -1e300; tmax = NAN;
        for (size_t i = 0; i < t.size(); i++) if (t[i] >= t0 && t[i] <= t1) { mn = std::min(mn, v[i]); if (v[i] > mx) { mx = v[i]; tmax = t[i]; } }
    }
    double interp(const std::vector<double>& v, double tt) const {
        size_t i = std::lower_bound(t.begin(), t.end(), tt) - t.begin();
        if (i == 0) return v[0]; if (i >= t.size()) return v.back();
        const double a = (tt - t[i - 1]) / (t[i] - t[i - 1]);
        return v[i - 1] + a * (v[i] - v[i - 1]);
    }
    // 2D-2 results over the last k full periods
    struct Periodic { double cd_max, cd_min, cl_max, cl_min, st, period, period_spread, dp_t0_half, t0, t1; bool valid; };
    Periodic periodic(int k = 3) const {
        Periodic r{}; r.valid = n_periods() >= k;
        if (!r.valid) return r;
        r.t0 = zc[zc.size() - 1 - k]; r.t1 = zc.back();
        double tcl, tcd;
        extrema(t, cd, r.t0, r.t1, r.cd_min, r.cd_max, tcd);
        extrema(t, cl, r.t0, r.t1, r.cl_min, r.cl_max, tcl);
        r.period = period_mean(k); r.period_spread = period_spread(k); r.st = 1.0 / r.period;
        // paper definition: t0 = time of cL max, dp evaluated at t0 + 1/(2 f)
        double th = tcl + 0.5 * r.period; if (th > t.back()) th -= r.period;
        r.dp_t0_half = interp(dp, th);
        return r;
    }
    // 2D-1: relative band of cD over the last `window` time units
    double cd_band(double window) const {
        if (t.empty() || t.back() < window) return NAN;
        double mn, mx, tm; extrema(t, cd, t.back() - window, t.back(), mn, mx, tm);
        return (mx - mn) / std::fabs(0.5 * (mx + mn));
    }
};
