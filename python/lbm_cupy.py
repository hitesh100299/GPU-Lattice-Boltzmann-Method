# First prototype of this project (CuPy and pygame), kept for history and as the starting point.
# It runs at Mach 0.35 with a velocity clamp, a density filter and periodic top and bottom;
# these issues are fixed in the CUDA solver (cuda/, see the README). Packages: python/requirements.txt.

import os
if os.name == "nt" and "CUDA_PATH" not in os.environ:
    print("Hint: CUDA_PATH is not set. If CuPy cannot find CUDA, set it to your toolkit folder, "
          r"for example C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")

import time
import cupy as cp
import numpy as np
import pygame

# ================================================================== #
#  parameters
# ================================================================== #
Nx      = 1600
Ny      = 400
tau     = 0.57
u0_max  = 0.20
Nt      = 1_000_000

# Both MUST be even: the kernel ping-pongs between two fixed argument tuples,
# so an even count always leaves the current state back in f_in.
STEPS_PER_FRAME_FAST   = 2000   # fast-forward through the laminar startup
STEPS_PER_FRAME_SMOOTH = 120    # playback once shedding begins
assert STEPS_PER_FRAME_FAST % 2 == 0 and STEPS_PER_FRAME_SMOOTH % 2 == 0

VMAX          = 0.05    # curl magnitude mapped to the top of the colormap
BLUR_PASSES   = 2        # display-only smoothing passes (0, 1 or 2)
SCALE         = 1        # window pixel scale (window = Nx*SCALE x Ny*SCALE)
CHECK_EVERY   = 30       # frames between divergence checks (costs a transfer)
PROFILE_EVERY = 30

# Anti-checkerboard filter: only needed at high Re with Smagorinsky active.
USE_FILTER   = True
FILTER_EVERY = 3

# ================================================================== #
#  LBM kernel
# ================================================================== #
kernel_src = r"""
extern "C" __global__
void lbm_step(
    const float* __restrict__ f_in,
    float*       __restrict__ f_out,
    const unsigned char* __restrict__ solid,
    const int Nx, const int Ny,
    const float tau_base, const float u0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int x = idx % Nx;
    int y = idx / Nx;

    const int   cx[9] = {0, 0, 1, 1,  1,  0, -1, -1, -1};
    const int   cy[9] = {0, 1, 1, 0, -1, -1, -1,  0,  1};
    const float  w[9] = {4.f/9.f,
                         1.f/9.f, 1.f/36.f, 1.f/9.f, 1.f/36.f,
                         1.f/9.f, 1.f/36.f, 1.f/9.f, 1.f/36.f};
    const int  opp[9] = {0, 5, 6, 7, 8, 1, 2, 3, 4};

    // inlet: flat equilibrium profile at u0
    if (x == 0 && !solid[idx]) {
        float u2 = 1.5f * u0 * u0;
        #pragma unroll
        for (int i=0;i<9;i++) {
            float cu = 3.f * cx[i] * u0;
            f_out[idx*9+i] = w[i]*(1.f+cu+0.5f*cu*cu-u2);
        }
        return;
    }
    // outlet: zero-gradient
    if (x == Nx-1) {
        int src = (y*Nx+(Nx-2))*9;
        #pragma unroll
        for (int i=0;i<9;i++) f_out[idx*9+i] = f_in[src+i];
        return;
    }

    // pull streaming
    float f[9];
    #pragma unroll
    for (int i=0;i<9;i++) {
        int sx=x-cx[i]; if(sx<0) sx+=Nx; else if(sx>=Nx) sx-=Nx;
        int sy=y-cy[i]; if(sy<0) sy+=Ny; else if(sy>=Ny) sy-=Ny;
        f[i] = f_in[(sy*Nx+sx)*9+i];
    }

    // solid: bounce-back
    if (solid[idx]) {
        #pragma unroll
        for (int i=0;i<9;i++) f_out[idx*9+i]=f[opp[i]];
        return;
    }

    float rho=0.f,ux=0.f,uy=0.f;
    #pragma unroll
    for (int i=0;i<9;i++){rho+=f[i];ux+=f[i]*cx[i];uy+=f[i]*cy[i];}
    float inv_rho=1.f/rho; ux*=inv_rho; uy*=inv_rho;
    float umag=sqrtf(ux*ux+uy*uy);
    if (umag>0.45f){float s=0.45f/umag;ux*=s;uy*=s;}

    float u2=1.5f*(ux*ux+uy*uy);
    float feq[9];
    #pragma unroll
    for (int i=0;i<9;i++){
        float cu=3.f*(cx[i]*ux+cy[i]*uy);
        feq[i]=rho*w[i]*(1.f+cu+0.5f*cu*cu-u2);
    }

    // Smagorinsky (Cs=0 disables it -> plain BGK, correct for low Re)
    float Qxx=0.f,Qxy=0.f,Qyy=0.f;
    #pragma unroll
    for (int i=0;i<9;i++){
        float fn=f[i]-feq[i];
        Qxx+=fn*cx[i]*cx[i]; Qxy+=fn*cx[i]*cy[i]; Qyy+=fn*cy[i]*cy[i];
    }
    float Q      = sqrtf(2.f*(Qxx*Qxx+2.f*Qxy*Qxy+Qyy*Qyy));
    float Cs     = 0.0f;
    float tau_eff= 0.5f*(tau_base+sqrtf(tau_base*tau_base+18.f*Cs*Cs*Q));
    float omega  = 1.f/tau_eff;
    #pragma unroll
    for (int i=0;i<9;i++) f_out[idx*9+i]=f[i]+omega*(feq[i]-f[i]);
}
"""

filter_src = r"""
extern "C" __global__
void filter_kernel(float* __restrict__ f,
                   const unsigned char* __restrict__ solid,
                   const int Nx, const int Ny)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=Nx*Ny) return;
    int x=idx%Nx, y=idx/Nx;
    if (solid[idx]||x==0||x==Nx-1||y==0||y==Ny-1) return;
    float rho_avg=0.f;
    for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++)
        for (int i=0;i<9;i++) rho_avg+=f[((y+dy)*Nx+(x+dx))*9+i];
    rho_avg/=9.f;
    float rho_loc=0.f;
    for (int i=0;i<9;i++) rho_loc+=f[idx*9+i];
    if (rho_loc<1e-6f) return;
    float scale=rho_avg/rho_loc;
    for (int i=0;i<9;i++) f[idx*9+i]*=scale;
}
"""

curl_src = r"""
extern "C" __global__
void curl_kernel(const float* __restrict__ f, float* __restrict__ curl,
                 const int Nx, const int Ny)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=Nx*Ny) return;
    int x=idx%Nx,y=idx/Nx;
    if (x==0||x==Nx-1||y==0||y==Ny-1){curl[idx]=0.f;return;}
    const int cx[9]={0,0,1,1,1,0,-1,-1,-1};
    const int cy[9]={0,1,1,0,-1,-1,-1,0,1};
    float ra=0,rb=0,rc=0,rd=0,uxa=0,uxb=0,uyc=0,uyd=0;
    int a=((y+1)*Nx+x)*9,b=((y-1)*Nx+x)*9;
    int c=(y*Nx+(x+1))*9,d=(y*Nx+(x-1))*9;
    #pragma unroll
    for (int i=0;i<9;i++){
        ra+=f[a+i];uxa+=f[a+i]*cx[i];
        rb+=f[b+i];uxb+=f[b+i]*cx[i];
        rc+=f[c+i];uyc+=f[c+i]*cy[i];
        rd+=f[d+i];uyd+=f[d+i]*cy[i];
    }
    curl[idx]=uxa/ra-uxb/rb-(uyc/rc-uyd/rd);
}
"""

gauss_src = r"""
extern "C" __global__
void gauss_blur(const float* __restrict__ src, float* __restrict__ dst,
                const int Nx, const int Ny)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=Nx*Ny) return;
    int x=idx%Nx, y=idx/Nx;
    if (x==0||x==Nx-1||y==0||y==Ny-1){dst[idx]=src[idx];return;}
    const float k[3][3] = {{0.0625f,0.125f,0.0625f},
                           {0.125f, 0.25f, 0.125f },
                           {0.0625f,0.125f,0.0625f}};
    float v=0.f;
    for (int dy=-1;dy<=1;dy++)
        for (int dx=-1;dx<=1;dx++)
            v += k[dy+1][dx+1] * src[(y+dy)*Nx+(x+dx)];
    dst[idx]=v;
}
"""

colour_src = r"""
__constant__ unsigned char LUT[256][3] = {
LUT_ROWS
};

extern "C" __global__
void colourise(const float* __restrict__ curl,
               const unsigned char* __restrict__ solid,
               unsigned char* __restrict__ rgb,
               const int Nx, const int Ny, const float vmax)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= Nx*Ny) return;
    int x = idx % Nx;
    int y = idx / Nx;

    int c;
    if (solid[idx]) {
        c = -1;                      // flag: draw the cylinder, not the field
    } else {
        float v = fabsf(curl[idx]) / vmax;
        v = fminf(fmaxf(v, 0.f), 1.f);
        c = (int)(v * 255.f + 0.5f);
    }

    // pygame surfarray wants (width, height, 3)
    int o = (x * Ny + y) * 3;
    if (c < 0) {
        rgb[o+0] = 38; rgb[o+1] = 38; rgb[o+2] = 45;
    } else {
        rgb[o+0] = LUT[c][0];
        rgb[o+1] = LUT[c][1];
        rgb[o+2] = LUT[c][2];
    }
}
""".replace("LUT_ROWS", r"""
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
""")

print("Compiling CUDA kernels...")
lbm_step      = cp.RawKernel(kernel_src, "lbm_step")
filter_kernel = cp.RawKernel(filter_src, "filter_kernel")
curl_kernel   = cp.RawKernel(curl_src,   "curl_kernel")
gauss_blur    = cp.RawKernel(gauss_src,  "gauss_blur")
colourise     = cp.RawKernel(colour_src, "colourise")
print("Ready.")

# ================================================================== #
#  host setup
# ================================================================== #
N   = Nx * Ny
cxs = np.array([0,0,1,1,1,0,-1,-1,-1], dtype=np.float32)
w   = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36], dtype=np.float32)

yy, xx   = np.mgrid[0:Ny, 0:Nx]
cyl_r    = int(22 * Ny / 180)
cx0, cy0 = Nx//5, Ny//2 + 2          # +2 breaks symmetry -> shedding starts sooner
dist2    = (xx - cx0)**2 + (yy - cy0)**2
cylinder = dist2 < cyl_r**2
solid_gpu = cp.array(cylinder.ravel().astype(np.uint8))

# uniform init matching the flat inlet
f0 = np.zeros((N, 9), dtype=np.float32)
for i in range(9):
    cu = 3.0 * cxs[i] * u0_max
    f0[:, i] = w[i] * (1.0 + cu + 0.5*cu**2 - 1.5*u0_max**2)
f0 += (0.002 * np.random.randn(N, 9)).astype(np.float32)

f_in      = cp.array(f0.ravel())
f_out     = cp.empty_like(f_in)
curl_buf  = cp.empty(N, dtype=cp.float32)
curl_blur = cp.empty(N, dtype=cp.float32)
rgb_gpu   = cp.empty(N*3, dtype=cp.uint8)

tau_f, u0_f = np.float32(tau), np.float32(u0_max)
vmax_f      = np.float32(VMAX)
threads = 256
blocks  = (N + threads - 1) // threads
grid, block = (blocks,), (threads,)

# scalars converted once; rebuilding these per launch costs real Python time
NX_I, NY_I = np.int32(Nx), np.int32(Ny)
args_a = (f_in,  f_out, solid_gpu, NX_I, NY_I, tau_f, u0_f)
args_b = (f_out, f_in,  solid_gpu, NX_I, NY_I, tau_f, u0_f)

nu = (2*tau - 1) / 6
Re = u0_max * (2*cyl_r) / nu

# ================================================================== #
#  batched stepping (CUDA graph with plain-loop fallback)
# ================================================================== #
def run_steps_plain(n):
    for k in range(n):
        lbm_step(grid, block, args_a if (k & 1) == 0 else args_b)
        if USE_FILTER and (k % FILTER_EVERY) == 0:
            filter_kernel(grid, block, (f_in, solid_gpu, NX_I, NY_I))

graph_cache = {}

def run_steps(n):
    entry = graph_cache.get(n)
    if entry is None:
        try:
            s = cp.cuda.Stream(non_blocking=True)
            with s:
                s.begin_capture()
                for k in range(n):
                    lbm_step(grid, block, args_a if (k & 1) == 0 else args_b)
                g = s.end_capture()
            graph_cache[n] = (g, s)
            print(f"CUDA graph captured: {n} steps per launch.")
            entry = graph_cache[n]
        except Exception as e:
            print(f"CUDA graph unavailable ({type(e).__name__}); plain launches.")
            graph_cache[n] = False
            entry = False
    if entry is False:
        run_steps_plain(n)
    else:
        g, s = entry
        g.launch(s)
        s.synchronize()

# ================================================================== #
#  pygame window
# ================================================================== #
pygame.init()
screen = pygame.display.set_mode((Nx*SCALE, Ny*SCALE))
pygame.display.set_caption("Karman Vortex Street")
surf  = pygame.Surface((Nx, Ny))
font  = pygame.font.SysFont("consolas", 16)
clock = pygame.time.Clock()

print(f"Re ~ {Re:.0f}   nu = {nu:.4f}   grid {Nx}x{Ny}")
print("ESC or close window to quit.")

# ================================================================== #
#  main loop
# ================================================================== #
it = frame = 0
shedding_started = False
fps = 0.0
t_sim = t_post = t_draw = 0.0
running = True

while running and it < Nt:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            running = False

    n = STEPS_PER_FRAME_SMOOTH if shedding_started else STEPS_PER_FRAME_FAST

    # ---- physics ----
    t0 = time.perf_counter()
    run_steps(n)                     # n is even -> state ends back in f_in
    cp.cuda.runtime.deviceSynchronize()
    t1 = time.perf_counter()
    it += n

    # ---- curl, blur, colourise (all on GPU) ----
    curl_kernel(grid, block, (f_in, curl_buf, NX_I, NY_I))
    sbuf, dbuf = curl_buf, curl_blur
    for _ in range(BLUR_PASSES):
        gauss_blur(grid, block, (sbuf, dbuf, NX_I, NY_I))
        sbuf, dbuf = dbuf, sbuf
    colourise(grid, block, (sbuf, solid_gpu, rgb_gpu, NX_I, NY_I, vmax_f))

    rgb = rgb_gpu.reshape(Nx, Ny, 3).get()      # one small uint8 transfer
    t2 = time.perf_counter()

    # divergence + shedding checks: occasional, they cost a transfer
    if frame % CHECK_EVERY == 0:
        peak = float(cp.abs(sbuf).max())
        if not np.isfinite(peak) or peak > 10.0:
            print(f"Diverged at iter {it}.")
            break
        if not shedding_started and peak > VMAX * 0.5:
            shedding_started = True
            print(f"Shedding detected at iter {it}; switching to smooth playback.")

    # ---- draw ----
    pygame.surfarray.blit_array(surf, rgb)
    pygame.transform.scale(surf, (Nx*SCALE, Ny*SCALE), screen)
    screen.blit(font.render(
        f"Re~{Re:.0f}  iter {it}  {n} steps/frame  {fps:.1f} FPS",
        True, (255,255,255)), (8, 6))
    pygame.display.flip()
    t3 = time.perf_counter()

    t_sim += t1-t0; t_post += t2-t1; t_draw += t3-t2
    fps = clock.get_fps()
    clock.tick()
    frame += 1

    if frame % PROFILE_EVERY == 0:
        tot = t_sim + t_post + t_draw
        print(f"[{frame:5d}] {n:5d} steps/frame | "
              f"sim {100*t_sim/tot:4.1f}%  post {100*t_post/tot:4.1f}%  "
              f"draw {100*t_draw/tot:4.1f}%  | "
              f"{n*PROFILE_EVERY/tot:9.0f} steps/s  {PROFILE_EVERY/tot:5.1f} FPS")
        t_sim = t_post = t_draw = 0.0

pygame.quit()
