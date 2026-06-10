import os
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot

# ================================================================== #
#  parameters
# ================================================================== #
Nx      = 1020
Ny      = 270
tau     = 0.55      # slightly higher — reduces checkerboard, keeps vortices strong  0.55
u0_max  = 0.30      # 0.28
Nt      = 1_000_000

STEPS_PER_FRAME = 40
VMAX            = 0.07
FILTER_EVERY    = 3     # more frequent anti-checkerboard filtering

# ================================================================== #
#  LBM kernel
# ================================================================== #
kernel_src = r'''
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

    if (x == 0 && !solid[idx]) {
        float yf    = (float)y / (float)(Ny-1);
        float u_loc = u0 * 4.f * yf * (1.f - yf);
        float u2    = 1.5f * u_loc * u_loc;
        #pragma unroll
        for (int i=0;i<9;i++) {
            float cu = 3.f * cx[i] * u_loc;
            f_out[idx*9+i] = w[i]*(1.f+cu+0.5f*cu*cu-u2);
        }
        return;
    }
    if (x == Nx-1) {
        int src = (y*Nx+(Nx-2))*9;
        #pragma unroll
        for (int i=0;i<9;i++) f_out[idx*9+i] = f_in[src+i];
        return;
    }

    float f[9];
    #pragma unroll
    for (int i=0;i<9;i++) {
        int sx=x-cx[i]; if(sx<0) sx+=Nx; else if(sx>=Nx) sx-=Nx;
        int sy=y-cy[i]; if(sy<0) sy+=Ny; else if(sy>=Ny) sy-=Ny;
        f[i] = f_in[(sy*Nx+sx)*9+i];
    }

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

    float Qxx=0.f,Qxy=0.f,Qyy=0.f;
    #pragma unroll
    for (int i=0;i<9;i++){
        float fn=f[i]-feq[i];
        Qxx+=fn*cx[i]*cx[i]; Qxy+=fn*cx[i]*cy[i]; Qyy+=fn*cy[i]*cy[i];
    }
    float Q      = sqrtf(2.f*(Qxx*Qxx+2.f*Qxy*Qxy+Qyy*Qyy));
    float Cs     = 0.18f;
    float tau_eff= 0.5f*(tau_base+sqrtf(tau_base*tau_base+18.f*Cs*Cs*Q));
    float omega  = 1.f/tau_eff;
    #pragma unroll
    for (int i=0;i<9;i++) f_out[idx*9+i]=f[i]+omega*(feq[i]-f[i]);
}
'''

# ================================================================== #
#  anti-checkerboard density filter
# ================================================================== #
filter_src = r'''
extern "C" __global__
void filter_kernel(
    float* __restrict__ f,
    const unsigned char* __restrict__ solid,
    const int Nx, const int Ny)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=Nx*Ny) return;
    int x=idx%Nx, y=idx/Nx;
    if (solid[idx]||x==0||x==Nx-1||y==0||y==Ny-1) return;
    float rho_avg=0.f;
    for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++)
        for (int i=0;i<9;i++)
            rho_avg+=f[((y+dy)*Nx+(x+dx))*9+i];
    rho_avg/=9.f;
    float rho_loc=0.f;
    for (int i=0;i<9;i++) rho_loc+=f[idx*9+i];
    if (rho_loc<1e-6f) return;
    float scale=rho_avg/rho_loc;
    for (int i=0;i<9;i++) f[idx*9+i]*=scale;
}
'''

# ================================================================== #
#  GPU Gaussian blur on curl — purely cosmetic, suppresses stripe artefacts
#  in the display without touching the physics arrays at all
# ================================================================== #
gauss_src = r'''
extern "C" __global__
void gauss_blur(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    const int Nx, const int Ny)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=Nx*Ny) return;
    int x=idx%Nx, y=idx/Nx;
    if (x==0||x==Nx-1||y==0||y==Ny-1){dst[idx]=src[idx];return;}

    // 3x3 Gaussian kernel weights
    const float k[3][3] = {
        {0.0625f, 0.125f, 0.0625f},
        {0.125f,  0.25f,  0.125f },
        {0.0625f, 0.125f, 0.0625f}
    };
    float val=0.f;
    for (int dy=-1;dy<=1;dy++)
        for (int dx=-1;dx<=1;dx++)
            val += k[dy+1][dx+1] * src[(y+dy)*Nx+(x+dx)];
    dst[idx]=val;
}
'''

curl_src = r'''
extern "C" __global__
void curl_kernel(
    const float* __restrict__ f,
    float* __restrict__ curl,
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
'''

print("Compiling CUDA kernels...")
lbm_step      = cp.RawKernel(kernel_src, "lbm_step")
filter_kernel = cp.RawKernel(filter_src, "filter_kernel")
gauss_blur    = cp.RawKernel(gauss_src,  "gauss_blur")
curl_kernel   = cp.RawKernel(curl_src,   "curl_kernel")
print("Ready.")

# ================================================================== #
#  host setup
# ================================================================== #
N = Nx * Ny
cxs = np.array([0,0,1,1,1,0,-1,-1,-1], dtype=np.float32)
w   = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36], dtype=np.float32)

yy, xx   = np.mgrid[0:Ny, 0:Nx]
cyl_r    = int(22 * Ny / 180)
cx0, cy0 = Nx//5, Ny//2
dist2    = (xx - cx0)**2 + (yy - cy0)**2
cylinder = dist2 < cyl_r**2
solid_gpu = cp.array(cylinder.ravel().astype(np.uint8))

# parabolic init
f0 = np.zeros((N, 9), dtype=np.float32)
for j in range(Ny):
    yf    = j / (Ny-1)
    u_loc = u0_max * 4.0 * yf * (1.0 - yf)
    for i in range(9):
        cu = 3.0 * cxs[i] * u_loc
        f0[j*Nx:(j+1)*Nx, i] = w[i]*(1.0+cu+0.5*cu**2-1.5*u_loc**2)
f0 += (0.001 * np.random.randn(N, 9)).astype(np.float32)

f_in      = cp.array(f0.ravel())
f_out     = cp.empty_like(f_in)
curl_buf  = cp.empty(N, dtype=cp.float32)
curl_blur = cp.empty(N, dtype=cp.float32)

tau_f   = np.float32(tau)
u0_f    = np.float32(u0_max)
threads = 256
blocks  = (N + threads - 1) // threads

# ================================================================== #
#  antialiased cylinder mask for display
# ================================================================== #
# smooth alpha falloff over 1.5px at the edge — clean circle, no jaggies
dist_f   = np.sqrt(dist2).astype(np.float32)
alpha    = np.clip((cyl_r - dist_f + 1.5) / 1.5, 0.0, 1.0)
cyl_rgba = np.zeros((Ny, Nx, 4), dtype=np.float32)
cyl_rgba[:, :, 0] = 0.12 * alpha
cyl_rgba[:, :, 1] = 0.12 * alpha
cyl_rgba[:, :, 2] = 0.15 * alpha
cyl_rgba[:, :, 3] = alpha

# ================================================================== #
#  matplotlib
# ================================================================== #
matplotlib.rcParams["toolbar"] = "None"
pyplot.ion()
fig, ax = pyplot.subplots(figsize=(16, 5), dpi=110)
fig.patch.set_facecolor("#050008")
ax.set_facecolor("#050008")

im = ax.imshow(
    np.zeros((Ny, Nx), dtype=np.float32),
    cmap="magma",
    vmin=0, vmax=VMAX,
    interpolation="bicubic",
    aspect="auto",
)
ax.imshow(cyl_rgba, aspect="auto", interpolation="bilinear")
ax.axis("off")
title = ax.set_title(
    f"Karman Vortex Street  |  u0={u0_max}  tau={tau}  Re~{int(u0_max*cyl_r*2/((2*tau-1)/6))}",
    fontsize=10, color="white"
)
fig.tight_layout(pad=0.2)
pyplot.show(block=False)
pyplot.pause(0.3)

# ================================================================== #
#  main loop
# ================================================================== #
it       = 0
step_num = 0
while it < Nt:
    for _ in range(STEPS_PER_FRAME):
        lbm_step(
            (blocks,), (threads,),
            (f_in, f_out, solid_gpu,
             np.int32(Nx), np.int32(Ny), tau_f, u0_f)
        )
        f_in, f_out = f_out, f_in
        step_num += 1
        if step_num % FILTER_EVERY == 0:
            filter_kernel(
                (blocks,), (threads,),
                (f_in, solid_gpu, np.int32(Nx), np.int32(Ny))
            )
    it += STEPS_PER_FRAME

    curl_kernel((blocks,), (threads,),
                (f_in, curl_buf, np.int32(Nx), np.int32(Ny)))

    # GPU Gaussian blur on curl — kills stripe artefacts in display only
    gauss_blur((blocks,), (threads,),
               (curl_buf, curl_blur, np.int32(Nx), np.int32(Ny)))
    gauss_blur((blocks,), (threads,),
               (curl_blur, curl_buf, np.int32(Nx), np.int32(Ny)))

    curl_2d = curl_buf.reshape(Ny, Nx).get()

    if not np.isfinite(curl_2d).all() or np.abs(curl_2d).max() > 10.0:
        print(f"Diverged at iter {it}.")
        break

    im.set_data(np.abs(curl_2d))
    title.set_text(
        f"Karman Vortex Street  |  u0={u0_max}  tau={tau}  iter={it}"
    )
    pyplot.pause(0.001)

pyplot.ioff()
pyplot.show(block=True)