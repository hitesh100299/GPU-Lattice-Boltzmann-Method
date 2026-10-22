// stream_copy.cu
// Measured peak device memory bandwidth: a plain copy kernel over 512 MiB
// buffers, best of 20 repetitions, GB/s = (bytes read + bytes written) / time.
// This measured value, not the datasheet, is the peak that the LBM bandwidth
// numbers are compared against.
//   nvcc -O3 -arch=native stream_copy.cu -o stream_copy
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

__global__ void copy_kernel(const float4* __restrict__ a, float4* __restrict__ b, size_t n)
{
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

int main()
{
    const size_t bytes = 512ull << 20;
    const size_t n = bytes / sizeof(float4);
    float4 *a, *b;
    CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes));
    CK(cudaMemset(a, 1, bytes));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int threads = 256; const int blocks = (int)((n + threads - 1) / threads);
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int mclk = 0, bus = 0;
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0); cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    std::printf("GPU: %s, memory clock %d kHz, bus width %d bit, buffers 2 x %zu MiB\n", prop.name, mclk, bus, bytes >> 20);
    float best = 1e30f, worst = 0.f, sum = 0.f;
    for (int r = 0; r < 23; r++) {                  // 3 warm-up, 20 timed
        CK(cudaEventRecord(e0));
        copy_kernel<<<blocks, threads>>>(a, b, n);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (r >= 3) { best = ms < best ? ms : best; worst = ms > worst ? ms : worst; sum += ms; }
    }
    const double gbs_best = 2.0 * bytes / (best * 1e-3) / 1e9;
    std::printf("copy kernel: best %.3f ms, mean %.3f ms, worst %.3f ms over 20 runs\n", best, sum / 20, worst);
    std::printf("bandwidth (read + write): best %.1f GB/s, mean %.1f GB/s\n", gbs_best, 2.0 * bytes / (sum / 20 * 1e-3) / 1e9);
    // cudaMemcpy device-to-device for comparison
    CK(cudaEventRecord(e0)); CK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice)); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    std::printf("cudaMemcpy D2D: %.3f ms, %.1f GB/s\n", ms, 2.0 * bytes / (ms * 1e-3) / 1e9);
    std::printf("RESULT_GBS %.1f\n", gbs_best);
    return 0;
}
