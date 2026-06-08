// Auto-generated mock for ggml-tilelang. See add_fp16.cu for context.

#include <cuda_runtime.h>
#include <stdint.h>

#define TILELANG_MUL_FP32_BLOCK 1024

namespace {

__global__ void tilelang_mul_fp32_kernel(const float * __restrict__ A,
                                         const float * __restrict__ B,
                                         float * __restrict__ C,
                                         int32_t N) {
    const int32_t idx = blockIdx.x * TILELANG_MUL_FP32_BLOCK + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

}  // namespace

extern "C" void tilelang_mul_fp32(void * A, void * B, void * C, int32_t N, cudaStream_t stream) {
    if (N <= 0) {
        return;
    }
    const int32_t grid = (N + TILELANG_MUL_FP32_BLOCK - 1) / TILELANG_MUL_FP32_BLOCK;
    tilelang_mul_fp32_kernel<<<grid, TILELANG_MUL_FP32_BLOCK, 0, stream>>>(
        reinterpret_cast<const float *>(A),
        reinterpret_cast<const float *>(B),
        reinterpret_cast<float *>(C),
        N);
}
