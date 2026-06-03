// Auto-generated mock for ggml-tilelang. See add_fp16.cu for context.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define TILELANG_MUL_FP16_BLOCK 1024

namespace {

__global__ void tilelang_mul_fp16_kernel(const __half * __restrict__ A,
                                         const __half * __restrict__ B,
                                         __half * __restrict__ C,
                                         int32_t N) {
    const int32_t idx = blockIdx.x * TILELANG_MUL_FP16_BLOCK + threadIdx.x;
    if (idx < N) {
        C[idx] = __hmul(A[idx], B[idx]);
    }
}

}  // namespace

extern "C" void tilelang_mul_fp16(void * A, void * B, void * C, int32_t N, cudaStream_t stream) {
    if (N <= 0) {
        return;
    }
    const int32_t grid = (N + TILELANG_MUL_FP16_BLOCK - 1) / TILELANG_MUL_FP16_BLOCK;
    tilelang_mul_fp16_kernel<<<grid, TILELANG_MUL_FP16_BLOCK, 0, stream>>>(
        reinterpret_cast<const __half *>(A),
        reinterpret_cast<const __half *>(B),
        reinterpret_cast<__half *>(C),
        N);
}
