// Auto-generated mock for ggml-tilelang. The real .cu file is produced by
// `scripts/compile_kernels.py` from the TileLang prim_func in
// `tilelang_kernels/elementwise.py`. The body below mirrors the kernel that
// TileLang would emit and is used as a build-able stand-in.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define TILELANG_ADD_FP16_BLOCK 1024

namespace {

__global__ void tilelang_add_fp16_kernel(const __half * __restrict__ A,
                                         const __half * __restrict__ B,
                                         __half * __restrict__ C,
                                         int32_t N) {
    const int32_t idx = blockIdx.x * TILELANG_ADD_FP16_BLOCK + threadIdx.x;
    if (idx < N) {
        C[idx] = __hadd(A[idx], B[idx]);
    }
}

}  // namespace

extern "C" void tilelang_add_fp16(void * A, void * B, void * C, int32_t N, cudaStream_t stream) {
    if (N <= 0) {
        return;
    }
    const int32_t grid = (N + TILELANG_ADD_FP16_BLOCK - 1) / TILELANG_ADD_FP16_BLOCK;
    tilelang_add_fp16_kernel<<<grid, TILELANG_ADD_FP16_BLOCK, 0, stream>>>(
        reinterpret_cast<const __half *>(A),
        reinterpret_cast<const __half *>(B),
        reinterpret_cast<__half *>(C),
        N);
}
