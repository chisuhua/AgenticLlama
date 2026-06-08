// =============================================================================
// CUTLASS-backed FP32 GEMM (with naive CUDA fallback) for ggml MUL_MAT.
//
// Layout (matches `cutlass_gemm_f32_sm80` in cutlass_kernels.h):
//   A : N x K row-major (lda >= K)   -- ggml src0 weights
//   B : M x K row-major (ldb >= K)   -- ggml src1 activations
//   C : M x N row-major (ldc >= N)   -- ggml dst output
//
//   C[m, n] = sum_k A[n, k] * B[m, k]      (i.e. C = B * A^T)
//
// All operands are float (FP32). Accumulation happens in FP32.
// =============================================================================

#include <cuda_runtime.h>

#include <cstddef>

#include "kernels/include/cutlass_kernels.h"

#ifdef GGML_CUTLASS_HAS_HEADERS
// Placeholder for CUTLASS instantiation (CollectiveBuilder over float).
// Filled in by the kernel-generation step.
#endif

namespace {

constexpr int kTileM = 16;
constexpr int kTileN = 16;

__global__ void naive_gemm_f32_kernel(
        const float * __restrict__ A,
        const float * __restrict__ B,
        float       * __restrict__ C,
        int M, int N, int K,
        int lda, int ldb, int ldc) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) {
        return;
    }

    const float * a_row = A + (size_t) n * lda;
    const float * b_row = B + (size_t) m * ldb;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += a_row[k] * b_row[k];
    }

    C[(size_t) m * ldc + n] = acc;
}

} // namespace

extern "C" cudaError_t cutlass_gemm_f32_sm80(
        const void * A,
        const void * B,
        void *       C,
        int M, int N, int K,
        int lda, int ldb, int ldc,
        void * workspace, size_t workspace_size,
        cudaStream_t stream) {

    if (M <= 0 || N <= 0 || K <= 0) {
        return cudaSuccess;
    }
    if (A == nullptr || B == nullptr || C == nullptr) {
        return cudaErrorInvalidValue;
    }

    (void) workspace;
    (void) workspace_size;

#ifdef GGML_CUTLASS_HAS_HEADERS
    // CUTLASS path placeholder; falls through to naive kernel until codegen
    // is wired up.
#endif

    dim3 block(kTileN, kTileM, 1);
    dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM, 1);

    naive_gemm_f32_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float *>(A),
        reinterpret_cast<const float *>(B),
        reinterpret_cast<float *>(C),
        M, N, K, lda, ldb, ldc);

    return cudaGetLastError();
}

extern "C" size_t cutlass_gemm_f32_sm80_workspace_size(int M, int N, int K) {
    (void) M; (void) N; (void) K;
    return 0;
}
