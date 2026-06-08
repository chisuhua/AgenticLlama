// =============================================================================
// CUTLASS-backed FP16 GEMM (with naive CUDA fallback) for ggml MUL_MAT.
//
// Layout (matches `cutlass_gemm_f16_sm80` in cutlass_kernels.h):
//   A : N x K row-major (lda >= K)   -- ggml src0 weights
//   B : M x K row-major (ldb >= K)   -- ggml src1 activations
//   C : M x N row-major (ldc >= N)   -- ggml dst output
//
//   C[m, n] = sum_k A[n, k] * B[m, k]      (i.e. C = B * A^T)
//
// All operands are __half (FP16). Accumulation happens in FP32 internally.
// Batched calls should be performed by the caller with offset pointers.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>

#include "kernels/include/cutlass_kernels.h"

#ifdef GGML_CUTLASS_HAS_HEADERS
// When CUTLASS headers are present, prefer the CollectiveBuilder path.
//   #include <cutlass/cutlass.h>
//   #include <cutlass/gemm/device/gemm_universal_adapter.h>
//   #include <cutlass/gemm/collective/collective_builder.hpp>
//   #include <cutlass/epilogue/collective/collective_builder.hpp>
//
// The actual CUTLASS instantiation is intentionally left out of this skeleton
// to keep build cost low; it should be filled in by the kernel-generation step
// (see scripts/generate_gemm_instances.py).
#endif

namespace {

constexpr int kTileM = 16;
constexpr int kTileN = 16;

// One thread per output element (m, n).
//   C[m, n] = sum_k A[n*lda + k] * B[m*ldb + k]
__global__ void naive_gemm_f16_kernel(
        const __half * __restrict__ A,
        const __half * __restrict__ B,
        __half       * __restrict__ C,
        int M, int N, int K,
        int lda, int ldb, int ldc) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) {
        return;
    }

    const __half * a_row = A + (size_t) n * lda;   // A[n, :]
    const __half * b_row = B + (size_t) m * ldb;   // B[m, :]

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float a = __half2float(a_row[k]);
        const float b = __half2float(b_row[k]);
        acc += a * b;
    }

    C[(size_t) m * ldc + n] = __float2half(acc);
}

} // namespace

extern "C" cudaError_t cutlass_gemm_f16_sm80(
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
    // ------------------------------------------------------------------------
    // CUTLASS path (placeholder).
    //
    //   using Gemm = cutlass::gemm::device::GemmUniversalAdapter<...>;
    //   Gemm::Arguments args{ ... };       // RowMajor A (NxK), RowMajor B
    //                                      // (MxK) but logically transposed
    //                                      // for B*A^T
    //   Gemm gemm_op;
    //   auto status = gemm_op.can_implement(args);
    //   if (status != cutlass::Status::kSuccess) goto fallback;
    //   status = gemm_op.initialize(args, workspace, stream);
    //   if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    //   status = gemm_op.run(stream);
    //   return (status == cutlass::Status::kSuccess) ? cudaSuccess
    //                                                : cudaErrorUnknown;
    //
    // Until codegen is wired up, fall through to the naive kernel.
#endif

    dim3 block(kTileN, kTileM, 1);
    dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM, 1);

    naive_gemm_f16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half *>(A),
        reinterpret_cast<const __half *>(B),
        reinterpret_cast<__half *>(C),
        M, N, K, lda, ldb, ldc);

    return cudaGetLastError();
}

extern "C" size_t cutlass_gemm_f16_sm80_workspace_size(int M, int N, int K) {
    (void) M; (void) N; (void) K;
    return 0;
}
