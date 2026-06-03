#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// GEMM kernels matching ggml MUL_MAT semantics.
//
// In ggml, MUL_MAT computes:
//   dst[m, n] = sum_k(src0[n, k] * src1[m, k])
// where:
//   src0 is the weight matrix, ne=[K, N], stored row-major as N rows of
//        K contiguous elements (so A[n, k] = A->data[n * K + k]).
//   src1 is the activation/input,    ne=[K, M], stored row-major as M rows of
//        K contiguous elements (so B[m, k] = B->data[m * K + k]).
//   dst  has ne=[N, M], laid out as M rows of N contiguous elements
//        (so C[m, n] = C->data[m * N + n]).
//
// Equivalently, this is the standard GEMM `C = B * A^T` with row-major A, B, C
// of dimensions:
//   A : N x K (lda >= K)
//   B : M x K (ldb >= K)
//   C : M x N (ldc >= N)
//
// All GEMM entry points below follow this convention. Batched calls (ne[2],
// ne[3] > 1) should be performed by the caller by looping over batches and
// invoking the 2-D kernel once per batch with appropriately offset pointers.
// =============================================================================

// -----------------------------------------------------------------------------
// FP16 GEMM (compute capability 8.0+ / Ampere baseline)
// -----------------------------------------------------------------------------
//
// `workspace` must point to at least `workspace_size` bytes of device memory;
// it may be NULL/0 if the kernel reports it does not need scratch.
//
// `stream` is the CUDA stream to launch on; NULL means cudaStreamPerThread.
//
// When CUTLASS headers are available (GGML_CUTLASS_HAS_HEADERS) the kernel
// dispatches to a CollectiveBuilder-generated kernel; otherwise a naive CUDA
// fallback is used.
cudaError_t cutlass_gemm_f16_sm80(
    const void * A,
    const void * B,
    void *       C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    void * workspace, size_t workspace_size,
    cudaStream_t stream);

// Returns the workspace size (bytes) required by `cutlass_gemm_f16_sm80` for a
// problem of shape (M, N, K). Implementations may always return 0.
size_t cutlass_gemm_f16_sm80_workspace_size(int M, int N, int K);

// -----------------------------------------------------------------------------
// FP32 GEMM
// -----------------------------------------------------------------------------
//
// Same convention as the FP16 variant. All operands are float32. Internal
// accumulation also happens in FP32.
cudaError_t cutlass_gemm_f32_sm80(
    const void * A,
    const void * B,
    void *       C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    void * workspace, size_t workspace_size,
    cudaStream_t stream);

size_t cutlass_gemm_f32_sm80_workspace_size(int M, int N, int K);

// -----------------------------------------------------------------------------
// Q4_0 x FP16 fused dequantize + GEMM
// -----------------------------------------------------------------------------
//
//   A_quant : Q4_0-packed weights, logically shaped [N, K] (N rows of K
//             quantised values), physically laid out as N * (K/QK4_0) blocks
//             of 18 bytes (block_q4_0). K must be a multiple of QK4_0 (=32).
//   B_fp16  : FP16 activations, M x K row-major (ldb implicitly = K).
//   C_fp16  : FP16 output,      M x N row-major (ldc implicitly = N).
//
// Computes  C[m, n] = sum_k dequant(A_quant)[n, k] * B_fp16[m, k]
// using a fused dequantisation prologue (no intermediate FP16 weight buffer).
cudaError_t cutlass_gemm_q4_0_f16_sm80(
    const void * A_quant,
    const void * B_fp16,
    void *       C_fp16,
    int M, int N, int K,
    void * workspace, size_t workspace_size,
    cudaStream_t stream);

size_t cutlass_gemm_q4_0_f16_sm80_workspace_size(int M, int N, int K);

// -----------------------------------------------------------------------------
// Q8_0 x FP16 fused dequantize + GEMM
// -----------------------------------------------------------------------------
//
//   A_quant : Q8_0-packed weights, logically shaped [N, K] (N rows of K
//             quantised values), physically laid out as N * (K/QK8_0) blocks
//             of 34 bytes (block_q8_0). K must be a multiple of QK8_0 (=32).
//   B_fp16  : FP16 activations, M x K row-major (ldb implicitly = K).
//   C_fp16  : FP16 output,      M x N row-major (ldc implicitly = N).
//
// Computes  C[m, n] = sum_k dequant(A_quant)[n, k] * B_fp16[m, k]
cudaError_t cutlass_gemm_q8_0_f16_sm80(
    const void * A_quant,
    const void * B_fp16,
    void *       C_fp16,
    int M, int N, int K,
    void * workspace, size_t workspace_size,
    cudaStream_t stream);

size_t cutlass_gemm_q8_0_f16_sm80_workspace_size(int M, int N, int K);

// -----------------------------------------------------------------------------
// Element-wise activations
// -----------------------------------------------------------------------------

// y = GELU(x), N elements, FP16 input/output.
cudaError_t cutlass_gelu_fp16(
    const void * input, void * output, int N, cudaStream_t stream);

// y = GELU(x), N elements, FP32 input/output.
cudaError_t cutlass_gelu_fp32(
    const void * input, void * output, int N, cudaStream_t stream);

// y = SILU(x), N elements, FP16 input/output.
cudaError_t cutlass_silu_fp16(
    const void * input, void * output, int N, cudaStream_t stream);

// y = SILU(x), N elements, FP32 input/output.
cudaError_t cutlass_silu_fp32(
    const void * input, void * output, int N, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
