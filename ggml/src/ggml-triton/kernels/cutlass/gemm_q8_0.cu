// =============================================================================
// Fused dequantize + GEMM for Q8_0 weights x FP16 activations.
//
// Block layout (matches ggml-common.h block_q8_0):
//   struct block_q8_0 {
//       ggml_half d;          // 2-byte FP16 scale
//       int8_t    qs[32];     // 32 signed 8-bit values
//   };                        // 34 bytes per 32 elements
//
// Dequantisation rule:
//   value[i] = (float) qs[i] * (float) d
//
// Layout (matches cutlass_kernels.h):
//   A_quant : N rows of (K/QK8_0) blocks, each 34 bytes.
//             Row n starts at byte offset  n * (K/QK8_0) * 34.
//   B_fp16  : M x K row-major.
//   C_fp16  : M x N row-major.
//   K must be a multiple of QK8_0 (=32).
//
//   C[m, n] = sum_k dequant(A_quant)[n, k] * B_fp16[m, k]
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include "kernels/include/cutlass_kernels.h"
#include "kernels/include/ggml_quant_traits.h"

namespace {

constexpr int kQK8_0    = ggml_cutlass::GgmlQ8_0Traits::kElementsPerBlock;       // 32
constexpr int kBytesQ8_0 = ggml_cutlass::GgmlQ8_0Traits::kBytesPerBlock;          // 34

constexpr int kTileM = 16;
constexpr int kTileN = 16;

__global__ void naive_gemm_q8_0_f16_kernel(
        const uint8_t * __restrict__ A_quant,
        const __half  * __restrict__ B_fp16,
        __half        * __restrict__ C_fp16,
        int M, int N, int K) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) {
        return;
    }

    const int n_blocks = K / kQK8_0;

    const uint8_t * row_base =
        A_quant + (size_t) n * n_blocks * kBytesQ8_0;

    const __half * b_row = B_fp16 + (size_t) m * K;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        const uint8_t * blk = row_base + (size_t) b * kBytesQ8_0;

        const uint16_t d_bits =
            (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
        const float d = __half2float(__ushort_as_half(d_bits));

        const int8_t * qs = reinterpret_cast<const int8_t *>(blk + 2);

        const __half * b_blk = b_row + (size_t) b * kQK8_0;

        #pragma unroll
        for (int i = 0; i < kQK8_0; ++i) {
            const float w = (float) qs[i] * d;
            const float x = __half2float(b_blk[i]);
            acc += w * x;
        }
    }

    C_fp16[(size_t) m * N + n] = __float2half(acc);
}

} // namespace

extern "C" cudaError_t cutlass_gemm_q8_0_f16_sm80(
        const void * A_quant,
        const void * B_fp16,
        void *       C_fp16,
        int M, int N, int K,
        void * workspace, size_t workspace_size,
        cudaStream_t stream) {

    if (M <= 0 || N <= 0 || K <= 0) {
        return cudaSuccess;
    }
    if (A_quant == nullptr || B_fp16 == nullptr || C_fp16 == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (K % kQK8_0 != 0) {
        return cudaErrorInvalidValue;
    }

    (void) workspace;
    (void) workspace_size;

#ifdef GGML_CUTLASS_HAS_HEADERS
    // Placeholder for a CUTLASS dequant-prologue + GEMM mainloop. The naive
    // kernel below is correct but unoptimised.
#endif

    dim3 block(kTileN, kTileM, 1);
    dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM, 1);

    naive_gemm_q8_0_f16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(A_quant),
        reinterpret_cast<const __half  *>(B_fp16),
        reinterpret_cast<__half        *>(C_fp16),
        M, N, K);

    return cudaGetLastError();
}

extern "C" size_t cutlass_gemm_q8_0_f16_sm80_workspace_size(int M, int N, int K) {
    (void) M; (void) N; (void) K;
    return 0;
}
