// =============================================================================
// Fused dequantize + GEMM for Q4_0 weights x FP16 activations.
//
// Block layout (matches ggml-common.h block_q4_0):
//   struct block_q4_0 {
//       ggml_half d;            // 2-byte FP16 scale
//       uint8_t   qs[16];       // 32 nibbles packed two-per-byte
//   };                          // 18 bytes per 32 elements
//
// Dequantisation rule:
//   value_lo[i] = (qs[i]       & 0x0F) - 8;  // even index 2*i
//   value_hi[i] = (qs[i] >>  4) & 0x0F - 8;  // odd  index 2*i+1
//   dequant     = value * (float) d
//
// Layout (matches cutlass_kernels.h):
//   A_quant : N rows of (K/QK4_0) blocks, each 18 bytes.
//             Row n starts at byte offset  n * (K/QK4_0) * 18.
//   B_fp16  : M x K row-major.
//   C_fp16  : M x N row-major.
//   K must be a multiple of QK4_0 (=32).
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

constexpr int kQK4_0    = ggml_cutlass::GgmlQ4_0Traits::kElementsPerBlock;       // 32
constexpr int kBytesQ4_0 = ggml_cutlass::GgmlQ4_0Traits::kBytesPerBlock;          // 18

constexpr int kTileM = 16;
constexpr int kTileN = 16;

// One thread per output element (m, n). For each of the K/32 weight blocks
// belonging to row n we load the FP16 scale, then expand the 32 nibbles into
// FP32 contributions against the matching 32 elements of B[m, :].
__global__ void naive_gemm_q4_0_f16_kernel(
        const uint8_t * __restrict__ A_quant,
        const __half  * __restrict__ B_fp16,
        __half        * __restrict__ C_fp16,
        int M, int N, int K) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) {
        return;
    }

    const int n_blocks = K / kQK4_0;

    // Pointer to the first block of weight row n.
    const uint8_t * row_base =
        A_quant + (size_t) n * n_blocks * kBytesQ4_0;

    const __half * b_row = B_fp16 + (size_t) m * K;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        const uint8_t * blk = row_base + (size_t) b * kBytesQ4_0;

        // 2-byte little-endian FP16 scale at the head of the block.
        __half d_h;
        // Use memcpy to avoid alignment issues; nvcc lowers it to ld.u16.
        uint16_t d_bits =
            (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
        d_h = __ushort_as_half(d_bits);
        const float d = __half2float(d_h);

        const uint8_t * qs = blk + 2;

        const __half * b_blk = b_row + (size_t) b * kQK4_0;

        // 16 packed bytes, each containing 2 nibbles. The ggml convention
        // places element 2*i in the low nibble and element 2*i+1 in the high
        // nibble, so we expand them in lockstep.
        #pragma unroll
        for (int i = 0; i < kQK4_0 / 2; ++i) {
            const uint8_t byte = qs[i];
            const int q_lo = (int) (byte & 0x0F) - 8;
            const int q_hi = (int) (byte >>   4) - 8;

            const float w_lo = (float) q_lo * d;
            const float w_hi = (float) q_hi * d;

            const float b_lo = __half2float(b_blk[2 * i + 0]);
            const float b_hi = __half2float(b_blk[2 * i + 1]);

            acc += w_lo * b_lo;
            acc += w_hi * b_hi;
        }
    }

    C_fp16[(size_t) m * N + n] = __float2half(acc);
}

} // namespace

extern "C" cudaError_t cutlass_gemm_q4_0_f16_sm80(
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
    if (K % kQK4_0 != 0) {
        // K must be a multiple of the Q4_0 block size.
        return cudaErrorInvalidValue;
    }

    (void) workspace;
    (void) workspace_size;

#ifdef GGML_CUTLASS_HAS_HEADERS
    // Placeholder for a CUTLASS dequant-prologue + GEMM mainloop. The naive
    // kernel below is correct but unoptimised and serves as the canonical
    // reference for the CUTLASS instantiation step.
#endif

    dim3 block(kTileN, kTileM, 1);
    dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM, 1);

    naive_gemm_q4_0_f16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(A_quant),
        reinterpret_cast<const __half  *>(B_fp16),
        reinterpret_cast<__half        *>(C_fp16),
        M, N, K);

    return cudaGetLastError();
}

extern "C" size_t cutlass_gemm_q4_0_f16_sm80_workspace_size(int M, int N, int K) {
    (void) M; (void) N; (void) K;
    return 0;
}
