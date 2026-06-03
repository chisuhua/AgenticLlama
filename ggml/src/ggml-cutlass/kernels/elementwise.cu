// =============================================================================
// Element-wise activations (GELU, SILU) for the CUTLASS backend.
//
// These kernels are intentionally simple: they read a contiguous tensor and
// write the activated result element-wise. Heavy fusion (e.g. with a preceding
// GEMM epilogue) is the job of the CUTLASS GEMM TU; this file only handles
// the standalone op case used when GELU/SILU appear as their own graph nodes.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernels/include/cutlass_kernels.h"

namespace {

// Constants for the tanh-based GELU approximation, matching the variant used
// throughout ggml (see ggml.c / ggml-cpu).
__device__ __forceinline__ float gelu_tanh_approx_f32(float x) {
    // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff       = 0.044715f;
    const float x3   = x * x * x;
    const float arg  = kSqrt2OverPi * (x + kCoeff * x3);
    return 0.5f * x * (1.0f + tanhf(arg));
}

__device__ __forceinline__ float silu_f32(float x) {
    // x / (1 + exp(-x))
    return x / (1.0f + expf(-x));
}

template <typename Op>
__global__ void elementwise_fp16_kernel(const __half * __restrict__ in,
                                        __half * __restrict__ out,
                                        int N, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float x = __half2float(in[idx]);
    out[idx] = __float2half(op(x));
}

template <typename Op>
__global__ void elementwise_fp32_kernel(const float * __restrict__ in,
                                        float * __restrict__ out,
                                        int N, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    out[idx] = op(in[idx]);
}

struct GeluFn { __device__ float operator()(float x) const { return gelu_tanh_approx_f32(x); } };
struct SiluFn { __device__ float operator()(float x) const { return silu_f32(x); } };

constexpr int kBlockSize = 256;

inline dim3 launch_grid(int N) {
    return dim3((N + kBlockSize - 1) / kBlockSize, 1, 1);
}

template <typename Op>
cudaError_t launch_fp16(const void * input, void * output, int N, cudaStream_t stream, Op op) {
    if (N <= 0) return cudaSuccess;
    if (input == nullptr || output == nullptr) return cudaErrorInvalidValue;

    elementwise_fp16_kernel<<<launch_grid(N), kBlockSize, 0, stream>>>(
        reinterpret_cast<const __half *>(input),
        reinterpret_cast<__half *>(output),
        N, op);
    return cudaGetLastError();
}

template <typename Op>
cudaError_t launch_fp32(const void * input, void * output, int N, cudaStream_t stream, Op op) {
    if (N <= 0) return cudaSuccess;
    if (input == nullptr || output == nullptr) return cudaErrorInvalidValue;

    elementwise_fp32_kernel<<<launch_grid(N), kBlockSize, 0, stream>>>(
        reinterpret_cast<const float *>(input),
        reinterpret_cast<float *>(output),
        N, op);
    return cudaGetLastError();
}

} // namespace

extern "C" cudaError_t cutlass_gelu_fp16(const void * input, void * output, int N, cudaStream_t stream) {
    return launch_fp16(input, output, N, stream, GeluFn{});
}

extern "C" cudaError_t cutlass_gelu_fp32(const void * input, void * output, int N, cudaStream_t stream) {
    return launch_fp32(input, output, N, stream, GeluFn{});
}

extern "C" cudaError_t cutlass_silu_fp16(const void * input, void * output, int N, cudaStream_t stream) {
    return launch_fp16(input, output, N, stream, SiluFn{});
}

extern "C" cudaError_t cutlass_silu_fp32(const void * input, void * output, int N, cudaStream_t stream) {
    return launch_fp32(input, output, N, stream, SiluFn{});
}
