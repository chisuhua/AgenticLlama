#pragma once

// Public C ABI exposed by the TileLang AOT-generated kernels.
//
// All launchers accept device pointers (allocated via cudaMalloc), the number
// of elements, and a CUDA stream. They return after enqueueing the kernel,
// callers are responsible for synchronization.

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// element-wise C = A + B
void tilelang_add_fp16(void * A, void * B, void * C, int32_t N, cudaStream_t stream);
void tilelang_add_fp32(void * A, void * B, void * C, int32_t N, cudaStream_t stream);

// element-wise C = A * B
void tilelang_mul_fp16(void * A, void * B, void * C, int32_t N, cudaStream_t stream);
void tilelang_mul_fp32(void * A, void * B, void * C, int32_t N, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
