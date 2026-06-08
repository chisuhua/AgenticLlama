#pragma once

// TileLang provider for the Triton multi-kernel backend.
//
// Wraps the existing AOT-compiled TileLang kernels
// (currently elementwise add/mul, fp16/fp32) into the provider registry.
// Activated by the CMake option GGML_TRITON_WITH_TILELANG.

#include "ggml.h"
#include "ggml-triton-provider.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of the launcher C ABI provided by TileLang AOT
// (mirrors ggml-tilelang/kernels/include/tilelang_kernels.h).
void tilelang_add_fp16(void * A, void * B, void * C, int32_t N, cudaStream_t stream);
void tilelang_add_fp32(void * A, void * B, void * C, int32_t N, cudaStream_t stream);
void tilelang_mul_fp16(void * A, void * B, void * C, int32_t N, cudaStream_t stream);
void tilelang_mul_fp32(void * A, void * B, void * C, int32_t N, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

// Register all built-in TileLang providers into the global registry.
// Called from ggml-triton-provider.cpp:ggml_triton_global_registry() when
// GGML_TRITON_WITH_TILELANG is defined.
void ggml_triton_register_tilelang_providers(ggml_triton_op_registry & registry);
