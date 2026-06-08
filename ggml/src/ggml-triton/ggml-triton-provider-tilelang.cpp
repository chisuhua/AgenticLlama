#include "ggml-triton-provider-tilelang.h"
#include "ggml-triton-context.h"
#include "ggml-impl.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

// The ggml-triton context uses CUstream (Driver API).
// The TileLang launchers expect cudaStream_t (Runtime API).
// They are the same underlying pointer type -- direct cast is safe.
static cudaStream_t to_cuda_stream(CUstream cu_stream) {
    return reinterpret_cast<cudaStream_t>(cu_stream);
}

// Element-wise requires src0/src1/dst to share shape and be contiguous.
static bool elementwise_shapes_compatible(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];
    if (a == nullptr || b == nullptr) return false;
    if (!ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, op)) return false;
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(op)) return false;
    return true;
}

// ----------------------------------------------------------------------------
// ADD FP16
// ----------------------------------------------------------------------------

static bool tilelang_add_f16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ADD) return false;
    if (op->type != GGML_TYPE_F16) return false;
    if (op->src[0]->type != GGML_TYPE_F16 || op->src[1]->type != GGML_TYPE_F16) return false;
    return elementwise_shapes_compatible(op);
}

static bool tilelang_add_f16_execute(struct ggml_backend_triton_context * ctx,
                                     const struct ggml_tensor * node) {
    const int64_t n64 = ggml_nelements(node);
    if (n64 > INT32_MAX) return false;
    tilelang_add_fp16(node->src[0]->data, node->src[1]->data, node->data,
                                     (int32_t) n64, to_cuda_stream(ctx->cu_stream));
    return true;
}

// ----------------------------------------------------------------------------
// ADD FP32
// ----------------------------------------------------------------------------

static bool tilelang_add_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ADD) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) return false;
    return elementwise_shapes_compatible(op);
}

static bool tilelang_add_f32_execute(struct ggml_backend_triton_context * ctx,
                                     const struct ggml_tensor * node) {
    const int64_t n64 = ggml_nelements(node);
    if (n64 > INT32_MAX) return false;
    tilelang_add_fp32(node->src[0]->data, node->src[1]->data, node->data,
                                     (int32_t) n64, to_cuda_stream(ctx->cu_stream));
    return true;
}

// ----------------------------------------------------------------------------
// MUL FP16
// ----------------------------------------------------------------------------

static bool tilelang_mul_f16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL) return false;
    if (op->type != GGML_TYPE_F16) return false;
    if (op->src[0]->type != GGML_TYPE_F16 || op->src[1]->type != GGML_TYPE_F16) return false;
    return elementwise_shapes_compatible(op);
}

static bool tilelang_mul_f16_execute(struct ggml_backend_triton_context * ctx,
                                     const struct ggml_tensor * node) {
    const int64_t n64 = ggml_nelements(node);
    if (n64 > INT32_MAX) return false;
    tilelang_mul_fp16(node->src[0]->data, node->src[1]->data, node->data,
                                     (int32_t) n64, to_cuda_stream(ctx->cu_stream));
    return true;
}

// ----------------------------------------------------------------------------
// MUL FP32
// ----------------------------------------------------------------------------

static bool tilelang_mul_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) return false;
    return elementwise_shapes_compatible(op);
}

static bool tilelang_mul_f32_execute(struct ggml_backend_triton_context * ctx,
                                     const struct ggml_tensor * node) {
    const int64_t n64 = ggml_nelements(node);
    if (n64 > INT32_MAX) return false;
    tilelang_mul_fp32(node->src[0]->data, node->src[1]->data, node->data,
                                     (int32_t) n64, to_cuda_stream(ctx->cu_stream));
    return true;
}

// ----------------------------------------------------------------------------
// Registration
// ----------------------------------------------------------------------------

void ggml_triton_register_tilelang_providers(ggml_triton_op_registry & registry) {
    // ADD FP16
    registry.register_impl(GGML_OP_ADD, {
        /* .name = */ "tilelang_add_f16",
        /* .provider = */ GGML_TRITON_PROVIDER_TILELANG,
        /* .supports = */ tilelang_add_f16_supports,
        /* .execute = */ tilelang_add_f16_execute,
        /* .priority = */ 100,
    });

    // ADD FP32
    registry.register_impl(GGML_OP_ADD, {
        /* .name = */ "tilelang_add_f32",
        /* .provider = */ GGML_TRITON_PROVIDER_TILELANG,
        /* .supports = */ tilelang_add_f32_supports,
        /* .execute = */ tilelang_add_f32_execute,
        /* .priority = */ 100,
    });

    // MUL FP16
    registry.register_impl(GGML_OP_MUL, {
        /* .name = */ "tilelang_mul_f16",
        /* .provider = */ GGML_TRITON_PROVIDER_TILELANG,
        /* .supports = */ tilelang_mul_f16_supports,
        /* .execute = */ tilelang_mul_f16_execute,
        /* .priority = */ 100,
    });

    // MUL FP32
    registry.register_impl(GGML_OP_MUL, {
        /* .name = */ "tilelang_mul_f32",
        /* .provider = */ GGML_TRITON_PROVIDER_TILELANG,
        /* .supports = */ tilelang_mul_f32_supports,
        /* .execute = */ tilelang_mul_f32_execute,
        /* .priority = */ 100,
    });
}
