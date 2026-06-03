#include "ggml-triton-provider.h"
#include "ggml-triton-context.h"

// Generated AOT launcher prototypes
#include "triton_kernels.h"

#include "ggml-impl.h"

#include <cuda.h>

#include <cstdint>

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

static enum ggml_unary_op triton_provider_get_unary_op(const struct ggml_tensor * t) {
    return (enum ggml_unary_op) ggml_get_op_params_i32(t, 0);
}

// ----------------------------------------------------------------------------
// GELU FP16 provider
// ----------------------------------------------------------------------------

static bool triton_gelu_fp16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY) {
        return false;
    }
    if (triton_provider_get_unary_op(op) != GGML_UNARY_OP_GELU) {
        return false;
    }
    if (op->type != GGML_TYPE_F16) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F16) {
        return false;
    }
    return true;
}

static bool triton_gelu_fp16_execute(struct ggml_backend_triton_context * ctx,
                                     const struct ggml_tensor * node) {
    const int64_t n_elements = ggml_nelements(node);
    const void *  src_data   = node->src[0]->data;
    void *        dst_data   = node->data;

    if (src_data == nullptr || dst_data == nullptr) {
        return false;
    }

    const CUdeviceptr d_in  = (CUdeviceptr) src_data;
    const CUdeviceptr d_out = (CUdeviceptr) dst_data;

    int rc = triton_launch_gelu_fp16_sm80(ctx->cu_stream, d_in, d_out, (int32_t) n_elements);
    return rc == 0;
}

// ----------------------------------------------------------------------------
// GELU FP32 provider
// ----------------------------------------------------------------------------

static bool triton_gelu_fp32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY) {
        return false;
    }
    if (triton_provider_get_unary_op(op) != GGML_UNARY_OP_GELU) {
        return false;
    }
    if (op->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    return true;
}

static bool triton_gelu_fp32_execute(struct ggml_backend_triton_context * ctx,
                                     const struct ggml_tensor * node) {
    const int64_t n_elements = ggml_nelements(node);
    const void *  src_data   = node->src[0]->data;
    void *        dst_data   = node->data;

    if (src_data == nullptr || dst_data == nullptr) {
        return false;
    }

    const CUdeviceptr d_in  = (CUdeviceptr) src_data;
    const CUdeviceptr d_out = (CUdeviceptr) dst_data;

    int rc = triton_launch_gelu_fp32_sm80(ctx->cu_stream, d_in, d_out, (int32_t) n_elements);
    return rc == 0;
}

// ----------------------------------------------------------------------------
// SILU FP16 provider
// ----------------------------------------------------------------------------

static bool triton_silu_fp16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY) {
        return false;
    }
    if (triton_provider_get_unary_op(op) != GGML_UNARY_OP_SILU) {
        return false;
    }
    if (op->type != GGML_TYPE_F16) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F16) {
        return false;
    }
    return true;
}

static bool triton_silu_fp16_execute(struct ggml_backend_triton_context * ctx,
                                     const struct ggml_tensor * node) {
    const int64_t n_elements = ggml_nelements(node);
    const void *  src_data   = node->src[0]->data;
    void *        dst_data   = node->data;

    if (src_data == nullptr || dst_data == nullptr) {
        return false;
    }

    const CUdeviceptr d_in  = (CUdeviceptr) src_data;
    const CUdeviceptr d_out = (CUdeviceptr) dst_data;

    int rc = triton_launch_silu_fp16_sm80(ctx->cu_stream, d_in, d_out, (int32_t) n_elements);
    return rc == 0;
}

// ----------------------------------------------------------------------------
// SILU FP32 provider
// ----------------------------------------------------------------------------

static bool triton_silu_fp32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY) {
        return false;
    }
    if (triton_provider_get_unary_op(op) != GGML_UNARY_OP_SILU) {
        return false;
    }
    if (op->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    return true;
}

static bool triton_silu_fp32_execute(struct ggml_backend_triton_context * ctx,
                                     const struct ggml_tensor * node) {
    const int64_t n_elements = ggml_nelements(node);
    const void *  src_data   = node->src[0]->data;
    void *        dst_data   = node->data;

    if (src_data == nullptr || dst_data == nullptr) {
        return false;
    }

    const CUdeviceptr d_in  = (CUdeviceptr) src_data;
    const CUdeviceptr d_out = (CUdeviceptr) dst_data;

    int rc = triton_launch_silu_fp32_sm80(ctx->cu_stream, d_in, d_out, (int32_t) n_elements);
    return rc == 0;
}

// ----------------------------------------------------------------------------
// Registration
// ----------------------------------------------------------------------------

void ggml_triton_register_builtin_providers(ggml_triton_op_registry & registry) {
    // GELU FP16
    registry.register_impl(GGML_OP_UNARY, {
        /* .name     = */ "triton_gelu_fp16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_gelu_fp16_supports,
        /* .execute  = */ triton_gelu_fp16_execute,
        /* .priority = */ 100,
    });

    // GELU FP32
    registry.register_impl(GGML_OP_UNARY, {
        /* .name     = */ "triton_gelu_fp32_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_gelu_fp32_supports,
        /* .execute  = */ triton_gelu_fp32_execute,
        /* .priority = */ 100,
    });

    // SILU FP16
    registry.register_impl(GGML_OP_UNARY, {
        /* .name     = */ "triton_silu_fp16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_silu_fp16_supports,
        /* .execute  = */ triton_silu_fp16_execute,
        /* .priority = */ 100,
    });

    // SILU FP32
    registry.register_impl(GGML_OP_UNARY, {
        /* .name     = */ "triton_silu_fp32_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_silu_fp32_supports,
        /* .execute  = */ triton_silu_fp32_execute,
        /* .priority = */ 100,
    });
}
