#include "ggml-triton-provider-cutlass.h"
#include "ggml-triton-context.h"

#include "cutlass_kernels.h"

#include "ggml-impl.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

// The ggml-triton context uses CUstream (Driver API).
// CUTLASS kernels expect cudaStream_t (Runtime API).
// They are the same underlying pointer type -- direct cast is safe.
static cudaStream_t to_cuda_stream(CUstream cu_stream) {
    return reinterpret_cast<cudaStream_t>(cu_stream);
}

static enum ggml_unary_op cutlass_provider_get_unary_op(const struct ggml_tensor * t) {
    return (enum ggml_unary_op) ggml_get_op_params_i32(t, 0);
}

// Check that a tensor is contiguous (row-major, no padding between rows).
static bool tensor_is_contiguous(const struct ggml_tensor * t) {
    if (t == nullptr) {
        return false;
    }
    return ggml_is_contiguous(t);
}

// ----------------------------------------------------------------------------
// MUL_MAT shared helpers
//
// ggml MUL_MAT semantics:
//   src0  ne=[K, N, b2, b3]   -- weight matrix
//   src1  ne=[K, M, b2, b3]   -- activations
//   dst   ne=[N, M, b2, b3]   -- output
//
//   dst[..., m, n] = sum_k(src0[..., n, k] * src1[..., m, k])
//
// CUTLASS kernels expect:
//   A : N x K row-major (lda = K)   <- src0
//   B : M x K row-major (ldb = K)   <- src1
//   C : M x N row-major (ldc = N)   <- dst
//
// Batches (ne[2]*ne[3] > 1) are looped at the provider level, with src0
// optionally broadcasting along ne[2]/ne[3] when its size is 1.
// ----------------------------------------------------------------------------

static bool mul_mat_shapes_compatible(const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (!tensor_is_contiguous(src0) || !tensor_is_contiguous(src1) || !tensor_is_contiguous(op)) {
        return false;
    }

    // ne[0] (K) must match between src0 and src1.
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }

    // Allow broadcasting of src0 along ne[2]/ne[3]: either matches src1, or 1.
    for (int d = 2; d < 4; ++d) {
        if (src0->ne[d] != src1->ne[d] && src0->ne[d] != 1) {
            return false;
        }
    }

    return true;
}

// ----------------------------------------------------------------------------
// MUL_MAT FP16 provider (priority = 120)
// ----------------------------------------------------------------------------

static bool cutlass_mul_mat_f16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F16) {
        return false;
    }

    return mul_mat_shapes_compatible(op);
}

static bool cutlass_mul_mat_f16_execute(struct ggml_backend_triton_context * ctx,
                                        const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        return false;
    }

    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int M = (int) src1->ne[1];

    const int lda = K;  // A (src0) is N x K row-major
    const int ldb = K;  // B (src1) is M x K row-major
    const int ldc = N;  // C (dst)  is M x N row-major

    cudaStream_t stream = to_cuda_stream(ctx->cu_stream);

    const int64_t src0_b2 = src0->ne[2];
    const int64_t src0_b3 = src0->ne[3];

    for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
            const int64_t a_i2 = (src0_b2 == 1) ? 0 : i2;
            const int64_t a_i3 = (src0_b3 == 1) ? 0 : i3;

            const char * A_ptr = (const char *) src0->data
                + a_i3 * src0->nb[3]
                + a_i2 * src0->nb[2];
            const char * B_ptr = (const char *) src1->data
                + i3 * src1->nb[3]
                + i2 * src1->nb[2];
            char * C_ptr = (char *) node->data
                + i3 * node->nb[3]
                + i2 * node->nb[2];

            cudaError_t err = cutlass_gemm_f16_sm80(
                A_ptr, B_ptr, C_ptr,
                M, N, K, lda, ldb, ldc,
                nullptr, 0,
                stream);
            if (err != cudaSuccess) {
                return false;
            }
        }
    }
    return true;
}

// ----------------------------------------------------------------------------
// MUL_MAT FP32 provider (priority = 120)
// ----------------------------------------------------------------------------

static bool cutlass_mul_mat_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }

    return mul_mat_shapes_compatible(op);
}

static bool cutlass_mul_mat_f32_execute(struct ggml_backend_triton_context * ctx,
                                        const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        return false;
    }

    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int M = (int) src1->ne[1];

    const int lda = K;
    const int ldb = K;
    const int ldc = N;

    cudaStream_t stream = to_cuda_stream(ctx->cu_stream);

    const int64_t src0_b2 = src0->ne[2];
    const int64_t src0_b3 = src0->ne[3];

    for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
            const int64_t a_i2 = (src0_b2 == 1) ? 0 : i2;
            const int64_t a_i3 = (src0_b3 == 1) ? 0 : i3;

            const char * A_ptr = (const char *) src0->data
                + a_i3 * src0->nb[3]
                + a_i2 * src0->nb[2];
            const char * B_ptr = (const char *) src1->data
                + i3 * src1->nb[3]
                + i2 * src1->nb[2];
            char * C_ptr = (char *) node->data
                + i3 * node->nb[3]
                + i2 * node->nb[2];

            cudaError_t err = cutlass_gemm_f32_sm80(
                A_ptr, B_ptr, C_ptr,
                M, N, K, lda, ldb, ldc,
                nullptr, 0,
                stream);
            if (err != cudaSuccess) {
                return false;
            }
        }
    }
    return true;
}

// ----------------------------------------------------------------------------
// MUL_MAT Q4_0 / Q8_0 fused dequant providers (priority = 130)
// ----------------------------------------------------------------------------

// Common shape/contiguity check for a quantized weight + FP16 activation
// MUL_MAT. `weight_type` selects the weight quantisation; activation must be
// FP16 and the K dimension must be a multiple of 32.
static bool cutlass_mul_mat_quant_f16_supports(const struct ggml_tensor * op,
                                               enum ggml_type weight_type) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (src0->type != weight_type) {
        return false;
    }
    if (src1->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }
    if ((src0->ne[0] % 32) != 0) {
        return false;
    }
    for (int d = 2; d < 4; ++d) {
        if (src0->ne[d] != src1->ne[d] && src0->ne[d] != 1) {
            return false;
        }
    }
    return true;
}

// Helper that does the actual batched dispatch. `kernel` is one of
// `cutlass_gemm_q4_0_f16_sm80` / `cutlass_gemm_q8_0_f16_sm80`.
typedef cudaError_t (*quant_gemm_kernel_t)(
    const void *, const void *, void *,
    int, int, int,
    void *, size_t,
    cudaStream_t);

static bool cutlass_mul_mat_quant_f16_execute(struct ggml_backend_triton_context * ctx,
                                              const struct ggml_tensor * node,
                                              quant_gemm_kernel_t kernel) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        return false;
    }

    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int M = (int) src1->ne[1];

    cudaStream_t stream = to_cuda_stream(ctx->cu_stream);

    const int64_t src0_b2 = src0->ne[2];
    const int64_t src0_b3 = src0->ne[3];

    for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
            const int64_t a_i2 = (src0_b2 == 1) ? 0 : i2;
            const int64_t a_i3 = (src0_b3 == 1) ? 0 : i3;

            const char * A_ptr = (const char *) src0->data
                + a_i3 * src0->nb[3]
                + a_i2 * src0->nb[2];
            const char * B_ptr = (const char *) src1->data
                + i3 * src1->nb[3]
                + i2 * src1->nb[2];
            char * C_ptr = (char *) node->data
                + i3 * node->nb[3]
                + i2 * node->nb[2];

            cudaError_t err = kernel(
                A_ptr, B_ptr, C_ptr,
                M, N, K,
                nullptr, 0,
                stream);
            if (err != cudaSuccess) {
                return false;
            }
        }
    }
    return true;
}

static bool cutlass_mul_mat_q4_0_supports(const struct ggml_tensor * op) {
    return cutlass_mul_mat_quant_f16_supports(op, GGML_TYPE_Q4_0);
}

static bool cutlass_mul_mat_q4_0_execute(struct ggml_backend_triton_context * ctx,
                                         const struct ggml_tensor * node) {
    return cutlass_mul_mat_quant_f16_execute(ctx, node, cutlass_gemm_q4_0_f16_sm80);
}

static bool cutlass_mul_mat_q8_0_supports(const struct ggml_tensor * op) {
    return cutlass_mul_mat_quant_f16_supports(op, GGML_TYPE_Q8_0);
}

static bool cutlass_mul_mat_q8_0_execute(struct ggml_backend_triton_context * ctx,
                                         const struct ggml_tensor * node) {
    return cutlass_mul_mat_quant_f16_execute(ctx, node, cutlass_gemm_q8_0_f16_sm80);
}

// ----------------------------------------------------------------------------
// GELU FP16 provider (priority = 90, lower than Triton)
// ----------------------------------------------------------------------------

static bool cutlass_gelu_fp16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY) {
        return false;
    }
    if (cutlass_provider_get_unary_op(op) != GGML_UNARY_OP_GELU) {
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

static bool cutlass_gelu_fp16_execute(struct ggml_backend_triton_context * ctx,
                                      const struct ggml_tensor * node) {
    const int64_t n_elements = ggml_nelements(node);
    const void *  src_data   = node->src[0]->data;
    void *        dst_data   = node->data;

    if (src_data == nullptr || dst_data == nullptr) {
        return false;
    }

    cudaStream_t stream = to_cuda_stream(ctx->cu_stream);

    cudaError_t err = cutlass_gelu_fp16(src_data, dst_data, (int) n_elements, stream);
    return (err == cudaSuccess);
}

// ----------------------------------------------------------------------------
// GELU FP32 provider (priority = 90, lower than Triton)
// ----------------------------------------------------------------------------

static bool cutlass_gelu_fp32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY) {
        return false;
    }
    if (cutlass_provider_get_unary_op(op) != GGML_UNARY_OP_GELU) {
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

static bool cutlass_gelu_fp32_execute(struct ggml_backend_triton_context * ctx,
                                      const struct ggml_tensor * node) {
    const int64_t n_elements = ggml_nelements(node);
    const void *  src_data   = node->src[0]->data;
    void *        dst_data   = node->data;

    if (src_data == nullptr || dst_data == nullptr) {
        return false;
    }

    cudaStream_t stream = to_cuda_stream(ctx->cu_stream);

    cudaError_t err = cutlass_gelu_fp32(src_data, dst_data, (int) n_elements, stream);
    return (err == cudaSuccess);
}

// ----------------------------------------------------------------------------
// Registration
// ----------------------------------------------------------------------------

void ggml_triton_register_cutlass_providers(ggml_triton_op_registry & registry) {
    // MUL_MAT Q4_0 -- highest priority (130). Fused dequant GEMM is the key
    // use case for quantised weights, so prefer it whenever applicable.
    registry.register_impl(GGML_OP_MUL_MAT, {
        /* .name     = */ "cutlass_gemm_q4_0_f16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_CUTLASS,
        /* .supports = */ cutlass_mul_mat_q4_0_supports,
        /* .execute  = */ cutlass_mul_mat_q4_0_execute,
        /* .priority = */ 130,
    });

    // MUL_MAT Q8_0 -- highest priority (130).
    registry.register_impl(GGML_OP_MUL_MAT, {
        /* .name     = */ "cutlass_gemm_q8_0_f16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_CUTLASS,
        /* .supports = */ cutlass_mul_mat_q8_0_supports,
        /* .execute  = */ cutlass_mul_mat_q8_0_execute,
        /* .priority = */ 130,
    });

    // MUL_MAT FP16 -- higher than Triton (120 > 100)
    registry.register_impl(GGML_OP_MUL_MAT, {
        /* .name     = */ "cutlass_gemm_f16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_CUTLASS,
        /* .supports = */ cutlass_mul_mat_f16_supports,
        /* .execute  = */ cutlass_mul_mat_f16_execute,
        /* .priority = */ 120,
    });

    // MUL_MAT FP32 -- higher than Triton (120 > 100)
    registry.register_impl(GGML_OP_MUL_MAT, {
        /* .name     = */ "cutlass_gemm_f32_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_CUTLASS,
        /* .supports = */ cutlass_mul_mat_f32_supports,
        /* .execute  = */ cutlass_mul_mat_f32_execute,
        /* .priority = */ 120,
    });

    // GELU FP16 -- lower priority than Triton (90 < 100), acts as backup
    registry.register_impl(GGML_OP_UNARY, {
        /* .name     = */ "cutlass_gelu_fp16",
        /* .provider = */ GGML_TRITON_PROVIDER_CUTLASS,
        /* .supports = */ cutlass_gelu_fp16_supports,
        /* .execute  = */ cutlass_gelu_fp16_execute,
        /* .priority = */ 90,
    });

    // GELU FP32 -- lower priority than Triton (90 < 100), acts as backup
    registry.register_impl(GGML_OP_UNARY, {
        /* .name     = */ "cutlass_gelu_fp32",
        /* .provider = */ GGML_TRITON_PROVIDER_CUTLASS,
        /* .supports = */ cutlass_gelu_fp32_supports,
        /* .execute  = */ cutlass_gelu_fp32_execute,
        /* .priority = */ 90,
    });
}
