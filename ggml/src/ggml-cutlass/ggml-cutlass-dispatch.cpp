// =============================================================================
// CUTLASS backend op dispatch.
// =============================================================================

#include <cstdio>

#include <cuda_runtime.h>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cutlass-dispatch.h"
#include "ggml-cutlass-context.h"

#include "kernels/include/cutlass_kernels.h"

namespace ggml_cutlass {

// -----------------------------------------------------------------------------
// supports_op
// -----------------------------------------------------------------------------

static bool supports_mul_mat(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    // Skeleton: only FP16 x FP16 -> FP16 with a 2-D, contiguous, non-broadcast
    // case. Higher-rank batched mulmats and quantised weights will be lit up
    // by later tasks once the corresponding GEMM kernels exist.
    if (src0->type != GGML_TYPE_F16 ||
        src1->type != GGML_TYPE_F16 ||
        op->type   != GGML_TYPE_F16) {
        return false;
    }

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    // No broadcasting in higher dimensions for the skeleton.
    if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }

    return true;
}

static bool supports_unary(const ggml_tensor * op) {
    if (ggml_get_unary_op(op) != GGML_UNARY_OP_GELU) {
        return false;
    }
    if (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (op->src[0]->type != op->type) {
        return false;
    }
    return true;
}

bool supports_op(const ggml_tensor * op) {
    switch (op->op) {
        // No-op layout transformations are always cheap.
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
            return supports_mul_mat(op);

        case GGML_OP_UNARY:
            return supports_unary(op);

        default:
            return false;
    }
}

// -----------------------------------------------------------------------------
// compute_op
// -----------------------------------------------------------------------------

static ggml_status compute_mul_mat(ggml_backend_cutlass_context & ctx, ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0]; // weights : ne[0]=K (inner), ne[1]=N
    const ggml_tensor * src1 = node->src[1]; // activations: ne[0]=K, ne[1]=M

    // ggml MUL_MAT semantics:
    //   src0 has shape (K, N)   -- weights, ne[0] is the inner / contraction dim
    //   src1 has shape (K, M)   -- activations
    //   dst  has shape (N, M)   -- result; ne[0]=N, ne[1]=M
    //
    // Both src0 and src1 are contiguous and store their innermost dimension
    // (K) as the fastest-varying axis. Viewed as standard row-major matrices
    // they are:
    //   src0 -> A : N x K row-major (lda = K)
    //   src1 -> B : M x K row-major (ldb = K)
    //   dst  -> C : M x N row-major (ldc = N)
    //
    // and `cutlass_gemm_f16_sm80` computes  C[m, n] = sum_k A[n, k] * B[m, k]
    // which matches the ggml definition exactly.
    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int M = (int) src1->ne[1];

    const int lda = (int) (src0->nb[1] / ggml_type_size(src0->type)); // = K
    const int ldb = (int) (src1->nb[1] / ggml_type_size(src1->type)); // = K
    const int ldc = (int) (node->nb[1] / ggml_type_size(node->type)); // = N

    cudaError_t err = cutlass_gemm_f16_sm80(
        /* A           = */ src0->data,
        /* B           = */ src1->data,
        /* C           = */ node->data,
        /* M           = */ M,
        /* N           = */ N,
        /* K           = */ K,
        /* lda         = */ lda,
        /* ldb         = */ ldb,
        /* ldc         = */ ldc,
        /* workspace   = */ ctx.workspace,
        /* ws_size     = */ ctx.workspace_size,
        /* stream      = */ ctx.stream);

    if (err != cudaSuccess) {
        GGML_LOG_ERROR("CUTLASS backend: GEMM kernel failed: %s\n", cudaGetErrorString(err));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status compute_unary(ggml_backend_cutlass_context & ctx, ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    const int N = (int) ggml_nelements(node);

    cudaError_t err = cudaSuccess;
    switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_GELU:
            if (node->type == GGML_TYPE_F16) {
                err = cutlass_gelu_fp16(src->data, node->data, N, ctx.stream);
            } else {
                err = cutlass_gelu_fp32(src->data, node->data, N, ctx.stream);
            }
            break;
        default:
            return GGML_STATUS_FAILED;
    }

    if (err != cudaSuccess) {
        GGML_LOG_ERROR("CUTLASS backend: unary kernel failed: %s\n", cudaGetErrorString(err));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

ggml_status compute_op(ggml_backend_cutlass_context & ctx, ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return GGML_STATUS_SUCCESS;

        case GGML_OP_MUL_MAT:
            return compute_mul_mat(ctx, node);

        case GGML_OP_UNARY:
            return compute_unary(ctx, node);

        default:
            GGML_LOG_ERROR("CUTLASS backend: unsupported op %s\n", ggml_op_desc(node));
            return GGML_STATUS_FAILED;
    }
}

} // namespace ggml_cutlass
