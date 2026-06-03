// ggml-tilelang-dispatch.cpp - selects the right TileLang AOT-compiled kernel
// for a given ggml node and launches it via the standard CUDA Runtime API.
//
// All kernel launchers are exposed by the AOT-generated kernels through
// `tilelang_kernels.h` as plain `extern "C"` functions.

#include "ggml-tilelang-dispatch.h"
#include "ggml-tilelang-context.h"
#include "ggml-impl.h"

#include "tilelang_kernels.h"

#include <cstdint>

// ---------------------------------------------------------------------------
// supports_op
// ---------------------------------------------------------------------------

static bool tilelang_is_supported_type(enum ggml_type t) {
    return t == GGML_TYPE_F16 || t == GGML_TYPE_F32;
}

bool ggml_backend_tilelang_supports_op(const struct ggml_tensor * op) {
    if (op == nullptr) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL: {
            const ggml_tensor * a = op->src[0];
            const ggml_tensor * b = op->src[1];
            if (a == nullptr || b == nullptr) {
                return false;
            }
            // Only same dtype across operands and output.
            if (a->type != op->type || b->type != op->type) {
                return false;
            }
            if (!tilelang_is_supported_type(op->type)) {
                return false;
            }
            // Element-wise on identically-shaped contiguous tensors.
            if (!ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, op)) {
                return false;
            }
            if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(op)) {
                return false;
            }
            return true;
        }
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

static enum ggml_status tilelang_launch_elementwise(struct ggml_tensor * node, cudaStream_t stream) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];

    const int64_t n64 = ggml_nelements(node);
    if (n64 > INT32_MAX) {
        GGML_TILELANG_LOG_ERROR("[tilelang] element count %lld exceeds int32 launcher API\n",
                                (long long) n64);
        return GGML_STATUS_FAILED;
    }
    const int32_t N = (int32_t) n64;

    void * A = a->data;
    void * B = b->data;
    void * C = node->data;

    switch (node->op) {
        case GGML_OP_ADD:
            if (node->type == GGML_TYPE_F16) {
                tilelang_add_fp16(A, B, C, N, stream);
            } else {
                tilelang_add_fp32(A, B, C, N, stream);
            }
            return GGML_STATUS_SUCCESS;
        case GGML_OP_MUL:
            if (node->type == GGML_TYPE_F16) {
                tilelang_mul_fp16(A, B, C, N, stream);
            } else {
                tilelang_mul_fp32(A, B, C, N, stream);
            }
            return GGML_STATUS_SUCCESS;
        default:
            return GGML_STATUS_FAILED;
    }
}

enum ggml_status ggml_backend_tilelang_dispatch(struct ggml_tensor * node, cudaStream_t stream) {
    if (node == nullptr) {
        return GGML_STATUS_FAILED;
    }
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return GGML_STATUS_SUCCESS;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return tilelang_launch_elementwise(node, stream);
        default:
            GGML_TILELANG_LOG_ERROR("[tilelang] unsupported op %s in dispatch\n", ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
    }
}
