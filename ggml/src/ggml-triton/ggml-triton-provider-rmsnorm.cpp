// ggml/src/ggml-triton/ggml-triton-provider-rmsnorm.cpp
//
// B.1 RMSNorm AOT provider (Stage 1, constexpr eps + USE_WEIGHT).
// See docs/development/ROADMAP.md §3 Phase B.1 for context.
//
// Math reference (bit-equivalent):
//   ggml/src/ggml-cpu/ops.cpp:3758-3821
//     ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>
//   ggml/src/ggml-triton/ggml-triton-provider-cpu.cpp:464-510
//     cpu_rms_norm_f32_execute
//
// Per-row computation:
//   y[i] = x[i] * rsqrt(mean(x[i]^2) + eps) * w[i]    (weighted variant)
//   y[i] = x[i] * rsqrt(mean(x[i]^2) + eps)           (unweighted variant, src[1]==nullptr)
//
// Per Oracle review: MiniMind-3 (and tests/test-backend-ops.cpp's test_rms_norm)
// call ggml_rms_norm(ctx, a, eps) WITHOUT a weight tensor, so src[1]==nullptr is
// the COMMON case, not the exception. We ship BOTH variants: unweighted and
// weighted. Each variant has fp16 and fp32 specializations, total 4 impls.
//
// The kernel source is triton_kernels/rms_norm.py; the AOT launcher
// signatures are emitted by scripts/compile_kernels.py into
// ggml/src/ggml-triton/kernels/generated/rms_norm_{unweighted,weighted}_fp{16,32}_sm80.{h,c}.
//
// Stage 2 retro-fix: launch grid is num_blocks (= ne[1]*ne[2]*ne[3]) via
// grid_mode="exact"; the unweighted launcher also takes a dummy d_w (= 0)
// because the kernel reserves the slot for both variants.

#include "ggml-triton-provider-rmsnorm.h"
#include "ggml-triton-provider.h"
#include "ggml-triton-context.h"

#include "kernels/include/triton_kernels.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>


// --- shared predicates (extracted to avoid 4x duplication) ------------------

static inline bool rms_norm_eps_matches_stage1(const struct ggml_tensor * node) {
    // Stage 1: eps is a tl.constexpr (= 1e-6) baked into the AOT launcher.
    // We use a tolerance-bounded comparison (per Oracle review §6) so a
    // 1-ULP rounding drift in the runtime eps still routes to the triton
    // AOT path; a bit-exact `!=` would falsely fall back to CPU for any
    // computed value (e.g. from JSON or arithmetic). Stage 2 will thread
    // runtime eps through the launcher and remove this gate.
    float eps_runtime = 0.0f;
    std::memcpy(&eps_runtime, node->op_params, sizeof(float));
    return std::fabs(eps_runtime - 1.0e-6f) <= 1.0e-7f;
}

static inline bool rms_norm_row_fits_stage1(const struct ggml_tensor * node) {
    // Stage 1 constraint: BLOCK_SIZE (=1024) must be >= the row length ne00.
    // The kernel masks out-of-range loads (other=0.0) so sum-of-squares is
    // correct for any N <= BLOCK_SIZE, but the row is laid out for one
    // program per row. Reject rows larger than 1024 to keep semantics
    // correct; a follow-up adds a multi-block variant.
    return node->src[0]->ne[0] <= 1024;
}

// Total row count = ne[1] * ne[2] * ne[3]; used as the C grid size via
// grid_mode="exact" (one Triton program per row).
static inline int32_t rms_norm_num_blocks(const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    return (int32_t)(src0->ne[1] * src0->ne[2] * src0->ne[3]);
}


// --- unweighted / fp16 ------------------------------------------------------

static bool triton_rms_norm_unweighted_fp16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM) return false;
    if (op->type != GGML_TYPE_F16) return false;
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F16) return false;
    if (op->src[1] != nullptr) return false;
    if (!rms_norm_row_fits_stage1(op)) return false;
    if (!rms_norm_eps_matches_stage1(op)) return false;
    return true;
}

static bool triton_rms_norm_unweighted_fp16_execute(struct ggml_backend_triton_context * ctx,
                                                   const struct ggml_tensor * node) {
    if (node->src[0]->data == nullptr || node->data == nullptr) {
        return false;
    }
    const CUdeviceptr d_x        = (CUdeviceptr) node->src[0]->data;
    const CUdeviceptr d_y        = (CUdeviceptr) node->data;
    const int32_t     N          = (int32_t) node->src[0]->ne[0];
    const int32_t     num_blocks = rms_norm_num_blocks(node);
    const int rc = triton_launch_rms_norm_unweighted_fp16_sm80(
        ctx->cu_stream, d_x, d_y, 0 /* d_w dummy: USE_WEIGHT=0 in kernel */, N, num_blocks);
    return rc == 0;
}


// --- weighted / fp16 --------------------------------------------------------

static bool triton_rms_norm_weighted_fp16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM) return false;
    if (op->type != GGML_TYPE_F16) return false;
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F16) return false;
    if (op->src[1] == nullptr || op->src[1]->type != GGML_TYPE_F16) return false;
    if (!rms_norm_row_fits_stage1(op)) return false;
    if (!rms_norm_eps_matches_stage1(op)) return false;
    return true;
}

static bool triton_rms_norm_weighted_fp16_execute(struct ggml_backend_triton_context * ctx,
                                                 const struct ggml_tensor * node) {
    if (node->src[0]->data == nullptr || node->src[1]->data == nullptr || node->data == nullptr) {
        return false;
    }
    const CUdeviceptr d_x        = (CUdeviceptr) node->src[0]->data;
    const CUdeviceptr d_w        = (CUdeviceptr) node->src[1]->data;
    const CUdeviceptr d_y        = (CUdeviceptr) node->data;
    const int32_t     N          = (int32_t) node->src[0]->ne[0];
    const int32_t     num_blocks = rms_norm_num_blocks(node);
    const int rc = triton_launch_rms_norm_weighted_fp16_sm80(
        ctx->cu_stream, d_x, d_y, d_w, N, num_blocks);
    return rc == 0;
}


// --- unweighted / fp32 ------------------------------------------------------

static bool triton_rms_norm_unweighted_fp32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32) return false;
    if (op->src[1] != nullptr) return false;
    if (!rms_norm_row_fits_stage1(op)) return false;
    if (!rms_norm_eps_matches_stage1(op)) return false;
    return true;
}

static bool triton_rms_norm_unweighted_fp32_execute(struct ggml_backend_triton_context * ctx,
                                                   const struct ggml_tensor * node) {
    if (node->src[0]->data == nullptr || node->data == nullptr) {
        return false;
    }
    const CUdeviceptr d_x        = (CUdeviceptr) node->src[0]->data;
    const CUdeviceptr d_y        = (CUdeviceptr) node->data;
    const int32_t     N          = (int32_t) node->src[0]->ne[0];
    const int32_t     num_blocks = rms_norm_num_blocks(node);
    const int rc = triton_launch_rms_norm_unweighted_fp32_sm80(
        ctx->cu_stream, d_x, d_y, 0 /* d_w dummy: USE_WEIGHT=0 in kernel */, N, num_blocks);
    return rc == 0;
}


// --- weighted / fp32 --------------------------------------------------------

static bool triton_rms_norm_weighted_fp32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32) return false;
    if (op->src[1] == nullptr || op->src[1]->type != GGML_TYPE_F32) return false;
    if (!rms_norm_row_fits_stage1(op)) return false;
    if (!rms_norm_eps_matches_stage1(op)) return false;
    return true;
}

static bool triton_rms_norm_weighted_fp32_execute(struct ggml_backend_triton_context * ctx,
                                                 const struct ggml_tensor * node) {
    if (node->src[0]->data == nullptr || node->src[1]->data == nullptr || node->data == nullptr) {
        return false;
    }
    const CUdeviceptr d_x        = (CUdeviceptr) node->src[0]->data;
    const CUdeviceptr d_w        = (CUdeviceptr) node->src[1]->data;
    const CUdeviceptr d_y        = (CUdeviceptr) node->data;
    const int32_t     N          = (int32_t) node->src[0]->ne[0];
    const int32_t     num_blocks = rms_norm_num_blocks(node);
    const int rc = triton_launch_rms_norm_weighted_fp32_sm80(
        ctx->cu_stream, d_x, d_y, d_w, N, num_blocks);
    return rc == 0;
}


// --- registration -----------------------------------------------------------

void ggml_triton_register_rmsnorm_providers(ggml_triton_op_registry & registry) {
    registry.register_impl(GGML_OP_RMS_NORM, {
        /* .name     = */ "triton_rms_norm_unweighted_fp16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_rms_norm_unweighted_fp16_supports,
        /* .execute  = */ triton_rms_norm_unweighted_fp16_execute,
        /* .priority = */ 100,
    });
    registry.register_impl(GGML_OP_RMS_NORM, {
        /* .name     = */ "triton_rms_norm_weighted_fp16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_rms_norm_weighted_fp16_supports,
        /* .execute  = */ triton_rms_norm_weighted_fp16_execute,
        /* .priority = */ 100,
    });
    registry.register_impl(GGML_OP_RMS_NORM, {
        /* .name     = */ "triton_rms_norm_unweighted_fp32_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_rms_norm_unweighted_fp32_supports,
        /* .execute  = */ triton_rms_norm_unweighted_fp32_execute,
        /* .priority = */ 100,
    });
    registry.register_impl(GGML_OP_RMS_NORM, {
        /* .name     = */ "triton_rms_norm_weighted_fp32_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_rms_norm_weighted_fp32_supports,
        /* .execute  = */ triton_rms_norm_weighted_fp32_execute,
        /* .priority = */ 100,
    });
}
