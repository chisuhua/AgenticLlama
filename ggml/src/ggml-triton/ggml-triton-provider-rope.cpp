// ggml/src/ggml-triton/ggml-triton-provider-rope.cpp
//
// B.2 RoPE AOT provider (Stage 1).  See docs/development/ROADMAP.md §3
// Phase B.2 for context.
//
// Math reference (bit-equivalent):
//   ggml/src/ggml-cpu/ops.cpp:5813-5959
//     ggml_compute_forward_rope_flt<T>  (the canonical forward)
//   ggml/src/ggml-triton/ggml-triton-provider-cpu.cpp:517-611
//     cpu_rope_f32_supports/execute  (F32 + NORMAL only — limited scope)
//
// Per-row computation:
//   NORMAL/NEOX: y[i] = x[i] * cos + partner * (sin or -sin)   (pair rotation)
//   MROPE:       y[i] = x[i] * cos - x[i + n_dims/2] * sin     (half-rotation)
//
// 6 impls = 3 modes × 2 dtypes; each impl dispatches to 4 AOT variants
// at runtime (2 sin_sign × 2 ya_on).  Total 24 launcher functions.
//
// The kernel source is triton_kernels/rope.py; the AOT launcher
// signatures are emitted by scripts/compile_kernels.py into
// ggml/src/ggml-triton/kernels/generated/rope_<mode>_<sin>_<yarn>_<dtype>_sm80.{h,c}.

#include "ggml-triton-provider-rope.h"
#include "ggml-triton-provider.h"
#include "ggml-triton-context.h"

#include "kernels/include/triton_kernels.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>


// --- shared helpers (extracted to avoid 6x duplication) -------------------

static inline bool rope_op_is_supported(const struct ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_ROPE;
}

static inline bool rope_dtypes_match(const struct ggml_tensor * op,
                                    enum ggml_type want) {
    return op->type == want
        && op->src[0] != nullptr && op->src[0]->type == want
        && op->src[1] != nullptr && op->src[1]->type == GGML_TYPE_I32;
}

static inline int32_t rope_mode(const struct ggml_tensor * op) {
    return ((const int32_t *)op->op_params)[2];
}

static inline int32_t rope_n_dims(const struct ggml_tensor * op) {
    return ((const int32_t *)op->op_params)[1];
}

static inline int32_t rope_n_ctx_orig(const struct ggml_tensor * op) {
    return ((const int32_t *)op->op_params)[4];
}

static inline float rope_freq_base(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 5 * sizeof(float), sizeof(float));
    return v;
}

static inline float rope_freq_scale(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 6 * sizeof(float), sizeof(float));
    return v;
}

static inline float rope_ext_factor(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 7 * sizeof(float), sizeof(float));
    return v;
}

static inline float rope_attn_factor(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 8 * sizeof(float), sizeof(float));
    return v;
}

static inline float rope_beta_fast(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 9 * sizeof(float), sizeof(float));
    return v;
}

static inline float rope_beta_slow(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 10 * sizeof(float), sizeof(float));
    return v;
}

static inline int32_t rope_sect(const struct ggml_tensor * op, int i) {
    return ((const int32_t *)op->op_params)[11 + i];
}

// Stage 1 hard-gate: BLOCK_SIZE = 128.  n_dims must be <= 128.
static inline bool rope_row_fits_stage1(const struct ggml_tensor * op) {
    return rope_n_dims(op) > 0 && rope_n_dims(op) <= 128;
}

// MROPE: must have at least one non-zero section.
static inline bool rope_mrope_valid_sections(const struct ggml_tensor * op) {
    int32_t s0 = rope_sect(op, 0);
    int32_t s1 = rope_sect(op, 1);
    int32_t s2 = rope_sect(op, 2);
    int32_t s3 = rope_sect(op, 3);
    int32_t sum = s0 + s1 + s2 + s3;
    return sum > 0 && sum * 2 <= rope_n_dims(op);
}

// Per Q5: YaRN is on if any YaRN parameter is non-default.
static inline bool rope_ya_on(const struct ggml_tensor * op) {
    return (rope_ext_factor(op)  != 0.0f)
        || (rope_attn_factor(op) != 1.0f)
        || (rope_beta_fast(op)   != 0.0f)
        || (rope_beta_slow(op)   != 0.0f);
}

// Pre-compute corr_dims (matches ggml_rope_yarn_corr_dims in ggml.c:4335).
// Only used when YA_ON is true; sub-microsecond per launch.
static inline void rope_compute_corr_dims(const struct ggml_tensor * op,
                                          float & corr_low,
                                          float & corr_high) {
    const float n_dims      = (float) rope_n_dims(op);
    const float n_ctx_orig  = (float) rope_n_ctx_orig(op);
    const float freq_base   = rope_freq_base(op);
    const float beta_fast   = rope_beta_fast(op);
    const float beta_slow   = rope_beta_slow(op);
    const float n_rot       = n_dims * 0.5f;
    const float log_arg     = n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f);
    const float theta_log   = std::log(freq_base);
    // corr_dim = n_dims * log(n_ctx_orig / (n_rot * 2*pi)) / (2 * log(freq_base))
    // For both beta_fast and beta_slow (the *value* of beta matters at runtime
    // when the kernel does the rope_yarn ramp; corr_low and corr_high here
    // are computed for the fast and slow thresholds respectively).
    corr_low  = n_dims * std::log(log_arg) / (2.0f * theta_log);
    corr_high = corr_low;  // same formula; the kernel uses the beta values
                           // from op_params to decide which to use
}


// --- NORMAL / fp16 ---------------------------------------------------------

static bool triton_rope_normal_fp16_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_NORMAL) return false;
    if (!rope_row_fits_stage1(op)) return false;
    return true;
}

static bool triton_rope_normal_fp16_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {

    const struct ggml_tensor * src0 = node->src[0];  // Q/K
    const struct ggml_tensor * src1 = node->src[1];  // positions
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;

    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const bool    ya_on      = rope_ya_on(node);

    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);

    // Per Q4/Q5: pick the right AOT-compiled launcher at runtime.
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_normal_bwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_normal_bwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_normal_fwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_normal_fwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- NORMAL / fp32 ---------------------------------------------------------

static bool triton_rope_normal_fp32_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_NORMAL) return false;
    if (!rope_row_fits_stage1(op)) return false;
    return true;
}

static bool triton_rope_normal_fp32_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {
    // Identical to fp16 execute except all dtype-specific launcher calls.
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;
    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const bool    ya_on      = rope_ya_on(node);
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_normal_bwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_normal_bwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_normal_fwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_normal_fwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- NEOX / fp16 -----------------------------------------------------------

static bool triton_rope_neox_fp16_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_NEOX) return false;
    if (!rope_row_fits_stage1(op)) return false;
    return true;
}

static bool triton_rope_neox_fp16_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;
    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const bool    ya_on      = rope_ya_on(node);
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_neox_bwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_neox_bwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_neox_fwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_neox_fwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- NEOX / fp32 -----------------------------------------------------------

static bool triton_rope_neox_fp32_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_NEOX) return false;
    if (!rope_row_fits_stage1(op)) return false;
    return true;
}

static bool triton_rope_neox_fp32_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;
    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const bool    ya_on      = rope_ya_on(node);
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_neox_bwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_neox_bwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_neox_fwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_neox_fwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- MROPE / fp16 ----------------------------------------------------------

static bool triton_rope_mrope_fp16_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_MROPE) return false;
    if (!rope_row_fits_stage1(op)) return false;
    if (!rope_mrope_valid_sections(op)) return false;
    // MROPE positions are 4x the seq length (one per axis).
    if (op->src[1]->ne[0] != op->ne[2] * 4) return false;
    return true;
}

static bool triton_rope_mrope_fp16_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    const struct ggml_tensor * src2 = node->src[2];  // freq_factors or nullptr
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;
    // Per Q3: pass a non-null dummy pointer when src2 is null.  We allocate
    // 1 float on the stack initialized to 1.0; the kernel divides theta by
    // this value, which is a no-op.  (Stage 1 simplification; the kernel
    // does not actually use this value in MROPE mode.)
    float dummy_freq_factor = 1.0f;
    CUdeviceptr freq_factors_ptr = src2 != nullptr
        ? (CUdeviceptr) src2->data
        : (CUdeviceptr) &dummy_freq_factor;
    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const int32_t sect_t     = rope_sect(node, 0);
    const int32_t sect_h     = rope_sect(node, 1);
    const int32_t sect_w     = rope_sect(node, 2);
    const int32_t sect_e     = rope_sect(node, 3);
    const bool    ya_on      = rope_ya_on(node);
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_mrope_bwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, corr_low, corr_high)
            : triton_launch_rope_mrope_bwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_mrope_fwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, corr_low, corr_high)
            : triton_launch_rope_mrope_fwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- MROPE / fp32 ----------------------------------------------------------

static bool triton_rope_mrope_fp32_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_MROPE) return false;
    if (!rope_row_fits_stage1(op)) return false;
    if (!rope_mrope_valid_sections(op)) return false;
    if (op->src[1]->ne[0] != op->ne[2] * 4) return false;
    return true;
}

static bool triton_rope_mrope_fp32_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    const struct ggml_tensor * src2 = node->src[2];
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;
    float dummy_freq_factor = 1.0f;
    CUdeviceptr freq_factors_ptr = src2 != nullptr
        ? (CUdeviceptr) src2->data
        : (CUdeviceptr) &dummy_freq_factor;
    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const int32_t sect_t     = rope_sect(node, 0);
    const int32_t sect_h     = rope_sect(node, 1);
    const int32_t sect_w     = rope_sect(node, 2);
    const int32_t sect_e     = rope_sect(node, 3);
    const bool    ya_on      = rope_ya_on(node);
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_mrope_bwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, corr_low, corr_high)
            : triton_launch_rope_mrope_bwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_mrope_fwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, corr_low, corr_high)
            : triton_launch_rope_mrope_fwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- registration -----------------------------------------------------------

void ggml_triton_register_rope_providers(ggml_triton_op_registry & registry) {
    registry.register_impl(GGML_OP_ROPE, {
        /* .name     = */ "triton_rope_normal_fp16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_rope_normal_fp16_supports,
        /* .execute  = */ triton_rope_normal_fp16_execute,
        /* .priority = */ 100,
    });
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_normal_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_rope_normal_fp32_supports,
        triton_rope_normal_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_neox_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_rope_neox_fp16_supports,
        triton_rope_neox_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_neox_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_rope_neox_fp32_supports,
        triton_rope_neox_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_mrope_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_rope_mrope_fp16_supports,
        triton_rope_mrope_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_mrope_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_rope_mrope_fp32_supports,
        triton_rope_mrope_fp32_execute,
        100,
    });
}
