// ggml/src/ggml-triton/ggml-triton-provider-flash-attn.cpp
//
// B.3 FlashAttn AOT provider (Stage 1).  See docs/development/ROADMAP.md
// §3 Phase B.3 for context.
//
// Math reference (bit-equivalent):
//   ggml/src/ggml-cpu/ops.cpp:8846   ggml_compute_forward_flash_attn_ext_f16 (prefill)
//   ggml/src/ggml-cpu/ops.cpp:8248   ggml_compute_forward_flash_attn_ext_f16_one_chunk (decode)
//   ggml/src/ggml-cpu/ops.cpp:8776   ggml_flash_attn_ext_reduce_partials (host CPU reduce)
//
// 12 impls = 3 head_dim × 2 dtype × 2 kernel (prefill, decode); each impl
// backed by 1 AOT-compiled launcher. Total 12 launcher functions.
//
// Decode execute is multi-step (per design spec §1.5):
//   1. Launch kernel (writes partials to device scratch)
//   2. cuMemcpyDtoHAsync of scratch to host
//   3. cuStreamSynchronize (barrier so kernel writes visible to CPU)
//   4. CPU reduce pass over partials (writes final dst)
//   5. Return
//
// First persistent per-call state in the Triton subsystem (B.1/B.2
// launchers are stateless).  Scratch is allocated lazily on first decode
// call (ensure_decode_scratch); freed in ggml_backend_triton_free.

#include "ggml-triton-provider-flash-attn.h"
#include "ggml-triton-provider.h"
#include "ggml-triton-context.h"

#include "kernels/include/triton_kernels.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>


// --- 4.A: tensor dim helpers --------------------------------------------

static inline int64_t op_neq0(const ggml_tensor * op) { return op->src[0]->ne[0]; }
static inline int64_t op_neq1(const ggml_tensor * op) { return op->src[0]->ne[1]; }
static inline int64_t op_neq2(const ggml_tensor * op) { return op->src[0]->ne[2]; }
static inline int64_t op_neq3(const ggml_tensor * op) { return op->src[0]->ne[3]; }
static inline int64_t op_nek1(const ggml_tensor * op) { return op->src[1]->ne[1]; }
static inline int64_t op_nek2(const ggml_tensor * op) { return op->src[1]->ne[2]; }
static inline int64_t op_nev2(const ggml_tensor * op) { return op->src[2]->ne[2]; }


// --- 4.B: shape constraints ----------------------------------------------

static inline bool op_is_flash_attn(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_FLASH_ATTN_EXT;
}

static inline bool op_mask_is_null(const ggml_tensor * op) {
    return op->src[3] == nullptr;
}

static inline bool op_is_mha(const ggml_tensor * op) {
    return op_neq2(op) == op_nek2(op) && op_neq2(op) == op_nev2(op);
}

static inline bool op_n_heads_supported(const ggml_tensor * op) {
    return op_neq2(op) * op_neq3(op) <= 32;
}

static inline bool op_dtypes_match(const ggml_tensor * op, enum ggml_type want) {
    return op->type == want
        && op->src[0]->type == want
        && op->src[1]->type == want
        && op->src[2]->type == want;
}

static inline bool op_is_contiguous(const ggml_tensor * t) {
    return t->nb[0] == ggml_type_size(t->type)
        && t->nb[1] == t->nb[0] * t->ne[0]
        && t->nb[2] == t->nb[1] * t->ne[1]
        && t->nb[3] == t->nb[2] * t->ne[2];
}

static inline bool op_head_dim_in_set(const ggml_tensor * op, int32_t want) {
    int32_t dk = (int32_t)op_neq0(op);
    return dk == want;
}


// --- 4.D: scale precomputation ------------------------------------------

static inline float op_scale(const ggml_tensor * op) {
    return 1.0f / sqrtf((float)op_neq0(op));
}


// --- 4.E: scratch alloc / resize ----------------------------------------

static inline int ensure_decode_scratch(ggml_backend_triton_context * ctx, size_t needed) {
    if (ctx->decode_scratch_size >= needed) return 0;
    if (ctx->decode_scratch) {
        cuMemFree(ctx->decode_scratch);
        ctx->decode_scratch = 0;
    }
    if (ctx->decode_scratch_host) {
        free(ctx->decode_scratch_host);
        ctx->decode_scratch_host = nullptr;
    }
    if (cuMemAlloc(&ctx->decode_scratch, needed) != CUDA_SUCCESS) return -1;
    ctx->decode_scratch_host = (float*)malloc(needed);
    if (!ctx->decode_scratch_host) {
        cuMemFree(ctx->decode_scratch);
        ctx->decode_scratch = 0;
        return -1;
    }
    ctx->decode_scratch_size = needed;
    return 0;
}


// --- PREFILL / HD=64 / fp16 ---------------------------------------------

static bool triton_flash_attn_prefill_hd64_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 64))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd64_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd64_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- PREFILL / HD=64 / fp32 ---------------------------------------------

static bool triton_flash_attn_prefill_hd64_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 64))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd64_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd64_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- PREFILL / HD=96 / fp16 ---------------------------------------------

static bool triton_flash_attn_prefill_hd96_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 96))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd96_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd96_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- PREFILL / HD=96 / fp32 ---------------------------------------------

static bool triton_flash_attn_prefill_hd96_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 96))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd96_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd96_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- PREFILL / HD=128 / fp16 --------------------------------------------

static bool triton_flash_attn_prefill_hd128_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 128))     return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd128_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd128_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- PREFILL / HD=128 / fp32 --------------------------------------------

static bool triton_flash_attn_prefill_hd128_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 128))     return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd128_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd128_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- DECODE / HD=64 / fp16 ----------------------------------------------
// (Decode is 4-step: kernel + D2H + sync + CPU reduce.  See design §4.4.)

static bool triton_flash_attn_decode_hd64_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 64))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd64_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 64;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd64_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- DECODE / HD=64 / fp32 ----------------------------------------------

static bool triton_flash_attn_decode_hd64_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 64))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd64_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 64;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd64_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- DECODE / HD=96 / fp16 ----------------------------------------------

static bool triton_flash_attn_decode_hd96_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 96))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd96_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 96;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd96_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- DECODE / HD=96 / fp32 ----------------------------------------------

static bool triton_flash_attn_decode_hd96_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 96))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd96_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 96;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd96_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- DECODE / HD=128 / fp16 ---------------------------------------------

static bool triton_flash_attn_decode_hd128_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 128))     return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd128_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 128;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd128_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- DECODE / HD=128 / fp32 ---------------------------------------------

static bool triton_flash_attn_decode_hd128_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 128))     return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd128_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 128;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd128_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- registration -----------------------------------------------------------

void ggml_triton_register_flash_attn_providers(ggml_triton_op_registry & registry) {
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd64_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd64_fp16_supports,
        triton_flash_attn_prefill_hd64_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd64_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd64_fp32_supports,
        triton_flash_attn_prefill_hd64_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd96_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd96_fp16_supports,
        triton_flash_attn_prefill_hd96_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd96_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd96_fp32_supports,
        triton_flash_attn_prefill_hd96_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd128_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd128_fp16_supports,
        triton_flash_attn_prefill_hd128_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd128_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd128_fp32_supports,
        triton_flash_attn_prefill_hd128_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd64_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd64_fp16_supports,
        triton_flash_attn_decode_hd64_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd64_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd64_fp32_supports,
        triton_flash_attn_decode_hd64_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd96_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd96_fp16_supports,
        triton_flash_attn_decode_hd96_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd96_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd96_fp32_supports,
        triton_flash_attn_decode_hd96_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd128_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd128_fp16_supports,
        triton_flash_attn_decode_hd128_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd128_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd128_fp32_supports,
        triton_flash_attn_decode_hd128_fp32_execute,
        100,
    });
}
