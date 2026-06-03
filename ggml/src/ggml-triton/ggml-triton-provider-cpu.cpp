#include "ggml-triton-provider.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdint>

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

static enum ggml_unary_op cpu_provider_get_unary_op(const struct ggml_tensor * t) {
    return (enum ggml_unary_op) ggml_get_op_params_i32(t, 0);
}

static inline float cpu_gelu_f32(float x) {
    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static inline float cpu_silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// ----------------------------------------------------------------------------
// GELU FP32 provider
// ----------------------------------------------------------------------------

static bool cpu_gelu_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY) {
        return false;
    }
    if (cpu_provider_get_unary_op(op) != GGML_UNARY_OP_GELU) {
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

static bool cpu_gelu_f32_execute(struct ggml_backend_triton_context * /*ctx*/,
                                 const struct ggml_tensor * node) {
    const int64_t n_elements = ggml_nelements(node);
    const float * src_data   = (const float *) node->src[0]->data;
    float *       dst_data   = (float *) node->data;

    if (src_data == nullptr || dst_data == nullptr) {
        return false;
    }

    for (int64_t i = 0; i < n_elements; ++i) {
        dst_data[i] = cpu_gelu_f32(src_data[i]);
    }

    return true;
}

// ----------------------------------------------------------------------------
// GELU FP16 provider
// ----------------------------------------------------------------------------

static bool cpu_gelu_f16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY) {
        return false;
    }
    if (cpu_provider_get_unary_op(op) != GGML_UNARY_OP_GELU) {
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

static bool cpu_gelu_f16_execute(struct ggml_backend_triton_context * /*ctx*/,
                                 const struct ggml_tensor * node) {
    const int64_t n_elements = ggml_nelements(node);
    const ggml_fp16_t * src_data = (const ggml_fp16_t *) node->src[0]->data;
    ggml_fp16_t *       dst_data = (ggml_fp16_t *) node->data;

    if (src_data == nullptr || dst_data == nullptr) {
        return false;
    }

    for (int64_t i = 0; i < n_elements; ++i) {
        const float x = ggml_fp16_to_fp32(src_data[i]);
        dst_data[i]   = ggml_fp32_to_fp16(cpu_gelu_f32(x));
    }

    return true;
}

// ----------------------------------------------------------------------------
// SILU FP32 provider
// ----------------------------------------------------------------------------

static bool cpu_silu_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY) {
        return false;
    }
    if (cpu_provider_get_unary_op(op) != GGML_UNARY_OP_SILU) {
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

static bool cpu_silu_f32_execute(struct ggml_backend_triton_context * /*ctx*/,
                                 const struct ggml_tensor * node) {
    const int64_t n_elements = ggml_nelements(node);
    const float * src_data   = (const float *) node->src[0]->data;
    float *       dst_data   = (float *) node->data;

    if (src_data == nullptr || dst_data == nullptr) {
        return false;
    }

    for (int64_t i = 0; i < n_elements; ++i) {
        dst_data[i] = cpu_silu_f32(src_data[i]);
    }

    return true;
}

// ----------------------------------------------------------------------------
// SILU FP16 provider
// ----------------------------------------------------------------------------

static bool cpu_silu_f16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY) {
        return false;
    }
    if (cpu_provider_get_unary_op(op) != GGML_UNARY_OP_SILU) {
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

static bool cpu_silu_f16_execute(struct ggml_backend_triton_context * /*ctx*/,
                                 const struct ggml_tensor * node) {
    const int64_t n_elements = ggml_nelements(node);
    const ggml_fp16_t * src_data = (const ggml_fp16_t *) node->src[0]->data;
    ggml_fp16_t *       dst_data = (ggml_fp16_t *) node->data;

    if (src_data == nullptr || dst_data == nullptr) {
        return false;
    }

    for (int64_t i = 0; i < n_elements; ++i) {
        const float x = ggml_fp16_to_fp32(src_data[i]);
        dst_data[i]   = ggml_fp32_to_fp16(cpu_silu_f32(x));
    }

    return true;
}

// ----------------------------------------------------------------------------
// MUL_MAT — Matrix multiplication
// dst = src0^T × src1, where src0 is [ne00, ne01], src1 is [ne10, ne11]
// dst is [ne01, ne11]
// Actually: dst[i,j] = sum_k src0[k,i] * src1[k,j]
// ----------------------------------------------------------------------------

static bool cpu_mul_mat_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    if (op->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    // Support F32×F32→F32 and F16×F32→F32
    if (op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) {
        return false;
    }
    return true;
}

static bool cpu_mul_mat_f32_execute(struct ggml_backend_triton_context * /*ctx*/,
                                    const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        return false;
    }

    // ggml mul_mat semantics: dst = src0^T × src1
    // src0: [ne00, ne01] = [K, M], dst row dim = ne01
    // src1: [ne10, ne11] = [K, N], dst col dim = ne11
    // dst:  [ne0, ne1] = [ne01, ne11] = [M, N]
    // dst[i0, i1] = sum_k src0[k, i0] * src1[k, i1]

    const int64_t ne00 = src0->ne[0]; // K (inner dim)
    const int64_t ne01 = src0->ne[1]; // M (rows of dst)
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne11 = src1->ne[1]; // N (cols of dst)
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const bool src0_f16 = (src0->type == GGML_TYPE_F16);

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];
    const int64_t nb1  = node->nb[1];
    const int64_t nb2  = node->nb[2];
    const int64_t nb3  = node->nb[3];

    // Batch dimensions: broadcast src0 across src1 if needed
    const int64_t ne2 = ne12;
    const int64_t ne3 = ne13;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            // Broadcast: src0 may have ne02=1 or ne03=1
            const int64_t i02 = i2 % ne02;
            const int64_t i03 = i3 % ne03;

            // Pointer to src0's batch slice
            const char * s0_base = (const char *)src0->data + i02*nb02 + i03*nb03;
            // Pointer to src1's batch slice
            const char * s1_base = (const char *)src1->data + i2*nb12 + i3*nb13;
            // Pointer to dst's batch slice
            char * d_base = (char *)node->data + i2*nb2 + i3*nb3;

            for (int64_t i1 = 0; i1 < ne11; i1++) {
                // src1 row i1: data at s1_base + i1*nb11
                const char * s1_row = s1_base + i1*nb11;

                for (int64_t i0 = 0; i0 < ne01; i0++) {
                    // src0 row i0: data at s0_base + i0*nb01
                    const char * s0_row = s0_base + i0*nb01;

                    float sum = 0.0f;
                    for (int64_t k = 0; k < ne00; k++) {
                        float a;
                        if (src0_f16) {
                            a = ggml_fp16_to_fp32(*(const ggml_fp16_t *)(s0_row + k*sizeof(ggml_fp16_t)));
                        } else {
                            a = *(const float *)(s0_row + k*sizeof(float));
                        }

                        const float b = *(const float *)(s1_row + k*sizeof(float));
                        sum += a * b;
                    }

                    float * d = (float *)(d_base + i0*node->nb[0] + i1*nb1);
                    *d = sum;
                }
            }
        }
    }

    return true;
}

// ----------------------------------------------------------------------------
// ADD — Element-wise addition with broadcasting
// ----------------------------------------------------------------------------

static bool cpu_add_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ADD) {
        return false;
    }
    if (op->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[1] == nullptr || op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    return true;
}

static bool cpu_add_f32_execute(struct ggml_backend_triton_context * /*ctx*/,
                                const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        return false;
    }

    // Iterate over dst dimensions, use modulo for broadcasting src1
    const int64_t ne0 = node->ne[0];
    const int64_t ne1 = node->ne[1];
    const int64_t ne2 = node->ne[2];
    const int64_t ne3 = node->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];
    const int64_t nb1  = node->nb[1];
    const int64_t nb2  = node->nb[2];
    const int64_t nb3  = node->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                // Broadcast indices for src1
                const int64_t i13 = i3 % ne13;
                const int64_t i12 = i2 % ne12;
                const int64_t i11 = i1 % ne11;

                const float * s0_row = (const float *)((const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                const float * s1_row = (const float *)((const char *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                float *       d_row  = (float *)      ((char *)node->data   + i1*nb1  + i2*nb2  + i3*nb3);

                if (ne10 == ne0) {
                    // No broadcasting along dim0
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        d_row[i0] = s0_row[i0] + s1_row[i0];
                    }
                } else {
                    // src1 broadcasts along dim0
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        const float * s1_el = (const float *)((const char *)src1->data + i11*nb11 + i12*nb12 + i13*nb13 + (i0 % ne10)*nb10);
                        d_row[i0] = s0_row[i0] + *s1_el;
                    }
                }
            }
        }
    }

    return true;
}

// ----------------------------------------------------------------------------
// MUL — Element-wise multiplication with broadcasting
// ----------------------------------------------------------------------------

static bool cpu_mul_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL) {
        return false;
    }
    if (op->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[1] == nullptr || op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    return true;
}

static bool cpu_mul_f32_execute(struct ggml_backend_triton_context * /*ctx*/,
                                const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        return false;
    }

    const int64_t ne0 = node->ne[0];
    const int64_t ne1 = node->ne[1];
    const int64_t ne2 = node->ne[2];
    const int64_t ne3 = node->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];
    const int64_t nb1  = node->nb[1];
    const int64_t nb2  = node->nb[2];
    const int64_t nb3  = node->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t nb10 = src1->nb[0];

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const int64_t i13 = i3 % ne13;
                const int64_t i12 = i2 % ne12;
                const int64_t i11 = i1 % ne11;

                const float * s0_row = (const float *)((const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                float *       d_row  = (float *)      ((char *)node->data   + i1*nb1  + i2*nb2  + i3*nb3);

                if (ne10 == ne0) {
                    // No broadcasting along dim0
                    const float * s1_row = (const float *)((const char *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        d_row[i0] = s0_row[i0] * s1_row[i0];
                    }
                } else {
                    // src1 broadcasts along dim0
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        const float * s1_el = (const float *)((const char *)src1->data + i11*nb11 + i12*nb12 + i13*nb13 + (i0 % ne10)*nb10);
                        d_row[i0] = s0_row[i0] * *s1_el;
                    }
                }
            }
        }
    }

    return true;
}

// ----------------------------------------------------------------------------
// RMS_NORM — RMS Normalization
// rms = sqrt(mean(x^2) + eps), dst = x / rms
// ----------------------------------------------------------------------------

static bool cpu_rms_norm_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM) {
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

static bool cpu_rms_norm_f32_execute(struct ggml_backend_triton_context * /*ctx*/,
                                     const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];

    if (src0->data == nullptr || node->data == nullptr) {
        return false;
    }

    float eps;
    memcpy(&eps, node->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb1  = node->nb[1];
    const int64_t nb2  = node->nb[2];
    const int64_t nb3  = node->nb[3];

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const float * x = (const float *)((const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03);

                float sum_sq = 0.0f;
                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    sum_sq += x[i0] * x[i0];
                }

                const float mean = sum_sq / ne00;
                const float scale = 1.0f / sqrtf(mean + eps);

                float * y = (float *)((char *)node->data + i1*nb1 + i2*nb2 + i3*nb3);

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    y[i0] = x[i0] * scale;
                }
            }
        }
    }

    return true;
}

// ----------------------------------------------------------------------------
// ROPE — Rotary Position Embedding
// Support mode 0 (GGML_ROPE_TYPE_NORMAL) for F32
// ----------------------------------------------------------------------------

static bool cpu_rope_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ROPE) {
        return false;
    }
    if (op->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    // Only support standard rope mode for now
    const int mode = ((const int32_t *)op->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL) {
        return false;
    }
    return true;
}

static bool cpu_rope_f32_execute(struct ggml_backend_triton_context * /*ctx*/,
                                 const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1]; // position indices (int32)

    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        return false;
    }

    // Read op_params
    // op_params layout: [n_past, n_dims, mode, n_ctx, n_ctx_orig,
    //                    freq_base(f32), freq_scale(f32), ext_factor(f32),
    //                    attn_factor(f32), beta_fast(f32), beta_slow(f32)]
    const int n_dims     = ((const int32_t *)node->op_params)[1];

    float freq_base, freq_scale;
    memcpy(&freq_base,  (const int32_t *)node->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (const int32_t *)node->op_params + 6, sizeof(float));

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb1  = node->nb[1];
    const int64_t nb2  = node->nb[2];
    const int64_t nb3  = node->nb[3];

    const int32_t * pos = (const int32_t *)src1->data;

    // theta_scale = freq_base^(-2/n_dims)
    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    // For GGML_ROPE_TYPE_NORMAL (mode 0):
    // Pairs are consecutive: (x[0], x[1]), (x[2], x[3]), ...
    // For each pair (x, y) at dimension i:
    //   theta = pos * freq_base^(-2i/n_dims) * freq_scale
    //   x' = x*cos(theta) - y*sin(theta)
    //   y' = x*sin(theta) + y*cos(theta)

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = pos[i2]; // position for this sequence position

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const float * src = (const float *)((const char *)src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                float *       dst = (float *)      ((char *)node->data   + i3*nb3  + i2*nb2  + i1*nb1);

                float cur_theta = (float)p * freq_scale;

                // Apply rotation to pairs within n_dims
                for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                    const float cos_theta = cosf(cur_theta);
                    const float sin_theta = sinf(cur_theta);

                    const float x0 = src[i0 + 0];
                    const float x1 = src[i0 + 1];

                    dst[i0 + 0] = x0 * cos_theta - x1 * sin_theta;
                    dst[i0 + 1] = x0 * sin_theta + x1 * cos_theta;

                    cur_theta *= theta_scale;
                }

                // Copy remaining dimensions unchanged
                for (int64_t i0 = n_dims; i0 < ne0; i0++) {
                    dst[i0] = src[i0];
                }
            }
        }
    }

    return true;
}

// ----------------------------------------------------------------------------
// SOFT_MAX — Softmax along ne0 dimension
// exp(x - max) / sum(exp(x - max))
// May have scale in op_params[0], mask in src1 (additive)
// ----------------------------------------------------------------------------

static bool cpu_soft_max_f32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_SOFT_MAX) {
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

static bool cpu_soft_max_f32_execute(struct ggml_backend_triton_context * /*ctx*/,
                                     const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1]; // mask (may be NULL)

    if (src0->data == nullptr || node->data == nullptr) {
        return false;
    }

    // Read scale from op_params
    float scale = 1.0f;
    memcpy(&scale, (const float *)node->op_params + 0, sizeof(float));

    const int64_t ne0 = node->ne[0];
    const int64_t ne1 = node->ne[1];
    const int64_t ne2 = node->ne[2];
    const int64_t ne3 = node->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb1  = node->nb[1];
    const int64_t nb2  = node->nb[2];
    const int64_t nb3  = node->nb[3];

    // Mask dimensions and strides
    const int64_t ne11 = src1 ? src1->ne[1] : 1;
    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;
    const int64_t nb11 = src1 ? src1->nb[1] : 0;
    const int64_t nb12 = src1 ? src1->nb[2] : 0;
    const int64_t nb13 = src1 ? src1->nb[3] : 0;
    const bool mask_f16 = src1 && src1->type == GGML_TYPE_F16;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const float * sp = (const float *)((const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                float *       dp = (float *)      ((char *)node->data   + i1*nb1  + i2*nb2  + i3*nb3);

                // Apply scale and optional mask, find max
                float max_val = -INFINITY;
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    float val = sp[i0] * scale;

                    // Additive mask
                    if (src1) {
                        const int64_t mi1 = i1 % ne11;
                        const int64_t mi2 = i2 % ne12;
                        const int64_t mi3 = i3 % ne13;
                        if (mask_f16) {
                            const ggml_fp16_t * mp = (const ggml_fp16_t *)((const char *)src1->data + mi1*nb11 + mi2*nb12 + mi3*nb13);
                            val += ggml_fp16_to_fp32(mp[i0]);
                        } else {
                            const float * mp = (const float *)((const char *)src1->data + mi1*nb11 + mi2*nb12 + mi3*nb13);
                            val += mp[i0];
                        }
                    }

                    dp[i0] = val;
                    if (val > max_val) {
                        max_val = val;
                    }
                }

                // Compute exp(x - max) and sum
                float sum = 0.0f;
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    dp[i0] = expf(dp[i0] - max_val);
                    sum += dp[i0];
                }

                // Normalize
                const float inv_sum = 1.0f / sum;
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    dp[i0] *= inv_sum;
                }
            }
        }
    }

    return true;
}

// ----------------------------------------------------------------------------
// Registration
// ----------------------------------------------------------------------------

void ggml_triton_register_cpu_providers(ggml_triton_op_registry & registry) {
    // GELU FP32
    registry.register_impl(GGML_OP_UNARY, {
        /* .name     = */ "cpu_gelu_f32",
        /* .provider = */ GGML_TRITON_PROVIDER_CPU,
        /* .supports = */ cpu_gelu_f32_supports,
        /* .execute  = */ cpu_gelu_f32_execute,
        /* .priority = */ 50,
    });

    // GELU FP16
    registry.register_impl(GGML_OP_UNARY, {
        /* .name     = */ "cpu_gelu_f16",
        /* .provider = */ GGML_TRITON_PROVIDER_CPU,
        /* .supports = */ cpu_gelu_f16_supports,
        /* .execute  = */ cpu_gelu_f16_execute,
        /* .priority = */ 50,
    });

    // SILU FP32
    registry.register_impl(GGML_OP_UNARY, {
        /* .name     = */ "cpu_silu_f32",
        /* .provider = */ GGML_TRITON_PROVIDER_CPU,
        /* .supports = */ cpu_silu_f32_supports,
        /* .execute  = */ cpu_silu_f32_execute,
        /* .priority = */ 50,
    });

    // SILU FP16
    registry.register_impl(GGML_OP_UNARY, {
        /* .name     = */ "cpu_silu_f16",
        /* .provider = */ GGML_TRITON_PROVIDER_CPU,
        /* .supports = */ cpu_silu_f16_supports,
        /* .execute  = */ cpu_silu_f16_execute,
        /* .priority = */ 50,
    });

    // MUL_MAT F32/F16×F32→F32
    registry.register_impl(GGML_OP_MUL_MAT, {
        /* .name     = */ "cpu_mul_mat_f32",
        /* .provider = */ GGML_TRITON_PROVIDER_CPU,
        /* .supports = */ cpu_mul_mat_f32_supports,
        /* .execute  = */ cpu_mul_mat_f32_execute,
        /* .priority = */ 50,
    });

    // ADD F32
    registry.register_impl(GGML_OP_ADD, {
        /* .name     = */ "cpu_add_f32",
        /* .provider = */ GGML_TRITON_PROVIDER_CPU,
        /* .supports = */ cpu_add_f32_supports,
        /* .execute  = */ cpu_add_f32_execute,
        /* .priority = */ 50,
    });

    // MUL F32
    registry.register_impl(GGML_OP_MUL, {
        /* .name     = */ "cpu_mul_f32",
        /* .provider = */ GGML_TRITON_PROVIDER_CPU,
        /* .supports = */ cpu_mul_f32_supports,
        /* .execute  = */ cpu_mul_f32_execute,
        /* .priority = */ 50,
    });

    // RMS_NORM F32
    registry.register_impl(GGML_OP_RMS_NORM, {
        /* .name     = */ "cpu_rms_norm_f32",
        /* .provider = */ GGML_TRITON_PROVIDER_CPU,
        /* .supports = */ cpu_rms_norm_f32_supports,
        /* .execute  = */ cpu_rms_norm_f32_execute,
        /* .priority = */ 50,
    });

    // ROPE F32 (mode 0 = NORMAL)
    registry.register_impl(GGML_OP_ROPE, {
        /* .name     = */ "cpu_rope_f32",
        /* .provider = */ GGML_TRITON_PROVIDER_CPU,
        /* .supports = */ cpu_rope_f32_supports,
        /* .execute  = */ cpu_rope_f32_execute,
        /* .priority = */ 50,
    });

    // SOFT_MAX F32
    registry.register_impl(GGML_OP_SOFT_MAX, {
        /* .name     = */ "cpu_soft_max_f32",
        /* .provider = */ GGML_TRITON_PROVIDER_CPU,
        /* .supports = */ cpu_soft_max_f32_supports,
        /* .execute  = */ cpu_soft_max_f32_execute,
        /* .priority = */ 50,
    });
}
