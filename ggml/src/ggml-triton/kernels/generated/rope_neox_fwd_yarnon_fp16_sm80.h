        #pragma once
        #include <cuda.h>
        #include <stdint.h>

        #ifdef __cplusplus
        extern "C" {
        #endif

        int triton_launch_rope_neox_fwd_yarnon_fp16_sm80(
            CUstream    stream,
CUdeviceptr a,
CUdeviceptr b,
int32_t     n_dims,
int32_t     n_ctx_orig,
float       freq_base,
float       freq_scale,
float       ext_factor,
float       attn_factor,
float       beta_fast,
float       beta_slow,
float       corr_low,
float       corr_high,
int32_t     rows);

        #ifdef __cplusplus
        }
        #endif
