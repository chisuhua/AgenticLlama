        #pragma once
        #include <cuda.h>
        #include <stdint.h>

        #ifdef __cplusplus
        extern "C" {
        #endif

        int triton_launch_flash_attn_prefill_hd128_fp16_sm80(
            CUstream    stream,
CUdeviceptr q,
CUdeviceptr k,
CUdeviceptr v,
CUdeviceptr dst,
int32_t     neq1,
int32_t     neq2,
int32_t     neq3,
int32_t     nek1,
int32_t     S,
int32_t     n_heads,
int32_t     rows,
int32_t     num_q_blocks,
float       scale);

        #ifdef __cplusplus
        }
        #endif
