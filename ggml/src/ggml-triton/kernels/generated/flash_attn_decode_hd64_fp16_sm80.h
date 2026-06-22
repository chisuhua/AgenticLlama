        #pragma once
        #include <cuda.h>
        #include <stdint.h>

        #ifdef __cplusplus
        extern "C" {
        #endif

        int triton_launch_flash_attn_decode_hd64_fp16_sm80(
            CUstream    stream,
CUdeviceptr q,
CUdeviceptr k,
CUdeviceptr v,
CUdeviceptr dst,
CUdeviceptr scratch,
int32_t     neq1,
int32_t     neq2,
int32_t     neq3,
int32_t     nek1,
int32_t     S,
int32_t     n_heads,
int32_t     q_pos,
int32_t     num_kv_chunks,
int32_t     rows,
float       scale);

        #ifdef __cplusplus
        }
        #endif
