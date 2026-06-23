        #pragma once
        #include <cuda.h>
        #include <stdint.h>

        #ifdef __cplusplus
        extern "C" {
        #endif

        int triton_launch_rms_norm_weighted_fp32_sm80(
            CUstream    stream,
CUdeviceptr d_x,
CUdeviceptr d_y,
CUdeviceptr d_w,
int32_t     N,
int32_t     num_blocks);

        #ifdef __cplusplus
        }
        #endif
