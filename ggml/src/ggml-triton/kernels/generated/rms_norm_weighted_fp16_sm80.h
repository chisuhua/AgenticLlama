        #pragma once
        #include <cuda.h>
        #include <stdint.h>

        #ifdef __cplusplus
        extern "C" {
        #endif

        int triton_launch_rms_norm_weighted_fp16_sm80(
            CUstream    stream,
CUdeviceptr d_x,
CUdeviceptr d_w,
CUdeviceptr d_y,
int32_t     N);

        #ifdef __cplusplus
        }
        #endif
