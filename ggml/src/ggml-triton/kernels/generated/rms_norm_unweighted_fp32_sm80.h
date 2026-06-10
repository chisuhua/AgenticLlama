        #pragma once
        #include <cuda.h>
        #include <stdint.h>

        #ifdef __cplusplus
        extern "C" {
        #endif

        int triton_launch_rms_norm_unweighted_fp32_sm80(
            CUstream    stream,
CUdeviceptr d_x,
CUdeviceptr d_y,
int32_t     N);

        #ifdef __cplusplus
        }
        #endif
