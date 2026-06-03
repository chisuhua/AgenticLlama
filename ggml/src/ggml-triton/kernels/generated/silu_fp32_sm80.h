#pragma once
#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int triton_launch_silu_fp32_sm80(CUstream stream,
                         CUdeviceptr d_in,
                         CUdeviceptr d_out,
                         int32_t     N);

#ifdef __cplusplus
}
#endif
