#pragma once

// Aggregated header that pulls in every AOT-generated kernel launcher.
//
// Each generated file declares a launcher of the form:
//   int triton_launch_<kernel>_<dtype>_<arch>(CUstream stream,
//                                             CUdeviceptr ...,
//                                             int32_t N);
//
// The launchers wrap a cuModuleLoadData / cuModuleGetFunction / cuLaunchKernel
// triple where the CUBIN payload was produced by Triton AOT compilation.
//
// Pointer-slot counts per kernel family (B.1 onwards):
//   gelu, silu, rms_norm_unweighted  :  2 ptrs (in, out) + N
//   rms_norm_weighted                 :  3 ptrs (x, w, y) + N

#ifdef __cplusplus
extern "C" {
#endif

#include "gelu_fp16_sm80.h"
#include "gelu_fp32_sm80.h"
#include "silu_fp16_sm80.h"
#include "silu_fp32_sm80.h"
#include "rms_norm_unweighted_fp16_sm80.h"
#include "rms_norm_unweighted_fp32_sm80.h"
#include "rms_norm_weighted_fp16_sm80.h"
#include "rms_norm_weighted_fp32_sm80.h"

#ifdef __cplusplus
}
#endif
