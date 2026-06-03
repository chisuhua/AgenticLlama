#pragma once

// Aggregated header that pulls in every AOT-generated kernel launcher.
//
// Each generated file declares a launcher of the form:
//   int triton_launch_<kernel>_<dtype>_<arch>(CUstream stream,
//                                             CUdeviceptr in,
//                                             CUdeviceptr out,
//                                             int32_t N);
//
// The launchers wrap a cuModuleLoadData / cuModuleGetFunction / cuLaunchKernel
// triple where the CUBIN payload was produced by Triton AOT compilation.

#ifdef __cplusplus
extern "C" {
#endif

#include "gelu_fp16_sm80.h"
#include "gelu_fp32_sm80.h"
#include "silu_fp16_sm80.h"
#include "silu_fp32_sm80.h"

#ifdef __cplusplus
}
#endif
