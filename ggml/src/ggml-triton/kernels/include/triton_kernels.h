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
// Pointer-slot counts per kernel family (B.3 onwards):
//   gelu, silu, rms_norm_unweighted  :  2 ptrs (in, out) + N
//   rms_norm_weighted                 :  3 ptrs (x, w, y) + N
//   rope_normal, rope_neox            :  2 ptrs (a, b) + 11 scalar args
//   rope_mrope                        :  3 ptrs (a, b, freq_factors) + 13 scalar args
//   flash_attn_prefill               :  4 ptrs (q, k, v, dst) + 8 ints + 1 float
//   flash_attn_decode                :  5 ptrs (q, k, v, dst, scratch) + 9 ints + 1 float

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
#include "rope_normal_fwd_yarnoff_fp16_sm80.h"
#include "rope_normal_fwd_yarnoff_fp32_sm80.h"
#include "rope_normal_fwd_yarnon_fp16_sm80.h"
#include "rope_normal_fwd_yarnon_fp32_sm80.h"
#include "rope_normal_bwd_yarnoff_fp16_sm80.h"
#include "rope_normal_bwd_yarnoff_fp32_sm80.h"
#include "rope_normal_bwd_yarnon_fp16_sm80.h"
#include "rope_normal_bwd_yarnon_fp32_sm80.h"
#include "rope_neox_fwd_yarnoff_fp16_sm80.h"
#include "rope_neox_fwd_yarnoff_fp32_sm80.h"
#include "rope_neox_fwd_yarnon_fp16_sm80.h"
#include "rope_neox_fwd_yarnon_fp32_sm80.h"
#include "rope_neox_bwd_yarnoff_fp16_sm80.h"
#include "rope_neox_bwd_yarnoff_fp32_sm80.h"
#include "rope_neox_bwd_yarnon_fp16_sm80.h"
#include "rope_neox_bwd_yarnon_fp32_sm80.h"
#include "rope_mrope_fwd_yarnoff_fp16_sm80.h"
#include "rope_mrope_fwd_yarnoff_fp32_sm80.h"
#include "rope_mrope_fwd_yarnon_fp16_sm80.h"
#include "rope_mrope_fwd_yarnon_fp32_sm80.h"
#include "rope_mrope_bwd_yarnoff_fp16_sm80.h"
#include "rope_mrope_bwd_yarnoff_fp32_sm80.h"
#include "rope_mrope_bwd_yarnon_fp16_sm80.h"
#include "rope_mrope_bwd_yarnon_fp32_sm80.h"
#include "flash_attn_prefill_hd64_fp16_sm80.h"
#include "flash_attn_prefill_hd64_fp32_sm80.h"
#include "flash_attn_prefill_hd96_fp16_sm80.h"
#include "flash_attn_prefill_hd96_fp32_sm80.h"
#include "flash_attn_prefill_hd128_fp16_sm80.h"
#include "flash_attn_prefill_hd128_fp32_sm80.h"
#include "flash_attn_decode_hd64_fp16_sm80.h"
#include "flash_attn_decode_hd64_fp32_sm80.h"
#include "flash_attn_decode_hd96_fp16_sm80.h"
#include "flash_attn_decode_hd96_fp32_sm80.h"
#include "flash_attn_decode_hd128_fp16_sm80.h"
#include "flash_attn_decode_hd128_fp32_sm80.h"

#ifdef __cplusplus
}
#endif
