"""Triton RMSNorm kernel for the ggml-triton backend (B.1).

Computes  y = x * rsqrt(mean(x*x) + eps) * weight   row-wise
        or  y = x * rsqrt(mean(x*x) + eps)             when USE_WEIGHT=0

The kernel is AOT-compiled by ``scripts/compile_kernels.py`` for the
``(dtype, arch)`` combinations declared in ``scripts/kernel_registry.json``.
Two kernel families share this Python source:

- ``rms_norm_unweighted``: USE_WEIGHT=0, launcher passes a dummy w_ptr (= 0)
  plus (N, num_blocks) as runtime args
- ``rms_norm_weighted``:   USE_WEIGHT=1, launcher passes (d_w, N, num_blocks)

Stage 2 retro-fix: N is now a runtime int32 (was constexpr in Stage 1) so the
host can pass ne[0] per call.  ``num_blocks`` is the total row count
(ne[1]*ne[2]*ne[3]) and is the C grid size (``grid_mode = "exact"``).  The
kernel uses ``pid * N`` to address the right row.  ``eps``, ``BLOCK_SIZE``,
and ``USE_WEIGHT`` stay ``tl.constexpr`` so the AOT launcher ABI is fixed.
"""

import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,                  # *T   — input
    y_ptr,                  # *T   — output
    w_ptr,                  # *T   — weight (only dereferenced when USE_WEIGHT=1;
                            #        host passes 0 in the unweighted variant
                            #        because Triton requires the slot to exist)
    N,                      # i32  — row length (runtime, was constexpr in Stage 1)
    num_blocks,             # i32  — total number of rows = ne[1]*ne[2]*ne[3]
    BLOCK_SIZE: tl.constexpr,  # i32  — power-of-two tile; Stage 1: BLOCK_SIZE >= N
    EPS: tl.constexpr,      # fp32 — epsilon
    USE_WEIGHT: tl.constexpr,  # i1  — when 0, skip the w load and the weight mul
):
    # One program per row.  Row is at program_id(0).
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = pid * N + col_offsets
    mask = col_offsets < N

    # Load x in fp32 for accumulation regardless of input dtype.  Other=0.0
    # so masked-off elements contribute 0 to the sum-of-squares.
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)

    # sum-of-squares; BLOCK_SIZE >= N guarantees all real elements are
    # in the first N lanes and the masked tail is 0.
    sum_sq = tl.sum(x * x, axis=0)
    mean = sum_sq / N
    scale = 1.0 / tl.sqrt(mean + EPS)

    if USE_WEIGHT:
        w = tl.load(w_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x * scale * w).to(y_ptr.dtype.element_ty)
    else:
        y = (x * scale).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row_offsets, y, mask=mask)
