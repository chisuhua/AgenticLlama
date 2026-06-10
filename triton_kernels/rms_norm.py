"""Triton RMSNorm kernel for the ggml-triton backend (B.1).

Computes  y = x * rsqrt(mean(x*x) + eps) * weight   row-wise
        or  y = x * rsqrt(mean(x*x) + eps)             when USE_WEIGHT=0

The kernel is AOT-compiled by ``scripts/compile_kernels.py`` for the
``(dtype, arch)`` combinations declared in ``scripts/kernel_registry.json``.
Two kernel families share this Python source:

- ``rms_norm_unweighted``: USE_WEIGHT=0, launcher takes (x_ptr, y_ptr, N)
- ``rms_norm_weighted``:   USE_WEIGHT=1, launcher takes (x_ptr, w_ptr, y_ptr, N)

Stage 1 design: ``eps``, ``BLOCK_SIZE``, and ``USE_WEIGHT`` are all
``tl.constexpr`` so the AOT launcher signature is fixed at AOT-compile
time and matches the GELU/SiLU launcher's parameter slot count for
the unweighted variant (2 ptrs + N). Stage 2 will promote ``eps`` to
a runtime float argument and extend ``compile_kernels.py``'s launcher
emitter.
"""

import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,                  # *T   — input
    y_ptr,                  # *T   — output
    w_ptr,                  # *T   — weight (only dereferenced when USE_WEIGHT=1;
                            #        pass a dummy pointer in the unweighted variant
                            #        because Triton requires the slot to exist)
    N: tl.constexpr,        # i32  — row length (constexpr so BLOCK_SIZE >= N)
    BLOCK_SIZE: tl.constexpr,  # i32  — power-of-two tile (Stage 1: BLOCK_SIZE >= N)
    EPS: tl.constexpr,      # fp32 — epsilon (constexpr in Stage 1)
    USE_WEIGHT: tl.constexpr,  # i1  — when 0, skip the w load and the weight mul
):
    # Single program per row.  Row is at program_id(0).
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load x in fp32 for accumulation regardless of input dtype.  Other=0.0
    # so masked-off elements contribute 0 to the sum-of-squares.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # sum-of-squares; BLOCK_SIZE >= N guarantees all real elements are
    # in the first N lanes and the masked tail is 0.
    sum_sq = tl.sum(x * x, axis=0)
    mean = sum_sq / N
    scale = 1.0 / tl.sqrt(mean + EPS)

    if USE_WEIGHT:
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x * scale * w).to(y_ptr.dtype.element_ty)
    else:
        y = (x * scale).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offsets, y, mask=mask)
