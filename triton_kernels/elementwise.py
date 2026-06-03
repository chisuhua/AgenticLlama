"""Triton elementwise kernels used by the ggml-triton backend.

These kernels are compiled ahead-of-time by ``scripts/compile_kernels.py`` for
the dtype/arch combinations declared in ``scripts/kernel_registry.json``. The
resulting CUBIN is embedded into a generated ``.c`` launcher under
``ggml/src/ggml-triton/kernels/generated``.
"""

import triton
import triton.language as tl


@triton.jit
def gelu_kernel(input_ptr, output_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input_ptr + offsets, mask=mask)
    # GELU tanh approximation (matches ggml CUDA reference).
    out = 0.5 * x * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    tl.store(output_ptr + offsets, out, mask=mask)


@triton.jit
def silu_kernel(input_ptr, output_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input_ptr + offsets, mask=mask)
    out = x * tl.sigmoid(x)
    tl.store(output_ptr + offsets, out, mask=mask)
