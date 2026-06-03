"""TileLang element-wise kernel definitions used by the ggml-tilelang backend.

These functions return TileLang `prim_func`s that can be lowered to CUDA source
by the AOT pipeline driven from `scripts/compile_kernels.py`. The skeleton ships
a checked-in `kernels/generated/` directory so the C++ side can be built without
actually invoking TileLang.
"""

import tilelang as tl  # noqa: F401  (re-exported by some TileLang builds)
import tilelang.language as T


def make_add_kernel(N, BLOCK_SIZE=1024):
    @T.prim_func
    def add_kernel(
        A: T.Buffer((N,), "float16"),
        B: T.Buffer((N,), "float16"),
        C: T.Buffer((N,), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_SIZE), threads=BLOCK_SIZE) as bx:
            for i in T.vectorized(BLOCK_SIZE):
                idx = bx * BLOCK_SIZE + i
                if idx < N:
                    C[idx] = A[idx] + B[idx]

    return add_kernel


def make_mul_kernel(N, BLOCK_SIZE=1024):
    @T.prim_func
    def mul_kernel(
        A: T.Buffer((N,), "float16"),
        B: T.Buffer((N,), "float16"),
        C: T.Buffer((N,), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_SIZE), threads=BLOCK_SIZE) as bx:
            for i in T.vectorized(BLOCK_SIZE):
                idx = bx * BLOCK_SIZE + i
                if idx < N:
                    C[idx] = A[idx] * B[idx]

    return mul_kernel


def make_add_kernel_fp32(N, BLOCK_SIZE=1024):
    @T.prim_func
    def add_kernel_fp32(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_SIZE), threads=BLOCK_SIZE) as bx:
            for i in T.vectorized(BLOCK_SIZE):
                idx = bx * BLOCK_SIZE + i
                if idx < N:
                    C[idx] = A[idx] + B[idx]

    return add_kernel_fp32


def make_mul_kernel_fp32(N, BLOCK_SIZE=1024):
    @T.prim_func
    def mul_kernel_fp32(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_SIZE), threads=BLOCK_SIZE) as bx:
            for i in T.vectorized(BLOCK_SIZE):
                idx = bx * BLOCK_SIZE + i
                if idx < N:
                    C[idx] = A[idx] * B[idx]

    return mul_kernel_fp32
