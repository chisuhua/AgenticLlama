#!/usr/bin/env python3
"""Driver for the Triton AOT compilation step used by ggml-triton.

The script reads ``kernel_registry.json``, imports each kernel listed in it
(see ``triton_kernels/elementwise.py``) and asks Triton to compile a CUBIN per
``(dtype, arch)`` variant. Each compiled artifact is then emitted as a small C
file under ``ggml/src/ggml-triton/kernels/generated`` containing:

  * the CUBIN bytes embedded as a ``static const unsigned char`` array, and
  * a ``triton_launch_<kernel>_<dtype>_<arch>`` launcher implemented with the
    CUDA Driver API (``cuModuleLoadData`` + ``cuLaunchKernel``).

The shape of the generated files matches the hand-written placeholders that
ship in the repository, so the C/C++ side never sees a build break even when
Triton is not installed.

Triton is imported lazily so that builds without Triton can still re-run this
script and produce the placeholder layout (handy for CI on machines without
GPUs).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    dtype: str
    arch:  str
    specialise: Dict[str, Any]
    signature: str

    @property
    def tag(self) -> str:
        # Fold the constexpr-specialised axes that the AOT driver cares about
        # into the filename so each variant emits its own .c/.h pair.
        #
        # Recognised axes (all optional; only present axes are folded):
        #   * HEAD_DIM  (B.3) — folded as "hd{N}" prefix; ensures 3 head_dim
        #                        variants of the same kernel/dtype do not
        #                        collide on filename.
        #   * SIN_SIGN  (B.2) — folded as "fwd" (>0) or "bwd" (<=0).
        #   * YA_ON     (B.2) — folded as "yarnon" (!=0) or "yarnoff" (==0).
        #
        # Variants that do not specialise any of these axes (GELU/SiLU,
        # RMSNorm) keep the pre-B.2 "<dtype>_<arch>" filename to preserve
        # byte-compat with the existing committed launchers.
        parts = []
        if "HEAD_DIM" in self.specialise:
            parts.append(f"hd{int(self.specialise['HEAD_DIM'])}")
        if "SIN_SIGN" in self.specialise:
            parts.append("fwd" if int(self.specialise["SIN_SIGN"]) > 0 else "bwd")
        if "YA_ON" in self.specialise:
            parts.append("yarnon" if int(self.specialise["YA_ON"]) != 0 else "yarnoff")
        parts.append(self.dtype)
        parts.append(self.arch)
        return "_".join(parts)


@dataclass
class Kernel:
    name: str
    module: str
    function: str
    variants: List[Variant]


def _parse_registry(path: Path) -> List[Kernel]:
    data = json.loads(path.read_text())
    kernels: List[Kernel] = []
    for entry in data.get("kernels", []):
        variants = [
            Variant(
                dtype=v["dtype"],
                arch=v["arch"],
                specialise=v.get("specialise", {}),
                signature=v.get("signature", ""),
            )
            for v in entry.get("variants", [])
        ]
        kernels.append(
            Kernel(
                name=entry["name"],
                module=entry["module"],
                function=entry["function"],
                variants=variants,
            )
        )
    return kernels


# ---------------------------------------------------------------------------
# CUBIN compilation
# ---------------------------------------------------------------------------

def _compile_with_triton(kernel: Kernel, variant: Variant) -> Optional[bytes]:
    """Try to AOT-compile the kernel with Triton; return the CUBIN bytes.

    On any failure (Triton missing, no GPU, signature error, ...) returns
    ``None`` so the caller can fall back to a placeholder.

    This implements the canonical Triton 3.7.0 AOT pattern, modeled on
    ``triton/tools/compile.py``. The key differences from the older 2.x API:

    * We build an ``ASTSource`` from the loaded ``JITFunction`` plus a parsed
      signature + constants dict, rather than passing the JITFunction directly.
    * We pass a ``GPUTarget(backend, arch, warp_size)`` and an ``options`` dict
      that the backend has already validated via ``backend.parse_options(...)``.

    See ``triton/tools/compile.py`` in the installed Triton distribution for
    the upstream reference.
    """
    try:
        triton = importlib.import_module("triton")
    except Exception as exc:  # pragma: no cover - exercised on machines w/o triton
        print(f"[triton-aot] triton not available: {exc}", file=sys.stderr)
        return None

    try:
        mod = importlib.import_module(kernel.module)
        fn = getattr(mod, kernel.function)
    except Exception as exc:
        print(f"[triton-aot] failed to import {kernel.module}.{kernel.function}: {exc}",
              file=sys.stderr)
        return None

    # Parse "sm80" -> arch 80.
    arch_num = None
    if variant.arch.startswith("sm"):
        try:
            arch_num = int(variant.arch[2:])
        except ValueError:
            arch_num = None
    if arch_num is None:
        print(f"[triton-aot] unrecognized arch tag {variant.arch!r} "
              f"(expected 'sm<num>'); skipping {kernel.name}/{variant.tag}",
              file=sys.stderr)
        return None

    # Parse the signature string ("*fp16,*fp16,i32,1024") into the dict form
    # that ASTSource expects. Each comma-separated entry is either:
    #   * a type with an optional divisibility hint  ("*fp16", "*fp16:16")
    #   * or a constexpr value                        ("1024")
    # The kernel arg names come from the JITFunction we just loaded.
    def _parse_constexpr(s: str):
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return None

    try:
        sig_parts = [p.strip() for p in variant.signature.split(",")]
        hints = {}
        for i, s in enumerate(sig_parts):
            if ":" in s:
                base, hint = s.split(":", 1)
                h = _parse_constexpr(hint)
                if h is not None:
                    hints[(i,)] = h
                sig_parts[i] = base

        # Constants: any non-pointer entries that look like numbers.
        constants = {}
        for i, s in enumerate(sig_parts):
            if s.startswith("*"):
                continue
            v = _parse_constexpr(s)
            if v is not None:
                constants[fn.arg_names[i]] = v

        # Mark divisibility-1 hints as constants too.
        for (i,), h in hints.items():
            if h == 1:
                constants[fn.arg_names[i]] = h

        # The signature dict maps arg_name -> type-string. constexprs get
        # replaced by the string 'constexpr'.
        sig_dict = {fn.arg_names[i]: s for i, s in enumerate(sig_parts)}
        for k in constants:
            sig_dict[k] = "constexpr"

        # Attrs for divisibility hints (only the "16" hint is honoured by
        # the CUDA backend today).
        attrs = {fn.arg_names[i]: [["tt.divisibility", 16]]
                 for (i,), h in hints.items() if h == 16}

        # Make sure the JITFunction has its binder built (needed for ASTSource
        # to introspect arg names in 3.7+).
        try:
            fn.create_binder()
        except RuntimeError as exc:
            # The 3.7+ JIT runtime calls `driver.active.get_current_target()`
            # inside create_binder(), which requires a GPU driver to be
            # loaded. On a machine with no NVIDIA driver (e.g. a CI runner
            # or a Triton-AOT dev box), this raises
            #     RuntimeError: 0 active drivers ([]). There should only be one.
            # In that case AOT is not possible here — bail out so the caller
            # can fall back to the placeholder CUBIN path.
            if "active drivers" in str(exc):
                print(f"[triton-aot] no GPU driver available on this host "
                      f"(triton.runtime.driver.active is empty); "
                      f"falling back to placeholder CUBIN for "
                      f"{kernel.name}/{variant.tag}", file=sys.stderr)
                return None
            raise

        src = fn.ASTSource(fn=fn, constexprs=constants,
                           signature=sig_dict, attrs=attrs)

        from triton.compiler.compiler import GPUTarget
        target = GPUTarget("cuda", arch_num, 32)

        from triton.compiler import make_backend
        backend = make_backend(target)
        # num_warps comes from the kernel_registry default; the constexpr
        # BLOCK_SIZE (etc.) is encoded in `constants` already.
        num_warps = int(variant.specialise.get("NUM_WARPS", 4))
        parsed = backend.parse_options({"num_warps": num_warps, "num_stages": 1})
        options = parsed.__dict__

        compiled = triton.compile(src, target=target, options=options)
    except Exception as exc:
        print(f"[triton-aot] triton.compile failed for "
              f"{kernel.name}/{variant.tag}: {exc}", file=sys.stderr)
        return None

    # The 3.7+ API exposes the cubin via ccinfo.asm[backend.binary_ext] where
    # binary_ext is "cubin" for the CUDA backend. Look it up on the backend
    # instance (3.7 stores it as an instance attribute, not class attr).
    asm = getattr(compiled, "asm", {}) or {}
    bin_ext = getattr(backend, "binary_ext", "cubin")
    cubin = asm.get(bin_ext) or asm.get("cubin")
    if cubin is None:
        print(f"[triton-aot] no cubin produced for {kernel.name}/{variant.tag} "
              f"(asm keys: {list(asm.keys())})", file=sys.stderr)
        return None
    if isinstance(cubin, str):
        cubin = cubin.encode("latin-1")
    return bytes(cubin)


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------

_PLACEHOLDER_CUBIN = bytes([
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
])


# Per-kernel launcher-arg layout.  The emitter uses the value to pick the
# header signature, the function-body parameter list, and the args[] array.
# Each shape is a dict with two keys:
#   * "params":     list of (c_type, c_name) pairs in source order
#                   (after the CUstream stream parameter that
#                   _format_params_lines prepends).
#   * "grid_param": name of the param used in the grid-size calculation
#                   "(grid = (param + block - 1) / block)".  Elementwise
#                   and RMSNorm use "N" (one program per element-block);
#                   RoPE uses "rows" (one program per row, where each
#                   program processes a full n_dims vector).
#
# GELU/SILU use "default" — the unchanged elementwise (in, out, N) ABI
# that pre-existed PR #1.
LAUNCHER_SHAPES = {
    "default": {
        "params": [
            ("CUdeviceptr", "d_in"),
            ("CUdeviceptr", "d_out"),
            ("int32_t",     "N"),
        ],
        "grid_param": "N",
    },
    # B.1 RMSNorm unweighted: y = x * rsqrt(mean(x*x) + eps)
    # Stage 2 retro-fix: d_w is a dummy (host passes 0); grid_mode="exact"
    # so the C grid equals the host-computed num_blocks (= ne[1]*ne[2]*ne[3]),
    # not (N + block - 1) / block.
    "rms_norm_unweighted": {
        "params": [
            ("CUdeviceptr", "d_x"),
            ("CUdeviceptr", "d_y"),
            ("CUdeviceptr", "d_w"),
            ("int32_t",     "N"),
            ("int32_t",     "num_blocks"),
        ],
        "grid_param": "num_blocks",
        "grid_mode":  "exact",
    },
    # B.1 RMSNorm weighted: y = x * rsqrt(mean(x*x) + eps) * w
    "rms_norm_weighted": {
        "params": [
            ("CUdeviceptr", "d_x"),
            ("CUdeviceptr", "d_w"),
            ("CUdeviceptr", "d_y"),
            ("int32_t",     "N"),
            ("int32_t",     "num_blocks"),
        ],
        "grid_param": "num_blocks",
        "grid_mode":  "exact",
    },
    # B.2 RoPE shapes.  Per-mode ABI (counted after the stream param that
    # _format_params_lines prepends), and a trailing "rows" runtime arg
    # (the number of rows in the input tensor = ne[1]*ne[2]*ne[3]):
    #   rope_normal/rope_neox: 2 ptrs + 10 scalar args + 1 rows = 13 args
    #   rope_mrope:             3 ptrs + 14 scalar args + 1 rows = 18 args
    # The C++ provider passes `rows = src0->ne[1]*src0->ne[2]*src0->ne[3]`
    # as the last arg; the grid is then `(rows + block - 1) / block`,
    # i.e. one Triton program per row.
    # See commit b815d418b — earlier draft had 18 runtime args
    # (off-by-one), corrected to 17; this revision adds 1 (the rows
    # arg) on top, taking rope_normal/rope_neox to 13 and rope_mrope
    # to 18 runtime args.
    "rope_normal": {
        "params": [
            ("CUdeviceptr", "a"),
            ("CUdeviceptr", "b"),
            ("int32_t",     "n_dims"),
            ("int32_t",     "n_ctx_orig"),
            ("float",       "freq_base"),
            ("float",       "freq_scale"),
            ("float",       "ext_factor"),
            ("float",       "attn_factor"),
            ("float",       "beta_fast"),
            ("float",       "beta_slow"),
            ("float",       "corr_low"),
            ("float",       "corr_high"),
            ("int32_t",     "rows"),
        ],
        "grid_param": "rows",
        "grid_mode":  "exact",
    },
    "rope_neox": {
        "params": [
            ("CUdeviceptr", "a"),
            ("CUdeviceptr", "b"),
            ("int32_t",     "n_dims"),
            ("int32_t",     "n_ctx_orig"),
            ("float",       "freq_base"),
            ("float",       "freq_scale"),
            ("float",       "ext_factor"),
            ("float",       "attn_factor"),
            ("float",       "beta_fast"),
            ("float",       "beta_slow"),
            ("float",       "corr_low"),
            ("float",       "corr_high"),
            ("int32_t",     "rows"),
        ],
        "grid_param": "rows",
        "grid_mode":  "exact",
    },
    "rope_mrope": {
        "params": [
            ("CUdeviceptr", "a"),
            ("CUdeviceptr", "b"),
            ("CUdeviceptr", "freq_factors"),
            ("int32_t",     "n_dims"),
            ("int32_t",     "n_ctx_orig"),
            ("float",       "freq_base"),
            ("float",       "freq_scale"),
            ("float",       "ext_factor"),
            ("float",       "attn_factor"),
            ("float",       "beta_fast"),
            ("float",       "beta_slow"),
            ("int32_t",     "sect_t"),
            ("int32_t",     "sect_h"),
            ("int32_t",     "sect_w"),
            ("int32_t",     "sect_e"),
            ("float",       "corr_low"),
            ("float",       "corr_high"),
            ("int32_t",     "rows"),
        ],
        "grid_param": "rows",
        "grid_mode":  "exact",
    },
    # B.3 FlashAttn shapes (2 kernels x per-kernel ABI per Oracle #1 fix).
    #   prefill: 4 ptrs (q, k, v, dst) + 8 ints (neq1, neq2, neq3, nek1, S,
    #            n_heads, rows, num_q_blocks) + 1 float (scale)
    #            = 13 args after stream.
    #   decode:  5 ptrs (q, k, v, dst, scratch) + 9 ints (neq1=1, neq2, neq3,
    #            nek1, S, n_heads, q_pos, num_kv_chunks, rows) + 1 float
    #            (scale) = 15 args after stream.
    # Grid is 2D: program_id(0) = q_block (prefill) or kv_chunk (decode),
    # program_id(1) = head/batch row.  Both axes use "exact" mode (host
    # pre-computes the per-tile count) — Oracle #5 fix.  Per Oracle #1 fix,
    # 2D grid requires both grid_param + grid_param_y on the shape.
    "flash_attn_prefill": {
        "params": [
            ("CUdeviceptr", "q"),
            ("CUdeviceptr", "k"),
            ("CUdeviceptr", "v"),
            ("CUdeviceptr", "dst"),
            ("int32_t",     "neq1"),
            ("int32_t",     "neq2"),
            ("int32_t",     "neq3"),
            ("int32_t",     "nek1"),
            ("int32_t",     "S"),
            ("int32_t",     "n_heads"),
            ("int32_t",     "rows"),
            ("int32_t",     "num_q_blocks"),
            ("float",       "scale"),
        ],
        "grid_param":   "num_q_blocks",
        "grid_mode":    "exact",
        "grid_param_y": "rows",
        "grid_mode_y":  "exact",
    },
    "flash_attn_decode": {
        "params": [
            ("CUdeviceptr", "q"),
            ("CUdeviceptr", "k"),
            ("CUdeviceptr", "v"),
            ("CUdeviceptr", "dst"),
            ("CUdeviceptr", "scratch"),
            ("int32_t",     "neq1"),
            ("int32_t",     "neq2"),
            ("int32_t",     "neq3"),
            ("int32_t",     "nek1"),
            ("int32_t",     "S"),
            ("int32_t",     "n_heads"),
            ("int32_t",     "q_pos"),
            ("int32_t",     "num_kv_chunks"),
            ("int32_t",     "rows"),
            ("float",       "scale"),
        ],
        "grid_param":   "num_kv_chunks",
        "grid_mode":    "exact",
        "grid_param_y": "rows",
        "grid_mode_y":  "exact",
    },
}


def _shape_for(kernel_name: str):
    return LAUNCHER_SHAPES.get(kernel_name, LAUNCHER_SHAPES["default"])


def _format_params_lines(shape, include_cu_stream: bool = True) -> str:
    """Render the function-parameter block as a multi-line string."""
    lines = []
    if include_cu_stream:
        lines.append("CUstream    stream")
    for c_type, c_name in shape["params"]:
        lines.append(f"{c_type:11s} {c_name}")
    return ",\n    ".join(lines)


def _format_arg_addrs(shape) -> str:
    return ", ".join(f"(void *) &{c_name}" for _c_type, c_name in shape["params"])


def _format_byte_array(blob: bytes, indent: str = "    ", per_line: int = 12) -> str:
    parts = []
    for i in range(0, len(blob), per_line):
        chunk = blob[i:i + per_line]
        parts.append(indent + ", ".join(f"0x{b:02x}" for b in chunk))
    return ",\n".join(parts)


def _format_grid_expr(shape: dict, axis: str = "x") -> str:
    """Render the C expression for the grid size on the given axis.

    Modes (per Oracle #5 fix):
      * "divide" (default): ceil(grid_param / divisor).  divisor is
        shape["grid_divisor"] if set, else falls back to the literal
        C identifier "block" (the kTritonBlockSize_<name> local in the
        generated launcher).  This matches the pre-B.3 behaviour for
        B.1/B.2 entries exactly — backward compat preserved.
      * "exact":  use grid_param as-is (host has already pre-computed
        the per-tile count).  Used by B.3 FlashAttn entries.

    axis="x" reads shape["grid_param"] / shape["grid_mode"];
    axis="y" reads shape["grid_param_y"] / shape["grid_mode_y"]
    (defaulting grid_mode_y to the x-axis mode for symmetry).
    """
    if axis == "y":
        param = shape.get("grid_param_y")
        if param is None:
            raise ValueError(f"axis='y' but no grid_param_y in shape: {shape!r}")
        mode = shape.get("grid_mode_y", shape.get("grid_mode", "divide"))
    else:
        param = shape.get("grid_param")
        if param is None:
            raise ValueError(f"no grid_param in shape: {shape!r}")
        mode = shape.get("grid_mode", "divide")
    if mode == "exact":
        return f"(int)({param})"
    if mode == "divide":
        divisor = shape.get("grid_divisor")
        if divisor is None:
            return f"(int)(({param} + block - 1) / block)"
        return f"(int)(({param} + {divisor} - 1) / {divisor})"
    raise ValueError(f"Unknown grid_mode: {mode!r}")


def _emit_header(out_dir: Path, kernel: Kernel, variant: Variant) -> Path:
    name = f"{kernel.name}_{variant.tag}"
    # Byte-for-byte compatibility with the pre-B.1 GELU/SiLU headers for
    # the "default" (elementwise) shape: same multi-line alignment.
    if kernel.name in LAUNCHER_SHAPES and kernel.name != "default":
        shape = _shape_for(kernel.name)
        params_lines = _format_params_lines(shape, include_cu_stream=True)
        text = textwrap.dedent(f"""\
            #pragma once
            #include <cuda.h>
            #include <stdint.h>

            #ifdef __cplusplus
            extern "C" {{
            #endif

            int triton_launch_{name}(
                {params_lines});

            #ifdef __cplusplus
            }}
            #endif
            """)
    else:
        text = textwrap.dedent(f"""\
            #pragma once
            #include <cuda.h>
            #include <stdint.h>

            #ifdef __cplusplus
            extern "C" {{
            #endif

            int triton_launch_{name}(CUstream stream,
                                     CUdeviceptr d_in,
                                     CUdeviceptr d_out,
                                     int32_t     N);

            #ifdef __cplusplus
            }}
            #endif
            """)
    p = out_dir / f"{name}.h"
    p.write_text(text)
    return p


def _emit_source(out_dir: Path, kernel: Kernel, variant: Variant,
                 cubin: bytes, kernel_symbol: str, block_size: int) -> Path:
    name = f"{kernel.name}_{variant.tag}"
    cubin_array = _format_byte_array(cubin if cubin else _PLACEHOLDER_CUBIN)

    # Default (elementwise) shape: byte-for-byte compatible with pre-B.1
    # GELU/SiLU sources.  New shapes use the parameterised formatter.
    if kernel.name in LAUNCHER_SHAPES and kernel.name != "default":
        shape = _shape_for(kernel.name)
        params_lines = _format_params_lines(shape, include_cu_stream=True)
        arg_addrs = _format_arg_addrs(shape)
        # Per Oracle #1 fix: 2D grid when grid_param_y is present.
        # 1D path preserved byte-compat with B.1/B.2 entries (no grid_param_y).
        if "grid_param_y" in shape:
            grid_x_expr = _format_grid_expr(shape, "x")
            grid_y_expr = _format_grid_expr(shape, "y")
            grid_decl = (
                f"const int grid_x = {grid_x_expr};\n"
                f"                const int grid_y = {grid_y_expr};"
            )
            launch_dims = "grid_x, grid_y, 1,\n                                            block, 1, 1,"
        else:
            grid_x_expr = _format_grid_expr(shape, "x")
            grid_decl = f"const int grid  = {grid_x_expr};"
            launch_dims = "grid, 1, 1,\n                                            block, 1, 1,"
        text = textwrap.dedent(f"""\
            // AUTO-GENERATED by scripts/compile_kernels.py — do not edit by hand.

            #include "{name}.h"

            #include <stddef.h>
            #include <stdint.h>

            static const unsigned char kTritonCubin_{name}[] = {{
            {cubin_array}
            }};
            static const size_t kTritonCubinSize_{name} = sizeof(kTritonCubin_{name});

            static const char kTritonKernelName_{name}[] = "{kernel_symbol}";
            static const int  kTritonBlockSize_{name}    = {block_size};

            static CUmodule   g_module      = NULL;
            static CUfunction g_function    = NULL;
            static int        g_load_failed = 0;

            static int load_module_once(void) {{
                if (g_function) return 0;
                if (g_load_failed) return -1;

                CUresult r = cuModuleLoadData(&g_module, kTritonCubin_{name});
                if (r != CUDA_SUCCESS) {{ g_load_failed = 1; return -1; }}
                r = cuModuleGetFunction(&g_function, g_module, kTritonKernelName_{name});
                if (r != CUDA_SUCCESS) {{ g_load_failed = 1; return -1; }}
                return 0;
            }}

            int triton_launch_{name}(
                {params_lines}) {{
                if (load_module_once() != 0) {{
                    return -1;
                }}

                void * args[] = {{ {arg_addrs} }};
                const int block = kTritonBlockSize_{name};
                {grid_decl}

                CUresult r = cuLaunchKernel(g_function,
                                            {launch_dims}
                                            0, stream,
                                            args, NULL);
                return (r == CUDA_SUCCESS) ? 0 : -1;
                (void) kTritonCubinSize_{name};
            }}
            """)
    else:
        text = textwrap.dedent(f"""\
            // AUTO-GENERATED by scripts/compile_kernels.py — do not edit by hand.

            #include "{name}.h"

            #include <stddef.h>
            #include <stdint.h>

            static const unsigned char kTritonCubin_{name}[] = {{
            {cubin_array}
            }};
            static const size_t kTritonCubinSize_{name} = sizeof(kTritonCubin_{name});

            static const char kTritonKernelName_{name}[] = "{kernel_symbol}";
            static const int  kTritonBlockSize_{name}    = {block_size};

            static CUmodule   g_module      = NULL;
            static CUfunction g_function    = NULL;
            static int        g_load_failed = 0;

            static int load_module_once(void) {{
                if (g_function) return 0;
                if (g_load_failed) return -1;

                CUresult r = cuModuleLoadData(&g_module, kTritonCubin_{name});
                if (r != CUDA_SUCCESS) {{ g_load_failed = 1; return -1; }}
                r = cuModuleGetFunction(&g_function, g_module, kTritonKernelName_{name});
                if (r != CUDA_SUCCESS) {{ g_load_failed = 1; return -1; }}
                return 0;
            }}

            int triton_launch_{name}(CUstream    stream,
                                     CUdeviceptr d_in,
                                     CUdeviceptr d_out,
                                     int32_t     N) {{
                if (load_module_once() != 0) {{
                    return -1;
                }}

                void * args[] = {{ (void *) &d_in, (void *) &d_out, (void *) &N }};
                const int block = kTritonBlockSize_{name};
                const int grid  = (int)((N + block - 1) / block);

                CUresult r = cuLaunchKernel(g_function,
                                            grid, 1, 1,
                                            block, 1, 1,
                                            0, stream,
                                            args, NULL);
                return (r == CUDA_SUCCESS) ? 0 : -1;
                (void) kTritonCubinSize_{name};
            }}
            """)
    p = out_dir / f"{name}.c"
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Triton AOT compiler driver")
    parser.add_argument("--registry", required=True, type=Path,
                        help="Path to kernel_registry.json")
    parser.add_argument("--kernels", required=True, type=Path,
                        help="Directory containing the triton_kernels python package")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for generated .c/.h files")
    args = parser.parse_args()

    if not args.registry.is_file():
        print(f"[triton-aot] registry file not found: {args.registry}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # Make `triton_kernels` importable.
    parent = args.kernels.resolve().parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

    registry = _parse_registry(args.registry)

    default_block = json.loads(args.registry.read_text()).get("default_block_size", 1024)

    n_real = 0
    n_placeholder = 0
    for kernel in registry:
        for variant in kernel.variants:
            cubin = _compile_with_triton(kernel, variant)
            block_size = int(variant.specialise.get("BLOCK_SIZE", default_block))
            kernel_symbol = f"{kernel.function}_{variant.dtype}"

            _emit_header(args.out, kernel, variant)
            _emit_source(args.out, kernel, variant,
                         cubin or b"",
                         kernel_symbol=kernel_symbol,
                         block_size=block_size)

            if cubin:
                n_real += 1
            else:
                n_placeholder += 1

    print(f"[triton-aot] wrote {n_real} real and {n_placeholder} placeholder kernel(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
