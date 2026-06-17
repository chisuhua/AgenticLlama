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
        # into the filename so each variant emits its own .c/.h pair.  This is
        # what enables 24 distinct launchers for RoPE (3 modes x 2 sin x 2 yarn
        # x 2 dtype).  Variants that do not specialise these axes (GELU/SiLU,
        # RMSNorm) keep the pre-B.2 "<dtype>_<arch>" filename to preserve
        # byte-compat with the existing committed launchers.
        if "SIN_SIGN" in self.specialise or "YA_ON" in self.specialise:
            sin  = "fwd"     if int(self.specialise["SIN_SIGN"]) > 0 else "bwd"
            yarn = "yarnon"  if int(self.specialise["YA_ON"])    != 0 else "yarnoff"
            return f"{sin}_{yarn}_{self.dtype}_{self.arch}"
        return f"{self.dtype}_{self.arch}"


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
    "rms_norm_unweighted": {
        "params": [
            ("CUdeviceptr", "d_x"),
            ("CUdeviceptr", "d_y"),
            ("int32_t",     "N"),
        ],
        "grid_param": "N",
    },
    # B.1 RMSNorm weighted: y = x * rsqrt(mean(x*x) + eps) * w
    "rms_norm_weighted": {
        "params": [
            ("CUdeviceptr", "d_x"),
            ("CUdeviceptr", "d_w"),
            ("CUdeviceptr", "d_y"),
            ("int32_t",     "N"),
        ],
        "grid_param": "N",
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
                const int grid  = (int)(({shape["grid_param"]} + block - 1) / block);

                CUresult r = cuLaunchKernel(g_function,
                                            grid, 1, 1,
                                            block, 1, 1,
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
