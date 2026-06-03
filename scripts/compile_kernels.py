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

    # Translate "sm80" -> compute capability for Triton's compile API.
    cc = None
    if variant.arch.startswith("sm"):
        try:
            cc = int(variant.arch[2:])
        except ValueError:
            cc = None

    try:
        compiled = triton.compile(
            fn,
            signature=variant.signature,
            constants=variant.specialise,
            cc=cc,
        )
    except Exception as exc:
        print(f"[triton-aot] triton.compile failed for "
              f"{kernel.name}/{variant.tag}: {exc}", file=sys.stderr)
        return None

    cubin = getattr(compiled, "asm", {}).get("cubin")
    if cubin is None:
        print(f"[triton-aot] no cubin produced for {kernel.name}/{variant.tag}",
              file=sys.stderr)
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


def _format_byte_array(blob: bytes, indent: str = "    ", per_line: int = 12) -> str:
    parts = []
    for i in range(0, len(blob), per_line):
        chunk = blob[i:i + per_line]
        parts.append(indent + ", ".join(f"0x{b:02x}" for b in chunk))
    return ",\n".join(parts)


def _emit_header(out_dir: Path, kernel: Kernel, variant: Variant) -> Path:
    name = f"{kernel.name}_{variant.tag}"
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
