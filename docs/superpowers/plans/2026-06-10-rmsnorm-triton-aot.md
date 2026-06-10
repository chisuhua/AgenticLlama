# B.1 — RMSNorm Triton AOT Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Triton AOT RMSNorm provider (`GGML_OP_RMS_NORM`, fp16 + fp32) to the ggml-triton backend so Qwen3-style transformer blocks (the MiniMind-3 architecture) can run their normalization layers on the GPU path instead of falling back to ggml-cpu.

**Architecture:** Mirror the existing AOT elementwise pattern (GELU/SILU): one Triton DSL source per op → `scripts/compile_kernels.py` AOT-compiles to CUBIN → C launcher wraps `cuModuleLoadData`/`cuLaunchKernel` → C++ provider registers `(GGML_OP_RMS_NORM, dtype)` tuples into the existing `ggml_triton_op_registry`. Two stages: (Stage 1) prove the pipeline with `eps` as a `tl.constexpr` (no driver change) — this gets a working registration + test-bed; (Stage 2) extend the driver to thread runtime `eps` as a launcher parameter for production use.

**Tech Stack:** Triton 3.7.0 (AOT via `JITFunction.ASTSource` + `triton.compile`), CUDA Driver API (cuModule/cuLaunchKernel), C++17, CMake 3.18+, existing `ggml_triton_kernel_impl` function-pointer interface. Reference math: `ggml/src/ggml-cpu/ops.cpp:3758-3821` (`ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>`) and the in-tree CPU provider at `ggml/src/ggml-triton/ggml-triton-provider-cpu.cpp:464-510`.

---

## Critical environment caveat (read first)

> **Phase 0 audit finding (from `docs/superpowers/plans/2026-06-09-phase-0-audit.md` §0.4):** `scripts/compile_kernels.py` uses the pre-3.7.0 AOT API (`triton.compile(signature=...)` / `constants=` / `cc=`). On Triton 3.7.0 these kwargs were removed. The script falls back to a 16-byte ELF-magic placeholder CUBIN when run on this CPU-only host. This means **on the CPU-only dev box, every AOT step below produces a working C launcher backed by a stub CUBIN** — build succeeds, `test-backend-ops` exercises the dispatcher + provider + launcher load + (placeholder) launch path end-to-end, and the new provider IS reachable. **But the kernel itself does no real compute on this box.** Numeric correctness against `ggml-cpu` is only verifiable on a real GPU host (Triton 3.7.0 + SM80+ driver). The plan below accepts this and marks Stage 1 exit as "functional registration on CPU-only box; numeric verification deferred to GPU host per ROADMAP Phase A".

If you are on a GPU host with a working Triton 3.7.0 + CUDA 11.0+ + NVIDIA driver, you can also patch `scripts/compile_kernels.py` to the 3.7.0 API — but **that patch is out of scope for B.1** (it is a separate Phase 0 follow-up referenced in the audit).

---

## File map — what each new file does

| New file | Responsibility |
|---|---|
| `triton_kernels/rms_norm.py` | Triton DSL source: one `@triton.jit` function `rms_norm_kernel(x_ptr, w_ptr, y_ptr, N, BLOCK_SIZE)` computing `y = x * rsqrt(mean(x²) + eps) * w` (Stage 1: `eps` baked as a `tl.constexpr` parameter; Stage 2: also accepts runtime `eps`). |
| `ggml/src/ggml-triton/kernels/generated/rms_norm_unweighted_fp16_sm80.{h,c}` | AOT-generated launcher (do not hand-edit). Header declares `triton_launch_rms_norm_unweighted_fp16_sm80`; source embeds the CUBIN and implements the launcher. |
| `ggml/src/ggml-triton/kernels/generated/rms_norm_unweighted_fp32_sm80.{h,c}` | Same as above for fp32. |
| `ggml/src/ggml-triton/kernels/generated/rms_norm_weighted_fp16_sm80.{h,c}` | Same as above for the weighted variant (3 ptrs). |
| `ggml/src/ggml-triton/kernels/generated/rms_norm_weighted_fp32_sm80.{h,c}` | Same as above for fp32 weighted. |
| `ggml/src/ggml-triton/ggml-triton-provider-rmsnorm.h` | Declares `ggml_triton_register_rmsnorm_providers(ggml_triton_op_registry &)`. |
| `ggml/src/ggml-triton/ggml-triton-provider-rmsnorm.cpp` | Defines `supports`/`execute` pairs for fp16 and fp32; calls `registry.register_impl(GGML_OP_RMS_NORM, {...})` twice (priority 100 — GPU AOT path). |

| Modified file | What changes |
|---|---|
| `scripts/kernel_registry.json` | Append a new `"name": "rms_norm"` entry with fp16 + fp32 / sm80 variants. Stage 2: extend signature to include runtime `eps` and `weight_ptr`. |
| `scripts/compile_kernels.py` | **Stage 1**: no change. **Stage 2**: extend `_emit_header` and `_emit_source` so the launcher signature includes an extra `CUdeviceptr w` (weight) parameter and a `float eps` parameter; extend signature parser to accept `*T` ptr and `fp32` eps. |
| `ggml/src/ggml-triton/kernels/include/triton_kernels.h` | Append 4 includes: `rms_norm_{unweighted,weighted}_fp{16,32}_sm80.h`. |
| `ggml/src/ggml-triton/ggml-triton-provider.cpp` | Inside the `std::call_once` block at lines 60-71, append `ggml_triton_register_rmsnorm_providers(registry);` (gated by `#ifdef GGML_TRITON_HAS_RMSNORM` only if you add a CMake option in Step 14). |
| `ggml/src/ggml-triton/ggml-triton.cpp` | In `ggml_backend_triton_init`, append `ggml_triton_register_rmsnorm_providers(ctx->op_registry);` in **both** the CPU-only branch (around line 419) and the GPU branch (around line 474). |
| `ggml/src/ggml-triton/CMakeLists.txt` | Append 4 generated .c files (`rms_norm_{unweighted,weighted}_fp{16,32}_sm80.c`) to `GGML_TRITON_GENERATED_SRC`; append `ggml-triton-provider-rmsnorm.cpp` to the `ggml_add_backend_library` call. |
| `tests/test-triton-registry.cpp` | Add Assert 4 mirroring the GELU pattern (lines 26-39) for `GGML_OP_RMS_NORM` — assert the CPU fp32 provider is registered (proves the existing CPU reference path is reachable; the triton AOT entries will be added by the provider file in the same change set). |

| Out of scope for B.1 (explicit) |
|---|
| Patching `scripts/compile_kernels.py` for Triton 3.7.0's new `triton.compile(src, target, options)` API. This is the Phase 0 audit §0.4 follow-up and is needed only for **real** GPU numeric verification. |
| ROCm / AMD GPU variants (Phase D.4). |
| RoPE (B.2) and FlashAttn (B.3) — separate plans. |

---

## Stage 1 — Minimum viable RMSNorm (eps as constexpr)

### Task 1: Write the failing registry test

**Files:**
- Modify: `tests/test-triton-registry.cpp:65-84` (after the TileLang assert block)

- [ ] **Step 1: Add Assert 4 for GGML_OP_RMS_NORM**

Open `tests/test-triton-registry.cpp` and locate the end of the TileLang assert block (the closing `}` of the `if (!found_tilelang_add)` block at line 84). Immediately after, add:

```cpp
    // Assert 4 (B.1): the Triton AOT RMSNorm provider (fp16 + fp32) must be
    // registered for GGML_OP_RMS_NORM. The CPU fp32 entry already exists
    // (see ggml-triton-provider-cpu.cpp:785-790) — we are specifically
    // asserting that the *triton AOT* entries get added by the new
    // ggml-triton-provider-rmsnorm.{h,cpp} files (Task 8) and are reachable
    // from the global registry.
    bool found_triton_rms_norm_fp16 = false;
    bool found_triton_rms_norm_fp32 = false;
    if (auto * impls = reg.get_impls(GGML_OP_RMS_NORM)) {
        for (const auto & impl : *impls) {
            const std::string n = impl.name;
            if (impl.provider == GGML_TRITON_PROVIDER_TRITON && n.find("fp16") != std::string::npos) {
                found_triton_rms_norm_fp16 = true;
                std::printf("found RMS_NORM impl: name=%s provider=%d priority=%d\n",
                            impl.name, (int) impl.provider, impl.priority);
            }
            if (impl.provider == GGML_TRITON_PROVIDER_TRITON && n.find("fp32") != std::string::npos) {
                found_triton_rms_norm_fp32 = true;
                std::printf("found RMS_NORM impl: name=%s provider=%d priority=%d\n",
                            impl.name, (int) impl.provider, impl.priority);
            }
        }
    }
    if (!found_triton_rms_norm_fp16) {
        std::fprintf(stderr, "FAIL: triton AOT RMS_NORM fp16 provider not registered in global registry\n");
        return 4;
    }
    if (!found_triton_rms_norm_fp32) {
        std::fprintf(stderr, "FAIL: triton AOT RMS_NORM fp32 provider not registered in global registry\n");
        return 5;
    }
    std::printf("Assert 4 passed: triton AOT RMS_NORM fp16 + fp32 providers are registered\n");
```

(The `return 4;` / `return 5;` codes are intentionally distinct from the existing `return 1/2/3` to make the new failure mode easy to spot.)

- [ ] **Step 2: Build and run the test — expect FAIL**

```bash
cmake --build build-master --config Release --target test-triton-registry -j$(nproc)
./build-master/bin/test-triton-registry
```

Expected: exit code `4` with stderr `FAIL: triton AOT RMS_NORM fp16 provider not registered in global registry`. The existing CPU fp32 RMSNorm (`cpu_rms_norm_f32`) is reachable, but the triton AOT entries aren't yet — they're added by Tasks 7-10. (To confirm the test binary is actually rebuilt with your new assert, look for `Assert 4` in the output before the FAIL line; if the binary exits at 0 without `Assert 4` printed, your edit didn't compile in.)

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test-triton-registry.cpp
git commit -m "test(triton-registry): add Assert 4 for triton AOT RMS_NORM providers

Stage 1 of B.1 (RMSNorm provider per docs/development/ROADMAP.md).
Asserts the triton_rms_norm_fp{16,32}_sm80 impls are present in the
global registry; the assert will pass once Tasks 7-10 link the new
ggml-triton-provider-rmsnorm.cpp and wire both registration sites."
```

---

### Task 2: Add the kernel_registry.json entries

**Files:**
- Modify: `scripts/kernel_registry.json:7-44` (the `kernels` array)

**Why two entries (not one)**: per Oracle review, MiniMind-3 (and `test_rms_norm`) call `ggml_rms_norm(ctx, a, eps)` *without* a weight tensor — the graph then applies the weight as a separate `ggml_mul` op. So Stage 1 ships **two kernel variants**: `rms_norm_unweighted` (2 ptrs, signature mirrors GELU/SiLU exactly) and `rms_norm_weighted` (3 ptrs, used when src[1] is present). The unweighted variant lets the existing launcher ABI stay untouched — no driver change, no GELU/SiLU call-site updates.

- [ ] **Step 1: Append both entries at the start of the `kernels` array**

Open `scripts/kernel_registry.json`. Locate the line that opens the `kernels` array (`"kernels": [`). Immediately after the `[`, insert two entries separated by a comma:

```json
  "kernels": [
    {
      "name": "rms_norm_unweighted",
      "module": "triton_kernels.rms_norm",
      "function": "rms_norm_kernel",
      "variants": [
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 1024, "EPS": 1.0e-6, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,i32,1024,1e-06,0"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 1024, "EPS": 1.0e-6, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,i32,1024,1e-06,0"
        }
      ]
    },
    {
      "name": "rms_norm_weighted",
      "module": "triton_kernels.rms_norm",
      "function": "rms_norm_kernel",
      "variants": [
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 1024, "EPS": 1.0e-6, "USE_WEIGHT": 1 },
          "signature": "*fp16,*fp16,*fp16,i32,1024,1e-06,1"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 1024, "EPS": 1.0e-6, "USE_WEIGHT": 1 },
          "signature": "*fp32,*fp32,*fp32,i32,1024,1e-06,1"
        }
      ]
    },
    {
      "name": "gelu",

- [ ] **Step 2: Validate JSON syntax**

```bash
python3 -c "import json; json.load(open('scripts/kernel_registry.json'))" && echo "JSON OK"
```

Expected: `JSON OK`. If it errors, fix the JSON (the comma placement above the new entry is the most common mistake — a trailing comma after the last element of a JSON array is invalid).

- [ ] **Step 3: Commit**

```bash
git add scripts/kernel_registry.json
git commit -m "scripts: register rms_norm kernel in kernel_registry.json

B.1 Stage 1. Stage 2 will extend signature with runtime eps once
the launcher ABI extension lands."
```

---

### Task 3: Write the Triton DSL source

**Files:**
- Create: `triton_kernels/rms_norm.py`

- [ ] **Step 1: Create the file**

Create `triton_kernels/rms_norm.py`:

```python
"""Triton RMSNorm kernel for the ggml-triton backend (B.1).

Computes  y = x * rsqrt(mean(x*x) + eps) * weight   row-wise.

The kernel is AOT-compiled by ``scripts/compile_kernels.py`` for the
``(dtype, arch)`` combinations declared in ``scripts/kernel_registry.json``.

Stage 1 design: ``eps`` and ``BLOCK_SIZE`` are ``tl.constexpr`` so the
AOT launcher signature is the standard ``(in, weight, out, N)`` quad
plus N — no driver change required.

Stage 2 will promote ``eps`` to a runtime float argument and extend
``compile_kernels.py``'s launcher emitter.
"""

import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,                # *T   — input
    y_ptr,                # *T   — output
    w_ptr,                # *T   — weight (only when USE_WEIGHT=1; otherwise pass a
                          #        dummy pointer — kernel never dereferences it)
    N: tl.constexpr,      # int32 — row length (constexpr so BLOCK_SIZE >= N holds)
    BLOCK_SIZE: tl.constexpr,  # int32 — power-of-two tile; Stage 1 requires BLOCK_SIZE >= N
    EPS: tl.constexpr,    # fp32  — epsilon (constexpr in Stage 1)
    USE_WEIGHT: tl.constexpr,  # i1 — when 0, skip the w load and the weight mul
):
    # Single program per row.  Row is at program_id(0).
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load x in fp32 for accumulation regardless of input dtype.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # sum-of-squares (block-local; row fits in one BLOCK_SIZE because BLOCK_SIZE >= N).
    sum_sq = tl.sum(x * x, axis=0)
    mean = sum_sq / N
    scale = 1.0 / tl.sqrt(mean + EPS)

    if USE_WEIGHT:
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x * scale * w).to(y_ptr.dtype.element_ty)
    else:
        y = (x * scale).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offsets, y, mask=mask)
```

**Notes:**
- The kernel keeps a `w_ptr` slot in the signature for both variants, but the `USE_WEIGHT` constexpr branches the body. This keeps the AOT launcher ABI consistent between unweighted and weighted variants (3 ptrs in both cases: `x`, `y`, `w`) — but the **unweighted launcher is exposed as a separate 2-arg function** at the C level (`triton_launch_rms_norm_unweighted_fp16_sm80(stream, x, y, N)`) and the weighted one as 3-arg (`triton_launch_rms_norm_weighted_fp16_sm80(stream, x, w, y, N)`). This is achieved in Task 4 by introducing a small per-kernel launcher-shape map rather than rewriting the global emitter.
- The cast `.to(y_ptr.dtype.element_ty)` returns the output dtype (fp16 or fp32). Verify Triton ≥ 3.7 supports `pointer.dtype.element_ty`; if not, derive from a `IS_FP16: tl.constexpr` parameter (covered by the existing 2 dtype variants).

- [ ] **Step 2: Commit**

```bash
git add triton_kernels/rms_norm.py
git commit -m "triton_kernels: add rms_norm DSL kernel (Stage 1, constexpr eps)

B.1 of docs/development/ROADMAP.md. Stage 2 will switch eps to a
runtime launcher parameter."
```

---

### Task 4: Add a per-kernel launcher-shape map to compile_kernels.py (no global ABI change)

**Files:**
- Modify: `scripts/compile_kernels.py` — add a small lookup table mapping kernel `name` → launcher arg layout; branch `_emit_header` / `_emit_source` on it. **Do not** modify the existing elementwise path that GELU/SILU already use.

**Why this approach** (per Oracle review, blocking finding #2 and #3): a global launcher ABI change would regenerate all four GELU/SILU `.c/.h` files in Task 5 with the new signature, and the existing call sites in `ggml-triton-provider-triton.cpp` (lines 54, 91, 128, 165) would then have wrong-arity calls — a hard compile error. A global change also produces a transient git history where GELU/SiLU .c files are "dirty" between Task 4 and Task 5 even though their semantically required call sites haven't moved.

Instead, we add a per-kernel `LAUNCHER_SHAPES` table; the emitter picks the shape based on the kernel's `name` field in the registry. GELU/SILU take the "elementwise" path (unchanged); the two new RMSNorm entries take the "rms_norm_unweighted" (2 ptrs) and "rms_norm_weighted" (3 ptrs) shapes. Their generated files have different launcher function names, so they cannot collide with each other or with GELU/SILU.

- [ ] **Step 1: Read the existing launcher emitter to ground the edit**

```bash
grep -n "_emit_header\|_emit_source\|def _parse_signature\|launcher\|triton_launch" scripts/compile_kernels.py
```

- [ ] **Step 2: Add a launcher-shape map near the top of the file**

After the imports and before the first function, add:

```python
# Per-kernel launcher-arg layout.  The emitter uses the value to pick the
# header signature, the function-body parameter list, and the args[] array.
# Shape is a list of (c_type, c_name) pairs in source order (after CUstream).
LAUNCHER_SHAPES = {
    # Default for GELU / SILU: (in, out, N)
    "default": [
        ("CUdeviceptr", "d_in"),
        ("CUdeviceptr", "d_out"),
        ("int32_t",     "N"),
    ],
    # B.1 RMSNorm unweighted: (x, y, N)  -- 2 ptrs, no weight
    "rms_norm_unweighted": [
        ("CUdeviceptr", "d_x"),
        ("CUdeviceptr", "d_y"),
        ("int32_t",     "N"),
    ],
    # B.1 RMSNorm weighted: (x, w, y, N)  -- 3 ptrs, weight present
    "rms_norm_weighted": [
        ("CUdeviceptr", "d_x"),
        ("CUdeviceptr", "d_w"),
        ("CUdeviceptr", "d_y"),
        ("int32_t",     "N"),
    ],
}
```

- [ ] **Step 3: Branch the header emitter on `kernel.name`**

In `_emit_header`, replace the hard-coded signature block with:

```python
def _emit_header(name, dtype, arch):
    shape = LAUNCHER_SHAPES.get(name, LAUNCHER_SHAPES["default"])
    params = ["    CUstream    stream"]
    for c_type, c_name in shape:
        params.append(f"    {c_type:11s} {c_name}")
    param_str = ",\n".join(params) + ");"
    return f"""#pragma once
#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

int triton_launch_{name}_{dtype}_{arch}(
{param_str}

#ifdef __cplusplus
}}
#endif
"""
```

(The original `param_str` formatting in the existing emitter may use slightly different indentation; mirror the existing style and only swap the variable content.)

- [ ] **Step 4: Branch the source emitter on `kernel.name`**

In `_emit_source`, the args[] array and the function-body parameter list both derive from the same shape. Replace the hard-coded 3-element `args[]` with:

```python
def _emit_source(name, dtype, arch, cubin_bytes):
    shape = LAUNCHER_SHAPES.get(name, LAUNCHER_SHAPES["default"])
    arg_addrs = [f"(void *) &{c_name}" for _c_type, c_name in shape]
    args_decl = ", ".join(arg_addrs)
    body_params = ["CUstream    stream"] + [f"{c_type:11s} {c_name}" for c_type, c_name in shape]
    body_params_str = ",\n    ".join(body_params)
    return f"""// AUTO-GENERATED by scripts/compile_kernels.py — do not edit by hand.

#include "{name}_{dtype}_{arch}.h"

#include <stddef.h>
#include <stdint.h>

static const unsigned char kTritonCubin_{name}_{dtype}_{arch}[] = {{
{cubin_bytes}
}};
static const size_t kTritonCubinSize_{name}_{dtype}_{arch} = sizeof(kTritonCubin_{name}_{dtype}_{arch});

static const char kTritonKernelName_{name}_{dtype}_{arch}[] = "<TODO: actual kernel symbol>";
static const int  kTritonBlockSize_{name}_{dtype}_{arch}    = 1024;

static CUmodule   g_module      = NULL;
static CUfunction g_function    = NULL;
static int        g_load_failed = 0;

static int load_module_once(void) {{
    if (g_function) return 0;
    if (g_load_failed) return -1;

    CUresult r = cuModuleLoadData(&g_module, kTritonCubin_{name}_{dtype}_{arch});
    if (r != CUDA_SUCCESS) {{ g_load_failed = 1; return -1; }}
    r = cuModuleGetFunction(&g_function, g_module, kTritonKernelName_{name}_{dtype}_{arch});
    if (r != CUDA_SUCCESS) {{ g_load_failed = 1; return -1; }}
    return 0;
}}

int triton_launch_{name}_{dtype}_{arch}(
    {body_params_str}) {{
    if (load_module_once() != 0) {{
        return -1;
    }}

    void * args[] = {{ {args_decl} }};
    const int block = kTritonBlockSize_{name}_{dtype}_{arch};
    const int grid  = (int)((N + block - 1) / block);

    CUresult r = cuLaunchKernel(g_function,
                                grid, 1, 1,
                                block, 1, 1,
                                0, stream,
                                args, NULL);
    return (r == CUDA_SUCCESS) ? 0 : -1;
    (void) kTritonCubinSize_{name}_{dtype}_{arch};
}}
"""
```

(If the existing emitter stores `cubin_bytes` as a Python list of `int`s and formats them itself, preserve that flow; the snippet above is illustrative — match the existing style byte-for-byte except for the shape-derived substitutions.)

- [ ] **Step 5: Confirm the existing GELU/SILU launcher shape is preserved**

```bash
git diff scripts/compile_kernels.py | head -40
```

Expected: the diff shows only the **additions** of `LAUNCHER_SHAPES` and the shape-branched `_emit_header` / `_emit_source`. The old hard-coded `triton_launch_gelu_fp16_sm80(CUstream, CUdeviceptr d_in, CUdeviceptr d_out, int32_t N)` signature must still appear in the rendered output for `name == "gelu"`. If you accidentally hard-coded the new shape, the `git diff` for `ggml/src/ggml-triton/kernels/generated/gelu_*.c` (regenerated in Task 5) will show those files changed, and Task 5 will catch that.

- [ ] **Step 6: Commit**

```bash
git add scripts/compile_kernels.py
git commit -m "scripts: add per-kernel launcher-shape map (rms_norm unweighted/weighted)

B.1 Stage 1. Elementwise (GELU/SILU) launchers continue to use the
unchanged (in, out, N) shape; the two new rms_norm entries use
(x, y, N) and (x, w, y, N) respectively. Their launcher function
names differ (rms_norm_unweighted_* vs rms_norm_weighted_*) so they
cannot collide with each other or with the elementwise launchers."
```

---

### Task 5: Generate the launcher .c/.h files (AOT driver run)

**Files:**
- Create: `ggml/src/ggml-triton/kernels/generated/rms_norm_unweighted_fp16_sm80.h`
- Create: `ggml/src/ggml-triton/kernels/generated/rms_norm_unweighted_fp16_sm80.c`
- Create: `ggml/src/ggml-triton/kernels/generated/rms_norm_unweighted_fp32_sm80.h`
- Create: `ggml/src/ggml-triton/kernels/generated/rms_norm_unweighted_fp32_sm80.c`
- Create: `ggml/src/ggml-triton/kernels/generated/rms_norm_weighted_fp16_sm80.h`
- Create: `ggml/src/ggml-triton/kernels/generated/rms_norm_weighted_fp16_sm80.c`
- Create: `ggml/src/ggml-triton/kernels/generated/rms_norm_weighted_fp32_sm80.h`
- Create: `ggml/src/ggml-triton/kernels/generated/rms_norm_weighted_fp32_sm80.c`

- [ ] **Step 1: Run the AOT driver**

```bash
python3 scripts/compile_kernels.py \
    --registry scripts/kernel_registry.json \
    --kernels  triton_kernels \
    --out      ggml/src/ggml-triton/kernels/generated
```

Expected output (CPU-only fallback path per Phase 0 audit §0.4):

```
[compile_kernels] building rms_norm_unweighted/fp16/sm80 ...
[compile_kernels] NOTE: no CUDA driver detected, emitting placeholder CUBIN (16-byte ELF magic)
[compile_kernels] wrote .../rms_norm_unweighted_fp16_sm80.h
[compile_kernels] wrote .../rms_norm_unweighted_fp16_sm80.c
[compile_kernels] building rms_norm_unweighted/fp32/sm80 ...
[compile_kernels] wrote .../rms_norm_unweighted_fp32_sm80.h
[compile_kernels] wrote .../rms_norm_unweighted_fp32_sm80.c
[compile_kernels] building rms_norm_weighted/fp16/sm80 ...
[compile_kernels] wrote .../rms_norm_weighted_fp16_sm80.h
[compile_kernels] wrote .../rms_norm_weighted_fp16_sm80.c
[compile_kernels] building rms_norm_weighted/fp32/sm80 ...
[compile_kernels] wrote .../rms_norm_weighted_fp32_sm80.h
[compile_kernels] wrote .../rms_norm_weighted_fp32_sm80.c
```

**Critical**: the placeholder CUBIN path is the design-fallback (per Phase 0 audit); each `.c` file embeds a 16-byte ELF magic stub. On a real GPU host with a working Triton 3.7.0 + CUDA driver, the same command emits real CUBINs.

- [ ] **Step 2: Verify the EXISTING GELU/SILU launchers are UNCHANGED**

```bash
git diff ggml/src/ggml-triton/kernels/generated/gelu_fp16_sm80.c ggml/src/ggml-triton/kernels/generated/gelu_fp16_sm80.h ggml/src/ggml-triton/kernels/generated/silu_fp16_sm80.c ggml/src/ggml-triton/kernels/generated/silu_fp16_sm80.h
```

Expected: empty diff. If `git diff` shows ANY changes to the existing 4 GELU/SiLU `.c/.h` files, Task 4's per-kernel shape map leaked into the default path — fix the `LAUNCHER_SHAPES` lookup logic and re-run Task 5.

- [ ] **Step 3: Verify the new unweighted header shape**

```bash
cat ggml/src/ggml-triton/kernels/generated/rms_norm_unweighted_fp16_sm80.h
```

Expected: 2 ptrs (`d_x`, `d_y`) + `int32_t N` — mirrors the GELU launcher layout:

```c
#pragma once
#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int triton_launch_rms_norm_unweighted_fp16_sm80(CUstream    stream,
                                                CUdeviceptr d_x,
                                                CUdeviceptr d_y,
                                                int32_t     N);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 4: Verify the new weighted header shape**

```bash
cat ggml/src/ggml-triton/kernels/generated/rms_norm_weighted_fp16_sm80.h
```

Expected: 3 ptrs (`d_x`, `d_w`, `d_y`) + `int32_t N`:

```c
#pragma once
#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int triton_launch_rms_norm_weighted_fp16_sm80(CUstream    stream,
                                              CUdeviceptr d_x,
                                              CUdeviceptr d_w,
                                              CUdeviceptr d_y,
                                              int32_t     N);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 5: Verify the new source files contain the right arg counts**

```bash
grep -n "args\[\]" ggml/src/ggml-triton/kernels/generated/rms_norm_unweighted_fp16_sm80.c
grep -n "args\[\]" ggml/src/ggml-triton/kernels/generated/rms_norm_weighted_fp16_sm80.c
```

Expected: the unweighted one has 3 elements in `args[]` (d_x, d_y, N), the weighted one has 4 (d_x, d_w, d_y, N). If either file is missing or has the wrong count, the shape map in Task 4 isn't being applied — re-check.

- [ ] **Step 6: Commit**

```bash
git add ggml/src/ggml-triton/kernels/generated/rms_norm_{unweighted,weighted}_fp{16,32}_sm80.{h,c}
git commit -m "ggml-triton: AOT-generate rms_norm launchers (placeholder CUBIN on CPU-only host)

B.1 Stage 1. Two kernel families per Oracle review: rms_norm_unweighted
(2 ptrs, used when src[1]==nullptr — MiniMind-3, test_rms_norm) and
rms_norm_weighted (3 ptrs, used when src[1] is present). Per Phase 0
audit §0.4 the AOT path emits 16-byte ELF-magic placeholders on this
host. The C launcher ABI is real; numeric verification requires a
GPU host (out of scope for B.1 on this box)."
```

---

### Task 6: Wire the aggregated include header

**Files:**
- Modify: `ggml/src/ggml-triton/kernels/include/triton_kernels.h`

- [ ] **Step 1: Add the four includes**

Open `ggml/src/ggml-triton/kernels/include/triton_kernels.h`. After the `silu` includes (around line 19), add:

```c
#include "rms_norm_unweighted_fp16_sm80.h"
#include "rms_norm_unweighted_fp32_sm80.h"
#include "rms_norm_weighted_fp16_sm80.h"
#include "rms_norm_weighted_fp32_sm80.h"
```

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-triton/kernels/include/triton_kernels.h
git commit -m "ggml-triton: pull rms_norm launchers into aggregated header (4 files)

B.1 Stage 1. The two unweighted entries cover MiniMind-3 / test_rms_norm
(src[1]==nullptr); the two weighted entries cover the fused-RMSNorm path
(src[1] is a weight tensor). Per Oracle review, weight is optional, so
both families ship in Stage 1."
```

---

### Task 7: Declare the new provider header

**Files:**
- Create: `ggml/src/ggml-triton/ggml-triton-provider-rmsnorm.h`

- [ ] **Step 1: Create the header file**

```cpp
// ggml/src/ggml-triton/ggml-triton-provider-rmsnorm.h
//
// B.1 (RMSNorm provider) — see docs/development/ROADMAP.md §3 Phase B.
//
// Mirrors the shape of ggml-triton-provider-cutlass.h and
// ggml-triton-provider-tilelang.h: one free C++ registration function
// to be called from both ggml-triton-provider.cpp (global registry) and
// ggml-triton.cpp (per-context registry).

#pragma once

struct ggml_triton_op_registry;

#ifdef __cplusplus
extern "C" {
#endif

void ggml_triton_register_rmsnorm_providers(ggml_triton_op_registry & registry);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-triton/ggml-triton-provider-rmsnorm.h
git commit -m "ggml-triton: declare rmsnorm provider registration function"
```

---

### Task 8: Implement the new provider (TDD: TDD-style — wire, see fail, then fill in supports/execute)

**Files:**
- Create: `ggml/src/ggml-triton/ggml-triton-provider-rmsnorm.cpp`

- [ ] **Step 1: Create the file with the register function only (no supports/execute yet)**

```cpp
// ggml/src/ggml-triton/ggml-triton-provider-rmsnorm.cpp
//
// B.1 RMSNorm AOT provider (Stage 1, constexpr eps). See
// docs/development/ROADMAP.md §3 Phase B.1 for context.
//
// Math reference (bit-equivalent):
//   ggml/src/ggml-cpu/ops.cpp:3758-3821
//     ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>
//   ggml/src/ggml-triton/ggml-triton-provider-cpu.cpp:464-510
//     cpu_rms_norm_f32_execute
//
// Per-row computation:
//   y[i] = x[i] * rsqrt(mean(x*x) + eps) * w[i]    (weighted variant)
//   y[i] = x[i] * rsqrt(mean(x*x) + eps)           (unweighted variant, src[1]==nullptr)
//
// Per Oracle review: MiniMind-3 (and tests/test-backend-ops.cpp's test_rms_norm)
// call ggml_rms_norm(ctx, a, eps) WITHOUT a weight tensor, so src[1]==nullptr is
// the COMMON case, not the exception. We ship BOTH variants: unweighted and
// weighted. Each variant has fp16 and fp32 specializations, total 4 impls.
//
// The kernel source is triton_kernels/rms_norm.py; the AOT launcher
// signatures are emitted by scripts/compile_kernels.py into
// ggml/src/ggml-triton/kernels/generated/rms_norm_{unweighted,weighted}_fp{16,32}_sm80.{h,c}.

#include "ggml-triton-provider-rmsnorm.h"
#include "ggml-triton-provider.h"

#include "kernels/include/triton_kernels.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>


// --- shared predicates (extracted to avoid 4x duplication) ------------------

static inline bool rms_norm_eps_matches_stage1(const struct ggml_tensor * node) {
    // Stage 1: eps is a tl.constexpr (= 1e-6) baked into the AOT launcher.
    // We use a tolerance-bounded comparison (per Oracle review §6) so a
    // 1-ULP rounding drift in the runtime eps still routes to the triton
    // AOT path; a bit-exact `!=` would falsely fall back to CPU for any
    // computed value (e.g. from JSON or arithmetic). Stage 2 will thread
    // runtime eps through the launcher and remove this gate.
    float eps_runtime = 0.0f;
    std::memcpy(&eps_runtime, node->op_params, sizeof(float));
    return std::fabsf(eps_runtime - 1.0e-6f) <= 1.0e-7f;
}

static inline bool rms_norm_row_fits_stage1(const struct ggml_tensor * node) {
    // Stage 1 constraint: BLOCK_SIZE (=1024) must be >= the row length ne00.
    // The kernel masks out-of-range loads (other=0.0) so sum-of-squares is
    // correct for any N <= BLOCK_SIZE, but the literal `N` is a constexpr
    // and the row is laid out for one program per row. Reject rows larger
    // than 1024 to keep semantics correct; a follow-up adds a multi-block
    // variant.
    return node->src[0]->ne[0] <= 1024;
}


// --- unweighted / fp16 ------------------------------------------------------

static bool triton_rms_norm_unweighted_fp16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM) return false;
    if (op->type != GGML_TYPE_F16) return false;
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F16) return false;
    // UNWEIGHTED: src[1] must be NULL.  Weighted variant handles src[1] != null.
    if (op->src[1] != nullptr) return false;
    if (!rms_norm_row_fits_stage1(op)) return false;
    if (!rms_norm_eps_matches_stage1(op)) return false;
    return true;
}

static bool triton_rms_norm_unweighted_fp16_execute(struct ggml_backend_triton_context * ctx,
                                                   const struct ggml_tensor * node) {
    if (node->src[0]->data == nullptr || node->data == nullptr) {
        return false;
    }
    const CUdeviceptr d_x = (CUdeviceptr) node->src[0]->data;
    const CUdeviceptr d_y = (CUdeviceptr) node->data;
    const int32_t     N   = (int32_t) node->src[0]->ne[0];
    const int rc = triton_launch_rms_norm_unweighted_fp16_sm80(
        ctx->cu_stream, d_x, d_y, N);
    return rc == 0;
}


// --- weighted / fp16 --------------------------------------------------------

static bool triton_rms_norm_weighted_fp16_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM) return false;
    if (op->type != GGML_TYPE_F16) return false;
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F16) return false;
    // WEIGHTED: src[1] must be a same-dtype weight tensor.
    if (op->src[1] == nullptr || op->src[1]->type != GGML_TYPE_F16) return false;
    if (!rms_norm_row_fits_stage1(op)) return false;
    if (!rms_norm_eps_matches_stage1(op)) return false;
    return true;
}

static bool triton_rms_norm_weighted_fp16_execute(struct ggml_backend_triton_context * ctx,
                                                 const struct ggml_tensor * node) {
    if (node->src[0]->data == nullptr || node->src[1]->data == nullptr || node->data == nullptr) {
        return false;
    }
    const CUdeviceptr d_x = (CUdeviceptr) node->src[0]->data;
    const CUdeviceptr d_w = (CUdeviceptr) node->src[1]->data;
    const CUdeviceptr d_y = (CUdeviceptr) node->data;
    const int32_t     N   = (int32_t) node->src[0]->ne[0];
    const int rc = triton_launch_rms_norm_weighted_fp16_sm80(
        ctx->cu_stream, d_x, d_w, d_y, N);
    return rc == 0;
}


// --- unweighted / fp32 ------------------------------------------------------

static bool triton_rms_norm_unweighted_fp32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32) return false;
    if (op->src[1] != nullptr) return false;
    if (!rms_norm_row_fits_stage1(op)) return false;
    if (!rms_norm_eps_matches_stage1(op)) return false;
    return true;
}

static bool triton_rms_norm_unweighted_fp32_execute(struct ggml_backend_triton_context * ctx,
                                                   const struct ggml_tensor * node) {
    if (node->src[0]->data == nullptr || node->data == nullptr) {
        return false;
    }
    const CUdeviceptr d_x = (CUdeviceptr) node->src[0]->data;
    const CUdeviceptr d_y = (CUdeviceptr) node->data;
    const int32_t     N   = (int32_t) node->src[0]->ne[0];
    const int rc = triton_launch_rms_norm_unweighted_fp32_sm80(
        ctx->cu_stream, d_x, d_y, N);
    return rc == 0;
}


// --- weighted / fp32 --------------------------------------------------------

static bool triton_rms_norm_weighted_fp32_supports(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32) return false;
    if (op->src[1] == nullptr || op->src[1]->type != GGML_TYPE_F32) return false;
    if (!rms_norm_row_fits_stage1(op)) return false;
    if (!rms_norm_eps_matches_stage1(op)) return false;
    return true;
}

static bool triton_rms_norm_weighted_fp32_execute(struct ggml_backend_triton_context * ctx,
                                                 const struct ggml_tensor * node) {
    if (node->src[0]->data == nullptr || node->src[1]->data == nullptr || node->data == nullptr) {
        return false;
    }
    const CUdeviceptr d_x = (CUdeviceptr) node->src[0]->data;
    const CUdeviceptr d_w = (CUdeviceptr) node->src[1]->data;
    const CUdeviceptr d_y = (CUdeviceptr) node->data;
    const int32_t     N   = (int32_t) node->src[0]->ne[0];
    const int rc = triton_launch_rms_norm_weighted_fp32_sm80(
        ctx->cu_stream, d_x, d_w, d_y, N);
    return rc == 0;
}


// --- registration -----------------------------------------------------------

void ggml_triton_register_rmsnorm_providers(ggml_triton_op_registry & registry) {
    registry.register_impl(GGML_OP_RMS_NORM, {
        /* .name     = */ "triton_rms_norm_unweighted_fp16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_rms_norm_unweighted_fp16_supports,
        /* .execute  = */ triton_rms_norm_unweighted_fp16_execute,
        /* .priority = */ 100,
    });
    registry.register_impl(GGML_OP_RMS_NORM, {
        /* .name     = */ "triton_rms_norm_weighted_fp16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_rms_norm_weighted_fp16_supports,
        /* .execute  = */ triton_rms_norm_weighted_fp16_execute,
        /* .priority = */ 100,
    });
    registry.register_impl(GGML_OP_RMS_NORM, {
        /* .name     = */ "triton_rms_norm_unweighted_fp32_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_rms_norm_unweighted_fp32_supports,
        /* .execute  = */ triton_rms_norm_unweighted_fp32_execute,
        /* .priority = */ 100,
    });
    registry.register_impl(GGML_OP_RMS_NORM, {
        /* .name     = */ "triton_rms_norm_weighted_fp32_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_rms_norm_weighted_fp32_supports,
        /* .execute  = */ triton_rms_norm_weighted_fp32_execute,
        /* .priority = */ 100,
    });
}
```

**Important — interaction with the existing CPU provider**: the dispatcher picks the highest-priority impl whose `supports` returns true. The existing `cpu_rms_norm_f32` (priority 50) will be selected for any fp32 row where the triton AOT impls' `supports` return false (e.g. `ne00 > 1024` or `eps` outside the 1e-6 ± 1e-7 tolerance band). The unweighted and weighted AOT impls are mutually exclusive (one requires src[1]==nullptr, the other requires src[1]!=nullptr), so the dispatcher can never pick the wrong one for a given node. The per-node fallback to CPU works per-node, not per-graph: per Oracle §7, the ggml scheduler can still assign some nodes to ggml-triton and others to ggml-cpu within the same graph if needed (though for the single-backend tests in this plan, the harness uses one backend at a time).

- [ ] **Step 2: Commit (do NOT build yet — registration sites are not wired)**

```bash
git add ggml/src/ggml-triton/ggml-triton-provider-rmsnorm.cpp
git commit -m "ggml-triton: implement RMSNorm unweighted + weighted AOT providers (4 impls)

B.1 Stage 1. Per Oracle review §1: MiniMind-3 and test_rms_norm call
ggml_rms_norm() WITHOUT a weight tensor, so the unweighted variant
(src[1]==nullptr) is the common case. We ship both unweighted and
weighted, each in fp16 and fp32, total 4 impls.

Math: y = x * rsqrt(mean(x*x) + eps) [unweighted] or
      y = x * rsqrt(mean(x*x) + eps) * w [weighted]
matching ggml-cpu reference (ops.cpp:3758-3821) and the existing CPU
provider (provider-cpu.cpp:464-510).

Row length constrained to ne00 <= 1024 (Stage 1 BLOCK_SIZE=1024);
multi-block variant deferred to a follow-up. Stage 1 eps is baked at
1e-6; nodes with eps outside the 1e-6 ± 1e-7 tolerance band fall back
to the CPU provider. Stage 2 will thread runtime eps through the
launcher and remove this gate."
```

---

### Task 9: Wire the new provider into CMakeLists.txt

**Files:**
- Modify: `ggml/src/ggml-triton/CMakeLists.txt`

- [ ] **Step 1: Add the generated .c files to GENERATED_SRC**

Locate `set(GGML_TRITON_GENERATED_SRC ...)` (around line 46). Append the two new entries:

```cmake
    ${TRITON_GENERATED_DIR}/rms_norm_unweighted_fp16_sm80.c
    ${TRITON_GENERATED_DIR}/rms_norm_unweighted_fp32_sm80.c
    ${TRITON_GENERATED_DIR}/rms_norm_weighted_fp16_sm80.c
    ${TRITON_GENERATED_DIR}/rms_norm_weighted_fp32_sm80.c
```

- [ ] **Step 2: Add the provider .cpp to the backend library**

Locate the `ggml_add_backend_library(ggml-triton ...)` call (around line 58-65). Inside the list, append:

```cmake
    ggml-triton-provider-rmsnorm.cpp
```

- [ ] **Step 3: Verify the cmake syntax**

```bash
cmake -B build-master -DGGML_TRITON=ON 2>&1 | tail -25
```

Expected: cmake configure succeeds, and you see the new file paths in the `ggml-triton` target's source list. If cmake errors with "file not found" on any of the four new `.c` files or the provider `.cpp`, double-check the paths.

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-triton/CMakeLists.txt
git commit -m "ggml-triton: link rms_norm provider + 4 generated launchers into backend library"
```

---

### Task 10: Wire global registration (both registries)

**Files:**
- Modify: `ggml/src/ggml-triton/ggml-triton-provider.cpp` (global registry)
- Modify: `ggml/src/ggml-triton/ggml-triton.cpp` (per-context registry, two branches)

- [ ] **Step 1: Global registry — `ggml-triton-provider.cpp`**

Open `ggml/src/ggml-triton/ggml-triton-provider.cpp`. Inside the `std::call_once` block of `ggml_triton_global_registry()` (around line 60-71), add the call. It should sit alongside the existing provider registrations:

```cpp
        ggml_triton_register_cpu_providers(registry);
        ggml_triton_register_builtin_providers(registry);
        ggml_triton_register_rmsnorm_providers(registry);   // <-- new (B.1)
#ifdef GGML_TRITON_HAS_CUTLASS
        ggml_triton_register_cutlass_providers(registry);
#endif
#ifdef GGML_TRITON_HAS_TILELANG
        ggml_triton_register_tilelang_providers(registry);
#endif
```

Add the corresponding include at the top of the file:

```cpp
#include "ggml-triton-provider-rmsnorm.h"
```

- [ ] **Step 2: Per-context registry — `ggml-triton.cpp` CPU-only branch**

Locate the CPU-only branch in `ggml_backend_triton_init` (around line 413-420). Add the call after the existing registrations:

```cpp
    ggml_triton_register_builtin_providers(ctx->op_registry);
    ggml_triton_register_cpu_providers(ctx->op_registry);
    ggml_triton_register_rmsnorm_providers(ctx->op_registry);   // <-- new (B.1)
```

- [ ] **Step 3: Per-context registry — `ggml-triton.cpp` GPU branch**

Locate the GPU branch (around line 472-477). Add the same call:

```cpp
    ggml_triton_register_builtin_providers(ctx->op_registry);
    ggml_triton_register_rmsnorm_providers(ctx->op_registry);   // <-- new (B.1)
#ifdef GGML_TRITON_HAS_CUTLASS
    ggml_triton_register_cutlass_providers(ctx->op_registry);
#endif
```

(Stage 1 only needs the triton AOT path; the existing `cpu_rms_norm_f32` is already registered by `ggml_triton_register_cpu_providers` which the GPU branch does NOT call — that's the design. The triton AOT entries have priority 100. Per Oracle §7: when a node's `supports` returns false (e.g. `ne00 > 1024` or `eps` outside tolerance), the dispatcher does NOT reject the whole graph from ggml-triton — it per-node assigns that specific op to the ggml-cpu backend and routes the other ops to ggml-triton. So a model graph mixing small-row and large-row RMS_NORM nodes will still see partial ggml-triton acceleration on the small rows.)

- [ ] **Step 4: Build and run the failing-then-passing test**

```bash
cmake --build build-master --config Release --target test-triton-registry -j$(nproc)
./build-master/bin/test-triton-registry
```

Expected (post-fix): exit 0 with output `... Assert 4 passed: triton AOT RMS_NORM fp16 + fp32 providers are registered`. The global registry now contains all 5 RMS_NORM impls (CPU + 4 triton AOT); Assert 4 specifically checks that the triton AOT fp16 + fp32 entries are present. (The CPU fp32 is reachable too — see Assert 1-style checks for the existing CPU providers — but Assert 4's job is to gate the new code path.)

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-triton/ggml-triton-provider.cpp ggml/src/ggml-triton/ggml-triton.cpp
git commit -m "ggml-triton: register rmsnorm provider in both global and per-context registries

B.1 Stage 1 complete: RMS_NORM dispatchable via global and per-context
registries. Assert 4 in test-triton-registry.cpp now passes."
```

---

### Task 11: Wire test-backend-ops path — provider reachable

**Files:**
- Build: `tests/test-backend-ops` against the new `ggml-triton` library

- [ ] **Step 1: Rebuild test-backend-ops**

```bash
cmake --build build-master --config Release --target test-backend-ops -j$(nproc)
```

- [ ] **Step 2: Run the RMS_NORM test cases (CPU-only path)**

```bash
./build-master/bin/test-backend-ops RMS_NORM
```

Expected: existing test_rms_norm cases all pass (F32, F16, etc., with the existing CPU provider as the path being exercised). The new triton AOT provider is **not yet selected** on this host because the test framework only iterates over the built-in backends; it does not auto-construct a ggml-triton backend. The triton AOT path is exercised by the Level-4 cross-backend consistency check below.

- [ ] **Step 3: Run the triton-registry test — verify the new impls are reachable**

```bash
./build-master/bin/test-triton-registry
```

Expected: prints `Assert 1/2/3/4 passed`, exits 0. (If you wired the global registry correctly, the `get_impls(GGML_OP_RMS_NORM)` lookup returns 5 impls: `cpu_rms_norm_f32` + 4 triton AOT entries.)

---

### Task 12: Add a Level-4 cross-backend consistency check (optional, manual on GPU host)

This task is not blockable on the CPU-only host — it requires a real GPU with working Triton 3.7.0 + CUDA driver (per Phase 0 audit §0.4). It is listed here for completeness so the engineer knows what to run on a GPU host once available.

- [ ] **Step 1: On GPU host, run with cross-backend harness**

```bash
cd build && ./bin/test-backend-ops --backends CPU,TRITON
```

The harness iterates each test case twice (once per backend) and diffs. `test_rms_norm` (tests/test-backend-ops.cpp:3371) has `grad_precise()=true`, which selects the tight tolerance ladder. Expected diff vs CPU reference: ≤ 1e-3 fp16 (well within the precision target stated in ROADMAP §3 Phase A exit criteria). If the diff exceeds 1e-3, the likely cause is the Stage 1 `eps` baked at 1e-6 vs the actual model eps — Stage 2 removes this constraint by threading runtime eps.

- [ ] **Step 2: Capture the result in the docs**

Append a row to `docs/performance/unified-backend.md` (the Phase A/B 4-mode comparison table) recording the GPU host's hostname, commit SHA, and the cross-backend RMS_NORM diff. This is the same evidence capture ROADMAP §3 Phase C.2 prescribes.

---

### Task 13: MiniMind-3 end-to-end smoke (Level 2)

- [ ] **Step 1: Run the model with the ggml-triton backend in DEBUG log mode**

```bash
GGML_TRITON_LOG_LEVEL=DEBUG ./build-master/bin/llama-cli \
    -m minimind-3-F16.gguf \
    -p "1+1等于几" \
    -n 50
```

Look in the log for lines like `ggml-triton: selected impl triton_rms_norm_unweighted_fp16_sm80 for node ...` (or `_weighted_`). The presence of such a line proves the dispatcher routed the RMS_NORM node to your new provider. (Note: per the optional follow-up at the end of this plan, this log line is currently NOT emitted by upstream `ggml-triton-dispatch.cpp` — if you want it, add the `GGML_LOG_DEBUG` line from "Optional follow-up" first; otherwise the proof of routing is the registry unit test passing in Task 10.)

Expected outcome: the model produces a sensible Chinese-language response (this is the Level-2 smoke per `test-pyramid.md` line 169-177). The output should be identical to the CPU-only baseline because all RNG / sampling is upstream of the provider.

If the dispatcher does NOT log `triton_rms_norm_*`, the most likely cause is that the global registry's `supports` query (called by `ggml_backend_triton_device_supports_op`) returned false — re-check Task 8's supports predicates: `op->src[1] != nullptr` and `op->src[0]->ne[0] <= 1024`. If MiniMind-3 uses a larger hidden dim or omits the weight, the graph assignment will skip ggml-triton and fall back to CPU.

- [ ] **Step 2: Commit any doc updates**

If you recorded the log line in `docs/performance/unified-backend.md`, commit that. Otherwise no commit.

---

### Task 14: Optional — gate behind CMake option (recommended for CI)

**Files:**
- Modify: `ggml/src/ggml-triton/CMakeLists.txt` (add `option(GGML_TRITON_WITH_RMSNORM ...)`)
- Modify: `ggml/src/ggml-triton/ggml-triton-provider.cpp` (gate the call)
- Modify: `ggml/src/ggml-triton/ggml-triton.cpp` (gate the call in both branches)

The MIRROR pattern at CMakeLists.txt lines 90-105 (CUTLASS) and 132-161 (TileLang) is the template. This task is OPTIONAL for B.1 Stage 1 because the placeholder CUBIN + provider file are non-intrusive — but it isolates the new code behind a CI flag in case the multi-block constraint (ne00 > 1024) bites a downstream user.

- [ ] **Step 1: Add the CMake option**

```cmake
option(GGML_TRITON_WITH_RMSNORM "ggml-triton: enable RMSNorm provider" ON)
```

- [ ] **Step 2: Gate the sources**

```cmake
if (GGML_TRITON_WITH_RMSNORM)
    list(APPEND GGML_TRITON_GENERATED_SRC
        ${TRITON_GENERATED_DIR}/rms_norm_unweighted_fp16_sm80.c
        ${TRITON_GENERATED_DIR}/rms_norm_unweighted_fp32_sm80.c
        ${TRITON_GENERATED_DIR}/rms_norm_weighted_fp16_sm80.c
        ${TRITON_GENERATED_DIR}/rms_norm_weighted_fp32_sm80.c)
    list(APPEND GGML_TRITON_SOURCES ggml-triton-provider-rmsnorm.cpp)
    target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_HAS_RMSNORM)
endif()
```

- [ ] **Step 3: Gate the registration calls** (in `ggml-triton-provider.cpp` and `ggml-triton.cpp`)

Wrap each new `ggml_triton_register_rmsnorm_providers(...)` call:

```cpp
#ifdef GGML_TRITON_HAS_RMSNORM
    ggml_triton_register_rmsnorm_providers(registry);
#endif
```

- [ ] **Step 4: Build with option off, verify clean**

```bash
cmake -B build-master -DGGML_TRITON=ON -DGGML_TRITON_WITH_RMSNORM=OFF
cmake --build build-master -j$(nproc)
./build-master/bin/test-triton-registry
```

Expected: the test still passes (Assert 4 still passes because the CPU provider is independent of this flag). The triton AOT entries are absent from the global registry — verifiable by adding a temporary debug print in the test.

- [ ] **Step 5: Build with option on (default), confirm registration is back**

```bash
cmake -B build-master -DGGML_TRITON=ON -DGGML_TRITON_WITH_RMSNORM=ON
cmake --build build-master -j$(nproc)
./build-master/bin/test-triton-registry
```

Expected: all 4 asserts pass, including the triton AOT entries' names in the "found RMS_NORM impl" log line.

- [ ] **Step 6: Commit**

```bash
git add ggml/src/ggml-triton/CMakeLists.txt ggml/src/ggml-triton/ggml-triton-provider.cpp ggml/src/ggml-triton/ggml-triton.cpp
git commit -m "ggml-triton: gate RMSNorm provider behind GGML_TRITON_WITH_RMSNORM (default ON)

B.1 polish. Mirrors the GGML_TRITON_WITH_CUTLASS / GGML_TRITON_WITH_TILELANG
option pattern. CI can flip the flag to bisect regressions without
unplugging the entire provider file."
```

---

### Task 15: Update test-pyramid.md (B.1 coverage marker)

**Files:**
- Modify: `docs/development/test-pyramid.md`

Per Oracle §9b: the test-pyramid doc is the canonical "which tests cover which changes" reference. Adding a one-line marker for B.1 makes it discoverable to future readers that the ggml-triton RMSNorm op family is now part of the covered set.

- [ ] **Step 1: Add an entry to the test-pyramid's op-coverage table**

Open `docs/development/test-pyramid.md`. Locate the Level-3 row that lists ggml op families (around line 60-84 of the file — the "改了什么 / 该跑的测试 / 为什么是这些" table). The current row already says "**任何 ggml op**（matmul / rope / norm / softmax / silu / quant / dequant / ...）" which is generic. Below the table (or appended to the row description), add a brief sub-list of ops with active Triton AOT providers as of B.1:

```markdown
> **ggml-triton op coverage as of B.1 (RMSNorm):**
> - `GGML_OP_UNARY` (GELU, SILU — fp16 + fp32)
> - `GGML_OP_RMS_NORM` (unweighted + weighted — fp16 + fp32, 4 impls)
> - `GGML_OP_ADD` (TileLang, conditional on `GGML_TRITON_HAS_TILELANG`)
> - `GGML_OP_MUL_MAT` (CUTLASS, conditional on `GGML_TRITON_HAS_CUTLASS`)
```

This mirrors the "rms_norm" entry you'll see in `scripts/kernel_registry.json` after Tasks 2-5 land.

- [ ] **Step 2: Commit**

```bash
git add docs/development/test-pyramid.md
git commit -m "docs: mark RMSNorm as covered op in test-pyramid (B.1)

B.1 of docs/development/ROADMAP.md. The test-pyramid's Level-3 row
now lists which ggml ops have active ggml-triton AOT providers,
making it easier for future contributors to find the right test
target when working on a new op."
```

---

## Stage 1 exit criteria

- [ ] `test-triton-registry` exits 0 with all 4 asserts passing
- [ ] `test-backend-ops RMS_NORM` exits 0 on CPU-only (existing CPU provider path)
- [ ] The global and per-context ggml-triton registries each contain 5 RMS_NORM impls:
  - `cpu_rms_norm_f32` (existing)
  - `triton_rms_norm_unweighted_fp16_sm80` (new)
  - `triton_rms_norm_weighted_fp16_sm80` (new)
  - `triton_rms_norm_unweighted_fp32_sm80` (new)
  - `triton_rms_norm_weighted_fp32_sm80` (new)
- [ ] MiniMind-3 smoke (Level 2) runs end-to-end; if the dispatcher routed any RMS_NORM to the new triton AOT provider, the DEBUG log shows `triton_rms_norm_*`. (On the CPU-only box, the placeholder CUBIN means the kernel itself does no real work; the dispatcher + launcher + provider path IS exercised.)

## Stage 2 — Promote `eps` to a runtime launcher parameter (deferred, on GPU host)

Once a GPU host is available (and `scripts/compile_kernels.py` is patched for Triton 3.7.0 per the Phase 0 audit §0.4 follow-up), the next iteration is:

1. **Driver extension** (`scripts/compile_kernels.py`):
   - **Versioned launcher naming convention** (per Oracle §8): introduce a `_v2` suffix on the rms_norm launcher names. This isolates the new ABI from the Stage 1 ABI so the GELU/SiLU launchers (whose ABI must remain stable per their existing call sites) are never affected. The `LAUNCHER_SHAPES` map from Task 4 gets two new entries: `rms_norm_unweighted_v2` and `rms_norm_weighted_v2`, with shape `[..., ("float", "eps")]` appended to the parameter list.
   - Extend `_parse_signature` to recognize a new token class: `eps:fp32` (a runtime float parameter, not a constexpr). The token's leading-letter heuristic in the current parser must distinguish a literal `1e-06` (constexpr) from a named `eps:fp32` (runtime arg) — likely by adding a new branch on `:` in the token.
   - Extend `_emit_header` so the launcher signature adds a `float eps` parameter after the ptrs and before `N`.
   - Extend `_emit_source` so the `args[]` array adds `(void *) &eps`.
2. **DSL change** (`triton_kernels/rms_norm.py`): remove `EPS: tl.constexpr` from the signature; replace with `eps: float` runtime arg, passed to `tl.sqrt(mean + eps)`.
3. **Registry update** (`scripts/kernel_registry.json`): add NEW entries with the `_v2` suffix, e.g. `rms_norm_unweighted_v2`, with signature strings like `*fp16,*fp16,i32,1024,eps:fp32` (note: no trailing constexpr float; the `eps:fp32` token IS a runtime arg). Do NOT remove the Stage 1 entries — they are needed to keep the unweighted-only path working in environments where Stage 2 hasn't been re-deployed.
4. **Provider change** (`ggml-triton-provider-rmsnorm.cpp`): add 4 new impls (`triton_rms_norm_unweighted_v2_*` and `triton_rms_norm_weighted_v2_*` in fp16 and fp32) that read eps from `node->op_params` via `memcpy(&eps, node->op_params, sizeof(float));` and pass it to the new `_v2` launchers. REMOVE the `rms_norm_eps_matches_stage1` gate from the v2 supports predicates — the runtime eps is now passed in, so any value is acceptable.
5. **Regenerate launchers** with the updated `compile_kernels.py`. Verify GELU/SiLU launchers are STILL UNTOUCHED via `git diff` (per Oracle review).
6. **Re-run Tasks 11/12/13**: registry unit test, `test-backend-ops RMS_NORM` (now with `eps=1e-7` and other non-1e-6 values, if test cases are extended), MiniMind-3 smoke (Level 2).

**ABI compatibility note**: the `_v2` suffix convention is the surgical fix for the global-ABI problem Oracle §8 raised. It means Stage 1 and Stage 2 launchers can coexist (e.g. during a phased rollout), and the existing GELU/SiLU ABI is preserved bit-for-bit. The cost is naming ugliness (`_unweighted_v2_fp16_sm80` is a mouthful); a future cleanup could collapse both stages once Stage 1 is retired.

This is a separate sub-plan; it is referenced here only so Stage 1 does not bake in an `eps` constraint that would need a different launch path to undo.

## Out of scope (explicit)

- Real CUBIN generation (requires Triton 3.7.0 AOT API patch — Phase 0 audit §0.4 follow-up).
- Multi-block RMSNorm for `ne00 > 1024` (mentioned in the supports predicate; a follow-up plan will add a `BLOCK_SIZE` family of variants).
- Backward-warp RMSNorm (`GGML_OP_RMS_NORM_BACK` — used for training; not in MiniMind-3 inference).
- RoPE (B.2) and FlashAttn (B.3) — separate plans per ROADMAP.

## Verification matrix (summary)

| Test | Command | Stage 1 expected |
|---|---|---|
| Registry unit | `./build-master/bin/test-triton-registry` | exit 0, 4 asserts pass |
| Op correctness (CPU path) | `./build-master/bin/test-backend-ops RMS_NORM` | exit 0, all F32/F16 cases pass |
| Cross-backend (GPU path) | `./bin/test-backend-ops --backends CPU,TRITON` | deferred to GPU host |
| Level 2 smoke | `./build-master/bin/llama-cli -m minimind-3-F16.gguf -p "1+1等于几" -n 50` (sensible output); if `GGML_LOG_LEVEL=DEBUG` is set and a debug log line is added (see Optional follow-up below), the log shows `triton_rms_norm_*` for any RMSNorm nodes the dispatcher routes to the new AOT path |

## Optional follow-up (deferred, not blocking Stage 1)

Per Oracle §9a: `ggml-triton-dispatch.cpp:59` only logs UNSUPPORTED ops, not selected impls. A 2-line addition would surface selection events via `GGML_LOG_DEBUG`:

```cpp
// In ggml-triton-dispatch.cpp, after `ctx->op_registry.select(node)` returns non-null
GGML_LOG_DEBUG("ggml-triton: selected impl %s for op %s\n",
               impl->name, ggml_op_name(node->op));
```

This is **not required for Stage 1 exit** (the registry unit test + test-backend-ops already prove the dispatch path works). It is recommended for the next time someone debugs "why isn't my op hitting ggml-triton?" — left as a follow-up so this plan stays focused.

## Related docs

- `docs/development/ROADMAP.md` §3 Phase B.1 — the source-of-truth for this task
- `docs/development/test-pyramid.md` — Level 2/3/4 references
- `docs/development/ggml-custom-backends.md` §"Adding a new op" — the 4-step loop
- `docs/superpowers/plans/2026-06-09-phase-0-audit.md` §0.4 — the placeholder CUBIN caveat
- `ggml/src/ggml-cpu/ops.cpp:3758-3821` — bit-equivalent reference math
- `ggml/src/ggml-triton/ggml-triton-provider-cpu.cpp:464-510` — in-tree CPU provider
- `tests/test-backend-ops.cpp:3371-3424` and `:8233-8258` — existing test cases
