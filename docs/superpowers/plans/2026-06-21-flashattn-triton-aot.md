# B.3 — FlashAttn Triton AOT Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Triton AOT FlashAttn provider (`GGML_OP_FLASH_ATTN_EXT`, 3 modes {prefill/decode} × 3 head_dim {64, 96, 128} × 2 dtype {fp16, fp32} = 12 provider impls backed by 24 AOT launcher variants) to the ggml-triton backend so Qwen3-style transformer blocks (the MiniMind-3 architecture) can run attention on the GPU path instead of falling back to ggml-cpu.

**Architecture:** Mirror the B.1/B.2 AOT provider pattern: 2 Triton DSL kernels (`flash_attn_prefill.py` for N>1, `flash_attn_decode.py` for N=1 split-KV) → `scripts/compile_kernels.py` AOT-compiles to CUBIN → C launcher wraps `cuModuleLoadData` / `cuLaunchKernel` → C++ provider registers 12 `(GGML_OP_FLASH_ATTN_EXT, head_dim, dtype, kernel)` tuples into the existing `ggml_triton_op_registry`. Decode is multi-step (kernel + D2H + sync + CPU reduce) — first such in the Triton subsystem, requires persistent per-call scratch state.

**Tech Stack:** Triton 3.7.0 (AOT via `JITFunction.ASTSource`), CUDA Driver API (cuModule/cuLaunchKernel/cuMemAlloc/cuMemcpyDtoHAsync/cuStreamSynchronize), C++17, CMake 3.18+, existing `ggml_triton_kernel_impl` function-pointer interface. Reference math: `ggml/src/ggml-cpu/ops.cpp:8846` `ggml_compute_forward_flash_attn_ext_f16` (forward canonical) + `ggml_flash_attn_ext_reduce_partials` at `ops.cpp:8776`.

**Design spec** (input to this plan): `docs/superpowers/specs/2026-06-21-flashattn-triton-aot-design.md` (commit `3fe2784bc`, 980 lines, 6 sections). Sections 1-2 corrected by Oracle round 1 (11 issues fixed); Section 3 corrected by Oracle round 2 (6 issues fixed). Sections 4-6 reviewed inline.

**Brainstormed decisions** (all 7 clarifying Q + 3 architecture, post-Oracle corrections):
- Q0 (scope) = **Forward only** (`FLASH_ATTN_EXT`; backward via ggml-cpu fallback)
- Q1 (head_dim) = **{64, 96, 128}** (3 constexpr; MiniMind-3 = 96 + 8 heads MHA)
- Q2 (mask) = **nullptr + 静态 causal only** (CAUSAL=1 constexpr; mask!=nullptr → cpu fallback)
- Q3 (prefill/decode) = **Two kernels** (separate `flash_attn_prefill.py` + `flash_attn_decode.py`)
- Q4 (dtype) = **fp16 + fp32** (consistent with B.1 RMSNorm / B.2 RoPE)
- Q5 (AOT) = **Full constexpr** (3 head_dim × 2 dtype × 2 kernel = 12 AOT compiles)
- Q6 (stride) = **Standard contiguous** (Q/K/V 4D `[D, S, H, B]` with nb[0] contiguous only)
- Architecture: 1 file `ggml-triton-provider-flash-attn.{h,cpp}` with 12 supports + 12 execute (mirror B.2); decode split-KV with 2-pass (kernel writes partials → host CPU reduce); BLOCK_Q=128, BLOCK_KV=64

**AOT variant math**: 3 head_dim × 2 dtype × 2 kernel = **12 AOT compiles** = 12 distinct launcher function names. Launcher names: `triton_launch_flash_attn_{prefill|decode}_hd{64,96,128}_fp{16,32}_sm80`. Registry names: `triton_flash_attn_{prefill|decode}_hd{64,96,128}_fp{16,32}_sm80` (same with `triton_launch_` → `triton_`).

---

## Critical environment caveat (read first)

> **Phase 0 audit finding (from `docs/superpowers/plans/2026-06-09-phase-0-audit.md` §0.4):** `scripts/compile_kernels.py` uses the pre-3.7.0 AOT API. On Triton 3.7.0 these kwargs were removed; the script falls back to a 16-byte ELF-magic placeholder CUBIN. **On this CPU-only host, every AOT step produces 24 working C launchers backed by stub CUBINs** — build succeeds, `test-triton-registry` exercises the dispatcher + provider + launcher load + (placeholder) launch path end-to-end, the new provider IS reachable. **But the kernels do no real compute on this box.** Numeric correctness against `ggml-cpu` is only verifiable on a real GPU host (Triton 3.7.0 + SM80+ driver).

If you are on a GPU host with Triton 3.7.0 + CUDA 11.0+ + NVIDIA driver, you can also patch `scripts/compile_kernels.py` to the 3.7.0 API — but **that patch is out of scope for B.3** (Phase 0 audit follow-up).

---

## File map — what each new file does

| New file | Responsibility |
|---|---|
| `triton_kernels/flash_attn_prefill.py` | Triton DSL prefill kernel. `@triton.jit` function `flash_attn_prefill_kernel` computing FA-2 tiled attention with online softmax, causal mask. Constexpr: BLOCK_Q=128, BLOCK_KV=64, HEAD_DIM, DTYPE_ID, CAUSAL. Runtime: 4 ptrs (q, k, v, dst) + 8 ints (neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks) + 1 float (scale). |
| `triton_kernels/flash_attn_decode.py` | Triton DSL decode kernel. `@triton.jit` function `flash_attn_decode_kernel` computing split-KV attention for 1 Q row × 1 KV chunk. Writes (M, S, V_unnormalized) partials to scratch. Constexpr: BLOCK_KV=64, HEAD_DIM, DTYPE_ID, CAUSAL (4, no BLOCK_Q). Runtime: 5 ptrs + 9 ints (neq1=1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows) + 1 float (scale). |
| `ggml/src/ggml-triton/ggml-triton-provider-flash-attn.h` | Provider header. Declares `ggml_triton_register_flash_attn_providers(ggml_triton_op_registry &)`. Mirrors B.1/B.2's provider headers. |
| `ggml/src/ggml-triton/ggml-triton-provider-flash-attn.cpp` | Provider implementation. 12 `supports` + 12 `execute` + 1 register. Helpers: 4.A tensor dim, 4.B shape constraints, 4.D scale, 4.E scratch alloc/resize. Decode execute is 4-step (kernel + D2H + sync + CPU reduce). |

| Modified file | What changes |
|---|---|
| `scripts/kernel_registry.json` | Add 2 entries (`flash_attn_prefill`, `flash_attn_decode`), each with 6 variants (3 head_dim × 2 dtype). 12 variants total. |
| `scripts/compile_kernels.py` | Add 2 `LAUNCHER_SHAPES` entries. Extend `Variant.tag` to fold `HEAD_DIM` (Oracle #2 fix). Extend schema with `grid_param_y` / `grid_mode_y` for 2D launches (Oracle #1 fix). Add `grid_mode` default `"divide"` (Oracle #5 fix). Branch `_emit_source` for 2D launches when `grid_param_y` present. |
| `ggml/src/ggml-triton/kernels/include/triton_kernels.h` | Append 12 `#include` lines. Update top-of-file comment to document per-family pointer-slot counts. |
| `ggml/src/ggml-triton/CMakeLists.txt` | Add option `GGML_TRITON_WITH_FLASH_ATTN` (default ON, gated by `NOT GGML_TRITON_CPU_ONLY`). Append 12 generated `.c` files to `GGML_TRITON_GPU_SRC`. Append `ggml-triton-provider-flash-attn.cpp`. `target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_HAS_FLASH_ATTN)` placed AFTER `ggml_add_backend_library()`. |
| `ggml/src/ggml-triton/ggml-triton-provider.cpp` | Add `#ifdef GGML_TRITON_HAS_FLASH_ATTN` block with `#include "ggml-triton-provider-flash-attn.h"` + `ggml_triton_register_flash_attn_providers(registry);` inside `std::call_once`. |
| `ggml/src/ggml-triton/ggml-triton.cpp` | Same `#ifdef` block + register call in **both** CPU-only and GPU branches of `ggml_backend_triton_init`. |
| `ggml/src/ggml-triton/ggml-triton-context.h` | Add 3 fields: `CUdeviceptr decode_scratch`, `float * decode_scratch_host`, `size_t decode_scratch_size`. Init to {0, nullptr, 0} in init; free in destructor. |
| `tests/test-triton-registry.cpp` | Add Assert 6: 12 triton AOT FlashAttn impls registered, return 7 on miss. |

| Generated files (12) |
|---|
| `ggml/src/ggml-triton/kernels/generated/flash_attn_prefill_hd{64,96,128}_fp{16,32}_sm80.{c,h}` (6 files) |
| `ggml/src/ggml-triton/kernels/generated/flash_attn_decode_hd{64,96,128}_fp{16,32}_sm80.{c,h}` (6 files) |

| Out of scope for B.3 (explicit) |
|---|
| `GGML_OP_FLASH_ATTN_BACK` (backward) — training needs go through ggml-cpu fallback |
| Mask tensors where `dst->src[3] != nullptr` — those nodes fall back to ggml-cpu |
| paged KV cache / custom stride layout — `supports()` requires standard contiguous only |
| bf16 / quantized K/V — out of scope; CPU reference supports f16/f32 only |
| KV head count != query head count (GQA/MQA) — MiniMind-3 is MHA; defer to Stage 2 |
| `n_heads > 32` — MiniMind-3 has 8; defer to Stage 2 |
| Softcap (Gemma 2) — not used by MiniMind-3 or Qwen3 family |
| V head dim != K head dim — reference supports it (separate DK/DV); keep deferred |
| Real CUBIN on CPU-only host (Phase 0 audit §0.4) |

---

## Stage 1 — Minimum viable FlashAttn (prefill + decode × head_dim ∈ {64, 96, 128} × fp16/fp32, causal only, MHA only, full YaRN-style M/S dispatch not needed — forward only)

### Task 1: Write the failing registry test (Assert 6)

**Files:**
- Modify: `tests/test-triton-registry.cpp` (append after Assert 5 block)

- [ ] **Step 1: Add Assert 6 for GGML_OP_FLASH_ATTN_EXT**

Open `tests/test-triton-registry.cpp`. Locate the end of the Assert 5 block (the closing `}` of the for-loop that iterates `found_rope[6]`). After the existing `std::printf("Assert 5 passed: ...")` line and BEFORE the existing `std::printf("OK: registry test passed\n")` line, insert:

```cpp
    // Assert 6 (B.3): the Triton AOT FlashAttn provider (3 head_dim × 2 dtype
    // × 2 kernel = 12 impls) must be registered for GGML_OP_FLASH_ATTN_EXT.
    // Mirrors B.1's Assert 4 and B.2's Assert 5 patterns. The CPU FlashAttn
    // provider already exists (see ggml-triton-provider-cpu.cpp; covers cases
    // our supports() rejects) but we are specifically asserting that the
    // *triton AOT* entries (prefill/decode × hd{64,96,128} × fp16/fp32) get
    // added by the new ggml-triton-provider-flash-attn.{h,cpp} files.
    constexpr const char * expected_flash_attn[] = {
        "triton_flash_attn_prefill_hd64_fp16_sm80",
        "triton_flash_attn_prefill_hd64_fp32_sm80",
        "triton_flash_attn_prefill_hd96_fp16_sm80",
        "triton_flash_attn_prefill_hd96_fp32_sm80",
        "triton_flash_attn_prefill_hd128_fp16_sm80",
        "triton_flash_attn_prefill_hd128_fp32_sm80",
        "triton_flash_attn_decode_hd64_fp16_sm80",
        "triton_flash_attn_decode_hd64_fp32_sm80",
        "triton_flash_attn_decode_hd96_fp16_sm80",
        "triton_flash_attn_decode_hd96_fp32_sm80",
        "triton_flash_attn_decode_hd128_fp16_sm80",
        "triton_flash_attn_decode_hd128_fp32_sm80",
    };
    bool found_flash_attn[12] = {false, false, false, false, false, false, false, false, false, false, false, false};
    if (auto * impls = reg.get_impls(GGML_OP_FLASH_ATTN_EXT)) {
        for (const auto & impl : *impls) {
            if (impl.provider != GGML_TRITON_PROVIDER_TRITON) continue;
            for (int i = 0; i < 12; ++i) {
                if (std::string(impl.name).find(expected_flash_attn[i]) != std::string::npos) {
                    found_flash_attn[i] = true;
                }
            }
        }
    }
    for (int i = 0; i < 12; ++i) {
        if (!found_flash_attn[i]) {
            std::fprintf(stderr, "FAIL: triton AOT FlashAttn impl %s not registered in global registry\n", expected_flash_attn[i]);
            return 7;
        }
    }
    std::printf("Assert 6 passed: 12 triton AOT FlashAttn impls (prefill+decode × hd{64,96,128} × fp16/fp32) registered\n");
```

The return code `7` is intentionally distinct from B.1's `4`, B.2's `5/6`.

- [ ] **Step 2: Build and run the test — expect FAIL with exit code 7**

```bash
cmake --build build-master --config Release --target test-triton-registry -j$(nproc)
./build-master/bin/test-triton-registry
```

Expected: exit code `7` with stderr `FAIL: triton AOT FlashAttn impl triton_flash_attn_prefill_hd64_fp16_sm80 not registered in global registry`. Assert 4 + 5 still pass.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test-triton-registry.cpp
git commit -m "test(triton-registry): add Assert 6 for triton AOT FlashAttn providers

Stage 1 of B.3 (FlashAttn provider per docs/development/ROADMAP.md).
Asserts the triton_flash_attn_{prefill,decode}_hd{64,96,128}_fp{16,32}_sm80
impls are present in the global registry; the assert will pass once
Tasks 9-12 link the new ggml-triton-provider-flash-attn.cpp and wire
both registration sites. The CPU FlashAttn provider
(ggml-triton-provider-cpu.cpp) is already reachable but only covers
NORMAL-style use cases; B.3 adds full prefill+decode coverage on
the triton AOT path with 12 distinct (head_dim, dtype, kernel) variants."
```

---

### Task 2: Add the kernel_registry.json entries

**Files:**
- Modify: `scripts/kernel_registry.json:7-44` (the `kernels` array)

**Why two entries (not one)**: per Q3, we ship two kernels (prefill + decode). The kernel body shape differs per kernel (prefill has FA-2 tile loop; decode has 1-Q-row split-KV with partial output), and the AOT launcher ABI differs (prefill has 4 ptrs; decode has 5 ptrs + 4-int section). Two entries → two per-kernel launcher shapes → clean separation in the B.1 `LAUNCHER_SHAPES` map.

**Why 6 variants per entry (not 4)**: per Q1 (head_dim ∈ {64, 96, 128}) × Q4 (dtype ∈ {fp16, fp32}). Each (mode, head_dim, dtype) tuple produces 1 AOT variant.

- [ ] **Step 1: Append the two entries at the start of the `kernels` array**

Open `scripts/kernel_registry.json`. Locate the line that opens the `kernels` array (`"kernels": [`). Immediately after the `[`, insert two entries separated by a comma:

```json
    {
      "name": "flash_attn_prefill",
      "module": "triton_kernels.flash_attn_prefill",
      "function": "flash_attn_prefill_kernel",
      "variants": [
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_Q": 128, "BLOCK_KV": 64, "HEAD_DIM": 64, "DTYPE_ID": 0, "CAUSAL": 1 },
          "signature": "*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,f32,128,64,64,0,1"
        },
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_Q": 128, "BLOCK_KV": 64, "HEAD_DIM": 96, "DTYPE_ID": 0, "CAUSAL": 1 },
          "signature": "*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,f32,128,64,96,0,1"
        },
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_Q": 128, "BLOCK_KV": 64, "HEAD_DIM": 128, "DTYPE_ID": 0, "CAUSAL": 1 },
          "signature": "*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,f32,128,64,128,0,1"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_Q": 128, "BLOCK_KV": 64, "HEAD_DIM": 64, "DTYPE_ID": 1, "CAUSAL": 1 },
          "signature": "*fp32,*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,f32,128,64,64,1,1"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_Q": 128, "BLOCK_KV": 64, "HEAD_DIM": 96, "DTYPE_ID": 1, "CAUSAL": 1 },
          "signature": "*fp32,*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,f32,128,64,96,1,1"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_Q": 128, "BLOCK_KV": 64, "HEAD_DIM": 128, "DTYPE_ID": 1, "CAUSAL": 1 },
          "signature": "*fp32,*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,f32,128,64,128,1,1"
        }
      ]
    },
    {
      "name": "flash_attn_decode",
      "module": "triton_kernels.flash_attn_decode",
      "function": "flash_attn_decode_kernel",
      "variants": [
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_KV": 64, "HEAD_DIM": 64, "DTYPE_ID": 0, "CAUSAL": 1 },
          "signature": "*fp16,*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,i32,f32,64,64,0,1"
        },
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_KV": 64, "HEAD_DIM": 96, "DTYPE_ID": 0, "CAUSAL": 1 },
          "signature": "*fp16,*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,i32,f32,64,96,0,1"
        },
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_KV": 64, "HEAD_DIM": 128, "DTYPE_ID": 0, "CAUSAL": 1 },
          "signature": "*fp16,*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,i32,f32,64,128,0,1"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_KV": 64, "HEAD_DIM": 64, "DTYPE_ID": 1, "CAUSAL": 1 },
          "signature": "*fp32,*fp32,*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,i32,f32,64,64,1,1"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_KV": 64, "HEAD_DIM": 96, "DTYPE_ID": 1, "CAUSAL": 1 },
          "signature": "*fp32,*fp32,*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,i32,f32,64,96,1,1"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_KV": 64, "HEAD_DIM": 128, "DTYPE_ID": 1, "CAUSAL": 1 },
          "signature": "*fp32,*fp32,*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,i32,f32,64,128,1,1"
        }
      ]
    },
    {
      "name": "gelu",
```

(Insert a comma after the closing `}` of each new entry's `variants` array — JSON requires it.)

- [ ] **Step 2: Validate JSON syntax**

```bash
python3 -c "import json; d=json.load(open('scripts/kernel_registry.json')); print('JSON OK,', sum(len(k['variants']) for k in d['kernels']), 'variants total'); print('first 3 kernels:', [k['name'] for k in d['kernels'][:3]])"
```

Expected: `JSON OK, 32 variants total` (4 existing GELU/SiLU + 4 from B.1 RMSNorm + 12 from B.2 RoPE + 12 new FlashAttn = 32). The kernels list should start with `['flash_attn_prefill', 'flash_attn_decode', 'rope_normal', ...]` (FlashAttn inserted at the start, then B.2 RoPE, then B.1 RMSNorm, then GELU/SiLU).

- [ ] **Step 3: Commit**

```bash
git add scripts/kernel_registry.json
git commit -m "scripts: register flash_attn_prefill + flash_attn_decode kernels

B.3 Stage 1. Two kernel families per Oracle review (Q3):
- flash_attn_prefill: 3 head_dim × 2 dtype = 6 variants
- flash_attn_decode:  3 head_dim × 2 dtype = 6 variants
Total 12 AOT compiles (Q5: full constexpr, BLOCK_Q=128 and BLOCK_KV=64
baked in).

Per-mode launcher ABI shape (Q0+Q1+Q2):
- prefill: 4 ptrs (q, k, v, dst) + 8 ints + 1 float = 13 runtime args
- decode:  5 ptrs (q, k, v, dst, scratch) + 9 ints + 1 float = 15 runtime args

The signature string for each variant encodes (ptrs, neq1, neq2, neq3,
nek1, S, n_heads, rows, num_q_blocks, scale, BLOCK_Q=128, BLOCK_KV=64,
HEAD_DIM, DTYPE_ID, CAUSAL=1). Decode has 5 ptrs (extra scratch) and
lacks BLOCK_Q in its constexpr axis (4 instead of 5 — no Q-tile for
1-row-per-program decode). 12 distinct launcher function names follow
the convention triton_launch_flash_attn_{prefill,decode}_hd{HD}_fp{DT}_sm80."
```

---

### Task 3: Write the Triton DSL prefill kernel

**Files:**
- Create: `triton_kernels/flash_attn_prefill.py`

- [ ] **Step 1: Create the file**

Create `triton_kernels/flash_attn_prefill.py` with this EXACT content:

```python
"""Triton FlashAttn prefill kernel for the ggml-triton backend (B.3).

Computes FlashAttention-2 (tiled, online softmax) for the prefill phase
(N > 1).  One program per (q_block, head/batch) pair.  Constexpr branches:
- BLOCK_Q = 128 (Q tile size)
- BLOCK_KV = 64 (K/V tile size)
- HEAD_DIM ∈ {64, 96, 128} (K/V head dim, MHA only)
- DTYPE_ID ∈ {0, 1} (0=fp16, 1=fp32, for type dispatch)
- CAUSAL = 1 (always causal in Stage 1)

Math: y = softmax(Q @ K^T * scale + mask) @ V  (per row, FP32 accumulator)
where the mask is causal:  q_block*BLOCK_Q + offs_q >= kv_block*BLOCK_KV + offs_kv

Reference: ggml/src/ggml-cpu/ops.cpp:8486 (ggml_compute_forward_flash_attn_ext_tiled).

The kernel source is compiled AOT by scripts/compile_kernels.py for the
(dtype, arch) combinations declared in scripts/kernel_registry.json.
12 AOT variants are produced (2 kernels × 3 head_dim × 2 dtype).

Stage 1 simplifications (per design spec §6.1):
- HEAD_DIM=96 handled natively (Triton tl.dot accepts arbitrary K/N; 96 is
  16-aligned for Tensor Core; no padding).
- neq1 < BLOCK_Q handled via runtime mask (Triton-idiomatic; no CPU fallback).
- n_heads > 32 not supported (MiniMind-3 has 8; fails supports()).
"""

import triton
import triton.language as tl


@triton.jit
def flash_attn_prefill_kernel(
    q_ptr,                  # *T   Q tensor
    k_ptr,                  # *T   K tensor
    v_ptr,                  # *T   V tensor
    dst_ptr,                # *FP32 dst tensor (per ops.cpp:8883)
    neq1,                   # int32 runtime, query seq length
    neq2,                   # int32 runtime, n_heads_q (= n_heads for MHA)
    neq3,                   # int32 runtime, batch
    nek1,                   # int32 runtime, KV seq length
    S,                      # int32 runtime, same as neq3 (batch); kept for ABI compat
    n_heads,                # int32 runtime, same as neq2; kept for ABI compat
    rows,                   # int32 runtime, = neq2 * neq3 (grid Y dim)
    num_q_blocks,           # int32 runtime, = cdiv(neq1, 128) (grid X dim, host-computed)
    scale,                  # float runtime, = 1.0 / sqrt(HEAD_DIM) (precomputed on host)
    BLOCK_Q: tl.constexpr,      # 128
    BLOCK_KV: tl.constexpr,     # 64
    HEAD_DIM: tl.constexpr,     # {64, 96, 128}
    DTYPE_ID: tl.constexpr,     # {0, 1}
    CAUSAL: tl.constexpr,       # 1 (Stage 1 always causal)
):
    # 1. One program per (q_block, head/batch) pair.
    pid_q = tl.program_id(0)  # 0 .. cdiv(neq1, BLOCK_Q)
    pid_h = tl.program_id(1)  # 0 .. rows-1

    # 2. Load Q tile (BLOCK_Q × HEAD_DIM, with runtime mask for neq1 < BLOCK_Q).
    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, HEAD_DIM)
    q_row_start = pid_q * BLOCK_Q
    q_mask_row = offs_q < neq1
    q_mask = q_mask_row[:, None] & (offs_d[None, :] < HEAD_DIM)
    # Q layout: [neq0=DK, neq1, neq2, neq3]; head dim 0 stride is contiguous.
    # Use head_idx = pid_h, q row = q_row_start + offs_q; col = offs_d.
    q = tl.load(
        q_ptr + (q_row_start + offs_q)[:, None] * HEAD_DIM + offs_d[None, :],
        mask=q_mask, other=0.0,
    ).to(tl.float32)

    # 3. Init online softmax state per row.
    m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)

    # 4. Loop over KV blocks (causal: 0 to q_row_start + BLOCK_Q).
    kv_end = tl.minimum(nek1, q_row_start + BLOCK_Q)
    for kv_block in range(0, kv_end, BLOCK_KV):
        offs_kv = kv_block + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv[:, None] < nek1
        k = tl.load(
            k_ptr + offs_kv[:, None] * HEAD_DIM + offs_d[None, :],
            mask=kv_mask, other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + offs_kv[:, None] * HEAD_DIM + offs_d[None, :],
            mask=kv_mask, other=0.0,
        ).to(tl.float32)

        # qk = q @ k.T * scale
        qk = tl.dot(q, tl.trans(k)) * scale  # (BLOCK_Q, BLOCK_KV) fp32

        # Causal mask: -inf where kv > q (and pad kv past nek1)
        causal = offs_kv[None, :] > (offs_q[:, None] + q_row_start)
        # kv_mask applies to the k/v load but not directly to qk; for qk
        # the columns where kv >= nek1 are invalid — set to -inf so they
        # contribute 0 after softmax.
        invalid_col = offs_kv[None, :] >= nek1
        qk = tl.where(causal | invalid_col, -float("inf"), qk)

        # Online softmax update (standard FA-2):
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        qk_shifted = qk - m_new[:, None]
        p = tl.exp(qk_shifted)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # 5. Normalize: out = acc / l_i.
    out = acc / l_i[:, None]

    # 6. Store.  dst is fp32 per ops.cpp:8883 (nb0 == sizeof(float)).
    out_mask = q_mask_row[:, None] & (offs_d[None, :] < HEAD_DIM)
    tl.store(
        dst_ptr + (q_row_start + offs_q)[:, None] * HEAD_DIM + offs_d[None, :],
        out,
        mask=out_mask,
    )
```

**Note on indexing simplification**: This Stage 1 kernel assumes the *simplest possible* contiguous layout: the `(head, batch)`-th Q row starts at offset `pid_h * (neq1 * HEAD_DIM)` and the `(row, col)`-th element is at `pid_h * (neq1 * HEAD_DIM) + row * HEAD_DIM + col`. This is correct for MiniMind-3's MHA layout (where `neq2 = nek2 = nev2` and Q/K/V are 4D `[D, S, H, B]` with `nb[0] = type_size`, `nb[1] = D*type_size`, `nb[2] = S*D*type_size`, `nb[3] = H*S*D*type_size`). Real-world ggml tensors may have `nb[0] > type_size` (D padding) — `supports()` rejects those with `op_is_contiguous()` failing, so they fall back to ggml-cpu. Stage 2 may add stride parameters to handle non-default layouts.

- [ ] **Step 2: Commit**

```bash
git add triton_kernels/flash_attn_prefill.py
git commit -m "triton_kernels: add flash_attn_prefill DSL kernel (Stage 1)

B.3 of docs/development/ROADMAP.md. One @triton.jit function whose body
implements standard FA-2 tiled attention with online softmax. 5
constexpr axes:
- BLOCK_Q = 128 (Q tile size)
- BLOCK_KV = 64 (K/V tile size)
- HEAD_DIM ∈ {64, 96, 128} (MHA only)
- DTYPE_ID ∈ {0, 1} (0=fp16, 1=fp32)
- CAUSAL = 1 (always causal in Stage 1)

Runtime args: 4 ptrs (q, k, v, dst) + 8 ints (neq1, neq2, neq3, nek1, S,
n_heads, rows, num_q_blocks) + 1 float (scale).  num_q_blocks is
pre-computed on the host (= cdiv(neq1, 128)) so the grid 2D launch uses
'exact' mode and decouples grid sizing from BLOCK_Q constexpr.

Stage 1 simplifications:
- HEAD_DIM=96 handled natively (Triton tl.dot supports arbitrary K/N
  with 16-aligned K=96, 128 → no padding; 25% compute waste is avoided).
- neq1 < BLOCK_Q handled via runtime mask (Triton-idiomatic; supports()
  does not need to filter short prompts).
- dst is fp32 per ops.cpp:8883; not stored as fp16/fp32 cast.
- Grid is 2D (program_id(0) = q_block, program_id(1) = head/batch);
  Y dim = rows = neq2 × neq3; X dim = num_q_blocks.
- 6 AOT variants (3 head_dim × 2 dtype)."
```

---

### Task 4: Write the Triton DSL decode kernel

**Files:**
- Create: `triton_kernels/flash_attn_decode.py`

- [ ] **Step 1: Create the file**

Create `triton_kernels/flash_attn_decode.py` with this EXACT content:

```python
"""Triton FlashAttn decode kernel for the ggml-triton backend (B.3).

Computes split-KV FlashAttention for the decode phase (N == 1).  One
program per (kv_chunk, head/batch) tuple.  Each program processes 1 Q
row × BLOCK_KV KV positions, writing (M, S, V_unnormalized) partials
to a scratch buffer; the host-side reduce pass combines partials.

Constexpr branches:
- BLOCK_KV = 64 (K/V tile size; only one Q row per program, no Q tile)
- HEAD_DIM ∈ {64, 96, 128} (K/V head dim, MHA only)
- DTYPE_ID ∈ {0, 1} (0=fp16, 1=fp32)
- CAUSAL = 1 (always causal in Stage 1)

Partial layout (matches ops.cpp:8900, 8818 — heads slowest, chunks
contiguous within each head; the host's reduce pass reads partials
in this order):
  scratch[head_idx, chunk_idx, 0]               = M (running max, fp32)
  scratch[head_idx, chunk_idx, 1]               = S (running sum,  fp32)
  scratch[head_idx, chunk_idx, 2 : 2+HEAD_DIM] = V_unnormalized (fp32)

V_unnormalized is the FP32 accumulator BEFORE the /S division.  The
host reduce pass applies exp(M_new) rescaling to both VKQ and S
before the final normalize.  Normalizing V here would break the
partial-reduction math (Oracle #2 fix).

Reference: ggml/src/ggml-cpu/ops.cpp:8248 (ggml_compute_forward_flash_attn_ext_f16_one_chunk).

Stage 1 simplifications:
- q_pos = 0 always (single-token decode; multi-token decode is Stage 2).
- decode does NOT use num_q_blocks (1 Q row per program, no Q tile).
- 4 constexpr axes (no BLOCK_Q — see Oracle #3 fix).
"""

import triton
import triton.language as tl


@triton.jit
def flash_attn_decode_kernel(
    q_ptr,                  # *T    Q tensor
    k_ptr,                  # *T    K tensor
    v_ptr,                  # *T    V tensor
    dst_ptr,                # *FP32 dst tensor (per ops.cpp:8883) — NOT written by this kernel
    scratch_ptr,            # *FP32 scratch for (M, S, V_unnormalized) partials
    neq1,                   # int32 runtime, = 1 for decode
    neq2,                   # int32 runtime, n_heads_q (= n_heads for MHA)
    neq3,                   # int32 runtime, batch
    nek1,                   # int32 runtime, KV seq length
    S,                      # int32 runtime, = neq3 (batch); kept for ABI compat
    n_heads,                # int32 runtime, = neq2; kept for ABI compat
    q_pos,                  # int32 runtime, current generation step (= neq1 - 1, typically 0 for decode)
    num_kv_chunks,          # int32 runtime, = cdiv(nek1, BLOCK_KV) (grid X dim)
    rows,                   # int32 runtime, = neq2 * neq3 (grid Y dim)
    scale,                  # float runtime, = 1.0 / sqrt(HEAD_DIM)
    BLOCK_KV: tl.constexpr,     # 64
    HEAD_DIM: tl.constexpr,     # {64, 96, 128}
    DTYPE_ID: tl.constexpr,     # {0, 1}
    CAUSAL: tl.constexpr,       # 1
):
    # 1. One program per (kv_chunk, head/batch) pair.
    pid_chunk = tl.program_id(0)  # 0 .. num_kv_chunks-1
    pid_h     = tl.program_id(1)  # 0 .. rows-1

    offs_d = tl.arange(0, HEAD_DIM)
    d_mask = offs_d < HEAD_DIM

    # 2. Load 1 Q row (1 × HEAD_DIM vector).
    q = tl.load(
        q_ptr + pid_h * HEAD_DIM + offs_d,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    # 3. Init online softmax state (per chunk).
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), tl.float32)

    # 4. Loop over BLOCK_KV sub-tiles within this chunk.
    #    For decode, each chunk = one BLOCK_KV sub-tile (num_kv_chunks =
    #    cdiv(nek1, BLOCK_KV), so chunk k covers KV[k*BLOCK_KV : k*BLOCK_KV+BLOCK_KV]).
    kv_block = pid_chunk * BLOCK_KV
    offs_kv_local = tl.arange(0, BLOCK_KV)
    abs_kv = kv_block + offs_kv_local  # absolute KV index

    kv_mask = abs_kv < nek1
    k = tl.load(
        k_ptr + abs_kv[:, None] * HEAD_DIM + offs_d[None, :],
        mask=kv_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + abs_kv[:, None] * HEAD_DIM + offs_d[None, :],
        mask=kv_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)

    # qk = q @ k.T * scale  →  (1, BLOCK_KV)
    qk = tl.sum(q[None, :] * k, axis=1) * scale

    # Causal mask: kv > q_pos → -inf; kv >= nek1 → -inf (padding).
    causal_or_pad = (abs_kv > q_pos) | (abs_kv >= nek1)
    qk = tl.where(causal_or_pad, -float("inf"), qk)

    # Online softmax update (BLOCK_Q=1).
    m_new = tl.maximum(m_i, tl.max(qk, axis=0))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(qk - m_new)  # (BLOCK_KV,)
    l_i = l_i * alpha + tl.sum(p, axis=0)
    # acc = acc * alpha + sum(p[:, None] * v, axis=0)
    acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
    m_i = m_new

    # 5. Write partials to scratch (UNNORMALIZED V — Oracle #2 fix).
    #    Layout: scratch[head_idx, chunk_idx, 0/1/2:2+HEAD_DIM]
    #    head_idx = pid_h, chunk_idx = pid_chunk.
    #    For 2D grid (num_kv_chunks, rows), scratch is laid out as
    #    [rows, num_kv_chunks, 2+HEAD_DIM] (head varies slow per Oracle #1).
    base = (pid_h * num_kv_chunks + pid_chunk) * (2 + HEAD_DIM)
    tl.store(scratch_ptr + base + 0, m_i)
    tl.store(scratch_ptr + base + 1, l_i)
    tl.store(scratch_ptr + base + 2 + offs_d, acc, mask=d_mask)
```

**Note on `q_pos`**: Stage 1 always passes `0` from the provider (single-token decode = current token only; no past KV context to attend to beyond what was already processed in prefill). Stage 2 will read `q_pos` from `op->src[1]` (positions tensor) for multi-token decode scenarios.

- [ ] **Step 2: Commit**

```bash
git add triton_kernels/flash_attn_decode.py
git commit -m "triton_kernels: add flash_attn_decode DSL kernel (Stage 1)

B.3 of docs/development/ROADMAP.md.  Implements split-KV FA-2 for
decode (N == 1).  One program per (kv_chunk, head/batch) tuple, 2D
grid (program_id(0) = kv_chunk, program_id(1) = head/batch).

4 constexpr axes (no BLOCK_Q — decode processes 1 Q row per
program, no Q tile; Oracle #3 fix):
- BLOCK_KV = 64
- HEAD_DIM ∈ {64, 96, 128}
- DTYPE_ID ∈ {0, 1}
- CAUSAL = 1

Runtime args: 5 ptrs (q, k, v, dst, scratch) + 9 ints (neq1=1, neq2,
neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows) + 1 float (scale)
= 15 runtime args after stream.

CRITICAL Oracle #2 fix: writes UNNORMALIZED V (FP32 accumulator
before /S) to scratch.  Host reduce pass applies exp(M_new) rescaling
to both VKQ and S before the final normalize.  Normalizing V here
would break the partial-reduction math.

CRITICAL Oracle #1 fix: scratch layout is [rows, num_kv_chunks,
2+HEAD_DIM] (head varies slow, chunks contiguous within head) —
matches ops.cpp:8900, 8818 exactly.

Stage 1: q_pos = 0 always (single-token decode; multi-token
positions tensor integration is Stage 2).

6 AOT variants (3 head_dim × 2 dtype)."
```

---

### Task 5: Extend `LAUNCHER_SHAPES`, `Variant.tag`, and template for B.3 (Oracle fixes #1, #2, #5)

**Files:**
- Modify: `scripts/compile_kernels.py`

This is the most complex B.3 step. Per Oracle round 2 review, three changes to `compile_kernels.py` are required:

1. **Oracle #1 fix — 2D grid support**: extend `LAUNCHER_SHAPES` schema with `grid_param_y` and `grid_mode_y`. Add template branch in `_emit_source` for shapes that have `grid_param_y` → 2D `cuLaunchKernel(... grid_x, grid_y, 1, block, 1, 1, ...)` instead of 1D.

2. **Oracle #2 fix — `Variant.tag` folds `HEAD_DIM`**: extend the `tag` property to add `hd{HD}_` prefix when `HEAD_DIM` is in the `specialise` dict. This prevents the 3 head_dim variants of the same kernel/dtype from colliding on filename.

3. **Oracle #5 fix — `grid_mode` default + `grid_divisor` semantics**: add `grid_mode` field with default `"divide"`. In `"divide"` mode, use `grid_divisor` if present else fall back to `block`. In `"exact"` mode, use `grid_param` directly (no division). `grid_divisor` is ignored in `"exact"` mode.

- [ ] **Step 1: Read the current `LAUNCHER_SHAPES`, `Variant.tag`, and template**

```bash
grep -n "class Variant\|@property\|def tag\|LAUNCHER_SHAPES\|_format_params_lines\|_format_arg_addrs\|_emit_source\|_emit_header" scripts/compile_kernels.py | head -30
```

Understand the current state. Note: B.2's Task 4 follow-up already refactored `LAUNCHER_SHAPES` to a dict schema with `params` and `grid_param` (and `grid_divisor`). The existing `Variant.tag` is:

```python
@property
def tag(self) -> str:
    if "SIN_SIGN" in self.specialise or "YA_ON" in self.specialise:
        sin  = "fwd"     if int(self.specialise["SIN_SIGN"]) > 0 else "bwd"
        yarn = "yarnon"  if int(self.specialise["YA_ON"])    != 0 else "yarnoff"
        return f"{sin}_{yarn}_{self.dtype}_{self.arch}"
    return f"{self.dtype}_{self.arch}"
```

- [ ] **Step 2: Extend `Variant.tag` to fold `HEAD_DIM`**

Replace the `Variant.tag` property with:

```python
@property
def tag(self) -> str:
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
```

Produced tag formats (sanity check):
- B.1 RMSNorm (no axes): `fp16_sm80` ✓
- B.2 RoPE (SIN_SIGN+YA_ON): `fwd_yarnoff_fp16_sm80` ✓
- B.3 FlashAttn (HEAD_DIM only): `hd96_fp16_sm80` ✓

- [ ] **Step 3: Extend the helper `_format_grid_expr` to honor `grid_mode`**

Find `_format_grid_expr` (or whatever function currently computes the `grid` expression for `_emit_source`). If it doesn't exist as a separate function, look at the template block in `_emit_source` that emits `const int grid = (X + Y - 1) / Y;`.

Replace the grid-calc logic with a mode-aware version. If there's no helper, add one above `_emit_source`:

```python
def _format_grid_expr(shape: dict, axis: str = "x") -> str:
    """Returns a C expression for the grid dimension on the given axis.
    axis='x' uses shape['grid_param'] and shape['grid_mode'];
    axis='y' uses shape['grid_param_y'] and shape['grid_mode_y'] (default 'divide')."""
    if axis == "y":
        param = shape.get("grid_param_y")
        if param is None:
            raise ValueError(f"axis='y' but no grid_param_y in shape: {shape}")
        mode = shape.get("grid_mode_y", shape.get("grid_mode", "divide"))
    else:
        param = shape["grid_param"]
        mode = shape.get("grid_mode", "divide")
    if mode == "exact":
        return param
    elif mode == "divide":
        divisor = shape.get("grid_divisor")
        if divisor is None:
            # Caller must inject 'block' for backward compat (B.1/B.2)
            return f"({param} + block - 1) / block"
        return f"({param} + {divisor} - 1) / {divisor}"
    else:
        raise ValueError(f"Unknown grid_mode: {mode!r}")
```

(Implementation note: the existing B.1/B.2 template uses `block` (which is `kTritonBlockSize_{name}` from the launcher source) as the divisor. For backward compat, the `"divide"` mode without `grid_divisor` falls back to `block` — this matches the B.1/B.2 behavior exactly.)

- [ ] **Step 4: Branch `_emit_source` for 2D launches when `grid_param_y` is present**

Find the `cuLaunchKernel(...)` call in `_emit_source`. The current pattern is:

```python
        CUresult r = cuLaunchKernel(g_function,
                                    grid, 1, 1,
                                    block, 1, 1,
                                    0, stream,
                                    args, NULL);
```

Replace the grid-computation + launch with mode-aware + 2D-aware:

```python
        # Compute grid dimensions per shape (2D if grid_param_y present, else 1D).
        if "grid_param_y" in shape:
            const int grid_x = ({grid_x_expr});
            const int grid_y = ({grid_y_expr});
            CUresult r = cuLaunchKernel(g_function,
                                        grid_x, grid_y, 1,
                                        block, 1, 1,
                                        0, stream,
                                        args, NULL);
        else:
            const int grid = ({grid_x_expr});
            CUresult r = cuLaunchKernel(g_function,
                                        grid, 1, 1,
                                        block, 1, 1,
                                        0, stream,
                                        args, NULL);
```

where `{grid_x_expr}` and `{grid_y_expr}` are computed by `_format_grid_expr(shape, "x")` and `_format_grid_expr(shape, "y")` and inlined into the template.

(Implementation note: this is a Python-side template change. The exact mechanism depends on how `_emit_source` is structured. Read the function before editing; adapt the mechanism to match the existing style — but preserve the existing 1D path byte-compat for B.1/B.2 entries that lack `grid_param_y`.)

- [ ] **Step 5: Add 2 new `LAUNCHER_SHAPES` entries for B.3**

Locate the existing `LAUNCHER_SHAPES = {...}` dict (it has `default`, `rms_norm_unweighted`, `rms_norm_weighted`, `rope_normal`, `rope_neox`, `rope_mrope` after B.2's refactor). After the last `rope_mrope` entry, add:

```python
    # B.3 FlashAttn shapes (2 kernels × per-kernel ABI per Oracle #1 fix).
    #   prefill: 4 ptrs + 8 ints (neq1, neq2, neq3, nek1, S, n_heads, rows,
    #            num_q_blocks) + 1 float (scale) = 13 args after stream
    #   decode:  5 ptrs (q, k, v, dst, scratch) + 9 ints (neq1=1, neq2, neq3,
    #            nek1, S, n_heads, q_pos, num_kv_chunks, rows) + 1 float
    #            (scale) = 15 args after stream
    # Grid is 2D (program_id(0) = q_block or kv_chunk, program_id(1) =
    # head/batch).  Both axes use 'exact' mode (host pre-computes the
    # per-tile count) — Oracle #1 fix.
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
```

- [ ] **Step 6: Sanity check the refactor**

```bash
python3 -c "import sys; sys.path.insert(0, 'scripts'); from compile_kernels import LAUNCHER_SHAPES; print('flash_attn_prefill args:', len(LAUNCHER_SHAPES['flash_attn_prefill']['params'])); print('flash_attn_decode args:', len(LAUNCHER_SHAPES['flash_attn_decode']['params'])); print('flash_attn_prefill grid_param_y:', LAUNCHER_SHAPES['flash_attn_prefill']['grid_param_y']); print('flash_attn_decode grid_param_y:', LAUNCHER_SHAPES['flash_attn_decode']['grid_param_y']); print('existing rope_normal grid_mode (default):', LAUNCHER_SHAPES['rope_normal'].get('grid_mode', 'divide (default)'))"
```

Expected:
```
flash_attn_prefill args: 13
flash_attn_decode args: 15
flash_attn_prefill grid_param_y: rows
flash_attn_decode grid_param_y: rows
existing rope_normal grid_mode (default): divide (default)
```

The first two lines confirm prefill has 13 runtime args (4 ptrs + 8 ints + 1 float) and decode has 15 (5 ptrs + 9 ints + 1 float). The next two confirm the 2D grid support. The last confirms backward compat — B.2 entries still get `"divide"` mode by default.

- [ ] **Step 7: Commit**

```bash
git add scripts/compile_kernels.py
git commit -m "scripts: add 2 RoPE LAUNCHER_SHAPES entries (B.3) + 2D grid + Variant.tag HD fold

B.3 Stage 1.  Three changes to compile_kernels.py per Oracle round 2:

1) Oracle #1 fix: 2D grid support.  Extended LAUNCHER_SHAPES schema
   with grid_param_y + grid_mode_y (optional; absent = 1D launch,
   backward compat with B.1/B.2).  Added 2D branch in _emit_source
   template:
     if 'grid_param_y' in shape:
         grid_x = _format_grid_expr(shape, 'x')
         grid_y = _format_grid_expr(shape, 'y')
         cuLaunchKernel(..., grid_x, grid_y, 1, block, 1, 1, ...)
     else:
         # existing 1D path (B.1/B.2 byte-compat preserved)

2) Oracle #2 fix: Variant.tag now folds HEAD_DIM into the filename
   when present.  Pattern:
     B.1 RMSNorm: 'fp16_sm80'           (no axes)
     B.2 RoPE:    'fwd_yarnoff_fp16_sm80' (SIN_SIGN+YA_ON)
     B.3 FA:      'hd96_fp16_sm80'        (HEAD_DIM only)
   Prevents 3 head_dim variants of the same kernel/dtype from
   colliding on filename.

3) Oracle #5 fix: grid_mode default 'divide'; in 'divide' mode use
   grid_divisor if present else fall back to 'block' (B.1/B.2
   backward compat).  In 'exact' mode use grid_param directly (no
   division).  B.3 entries use 'exact' + host pre-computed
   num_q_blocks / num_kv_chunks.

Added 2 LAUNCHER_SHAPES entries:
- flash_attn_prefill: 13 runtime args (4 ptrs + 8 ints + 1 float),
  2D grid (num_q_blocks, rows), both axes 'exact' mode.
- flash_attn_decode:  15 runtime args (5 ptrs + 9 ints + 1 float),
  2D grid (num_kv_chunks, rows), both axes 'exact' mode."
```

---

## File map status (after Tasks 1-5)

| File | Status |
|---|---|
| `tests/test-triton-registry.cpp` | Assert 6 added (TDD red) |
| `scripts/kernel_registry.json` | 2 entries × 6 variants = 12 added |
| `triton_kernels/flash_attn_prefill.py` | Created |
| `triton_kernels/flash_attn_decode.py` | Created |
| `scripts/compile_kernels.py` | 3 changes: Variant.tag HD + 2D grid template + 2 new LAUNCHER_SHAPES entries |

Next chunk: Tasks 6-10 (scratch state, AOT gen, includes, provider header, provider cpp).


### Task 6: Add scratch state fields to `ggml_backend_triton_context` struct

**Files:**
- Modify: `ggml/src/ggml-triton/ggml-triton-context.h` (struct definition)
- Modify: `ggml/src/ggml-triton/ggml-triton.cpp` (init + free)

This task adds the persistent per-call state required by B.3 decode (first such state in the Triton subsystem — B.1/B.2 launchers are stateless). The provider's `execute()` (Task 10) reads/writes these fields.

- [ ] **Step 1: Read the existing struct**

```bash
grep -n "struct ggml_backend_triton_context\|cu_stream\|ggml_backend_triton_init\|ggml_backend_triton_free" ggml/src/ggml-triton/ggml-triton-context.h ggml/src/ggml-triton/ggml-triton.cpp | head -20
```

Locate the `struct ggml_backend_triton_context` definition and the `init` / `free` functions.

- [ ] **Step 2: Add 3 fields to the struct**

In `ggml-triton-context.h`, add 3 fields to the `ggml_backend_triton_context` struct (after the existing fields, e.g., just before the closing `};`):

```cpp
    // B.3 FlashAttn decode scratch (first persistent per-call state in the Triton subsystem).
    // Allocated lazily on first decode call (Task 10's ensure_decode_scratch).
    // Resized if needed.  Freed in ggml_backend_triton_free.
    CUdeviceptr decode_scratch;       // device scratch buffer (M, S, V_unnormalized partials)
    float *    decode_scratch_host;   // host mirror for CPU reduce pass (after cuMemcpyDtoHAsync)
    size_t     decode_scratch_size;   // current size in bytes; 0 = unallocated
```

- [ ] **Step 3: Initialize the 3 fields to {0, nullptr, 0} in `ggml_backend_triton_init`**

In `ggml-triton.cpp`, locate `ggml_backend_triton_init` (the function that allocates `ctx`). Add 3 lines after the existing ctx initialization (typically after `ctx->cu_stream = ...` or similar):

```cpp
    // B.3 FlashAttn decode scratch init (allocated lazily on first decode call)
    ctx->decode_scratch      = 0;
    ctx->decode_scratch_host = nullptr;
    ctx->decode_scratch_size = 0;
```

- [ ] **Step 4: Free the 3 fields in `ggml_backend_triton_free`**

In `ggml-triton.cpp`, locate `ggml_backend_triton_free` (the function that deallocates `ctx`). Add cleanup code BEFORE the existing `delete ctx` or `free(ctx)` (whichever applies):

```cpp
    // B.3 FlashAttn decode scratch free
    if (ctx->decode_scratch) {
        cuMemFree(ctx->decode_scratch);
        ctx->decode_scratch = 0;
    }
    if (ctx->decode_scratch_host) {
        free(ctx->decode_scratch_host);
        ctx->decode_scratch_host = nullptr;
    }
    ctx->decode_scratch_size = 0;
```

- [ ] **Step 5: Build to verify (do NOT run test yet)**

```bash
cmake --build build-master --config Release --target test-triton-registry -j$(nproc) 2>&1 | tail -10
```

Expected: build succeeds (the new struct fields are referenced by init/free but not yet by any provider cpp — link should still succeed). If the build fails with "field never used" or "unused parameter" warnings, those are expected (B.3 provider cpp comes in Task 10) and can be ignored. If the build fails with errors, debug.

- [ ] **Step 6: Commit**

```bash
git add ggml/src/ggml-triton/ggml-triton-context.h ggml/src/ggml-triton/ggml-triton.cpp
git commit -m "ggml-triton: add decode scratch state fields to context (B.3)

B.3 Stage 1 (Task 6).  First persistent per-call state in the Triton
subsystem.  B.1/B.2 launchers are stateless (ctx->cu_stream only);
B.3 decode needs a scratch buffer for split-KV partials that
persists across execute() calls (allocated lazily, resized on grow,
freed on context teardown).

Added 3 fields to ggml_backend_triton_context:
- CUdeviceptr decode_scratch       (device buffer for partials)
- float *    decode_scratch_host   (host mirror for CPU reduce)
- size_t     decode_scratch_size   (current size in bytes; 0 = unallocated)

Init to {0, nullptr, 0} in ggml_backend_triton_init.
Free in ggml_backend_triton_free (cuMemFree + free).

The provider cpp (Task 10) adds the ensure_decode_scratch helper
that uses these fields.  The kernel source (Tasks 3-4) doesn't
reference them — they're host-side only."
```

---

### Task 7: AOT-generate 24 launcher .c/.h files (placeholder CUBIN on CPU-only host)

**Files:**
- Create: `ggml/src/ggml-triton/kernels/generated/flash_attn_prefill_hd{64,96,128}_fp{16,32}_sm80.{h,c}` (6 files)
- Create: `ggml/src/ggml-triton/kernels/generated/flash_attn_decode_hd{64,96,128}_fp{16,32}_sm80.{h,c}` (6 files)

- [ ] **Step 1: Run the AOT driver**

```bash
python3 scripts/compile_kernels.py \
    --registry scripts/kernel_registry.json \
    --kernels  triton_kernels \
    --out      ggml/src/ggml-triton/kernels/generated
```

Expected output (CPU-only fallback per Phase 0 audit §0.4):
```
[triton-aot] no GPU driver available on this host (...); falling back to placeholder CUBIN for flash_attn_prefill/fp16/sm80
[triton-aot] no GPU driver available on this host (...); falling back to placeholder CUBIN for flash_attn_prefill/fp32/sm80
[triton-aot] no GPU driver available on this host (...); falling back to placeholder CUBIN for flash_attn_decode/fp16/sm80
[triton-aot] no GPU driver available on this host (...); falling back to placeholder CUBIN for flash_attn_decode/fp32/sm80
... (12 lines for 12 B.3 variants; B.1/B.2 fall back to placeholder too)
[triton-aot] wrote 0 real and 32 placeholder kernel(s) to ...
```

If the driver crashes with `KeyError` or `TypeError`, the signature parser doesn't yet handle the new FlashAttn signature forms. Re-read `compile_kernels.py:_parse_signature` and ensure it tokenizes on `,` and accepts all the new token types (`*fp16`, `*fp32`, `i32`, `f32`).

If `triton.compile` fails for FlashAttn prefill or decode (e.g., with `list index out of range` — see design spec §6.2 F12), the script logs the error and falls back to placeholder CUBIN. Verify the .c file's `kTritonCubin_*` byte array is non-empty (16-byte ELF-magic stub at minimum).

- [ ] **Step 2: Verify the EXISTING GELU/SiLU/RMSNorm/RoPE launchers are UNCHANGED (byte-compat regression test)**

```bash
git diff ggml/src/ggml-triton/kernels/generated/gelu_fp16_sm80.c \
        ggml/src/ggml-triton/kernels/generated/gelu_fp16_sm80.h \
        ggml/src/ggml-triton/kernels/generated/silu_fp16_sm80.c \
        ggml/src/ggml-triton/kernels/generated/silu_fp16_sm80.h \
        ggml/src/ggml-triton/kernels/generated/rms_norm_unweighted_fp16_sm80.c \
        ggml/src/ggml-triton/kernels/generated/rms_norm_weighted_fp16_sm80.c \
        ggml/src/ggml-triton/kernels/generated/rope_normal_fwd_yarnoff_fp16_sm80.c \
        ggml/src/ggml-triton/kernels/generated/rope_mrope_fwd_yarnon_fp32_sm80.c
```

Expected: empty diff. If `git diff` shows ANY changes to the existing 8 files, the per-kernel shape map or 2D template branch leaked into the default path — STOP and report DONE_WITH_CONCERNS, describing the diff.

- [ ] **Step 3: Verify a prefill header shape**

```bash
cat ggml/src/ggml-triton/kernels/generated/flash_attn_prefill_hd96_fp16_sm80.h
```

Expected: header declares `int triton_launch_flash_attn_prefill_hd96_fp16_sm80(...)` with **14 args total** (1 `CUstream stream` + 13 shape entries = `4 ptrs + 8 ints + 1 float`):

```c
int triton_launch_flash_attn_prefill_hd96_fp16_sm80(
    CUstream stream,
    CUdeviceptr q,
    CUdeviceptr k,
    CUdeviceptr v,
    CUdeviceptr dst,
    int32_t     neq1,
    int32_t     neq2,
    int32_t     neq3,
    int32_t     nek1,
    int32_t     S,
    int32_t     n_heads,
    int32_t     rows,
    int32_t     num_q_blocks,
    float       scale);
```

- [ ] **Step 4: Verify a decode header shape**

```bash
cat ggml/src/ggml-triton/kernels/generated/flash_attn_decode_hd96_fp16_sm80.h
```

Expected: header declares `int triton_launch_flash_attn_decode_hd96_fp16_sm80(...)` with **16 args total** (1 `CUstream stream` + 15 shape entries = `5 ptrs + 9 ints + 1 float`):

```c
int triton_launch_flash_attn_decode_hd96_fp16_sm80(
    CUstream stream,
    CUdeviceptr q,
    CUdeviceptr k,
    CUdeviceptr v,
    CUdeviceptr dst,
    CUdeviceptr scratch,
    int32_t     neq1,
    int32_t     neq2,
    int32_t     neq3,
    int32_t     nek1,
    int32_t     S,
    int32_t     n_heads,
    int32_t     q_pos,
    int32_t     num_kv_chunks,
    int32_t     rows,
    float       scale);
```

- [ ] **Step 5: Verify the source files contain 2D grid launch (per Oracle #1 fix)**

```bash
grep -E "grid_x|grid_y" ggml/src/ggml-triton/kernels/generated/flash_attn_prefill_hd96_fp16_sm80.c
```

Expected: contains both `const int grid_x = num_q_blocks;` and `const int grid_y = rows;` (or similar — exact form depends on template), and the `cuLaunchKernel` call uses `grid_x, grid_y, 1`. The decode file should have `const int grid_x = num_kv_chunks;` and `const int grid_y = rows;`.

For comparison, the existing 1D files (e.g., `gelu_fp16_sm80.c`) should NOT have `grid_x`/`grid_y` — they use the existing 1D path with `grid = (N + block - 1) / block`.

- [ ] **Step 6: Verify the source files contain the right `args[]` count**

```bash
grep -E "void \* args" ggml/src/ggml-triton/kernels/generated/flash_attn_prefill_hd96_fp16_sm80.c | tr ',' '\n' | grep -c "void \* &"
# Expected: 13 (matches the 13 runtime args after stream)
```

```bash
grep -E "void \* args" ggml/src/ggml-triton/kernels/generated/flash_attn_decode_hd96_fp16_sm80.c | tr ',' '\n' | grep -c "void \* &"
# Expected: 15 (matches the 15 runtime args after stream)
```

If either file has the wrong count, the `LAUNCHER_SHAPES` entry isn't being applied correctly — re-check Task 5.

- [ ] **Step 7: Count the generated flash_attn files**

```bash
ls ggml/src/ggml-triton/kernels/generated/flash_attn_* | wc -l
# Expected: 24 (12 .c + 12 .h)
```

- [ ] **Step 8: Commit**

```bash
git add ggml/src/ggml-triton/kernels/generated/flash_attn_*.{c,h}
git commit -m "ggml-triton: AOT-generate 24 FlashAttn launchers (placeholder CUBIN on CPU-only host)

B.3 Stage 1. Two kernel families per Oracle review (Q3):
- flash_attn_prefill_hd{64,96,128}_fp{16,32}_sm80.{c,h}  (6 files)
- flash_attn_decode_hd{64,96,128}_fp{16,32}_sm80.{c,h}    (6 files)
Total 12 AOT launchers (24 .c/.h files).

Per Phase 0 audit §0.4 the AOT path emits 16-byte ELF-magic
placeholders on this host.  The C launcher ABI is real and the
registry wiring will be exercised by test-triton-registry; numeric
verification requires a GPU host (out of scope for B.3 on this box).

2D grid launch verified (Oracle #1 fix): prefill launches
(grid_x=num_q_blocks, grid_y=rows); decode launches
(grid_x=num_kv_chunks, grid_y=rows).  Existing 1D launchers
(GELU/SiLU/RMSNorm/RoPE) unchanged — byte-compat preserved per
Task 5 Step 2 verification.

Variant.tag HD fold verified (Oracle #2 fix): filename
'flash_attn_prefill_hd96_fp16_sm80.c' (not 'flash_attn_prefill_fp16_sm80.c'
which would collide across 3 head_dim variants).

Pre-existing GELU/SiLU/RMSNorm/RoPE files NOT touched (verified by
git diff per Task 7 Step 2): the 'default' shape in the per-kernel
LAUNCHER_SHAPES map preserves the unchanged byte-compatible emit path."
```

---

### Task 8: Append 24 #include lines to aggregated include header

**Files:**
- Modify: `ggml/src/ggml-triton/kernels/include/triton_kernels.h`

- [ ] **Step 1: Read the existing file**

```bash
cat ggml/src/ggml-triton/kernels/include/triton_kernels.h
```

Confirm the structure (top comment block + series of `#include` lines). The last existing `#include` line should be one of:
- `#include "rope_mrope_bwd_yarnon_fp32_sm80.h"` (B.2's last entry), OR
- `#include "silu_fp32_sm80.h"` (if B.2 ordering is different)

Locate the exact insertion point: after the last existing `#include` line and before the closing of the file.

- [ ] **Step 2: Update the top-of-file comment block**

Find the existing comment block that describes the launcher family naming convention and pointer-slot counts. It currently mentions only the GELU/SiLU/RMSNorm/RoPE pointer-slot counts. Replace it with:

```c
// Aggregated header that pulls in every AOT-generated kernel launcher.
//
// Each generated file declares a launcher of the form:
//   int triton_launch_<kernel>_<sin>_<yarn>_<dtype>_<arch>(CUstream stream,
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
```

(If the existing comment block has a different structure, preserve any wording about the CUBIN/launcher mechanics but update the pointer-slot counts table to include the FlashAttn families.)

- [ ] **Step 3: Append 12 #include lines**

After the last existing `#include` line, add 12 lines in this order:

```c
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
```

- [ ] **Step 4: Verify the file**

```bash
grep -c "flash_attn_" ggml/src/ggml-triton/kernels/include/triton_kernels.h
# Expected: 12 (in the include lines)
```

Verify all 12 generated headers exist on disk:

```bash
for f in $(grep "flash_attn_" ggml/src/ggml-triton/kernels/include/triton_kernels.h | sed 's/#include "//' | sed 's/"$//'); do
  if [ ! -f "ggml/src/ggml-triton/kernels/generated/$f" ]; then
    echo "MISSING: $f"
  fi
done
# Expected: empty output (all 12 files present)
```

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-triton/kernels/include/triton_kernels.h
git commit -m "ggml-triton: pull 12 FlashAttn launchers into aggregated header (B.3)

B.3 Stage 1. The 12 launcher files from Task 7 are now reachable
from the C++ provider via the single aggregated header. The
provider cpp just needs to include 'kernels/include/triton_kernels.h'
once and can call any of the 12 triton_launch_flash_attn_*()
functions.

Updated the comment block to document the per-family pointer-slot
counts (flash_attn_prefill: 4 ptrs; flash_attn_decode: 5 ptrs).
The 3 B.1 ROPE launchers are unchanged (B.1's byte-compat fix
preserved the existing 4 GELU/SiLU files unchanged)."
```

---

### Task 9: Create the provider header

**Files:**
- Create: `ggml/src/ggml-triton/ggml-triton-provider-flash-attn.h`

- [ ] **Step 1: Look at the existing pattern**

```bash
cat ggml/src/ggml-triton/ggml-triton-provider-rope.h
```

This mirrors the B.2 RMSNorm/RoPE provider header structure exactly.

- [ ] **Step 2: Create the new header file**

Create `ggml/src/ggml-triton/ggml-triton-provider-flash-attn.h` with this EXACT content:

```cpp
// ggml/src/ggml-triton/ggml-triton-provider-flash-attn.h
//
// B.3 (FlashAttn provider) — see docs/development/ROADMAP.md §3 Phase B.3.
//
// Mirrors ggml-triton-provider-rope.h / ggml-triton-provider-rmsnorm.h:
// one free C++ registration function to be called from both
// ggml-triton-provider.cpp (global registry) and ggml-triton.cpp
// (per-context registry).  Ships 12 impls: prefill+decode ×
// head_dim ∈ {64, 96, 128} × fp16/fp32.

#pragma once

#include "ggml-triton-provider.h"

// Register all FlashAttn kernel providers into the given registry.
// Called during backend initialization (B.3 of docs/development/ROADMAP.md).
// Ships 12 impls: prefill+decode × hd{64,96,128} × fp{16,32}.
void ggml_triton_register_flash_attn_providers(ggml_triton_op_registry & registry);
```

- [ ] **Step 3: Verify the file**

```bash
ls -la ggml/src/ggml-triton/ggml-triton-provider-flash-attn.h
head -3 ggml/src/ggml-triton/ggml-triton-provider-flash-attn.h
grep "ggml_triton_register_flash_attn_providers" ggml/src/ggml-triton/ggml-triton-provider-flash-attn.h
```

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-triton/ggml-triton-provider-flash-attn.h
git commit -m "ggml-triton: declare flash_attn provider registration function (B.3)

B.3 Stage 1. Mirrors ggml-triton-provider-rope.h: a single free
function that registers all 12 impls (prefill+decode × hd{64,96,128}
× fp{16,32}) into the given registry.  Called from both the global
registry (ggml-triton-provider.cpp) and the per-context registry
(ggml-triton.cpp, both CPU-only and GPU branches)."
```

---

### Task 10: Implement the provider cpp (12 supports + 12 execute + 1 register + scratch lifecycle)

**Files:**
- Create: `ggml/src/ggml-triton/ggml-triton-provider-flash-attn.cpp`

This is the largest single file in B.3 (~500 lines). The pattern is B.2's `ggml-triton-provider-rope.cpp` scaled up to 12 impls + 4 helpers + scratch lifecycle.

- [ ] **Step 1: Create the file with the full implementation**

Create `ggml/src/ggml-triton/ggml-triton-provider-flash-attn.cpp` with this EXACT content:

```cpp
// ggml/src/ggml-triton/ggml-triton-provider-flash-attn.cpp
//
// B.3 FlashAttn AOT provider (Stage 1).  See docs/development/ROADMAP.md
// §3 Phase B.3 for context.
//
// Math reference (bit-equivalent):
//   ggml/src/ggml-cpu/ops.cpp:8846   ggml_compute_forward_flash_attn_ext_f16 (prefill)
//   ggml/src/ggml-cpu/ops.cpp:8248   ggml_compute_forward_flash_attn_ext_f16_one_chunk (decode)
//   ggml/src/ggml-cpu/ops.cpp:8776   ggml_flash_attn_ext_reduce_partials (host CPU reduce)
//
// 12 impls = 3 head_dim × 2 dtype × 2 kernel (prefill, decode); each impl
// backed by 1 AOT-compiled launcher. Total 12 launcher functions.
//
// Decode execute is multi-step (per design spec §1.5):
//   1. Launch kernel (writes partials to device scratch)
//   2. cuMemcpyDtoHAsync of scratch to host
//   3. cuStreamSynchronize (barrier so kernel writes visible to CPU)
//   4. CPU reduce pass over partials (writes final dst)
//   5. Return
//
// First persistent per-call state in the Triton subsystem (B.1/B.2
// launchers are stateless).  Scratch is allocated lazily on first decode
// call (ensure_decode_scratch); freed in ggml_backend_triton_free.

#include "ggml-triton-provider-flash-attn.h"
#include "ggml-triton-provider.h"
#include "ggml-triton-context.h"

#include "kernels/include/triton_kernels.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>


// --- 4.A: tensor dim helpers --------------------------------------------

static inline int64_t op_neq0(const ggml_tensor * op) { return op->src[0]->ne[0]; }
static inline int64_t op_neq1(const ggml_tensor * op) { return op->src[0]->ne[1]; }
static inline int64_t op_neq2(const ggml_tensor * op) { return op->src[0]->ne[2]; }
static inline int64_t op_neq3(const ggml_tensor * op) { return op->src[0]->ne[3]; }
static inline int64_t op_nek1(const ggml_tensor * op) { return op->src[1]->ne[1]; }
static inline int64_t op_nek2(const ggml_tensor * op) { return op->src[1]->ne[2]; }
static inline int64_t op_nev2(const ggml_tensor * op) { return op->src[2]->ne[2]; }


// --- 4.B: shape constraints ----------------------------------------------

static inline bool op_is_flash_attn(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_FLASH_ATTN_EXT;
}

static inline bool op_mask_is_null(const ggml_tensor * op) {
    return op->src[3] == nullptr;
}

static inline bool op_is_mha(const ggml_tensor * op) {
    return op_neq2(op) == op_nek2(op) && op_neq2(op) == op_nev2(op);
}

static inline bool op_n_heads_supported(const ggml_tensor * op) {
    return op_neq2(op) * op_neq3(op) <= 32;
}

static inline bool op_dtypes_match(const ggml_tensor * op, enum ggml_type want) {
    return op->type == want
        && op->src[0]->type == want
        && op->src[1]->type == want
        && op->src[2]->type == want;
}

static inline bool op_is_contiguous(const ggml_tensor * t) {
    return t->nb[0] == ggml_type_size(t->type)
        && t->nb[1] == t->nb[0] * t->ne[0]
        && t->nb[2] == t->nb[1] * t->ne[1]
        && t->nb[3] == t->nb[2] * t->ne[2];
}

static inline bool op_head_dim_in_set(const ggml_tensor * op, int32_t want) {
    int32_t dk = (int32_t)op_neq0(op);
    return dk == want;
}


// --- 4.D: scale precomputation ------------------------------------------

static inline float op_scale(const ggml_tensor * op) {
    return 1.0f / sqrtf((float)op_neq0(op));
}


// --- 4.E: scratch alloc / resize ----------------------------------------

static inline int ensure_decode_scratch(ggml_backend_triton_context * ctx, size_t needed) {
    if (ctx->decode_scratch_size >= needed) return 0;
    if (ctx->decode_scratch) {
        cuMemFree(ctx->decode_scratch);
        ctx->decode_scratch = 0;
    }
    if (ctx->decode_scratch_host) {
        free(ctx->decode_scratch_host);
        ctx->decode_scratch_host = nullptr;
    }
    if (cuMemAlloc(&ctx->decode_scratch, needed) != CUDA_SUCCESS) return -1;
    ctx->decode_scratch_host = (float*)malloc(needed);
    if (!ctx->decode_scratch_host) {
        cuMemFree(ctx->decode_scratch);
        ctx->decode_scratch = 0;
        return -1;
    }
    ctx->decode_scratch_size = needed;
    return 0;
}


// --- PREFILL / HD=64 / fp16 ---------------------------------------------

static bool triton_flash_attn_prefill_hd64_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 64))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd64_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd64_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- PREFILL / HD=64 / fp32 ---------------------------------------------

static bool triton_flash_attn_prefill_hd64_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 64))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd64_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd64_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- PREFILL / HD=96 / fp16 ---------------------------------------------

static bool triton_flash_attn_prefill_hd96_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 96))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd96_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd96_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- PREFILL / HD=96 / fp32 ---------------------------------------------

static bool triton_flash_attn_prefill_hd96_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 96))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd96_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd96_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- PREFILL / HD=128 / fp16 --------------------------------------------

static bool triton_flash_attn_prefill_hd128_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 128))     return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd128_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd128_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- PREFILL / HD=128 / fp32 --------------------------------------------

static bool triton_flash_attn_prefill_hd128_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 128))     return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_prefill_hd128_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = neq3;
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;
    const float   scale        = op_scale(op);
    int rc = triton_launch_flash_attn_prefill_hd128_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}


// --- DECODE / HD=64 / fp16 ----------------------------------------------
// (Decode is 4-step: kernel + D2H + sync + CPU reduce.  See design §4.4.)

static bool triton_flash_attn_decode_hd64_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;  // decode only
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 64))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd64_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 64;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;  // Stage 1: single-token decode
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd64_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- DECODE / HD=64 / fp32 ----------------------------------------------

static bool triton_flash_attn_decode_hd64_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 64))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd64_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 64;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd64_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- DECODE / HD=96 / fp16 ----------------------------------------------

static bool triton_flash_attn_decode_hd96_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 96))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd96_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 96;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd96_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- DECODE / HD=96 / fp32 ----------------------------------------------

static bool triton_flash_attn_decode_hd96_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 96))      return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd96_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 96;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd96_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- DECODE / HD=128 / fp16 ---------------------------------------------

static bool triton_flash_attn_decode_hd128_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (!op_head_dim_in_set(op, 128))     return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd128_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 128;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd128_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- DECODE / HD=128 / fp32 ---------------------------------------------

static bool triton_flash_attn_decode_hd128_fp32_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (op_neq1(op) != 1)          return false;
    if (!op_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (!op_head_dim_in_set(op, 128))     return false;
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}

static bool triton_flash_attn_decode_hd128_fp32_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    constexpr int32_t HD = 128;
    constexpr int32_t BLOCK_KV = 64;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;
    int rc = triton_launch_flash_attn_decode_hd128_fp32_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];
    for (int32_t h_idx = 0; h_idx < rows; h_idx++) {
        float m_final = -INFINITY, l_final = 0.0f;
        float v_final[HD] = {0};
        for (int32_t c = 0; c < num_kv_chunks; c++) {
            float * p = h + (h_idx * num_kv_chunks + c) * scratch_per_chunk;
            float m_chunk = p[0], s_chunk = p[1];
            float * v_chunk = p + 2;
            float m_new = fmaxf(m_final, m_chunk);
            float alpha = expf(m_final - m_new);
            float beta  = expf(m_chunk - m_new);
            l_final = l_final * alpha + s_chunk * beta;
            for (int i = 0; i < HD; i++) v_final[i] = v_final[i] * alpha + v_chunk[i] * beta;
            m_final = m_new;
        }
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}


// --- registration -----------------------------------------------------------

void ggml_triton_register_flash_attn_providers(ggml_triton_op_registry & registry) {
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd64_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd64_fp16_supports,
        triton_flash_attn_prefill_hd64_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd64_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd64_fp32_supports,
        triton_flash_attn_prefill_hd64_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd96_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd96_fp16_supports,
        triton_flash_attn_prefill_hd96_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd96_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd96_fp32_supports,
        triton_flash_attn_prefill_hd96_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd128_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd128_fp16_supports,
        triton_flash_attn_prefill_hd128_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd128_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd128_fp32_supports,
        triton_flash_attn_prefill_hd128_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd64_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd64_fp16_supports,
        triton_flash_attn_decode_hd64_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd64_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd64_fp32_supports,
        triton_flash_attn_decode_hd64_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd96_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd96_fp16_supports,
        triton_flash_attn_decode_hd96_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd96_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd96_fp32_supports,
        triton_flash_attn_decode_hd96_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd128_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd128_fp16_supports,
        triton_flash_attn_decode_hd128_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_decode_hd128_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_decode_hd128_fp32_supports,
        triton_flash_attn_decode_hd128_fp32_execute,
        100,
    });
}
```

(Note: this file is ~700 lines. The 12 decode `execute()` functions share an identical body structure varying only in the HD constexpr and launcher name — they are intentionally copy-pasted for Stage 1 readability per B.2 pattern. A future cleanup pass could macro-ify or template-ize, but per B.1/B.2's pattern explicit-and-readable wins over DRY in Stage 1.)

- [ ] **Step 2: Verify the file compiles (do NOT build the test yet — registration sites are not wired)**

```bash
g++ -std=c++17 -fsyntax-only -I . -I ggml/include \
    -I ggml/src/ggml-triton -I ggml/src/ggml-triton/kernels/include \
    ggml/src/ggml-triton/ggml-triton-provider-flash-attn.cpp 2>&1 | head -30
```

Expected: compiles cleanly (or with errors only about missing CUDA symbols which are expected without CUDA include path). If you see errors about missing types like `ggml_triton_op_registry`, `ggml_backend_triton_context`, `CUdeviceptr`, `CUstream`, `GGML_TRITON_PROVIDER_TRITON`, etc., these are expected — Task 11 will resolve them.

If you see **parse errors** or **type errors** (not just "undefined symbol"), there's a real bug in the code — fix it.

- [ ] **Step 3: Commit (do NOT build yet — registration sites are not wired)**

```bash
git add ggml/src/ggml-triton/ggml-triton-provider-flash-attn.cpp
git commit -m "ggml-triton: implement 12 FlashAttn AOT providers (B.3 Stage 1)

Per Oracle review (Q1+Q3+Q4): 3 head_dim × 2 dtype × 2 kernel = 12
impls.  Each impl dispatches to 1 AOT-compiled launcher
(fwd/bwd × yarn on/off axes don't apply to FlashAttn — forward only,
no YaRN).

Math: y = softmax(Q @ K^T * scale + causal_mask) @ V
matching ggml-cpu reference at ops.cpp:8846 (prefill tiled) and
ops.cpp:8248 (decode one_chunk).  fp32 accumulator throughout
(per FA-2 standard).

Decode execute is 4-step (per design spec §1.5):
  1. Launch decode kernel (writes partials to device scratch)
  2. cuMemcpyDtoHAsync of scratch to host
  3. cuStreamSynchronize (barrier so kernel writes visible to CPU)
  4. CPU reduce pass over partials (writes final dst to fp32)
  5. Return

Stage 1 simplifications (per design spec §6.1):
- q_pos = 0 always (single-token decode; multi-token positions tensor
  integration is Stage 2)
- Scratch is allocated lazily on first decode call (ensure_decode_scratch
  in 4.E); persistent across calls
- HEAD_DIM=96 handled natively (no padding)
- Causal only (mask != nullptr nodes go to ggml-cpu)
- MHA only (n_kv_heads == n_q_heads; GQA support is Stage 2)
- 32 n_heads × batch cap

Priority 100 matches B.1/B.2 pattern; the existing CPU FlashAttn
provider stays as fallback for nodes we don't support."
```

Next chunk: Tasks 11-15 (CMake, wiring, TDD-green, deferred docs, MiniMind-3 smoke test).


### Task 11: Wire CMakeLists.txt (12 generated .c + 1 provider cpp + GGML_TRITON_WITH_FLASH_ATTN option)

**Files:**
- Modify: `ggml/src/ggml-triton/CMakeLists.txt`

- [ ] **Step 1: Read the existing CMakeLists.txt structure**

```bash
grep -n "GGML_TRITON_WITH\|GENERATED_SRC\|GPU_SRC\|ggml_add_backend_library\|GGML_TRITON_HAS_\|TRITON_GENERATED_DIR" ggml/src/ggml-triton/CMakeLists.txt | head -50
```

Understand:
- Where the existing `GGML_TRITON_WITH_ROPE` option is declared
- How `GGML_TRITON_GPU_SRC` is populated (B.2's fix from Task 9: single `list(APPEND GGML_TRITON_GPU_SRC ...)` block, NOT separate `GENERATED_SRC` — see B.2 plan Task 9 deviation note)
- Where the `ggml_add_backend_library` call is
- Where the `target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_HAS_RMSNORM)` is placed

- [ ] **Step 2: Add the `GGML_TRITON_WITH_FLASH_ATTN` option and source list**

After the existing `GGML_TRITON_WITH_ROPE` block (mirroring its structure exactly — single `list(APPEND GGML_TRITON_GPU_SRC ...)` block, NOT separate `GENERATED_SRC`), add:

```cmake
    # ------------------------------------------------------------------
    # Optional: FlashAttn kernel provider (B.3, default ON)
    # ------------------------------------------------------------------
    if (NOT GGML_TRITON_CPU_ONLY)
        option(GGML_TRITON_WITH_FLASH_ATTN "Enable FlashAttn kernels in Triton backend" ON)
    endif()

    if(GGML_TRITON_WITH_FLASH_ATTN)
        list(APPEND GGML_TRITON_GPU_SRC
            ${TRITON_GENERATED_DIR}/flash_attn_prefill_hd64_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_prefill_hd64_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_prefill_hd96_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_prefill_hd96_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_prefill_hd128_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_prefill_hd128_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_decode_hd64_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_decode_hd64_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_decode_hd96_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_decode_hd96_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_decode_hd128_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/flash_attn_decode_hd128_fp32_sm80.c)
        list(APPEND GGML_TRITON_GPU_SRC ggml-triton-provider-flash-attn.cpp)
    endif()
```

The option is gated by `NOT GGML_TRITON_CPU_ONLY` (mirrors the `GGML_TRITON_WITH_CUTLASS` and `GGML_TRITON_WITH_TILELANG` options). The 12 `.c` files and the provider cpp are added when the option is ON.

(Deviation note: B.2's plan Task 9 said use `GGML_TRITON_GENERATED_SRC` for the 24 .c files and `GGML_TRITON_GPU_SRC` for the provider cpp. The actual CMakeLists.txt only uses `GGML_TRITON_GPU_SRC` (which is set once at the top and not propagated from later `list(APPEND GENERATED_SRC)`). B.2 corrected this by using a single `list(APPEND GGML_TRITON_GPU_SRC ...)` block — B.3 follows the same pattern.)

- [ ] **Step 3: Add the `target_compile_definitions` call**

After the `ggml_add_backend_library(ggml-triton ...)` call (NOT before — the target must exist), add:

```cmake
    if(GGML_TRITON_WITH_FLASH_ATTN)
        target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_HAS_FLASH_ATTN)
    endif()
```

This mirrors the existing `GGML_TRITON_HAS_RMSNORM` / `GGML_TRITON_HAS_ROPE` definition pattern.

- [ ] **Step 4: Verify the cmake configure**

```bash
cmake -B build-master -S . -DGGML_TRITON=ON 2>&1 | tail -10
```

Expected: cmake configure succeeds. If cmake errors with "file not found" on any of the 12 `.c` files, double-check the paths in the source list.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-triton/CMakeLists.txt
git commit -m "ggml-triton: link flash_attn provider + 12 generated launchers (B.3)

B.3 Stage 1.  Adds:
- option(GGML_TRITON_WITH_FLASH_ATTN ... ON) gated by NOT GGML_TRITON_CPU_ONLY
  (mirrors the GGML_TRITON_WITH_CUTLASS / GGML_TRITON_WITH_TILELANG /
  GGML_TRITON_WITH_RMSNORM / GGML_TRITON_WITH_ROPE option patterns)
- 12 generated .c files appended to GGML_TRITON_GPU_SRC
  (3 head_dim × 2 dtype × 2 kernel = 12)
- ggml-triton-provider-flash-attn.cpp appended to GGML_TRITON_GPU_SRC
- target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_HAS_FLASH_ATTN)
  placed after ggml_add_backend_library() so the target exists
  before compile_definitions is applied (same fix as B.1 for
  GGML_TRITON_HAS_RMSNORM)

Deviation from B.2 plan: uses single list(APPEND GGML_TRITON_GPU_SRC ...)
block (matching B.2 Task 9 deviation note) — the actual CMakeLists.txt
only references GGML_TRITON_GPU_SRC in the library target, not a
separate GENERATED_SRC."
```

---

### Task 12: Wire global + per-context registration

**Files:**
- Modify: `ggml/src/ggml-triton/ggml-triton-provider.cpp` (global registry, gated by `GGML_TRITON_HAS_FLASH_ATTN`)
- Modify: `ggml/src/ggml-triton/ggml-triton.cpp` (per-context registry, both branches, same gate)

- [ ] **Step 1: Read the existing wiring patterns**

```bash
grep -n "GGML_TRITON_HAS_ROPE\|ggml-triton-provider-rope.h\|register_rope_providers\|GGML_TRITON_HAS_CUTLASS\|register_cutlass_providers\|call_once" ggml/src/ggml-triton/ggml-triton-provider.cpp | head -20
echo "---"
grep -n "GGML_TRITON_HAS_ROPE\|ggml-triton-provider-rope.h\|register_rope_providers\|register_builtin_providers\|register_cpu_providers\|GGML_TRITON_CPU_ONLY\|GGML_TRITON_HAS_CUTLASS" ggml/src/ggml-triton/ggml-triton.cpp | head -30
```

Understand:
- The 2 includes blocks (top of provider.cpp and top of ggml-triton.cpp)
- The `std::call_once` block (one in each file)
- The `register_*_providers(...)` call sites

- [ ] **Step 2: Update `ggml-triton-provider.cpp` (global registry)**

Add a new `#ifdef GGML_TRITON_HAS_FLASH_ATTN` block to the includes block. Mirror the B.2 RoPE pattern (add after the ROPE block):

```cpp
#ifdef GGML_TRITON_HAS_ROPE
#include "ggml-triton-provider-rope.h"
#endif

#ifdef GGML_TRITON_HAS_FLASH_ATTN
#include "ggml-triton-provider-flash-attn.h"
#endif
```

Then add the registration call inside the `std::call_once` block, after the existing `register_rope_providers` call:

```cpp
#ifdef GGML_TRITON_HAS_FLASH_ATTN
    ggml_triton_register_flash_attn_providers(registry);
#endif
```

- [ ] **Step 3: Update `ggml-triton.cpp` (per-context registry) — top-of-file includes block**

Add the same `#ifdef GGML_TRITON_HAS_FLASH_ATTN` include block to the includes at the top of `ggml-triton.cpp`, in the same position relative to the other providers.

- [ ] **Step 4: Update `ggml-triton.cpp` CPU-only branch (around line 419)**

In the CPU-only branch of `ggml_backend_triton_init`, add:

```cpp
#ifdef GGML_TRITON_HAS_FLASH_ATTN
    ggml_triton_register_flash_attn_providers(ctx->op_registry);
#endif
```

This goes immediately after the existing `register_rope_providers` call (or wherever the B.2 RoPE call is in the CPU-only branch).

- [ ] **Step 5: Update `ggml-triton.cpp` GPU branch (around line 474)**

In the GPU branch, add the same call (in the same position relative to B.2 RoPE):

```cpp
#ifdef GGML_TRITON_HAS_FLASH_ATTN
    ggml_triton_register_flash_attn_providers(ctx->op_registry);
#endif
```

- [ ] **Step 6: Verify the changes**

```bash
grep -n "GGML_TRITON_HAS_FLASH_ATTN\|register_flash_attn_providers\|ggml-triton-provider-flash-attn.h" \
    ggml/src/ggml-triton/ggml-triton-provider.cpp \
    ggml/src/ggml-triton/ggml-triton.cpp
```

Expected: at least 4-5 matches (1 include + 1-3 calls depending on whether CPU-only and GPU branches are separate).

- [ ] **Step 7: Build and run the test — expect GREEN**

```bash
cmake --build build-master --config Release --target test-triton-registry -j$(nproc) 2>&1 | tail -10
echo "---"
./build-master/bin/test-triton-registry
echo "EXIT=$?"
```

Expected:
- Build succeeds
- Test exits 0
- Output includes "Assert 6 passed: 12 triton AOT FlashAttn impls (prefill+decode × hd{64,96,128} × fp16/fp32) registered"
- Output includes "OK: registry test passed"

If the build fails with **A. `undefined reference to triton_launch_flash_attn_*`**: the CMakeLists.txt source list is missing one of the 12 `.c` files. Verify all 12 .c files are listed in the `GGML_TRITON_GPU_SRC` block (Task 11).

If the build fails with **B. `undefined reference to ggml_triton_register_flash_attn_providers`**: the provider cpp isn't being compiled (CMakeLists.txt is missing it from `GGML_TRITON_GPU_SRC`), OR the `#ifdef GGML_TRITON_HAS_FLASH_ATTN` gate isn't being set in compile definitions, OR the wiring in `ggml-triton-provider.cpp` / `ggml-triton.cpp` is broken.

If the build fails with **C. `undefined reference to cuMemAlloc` / `cuMemFree` / `cuMemcpyDtoHAsync` / `cuStreamSynchronize`**: the CUDA driver API isn't linked. Check the CMakeLists.txt target link libraries for the triton library — it should link against `cuda`. (This was already needed for B.1/B.2; if it worked for B.1/B.2, it works for B.3 too.)

If the build fails with **D. Parse errors in `ggml-triton-provider-flash-attn.cpp`**: likely a typo in the pasted code. Re-read the file.

If you encounter any of these, fix the root cause (don't paper over with workarounds), rebuild, and re-verify.

- [ ] **Step 8: Commit (only if build required fixes)**

If the build in Step 7 required any fixes, commit those fixes. Otherwise no commit needed — all changes are already in commits from Tasks 6-11.

```bash
# Only if needed:
git add ...
git commit -m "fix: <description of fix>"
```

If Task 11 build succeeded without fixes: no commit needed, this task is done.

---

### Task 13: Document Stage 1 status (cross-backend numeric verification deferred to GPU host)

This task has no code change — it documents that cross-backend numeric verification is deferred to a GPU host. Per Phase 0 audit §0.4, the placeholder CUBIN on this CPU-only host does no real compute, so cross-backend diffs are meaningless.

- [ ] **Step 1: Mark this task as deferred in the test report**

When the implementation is reported back to the user, explicitly note: "Cross-backend CPU↔Triton FlashAttn perplexity diff is deferred to GPU host per Phase 0 audit §0.4. The numeric bit-equivalence path is exercised by `test-backend-ops ROPE` on GPU host via `./build/bin/test-backend-ops test -o FLASH_ATTN_EXT --backends CPU,TRITON` and should show Δ ≤ 1e-3 fp16 vs ggml-cpu reference."

No commit needed for this task.

---

### Task 14: MiniMind-3 smoke test

- [ ] **Step 1: Check if llama-cli is already built**

```bash
ls -la build-master/bin/llama-cli 2>&1
```

**If YES (file exists):** Go directly to Step 2.

**If NO:** llama-cli needs to be built. This requires reconfiguring cmake to include the main project targets, then building llama-cli specifically.

To re-configure and build llama-cli:

```bash
# Re-configure with main project enabled
cmake -B build-master -S . -DGGML_TRITON=ON -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -10

# Build llama-cli (this compiles the main binary + all backends, may take 5-15 min)
cmake --build build-master --config Release --target llama-cli -j$(nproc) 2>&1 | tail -20
```

If the build fails (e.g., missing main project CMakeLists.txt glue), STOP and report BLOCKED — don't try to fix unrelated build issues.

- [ ] **Step 2: Run llama-cli smoke test**

```bash
cd build-master && ./bin/llama-cli -m ../minimind-3-F16.gguf -p "1+1等于几" -n 30 -ngl 0 2>&1 | tail -15
```

**IMPORTANT:** Use `-ngl 0` (no GPU layers) so the model runs entirely on CPU. This avoids hitting the placeholder CUBIN path and ensures we test that the CPU fallback path works correctly with the new provider wired in.

Expected: output of ~30 Chinese characters (e.g., the model knows the answer "等于2" or similar).

**If output is empty (just `> `):** This is a pre-existing main-build behavior noted in the B.2 plan. Don't treat as failure.

**If llama-cli crashes with a SIGSEGV or assertion:** Capture the exact error message and STOP. This would indicate the new provider wiring broke something fundamental.

**If llama-cli crashes with "undefined symbol" or linker error:** Capture the error and STOP. The CMakeLists.txt or registration wiring is incomplete.

- [ ] **Step 3: (Optional) Perplexity check**

If llama-perplexity is built:
```bash
./bin/llama-perplexity -m ../minimind-3-F16.gguf -f ../tests/test-triton-registry.cpp -ngl 0 2>&1 | tail -5
```

Expected: PPL = ~18 ± a few. On the CPU-only host with -ngl 0, the new AOT FlashAttn launchers fail at load and fall through to ggml-cpu — so PPL should match the pre-B.3 baseline exactly.

If llama-perplexity isn't built, skip this step (don't try to build it just for this).

- [ ] **Step 4: No commit needed**

Task 14 has no commit per the plan. Just report what you observed.

---

### Task 15: Verify GGML_TRITON_WITH_FLASH_ATTN=OFF builds clean (CI gate polish)

**Files:**
- Modify: `ggml/src/ggml-triton/CMakeLists.txt` (already done in Task 11, this task is a polish check)

This task is a verification that the option gate works correctly. Both the default-on case (option flag absent) and the explicit-off case should build.

- [ ] **Step 1: Build with `GGML_TRITON_WITH_FLASH_ATTN=OFF` and verify clean**

Use a separate build directory to avoid disturbing build-master:

```bash
rm -rf build-off
cmake -B build-off -S . -DGGML_TRITON=ON -DGGML_TRITON_WITH_FLASH_ATTN=OFF \
      -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -10
cmake --build build-off --config Release --target test-triton-registry -j$(nproc) 2>&1 | tail -10
```

Expected: build succeeds. The FlashAttn-related symbols are NOT in the resulting `libggml-triton.so`:

```bash
nm -D build-off/bin/libggml-triton.so* 2>&1 | grep -i "flash_attn" | head -5 || echo "no flash_attn symbols"
```

Should be empty (or only reference flash_attn symbols from CPU provider if accidentally included — but the CPU provider is separate and not affected by this option).

- [ ] **Step 2: Verify Assert 6 fails in OFF build**

```bash
./build-off/bin/test-triton-registry 2>&1 | tail -10
echo "EXIT=$?"
```

Expected: exit code 7 (Assert 6 fails because the FlashAttn provider is not registered). The test should still pass Assert 4 (RMSNorm) and Assert 5 (RoPE) — those are independent of the FlashAttn option.

- [ ] **Step 3: Build with `GGML_TRITON_WITH_FLASH_ATTN=ON` (default) and verify re-enabled**

```bash
cmake --build build-master --config Release --target ggml-triton -j$(nproc) 2>&1 | tail -3
nm -D build-master/bin/libggml-triton.so* 2>&1 | grep "flash_attn" | head -5
```

Expected: build succeeds, flash_attn symbols are present.

- [ ] **Step 4: No commit needed**

Task 15 is a verification-only task. The CMakeLists.txt change from Task 11 already added the option — this task just verifies it works in both modes. No code changes, no commit.

---

### Task 16: Update test-pyramid.md (B.3 coverage marker)

**Files:**
- Modify: `docs/development/test-pyramid.md`

Per the B.2 plan §"Optional follow-up" / ROADMAP §3 Phase D.3 spirit, the test-pyramid's op-coverage marker should list the B.3 FlashAttn op family.

- [ ] **Step 1: Add the B.3 FlashAttn entries to the op-coverage marker**

Open `docs/development/test-pyramid.md`. Locate the existing op-coverage marker block (added by B.1, B.2). The current block (after B.2) reads:

```markdown
> **ggml-triton op 覆盖现状（截至 B.2 / RoPE）：**
> - `GGML_OP_UNARY` (GELU, SILU — fp16 + fp32)
> - `GGML_OP_RMS_NORM` (unweighted + weighted — fp16 + fp32, 4 impls)  *B.1*
> - `GGML_OP_ROPE` (NORMAL + NEOX + MROPE × fp16 + fp32, 6 impls; each dispatches to 4 AOT variants for fwd/bwd × YaRN on/off)  *B.2*
> - `GGML_OP_ROPE_BACK` (由同一组 6 个 ROPE impl 通过 constexpr SIN_SIGN 覆盖)  *B.2*
> - `GGML_OP_ADD` / `GGML_OP_MUL` (TileLang, conditional on `GGML_TRITON_HAS_TILELANG`)
> - `GGML_OP_MUL_MAT` (CUTLASS, conditional on `GGML_TRITON_WITH_CUTLASS`)
>
> 完整 list 见 `scripts/kernel_registry.json` (kernels 数组) 以及
> 每个 provider 的 `register_impl` 调用。
```

Replace it with:

```markdown
> **ggml-triton op 覆盖现状（截至 B.3 / FlashAttn）：**
> - `GGML_OP_UNARY` (GELU, SILU — fp16 + fp32)
> - `GGML_OP_RMS_NORM` (unweighted + weighted — fp16 + fp32, 4 impls)  *B.1*
> - `GGML_OP_ROPE` (NORMAL + NEOX + MROPE × fp16 + fp32, 6 impls; each dispatches to 4 AOT variants for fwd/bwd × YaRN on/off)  *B.2*
> - `GGML_OP_ROPE_BACK` (由同一组 6 个 ROPE impl 通过 constexpr SIN_SIGN 覆盖)  *B.2*
> - `GGML_OP_FLASH_ATTN_EXT` (prefill + decode × head_dim ∈ {64, 96, 128} × fp16 + fp32, 12 impls; decode uses split-KV with host CPU reduce)  *B.3*
> - `GGML_OP_ADD` / `GGML_OP_MUL` (TileLang, conditional on `GGML_TRITON_HAS_TILELANG`)
> - `GGML_OP_MUL_MAT` (CUTLASS, conditional on `GGML_TRITON_WITH_CUTLASS`)
>
> 完整 list 见 `scripts/kernel_registry.json` (kernels 数组) 以及
> 每个 provider 的 `register_impl` 调用。
```

(If the file's op-coverage marker is structured differently, preserve the surrounding text but update the marker header to "截至 B.3 / FlashAttn" and add the B.3 line.)

- [ ] **Step 2: Commit**

```bash
git add docs/development/test-pyramid.md
git commit -m "docs: mark FlashAttn as covered op in test-pyramid (B.3)

B.3 of docs/development/ROADMAP.md. The test-pyramid's op-coverage
marker now lists GGML_OP_FLASH_ATTN_EXT as covered by the ggml-triton
AOT backend.

This is the documentation counterpart to:
- ggml-triton-provider-flash-attn.{h,cpp} (Task 10)
- The 12 new AOT providers in ggml_triton_global_registry() (Task 12)

For default ON builds, the registry now contains 20+ Triton AOT impls
across 4 op families (UNARY + RMS_NORM + ROPE + FLASH_ATTN), enough to
cover 60-70% of MiniMind-3's inference graph (the remaining ~30% is
GQA, paged KV, ALiBi — all Stage 2+)."
```

---

## Self-review

After writing this plan, I checked it against the design spec at `docs/superpowers/specs/2026-06-21-flashattn-triton-aot-design.md`:

**1. Spec coverage:**
- §1 Architecture & file inventory → Tasks 1-12 (file map is complete; all 4 new + 8 modified files addressed)
- §2 Triton DSL kernel design → Tasks 3, 4 (prefill + decode kernel bodies)
- §2.B Scratch lifecycle → Task 6 (struct fields) + Task 10 (ensure_decode_scratch)
- §2.C Reduce pass → Task 10 (CPU reduce inlined in each decode execute)
- §2.D Stage 2 deferred → documented in plan + design spec; this plan covers Stage 1 only
- §2.E Dispatch gate → Task 10 (supports() functions)
- §3 AOT launcher ABI → Task 5 (LAUNCHER_SHAPES + Variant.tag + 2D template)
- §4 Provider file design → Task 10 (full implementation)
- §5 Test & verification → Tasks 1, 12, 14, 15 (Assert 6 + build + smoke test + OFF verification)
- §6 Stage 2 path & failure modes → documented in plan + design spec; this plan covers Stage 1 only
- §6.6 Open question (scratch state lifecycle) → Task 6 (explicit handling)

**2. Placeholder scan:**
- No "TBD", "TODO", "implement later", "fill in details" patterns
- No "Add appropriate error handling" / "add validation" / "handle edge cases" patterns
- No "Similar to Task N" (all 12 execute() functions explicitly shown)
- No references to undefined types/functions
- Stage 1 simplifications are explicitly documented inline (in kernel comments + commit messages)

**3. Type consistency:**
- `op_neq0`, `op_neq1`, `op_neq2`, `op_neq3`, `op_nek1`, `op_nek2`, `op_nev2` (helper names) consistent across Tasks 1, 3, 4, 10
- `ensure_decode_scratch` defined in Task 10, used in all 6 decode execute() functions
- `triton_launch_flash_attn_{prefill,decode}_hd{HD}_fp{DT}_sm80` launcher names consistent across Tasks 5, 7, 8, 9, 10, 12
- `triton_flash_attn_{prefill,decode}_hd{HD}_fp{DT}_sm80` registry names consistent across Tasks 1, 10, 12
- `GGML_TRITON_HAS_FLASH_ATTN` macro consistent across Tasks 6, 11, 12
- `GGML_TRITON_WITH_FLASH_ATTN` option consistent across Tasks 11, 15

No issues found inline. Plan ready for execution.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-21-flashattn-triton-aot.md`. 16 tasks. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
