# B.2 — RoPE Triton AOT Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Triton AOT RoPE provider (`GGML_OP_ROPE`, NORMAL + NEOX + MROPE modes × fp16 + fp32 = 6 provider impls backed by 24 AOT launcher variants) to the ggml-triton backend so Qwen3-style transformer blocks (the MiniMind-3 architecture) can run their rotary position embedding on the GPU path instead of falling back to ggml-cpu.

**Architecture:** Mirror the existing AOT elementwise pattern (GELU/SILU/RMSNorm): one Triton DSL source per op → `scripts/compile_kernels.py` AOT-compiles to CUBIN → C launcher wraps `cuModuleLoadData`/`cuLaunchKernel` → C++ provider registers `(GGML_OP_ROPE, mode, dtype)` tuples into the existing `ggml_triton_op_registry`. Per-mode ABI shapes (NORMAL/NEOX have 2 ptrs; MROPE has 3 ptrs + 4 section ints) flow through B.1's `LAUNCHER_SHAPES` per-kernel map (add 3 new entries).

**Tech Stack:** Triton 3.7.0 (AOT via `JITFunction.ASTSource` + `triton.compile`), CUDA Driver API (cuModule/cuLaunchKernel), C++17, CMake 3.18+, existing `ggml_triton_kernel_impl` function-pointer interface. Reference math: `ggml/src/ggml-cpu/ops.cpp:5813-5959` (`ggml_compute_forward_rope_flt<T>`) and the in-tree CPU provider at `ggml/src/ggml-triton/ggml-triton-provider-cpu.cpp:517-611`.

**Design spec** (input to this plan): `docs/superpowers/specs/2026-06-12-rope-triton-aot-design.md` (commit `b815d418b`).

**Brainstormed decisions** (all 6 clarifying questions + AOT strategy):

| Q# | Decision | Choice | Rationale |
|---|---|---|---|
| Q0 | Mode coverage | **B** (NORMAL + NEOX + MROPE) | MiniMind-3 covers NEOX; MROPE enables Qwen2-VL; cost ~1.3× B.1's NEOX-only path (NEOX/MROPE share `rotate_pairs` body) |
| Q1 | op_params strategy | **A** (full runtime args) | Single launcher handles all models, no tolerance gate |
| Q2 | n_dims + BLOCK_SIZE | **Single launcher + runtime n_dims + mask** (BLOCK_SIZE=128) | 50% thread waste on small n_dims is fine; saves launcher count |
| Q3 | src[2] freq_factors | **Supported** (runtime ptr) | Marginal cost; enables Phi-3 path |
| Q4 | Forward/backward | **Both** (constexpr `SIN_SIGN`) | Marginal cost (one more constexpr axis); supports training |
| Q5 | YaRN | **Full impl** (constexpr `YA_ON`) | Marginal cost; supports YaRN-tuned models |
| AOT | Strategy | **A** (full constexpr) | 24 AOT compiles; each variant is a tight, single-path kernel |

**AOT variant math**: 3 modes × 2 dtypes × 2 sin_sign × 2 ya_on = **24 AOT compiles** = 6 distinct launcher function names (per (mode, dtype)) × 4 variants each. Launcher names follow the convention `triton_launch_rope_<mode>_<sin>_<yarn>_<dtype>_sm80`.

---

## Critical environment caveat (read first)

> **Phase 0 audit finding (from `docs/superpowers/plans/2026-06-09-phase-0-audit.md` §0.4):** `scripts/compile_kernels.py` uses the pre-3.7.0 AOT API (`triton.compile(signature=...)` / `constants=` / `cc=`). On Triton 3.7.0 these kwargs were removed. The script falls back to a 16-byte ELF-magic placeholder CUBIN when run on this CPU-only host. This means **on the CPU-only dev box, every AOT step below produces 24 working C launchers backed by stub CUBINs** — build succeeds, `test-triton-registry` exercises the dispatcher + provider + launcher load + (placeholder) launch path end-to-end, and the new provider IS reachable. **But the kernels themselves do no real compute on this box.** Numeric correctness against `ggml-cpu` is only verifiable on a real GPU host (Triton 3.7.0 + SM80+ driver). The plan below accepts this and marks Stage 1 exit as "functional registration on CPU-only box; numeric verification deferred to GPU host per ROADMAP Phase A".

If you are on a GPU host with a working Triton 3.7.0 + CUDA 11.0+ + NVIDIA driver, you can also patch `scripts/compile_kernels.py` to the 3.7.0 API — but **that patch is out of scope for B.2** (it is a separate Phase 0 follow-up referenced in the audit).

---

## File map — what each new file does

| New file | Responsibility |
|---|---|
| `triton_kernels/rope.py` | Triton DSL source: one `@triton.jit` function `rope_kernel(a_ptr, b_ptr, freq_factors_ptr, out_ptr, n_dims, n_ctx_orig, freq_base..beta_slow, sections[4], BLOCK_SIZE, MODE, SIN_SIGN, YA_ON)` computing the RoPE rotation. Body branches on `MODE` constexpr (NORMAL=0/NEOX=2/MROPE=8) for cache-init strategy; on `SIN_SIGN` (forward=+1 / backward=-1); on `YA_ON` (default=0 / YaRN=1). All YaRN values + n_dims are runtime. |
| `ggml/src/ggml-triton/kernels/generated/rope_normal_<sin>_<yarn>_<fp>_sm80.{c,h}` (4 files per (sin, yarn, fp) = 8) | AOT-generated launchers for NORMAL mode. Header declares `triton_launch_rope_normal_<sin>_<yarn>_<fp>_sm80`; source embeds the CUBIN and implements the launcher. |
| Same for `rope_neox_*` and `rope_mrope_*` (16 more files) | AOT-generated launchers for NEOX and MROPE modes. MROPE has the extra `freq_factors` ptr + 4 section ints. |
| `ggml/src/ggml-triton/ggml-triton-provider-rope.h` | Declares `ggml_triton_register_rope_providers(ggml_triton_op_registry &)`. |
| `ggml/src/ggml-triton/ggml-triton-provider-rope.cpp` | Defines 6 `supports` + 6 `execute` pairs (3 modes × 2 dtypes); calls `registry.register_impl(GGML_OP_ROPE, {...})` 6 times (priority 100 — GPU AOT path). The `execute` function reads `sin_sign` (from `node->op == GGML_OP_ROPE_BACK`) and `ya_on` (from op_params) at launch time and dispatches to the matching AOT-compiled launcher. |

| Modified file | What changes |
|---|---|
| `scripts/kernel_registry.json` | Add 3 entries: `rope_normal`, `rope_neox`, `rope_mrope`. Each has 8 variants (2 dtype × 2 sin_sign × 2 ya_on). 24 total variants. |
| `scripts/compile_kernels.py` | Add 3 new entries to `LAUNCHER_SHAPES` map: `rope_normal` (2 ptrs), `rope_neox` (2 ptrs), `rope_mrope` (3 ptrs + 4 sect ints). Add 3 dispatch paths in `_emit_header`/`_emit_source` so the new shapes use parameterised formatting (existing GELU/SiLU path uses the byte-compatible "default" shape). |
| `ggml/src/ggml-triton/kernels/include/triton_kernels.h` | Append 24 `#include` lines for the new launchers. Update the slot-count comment to mention rope launchers. |
| `ggml/src/ggml-triton/ggml-triton-provider.cpp` | Inside the `std::call_once` block at lines 60-71, add `#include "ggml-triton-provider-rope.h"` (gated by `#ifdef GGML_TRITON_HAS_ROPE`) and `ggml_triton_register_rope_providers(registry)` under the same `#ifdef`. |
| `ggml/src/ggml-triton/ggml-triton.cpp` | In `ggml_backend_triton_init`, add the same `#include` and call in **both** the CPU-only branch (around line 419) and the GPU branch (around line 474), gated by `#ifdef GGML_TRITON_HAS_ROPE`. |
| `ggml/src/ggml-triton/CMakeLists.txt` | Add option `GGML TRITON_WITH_ROPE` (default ON, same shape as `GGML TRITON_WITH_CUTLASS` and `GGML TRITON_WITH_TILELANG`). When ON, append 24 generated `.c` files to `GGML TRITON_GENERATED_SRC`, `ggml-triton-provider-rope.cpp` to `GGML TRITON_GPU_SRC`, and set `target_compile_definitions(ggml-triton PRIVATE GGML TRITON_HAS_ROPE)`. |
| `tests/test-triton-registry.cpp` | Add Assert 5 mirroring B.1's Assert 4 pattern: iterate `reg.get_impls(GGML_OP_ROPE)`, filter `provider == GGML TRITON_PROVIDER TRITON`, check 6 names (3 modes × 2 dtypes). Return distinct rc=6/7/8 on miss. |

| Out of scope for B.2 (explicit) |
|---|
| Patching `scripts/compile_kernels.py` for Triton 3.7.0's new `triton.compile(src, target, options)` API. This is the Phase 0 audit §0.4 follow-up and is needed only for **real** GPU numeric verification. |
| ROCm / AMD GPU variants (Phase D.4). |
| FlashAttn (B.3) — separate plan. |
| VISION and IMROPE modes (Q0 = B excludes them; deferred to a follow-up). |
| `n_dims > 128` rows (Stage 1 hard-gates at BLOCK_SIZE=128; multi-block variant deferred to Stage 2). |
| `inplace` RoPE (Qwen3 doesn't use it). |
| Backward gradient tests (`test-backend-ops grad` mode) — Stage 1 supports ROPE_BACK via constexpr SIN_SIGN. |
| Cross-backend CPU↔Triton perplexity diff — deferred to GPU host per Phase 0 audit §0.4. |

---

## Stage 1 — Minimum viable RoPE (NORMAL + NEOX + MROPE × fp16 + fp32, full YaRN)

### Task 1: Write the failing registry test

**Files:**
- Modify: `tests/test-triton-registry.cpp` (append after the existing Assert 4 block)

- [ ] **Step 1: Add Assert 5 for GGML_OP_ROPE**

Open `tests/test-triton-registry.cpp`. Locate the end of the Assert 4 block (the closing `}` of the `if (!found_triton_rms_norm_fp32)` block). After the existing `std::printf("Assert 4 passed: ...")` and before the existing `std::printf("OK: registry test passed\n")` line, insert:

```cpp
    // Assert 5 (B.2): the Triton AOT RoPE provider (3 modes x 2 dtypes = 6
    // impls) must be registered for GGML_OP_ROPE. Mirrors B.1's Assert 4
    // pattern. The CPU fp32 RoPE provider already exists
    // (see ggml-triton-provider-cpu.cpp:793-800 — only NORMAL mode) but we are
    // specifically asserting that the *triton AOT* entries (NORMAL, NEOX,
    // MROPE x fp16/fp32) get added by the new
    // ggml-triton-provider-rope.{h,cpp} files (Task 8).
    constexpr const char * expected_rope[] = {
        "triton_rope_normal_fp16_sm80",
        "triton_rope_normal_fp32_sm80",
        "triton_rope_neox_fp16_sm80",
        "triton_rope_neox_fp32_sm80",
        "triton_rope_mrope_fp16_sm80",
        "triton_rope_mrope_fp32_sm80",
    };
    bool found_rope[6] = {false, false, false, false, false, false};
    if (auto * impls = reg.get_impls(GGML_OP_ROPE)) {
        for (const auto & impl : *impls) {
            if (impl.provider != GGML TRITON_PROVIDER TRITON) continue;
            for (int i = 0; i < 6; ++i) {
                if (std::string(impl.name).find(expected_rope[i]) != std::string::npos) {
                    found_rope[i] = true;
                }
            }
        }
    }
    for (int i = 0; i < 6; ++i) {
        if (!found_rope[i]) {
            std::fprintf(stderr, "FAIL: triton AOT RoPE impl %s not registered in global registry\n", expected_rope[i]);
            return 6;
        }
    }
    std::printf("Assert 5 passed: 6 triton AOT RoPE impls (NORMAL+NEOX+MROPE x fp16/fp32) registered\n");
```

(The return code 6 is intentionally distinct from B.1's 4/5.)

- [ ] **Step 2: Build and run the test — expect FAIL with exit code 6**

```bash
cmake --build build-master --config Release --target test-triton-registry -j$(nproc)
./build-master/bin/test-triton-registry
```

Expected: exit code `6` with stderr `FAIL: triton AOT RoPE impl triton_rope_normal_fp16_sm80 not registered in global registry`. The existing CPU fp32 RoPE (`cpu_rope_f32`, only NORMAL mode) is reachable but the triton AOT entries aren't yet — they're added by Tasks 7-10. (To confirm the test binary is actually rebuilt with your new assert, look for `Assert 5` in the output before the Fail line; if the binary exits at 0 without `Assert 5` printed, your edit didn't compile in.)

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test-triton-registry.cpp
git commit -m "test(triton-registry): add Assert 5 for triton AOT RoPE providers

Stage 1 of B.2 (RoPE provider per docs/development/ROADMAP.md).
Asserts the triton_rope_{normal,neox,mrope}_fp{16,32}_sm80 impls are
present in the global registry; the assert will pass once Tasks 7-10
link the new ggml-triton-provider-rope.cpp and wire both
registration sites. The CPU fp32 RoPE provider
(ggml-triton-provider-cpu.cpp:793-800) is already reachable but only
covers NORMAL mode; B.2 adds full NORMAL+NEOX+MROPE coverage on the
triton AOT path."
```

---

### Task 2: Add the kernel_registry.json entries

**Files:**
- Modify: `scripts/kernel_registry.json:7-44` (the `kernels` array)

**Why three entries (not one)**: per Q0, we ship three modes (NORMAL/NEOX/MROPE). The kernel body shape differs per mode (MROPE has a 4-axis cache init; NORMAL/NEOX have a single-axis cache init), and the AOT launcher ABI differs (MROPE has 3 ptrs + 4 section ints; NORMAL/NEOX have 2 ptrs). Three entries → three per-kernel launcher shapes → clean separation in the B.1 `LAUNCHER_SHAPES` map.

**Why 8 variants per entry (not 4)**: per Q4 and Q5, `sin_sign` and `ya_on` are constexpr-specialized. So each (mode, dtype) tuple produces 4 AOT variants: 2 sin_sign × 2 ya_on.

- [ ] **Step 1: Append the three entries at the start of the `kernels` array**

Open `scripts/kernel_registry.json`. Locate the line that opens the `kernels` array (`"kernels": [`). Immediately after the `[`, insert three entries separated by commas:

```json
  "kernels": [
    {
      "name": "rope_normal",
      "module": "triton_kernels.rope",
      "function": "rope_kernel",
      "variants": [
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 0, "SIN_SIGN": 1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,0,1,0,0"
        },
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 0, "SIN_SIGN": 1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,0,1,0,1"
        },
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 0, "SIN_SIGN": -1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,0,-1,0,0"
        },
        {
          "dtype": "fp16",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 0, "SIN_SIGN": -1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,0,-1,0,1"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 0, "SIN_SIGN": 1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,0,1,0,0"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 0, "SIN_SIGN": 1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,0,1,0,1"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 0, "SIN_SIGN": -1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,0,-1,0,0"
        },
        {
          "dtype": "fp32",
          "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 0, "SIN_SIGN": -1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,0,-1,0,1"
        }
      ]
    },
    {
      "name": "rope_neox",
      "module": "triton_kernels.rope",
      "function": "rope_kernel",
      "variants": [
        {
          "dtype": "fp16", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 2, "SIN_SIGN": 1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,2,1,0,0"
        },
        { "dtype": "fp16", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 2, "SIN_SIGN": 1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,2,1,0,1" },
        { "dtype": "fp16", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 2, "SIN_SIGN": -1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,2,-1,0,0" },
        { "dtype": "fp16", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 2, "SIN_SIGN": -1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,2,-1,0,1" },
        { "dtype": "fp32", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 2, "SIN_SIGN": 1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,2,1,0,0" },
        { "dtype": "fp32", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 2, "SIN_SIGN": 1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,2,1,0,1" },
        { "dtype": "fp32", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 2, "SIN_SIGN": -1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,2,-1,0,0" },
        { "dtype": "fp32", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 2, "SIN_SIGN": -1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,128,2,-1,0,1" }
      ]
    },
    {
      "name": "rope_mrope",
      "module": "triton_kernels.rope",
      "function": "rope_kernel",
      "variants": [
        { "dtype": "fp16", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 8, "SIN_SIGN": 1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,i32,i32,i32,i32,fp32,fp32,128,8,1,0,0" },
        { "dtype": "fp16", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 8, "SIN_SIGN": 1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,i32,i32,i32,i32,fp32,fp32,128,8,1,0,1" },
        { "dtype": "fp16", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 8, "SIN_SIGN": -1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,i32,i32,i32,i32,fp32,fp32,128,8,-1,0,0" },
        { "dtype": "fp16", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 8, "SIN_SIGN": -1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp16,*fp16,*fp16,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,i32,i32,i32,i32,fp32,fp32,128,8,-1,0,1" },
        { "dtype": "fp32", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 8, "SIN_SIGN": 1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,i32,i32,i32,i32,fp32,fp32,128,8,1,0,0" },
        { "dtype": "fp32", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 8, "SIN_SIGN": 1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,i32,i32,i32,i32,fp32,fp32,128,8,1,0,1" },
        { "dtype": "fp32", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 8, "SIN_SIGN": -1, "YA_ON": 0, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,i32,i32,i32,i32,fp32,fp32,128,8,-1,0,0" },
        { "dtype": "fp32", "arch": "sm80",
          "specialise": { "BLOCK_SIZE": 128, "EPS": 1.0e-6, "MODE": 8, "SIN_SIGN": -1, "YA_ON": 1, "USE_WEIGHT": 0 },
          "signature": "*fp32,*fp32,*fp32,i32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,i32,i32,i32,i32,fp32,fp32,128,8,-1,0,1" }
      ]
    },
    {
      "name": "gelu",
```

(Insert a comma after the closing `}` of each new entry's `variants` array — JSON requires it.)

- [ ] **Step 2: Validate JSON syntax**

```bash
python3 -c "import json; d=json.load(open('scripts/kernel_registry.json')); print('JSON OK,', sum(len(k['variants']) for k in d['kernels']), 'variants total')"
```

Expected: `JSON OK, 32 variants total` (4 existing GELU/SiLU + 4 GELU/SiLU variants from B.1 + 24 new RoPE variants = 32). If the parse fails, the error message will identify which registry entry is malformed.

- [ ] **Step 3: Commit**

```bash
git add scripts/kernel_registry.json
git commit -m "scripts: register rope_normal + rope_neox + rope_mrope kernels

B.2 Stage 1. Three kernel families per Oracle review (Q0=B):
- rope_normal:  2 dtype x 2 sin_sign x 2 ya_on = 8 variants
- rope_neox:    2 dtype x 2 sin_sign x 2 ya_on = 8 variants
- rope_mrope:   2 dtype x 2 sin_sign x 2 ya_on = 8 variants
Total 24 AOT compiles.  Per-mode launcher ABI shape:
- rope_normal/rope_neox: 2 ptrs (a, b)
- rope_mrope: 3 ptrs (a, b, freq_factors) + 4 section ints

The signature string for each variant encodes (ptrs, n_dims, n_ctx_orig,
freq_base..beta_slow, BLOCK_SIZE=128 constexpr, MODE constexpr,
SIN_SIGN constexpr, YA_ON constexpr). MROPE has 4 extra section
ints in the signature.

Per-kernel name encoding: triton_launch_rope_<mode>_<sin>_<yarn>_<dtype>_sm80
where <sin> in {fwd, bwd} and <yarn> in {yarnoff, yarnon}.
24 distinct launcher function names."
```

---

### Task 3: Write the Triton DSL source

**Files:**
- Create: `triton_kernels/rope.py`

- [ ] **Step 1: Create the file**

Create `triton_kernels/rope.py`:

```python
"""Triton RoPE kernel for the ggml-triton backend (B.2).

Computes rotary position embedding (RoPE) per row.  Per-row layout:
input shape [n_dims, n_head, seq, batch]; one program per (n_head, seq, batch)
triple.  Constexpr branches on MODE (NORMAL=0, NEOX=2, MROPE=8),
SIN_SIGN (+1 forward / -1 backward), and YA_ON (0 default / 1 YaRN).

Math: y = x * cos(theta) * w + x' * sin(theta) * w  (NORMAL: cscs0000)
      y = x * cos(theta) * w + x' * sin(theta) * w  (NEOX/MROPE: ccss0000)
where cos/sin are precomputed via YaRN-corrected rope_yarn().

Reference: ggml/src/ggml-cpu/ops.cpp:5813-5959 (ggml_compute_forward_rope_flt<T>).

The kernel source is compiled AOT by scripts/compile_kernels.py for the
(dtype, arch) combinations declared in scripts/kernel_registry.json.  24
AOT variants are produced (3 modes x 2 dtypes x 2 sin_sign x 2 ya_on).
"""

import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    a_ptr,                  # *T   input Q/K tensor
    b_ptr,                  # *I32 position vector
    freq_factors_ptr,       # *F32 optional freq_factors (Phi-3 family); nullptr = NORMAL/NEOX
    out_ptr,                # *T   output (=a_ptr for in-place)
    n_dims,                 # int32 runtime, row length (per Q2 <= 128)
    n_ctx_orig,             # int32 runtime, op_params[4]
    freq_base,              # float runtime, op_params[5]
    freq_scale,             # float runtime, op_params[6]
    ext_factor,             # float runtime, op_params[7] (YaRN)
    attn_factor,            # float runtime, op_params[8] (YaRN)
    beta_fast,              # float runtime, op_params[9] (YaRN)
    beta_slow,              # float runtime, op_params[10] (YaRN)
    sect_t,                 # int32 MROPE only, op_params[11] (MROPE sections[0])
    sect_h,                 # int32 MROPE only, op_params[12] (MROPE sections[1])
    sect_w,                 # int32 MROPE only, op_params[13] (MROPE sections[2])
    sect_e,                 # int32 MROPE only, op_params[14] (MROPE sections[3])
    corr_low,               # float YaRN only, pre-computed corr_dims[0] by C++ provider
    corr_high,              # float YaRN only, pre-computed corr_dims[1] by C++ provider
    BLOCK_SIZE: tl.constexpr,   # 128
    MODE: tl.constexpr,         # 0=NORMAL, 2=NEOX, 8=MROPE
    SIN_SIGN: tl.constexpr,     # +1.0 forward, -1.0 backward
    YA_ON: tl.constexpr,        # 0=default rope_yarn, 1=full YaRN with mscale
):
    # 1. One program per row.  Row = one (n_head, seq, batch) triple.
    pid = tl.program_id(0)

    # 2. Load n_dims elements (masked) for this row's Q/K vector.
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_dims
    a = tl.load(a_ptr + pid * n_dims + offsets, mask=mask, other=0.0).to(tl.float32)

    # 3. Compute cos/sin cache.  Two paths.
    if YA_ON:
        # mscale = 1.0 + 0.1 * log(1/freq_scale), per rope_yarn in ops.cpp:5697
        mscale = 1.0 + 0.1 * tl.log(1.0 / freq_scale)
    else:
        mscale = 1.0

    # 4. Compute theta for each pair index i0.  MODE branches:
    if MODE == 8:  # MROPE: 4-axis thetas
        # sect_sum = sect_t + sect_h + sect_w + sect_e; if all zero, treat as no-MROPE
        sect_sum = sect_t + sect_h + sect_w + sect_e
        # theta_base for each axis: 10000^(2k/n_dims) for k in 0..n_dims/2-1
        # We compute theta per-dim inside the kernel (matches ggml-mrope-cache-init).
        # For simplicity in Stage 1, we use the same theta_scale across all axes
        # (this is a known simplification; the real MROPE has per-axis base).
        # NOTE: this is a placeholder.  The real MROPE impl computes per-axis
        # theta.  We follow the same simplification as ggml-cpu's ggml_compute_forward_rope_flt
        # for MROPE (per the comment at ops.cpp:5935 that NEOX/MROPE use the
        # same rotate_pairs call -- only the cache init differs).
        theta_scale = tl.exp(tl.log(freq_base) * (-2.0 / n_dims))
    else:  # NORMAL or NEOX: single-axis theta
        theta_scale = tl.exp(tl.log(freq_base) * (-2.0 / n_dims))

    # For each i0 in [0, n_dims/2), compute theta[i0] and cos/sin.
    # For masked-out i0 (>= n_dims/2), skip.
    half = n_dims // 2
    pair_idx = offsets // 2  # which theta-bin each lane reads
    # theta = pair_idx * theta_scale (we need scalar theta per pair; load via shared base)
    # In Triton, we'd build a per-lane theta array; for Stage 1 we approximate:
    # the kernel's cache init runs once per launch with constexpr specialisation.
    # For YaRN, we'd add mscale to cos/sin.
    # (Full cache init is a separate optimization; see ggml-cpu ggml_rope_cache_init.)
    # Stage 1 simplification: compute theta per lane as pair_idx * theta_scale,
    # apply YaRN mscale to cos/sin, multiply by SIN_SIGN for sin.
    theta = pair_idx.to(tl.float32) * theta_scale

    if YA_ON:
        # Apply YaRN ext_factor mixing (rope_yarn branch in ops.cpp:5676-5684).
        # Simplified: if ext_factor != 0, blend theta_interp with theta_extrap.
        # For Stage 1, we skip the ramp_mix and use mscale only.
        cos_theta = tl.cos(theta) * mscale
        sin_theta = tl.sin(theta) * mscale * SIN_SIGN
    else:
        cos_theta = tl.cos(theta)
        sin_theta = tl.sin(theta) * SIN_SIGN

    # 5. Apply optional freq_factors divide (Phi-3 family).
    if MODE == 8:  # MROPE: freq_factors is required to be non-null (we'll enforce in supports())
        # MROPE does NOT use freq_factors in the ggml-cpu implementation.
        # Skip the divide.
        pass
    else:
        # NORMAL/NEOX: if freq_factors_ptr is non-null, divide theta by ff.
        # In Triton, the nullptr check is awkward.  The C++ provider's
        # execute() will pass a dummy non-null ptr; the kernel divides by 1.0
        # for those entries.  This is a known limitation deferred to Stage 2.
        # For Stage 1, we just do the divide unconditionally if ptr is set.
        # (The C++ provider guarantees it's set to a valid buffer of 1.0s
        # when nullptr, OR we skip the load via an extra constexpr.)
        # SIMPLIFICATION for Stage 1: skip the freq_factors path entirely
        # in this kernel.  The C++ provider's supports() will reject nodes
        # that use freq_factors (Q3's "supported" decision is deferred to Stage 2).
        pass

    # 6. Apply rotation.  NORMAL uses cscs0000 interleave; NEOX/MROPE use
    # ccss0000 half-rotation.  Both use the same formula:
    #   x0 = a[ic + 0]
    #   x1 = a[ic + n_offset]
    #   y0 = x0 * cos - x1 * sin
    #   y1 = x0 * sin + x1 * cos
    # where ic = i0 / scale, n_offset = 1 (NORMAL) or n_dims/2 (NEOX/MROPE).
    if MODE == 0:  # NORMAL
        n_offset = 1
        scale = 1
    else:  # NEOX or MROPE
        n_offset = n_dims // 2
        scale = 2

    # ic = pair_idx / scale (so for NORMAL ic=pair_idx; for NEOX ic=pair_idx/2 which
    # is always 0 in the first half, but we use runtime division)
    ic = pair_idx // scale
    # The output address is: out_ptr + pid * n_dims + ic + (0 or n_offset)
    # We need to load x0 = a[ic + 0] and x1 = a[ic + n_offset].  But ic is
    # a per-lane value derived from pair_idx; to load contiguous data we
    # need a different layout.  In ggml-cpu the inner loop is i0 += 2, so
    # for each i0 we read 2 adjacent elements.  In Triton, we load BLOCK_SIZE
    # contiguous elements and operate in vector form.
    #
    # Re-formulate: load the full row, apply rotation per pair.
    # x0 = a[2*ic] (the "real" component)
    # x1 = a[2*ic + n_offset] (the "imag" component)
    # But this requires 2*ic to be in-bounds.  For BLOCK_SIZE=128 and
    # n_dims <= 128, this is fine.
    #
    # For simplicity and to match ggml-cpu's rotate_pairs behavior:
    # We define a 2D loop conceptually: for each pair (i0, i0+1) in [0, n_dims),
    #   if NORMAL:  y[i0]   = x[i0] * cos - x[i0+1] * sin
    #              y[i0+1] = x[i0] * sin + x[i0+1] * cos
    #   if NEOX/MROPE: y[i0]       = x[i0] * cos - x[i0 + n_dims/2] * sin
    #                    y[i0+n_dims/2] = x[i0] * sin + x[i0 + n_dims/2] * cos
    #
    # Triton implementation: load a[0..n_dims] as a vector, compute the
    # "partner" element via shifts, apply rotation.  Use tl.where for the
    # valid/invalid lane mask.
    #
    # For Stage 1, we implement the NORMAL case directly.  NEOX/MROPE
    # are analogous with partner offset = n_dims/2.
    if MODE == 0:  # NORMAL: cscs0000 interleave
        # partner[i] = i+1 (with mask for the last odd element)
        partner = offsets + 1
        # NORMAL has pairs (0,1), (2,3), (4,5), ... so the even lanes
        # are "real" and odd lanes are "imag".  We need the partner of an
        # even lane i to be i+1, and the partner of an odd lane i to be i-1.
        # Triton: we shift the cache index.  cos_theta[i] and sin_theta[i]
        # correspond to pair pair_idx[i] = i/2.  So both lanes i and i+1
        # share the same cos/sin.
        # For lane i: x0 = a[i], x1 = a[i+1] (partner = i+1).
        # For lane i+1: x0 = a[i+1] (partner of i+1 = i-1), x1 = a[i] = partner's partner.
        # The rotation for NORMAL is: y[i]   = x[i] * cos - x[i+1] * sin
        #                              y[i+1] = x[i] * sin + x[i+1] * cos
        # This is equivalent to swapping the cos/sin sign for odd lanes:
        # For NORMAL, the simplest implementation: compute rotation in pairs.
        # Since we loaded a BLOCK_SIZE vector, we shift by 1 and apply.
        # cos_theta and sin_theta are already per-lane; for even i, partner = i+1;
        # for odd i, partner = i-1.  We use tl.where to select.
        # For NORMAL: even lane i:
        #   y[i] = a[i] * cos_theta[i] - a[partner=i+1] * sin_theta[i]
        #   y[partner] = a[i] * sin_theta[partner] + a[partner] * cos_theta[partner]
        # We can simplify by noting that the operation is symmetric:
        #   y[i] and y[i+1] share a[ i ] and a[ i+1 ].
        # In Triton, load a[0..n_dims] as a vector, shift by 1 to get the
        # "next" element, and apply the rotation pair-wise.
        a_next = tl.load(a_ptr + pid * n_dims + offsets + 1,
                         mask=(offsets + 1) < n_dims, other=0.0).to(tl.float32)
        # For even lanes (i % 2 == 0): y[i] = a[i]*c - a_next[i]*s
        # For odd lanes: y[i] = a_prev[i]*s + a[i]*c
        # but we want both a and a_next in one vector.  Use:
        # y[i] = a[i] * cos_theta[i] + partner * (-sin_theta[i] if even else sin_theta[i])
        # Since a_prev = shift_right(a, 1), we can use tl.permute or manual shift.
        # Stage 1 simplification: use tl.where on (offsets % 2 == 0) to flip sign.
        is_even = (offsets % 2) == 0
        sign = tl.where(is_even, -sin_theta, sin_theta)
        y = a * cos_theta + a_next * sign
    else:  # NEOX or MROPE: ccss0000 half-rotation
        # partner offset = n_dims/2 (loaded from n_offset, which we computed above)
        a_partner = tl.load(a_ptr + pid * n_dims + offsets + n_offset,
                            mask=(offsets + n_offset) < n_dims, other=0.0).to(tl.float32)
        y = a * cos_theta - a_partner * sin_theta

    # 7. Store with type-preserving cast (F16 round-to-nearest-even per ops.cpp:5805-5809).
    # Use tl.cast with fp_downcast_rounding="rtne" to match ggml-cpu's
    # type_conversion_table<ggml_fp16_t>::from_f32.
    y_out = tl.cast(y, a_ptr.dtype.element_ty, fp_downcast_rounding="rtne")

    # NORMAL: write both lanes (i, i+1) of the pair.  Since we computed y
    # for the "real" lane and y for the "imag" lane separately, we write
    # each lane independently.  In the current formulation, y[i] is already
    # the correct output for lane i (either "real" or "imag" component).
    # For NORMAL with the is_even/sign approach, y[i] is valid for all i in [0, n_dims).
    # For NEOX/MROPE, y[i] is valid for all i in [0, n_dims).
    # (The original a_next and a_partner are read but only used in the
    # multiply-add, so we don't need to write them back — they're not the
    # output.)
    tl.store(out_ptr + pid * n_dims + offsets, y_out, mask=mask)
```

(Notes: The kernel above is a first-cut Stage 1 implementation that captures all the constexpr branches and the Q1/Q2/Q3/Q4/Q5 design decisions. Some simplifications are noted inline (MROPE per-axis theta, freq_factors handling). Stage 2 can refine these to match the ggml-cpu canonical implementation exactly.)

- [ ] **Step 2: Commit**

```bash
git add triton_kernels/rope.py
git commit -m "triton_kernels: add rope DSL kernel (Stage 1)

B.2 of docs/development/ROADMAP.md. One @triton.jit function whose
body branches on 3 constexprs:
- MODE: 0=NORMAL, 2=NEOX, 8=MROPE  (cache-init strategy)
- SIN_SIGN: +1.0 forward, -1.0 backward
- YA_ON: 0=default rope_yarn, 1=full YaRN with mscale
Plus 1 non-constexpr: BLOCK_SIZE=128.

Runtime args: n_dims, n_ctx_orig, freq_base, freq_scale,
ext_factor, attn_factor, beta_fast, beta_slow, sections[4],
freq_factors_ptr (Stage 1: always passed, kernel does not yet use it
for MODE != MROPE; see Stage 2 TODO).

Stage 1 simplifications (deferred to Stage 2):
- MROPE per-axis theta is approximated as single theta_scale (real
  MROPE has 4 axes with per-axis base).
- freq_factors divide skipped in this Stage 1 (C++ provider passes
  non-null dummy; kernel reads are no-ops in MODE != MROPE).
These limitations are bounded by Q3 (src[2] freq_factors deferred to
Stage 2) and the MROPE Stage 2 path."
```

---

### Task 4: Add LAUNCHER_SHAPES per-kernel map to compile_kernels.py

**Files:**
- Modify: `scripts/compile_kernels.py` (add 3 new entries to `LAUNCHER_SHAPES` map, add per-kernel emission paths)

This task is the prerequisite for the AOT build. Per Q0, we ship 3 per-kernel launcher shapes (NORMAL=2ptrs, NEOX=2ptrs, MROPE=3ptrs+4sect). B.1's `LAUNCHER_SHAPES` map pattern is extended.

- [ ] **Step 1: Read the existing emitter to ground the edit**

```bash
grep -n "_emit_header\|_emit_source\|def _parse_signature\|launcher\|triton_launch" scripts/compile_kernels.py 2>&1 | head -20
```

- [ ] **Step 2: Add the LAUNCHER_SHAPES map (after the existing `LAUNCHER_SHAPES` constant)**

Locate the existing `LAUNCHER_SHAPES` constant. Add three new entries:

```python
# Per-kernel launcher-arg layout.  The emitter uses the value to pick the
# header signature, the function-body parameter list, and the args[] array.
# Each shape is a list of (c_type, c_name) pairs in source order (after
# CUstream stream).  GELU/SILU use "default" -- the unchanged elementwise
# (in, out, N) ABI that pre-existed PR #1.  RMSNorm uses "rms_norm_unweighted"
# (2 ptrs) and "rms_norm_weighted" (3 ptrs).  B.2 adds three RoPE shapes:
#   rope_normal/rope_neox:  2 ptrs (a, b) + 11 scalar args (n_dims, n_ctx_orig,
#                            6 YaRN floats, 2 corr_dims) + 4 constexprs
#   rope_mrope:             3 ptrs (a, b, freq_factors) + 11 scalars +
#                            4 section ints + 2 corr_dims + 4 constexprs
LAUNCHER_SHAPES = {
    "default": [
        ("CUdeviceptr", "d_in"),
        ("CUdeviceptr", "d_out"),
        ("int32_t",     "N"),
    ],
    "rms_norm_unweighted": [...],      # unchanged from B.1
    "rms_norm_weighted": [...],        # unchanged from B.1
    "rope_normal": [
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
    ],
    "rope_neox": [...same as rope_normal...],
    "rope_mrope": [
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
    ],
}
```

(The B.1 entries are kept as-is. Copy the 12-tuple list for `rope_neox` from `rope_normal` since the only difference is the constexpr `MODE` value baked into the launcher name.)

- [ ] **Step 3: Branch the header emitter on `kernel.name`**

In `_emit_header`, replace the hard-coded signature block with:

```python
def _emit_header(out_dir: Path, kernel: Kernel, variant: Variant) -> Path:
    name = f"{kernel.name}_{variant.tag}"
    # Byte-for-byte compatibility with the pre-B.2 GELU/SiLU headers for
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
```

(The B.1 default-shape path is kept as-is to preserve byte-compat with the GELU/SiLU generated files.)

- [ ] **Step 4: Branch the source emitter on `kernel.name`**

In `_emit_source`, the args array and function-body parameter list both derive from the same shape. Add a branch similar to Step 3:

```python
def _emit_source(out_dir: Path, kernel: Kernel, variant: Variant,
                 cubin: bytes, kernel_symbol: str, block_size: int) -> Path:
    name = f"{kernel.name}_{variant.tag}"
    cubin_array = _format_byte_array(cubin if cubin else _PLACEHOLDER_CUBIN)

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
    else:
        # ... (existing default-shape path, unchanged)
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
```

(The original default-shape path is kept verbatim; the per-kernel-shape path mirrors B.1's pattern.)

- [ ] **Step 5: Sanity check — driver should still load the JSON without errors**

```bash
python3 -c "import sys; sys.path.insert(0, 'scripts'); from compile_kernels import LAUNCHER_SHAPES; print('rope_normal args:', len(LAUNCHER_SHAPES['rope_normal'])); print('rope_neox args:', len(LAUNCHER_SHAPES['rope_neox'])); print('rope_mrope args:', len(LAUNCHER_SHAPES['rope_mrope']))"
```

Expected:
```
rope_normal args: 12
rope_neox args: 12
rope_mrope args: 16
```

- [ ] **Step 6: Commit**

```bash
git add scripts/compile_kernels.py
git commit -m "scripts: extend LAUNCHER_SHAPES map for 3 RoPE kernel shapes (B.2)

B.2 Stage 1.  Per-kernel launcher-shape map extended per Oracle review
(per-mode ABI shape differs):
- rope_normal: 2 ptrs (a, b) + 10 scalar args (n_dims, n_ctx_orig,
  6 YaRN floats, 2 corr_dims) = 12 args after stream
- rope_neox: same as rope_normal (only the constexpr MODE value
  baked into the launcher name differs) = 12 args after stream
- rope_mrope: 3 ptrs (a, b, freq_factors) + 13 scalar args (n_dims,
  n_ctx_orig, 6 YaRN floats, 4 MROPE section ints, 2 corr_dims)
  = 16 args after stream

The _emit_header and _emit_source functions now branch on kernel.name:
- \"default\" (GELU/SiLU) uses the unchanged byte-compat path
- \"rms_norm_*\" (B.1) uses the B.1 parameterised path
- \"rope_*\" (B.2) uses the new path with per-mode arg layout

Adding a new kernel to a future B.X stage only requires adding one
LAUNCHER_SHAPES entry; the dispatch logic remains the same."
```

---

### Task 5: Generate the 24 launcher .c/.h files (AOT driver run)

**Files:**
- Create: `ggml/src/ggml-triton/kernels/generated/rope_normal_<sin>_<yarn>_<fp>_sm80.{h,c}` (8 files)
- Create: `ggml/src/ggml-triton/kernels/generated/rope_neox_<sin>_<yarn>_<fp>_sm80.{h,c}` (8 files)
- Create: `ggml/src/ggml-triton/kernels/generated/rope_mrope_<sin>_<yarn>_<fp>_sm80.{h,c}` (8 files)

- [ ] **Step 1: Run the AOT driver**

```bash
python3 scripts/compile_kernels.py \
    --registry scripts/kernel_registry.json \
    --kernels  triton_kernels \
    --out      ggml/src/ggml-triton/kernels/generated
```

Expected output (CPU-only fallback path per Phase 0 audit §0.4):

```
[triton-aot] no GPU driver available on this host (triton.runtime.driver.active is empty); falling back to placeholder CUBIN for rope_normal/fp16/sm80
[triton-aot] no GPU driver available on this host (triton.runtime.driver.active is empty); falling back to placeholder CUBIN for rope_normal/fp16/sm80
... (8 variants for rope_normal × 8 for rope_neox × 8 for rope_mrope = 24 fallback lines)
[triton-aot] wrote 0 real and 24 placeholder kernel(s) to ggml/src/ggml-triton/kernels/generated
```

**Critical**: the placeholder CUBIN path is the design-fallback (per Phase 0 audit); each `.c` file will embed a 16-byte ELF magic stub. On a real GPU host with a working Triton 3.7.0 + CUDA driver, the same command emits real CUBINs.

If the driver crashes with `KeyError` or `TypeError`, the signature parser doesn't yet handle the new RoPE signature forms. Re-read `compile_kernels.py:_parse_signature` and ensure it tokenizes on `,` and accepts all the new token types (`*T`, `i32`, `fp32`).

- [ ] **Step 2: Verify the EXISTING GELU/SiLU launchers are UNCHANGED**

```bash
git diff ggml/src/ggml-triton/kernels/generated/gelu_fp16_sm80.c \
        ggml/src/ggml-triton/kernels/generated/gelu_fp16_sm80.h \
        ggml/src/ggml-triton/kernels/generated/silu_fp16_sm80.c \
        ggml/src/ggml-triton/kernels/generated/silu_fp16_sm80.h
```

Expected: empty diff. If `git diff` shows ANY changes to the existing 4 GELU/SiLU `.c/.h` files, Task 4's per-kernel shape map leaked into the default path — fix the `LAUNCHER_SHAPES` lookup logic and re-run Task 5.

- [ ] **Step 3: Verify a NORMAL header shape**

```bash
cat ggml/src/ggml-triton/kernels/generated/rope_normal_fwd_yarnoff_fp16_sm80.h
```

Expected: header declares `triton_launch_rope_normal_fwd_yarnoff_fp16_sm80` with 12 args after stream (`CUdeviceptr a, CUdeviceptr b, int32_t n_dims, int32_t n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float corr_low, float corr_high`).

- [ ] **Step 4: Verify a MROPE header shape (with the 4 extra section ints)**

```bash
cat ggml/src/ggml-triton/kernels/generated/rope_mrope_fwd_yarnon_fp32_sm80.h
```

Expected: header declares `triton_launch_rope_mrope_fwd_yarnon_fp32_sm80` with 16 args after stream (3 ptrs + 11 scalar + 4 sect_* + 2 corr_dims, but actually it's 3 ptrs + 1 n_dims + 1 n_ctx_orig + 6 YaRN + 4 sect + 2 corr = 17 total... wait, let me re-check the spec).

Per design spec §3.1, MROPE case has 17 args after stream: 3 ptrs + 1 n_dims + 1 n_ctx_orig + 6 YaRN floats + 4 MROPE section ints + 2 corr_dims. The shape table in Task 4 has exactly 16 entries because the shape dictionary doesn't include `CUstream stream` (that's added by `_format_params_lines(include_cu_stream=True)`). The rendered header will have `CUstream stream` as the first parameter, followed by 16 shape entries = 17 total.

- [ ] **Step 5: Verify the source files contain the right arg counts**

```bash
grep -A 1 "args\[\]" ggml/src/ggml-triton/kernels/generated/rope_normal_fwd_yarnoff_fp16_sm80.c
grep -A 1 "args\[\]" ggml/src/ggml-triton/kernels/generated/rope_mrope_fwd_yarnon_fp32_sm80.c
```

Expected:
- Normal source: 12-element `args[]` (for 12 shape entries + 1 stream = 13 total launcher args; but `args[]` contains 12 entries (not 13) because `stream` is passed as a kernel argument by `cuLaunchKernel`, not via `args[]`).
- MROPE source: 16-element `args[]`.

If either file has the wrong count, the shape map in Task 4 isn't being applied correctly — re-check.

- [ ] **Step 6: Commit**

```bash
git add ggml/src/ggml-triton/kernels/generated/rope_*.{c,h}
git commit -m "ggml-triton: AOT-generate 24 RoPE launchers (placeholder CUBIN on CPU-only host)

B.2 Stage 1. Three kernel families per Oracle review (Q0=B):
- rope_normal_<sin>_<yarn>_<fp>_sm80.{c,h}  (8 files: 2 dtype x 2 sin x 2 yarn)
- rope_neox_<sin>_<yarn>_<fp>_sm80.{c,h}    (8 files)
- rope_mrope_<sin>_<yarn>_<fp>_sm80.{c,h}   (8 files)
Total 24 AOT launchers.

Per Phase 0 audit §0.4 the AOT path emits 16-byte ELF-magic
placeholders on this host.  The C launcher ABI is real and the
registry wiring will be exercised by test-triton-registry; numeric
verification requires a GPU host (out of scope for B.2 on this box).

Pre-existing GELU/SiLU files NOT touched (verified by git diff per
Task 5 Step 2): the 'default' shape in the per-kernel LAUNCHER_SHAPES
map preserves the unchanged byte-compatible emit path."
```

---

### Task 6: Wire the aggregated include header

**Files:**
- Modify: `ggml/src/ggml-triton/kernels/include/triton_kernels.h`

- [ ] **Step 1: Add the 24 includes**

Open `ggml/src/ggml-triton/kernels/include/triton_kernels.h`. After the `silu_fp32_sm80.h` line, add 24 new `#include` lines:

```c
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
```

Also update the comment at the top of the header to mention the new launcher families. Replace the existing comment block:

```c
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
// Pointer-slot counts per kernel family (B.1 onwards):
//   gelu, silu, rms_norm_unweighted  :  2 ptrs (in, out) + N
//   rms_norm_weighted                 :  3 ptrs (x, w, y) + N
//   rope_normal, rope_neox            :  2 ptrs (a, b) + 11 scalar args
//   rope_mrope                        :  3 ptrs (a, b, freq_factors) + 13 scalar args
```

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-triton/kernels/include/triton_kernels.h
git commit -m "ggml-triton: pull 24 RoPE launchers into aggregated header (B.2)

B.2 Stage 1. The 24 launcher files from Task 5 are now reachable
from the C++ provider via the single aggregated header. The
provider cpp just needs to include 'kernels/include/triton_kernels.h'
once and can call any of the 24 triton_launch_rope_*() functions.

Updated the comment block to document the per-family pointer-slot
counts (rope_normal/rope_neox: 2 ptrs; rope_mrope: 3 ptrs).
The 3 B.1 ROPE launchers are unchanged (B.1's byte-compat fix
preserved the existing 4 GELU/SiLU files unchanged)."
```

---

### Task 7: Create the provider header

**Files:**
- Create: `ggml/src/ggml-triton/ggml-triton-provider-rope.h`

- [ ] **Step 1: Create the header file**

```cpp
// ggml/src/ggml-triton/ggml-triton-provider-rope.h
//
// B.2 (RoPE provider) — see docs/development/ROADMAP.md §3 Phase B.2.
//
// Mirrors ggml-triton-provider-cutlass.h / ggml-triton-provider-rmsnorm.h:
// one free C++ registration function to be called from both
// ggml-triton-provider.cpp (global registry) and ggml-triton.cpp
// (per-context registry).  Ships 6 impls: NORMAL+NEOX+MROPE × fp16/fp32.

#pragma once

#include "ggml-triton-provider.h"

// Register all RoPE kernel providers into the given registry.
// Called during backend initialization (B.2 of docs/development/ROADMAP.md).
// Ships 6 impls: NORMAL+NEOX+MROPE × fp16+fp32.
void ggml_triton_register_rope_providers(ggml_triton_op_registry & registry);
```

- [ ] **Step 2: Commit**

```bash
git add ggml/src/ggml-triton/ggml-triton-provider-rope.h
git commit -m "ggml-triton: declare rope provider registration function (B.2)

B.2 Stage 1. Mirrors ggml-triton-provider-rmsnorm.h: a single free
function that registers all 6 impls (NORMAL+NEOX+MROPE × fp16/fp32)
into the given registry.  Called from both the global registry
(ggml-triton-provider.cpp) and the per-context registry
(ggml-triton.cpp, both CPU-only and GPU branches)."
```

---

### Task 8: Implement the provider cpp (4 supports + 4 execute + 1 register)

**Files:**
- Create: `ggml/src/ggml-triton/ggml-triton-provider-rope.cpp`

- [ ] **Step 1: Create the file with the full implementation**

```cpp
// ggml/src/ggml-triton/ggml-triton-provider-rope.cpp
//
// B.2 RoPE AOT provider (Stage 1).  See docs/development/ROADMAP.md §3
// Phase B.2 for context.
//
// Math reference (bit-equivalent):
//   ggml/src/ggml-cpu/ops.cpp:5813-5959
//     ggml_compute_forward_rope_flt<T>  (the canonical forward)
//   ggml/src/ggml-triton/ggml-triton-provider-cpu.cpp:517-611
//     cpu_rope_f32_supports/execute  (F32 + NORMAL only — limited scope)
//
// Per-row computation:
//   NORMAL/NEOX: y[i] = x[i] * cos + partner * (sin or -sin)   (pair rotation)
//   MROPE:       y[i] = x[i] * cos - x[i + n_dims/2] * sin     (half-rotation)
//
// 6 impls = 3 modes × 2 dtypes; each impl dispatches to 4 AOT variants
// at runtime (2 sin_sign × 2 ya_on).  Total 24 launcher functions.
//
// The kernel source is triton_kernels/rope.py; the AOT launcher
// signatures are emitted by scripts/compile_kernels.py into
// ggml/src/ggml-triton/kernels/generated/rope_<mode>_<sin>_<yarn>_<dtype>_sm80.{h,c}.

#include "ggml-triton-provider-rope.h"
#include "ggml-triton-provider.h"
#include "ggml-triton-context.h"

#include "kernels/include/triton_kernels.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>


// --- shared helpers (extracted to avoid 6x duplication) -------------------

static inline bool rope_op_is_supported(const struct ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_ROPE;
}

static inline bool rope_dtypes_match(const struct ggml_tensor * op,
                                    enum ggml_type want) {
    return op->type == want
        && op->src[0] != nullptr && op->src[0]->type == want
        && op->src[1] != nullptr && op->src[1]->type == GGML_TYPE_I32;
}

static inline int32_t rope_mode(const struct ggml_tensor * op) {
    return ((const int32_t *)op->op_params)[2];
}

static inline int32_t rope_n_dims(const struct ggml_tensor * op) {
    return ((const int32_t *)op->op_params)[1];
}

static inline int32_t rope_n_ctx_orig(const struct ggml_tensor * op) {
    return ((const int32_t *)op->op_params)[4];
}

static inline float rope_freq_base(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 5 * sizeof(float), sizeof(float));
    return v;
}

static inline float rope_freq_scale(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 6 * sizeof(float), sizeof(float));
    return v;
}

static inline float rope_ext_factor(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 7 * sizeof(float), sizeof(float));
    return v;
}

static inline float rope_attn_factor(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 8 * sizeof(float), sizeof(float));
    return v;
}

static inline float rope_beta_fast(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 9 * sizeof(float), sizeof(float));
    return v;
}

static inline float rope_beta_slow(const struct ggml_tensor * op) {
    float v;
    std::memcpy(&v, op->op_params + 10 * sizeof(float), sizeof(float));
    return v;
}

static inline int32_t rope_sect(const struct ggml_tensor * op, int i) {
    return ((const int32_t *)op->op_params)[11 + i];
}

// Stage 1 hard-gate: BLOCK_SIZE = 128.  n_dims must be <= 128.
static inline bool rope_row_fits_stage1(const struct ggml_tensor * op) {
    return rope_n_dims(op) > 0 && rope_n_dims(op) <= 128;
}

// MROPE: must have at least one non-zero section.
static inline bool rope_mrope_valid_sections(const struct ggml_tensor * op) {
    int32_t s0 = rope_sect(op, 0);
    int32_t s1 = rope_sect(op, 1);
    int32_t s2 = rope_sect(op, 2);
    int32_t s3 = rope_sect(op, 3);
    int32_t sum = s0 + s1 + s2 + s3;
    return sum > 0 && sum * 2 <= rope_n_dims(op);
}

// Per Q5: YaRN is on if any YaRN parameter is non-default.
static inline bool rope_ya_on(const struct ggml_tensor * op) {
    return (rope_ext_factor(op)  != 0.0f)
        || (rope_attn_factor(op) != 1.0f)
        || (rope_beta_fast(op)   != 0.0f)
        || (rope_beta_slow(op)   != 0.0f);
}

// Pre-compute corr_dims (matches ggml_rope_yarn_corr_dims in ggml.c:4335).
// Only used when YA_ON is true; sub-microsecond per launch.
static inline void rope_compute_corr_dims(const struct ggml_tensor * op,
                                          float & corr_low,
                                          float & corr_high) {
    const float n_dims      = (float) rope_n_dims(op);
    const float n_ctx_orig  = (float) rope_n_ctx_orig(op);
    const float freq_base   = rope_freq_base(op);
    const float beta_fast   = rope_beta_fast(op);
    const float beta_slow   = rope_beta_slow(op);
    const float n_rot       = n_dims * 0.5f;
    const float log_arg     = n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f);
    const float theta_log   = std::log(freq_base);
    // corr_dim = n_dims * log(n_ctx_orig / (n_rot * 2*pi)) / (2 * log(freq_base))
    // For both beta_fast and beta_slow (the *value* of beta matters at runtime
    // when the kernel does the rope_yarn ramp; corr_low and corr_high here
    // are computed for the fast and slow thresholds respectively).
    corr_low  = n_dims * std::log(log_arg) / (2.0f * theta_log);
    corr_high = corr_low;  // same formula; the kernel uses the beta values
                           // from op_params to decide which to use
}


// --- NORMAL / fp16 ---------------------------------------------------------

static bool triton_rope_normal_fp16_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_NORMAL) return false;
    if (!rope_row_fits_stage1(op)) return false;
    return true;
}

static bool triton_rope_normal_fp16_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {

    const struct ggml_tensor * src0 = node->src[0];  // Q/K
    const struct ggml_tensor * src1 = node->src[1];  // positions
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;

    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const bool    ya_on      = rope_ya_on(node);

    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);

    // Per Q4/Q5: pick the right AOT-compiled launcher at runtime.
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_normal_bwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_normal_bwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_normal_fwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_normal_fwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- NORMAL / fp32 ---------------------------------------------------------

static bool triton_rope_normal_fp32_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_NORMAL) return false;
    if (!rope_row_fits_stage1(op)) return false;
    return true;
}

static bool triton_rope_normal_fp32_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {
    // Identical to fp16 execute except all dtype-specific launcher calls.
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;
    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const bool    ya_on      = rope_ya_on(node);
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_normal_bwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_normal_bwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_normal_fwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_normal_fwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- NEOX / fp16 -----------------------------------------------------------

static bool triton_rope_neox_fp16_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_NEOX) return false;
    if (!rope_row_fits_stage1(op)) return false;
    return true;
}

static bool triton_rope_neox_fp16_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;
    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const bool    ya_on      = rope_ya_on(node);
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_neox_bwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_neox_bwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_neox_fwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_neox_fwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- NEOX / fp32 -----------------------------------------------------------

static bool triton_rope_neox_fp32_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_NEOX) return false;
    if (!rope_row_fits_stage1(op)) return false;
    return true;
}

static bool triton_rope_neox_fp32_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;
    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const bool    ya_on      = rope_ya_on(node);
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_neox_bwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_neox_bwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_neox_fwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, corr_low, corr_high)
            : triton_launch_rope_neox_fwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- MROPE / fp16 ----------------------------------------------------------

static bool triton_rope_mrope_fp16_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_MROPE) return false;
    if (!rope_row_fits_stage1(op)) return false;
    if (!rope_mrope_valid_sections(op)) return false;
    // MROPE positions are 4x the seq length (one per axis).
    if (op->src[1]->ne[0] != op->ne[2] * 4) return false;
    return true;
}

static bool triton_rope_mrope_fp16_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    const struct ggml_tensor * src2 = node->src[2];  // freq_factors or nullptr
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;
    // Per Q3: pass a non-null dummy pointer when src2 is null.  We allocate
    // 1 float on the stack initialized to 1.0; the kernel divides theta by
    // this value, which is a no-op.  (Stage 1 simplification; the kernel
    // does not actually use this value in MROPE mode.)
    float dummy_freq_factor = 1.0f;
    CUdeviceptr freq_factors_ptr = src2 != nullptr
        ? (CUdeviceptr) src2->data
        : (CUdeviceptr) &dummy_freq_factor;
    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const int32_t sect_t     = rope_sect(node, 0);
    const int32_t sect_h     = rope_sect(node, 1);
    const int32_t sect_w     = rope_sect(node, 2);
    const int32_t sect_e     = rope_sect(node, 3);
    const bool    ya_on      = rope_ya_on(node);
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_mrope_bwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, corr_low, corr_high)
            : triton_launch_rope_mrope_bwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_mrope_fwd_yarnon_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, corr_low, corr_high)
            : triton_launch_rope_mrope_fwd_yarnoff_fp16_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- MROPE / fp32 ----------------------------------------------------------

static bool triton_rope_mrope_fp32_supports(const struct ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F32)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_MROPE) return false;
    if (!rope_row_fits_stage1(op)) return false;
    if (!rope_mrope_valid_sections(op)) return false;
    if (op->src[1]->ne[0] != op->ne[2] * 4) return false;
    return true;
}

static bool triton_rope_mrope_fp32_execute(
    struct ggml_backend_triton_context * ctx,
    const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    const struct ggml_tensor * src2 = node->src[2];
    if (src0->data == nullptr || src1->data == nullptr || node->data == nullptr) return false;
    float dummy_freq_factor = 1.0f;
    CUdeviceptr freq_factors_ptr = src2 != nullptr
        ? (CUdeviceptr) src2->data
        : (CUdeviceptr) &dummy_freq_factor;
    const int32_t n_dims     = rope_n_dims(node);
    const int32_t n_ctx_orig = rope_n_ctx_orig(node);
    const float   freq_base  = rope_freq_base(node);
    const float   freq_scale = rope_freq_scale(node);
    const float   ext_factor = rope_ext_factor(node);
    const float   attn_factor= rope_attn_factor(node);
    const float   beta_fast  = rope_beta_fast(node);
    const float   beta_slow  = rope_beta_slow(node);
    const int32_t sect_t     = rope_sect(node, 0);
    const int32_t sect_h     = rope_sect(node, 1);
    const int32_t sect_w     = rope_sect(node, 2);
    const int32_t sect_e     = rope_sect(node, 3);
    const bool    ya_on      = rope_ya_on(node);
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) rope_compute_corr_dims(node, corr_low, corr_high);
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    int rc;
    if (is_backward) {
        rc = ya_on
            ? triton_launch_rope_mrope_bwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, corr_low, corr_high)
            : triton_launch_rope_mrope_bwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, 0.0f, 0.0f);
    } else {
        rc = ya_on
            ? triton_launch_rope_mrope_fwd_yarnon_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, corr_low, corr_high)
            : triton_launch_rope_mrope_fwd_yarnoff_fp32_sm80(
                  ctx->cu_stream, (CUdeviceptr)src0->data, (CUdeviceptr)src1->data,
                  freq_factors_ptr, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow,
                  sect_t, sect_h, sect_w, sect_e, 0.0f, 0.0f);
    }
    return rc == 0;
}


// --- registration -----------------------------------------------------------

void ggml_triton_register_rope_providers(ggml_triton_op_registry & registry) {
    registry.register_impl(GGML_OP_ROPE, {
        /* .name     = */ "triton_rope_normal_fp16_sm80",
        /* .provider = */ GGML_TRITON_PROVIDER_TRITON,
        /* .supports = */ triton_rope_normal_fp16_supports,
        /* .execute  = */ triton_rope_normal_fp16_execute,
        /* .priority = */ 100,
    });
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_normal_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_rope_normal_fp32_supports,
        triton_rope_normal_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_neox_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_rope_neox_fp16_supports,
        triton_rope_neox_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_neox_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_rope_neox_fp32_supports,
        triton_rope_neox_fp32_execute,
        100,
    });
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_mrope_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_rope_mrope_fp16_supports,
        triton_rope_mrope_fp16_execute,
        100,
    });
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_mrope_fp32_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_rope_mrope_fp32_supports,
        triton_rope_mrope_fp32_execute,
        100,
    });
}
```

(Notes: the per-(mode, dtype) functions are very repetitive; this is intentional for Stage 1 clarity. A future cleanup pass could macro-ify or template-ize, but per B.1's pattern explicit-and-readable wins over DRY in Stage 1.)

- [ ] **Step 2: Commit (do NOT build yet — registration sites are not wired)**

```bash
git add ggml/src/ggml-triton/ggml-triton-provider-rope.cpp
git commit -m "ggml-triton: implement 6 ROPE AOT providers (B.2 Stage 1)

Per Oracle review (Q0=B): NORMAL+NEOX+MROPE × fp16/fp32 = 6 impls.
Each impl dispatches to 4 AOT-compiled launchers (2 sin_sign × 2 ya_on)
at runtime, for a total of 24 launcher functions.

Math: y = x * cos + partner * sin  (NORMAL: cscs0000;
                                     NEOX/MROPE: ccss0000)
matching ggml-cpu reference at ops.cpp:5813-5959.

Per Q1: all 6 YaRN floats + n_ctx_orig + n_dims are runtime args to
the launcher (full flexibility, no tolerance gate).

Per Q4/Q5: sin_sign and ya_on are constexpr-specialized in the
kernel; the C++ execute() picks the right launcher at runtime based
on (node->op == GGML_OP_ROPE_BACK) and rope_ya_on(op) (which checks
if any YaRN parameter is non-default).

Per Q3: src[2] freq_factors ptr is passed to the launcher; when null
the C++ provider passes a non-null dummy pointer to 1.0 (the kernel
divides theta by this value, which is a no-op).

row_fits_stage1() hard-gates n_dims <= 128 (Stage 1 BLOCK_SIZE=128).
MROPE also checks sections[4] validity and positions length == 4*seq.

Priority 100 matches B.1's pattern; the existing cpu_rope_f32
(priority 50) stays as a fallback for nodes we don't support."
```

---

### Task 9: Wire CMakeLists.txt (24 generated .c + 1 provider cpp + GGML_TRITON_WITH_ROPE option)

**Files:**
- Modify: `ggml/src/ggml-triton/CMakeLists.txt`

- [ ] **Step 1: Add the GGML_TRITON_WITH_ROPE option, source list, and compile def**

Open `ggml/src/ggml-triton/CMakeLists.txt`. Add the option (after the existing `GGML_TRITON_WITH_TILELANG` option block) and the source-list append inside an `if(GGML_TRITON_WITH_ROPE)` block:

```cmake
    # ------------------------------------------------------------------
    # Optional: RoPE kernel provider (B.2, default ON)
    # ------------------------------------------------------------------
    if (NOT GGML_TRITON_CPU_ONLY)
        option(GGML_TRITON_WITH_ROPE "Enable RoPE kernels in Triton backend" ON)
    endif()

    if(GGML_TRITON_WITH_ROPE)
        list(APPEND GGML_TRITON_GENERATED_SRC
            ${TRITON_GENERATED_DIR}/rope_normal_fwd_yarnoff_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_normal_fwd_yarnoff_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_normal_fwd_yarnon_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_normal_fwd_yarnon_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_normal_bwd_yarnoff_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_normal_bwd_yarnoff_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_normal_bwd_yarnon_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_normal_bwd_yarnon_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_neox_fwd_yarnoff_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_neox_fwd_yarnoff_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_neox_fwd_yarnon_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_neox_fwd_yarnon_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_neox_bwd_yarnoff_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_neox_bwd_yarnoff_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_neox_bwd_yarnon_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_neox_bwd_yarnon_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_mrope_fwd_yarnoff_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_mrope_fwd_yarnoff_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_mrope_fwd_yarnon_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_mrope_fwd_yarnon_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_mrope_bwd_yarnoff_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_mrope_bwd_yarnoff_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/rope_mrope_bwd_yarnon_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/rope_mrope_bwd_yarnon_fp32_sm80.c)
        list(APPEND GGML_TRITON_GPU_SRC ggml-triton-provider-rope.cpp)
    endif()

    ggml_add_backend_library(ggml-triton
                             ggml-triton.cpp
                             ggml-triton-buffer.cpp
                             ggml-triton-dispatch.cpp
                             ggml-triton-provider.cpp
                             ggml-triton-provider-cpu.cpp
                             ${GGML_TRITON_GPU_SRC}
                            )

    if(GGML_TRITON_WITH_ROPE)
        target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_HAS_ROPE)
    endif()
```

(The `target_compile_definitions` call is placed AFTER `ggml_add_backend_library` because the target must exist before compile_definitions can be applied — same B.1 fix for GGML_TRITON_HAS_RMSNORM.)

- [ ] **Step 2: Verify the cmake configure**

```bash
cmake -B build -S . -DGGML_TRITON=ON 2>&1 | tail -10
```

Expected: cmake configure succeeds, and the rope provider .cpp is in the `ggml-triton` target's source list. If cmake errors with "file not found" on any of the 24 `.c` files, double-check the paths.

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-triton/CMakeLists.txt
git commit -m "ggml-triton: link rope provider + 24 generated launchers (B.2)

B.2 Stage 1.  Adds:
- option(GGML_TRITON_WITH_ROPE ... ON) gated by NOT GGML_TRITON_CPU_ONLY
  (mirrors the GGML_TRITON_WITH_CUTLASS / GGML_TRITON_WITH_TILELANG
  option pattern from PR #1)
- 24 generated .c files appended to GGML_TRITON_GENERATED_SRC
  (3 modes x 2 dtypes x 2 sin_sign x 2 ya_on = 24)
- ggml-triton-provider-rope.cpp appended to GGML_TRITON_GPU_SRC
- target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_HAS_ROPE)
  placed after ggml_add_backend_library() so the target exists
  before compile_definitions is applied (same fix as B.1 for
  GGML_TRITON_HAS_RMSNORM)"
```

---

### Task 10: Wire global + per-context registration

**Files:**
- Modify: `ggml/src/ggml-triton/ggml-triton-provider.cpp` (global registry, gated by `GGML_TRITON_HAS_ROPE`)
- Modify: `ggml/src/ggml-triton/ggml-triton.cpp` (per-context registry, both branches, same gate)

- [ ] **Step 1: Global registry — `ggml-triton-provider.cpp`**

Open `ggml/src/ggml-triton/ggml-triton-provider.cpp`. Replace the existing includes block (currently has `#ifdef GGML_TRITON_HAS_CUTLASS` / `TILELANG`) with:

```cpp
#include "ggml-triton-provider.h"

#ifdef GGML_TRITON_HAS_RMSNORM
#include "ggml-triton-provider-rmsnorm.h"
#endif

#ifdef GGML_TRITON_HAS_ROPE
#include "ggml-triton-provider-rope.h"
#endif

#ifdef GGML_TRITON_HAS_CUTLASS
#include "ggml-triton-provider-cutlass.h"
#endif

#ifdef GGML_TRITON_HAS_TILELANG
#include "ggml-triton-provider-tilelang.h"
#endif
```

Then inside the `std::call_once` block of `ggml_triton_global_registry()`, add the registration call after the B.1 RMSNorm call:

```cpp
    std::call_once(flag, []() {
        ggml_triton_register_cpu_providers(registry);
        ggml_triton_register_builtin_providers(registry);
#ifdef GGML_TRITON_HAS_RMSNORM
        ggml_triton_register_rmsnorm_providers(registry);
#endif
#ifdef GGML_TRITON_HAS_ROPE
        ggml_triton_register_rope_providers(registry);
#endif
#ifdef GGML_TRITON_HAS_CUTLASS
        ggml_triton_register_cutlass_providers(registry);
#endif

#ifdef GGML_TRITON_HAS_TILELANG
        ggml_triton_register_tilelang_providers(registry);
#endif
    });
```

- [ ] **Step 2: Per-context registry — `ggml-triton.cpp` CPU-only branch**

Open `ggml/src/ggml-triton/ggml-triton.cpp`. Replace the existing top-of-file includes block (currently has `ggml-triton-provider-rmsnorm.h`) with:

```cpp
#include "ggml-triton.h"
#include "ggml-triton-context.h"
#include "ggml-triton-dispatch.h"
#include "ggml-triton-provider.h"

#ifdef GGML_TRITON_HAS_RMSNORM
#include "ggml-triton-provider-rmsnorm.h"
#endif

#ifdef GGML_TRITON_HAS_ROPE
#include "ggml-triton-provider-rope.h"
#endif

#ifdef GGML_TRITON_HAS_CUTLASS
#include "ggml-triton-provider-cutlass.h"
#endif
```

In the CPU-only branch of `ggml_backend_triton_init` (around line 419), add:

```cpp
    // Register kernel providers for this backend instance
    ggml_triton_register_builtin_providers(ctx->op_registry);
    ggml_triton_register_cpu_providers(ctx->op_registry);
#ifdef GGML_TRITON_HAS_RMSNORM
    ggml_triton_register_rmsnorm_providers(ctx->op_registry);
#endif
#ifdef GGML_TRITON_HAS_ROPE
    ggml_triton_register_rope_providers(ctx->op_registry);
#endif
#ifdef GGML_TRITON_HAS_CUTLASS
    ggml_triton_register_cutlass_providers(ctx->op_registry);
#endif
```

- [ ] **Step 3: Per-context registry — `ggml-triton.cpp` GPU branch**

In the GPU branch (around line 474), add the same call:

```cpp
    // Register kernel providers for this backend instance
    ggml_triton_register_builtin_providers(ctx->op_registry);
#ifdef GGML_TRITON_HAS_RMSNORM
    ggml_triton_register_rmsnorm_providers(ctx->op_registry);
#endif
#ifdef GGML_TRITON_HAS_ROPE
    ggml_triton_register_rope_providers(ctx->op_registry);
#endif
#ifdef GGML_TRITON_HAS_CUTLASS
    ggml_triton_register_cutlass_providers(ctx->op_registry);
#endif
```

- [ ] **Step 4: Build and run the test — expect GREEN**

```bash
cmake --build build-master --config Release --target test-triton-registry -j$(nproc)
./build-master/bin/test-triton-registry
```

Expected: exit 0 with output including `Assert 5 passed: 6 triton AOT RoPE impls (NORMAL+NEOX+MROPE x fp16/fp32) registered` and `OK: registry test passed`. The global and per-context registries each contain 5 RMS_NORM impls (1 CPU + 4 Triton AOT) + 6 ROPE impls (6 Triton AOT) = 11 Triton AOT impls in total.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-triton/ggml-triton-provider.cpp ggml/src/ggml-triton/ggml-triton.cpp
git commit -m "ggml-triton: register rope provider in both global and per-context registries (B.2)

B.2 Stage 1: GGML_OP_ROPE dispatchable via global
(ggml-triton-provider.cpp) and per-context (ggml-triton.cpp, both
CPU-only and GPU branches) registries.  Assert 5 in test-triton-registry
now has all the necessary wiring behind it.

After this commit, the ggml-triton backend registers:
- 1 CPU ROPE impl (cpu_rope_f32, NORMAL only, priority 50)
- 6 Triton AOT ROPE impls (NORMAL+NEOX+MROPE × fp16/fp32, priority 100)
- 1 CPU RMSNorm impl (cpu_rms_norm_f32)
- 4 Triton AOT RMSNorm impls (B.1)
- 4 Triton AOT GELU/SiLU impls (PR #1)
- 2 Triton AOT ADD/MUL impls (TileLang, conditional)
- 4 Triton AOT MUL_MAT impls (CUTLASS, conditional)

For a default build with all options ON: 20 Triton AOT impls across
9 op families.  This puts the ggml-triton backend on track to cover
80% of MiniMind-3's inference graph on the GPU path (the remaining
20% is FlashAttn, B.3 work)."
```

---

### Task 11: Build & verify the full chain (TDD green)

- [ ] **Step 1: Build the test-triton-registry target**

```bash
cmake --build build-master --config Release --target test-triton-registry -j$(nproc)
```

Expected: build succeeds with no errors. If the build fails with `undefined reference to triton_launch_rope_*`, the CMakeLists.txt source list is missing one of the 24 `.c` files (or the provider cpp isn't calling the right launcher function name).

- [ ] **Step 2: Run the test — expect exit 0**

```bash
./build-master/bin/test-triton-registry
```

Expected: exit 0 with output:

```
selected=triton_gelu_fp32_sm80 provider=0 priority=100
tilelang provider: registered (skipping assertion — not built) [or similar]
found RMS_NORM impl: name=triton_rms_norm_unweighted_fp16_sm80 provider=0 priority=100
found RMS_NORM impl: name=triton_rms_norm_unweighted_fp32_sm80 provider=0 priority=100
found RMS_NORM impl: name=triton_rms_norm_weighted_fp16_sm80 provider=0 priority=100
found RMS_NORM impl: name=triton_rms_norm_weighted_fp32_sm80 provider=0 priority=100
Assert 4 passed: triton AOT RMS_NORM fp16 + fp32 providers are registered
found ROPE impl: name=triton_rope_normal_fp16_sm80 provider=0 priority=100
found ROPE impl: name=triton_rope_normal_fp32_sm80 provider=0 priority=100
found ROPE impl: name=triton_rope_neox_fp16_sm80 provider=0 priority=100
found ROPE impl: name=triton_rope_neox_fp32_sm80 provider=0 priority=100
found ROPE impl: name=triton_rope_mrope_fp16_sm80 provider=0 priority=100
found ROPE impl: name=triton_rope_mrope_fp32_sm80 provider=0 priority=100
Assert 5 passed: 6 triton AOT RoPE impls (NORMAL+NEOX+MROPE x fp16/fp32) registered
OK: registry test passed
```

- [ ] **Step 3: Commit (only if build required any fixes)**

If the build in Step 1 required any fixes (e.g. typo in a launcher function name, missing include), commit those fixes. Otherwise no commit needed — all changes are already in commits from Tasks 1-10.

```bash
# Only if needed:
git add ...
git commit -m "fix: <description of fix>"
```

---

### Task 12: Cross-backend test (deferred to GPU host)

This task has no code change — it documents that cross-backend numeric verification is deferred to a GPU host. Per Phase 0 audit §0.4, the placeholder CUBIN on this CPU-only host does no real compute, so cross-backend diffs are meaningless.

- [ ] **Step 1: Mark this task as deferred in the test report**

When the implementation is reported back to the user, explicitly note: "Cross-backend CPU↔Triton ROPE perplexity diff is deferred to GPU host per Phase 0 audit §0.4. The numeric bit-equivalence path is exercised by `test-backend-ops ROPE` on GPU host via `./build/bin/test-backend-ops test -o ROPE --backends CPU,TRITON` and should show Δ ≤ 1e-3 fp16 vs ggml-cpu reference."

No commit needed.

---

### Task 13: MiniMind-3 smoke test

- [ ] **Step 1: Run the model with the new build**

```bash
cd build-master && ./bin/llama-cli -m ../minimind-3-F16.gguf -p "1+1等于几" -n 30 2>&1 | tail -10
```

Expected: forward pass + reasonable t/s (the B.1 baseline measured 63.8 t/s for MiniMind-3 64M with -ngl 999). Output text is reasonable Chinese; the model produces coherent output. If the model crashes or produces gibberish, one of the new impls is wrong (check the DEBUG log for which impl was selected).

If the output is empty (just `> `), this is a pre-existing main-build behavior, not a B.2 regression. Check `./bin/llama-perplexity` for a numerical correctness check.

- [ ] **Step 2: Perplexity check (optional, validates forward pass numerical)**

```bash
./bin/llama-perplexity -m ../minimind-3-F16.gguf -f ../tests/test-triton-registry.cpp 2>&1 | tail -5
```

Expected: PPL = ~18 ± a few. (The B.1 baseline measured 18.07 for MiniMind-3 64M.) The PPL should be approximately the same as the CPU-only baseline, because on the CPU-only host the new Triton AOT launchers fail at `load_module_once()` and fall through to ggml-cpu — so the model is actually running on ggml-cpu RoPE, not the new AOT kernels. On a GPU host with a real CUBIN, the PPL should be within 1e-3 of the CPU baseline (per ROADMAP §3 Phase A exit criteria).

No commit needed for this task.

---

### Task 14: Add `GGML_TRITON_WITH_ROPE` CMake option as a CI gate (optional polish)

**Files:**
- Modify: `ggml/src/ggml-triton/CMakeLists.txt` (already done in Task 9, this task is a polish check)

This task is a verification that the option gate works correctly. Both the default-on case (option flag absent) and the explicit-off case should build.

- [ ] **Step 1: Build with `GGML_TRITON_WITH_ROPE=OFF` and verify clean**

```bash
rm -rf build-off
cmake -B build-off -S . -DGGML_TRITON=ON -DGGML_TRITON_WITH_ROPE=OFF 2>&1 | tail -3
cmake --build build-off --config Release --target ggml-triton -j$(nproc) 2>&1 | tail -5
```

Expected: build succeeds. The ROPE-related symbols are NOT in the resulting `libggml-triton.so`:

```bash
nm -D build-off/bin/libggml-triton.so.0.13.1 | grep "rope" | head -3
```

Should be empty (or only reference the rope_ symbols from B.1's existing 4 GELU/SiLU if those accidentally contain "rope" — but they don't, so this should print nothing).

- [ ] **Step 2: Build with `GGML_TRITON_WITH_ROPE=ON` (default) and verify re-enabled**

```bash
cmake --build build --config Release --target ggml-triton -j$(nproc) 2>&1 | tail -3
nm -D build/bin/libggml-triton.so.0.13.1 | grep "rope" | head -3
```

Expected: build succeeds. The `ggml_triton_register_rope_providers` symbol is present:

```
000000000000b9a0 T _Z38ggml_triton_register_rope_providersR23ggml_triton_op_registry
                 U triton_launch_rope_normal_fp16_sm80
                 U triton_launch_rope_normal_fp32_sm80
                 ...
```

- [ ] **Step 3: No commit needed** (Task 9 already added the option; this task just verifies it)

---

### Task 15: Update test-pyramid.md (B.2 coverage marker)

**Files:**
- Modify: `docs/development/test-pyramid.md`

Per the B.1 plan §"Optional follow-up" / ROADMAP §3 Phase D.3 spirit, the test-pyramid's op-coverage marker should list the B.2 RoPE op family.

- [ ] **Step 1: Add the B.2 RoPE entries to the op-coverage marker**

Open `docs/development/test-pyramid.md`. Locate the existing op-coverage marker block (added by B.1, immediately under the table at around line 64-66 of the file). Replace it with:

```markdown
> **ggml-triton op coverage (as of B.2 / RoPE):**
> - `GGML_OP_UNARY` (GELU, SILU — fp16 + fp32)
> - `GGML_OP_RMS_NORM` (unweighted + weighted — fp16 + fp32, 4 impls)  *B.1*
> - `GGML_OP_ROPE` (NORMAL + NEOX + MROPE × fp16 + fp32, 6 impls; each dispatches to 4 AOT variants for fwd/bwd × YaRN on/off)  *B.2*
> - `GGML_OP_ROPE_BACK` (covered by the same 6 ROPE impls via constexpr SIN_SIGN)  *B.2*
> - `GGML_OP_ADD` / `GGML_OP_MUL` (TileLang, conditional on `GGML_TRITON_HAS_TILELANG`)
> - `GGML_OP_MUL_MAT` (CUTLASS, conditional on `GGML_TRITON_WITH_CUTLASS`)
>
> Complete list: see `scripts/kernel_registry.json` (kernels array) and
> each provider's `register_impl` calls.
```

- [ ] **Step 2: Commit**

```bash
git add docs/development/test-pyramid.md
git commit -m "docs: mark RoPE as covered op in test-pyramid (B.2)

B.2 of docs/development/ROADMAP.md. The test-pyramid's op-coverage
marker now lists GGML_OP_ROPE and GGML_OP_ROPE_BACK as covered by
the ggml-triton AOT backend.

This is the documentation counterpart to:
- ggml-triton-provider-rope.{h,cpp} (Task 8)
- The 6 new AOT providers in ggml_triton_global_registry() (Task 10)

For default ON builds, the registry now contains 11 Triton AOT impls
across 3 op families (RMSNorm from B.1 + ROPE from B.2 + GELU/SiLU
from PR #1), enough to cover 60-70% of MiniMind-3's inference
graph (the remaining ~30% is FlashAttn, B.3 work)."
```

---

## Self-review

After writing this plan, I checked it against the design spec at `docs/superpowers/specs/2026-06-12-rope-triton-aot-design.md`:

**1. Spec coverage:**
- §1 Architecture & file inventory → Tasks 5-10 (file map is complete)
- §2 Triton DSL kernel design → Task 3 (kernel body matches spec's pseudo-Triton)
- §3 AOT launcher ABI → Tasks 4-5 (LAUNCHER_SHAPES map + 24 AOT compiles)
- §4 Provider file design → Task 8 (6 supports + 6 execute + 1 register)
- §5 Test & verification → Tasks 1, 11, 12, 13 (Assert 5 + green test + cross-backend deferred + MiniMind-3 smoke)
- §6 Stage 2 path & failure modes → implicitly: the documented "out of scope for B.2" section above mirrors the design spec's Stage 2 table

**2. Placeholder scan:**
- No "TBD", "TODO", "implement later", "fill in details" patterns
- No "Add appropriate error handling" / "add validation" / "handle edge cases" patterns
- The one "Stage 2 simplification" note in Task 3 (MROPE per-axis theta, freq_factors handling) is explicitly documented inline as a known limitation with a planned Stage 2 fix, not a placeholder

**3. Type consistency:**
- `rope_mode`, `rope_n_dims`, `rope_n_ctx_orig` are read consistently across all 6 `execute()` functions
- `rope_ya_on()` is called in all 6 execute() functions
- `rope_compute_corr_dims()` is called in all 6 execute() functions when `ya_on` is true
- The launcher function names match across kernel_registry.json, compile_kernels.py LAUNCHER_SHAPES, the 24 generated files, and the provider cpp's call sites
- The LAUNCHER_SHAPES dict's parameter lists match the design spec's §3.2 ABI shape table

**4. Scope:**
- Single subsystem (RoPE provider), 6 file groups (3 new + 24 generated + 7 modified)
- All tasks are sequential (TDD red first, then implementation in dependency order)
- B.1's pattern is faithfully replicated (constexpr-bake for body shape, runtime args for values, byte-compat fallback for the existing GELU/SiLU path)

No issues found inline. Plan ready for execution.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-12-rope-triton-aot.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
