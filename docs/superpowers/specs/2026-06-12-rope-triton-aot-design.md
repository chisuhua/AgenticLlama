# B.2 RoPE Triton AOT Provider — Design

> **For agentic workers:** This is the design spec for the B.2 milestone of `docs/development/ROADMAP.md`. After approval, the next step is to invoke the **writing-plans** skill to produce the implementation plan. This spec is the input to that plan.
>
> **Status:** Awaiting user review (post-brainstorming, pre-plan).
> **Brainstormed:** 2026-06-12, via brainstorming skill, Oracle review (B recommendation over A/C).
> **B.1 reference implementation:** `docs/superpowers/plans/2026-06-10-rmsnorm-triton-aot.md` (commit `418f88b9f`).
> **Brainstorming decisions:**
> - Q0 (mode coverage) = **B** (NORMAL + NEOX + MROPE)
> - Q1 (op_params strategy) = **A** (full runtime args to launcher)
> - Q2 (n_dims + BLOCK_SIZE) = **single launcher + runtime n_dims + mask** (BLOCK_SIZE=128 constexpr)
> - Q3 (src[2] freq_factors) = **supported** (kernel takes a runtime ptr)
> - Q4 (forward/backward) = **both** (constexpr `SIN_SIGN` separates AOT variants)
> - Q5 (YaRN) = **full implementation** (constexpr `YA_ON` separates AOT variants)
> - AOT strategy = **A** (full constexpr; 24 AOT compiles = 3 modes × 2 dtypes × 2 sin_sign × 2 ya_on)

---

## 1. Architecture & file inventory

The B.2 implementation mirrors the B.1 4-step loop (kernel source → AOT launcher → provider → test) with the dimensions of variation: **3 modes × 2 dtypes × 2 sin_sign × 2 ya_on = 24 AOT compiles**.

### 1.1 New files (3)

| # | File | Purpose |
|---|---|---|
| 1 | `triton_kernels/rope.py` | Triton DSL source. One `@triton.jit` function whose body branches on `MODE`, `SIN_SIGN`, `YA_ON` constexprs. Covers NORMAL, NEOX, MROPE; forward and backward; YaRN off and on. |
| 2 | `ggml/src/ggml-triton/ggml-triton-provider-rope.h` | Provider header — declares `ggml_triton_register_rope_providers(ggml_triton_op_registry &)`. Mirrors B.1's `ggml-triton-provider-rmsnorm.h`. |
| 3 | `ggml/src/ggml-triton/ggml-triton-provider-rope.cpp` | Provider implementation — 6 `supports` + 6 `execute` functions (3 modes × 2 dtypes; sin_sign and ya_on are constexpr-specialized at AOT level, not runtime-dispatched in the provider), plus `ggml_triton_register_rope_providers()` registering all 6. Mirrors B.1's provider.cpp. |

### 1.2 Generated files (24 + 24 = 48)

`ggml/src/ggml-triton/kernels/generated/rope_<mode>_<sin>_<yarn>_<dtype>_sm80.{c,h}` (24 launcher pairs). Naming convention:

```
triton_launch_rope_<mode>_<sin>_<yarn>_<dtype>_sm80
                       ^     ^      ^
                       |     |      +-- yarnon or yarnoff
                       |     +-- fwd or bwd
                       +-- normal, neox, or mrope
```

24 AOT variants = 3 modes × 2 dtypes × 2 sin_sign × 2 ya_on.

### 1.3 Modified files (7)

| # | File | Change |
|---|---|---|
| 16 | `scripts/kernel_registry.json` | Add 3 entries: `rope_normal`, `rope_neox`, `rope_mrope`. Each has 8 variants (2 dtype × 2 sin_sign × 2 ya_on). |
| 17 | `scripts/compile_kernels.py` | Add 3 new entries to `LAUNCHER_SHAPES` map: `rope_normal` (2 ptrs), `rope_neox` (2 ptrs), `rope_mrope` (3 ptrs + 4 sect ints). The existing `default` shape stays unchanged. |
| 18 | `tests/test-triton-registry.cpp` | Add Assert 5: iterate `reg.get_impls(GGML_OP_ROPE)`, filter `provider == GGML_TRITON_PROVIDER_TRITON`, check 6 names (3 modes × 2 dtypes). rc=6/7/8 on miss. |
| 19 | `ggml/src/ggml-triton/CMakeLists.txt` | Add 24 generated `.c` files to `GGML_TRITON_GENERATED_SRC`; add `ggml-triton-provider-rope.cpp` to `GGML_TRITON_GPU_SRC`. New option `GGML_TRITON_WITH_ROPE` (default ON), gated by `GGML_TRITON_HAS_ROPE` macro. |
| 20 | `ggml/src/ggml-triton/ggml-triton-provider.cpp` | Inside `ggml_triton_global_registry()`'s `std::call_once`, add `ggml_triton_register_rope_providers(registry)` under `#ifdef GGML_TRITON_HAS_ROPE`. |
| 21 | `ggml/src/ggml-triton/ggml-triton.cpp` | Inside `ggml_backend_triton_init`, both CPU-only and GPU branches, add the same call under the same `#ifdef`. |
| 22 | `ggml/src/ggml-triton/kernels/include/triton_kernels.h` | Add 24 `#include` lines for the new launchers. Update the slot-count comment to mention rope launchers. |

### 1.4 Out of scope (per Q0–Q5 decisions)

- VISION and IMROPE modes (Q0 = B excludes them).
- `n_dims > 128` rows (current `BLOCK_SIZE=128` ceiling; for RoPE the ceiling is 128 since `BLOCK_SIZE` is 128 too).
- `inplace` RoPE (Qwen3 doesn't use it; defer to Stage 2 if needed).
- Backward gradient tests for autograd (`test-backend-ops grad` mode) — Stage 1 supports `ROPE_BACK` via constexpr `SIN_SIGN`, but `test_rope` already exercises both via the `for(fw:{true,false})` loop.
- Cross-backend CPU↔Triton perplexity diff — deferred to GPU host per Phase 0 audit §0.4.

---

## 2. Triton DSL kernel design

The kernel lives in `triton_kernels/rope.py`. 24 AOT variants = 3 modes × 2 sin_sign × 2 ya_on × 2 dtypes. Per the AOT strategy decision (A: full constexpr), each variant has a different **kernel body shape**, so the constexpr dimensions are real kernel-branches not just numeric specializations.

### 2.1 Constexpr list (kernel body shape)

| Constexpr | Values | Branch in kernel |
|---|---|---|
| `MODE: tl.constexpr` | `0`=NORMAL, `2`=NEOX, `8`=MROPE | Picks `n_offset` and `scale` at compute time (not AOT) — `n_offset = n_dims // 2` for NEOX/MROPE, `1` for NORMAL; `scale = 2` for non-NORMAL, `1` for NORMAL. MROPE also adds 4-axis theta fetch. |
| `SIN_SIGN: tl.constexpr` | `1.0` (fwd) or `-1.0` (bwd) | Multiplied into the `sin_theta` value during cache init. Per Q4, BWD is a separate AOT compile. |
| `YA_ON: tl.constexpr` | `0` (default rope_yarn) or `1` (full YaRN with `ext_factor`/`attn_factor`/`beta_fast`/`beta_slow`/`corr_dims`/`mscale`). Per Q5, full impl. |
| `BLOCK_SIZE: tl.constexpr` | `128` (single value per Q2). |

### 2.2 Kernel body shape (pseudo-Triton)

```python
@triton.jit
def rope_kernel(
    a_ptr, b_ptr, freq_factors_ptr,                # input tensors
    out_ptr,                                        # output (=a_ptr if inplace)
    n_dims,                                         # runtime, int32
    n_ctx_orig,                                     # runtime, int32
    freq_base, freq_scale,                          # runtime, fp32
    ext_factor, attn_factor,                        # runtime, fp32 (YaRN)
    beta_fast, beta_slow,                           # runtime, fp32 (YaRN)
    sect_t, sect_h, sect_w, sect_e,                 # runtime, int32 (MROPE only)
    BLOCK_SIZE: tl.constexpr,                       # 128
    MODE: tl.constexpr,                             # 0 / 2 / 8
    SIN_SIGN: tl.constexpr,                         # +1.0 / -1.0
    YA_ON: tl.constexpr,                            # 0 / 1
):
    # 1. One program per row. Row = one (head, seq_token).
    pid = tl.program_id(0)

    # 2. Load n_dims elements (masked) for this row's Q/K vector.
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_dims
    x = tl.load(a_ptr + pid * n_dims + offsets, mask=mask, other=0.0).to(tl.float32)

    # 3. Compute cos/sin cache. Two paths:
    if YA_ON:
        mscale = 1.0 + 0.1 * tl.log(1.0 / freq_scale)
        # corr_dims from beta_fast/beta_slow + n_dims + n_ctx_orig + freq_base
        # (computed in launcher emit, passed as 2 runtime floats — see Section 3)
    else:
        mscale = 1.0

    if MODE == 8:  # MROPE: 4-axis thetas
        # sector selection: section-of-2-pairs from sections[]
        ...
    else:  # NORMAL or NEOX: single-axis theta
        theta = pos * theta_scale  # theta_scale = freq_base^(-2/n_dims)

    if freq_factors_ptr != nullptr:
        ff = tl.load(freq_factors_ptr + offsets // 2, mask=mask, other=1.0)
        theta = theta / ff

    cos_theta = tl.cos(theta) * mscale
    sin_theta = tl.sin(theta) * mscale * SIN_SIGN

    # 4. Apply rotation (uniform across modes — matches ggml-cpu rotate_pairs):
    if MODE == 0:  # NORMAL: cscs0000 interleave
        n_offset = 1
        scale = 1
    else:           # NEOX, MROPE: ccss0000 half-rotation
        n_offset = n_dims // 2
        scale = 2
    # for i0 in [0, n) step 2:
    #   ic = i0 / scale
    #   x0 = x[ic + 0]; x1 = x[ic + n_offset]
    #   out[ic + 0]        = x0*cos - x1*sin
    #   out[ic + n_offset] = x0*sin + x1*cos
    ...
    y0 = x0 * cos_theta - x1 * sin_theta
    y1 = x0 * sin_theta + x1 * cos_theta
    tl.store(out_ptr + pid * n_dims + ic + 0,        y0.to(tl.float16 if dtype == fp16 else tl.float32), mask=mask)
    tl.store(out_ptr + pid * n_dims + ic + n_offset, y1.to(tl.float16 if dtype == fp16 else tl.float32), mask=mask)
```

### 2.3 Source/destination layout

- `a_ptr` is the Q (or K) tensor, shape `[n_dims, n_head, seq, batch]`, **contiguous** along the last dim (`nb0 = sizeof(T)`).
- Per-row stride: `n_dims` elements (one program processes one (head, seq, batch) triple).
- `b_ptr` is the position vector, `int32`, length `seq` (NORMAL/NEOX) or `seq*4` (MROPE).
- `out_ptr` is the same Q/K tensor (inplace) or a new tensor (out-of-place). Both have the same `[n_dims, n_head, seq, batch]` shape and `nb0 = sizeof(T)`.

### 2.4 Inline-with-host conversions

Per `ggml/src/ggml-cpu/ops.cpp:5805-5809`, F16 conversion is via `type_conversion_table<ggml_fp16_t>::to_f32 / from_f32` (round-to-nearest-even). Triton kernels must match this rounding — use `tl.cast(x, tl.float16, fp_downcast_rounding="rtne")` explicitly.

### 2.5 Edge cases

- **`n_dims == 0`**: same as `ggml-cpu` — no-op (the inner loop is empty, tail-copy skipped). The mask `offsets < n_dims` becomes all-false; writes are guarded.
- **`n_dims > BLOCK_SIZE`**: Stage 1's hard gate — `supports()` rejects and falls through to CPU. Multi-block variant deferred to Stage 2.
- **`freq_factors_ptr == nullptr`**: skip the freq_factors divide (the common case for Qwen3).
- **MROPE with default `sections[4] = {0, 0, 0, 0}`**: stage 1 treats this as "no MROPE in effect" and the supports() predicate asserts `sections_sum > 0` for MROPE entries.

---

## 3. AOT launcher ABI

The AOT launcher is the C function the C++ provider calls. 24 AOT variants share the same ABI shape (within a mode group) but with different constexpr specializations, so the launcher name encodes the variant.

### 3.1 Per-launcher C signature

```c
int triton_launch_rope_<mode>_<sin>_<yarn>_<dtype>_sm80(
    CUstream    stream,           // cuLaunchKernel argument
    CUdeviceptr a,                // input Q/K tensor (F16 or F32, n_dims×n_head×seq×batch)
    CUdeviceptr b,                // position vector (I32, length seq for normal/neox, seq*4 for mrope)
    CUdeviceptr freq_factors,     // optional I32, nullptr if no freq_factors (Q3)
    int32_t     n_dims,           // runtime row length, ≤ 128
    int32_t     n_ctx_orig,       // runtime, op_params[4]
    float       freq_base,        // runtime, op_params[5]
    float       freq_scale,       // runtime, op_params[6]
    float       ext_factor,       // runtime, op_params[7]
    float       attn_factor,      // runtime, op_params[8]
    float       beta_fast,        // runtime, op_params[9]
    float       beta_slow,        // runtime, op_params[10]
    int32_t     sect_t,           // MROPE only: runtime, op_params[11]
    int32_t     sect_h,           // MROPE only: runtime, op_params[12]
    int32_t     sect_w,           // MROPE only: runtime, op_params[13]
    int32_t     sect_e,           // MROPE only: runtime, op_params[14]
    float       corr_low,         // YaRN only: corr_dims[0] pre-computed by launcher
    float       corr_high);       // YaRN only: corr_dims[1] pre-computed by launcher
```

**Total args after stream**: **18** (3 ptrs + 1 n_dims + 1 n_ctx_orig + 6 YaRN floats + 4 MROPE section ints + 2 corr_dims).

### 3.2 Per-mode ABI shape (extends B.1's `LAUNCHER_SHAPES` pattern)

The ABI changes per `mode`:

| Mode | ptrs after stream | extras |
|---|---|---|
| `rope_normal` | `a, b` (2 ptrs) | no `freq_factors`, no `sections[]`, no `corr_dims[]` |
| `rope_neox` | `a, b` (2 ptrs) | no `freq_factors`, no `sections[]`, no `corr_dims[]` |
| `rope_mrope` | `a, b, freq_factors` (3 ptrs) | `sections[4]`, no `corr_dims[]` |

Three ABI shapes — matches the B.1 `LAUNCHER_SHAPES` dictionary pattern (3 entries: `default`, `rms_norm_unweighted`, `rms_norm_weighted`). `compile_kernels.py` will gain 3 new entries.

### 3.3 Launch sequence (what the C++ provider does)

```
ctx->cu_stream, d_a, d_b,                    // from node
freq_factors_ptr = d_freq_factors_or_NULL,   // Q3
n_dims, n_ctx_orig,                           // from op_params[1, 4]
freq_base..beta_slow,                         // op_params[5..10]
sect_t, sect_h, sect_w, sect_e,               // op_params[11..14] (MROPE only)
corr_low, corr_high =                          // YaRN only:
    compute_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow),
                                              // computed in the C++ provider BEFORE the launcher call
```

The `corr_dims[]` pre-computation is a small bit of C++ that mirrors `ggml_rope_yarn_corr_dims` (`ggml/src/ggml.c:4335`). It's a single closed-form per launch — sub-microsecond.

### 3.4 What gets serialized into the C launcher body

The C launcher body (one per AOT variant, e.g. `rope_neox_fwd_yarnoff_fp16.c`):

1. Load CUBIN (compile-time embedded)
2. `cuModuleLoadData` + `cuModuleGetFunction` (one-time cache)
3. Build `void* args[]` from the 18 launcher args
4. `cuLaunchKernel` with:
   - `grid = (n_rows,) = (n_head * seq * batch, 1, 1)` (one program per row)
   - `block = (BLOCK_SIZE, 1, 1)` = `(128, 1, 1)`

### 3.5 LAUNCHER_SHAPES extension in compile_kernels.py

Three new entries added to the B.1 `LAUNCHER_SHAPES` map:

```python
LAUNCHER_SHAPES = {
    "default": [...],                  # unchanged
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

### 3.6 What the launcher ABI is NOT

- **No `mode`, `n_offset`, `scale`, `sin_sign`, `ya_on` runtime args** — these are constexpr-specialized into the C function name.
- **No `BLOCK_SIZE` runtime arg** — constexpr-specialized.
- **No `out_ptr`** — out-of-place mode uses a different launcher (or the `inplace` path). Per Qwen3, RoPE is always out-of-place. Stage 1 supports out-of-place only.

---

## 4. Provider file design

**Files**:
- `ggml/src/ggml-triton/ggml-triton-provider-rope.h` (new)
- `ggml/src/ggml-triton/ggml-triton-provider-rope.cpp` (new)

### 4.1 Header

```cpp
#pragma once

#include "ggml-triton-provider.h"

// Register all RoPE kernel providers into the given registry.
// Called during backend initialization (B.2 of docs/development/ROADMAP.md).
// Ships 6 impls: NORMAL+NEOX+MROPE × fp16+fp32.
void ggml_triton_register_rope_providers(ggml_triton_op_registry & registry);
```

### 4.2 Provider cpp structure

The provider file has **6 `supports` functions + 6 `execute` functions + 1 registration function**, totaling 13 static functions. Plus shared inline helpers.

```
ggml-triton-provider-rope.cpp
├── shared helpers (3)
│   ├── rope_op_is_supported(op)        // op != null && op->op == GGML_OP_ROPE
│   ├── rope_dtypes_match(op, want)     // op->type == want && src[0/1] types match
│   ├── rope_mode(op)                   // op_params[2]
│   ├── rope_n_dims(op)                 // op_params[1]
│   ├── rope_n_ctx_orig(op)             // op_params[4]
│   ├── rope_freq_base(op)              // op_params[5] (float)
│   ├── rope_freq_scale(op)             // op_params[6]
│   ├── rope_ext_factor(op)             // op_params[7]
│   ├── rope_attn_factor(op)            // op_params[8]
│   ├── rope_beta_fast(op)              // op_params[9]
│   ├── rope_beta_slow(op)              // op_params[10]
│   ├── rope_sect(op, i)                // op_params[11+i]
│   ├── rope_row_fits_stage1(op)        // n_dims <= 128
│   ├── rope_is_mrope(op)               // mode == GGML_ROPE_TYPE_MROPE
│   ├── rope_ya_on(op)                  // ext_factor != 0 || attn_factor != 1 || beta != 0
│   └── rope_mrope_valid_sections(op)   // sections_sum > 0 && sections_sum * 2 <= n_dims
├── 6 supports() — one per (mode, dtype)
├── 6 execute() — one per (mode, dtype), dispatch to 4 launchers each
└── ggml_triton_register_rope_providers() — registers 6 impls at priority 100
```

### 4.3 Per-mode supports() predicates (6 functions)

Pattern (one for each `mode × dtype`):

```cpp
// E.g. triton_rope_normal_fp16_supports
static bool triton_rope_normal_fp16_supports(const ggml_tensor * op) {
    if (!rope_op_is_supported(op)) return false;
    if (!rope_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (rope_mode(op) != GGML_ROPE_TYPE_NORMAL) return false;
    if (rope_n_dims(op) <= 0) return false;
    if (!rope_row_fits_stage1(op)) return false;
    return true;
}
```

Differences across the 6:
- **dtype**: `GGML_TYPE_F16` or `GGML_TYPE_F32`
- **mode**: `GGML_ROPE_TYPE_NORMAL` (0), `GGML_ROPE_TYPE_NEOX` (2), or `GGML_ROPE_TYPE_MROPE` (8)
- For MROPE, add: `if (!rope_mrope_valid_sections(op)) return false;`
- For MROPE, add: `if (op->src[1] == nullptr || op->src[1]->ne[0] != op->ne[2] * 4) return false;` (MROPE positions are 4× seq length)

The 6 supports() functions do **not** dispatch on `sin_sign` (fwd/bwd) or `ya_on` (YaRN off/on) — those are constexpr-specialized in the kernel and selected at AOT-compile time. The provider's `execute()` function reads `ya_on` from the node at launch time and calls the matching launcher (yarnoff vs yarnon).

### 4.4 Per-mode execute() functions (6 functions)

```cpp
// E.g. triton_rope_normal_fp16_execute
static bool triton_rope_normal_fp16_execute(
    struct ggml_backend_triton_context * ctx,
    const ggml_tensor * node) {

    const ggml_tensor * src0 = node->src[0];  // Q or K tensor
    const ggml_tensor * src1 = node->src[1];  // position vector
    const ggml_tensor * src2 = node->src[2];  // freq_factors or nullptr
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

    // Pre-compute corr_dims if YaRN on. Mirrors ggml_rope_yarn_corr_dims (ggml.c:4335).
    float corr_low = 0.0f, corr_high = 0.0f;
    if (ya_on) {
        const float theta_base_log = std::log(freq_base);
        const float n_rot = n_dims / 2.0f;
        const float log_arg = n_ctx_orig / (n_rot * 2.0f * (float)M_PI);
        corr_low  = n_dims * std::log(log_arg) / (2.0f * theta_base_log);
        corr_high = corr_low;
    }

    // Branch on (sin_sign, ya_on) — there are 4 launchers per (mode, dtype) and we
    // pick the right one at runtime. Per Q4/Q5 the sin_sign and ya_on are
    // constexpr-specialized so there are 4 distinct AOT-compiled launchers.
    const bool is_backward = (node->op == GGML_OP_ROPE_BACK);
    const int rc =
        is_backward
            ? (ya_on
                ? triton_launch_rope_normal_bwd_yarnon_fp16_sm80(...)
                : triton_launch_rope_normal_bwd_yarnoff_fp16_sm80(...))
            : (ya_on
                ? triton_launch_rope_normal_fwd_yarnon_fp16_sm80(...)
                : triton_launch_rope_normal_fwd_yarnoff_fp16_sm80(...));
    return rc == 0;
}
```

MROPE variants add 4 `sect_*` args before `corr_low/corr_high` and the `freq_factors` ptr after `b`.

NEOX and NORMAL use the same launcher signature except for the `mode` constexpr (NORMAL=0, NEOX=2) baked into the launcher name.

### 4.5 Registration

```cpp
void ggml_triton_register_rope_providers(ggml_triton_op_registry & registry) {
    registry.register_impl(GGML_OP_ROPE, {
        "triton_rope_normal_fp16_sm80",
        GGML TRITON_PROVIDER TRITON,
        triton_rope_normal_fp16_supports,
        triton_rope_normal_fp16_execute,
        /* priority = */ 100,
    });
    // ... 5 more (normal_fp32, neox_fp16, neox_fp32, mrope_fp16, mrope_fp32)
}
```

Priority 100 matches B.1's pattern (Triton AOT providers).

### 4.6 What the provider does NOT do

- **No `inplace` dispatch.** Per Qwen3 model code, RoPE is always out-of-place. If `view_src == src[0]`, the provider falls through to CPU. (B.1 also did not dispatch inplace for RMSNorm; deferred to Stage 2 if needed.)
- **No fallback to `cpu_rope_*` provider.** The B.1 design registers at priority 100 and lets the dispatcher's "highest priority whose supports() returns true" select. NORMAL/NEOX/MROPE + F16/F32 will all hit Triton AOT; everything else falls through to ggml-cpu.

### 4.7 Dispatcher behavior

The dispatcher in `ggml/src/ggml-triton/ggml-triton-dispatch.cpp:40-67` already iterates `ctx->op_registry.select(node)`. With 6 new ROPE impls added, the per-node lookup is O(6) — negligible.

If both a ROPE and a ROPE_BACK node hit the same registry, the dispatcher's `select()` returns the highest-priority matches. Since B.2 registers only ROPE (not ROPE_BACK as a separate op), the dispatch for ROPE_BACK is automatic via the `node->op` check inside `execute()`: the supports() returns true for both ROPE and ROPE_BACK (per Q4), the execute() branches on `node->op == GGML_OP_ROPE_BACK` to pick fwd/bwd launcher.

This means **the dispatcher will route ROPE_BACK to the same impl** as ROPE. Per Q4, this is correct: B.2 supports both via the constexpr SIN_SIGN in the AOT variant.

---

## 5. Test & verification strategy

**Mirror of B.1 verification pattern**, with one new dimension (B.1 was 1 op family; B.2 is 3 mode families + 2 sin_sign + 2 ya_on = 24 AOT variants, all dispatchable from a single test).

### 5.1 Five-layer verification

| Layer | What it verifies | Mechanism | Stage 1 expected (CPU host) |
|---|---|---|---|
| **L1** | Provider registration in global registry | `tests/test-triton-registry.cpp` Assert 5 | 6 new ROPE impls found, exit 0 |
| **L2** | Provider registration in per-context registry | `tests/test-triton-registry.cpp` Assert 5 (after Step 10 wiring) | Same as L1 |
| **L3** | Op dispatch path end-to-end | `tests/test-backend-ops ROPE` on CPU baseline | 22 in-scope cases (3 modes × 2 dtypes, with YaRN subset) green; 29 out-of-scope cases (IMROPE=9, VISION=7, plus non-YaRN NORMAL/NEOX/MROPE in default form) green via ggml-cpu fallback |
| **L4** | End-to-end model inference doesn't crash | `llama-cli -m minimind-3-F16.gguf -p "1+1等于几" -n 50` | Forward pass + reasonable t/s; no crash; reasonable Chinese output |
| **L5** | Numerical correctness (CPU host: stub; GPU host: real) | `test_rope` cross-backend diff vs CPU reference | CPU host: `test-backend-ops --backends CPU,TRITON` defers to CPU; GPU host: ≤ 1e-3 fp16 (per Phase 0 audit §0.4) |

### 5.2 Assert 5 in test-triton-registry.cpp

Add a 5th assert block to `tests/test-triton-registry.cpp` mirroring B.1's Assert 4:

```cpp
// Assert 5 (B.2): the Triton AOT RoPE provider (3 modes × 2 dtypes = 6 impls)
// must be registered for GGML_OP_ROPE.
{
    constexpr const char * expected[] = {
        "triton_rope_normal_fp16_sm80",
        "triton_rope_normal_fp32_sm80",
        "triton_rope_neox_fp16_sm80",
        "triton_rope_neox_fp32_sm80",
        "triton_rope_mrope_fp16_sm80",
        "triton_rope_mrope_fp32_sm80",
    };
    bool found[6] = {false, false, false, false, false, false};
    if (auto * impls = reg.get_impls(GGML_OP_ROPE)) {
        for (const auto & impl : *impls) {
            if (impl.provider != GGML_TRITON_PROVIDER_TRITON) continue;
            for (int i = 0; i < 6; ++i) {
                if (std::string(impl.name).find(expected[i]) != std::string::npos) {
                    found[i] = true;
                }
            }
        }
    }
    for (int i = 0; i < 6; ++i) {
        if (!found[i]) {
            std::fprintf(stderr, "FAIL: triton AOT RoPE impl %s not registered\n", expected[i]);
            return 6;
        }
    }
    std::printf("Assert 5 passed: 6 triton AOT RoPE impls (NORMAL+NEOX+MROPE × fp16/fp32) registered\n");
}
```

Return code 6 distinguishes Assert 5 failure from B.1's 4/5. Pattern matches B.1's "filter by name + provider" without coupling to internal ordering.

### 5.3 Existing test coverage to reuse (zero new test cases)

`tests/test-backend-ops.cpp:4849` `test_rope` already has **51 instantiations** that exercise the B.2 path:

- **NORMAL (16 cases)**: shapes `{128,32,2,1}` n_dims=128, `{16,16,8192,1}` n_dims=16, `{64,5,4,3}` n_dims=10, etc. Modes NORMAL × types F32/F16.
- **NEOX (14 cases)**: `{64,1,2,1}` n_dims=64 (Falcon 7B), `{128,32,2,1}` n_dims=128. With forward=true and forward=false.
- **MROPE (13 cases)**: Qwen2-VL shape variants. With `sections[]` from `rope_sections[4]`.
- **IMROPE (9 cases)**: Qwen3-VL variants — **out of Stage 1 scope** (Q0 = B). Will fall through to CPU backend.
- **VISION (7 cases)**: Qwen2-VL ViT — **out of Stage 1 scope** (Q0 = B). Will fall through to CPU backend.
- **YaRN subset (5-10 cases)**: with `ef=0.7465, af=1.4245` (e.g. `tests/test-backend-ops.cpp:8773-8774`). Tests the `ya_on=1` path.

**The test harness iterates backends**: for each `test_rope` case, it tries ggml-cpu first, then ggml-triton (if registered). The ggml-triton `select()` will return the highest-priority impl whose `supports()` is true. For RoPE in scope, that's one of the 6 new Triton AOT impls. For out-of-scope modes (IMROPE/VISION), `supports()` returns false and the harness falls through to ggml-cpu.

### 5.4 Specific verification commands

| Test | Command | Expected on CPU host | Expected on GPU host |
|---|---|---|---|
| Registry unit | `cd build && ctest -R '^test-triton-registry$' --verbose` | exit 0, all 5 asserts pass | Same |
| Op dispatch (CPU baseline) | `ctest -R '^test-backend-ops$' --verbose` | exit 0, all 51 ROPE cases green | Same |
| MiniMind-3 smoke | `./build/bin/llama-cli -m minimind-3-F16.gguf -p "1+1等于几" -n 50` | Forward pass + reasonable t/s; no crash; reasonable Chinese output | Same |
| MiniMind-3 perplexity (optional) | `./build/bin/llama-perplexity -m minimind-3-F16.gguf -f <text>` | PPL = 18.07 ± 1e-3 (per B.1 baseline) | PPL unchanged |
| Cross-backend numeric | `./build/bin/test-backend-ops test -o ROPE --backends CPU,TRITON` | Skipped (CPU-only host) | Δ vs CPU reference ≤ 1e-3 fp16 (per Phase A exit criteria in ROADMAP §3) |

### 5.5 What "passes" on CPU host means for B.2

Per Phase 0 audit §0.4, the AOT driver emits 16-byte ELF-magic placeholder CUBINs on a host with no GPU. On CPU host:

- **L1 + L2 (Assert 5)**: pass if and only if all 6 names appear in `reg.get_impls(GGML_OP_ROPE)`. This verifies the **registration path** end-to-end (provider file linked, registration function called, registry inserted). It does NOT verify the kernel body — the placeholder CUBIN has no real Triton code.
- **L3 (`test-backend-ops ROPE`)**: passes because the test harness's `select()` returns the Triton AOT impl, the launcher `load_module_once()` returns `-1` (placeholder CUBIN can't be loaded), `execute()` returns false, the dispatcher falls through to ggml-cpu. So the test exercises the **fallback path** on the CPU host.
- **L4 (MiniMind-3 smoke)**: same — RoPE nodes go through the launcher-fails-fall-through-to-CPU path. The model still works because ggml-cpu has a working RoPE forward.
- **L5 (numerical)**: only verifiable on GPU host. CPU host always shows ggml-cpu numerics.

**Stage 1 exit criteria** (per the B.1 plan's analogous criteria):

- [ ] `test-triton-registry` exits 0 with all 5 asserts passing
- [ ] `test-backend-ops ROPE` exits 0 on CPU-only (all 22 in-scope cases + 29 fall-through-to-CPU cases)
- [ ] MiniMind-3 smoke runs end-to-end
- [ ] 6 ROPE impls (3 modes × 2 dtypes) at priority 100, plus 24 AOT launcher functions
- [ ] `RMSNORM` × 2 dtypes + `ROPE` × 3 modes × 2 dtypes = 10 Triton AOT impls total in the registry (after B.1 + B.2)
- [ ] `GGML_TRITON_WITH_ROPE` CMake option (default ON) gates the new code
- [ ] `ggml-custom-backends.md §3.7` updated with B.2's failure modes
- [ ] `ROADMAP.md` updated to mark B.2 done

**Stage 1 numerical correctness** (deferred to GPU host):

- [ ] `test-backend-ops --backends CPU,TRITON` shows Δ ≤ 1e-3 fp16 vs ggml-cpu reference for ROPE on GPU host
- [ ] MiniMind-3 perplexity diff vs CPU-only baseline ≤ 1e-3 (per ROADMAP §3 Phase A exit criteria)

### 5.6 Per-variant verification (for the implementer to track)

When implementing, the implementer should track **per-variant registration** separately:

| Variant | launcher name | registered in Assert 5? |
|---|---|---|
| `rope_normal_fwd_yarnoff_fp16` | yes | yes |
| `rope_normal_fwd_yarnoff_fp32` | yes | yes |
| `rope_normal_fwd_yarnon_fp16` | yes | (covered by same `_fp16` name match) |
| `rope_normal_fwd_yarnon_fp32` | yes | (covered) |
| `rope_normal_bwd_yarnoff_fp16` | yes | (covered) |
| `rope_normal_bwd_yarnoff_fp32` | yes | (covered) |
| ... × 3 modes × 4 = 12 entries | ... | (name match covers 4 per (mode, dtype)) |

The Assert 5 name-match loop uses `std::string::find` (substring match), so each `(mode, dtype)` entry matches all 4 `(sin_sign, ya_on)` AOT variants under that name. **Assert 5 checks for 6 names; 24 AOT launcher functions are exercised.**

### 5.7 Failure modes that should fail Assert 5 (deliberately)

- Forgetting to call `ggml_triton_register_rope_providers(registry)` in `ggml-triton-provider.cpp` → Assert 5 fails
- Forgetting to add `#include "ggml-triton-provider-rope.h"` in `ggml-triton.cpp` → per-context registry empty, Assert 5 still passes (L1 only) but L2 would fail
- Adding the 24 .c files to CMakeLists but forgetting to add the provider .cpp → Assert 5 fails (no linked symbols)
- Wrong `GGML TRITON_PROVIDER TRITON` enum value in registry → Assert 5 fails
- Adding the function under wrong op key (e.g. `GGML_OP_RMS_NORM` instead of `GGML_OP_ROPE`) → Assert 5 fails

These are the "did you wire it correctly?" failure modes that Assert 5 is designed to catch.

### 5.8 Out of scope for B.2 (per Q0–Q5 decisions)

- **VISION mode**: out of Stage 1. Falls through to ggml-cpu. No test impact.
- **IMROPE mode**: out of Stage 1. Same as VISION.
- **`inplace` ROPE**: out of Stage 1. Per Qwen3 model code, RoPE is always out-of-place. Will fall through to ggml-cpu if `view_src == src[0]`.
- **`n_dims > 128`**: out of Stage 1 (BLOCK_SIZE=128). Falls through to ggml-cpu. No test impact (test_rope's largest n_dims is 128).
- **Real CUBIN verification**: deferred to GPU host. Per Phase 0 audit §0.4.

---

## 6. Stage 2 path & failure modes

Per `ggml-custom-backends.md §3.6` (B.1's known-failure-modes table) and the B.1 plan §Stage 2, this section enumerates **what's deferred to Stage 2**, **what's intentionally out of Stage 1 scope**, and **what failure modes future maintainers must guard against**.

### 6.1 B.1-pattern failure-modes table (B.2 additions, in the same style)

| ID | Limit | Trigger | Behavior | Resolution path |
|---|---|---|---|---|
| **F1** | Placeholder CUBIN on CPU-only host | `scripts/compile_kernels.py` on a host with no GPU | `.c` files embed 16-byte ELF-magic stub. Launcher `load_module_once()` returns `-1`; `execute()` returns `false` → dispatcher falls through to ggml-cpu. | GPU host with Triton 3.7.0 + CUDA 11.0+ + NVIDIA driver (out of B.2 scope; needs `compile_kernels.py` patch per Phase 0 audit §0.4) |
| **F2** | `n_dims > 128` not supported | RoPE node's `src[0]->ne[0] > 128` (BLOCK_SIZE=128 ceiling). MiniMind-3 64M uses 64, 7B uses 128 — both in scope. Hypothetical Qwen-72B with head_dim=256 would be out of scope. | `supports()` returns `false` → per-node fallback to ggml-cpu. | Multi-block variant (`BLOCK_SIZE=256` or family). Deferred. |
| **F3** | `inplace` not supported | `ggml_rope_inplace(...)` API; rarely used in practice (Qwen3 doesn't use it). | `supports()` returns `false` (or provider not invoked at all if `view_src == src[0]`). | Add inplace launcher variants. Deferred. |
| **F4** | VISION/IMROPE not supported | Any `GGML_OP_ROPE` node with `mode` in `{8_mrope_with_visions, 40_imrope}`. | `supports()` returns `false` → ggml-cpu. | Per Q0: deferred to a follow-up (could be B.2.1). |
| **F5** | `compile_kernels.py` global ABI risk | If someone changes `_emit_header` to a global parameterised template | All 24 RoPE `.c/.h` files re-emitted with new ABI. Existing `ggml-triton-provider-rope.cpp`'s 24 call sites would be wrong-arity. | B.1's per-kernel `LAUNCHER_SHAPES` map isolates this — adding a new kernel still uses its own shape. |
| **F6** | No DEBUG log probe for impl selection | Want to see "selected impl X for op Y" at runtime | Dispatcher (`ggml-triton-dispatch.cpp:59`) only logs "unsupported", not "selected" | Add `GGML_LOG_DEBUG("ggml-triton: selected %s for op %s", impl->name, ggml_op_name(node->op))` (per B.1 plan §"Optional follow-up"). One-line change. |
| **F7** | TileLang/CUTLASS AOT re-emit side effect | Any `kernel_registry.json` edit triggers full re-emit of all kernel `.c/.h` (3 modes × 8 = 24 RoPE, plus 4 GELU/SiLU, plus 2 RMSNorm from B.1) | On a non-`default` path, may regenerate existing files with cosmetic diff (per B.1's `byte-compat` fix). | `git diff` should show only cosmetic changes for pre-existing GELU/SiLU files. Verify the LAUNCHER_SHAPES lookup logic isn't regressed. |
| **F8** | MROPE `sections[4]` not validated against `n_dims` | If `sections_sum * 2 > n_dims`, the MROPE cache init will read past the end of the row | `rope_mrope_valid_sections()` check in `supports()` (returns false on violation) | Already implemented in §4.3; defensive. |
| **F9** | F16 round mode mismatch | Triton AOT kernel uses different FP16 rounding than ggml-cpu | Numerical drift in `test_rope` F16 cases. May exceed 1e-3 fp16 tolerance on GPU host. | Use `tl.cast(..., fp_downcast_rounding="rtne")` explicitly (per §2.4). Verified by cross-backend `test-backend-ops --backends CPU,TRITON` on GPU host. |

### 6.2 Stage 2 path (mirror of B.1 plan §Stage 2)

B.2 Stage 1 already has **runtime n_dims, full YaRN support, src[2] freq_factors** (Q1-A, Q2, Q3, Q5 all chose runtime args). So **B.2 has fewer Stage 2 needs than B.1** — most of the "constrain to default" work is already done.

What's left for B.2 Stage 2:

| Stage 2 item | Trigger | Change |
|---|---|---|
| `n_dims > 128` multi-block | Real models with head_dim > 128 (e.g. Qwen-72B at head_dim=256, hypothetical 1B+ models) | Add `BLOCK_SIZE=256` kernel family, 24 more AOT compiles. Per Q2's rationale. |
| `inplace` RoPE launcher family | Some llama.cpp example code uses inplace (rare) | Add `rope_normal_inplace_*` etc. launches (24 more). |
| VISION/IMROPE | Qwen2-VL/Qwen3-VL users (community demand) | Add `vision` and `imrope` mode constexprs, 16 more AOT compiles. Reuses `rotate_pairs` body with different `(n, n_offset, scale)` per VISION. |
| `n_dims > 1024` more granular BLOCK_SIZE | Models with very large head_dim | Stage 2 can add BLOCK_SIZE=512 and BLOCK_SIZE=1024 kernel families |
| MROPE Stage 2 — per-section indep_sects path | VISION family | Reuse the `indep_sects` branch from CPU forward |

The `_v2` naming convention from B.1 plan is **not strictly needed for B.2** because B.2's AOT variant count is already correct (no tolerance-baked limitations to relax). New work in Stage 2 just adds new entries to the registry.

### 6.3 What's intentionally out of B.2 scope

- **Triton 3.7.0 AOT API patch** (per Phase 0 audit §0.4) — separate concern, blocks ALL real-CUBIN generation
- **Multi-GPU dispatch** — ggml-triton backend already supports multi-device; B.2 inherits
- **Cross-backend diff vs ggml-cpu in production CI** — needs GPU runner; deferred to Phase A
- **FlashAttention integration with RoPE** — B.3 separate work
- **Dynamic n_dims dispatch** — Stage 1 hard-gates at 128. Beyond 128 is ggml-cpu.

### 6.4 Documentation updates post-implementation

Per Section 1.3 (file inventory), 3 doc files get updated when B.2 lands:

| File | Update |
|---|---|
| `docs/development/ROADMAP.md` | §1: Task 4 status `🔄 进行中 (B.1 ✅, B.2 进行中)`. §2: B.2 marked ✅. §3 op table: ROPE row from `❌` to `✅ Triton AOT (B.2)`. §3: B.2 完成情况 subsection. |
| `docs/development/ggml-custom-backends.md` | §3.5: add 6 ROPE rows (3 modes × 2 dtypes). New §3.7: B.2 已知失败模式 (F1-F9 above). |
| `docs/development/test-pyramid.md` | Update op-coverage marker to include `GGML_OP_ROPE` and `GGML_OP_ROPE_BACK` (per B.1's RMSNorm pattern). |

### 6.5 What "B.2 is done" means — concrete definition

**B.2 Stage 1 exit criteria** (analogous to B.1):

- [ ] Assert 5 exits 0 in `test-triton-registry`
- [ ] `test-backend-ops ROPE` exits 0 on CPU-only host (all 22 in-scope cases green)
- [ ] MiniMind-3 smoke runs end-to-end; RoPE nodes dispatched to Triton AOT provider (verifiable via `ggml_log` if F6 is implemented; otherwise via load-module-once observation)
- [ ] 6 ROPE impls (3 modes × 2 dtypes) at priority 100, plus 24 AOT launcher functions
- [ ] `RMSNORM` × 2 dtypes + `ROPE` × 3 modes × 2 dtypes = 10 Triton AOT impls total in the registry (after B.1 + B.2)
- [ ] `GGML_TRITON_WITH_ROPE` CMake option (default ON) gates the new code
- [ ] `ggml-custom-backends.md §3.7` updated with B.2's failure modes
- [ ] `ROADMAP.md` updated to mark B.2 done

**Stage 1 numerical correctness** (deferred to GPU host):

- [ ] `test-backend-ops --backends CPU,TRITON` shows Δ ≤ 1e-3 fp16 vs ggml-cpu reference for ROPE on GPU host
- [ ] MiniMind-3 perplexity diff vs CPU-only baseline ≤ 1e-3 (per ROADMAP §3 Phase A exit criteria)

### 6.6 The "what's special about B.2 vs B.1" lessons-learned note

For the post-implementation write-up, B.2's key differences from B.1:

1. **24 AOT variants** vs B.1's 4 — the per-kernel naming convention (`<mode>_<sin>_<yarn>_<dtype>`) encodes more axes
2. **Three launcher-shape entries** vs B.1's two (MROPE adds a 3-ptr variant with `freq_factors`)
3. **No tolerance gate** in supports() — Q1-A chose full runtime args, so the kernel handles any value
4. **YaRN has pre-launch host math** (corr_dims[]) — B.1's eps was constexpr-baked; B.2's YaRN is more complex
5. **MROPE 4-axis theta** is a kernel-body branch (not just different args) — splits the kernel into non-MROPE/MROPE families with different cache-init code

These are worth recording in `ggml-custom-backends.md §3.7` as the "design decisions" rationale.

---

## Summary of decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| Q0 | Mode coverage | **B** (NORMAL+NEOX+MROPE) | MiniMind-3 covers NEOX; MROPE enables Qwen2-VL; cost ~1.3× B.1's NEOX-only path (NEOX/MROPE share `rotate_pairs` body) |
| Q1 | op_params strategy | **A** (full runtime args) | Single launcher handles all models, no tolerance gate to break |
| Q2 | n_dims + BLOCK_SIZE | **Single launcher + runtime n_dims + mask** | 50% thread waste on small n_dims is fine; saves launcher count |
| Q3 | src[2] freq_factors | **Supported** (runtime ptr) | Marginal cost; enables Phi-3 path |
| Q4 | Forward/backward | **Both** (constexpr SIN_SIGN) | Marginal cost (one more constexpr axis); supports training |
| Q5 | YaRN | **Full impl** (constexpr YA_ON) | Marginal cost; supports YaRN-tuned models |
| AOT | Strategy | **A** (full constexpr) | 24 AOT compiles; each variant is a tight, single-path kernel |

---

## Next step

After user approval, the writing-plans skill will produce the implementation plan (bite-sized tasks, 2-5 min each, TDD-strict, with verification commands at each step). The plan will follow B.1's structure (`docs/superpowers/plans/2026-06-10-rmsnorm-triton-aot.md`) as a template, adapting for B.2's larger AOT count (24 vs 4) and Stage 2 path.
