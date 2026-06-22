# B.3 FlashAttn Triton AOT Provider — Design

> **For agentic workers:** This is the design spec for the B.3 milestone of `docs/development/ROADMAP.md`. After approval, the next step is to invoke the **writing-plans** skill to produce the implementation plan. This spec is the input to that plan.
>
> **Status:** Sections 1 + 2 (corrected post-Oracle) + 2.D + 2.E committed (2026-06-21). Sections 3 (AOT launcher ABI), 4 (Provider file design), 5 (Test & verification), 6 (Stage 2 path & failure modes) pending — to be added in subsequent review rounds.
>
> **Brainstormed:** 2026-06-21, via brainstorming skill (Q0–Q6 + 3 architecture decisions), Oracle review on Sections 1+2 found 11 issues (2 critical, 4 high, 3 medium, 2 low) — all 11 applied in this spec.
>
> **B.1 reference implementation:** `docs/superpowers/plans/2026-06-10-rmsnorm-triton-aot.md` (commit `418f88b9f`).
> **B.2 reference implementation:** `docs/superpowers/plans/2026-06-12-rope-triton-aot.md` (commits `8acf85656`–`250d50695`).
>
> **Brainstorming decisions:**
> - Q0 (scope) = **Forward only** (`FLASH_ATTN_EXT`; backward via ggml-cpu fallback)
> - Q1 (head_dim range) = **{64, 96, 128}** (3 constexpr; MiniMind-3 = 96 + 8 heads MHA)
> - Q2 (mask support) = **nullptr + 静态 causal only** (CAUSAL=1 constexpr; mask!=nullptr → cpu fallback)
> - Q3 (prefill/decode strategy) = **Two kernels** (separate `flash_attn_prefill.py` + `flash_attn_decode.py`)
> - Q4 (dtype) = **fp16 + fp32** (consistent with B.1 RMSNorm / B.2 RoPE)
> - Q5 (AOT strategy) = **Full constexpr** (3 head_dim × 2 dtype × 2 kernel = 12 AOT compiles)
> - Q6 (stride layout) = **Standard contiguous** (Q/K/V 4D `[D, S, H, B]` with nb[0] contiguous only)
>
> **Architecture decisions (post-brainstorming):**
> - **Provider file count**: 1 file (`ggml-triton-provider-flash-attn.{h,cpp}`) with 12 supports + 12 execute (mirror B.2 pattern)
> - **Decode split-KV reduction**: 2-pass (kernel writes partials (M, S, V_unnormalized) to scratch → host CPU reduce, matches CPU reference `ops.cpp:8897-8925` `use_split_kv_path`)
> - **BLOCK sizes**: BLOCK_Q=128, BLOCK_KV=64 (standard FA-2, sm_80 SRAM fits 128×64=8K tile)

---

## 1. Architecture & file inventory

The B.3 implementation mirrors the B.1/B.2 4-step loop (kernel source → AOT launcher → provider → test) with the dimensions of variation: **3 head_dim × 2 dtype × 2 kernel (prefill/decode) = 12 AOT compiles**.

### 1.1 New files (3)

| # | File | Purpose |
|---|---|---|
| 1 | `triton_kernels/flash_attn_prefill.py` | Triton DSL prefill kernel. One `@triton.jit` function with `BLOCK_Q=128`, `BLOCK_KV=64`, `HEAD_DIM`, `DTYPE_ID`, `CAUSAL=1` constexprs. Standard FA-2 online-softmax algorithm. |
| 2 | `triton_kernels/flash_attn_decode.py` | Triton DSL decode kernel. One `@triton.jit` function. Split-KV: each program handles 1 Q row × 1 KV chunk; writes (M, S, V_unnormalized) partials to scratch. |
| 3 | `ggml/src/ggml-triton/ggml-triton-provider-flash-attn.{h,cpp}` | Provider implementation — 12 `supports` + 12 `execute` functions (3 head_dim × 2 dtype × 2 kernel) + 1 register function. Dispatches prefill vs decode based on `neq1`. **Decode execute is multi-step** (see §1.5). |

### 1.2 Modified files (7)

| # | File | What changes |
|---|---|---|
| 1 | `scripts/kernel_registry.json` | Add 2 entries (`flash_attn_prefill`, `flash_attn_decode`), each with 6 variants (3 head_dim × 2 dtype). 12 variants total. |
| 2 | `scripts/compile_kernels.py` | Add 2 `LAUNCHER_SHAPES` entries. Prefill: 4 ptrs + 6 ints (neq1, neq2, neq3, nek1, S, n_heads) + 1 int (rows) + 1 float (scale) + 5 constexpr = ~13 args after stream. Decode: 5 ptrs (q, k, v, dst, scratch) + 7 ints (neq1, neq2, neq3, nek1, S, n_heads, q_pos) + 2 ints (num_kv_chunks, rows) + 1 float (scale) + 5 constexpr = ~15 args after stream. |
| 3 | `ggml/src/ggml-triton/kernels/include/triton_kernels.h` | Append 12 `#include` lines for the new launchers. Update top-of-file comment block to document the per-family pointer-slot counts. |
| 4 | `ggml/src/ggml-triton/CMakeLists.txt` | Add option `GGML_TRITON_WITH_FLASH_ATTN` (default ON, gated by `NOT GGML_TRITON_CPU_ONLY`, mirrors `GGML_TRITON_WITH_RMS_NORM` / `GGML_TRITON_WITH_ROPE`). Append 12 generated `.c` files to `GGML_TRITON_GPU_SRC`. Append `ggml-triton-provider-flash-attn.cpp` to `GGML_TRITON_GPU_SRC`. `target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_HAS_FLASH_ATTN)` placed AFTER `ggml_add_backend_library()` (same fix as B.1 for `GGML_TRITON_HAS_RMSNORM`). |
| 5 | `ggml/src/ggml-triton/ggml-triton-provider.cpp` | Add `#ifdef GGML_TRITON_HAS_FLASH_ATTN` block with `#include "ggml-triton-provider-flash-attn.h"` + `ggml_triton_register_flash_attn_providers(registry);` inside `std::call_once`. |
| 6 | `ggml/src/ggml-triton/ggml-triton.cpp` | Same `#ifdef` block + register call in **both** CPU-only and GPU branches of `ggml_backend_triton_init`. |
| 7 | `tests/test-triton-registry.cpp` | Add Assert 6: iterate `reg.get_impls(GGML_OP_FLASH_ATTN_EXT)`, filter `provider == GGML_TRITON_PROVIDER_TRITON`, check 12 names (3 head_dim × 2 dtype × 2 kernel). Return distinct `rc=7` on miss. (Assert 4 = rc 4 for RMSNorm, Assert 5 = rc 5 for RoPE, Assert 6 = rc 7 for FlashAttn — preserves existing exit codes.) |

### 1.3 Generated files (12)

| Pattern | Count |
|---|---|
| `ggml/src/ggml-triton/kernels/generated/flash_attn_prefill_hd{64,96,128}_f{16,32}_sm80.{c,h}` | 6 |
| `ggml/src/ggml-triton/kernels/generated/flash_attn_decode_hd{64,96,128}_f{16,32}_sm80.{c,h}` | 6 |

### 1.4 Out of scope for B.3 (explicit)

- `GGML_OP_FLASH_ATTN_BACK` (backward) — training needs go through ggml-cpu fallback
- Mask tensors where `dst->src[3] != nullptr` — those nodes fall back to ggml-cpu
- paged KV cache / custom stride layout — `supports()` requires standard contiguous only
- bf16 / quantized K/V — out of scope; CPU reference supports f16/f32 only
- KV head count != query head count (GQA/MQA) — MiniMind-3 is MHA; defer to Stage 2
- `n_heads > 32` — MiniMind-3 has 8; defer to Stage 2
- Softcap (Gemma 2) — not used by MiniMind-3 or Qwen3 family; defer
- V head dim != K head dim — reference supports it (separate DK/DV); keep deferred

### 1.5 Note on decode multi-step execute (Oracle LOW #10)

> **Decode is the first multi-step execute() in the Triton subsystem.** B.1 RMSNorm and B.2 RoPE `execute()` calls exactly one CUDA launcher; B.3 decode's `execute()` does:
> 1. Launch decode kernel (writes partials to device scratch)
> 2. `cuMemcpyDtoHAsync` of scratch to host
> 3. `cuStreamSynchronize` (barrier so kernel writes visible to CPU)
> 4. CPU reduce pass over partials (writes final dst)
> 5. Return
>
> The provider struct gains new persistent state (`decode_scratch`, `decode_scratch_size`, possibly a pinned-host buffer). Lifecycle:
> - **Allocate**: lazy on first decode call via `cuMemAlloc`. Resize if current buffer too small.
> - **Free**: provider destructor (or context teardown).
> - **No race**: provider serializes `execute()` per node; `cuStreamSynchronize` ensures kernel writes are visible to CPU reduce.
>
> The provider's `register_impl` still returns a single function pointer per impl; the multi-step nature is internal to the `execute()` body.

### 1.6 Cumulative registry count after B.3

| 来源 | impls |
|---|---|
| B.1 RMSNorm | 4 Triton AOT + 1 CPU |
| B.2 RoPE | 6 Triton AOT + 1 CPU |
| **B.3 FlashAttn** | **12 Triton AOT** (待加) + 1 CPU (FlashAttn CPU provider exists, covers cases our supports() rejects) |
| PR #1 GELU/SiLU | 4 Triton AOT |
| PR #1 ADD/MUL (TileLang) | 2 Triton AOT (conditional) |
| PR #1 MUL_MAT (CUTLASS) | 4 Triton AOT (conditional) |
| **Total Triton AOT** | **28** (default ON) / 32 (with TileLang+CUTLASS) |

10 op families covered; Phase B exit criteria (`test-backend-ops 100% pass` for RMS_NORM/ROPE/FLASH_ATTN_EXT + `test-triton-registry` 3 new provider asserts) achieved.

---

## 2. Triton DSL kernel design

### 2.A `triton_kernels/flash_attn_prefill.py` — Standard FA-2 Tiled

**Algorithm reference**: `ggml-cpu/ops.cpp:8486` `ggml_compute_forward_flash_attn_ext_tiled`

**Constants (constexpr)**:
- `BLOCK_Q = 128` — query tile size
- `BLOCK_KV = 64` — key/value tile size
- `HEAD_DIM ∈ {64, 96, 128}` — K/V head dim (Q head dim == K head dim, MHA only)
- `DTYPE_ID ∈ {0, 1}` — 0=fp16, 1=fp32 (constexpr for type dispatch)
- `CAUSAL = 1` — always causal in Stage 1

**Runtime args** (4 ptrs + 7 ints + 1 float = 12 args after stream):
- `q_ptr, k_ptr, v_ptr, dst_ptr` — device pointers
- `{neq1, neq2, neq3, nek1, S, n_heads, rows}` — 7 int32s (`rows = neq2 * neq3`)
- `scale` — float, `1.0/sqrt(HEAD_DIM)` (precomputed on host)

**Grid**: `2D (cdiv(neq1, BLOCK_Q), rows)` where `rows = neq2 * neq3`

**Body outline** (5 stages):

1. **Load Q tile** (BLOCK_Q × HEAD_DIM):
   - `q = tl.load(q_ptr + head_idx * stride_q_h + (q_block * BLOCK_Q + offs_q[None, :]) * stride_q_d, mask=mask_q, other=0.0).to(tl.float32)`
   - Mask handles `neq1 < BLOCK_Q` (runtime mask, Triton-idiomatic — no CPU fallback for short prompts)
2. **Init online softmax state**:
   - `m_i = tl.full((BLOCK_Q,), -inf, tl.float32)`
   - `l_i = tl.zeros((BLOCK_Q,), tl.float32)`
   - `acc = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)`
3. **Loop over KV blocks** (causal: `0` to `q_block * BLOCK_Q + BLOCK_Q`):
   - For each `kv_block`:
     - Load K tile: `k = tl.load(k_ptr + ..., (kv_block * BLOCK_KV + offs_kv[:, None]) * stride_k_d)` (BLOCK_KV × HEAD_DIM)
     - Load V tile: `v = tl.load(v_ptr + ..., (kv_block * BLOCK_KV + offs_kv[:, None]) * stride_v_d)` (BLOCK_KV × HEAD_DIM)
     - `qk = tl.dot(q, tl.trans(k)) * scale` (BLOCK_Q × BLOCK_KV, fp32)
     - Causal mask: `qk = tl.where(offs_kv[None, :] + kv_block*BLOCK_KV > offs_q[:, None] + q_block*BLOCK_Q, -inf, qk)`
     - Online softmax (standard FA-2):
       ```
       m_new = tl.maximum(m_i, tl.max(qk, axis=1))
       alpha = tl.exp(m_i - m_new)
       qk -= m_new[:, None]                    # subtract max for numerical stability
       p = tl.exp(qk)                           # BLOCK_Q × BLOCK_KV
       l_i = l_i * alpha + tl.sum(p, axis=1)
       acc = acc * alpha[:, None] + tl.dot(p, v)
       m_i = m_new
       ```
4. **Normalize**: `out = acc / l_i[:, None]`
5. **Store** (cast per DTYPE_ID):
   - `tl.store(dst_ptr + ..., out.to(tl.float32), mask=...)` — dst is fp32 per `ops.cpp:8883` (`GGML_ASSERT(nb0 == sizeof(float))`)

**Stage 1 simplifications** (documented inline):
- No padding for HEAD_DIM=96 (Triton `tl.dot` accepts arbitrary K/N; 96 is 16-aligned for Tensor Core)
- Handles `neq1 < BLOCK_Q` via runtime mask (Triton-idiomatic, not CPU fallback)
- Doesn't handle `neq2 > 32` (MHA, MiniMind-3 has 8 heads)

---

### 2.B `triton_kernels/flash_attn_decode.py` — Split-KV (1 Q row × 1 KV chunk)

**Algorithm reference**: `ggml-cpu/ops.cpp:8248` `ggml_compute_forward_flash_attn_ext_f16_one_chunk`

**Constants**: same as prefill (BLOCK_KV=64, HEAD_DIM constexpr, CAUSAL=1)

**Runtime args** (5 ptrs + 8 ints + 1 float = 14 args after stream):
- `q_ptr, k_ptr, v_ptr, dst_ptr, scratch_ptr` — 5 device pointers
- `{neq1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows}` — 9 int32s (`q_pos = neq1 - 1`, current generation step for causal mask; `num_kv_chunks = cdiv(nek1, BLOCK_KV)`)
- `scale` — float

**Grid**: `2D (num_kv_chunks, rows)`

**Body outline** (5 stages):

1. **Load 1 Q row** (1 × HEAD_DIM):
   - `q = tl.load(q_ptr + head_idx * stride_q_h + (q_pos * stride_q_d) + offs_d[None, :], mask=mask_d, other=0.0).to(tl.float32)`
2. **Init online softmax state** (per chunk):
   - `m_i = -inf`, `l_i = 0`, `acc = tl.zeros((1, HEAD_DIM), tl.float32)`
3. **Loop over KV sub-tiles within this chunk**:
   - For each `sub_kv` in `[0, BLOCK_KV)` (decode processes one BLOCK_KV tile per chunk):
     - `abs_kv = chunk_idx * BLOCK_KV + sub_kv`
     - Load K[abs_kv, :], V[abs_kv, :]
     - `qk = tl.sum(q * k[None, :], axis=1) * scale` (1 × 1)
     - Causal mask: `qk = tl.where(abs_kv > q_pos, -inf, qk)`
     - Online softmax update (BLOCK_Q=1):
       ```
       m_new = tl.maximum(m_i, qk)
       alpha = tl.exp(m_i - m_new)
       p = tl.exp(qk - m_new)
       l_i = l_i * alpha + p
       acc = acc * alpha + p * v  # 1 × HEAD_DIM outer product
       m_i = m_new
       ```
4. **Write partials to scratch** (canonical layout, unnormalized V):
   - Per `(chunk_idx, head_idx)` slot in scratch, layout **`[q_head * batch + batch_idx][chunk_idx][M, S, V_unnormalized]`** (heads slowest, chunks contiguous within head — matches `ops.cpp:8900, 8818`).
   - **M**: `tl.store(scratch + base_M_offset, m_i)`
   - **S**: `tl.store(scratch + base_S_offset, l_i)`
   - **V_unnormalized**: `tl.store(scratch + base_V_offset + offs_d, acc[0, :])` — **the FP32 accumulator BEFORE `/S`**. Do NOT divide by `l_i` here; reduce pass handles rescaling.
5. **(No output write from this kernel — reduce pass writes dst)**

**Scratch size**: `n_heads × batch × num_kv_chunks × (2 + HEAD_DIM) × sizeof(float)` bytes.
- MiniMind-3 (8 heads, batch=1, num_chunks=64, head_dim=96): `8 × 1 × 64 × 98 × 4 = 200KB`

**Scratch allocation/lifecycle**:
- **Owner**: provider's per-context state (`ctx->decode_scratch`, `ctx->decode_scratch_size`).
- **Allocate**: `cuMemAlloc` on first decode call; persistent across `execute()` calls.
- **Resize**: if `n_heads × batch × num_kv_chunks × (2+HEAD_DIM) > current_size`, free + realloc.
- **Transfer before reduce**: `cuMemcpyDtoHAsync(scratch_host, scratch_device, size, ctx->cu_stream)` then `cuStreamSynchronize(ctx->cu_stream)` — reduce runs on host thread.
- **Free**: provider destructor (or ctx teardown).
- **No race**: provider serializes `execute()` per node; `cuStreamSynchronize` barrier ensures kernel writes are visible to CPU reduce.

---

### 2.C Reduce pass (host-side C++, not Triton)

**Why not a Triton kernel**: Reduction over a small `(num_kv_chunks × HEAD_DIM)` per-`(head, batch)` buffer is fast on CPU and avoids an extra AOT variant. Matches `ggml_flash_attn_ext_reduce_partials` (`ops.cpp:8776`) directly.

**Algorithm** (per `(head, batch)`):

```
m_final = -inf
l_final = 0.0
v_final = zeros(HEAD_DIM, fp32)
for chunk_idx in 0..num_kv_chunks:
    base = (head * batch + batch_idx) * num_kv_chunks * (2+HEAD_DIM) + chunk_idx * (2+HEAD_DIM)
    m_chunk = scratch_host[base + 0]                       # M_unnormalized
    s_chunk = scratch_host[base + 1]                       # S_chunk
    v_chunk = scratch_host[base + 2 : base + 2 + HEAD_DIM] # V_unnormalized
    m_new = max(m_final, m_chunk)
    alpha = exp(m_final - m_new)
    beta  = exp(m_chunk - m_new)
    l_final = l_final * alpha + s_chunk * beta
    v_final = v_final * alpha + v_chunk * beta
    m_final = m_new
dst[head, batch] = v_final / l_final   # final normalize, write to dst (fp32)
```

**Cost**: `O(num_kv_chunks × HEAD_DIM)` per `(head, batch)`. For MiniMind-3 (num_chunks=64, head=96, heads=8, batch=1): ~50K FLOPs on CPU, < 1ms.

---

## 2.D Stage 2+ explicitly deferred (Oracle #11)

| Deferred | Notes |
|---|---|
| Custom mask tensor (`dst->src[3] != nullptr`) | Reference computes `slope * mask_value` (ALiBi); MiniMind-3 doesn't use. |
| `logit_softcap` (tanh squash before mask) | Reference `ops.cpp:8385`; Gemma 2 style; out of scope. |
| `sinks` tensor (`dst->src[4]`) | Reference `ops.cpp:8446`; some chat templates use; Stage 2. |
| GQA/MQA (`neq2 != nek2`, broadcast factors) | Reference `ops.cpp:8296-8300`; Qwen3-198M-A64M MoE needs it. |
| bf16 / quantized K/V | Reference handles F16/F32 only in split-KV path (line 8894). |
| Paged KV cache | Not in reference at all; vLLM-style; Stage 2+. |
| Backward pass (`FLASH_ATTN_BACK`) | Out of scope per Q0; ggml-cpu `ops.cpp:8994` is reference. |
| V head dim ≠ K head dim (DV ≠ DK) | Reference supports it (separate DK/DV); keep deferred. |
| `n_heads > 32` | MiniMind-3 has 8; Qwen3-72B has 64; Stage 2. |
| Long-context (≥8K tokens) optimizations | FA-2 already O(n) memory; perf tuning for ≥8K is Stage 2. |

---

## 2.E Dispatch gate table (Oracle #6)

Provider's `supports()` returns true only when **all** conditions hold; otherwise returns false and the node falls back to ggml-cpu.

| Input shape | Kernel choice | `supports()` condition |
|---|---|---|
| `neq1 ≥ 1`, `nek1 ≥ 1`, `mask == nullptr` | **prefill** (when `neq1 > 1` or `neq1 % BLOCK_Q != 0` with runtime mask) | `neq1 >= 1`; `neq2 == nek2 == nev2` (MHA only); `neq2 * neq3 <= 32` (head count gate); `head_dim ∈ {64, 96, 128}`; `q.type == k.type == v.type ∈ {GGML_TYPE_F16, GGML_TYPE_F32}`; contiguous (`nb[0] == type_size` and `nb[i+1] == nb[i] * ne[i]` for all dims) |
| `neq1 == 1`, `nek1 ≥ 1`, `mask == nullptr` | **decode** (split-KV) | Same dtype/contiguous/head_dim gate as prefill; `neq1 == 1`; **no** `nek1 ≥ 512` CPU threading threshold (irrelevant on GPU — always route to decode) |
| Anything else (mask!=nullptr, non-contiguous, head_dim ∉ {64,96,128}, n_heads > 32, GQA) | CPU fallback | `supports()` returns false |

**Note**: `neq1 ∈ [1, BLOCK_Q)` is supported by both kernels (prefill handles via runtime mask; decode dispatches on `neq1 == 1` exactly).

---

## 3. AOT launcher ABI

### 3.1 Triton DSL kernel signatures (host-binding)

**Prefill**:
```python
@triton.jit
def flash_attn_prefill_kernel(
    q_ptr, k_ptr, v_ptr, dst_ptr,                          # 4 ptrs
    neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, # 8 int32
    scale,                                                   # float
    BLOCK_Q: tl.constexpr,                                   # 128
    BLOCK_KV: tl.constexpr,                                  # 64
    HEAD_DIM: tl.constexpr,                                  # {64, 96, 128}
    DTYPE_ID: tl.constexpr,                                  # {0, 1}
    CAUSAL: tl.constexpr,                                    # 1
):
```
**13 runtime args after stream**. `num_q_blocks = cdiv(neq1, 128)` is pre-computed on host (decouples grid sizing from BLOCK_Q constexpr; lets the template use `grid_mode="exact"` without per-shape divisor).

**Decode**:
```python
@triton.jit
def flash_attn_decode_kernel(
    q_ptr, k_ptr, v_ptr, dst_ptr, scratch_ptr,    # 5 ptrs
    neq1, neq2, neq3, nek1, S, n_heads, q_pos,     # 7 int32
    num_kv_chunks, rows,                          # 2 int32
    scale,                                        # float
    BLOCK_KV: tl.constexpr,                       # 64
    HEAD_DIM: tl.constexpr,                       # {64, 96, 128}
    DTYPE_ID: tl.constexpr,                       # {0, 1}
    CAUSAL: tl.constexpr,                         # 1
):
```
**15 runtime args after stream**. **4 constexpr** (no `BLOCK_Q` — decode processes 1 Q row per program; Q-tile size is irrelevant).

### 3.2 AOT launcher C signatures

| Kernel | launcher name | C signature (runtime args after stream) |
|---|---|---|
| prefill | `triton_launch_flash_attn_prefill_hd{HD}_fp{DT}_sm80` | `(stream, q, k, v, dst, neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale)` — **13 args** |
| decode | `triton_launch_flash_attn_decode_hd{HD}_fp{DT}_sm80` | `(stream, q, k, v, dst, scratch, neq1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale)` — **15 args** |

**Naming format**: use `fp16`/`fp32` (matches B.1/B.2 dtype strings, not `f16`/`f32` — keeps byte-compat with existing variant dtype keys).

**12 launcher names** (3 head_dim × 2 dtype × 2 kernel):
- `triton_launch_flash_attn_prefill_hd64_fp16_sm80` / `_fp32_sm80`
- `triton_launch_flash_attn_prefill_hd96_fp16_sm80` / `_fp32_sm80`
- `triton_launch_flash_attn_prefill_hd128_fp16_sm80` / `_fp32_sm80`
- `triton_launch_flash_attn_decode_hd64_fp16_sm80` / `_fp32_sm80`
- `triton_launch_flash_attn_decode_hd96_fp16_sm80` / `_fp32_sm80`
- `triton_launch_flash_attn_decode_hd128_fp16_sm80` / `_fp32_sm80`

Wait — re-check the decode launcher names. The 12 names should be:
- prefill: 3 head_dim × 2 dtype = 6 names (`hd{64,96,128}_fp{16,32}_sm80`)
- decode: 3 head_dim × 2 dtype = 6 names (`hd{64,96,128}_fp{16,32}_sm80`)

Total 12, all unique. (The example 5-line block above has a typo — `fp32_fp32_sm80` should be `fp32_sm80`. Correcting in the actual file below.)

**Corrected 12 launcher names**:
```
triton_launch_flash_attn_prefill_hd64_fp16_sm80
triton_launch_flash_attn_prefill_hd64_fp32_sm80
triton_launch_flash_attn_prefill_hd96_fp16_sm80
triton_launch_flash_attn_prefill_hd96_fp32_sm80
triton_launch_flash_attn_prefill_hd128_fp16_sm80
triton_launch_flash_attn_prefill_hd128_fp32_sm80
triton_launch_flash_attn_decode_hd64_fp16_sm80
triton_launch_flash_attn_decode_hd64_fp32_sm80
triton_launch_flash_attn_decode_hd96_fp16_sm80
triton_launch_flash_attn_decode_hd96_fp32_sm80
triton_launch_flash_attn_decode_hd128_fp16_sm80
triton_launch_flash_attn_decode_hd128_fp32_sm80
```

### 3.3 `LAUNCHER_SHAPES` entries (extended schema: `grid_param_y`, `grid_mode_y`)

Oracle §1 critical fix: grid must be 2D. Schema extended with `grid_param_y` / `grid_mode_y` (optional; absent → 1D launch, backward compatible with B.1/B.2 entries).

**`grid_mode` semantics** (Oracle §5 fix):
- Default: `shape.get("grid_mode", "divide")` (backward compat)
- `"divide"`: `grid = (grid_param + divisor - 1) / divisor`, where `divisor = shape.get("grid_divisor", block)`. If `grid_divisor` is absent, falls back to `block` (= `kTritonBlockSize_{name}` from the launcher source).
- `"exact"`: `grid = grid_param` (no division). `grid_divisor` is ignored.

```python
LAUNCHER_SHAPES = {
    # ... existing entries unchanged (default, rms_norm_*, rope_*) — backward compat via grid_mode default ...
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
```

### 3.4 Template branch (in `_emit_source`)

The launcher template branches on whether the shape has a Y-axis grid:

```python
if "grid_param_y" in shape:
    grid_x_expr = _format_grid_expr(shape, axis="x")
    grid_y_expr = _format_grid_expr(shape, axis="y")
    cuLaunch_kernel = (
        f"cuLaunchKernel(g_function, {grid_x_expr}, {grid_y_expr}, 1, "
        f"block, 1, 1, 0, stream, args, NULL)"
    )
else:
    # Existing 1D path (B.1/B.2 backward compat)
    grid_expr = _format_grid_expr(shape, axis="x")
    cuLaunch_kernel = (
        f"cuLaunchKernel(g_function, {grid_expr}, 1, 1, block, 1, 1, 0, stream, args, NULL)"
    )
```

`_format_grid_expr(shape, axis)` returns:
- `axis="x"`: `f"({shape['grid_param']} + {divisor} - 1) / {divisor}"` if `grid_mode="divide"`, else `shape["grid_param"]`
- `axis="y"`: same logic with `grid_param_y` / `grid_mode_y` (and `divisor` from `grid_divisor_y` if present, else same X divisor)

### 3.5 `Variant.tag` extension (Oracle #2 fix)

`compile_kernels.py:54-58` `Variant.tag` is extended to fold `HEAD_DIM` (and continues to fold `SIN_SIGN`/`YA_ON` for B.2 RoPE compatibility):

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

Produced tag formats:
- B.1 RMSNorm (no axes): `fp16_sm80` ✓ (unchanged)
- B.2 RoPE (SIN_SIGN+YA_ON): `fwd_yarnoff_fp16_sm80` ✓ (unchanged)
- **B.3 FlashAttn (HEAD_DIM only): `hd96_fp16_sm80`** ✓ (new; ensures 3 head_dim variants of same kernel/dtype don't collide)

### 3.6 AOT variant matrix (12 entries)

| kernel | HEAD_DIM | DTYPE | launcher name | sig token count |
|---|---|---|---|---|
| prefill | 64 | fp16 | `..._prefill_hd64_fp16_sm80` | 13 runtime + 5 constexpr = **18** |
| prefill | 64 | fp32 | `..._prefill_hd64_fp32_sm80` | 18 |
| prefill | 96 | fp16 | `..._prefill_hd96_fp16_sm80` | 18 |
| prefill | 96 | fp32 | `..._prefill_hd96_fp32_sm80` | 18 |
| prefill | 128 | fp16 | `..._prefill_hd128_fp16_sm80` | 18 |
| prefill | 128 | fp32 | `..._prefill_hd128_fp32_sm80` | 18 |
| decode | 64 | fp16 | `..._decode_hd64_fp16_sm80` | 15 runtime + **4** constexpr = **19** |
| decode | 64 | fp32 | `..._decode_hd64_fp32_sm80` | 19 |
| decode | 96 | fp16 | `..._decode_hd96_fp16_sm80` | 19 |
| decode | 96 | fp32 | `..._decode_hd96_fp32_sm80` | 19 |
| decode | 128 | fp16 | `..._decode_hd128_fp16_sm80` | 19 |
| decode | 128 | fp32 | `..._decode_hd128_fp32_sm80` | 19 |

(Oracle §3 fix: decode has 4 constexpr, not 5. Oracle §4 fix: decode has 15 runtime, not 14.)

### 3.7 Registry signatures (per kernel_registry.json)

Each variant's `signature` field:
- **prefill** (5 constexpr: `BLOCK_Q=128, BLOCK_KV=64, HEAD_DIM, DTYPE_ID=0/1, CAUSAL=1`):
  - `*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,f32,128,64,64,0,1` (HD=64, fp16)
  - `*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,f32,128,64,96,0,1` (HD=96, fp16)
  - `*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,f32,128,64,128,0,1` (HD=128, fp16)
  - (×2 for fp32: replace `*fp16` → `*fp32`, `DTYPE_ID` 0 → 1)
- **decode** (4 constexpr: `BLOCK_KV=64, HEAD_DIM, DTYPE_ID=0/1, CAUSAL=1` — **no BLOCK_Q**):
  - `*fp16,*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,i32,f32,64,64,0,1` (HD=64, fp16)
  - `*fp16,*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,i32,f32,64,96,0,1` (HD=96, fp16)
  - `*fp16,*fp16,*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,i32,f32,64,128,0,1` (HD=128, fp16)
  - (×2 for fp32)

### 3.8 B.2 RoPE bug: functional, not latent (Oracle LOW #6, info only)

The B.2 RoPE `rope.py:48-49` does 1 row per program (`pid = program_id(0)`, load `a_ptr + pid * n_dims`). The current launcher grid = `(rows + 127) / 128`. For MiniMind-3 with `rows=8` (or any rows < 128), grid = 1 → only `pid=0` runs → rows 1-7 unrotated. **Functional bug** (not latent), triggered by any multi-row RoPE input reaching the kernel.

**Stage 2 fix (not in B.3 scope)**: change B.2 `LAUNCHER_SHAPES["rope_normal"|"rope_neox"|"rope_mrope"]` from current (implicit `grid_mode="divide"`, `grid_param="rows"`, divisor 128) to explicit `grid_mode="exact"`, `grid_param="rows"`. Re-run `compile_kernels.py` to regenerate 24 B.2 .c files. Re-test B.2 (Assert 5 still passes; Stage 1 placeholder CUBIN unaffected).

### 3.9 B.1 RMSNorm has a separate `pid` bug (Oracle LOW #6 addendum)

**Distinct issue from B.2 RoPE** — do NOT conflate. RMSNorm's `grid_param="N"` (N = row length, e.g. 768) and `rms_norm.py:44` kernel does NOT use `pid` (`tl.load(x_ptr + offsets, ...)` always loads row 0). The grid computation `(N + 1023) / 1024` produces `grid=1` (for N=768) and the kernel always processes row 0. **Functional bug.**

**Stage 2 audit needed**: B.1 RMSNorm requires adding `pid = tl.program_id(0)` to the kernel, plus `num_blocks` runtime arg (host-computed `cdiv(N, 1024)`), plus `grid_mode="exact"`. Larger fix than B.2 RoPE.

**B.3 implication**: B.3 kernels use `pid = program_id(0)` and `program_id(1)` correctly from the start (Section 2 design), so B.3 is **immune to this bug class**. Only B.1/B.2 retro-fix needed in Stage 2.


---

## 4. Provider file design

### 4.1 File structure

`ggml/src/ggml-triton/ggml-triton-provider-flash-attn.{h,cpp}` mirrors B.2's `ggml-triton-provider-rope.{h,cpp}` structure:

**Header**:
```cpp
#pragma once
#include "ggml-triton-provider.h"

void ggml_triton_register_flash_attn_providers(ggml_triton_op_registry & registry);
```

**CPP** — top-level structure:
```
#include "..."
#include <cmath>          // sqrtf
#include <cstring>        // (none used)
#include <cuda.h>         // CUdeviceptr, cuMemAlloc, cuMemFree, cuMemcpyDtoHAsync, cuStreamSynchronize

// --- shared helpers (extracted to avoid 12x duplication) ---
// 4.A: tensor dim helpers
// 4.B: shape constraints
// 4.C: arg packers (inlined in execute() per B.2 style)
// 4.D: scale precomputation
// 4.E: scratch alloc/resize helper

// --- 12 supports() functions (3 head_dim × 2 dtype × 2 kernel) ---

// --- 12 execute() functions (same shape matrix) ---
// prefill execute(): single launcher call
// decode execute(): kernel + D2H + sync + CPU reduce (4-step)

// --- register function ---
void ggml_triton_register_flash_attn_providers(ggml_triton_op_registry & registry);
```

### 4.2 Helper functions (4.A dim, 4.B constraints, 4.D scale, 4.E scratch)

```cpp
// 4.A: tensor dim helpers
static inline int64_t op_neq0(const ggml_tensor * op) { return op->src[0]->ne[0]; }  // = DK
static inline int64_t op_neq1(const ggml_tensor * op) { return op->src[0]->ne[1]; }  // = N
static inline int64_t op_neq2(const ggml_tensor * op) { return op->src[0]->ne[2]; }  // = n_heads
static inline int64_t op_neq3(const ggml_tensor * op) { return op->src[0]->ne[3]; }  // = batch
static inline int64_t op_nek1(const ggml_tensor * op) { return op->src[1]->ne[1]; }  // = S (KV seq)
static inline int64_t op_nek2(const ggml_tensor * op) { return op->src[1]->ne[2]; }  // = n_kv_heads
static inline int64_t op_nev2(const ggml_tensor * op) { return op->src[2]->ne[2]; }  // = n_kv_heads
static inline int64_t op_ne0(const ggml_tensor * op)  { return op->ne[0]; }         // = DV

// 4.B: shape constraints
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

// 4.D: scale precomputation (1.0 / sqrt(HEAD_DIM))
static inline float op_scale(const ggml_tensor * op) {
    return 1.0f / sqrtf((float)op_neq0(op));
}

// 4.E: scratch alloc/resize helper
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
    if (!ctx->decode_scratch_host) return -1;
    ctx->decode_scratch_size = needed;
    return 0;
}
```

### 4.3 12 `supports()` functions

Naming: `triton_flash_attn_{prefill|decode}_hd{HD}_{fp16|fp32}_supports` (12 total).

Pattern (each ~6 lines):
```cpp
static bool triton_flash_attn_prefill_hd64_fp16_supports(const ggml_tensor * op) {
    if (!op_is_flash_attn(op))     return false;
    if (!op_mask_is_null(op))      return false;
    if (!op_dtypes_match(op, GGML_TYPE_F16)) return false;
    if (op_neq0(op) != 64)         return false;  // THIS function is for HD=64
    if (!op_is_mha(op))            return false;
    if (!op_n_heads_supported(op)) return false;
    if (!op_is_contiguous(op) || !op_is_contiguous(op->src[0])
        || !op_is_contiguous(op->src[1]) || !op_is_contiguous(op->src[2])) return false;
    return true;
}
// 11 more: 2 dtype (fp16/fp32) × 3 head_dim (64/96/128) × 2 kernel (prefill/decode)
```

### 4.4 12 `execute()` functions

**Prefill execute** (single launcher call, ~25 lines each):

```cpp
static bool triton_flash_attn_prefill_hd96_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;

    const int32_t neq1         = (int32_t)op_neq1(op);
    const int32_t neq2         = (int32_t)op_neq2(op);
    const int32_t neq3         = (int32_t)op_neq3(op);
    const int32_t nek1         = (int32_t)op_nek1(op);
    const int32_t S            = (int32_t)op_neq3(op);  // batch
    const int32_t n_heads      = neq2;
    const int32_t rows         = neq2 * neq3;
    const int32_t num_q_blocks = (neq1 + 127) / 128;   // BLOCK_Q=128 constexpr
    const float   scale        = op_scale(op);

    int rc = triton_launch_flash_attn_prefill_hd96_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        neq1, neq2, neq3, nek1, S, n_heads, rows, num_q_blocks, scale);
    return rc == 0;
}
```

**Decode execute** (multi-step: kernel + D2H + sync + CPU reduce, ~50 lines):

```cpp
static bool triton_flash_attn_decode_hd96_fp16_execute(
    ggml_backend_triton_context * ctx, const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr
        || op->data == nullptr) return false;

    constexpr int32_t HD = 96;                        // hardcoded per-kernel-fn
    constexpr int32_t BLOCK_KV = 64;
    const int32_t neq2          = (int32_t)op_neq2(op);
    const int32_t neq3          = (int32_t)op_neq3(op);
    const int32_t nek1          = (int32_t)op_nek1(op);
    const int32_t rows          = neq2 * neq3;
    const int32_t num_kv_chunks = (nek1 + BLOCK_KV - 1) / BLOCK_KV;
    const int32_t S             = neq3;
    const int32_t n_heads       = neq2;
    const int32_t q_pos         = 0;                    // decode: current token only
    const float   scale         = op_scale(op);
    const int32_t scratch_per_chunk = 2 + HD;          // M, S, V_unnormalized
    const size_t  scratch_size  = (size_t)rows * num_kv_chunks * scratch_per_chunk * sizeof(float);

    // 1. Lazy alloc / resize scratch
    if (ensure_decode_scratch(ctx, scratch_size) != 0) return false;

    // 2. Launch decode kernel (writes partials to device scratch)
    int rc = triton_launch_flash_attn_decode_hd96_fp16_sm80(
        ctx->cu_stream,
        (CUdeviceptr)q->data, (CUdeviceptr)k->data, (CUdeviceptr)v->data, (CUdeviceptr)op->data,
        (CUdeviceptr)ctx->decode_scratch,
        /*neq1=*/1, neq2, neq3, nek1, S, n_heads, q_pos, num_kv_chunks, rows, scale);
    if (rc != 0) return false;

    // 3. D2H async copy + stream sync
    cuMemcpyDtoHAsync(ctx->decode_scratch_host, ctx->decode_scratch,
                      scratch_size, ctx->cu_stream);
    cuStreamSynchronize(ctx->cu_stream);

    // 4. CPU reduce pass per (head, batch)
    float * h = ctx->decode_scratch_host;
    float * dst = (float*)op->data;
    const int64_t dst_nb1 = op->nb[1];                 // stride between heads (DV)
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
        // Write normalized V to dst[head, batch] (dst is fp32 per ops.cpp:8883)
        float * dst_h = dst + h_idx * dst_nb1 / sizeof(float);
        for (int i = 0; i < HD; i++) dst_h[i] = v_final[i] / l_final;
    }
    return true;
}
```

### 4.5 Provider struct new fields

`ggml-triton-context.h` (or wherever per-context state lives) gains:

```cpp
struct ggml_backend_triton_context {
    // ... existing fields ...
    CUdeviceptr decode_scratch;       // device scratch buffer (persistent)
    float *    decode_scratch_host;   // host mirror for CPU reduce
    size_t     decode_scratch_size;   // current size in bytes; 0 = unallocated
};
```

**Lifecycle** (per Section 1.5 + Section 2.B):
- **Alloc**: lazy on first decode call via `cuMemAlloc` (device) + `malloc` (host). 4.E `ensure_decode_scratch` handles resize.
- **Resize**: if `scratch_size > ctx->decode_scratch_size`, free + realloc.
- **Free**: in context destructor (`ggml_backend_triton_free`).

### 4.6 Register function

```cpp
void ggml_triton_register_flash_attn_providers(ggml_triton_op_registry & registry) {
    registry.register_impl(GGML_OP_FLASH_ATTN_EXT, {
        "triton_flash_attn_prefill_hd64_fp16_sm80",
        GGML_TRITON_PROVIDER_TRITON,
        triton_flash_attn_prefill_hd64_fp16_supports,
        triton_flash_attn_prefill_hd64_fp16_execute,
        100,
    });
    // ... 11 more: 2 dtype (fp16/fp32) × 3 head_dim (64/96/128) × 2 kernel (prefill/decode)
}
```

Registry names: `triton_flash_attn_{prefill|decode}_hd{HD}_{fp16|fp32}_sm80` (matches launcher function name with `triton_launch_` → `triton_` prefix per B.2 convention).

### 4.7 Implementation guidance (where to start in plan)

1. Copy `ggml-triton-provider-rope.{h,cpp}` as starting template
2. Replace 6 supports/execute pairs with 12 (3 head_dim × 2 dtype × 2 kernel)
3. Add helper sections 4.A/4.B/4.D/4.E
4. Add scratch state fields to `ggml_backend_triton_context` struct
5. Implement decode's multi-step execute (kernel + D2H + sync + CPU reduce)
6. Add `ensure_decode_scratch` (4.E)
7. Wire `register()` call in both `ggml-triton-provider.cpp` (global) and `ggml-triton.cpp` (per-context) — same pattern as B.1/B.2
8. Add Assert 6 in `tests/test-triton-registry.cpp`


---

## 5. Test & verification

### 5.1 Assert 6 in `tests/test-triton-registry.cpp`

**Goal**: Verify all 12 triton AOT FlashAttn impls are registered in the global registry. Mirrors B.1's Assert 4 and B.2's Assert 5 pattern.

**Where to add**: After B.2's Assert 5 block. Insert Assert 6 BEFORE the existing `OK: registry test passed` printf.

**Distinct exit code**: Assert 6 returns `rc=7` (Assert 4 = rc 4, Assert 5 = rc 6, Assert 6 = rc 7 — distinct codes for bisect).

**Code to insert** (pattern matches Assert 4/5):
```cpp
    // Assert 6 (B.3): the Triton AOT FlashAttn provider (3 head_dim × 2 dtype
    // × 2 kernel = 12 impls) must be registered for GGML_OP_FLASH_ATTN_EXT.
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

**Expected output** (after build):
```
Assert 4 passed: triton AOT RMS_NORM fp16 + fp32 providers are registered
Assert 5 passed: 6 triton AOT RoPE impls (NORMAL+NEOX+MROPE x fp16/fp32) registered
Assert 6 passed: 12 triton AOT FlashAttn impls (prefill+decode × hd{64,96,128} × fp16/fp32) registered
OK: registry test passed
EXIT=0
```

### 5.2 `test-backend-ops` coverage

`test-backend-ops` is the standard ggml op correctness test. Compares each op's output against a CPU reference implementation.

**B.3 FlashAttn coverage targets**:
- Both kernels (prefill N>1, decode N=1)
- Both dtype (fp16, fp32)
- Testable head_dim: 96 (MiniMind-3) and 128 (Qwen3-72B); 64 is optional
- Causal semantics (matches CPU reference)
- Edge cases: S=0 (empty KV), S=1, S=2048 (long context)

**Test command** (per test-pyramid.md Level 3):
```bash
./build/bin/test-backend-ops test -o FLASH_ATTN_EXT --backends CPU,TRITON -ngl 999
```

**Stage 1 vs GPU-host verification**:
- **Stage 1 (CPU-only host)**: `test-backend-ops` exercises the dispatch + provider + registry path. Numerical correctness cannot be verified (placeholder CUBIN does no real compute). Exit: build succeeds, dispatch paths reachable, no crash.
- **GPU host (Stage 2+)**: `Δ ≤ 1e-3` fp16 vs CPU reference (per Phase A exit criteria, `docs/performance/unified-backend.md`).

### 5.3 Verification matrix (per B.3 exit criteria)

| Check | Stage 1 (CPU-only) | Stage 2+ (GPU host) |
|---|---|---|
| `compile_kernels.py` regenerates 24 .c/.h without error | ✅ Required | ✅ Required |
| Generated headers match expected (signatures, arg counts) | ✅ Required | ✅ Required |
| GELU/SiLU/RMSNorm/RoPE byte-compat preserved | ✅ Required | ✅ Required |
| CMake configure succeeds (ON + OFF) | ✅ Required | ✅ Required |
| `test-triton-registry` exits 0 (Assert 4 + 5 + 6) | ✅ Required | ✅ Required |
| `test-triton-registry` exits 7 with `GGML_TRITON_WITH_FLASH_ATTN=OFF` | ✅ Required | ✅ Required |
| `test-backend-ops FLASH_ATTN_EXT` builds and dispatches | ✅ Required | ✅ Required |
| Numerical: `Δ ≤ 1e-3` fp16 vs CPU reference | ⏸ Deferred (Phase 0 audit §0.4) | ✅ Required |
| Numerical: `Δ ≤ 1e-5` fp32 vs CPU reference | ⏸ Deferred | ✅ Required |
| MiniMind-3 smoke test: PPL unchanged with B.3 enabled | ⏸ Deferred | ✅ Required |
| MiniMind-3 `test-llama-archs` builds + runs forward + writes GGUF | ⏸ Deferred | ✅ Required |

### 5.4 Test plan (TDD structure, mirrors B.2)

**TDD red step (B.3 Task 1)**: Add Assert 6 → run `test-triton-registry` → expect exit 7 (Assert 6 fails, Assert 4 + 5 still pass). Commit failing test.

**Implementation steps (B.3 Tasks 7-9)**: Wire provider cpp + CMakeLists + global/per-context registration. After each step, rebuild and re-run. Assert 6 transitions FAIL → PASS as registration is wired.

**TDD green step (B.3 Task 11)**: All 12 launchers + provider cpp wired. `test-triton-registry` exits 0 with all 6 Asserts passing.

**Build verification (B.3 Task 14)**: cmake with `GGML_TRITON_WITH_FLASH_ATTN=OFF` succeeds. Library has 0 FlashAttn symbols. `test-triton-registry` exits 7 (Assert 6 fails, others pass).

**Cross-backend numerical check (B.3 Task 12, deferred to GPU host)**: `test-backend-ops FLASH_ATTN_EXT --backends CPU,TRITON` shows `Δ ≤ 1e-3` for fp16 and `Δ ≤ 1e-5` for fp32. This is the actual Stage 1 exit criterion per Phase A.

### 5.5 Test file modifications summary

| File | Change | Task |
|---|---|---|
| `tests/test-triton-registry.cpp` | Add Assert 6 (12 launcher names, return 7 on miss) | B.3 Task 1 |
| `docs/development/test-pyramid.md` | Add B.3 coverage marker (12 new ROPE-style entries + Assert 6) | B.3 Task 15 |
| `docs/development/test-pyramid.md` (B.2 line) | No change (already there) | n/a |
| `tests/test-backend-ops` (if needed) | No source change; FLASH_ATTN_EXT auto-included via op registry | n/a |

### 5.6 CI gate matrix (per test-pyramid.md Level 1 + 2)

| Level | Command | Pass criteria |
|---|---|---|
| 1 | `cmake -B build -DGGML_TRITON=ON && cmake --build build --target test-triton-registry` | Build succeeds |
| 1 | `./build/bin/test-triton-registry` | Exit 0, all 6 Asserts pass |
| 2 | `cmake -B build-off -DGGML_TRITON=ON -DGGML_TRITON_WITH_FLASH_ATTN=OFF && cmake --build build-off --target test-triton-registry` | Build succeeds |
| 2 | `./build-off/bin/test-triton-registry` | Exit 7 (Assert 6 fails, others pass) |
| 3 | `./build/bin/test-backend-ops test -o FLASH_ATTN_EXT --backends CPU,TRITON` | Build succeeds (numeric check deferred) |
