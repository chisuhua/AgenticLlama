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