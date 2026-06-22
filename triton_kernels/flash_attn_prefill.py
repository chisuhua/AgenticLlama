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
