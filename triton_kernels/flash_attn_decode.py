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