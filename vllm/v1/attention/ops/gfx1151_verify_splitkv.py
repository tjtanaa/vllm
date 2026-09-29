# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GQA-packed split-KV verification attention for gfx1151 Qwen3.8 full attention.

The prefix kernel used for speculative verification launches one workgroup per
*query* head, so the 6 query heads of a GQA group each stream the same K/V (6x
the unique bytes) and a batch-1 decode only exposes 24 workgroups of 4 warps on
40 CUs. Measured in serving conditions (16 GiB scattered page pool, L2 flushed by
the surrounding weight streaming) that is ~435 us per layer at a 1.4K context
for 5.7 MB of unique KV.

This module packs the rows of one KV head's whole GQA group into a single tile
(each K/V byte read once) and splits the context into PART-token partitions
handled by separate workgroups, combining their (m, l, acc) states in a fixed
partition order, so the result is deterministic and free of atomics. Measured
with benchmarks/kernels/bench_gfx1151_attention_splitkv.py:

| context | prefix tile 16/128/8 | this kernel (part 256, BLOCK_N 16, 8 warps) |
| --- | --- | --- |
| 1024 | 336-343 us | 117 us |
| 1400 | 435-442 us | 137 us |
| 2048 | 564-567 us | 157 us |
| 3072 | 805-809 us | 322 us |

Relative L2 against an FP32 reference is slightly *better* than the prefix tile
(2.08-2.15e-3 versus 2.23-2.27e-3) because the partition combine is FP32.

Cache layout is the shipped paged ROCm layout: key_cache
[pages, kv_heads, dim // x, page, x] and value_cache [pages, kv_heads, dim, page].
"""

from __future__ import annotations

import torch
import triton.language as tl

from vllm.triton_utils import triton

# Measured best configuration (see module docstring).
DEFAULT_PART = 256
DEFAULT_BLOCK_N = 16
DEFAULT_WARPS = 8
DEFAULT_STAGES = 1
DEFAULT_REDUCE_BLOCK_D = 128
DEFAULT_REDUCE_WARPS = 4
# Packed rows per workgroup (query tokens x heads per program). A [64, 256]
# FP32 accumulator is the largest tile that stays out of VGPR spills at 8 warps
# on gfx1151; measured 163 us/layer versus 425 us/layer for the shipped tile.
MAX_ROWS = 64


def pick_heads_per_prog(num_heads: int, num_kv_heads: int, max_query_len: int) -> int:
    """Largest divisor of the GQA ratio whose packed tile fits MAX_ROWS.

    Packing all GQA heads makes each K/V byte be read once; when the query block
    is wide (for example 16 speculative tokens) the tile is split across
    ``num_kv_heads * (gqa // heads_per_prog)`` workgroups instead, so K/V is
    still read far fewer times than one-workgroup-per-query-head.
    """
    gqa = num_heads // num_kv_heads
    best = 1
    for divisor in range(1, gqa + 1):
        if gqa % divisor == 0 and max_query_len * divisor <= MAX_ROWS:
            best = divisor
    return best


@triton.jit
def _splitkv_main(
    Q,
    K_cache,
    V_cache,
    B_Loc,
    B_Seqlen,
    B_Start,
    Partials,  # [seqs, groups, parts, ROWS, DIM] fp32
    MetaM,  # [seqs, groups, parts, ROWS] fp32
    MetaL,  # [seqs, groups, parts, ROWS] fp32
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_kbl,
    stride_kx,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_vbl,
    stride_bls,
    stride_bll,
    stride_pp,
    stride_pg,
    stride_pa,
    stride_pr,
    stride_mp,
    stride_mg,
    stride_ma,
    stride_mr,
    GQA: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    GROUPS: tl.constexpr,  # kv_heads * GQA / HEADS_PER_PROG
    DIM: tl.constexpr,
    QLEN: tl.constexpr,
    ROWS: tl.constexpr,  # QLEN * HEADS_PER_PROG
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PART: tl.constexpr,
    PHYS_BLOCK: tl.constexpr,
    X: tl.constexpr,
):
    group = tl.program_id(0)
    seq = tl.program_id(1)
    part = tl.program_id(2)

    kv_head = group // (GQA // HEADS_PER_PROG)
    head_base = kv_head * GQA + (group % (GQA // HEADS_PER_PROG)) * HEADS_PER_PROG

    seq_len = tl.load(B_Seqlen + seq)
    q_start = tl.load(B_Start + seq)
    # Ragged batches: this sequence may carry fewer query rows than the padded
    # QLEN bound that sizes the grid.
    q_len_seq = tl.load(B_Start + seq + 1) - q_start
    ctx_len = seq_len - q_len_seq
    part_start = part * PART
    # Partitions cover the whole sequence: the block's own K/V is already in the
    # cache, and the causal mask below restricts each query row.
    if part_start >= seq_len:
        return

    rows = tl.arange(0, BLOCK_M)
    token = rows // HEADS_PER_PROG
    row_valid = (rows < ROWS) & (token < q_len_seq)
    q_head = head_base + rows % HEADS_PER_PROG
    q_pos = ctx_len + token  # absolute position; key j visible iff j <= q_pos

    offs_d = tl.arange(0, DIM)
    q = tl.load(
        Q
        + (q_start + token)[:, None] * stride_qt
        + q_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=row_valid[:, None],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)

    part_end = tl.minimum(part_start + PART, seq_len)
    for start_n in range(part_start, part_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < part_end
        page = tl.load(
            B_Loc + seq * stride_bls + (offs_n // PHYS_BLOCK) * stride_bll,
            mask=n_valid,
            other=0,
        ).to(tl.int64)
        in_page = offs_n % PHYS_BLOCK
        k = tl.load(
            K_cache
            + page[None, :] * stride_kbs
            + kv_head * stride_kh
            + (offs_d[:, None] // X) * stride_kd
            + in_page[None, :] * stride_kbl
            + (offs_d[:, None] % X) * stride_kx,
            mask=n_valid[None, :],
            other=0.0,
        )  # [DIM, BLOCK_N]
        qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale
        visible = (offs_n[None, :] <= q_pos[:, None]) & n_valid[None, :]
        qk = tl.where(visible & row_valid[:, None], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp(m_i - m_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        p = tl.math.exp(qk - m_safe[:, None])
        p = tl.where(visible & row_valid[:, None], p, 0.0)
        l_i = alpha * l_i + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            V_cache
            + page[:, None] * stride_vbs
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd
            + in_page[:, None] * stride_vbl,
            mask=n_valid[:, None],
            other=0.0,
        )  # [BLOCK_N, DIM]
        acc = tl.dot(p.to(v.dtype), v, acc=acc, out_dtype=tl.float32)
        m_i = m_new

    pbase = (
        seq * stride_pp + group * stride_pg + part * stride_pa + rows * stride_pr
    )
    mbase = seq * stride_mp + group * stride_mg + part * stride_ma + rows * stride_mr
    tl.store(Partials + pbase[:, None] + offs_d[None, :], acc,
             mask=row_valid[:, None])
    tl.store(MetaM + mbase, m_i, mask=row_valid)
    tl.store(MetaL + mbase, l_i, mask=row_valid)


@triton.jit
def _splitkv_reduce(
    Partials,
    MetaM,
    MetaL,
    Out,
    B_Seqlen,
    B_Start,
    stride_pp,
    stride_pg,
    stride_pa,
    stride_pr,
    stride_mp,
    stride_mg,
    stride_ma,
    stride_mr,
    stride_ot,
    stride_oh,
    stride_od,
    GQA: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    QLEN: tl.constexpr,
    ROWS: tl.constexpr,
    PART: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seq = tl.program_id(0)
    group = tl.program_id(1)
    d_blk = tl.program_id(2)

    kv_head = group // (GQA // HEADS_PER_PROG)
    head_base = kv_head * GQA + (group % (GQA // HEADS_PER_PROG)) * HEADS_PER_PROG

    seq_len = tl.load(B_Seqlen + seq)
    q_start = tl.load(B_Start + seq)
    q_len_seq = tl.load(B_Start + seq + 1) - q_start
    n_parts = tl.cdiv(seq_len, PART)

    rows = tl.arange(0, BLOCK_R)
    token = rows // HEADS_PER_PROG
    row_valid = (rows < ROWS) & (token < q_len_seq)
    q_head = head_base + rows % HEADS_PER_PROG
    offs_d = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)

    m_run = tl.full([BLOCK_R], float("-inf"), dtype=tl.float32)
    l_run = tl.zeros([BLOCK_R], dtype=tl.float32)
    acc = tl.zeros([BLOCK_R, BLOCK_D], dtype=tl.float32)

    pbase = seq * stride_pp + group * stride_pg + rows * stride_pr
    mbase = seq * stride_mp + group * stride_mg + rows * stride_mr
    for part in range(0, n_parts):
        m_p = tl.load(MetaM + mbase + part * stride_ma, mask=row_valid,
                      other=float("-inf"))
        l_p = tl.load(MetaL + mbase + part * stride_ma, mask=row_valid, other=0.0)
        a_p = tl.load(
            Partials + pbase[:, None] + part * stride_pa + offs_d[None, :],
            mask=row_valid[:, None],
            other=0.0,
        )
        m_new = tl.maximum(m_run, m_p)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        scale_old = tl.math.exp(m_run - m_safe)
        scale_old = tl.where(m_run == float("-inf"), 0.0, scale_old)
        scale_new = tl.math.exp(m_p - m_safe)
        scale_new = tl.where(m_p == float("-inf"), 0.0, scale_new)
        acc = acc * scale_old[:, None] + a_p * scale_new[:, None]
        l_run = l_run * scale_old + l_p * scale_new
        m_run = m_new

    l_safe = tl.where(l_run == 0.0, 1.0, l_run)
    res = acc / l_safe[:, None]
    tl.store(
        Out
        + (q_start + token)[:, None] * stride_ot
        + q_head[:, None] * stride_oh
        + offs_d[None, :] * stride_od,
        res.to(Out.dtype.element_ty),
        mask=row_valid[:, None],
    )




def splitkv_verify_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    output: torch.Tensor,
    *,
    max_query_len: int,
    max_seq_len: int,
    sm_scale: float,
    part: int = DEFAULT_PART,
    block_n: int = DEFAULT_BLOCK_N,
    num_warps: int = DEFAULT_WARPS,
    num_stages: int = DEFAULT_STAGES,
) -> torch.Tensor:
    """Run one cached-KV verification attention call. Returns the output."""
    num_heads, dim = query.shape[1], query.shape[2]
    num_seqs = seq_lens.numel()
    kv_heads = key_cache.shape[1]
    x = key_cache.shape[4]
    phys_block = key_cache.shape[3]
    gqa = num_heads // kv_heads
    heads_per_prog = pick_heads_per_prog(num_heads, kv_heads, max_query_len)
    groups = kv_heads * (gqa // heads_per_prog)
    rows = max_query_len * heads_per_prog
    block_m = max(16, triton.next_power_of_2(rows))
    max_parts = max(1, (max_seq_len + part - 1) // part)

    partials = torch.empty(
        (num_seqs, groups, max_parts, rows, dim), dtype=torch.float32,
        device=query.device,
    )
    meta = torch.empty(
        (2, num_seqs, groups, max_parts, rows), dtype=torch.float32,
        device=query.device,
    )
    meta_m, meta_l = meta[0], meta[1]

    _splitkv_main[(groups, num_seqs, max_parts)](
        query, key_cache, value_cache, block_table, seq_lens, query_start_loc,
        partials, meta_m, meta_l, sm_scale,
        query.stride(0), query.stride(1), query.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        key_cache.stride(3), key_cache.stride(4),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        value_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        partials.stride(0), partials.stride(1), partials.stride(2),
        partials.stride(3),
        meta.stride(1), meta.stride(2), meta.stride(3), meta.stride(4),
        gqa, heads_per_prog, groups, dim, max_query_len, rows,
        block_m, block_n, part, phys_block, x,
        num_warps=num_warps, num_stages=num_stages,
    )
    reduce_block_d = min(DEFAULT_REDUCE_BLOCK_D, dim)
    _splitkv_reduce[(
        num_seqs, groups, triton.cdiv(dim, reduce_block_d)
    )](
        partials, meta_m, meta_l, output, seq_lens, query_start_loc,
        partials.stride(0), partials.stride(1), partials.stride(2),
        partials.stride(3),
        meta.stride(1), meta.stride(2), meta.stride(3), meta.stride(4),
        output.stride(0), output.stride(1), output.stride(2),
        gqa, heads_per_prog, groups, dim, max_query_len, rows, part,
        max(16, triton.next_power_of_2(rows)), reduce_block_d,
        num_warps=DEFAULT_REDUCE_WARPS,
    )
    return output


def splitkv_supports(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_query_len: int,
    max_seq_len: int,
    causal: bool,
    window: int,
    sinks,
    dtype: torch.dtype,
) -> bool:
    """Contract for the packed split-KV verification path."""
    if head_size != 256 or num_kv_heads <= 0 or num_heads % num_kv_heads:
        return False
    # pick_heads_per_prog always yields a tile within MAX_ROWS, so any query
    # width up to 32 is supported; wider blocks keep the shipped prefix path.
    return (
        max_query_len <= 32
        and causal
        and window <= 0
        and sinks is None
        and dtype == torch.bfloat16
        and max_seq_len >= max_query_len
    )
