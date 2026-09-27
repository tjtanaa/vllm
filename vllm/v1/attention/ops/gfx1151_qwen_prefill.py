# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured gfx1151 prefix tiles, using the original attention arithmetic."""

from vllm.triton_utils import triton
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
)
from vllm.v1.attention.ops.prefix_prefill import _fwd_kernel


def launch_configured_prefill(
    q,
    k,
    v,
    out,
    key_cache,
    value_cache,
    table,
    starts,
    lengths,
    scale,
    query_len,
    causal,
    window,
    config,
    *,
    v_scale=None,
    sm_scale=None,
):
    """Internal BF16 prefix launch; callers validate capability and layouts.

    Single-token requests are deliberately skipped. The serving wrapper below
    retains the original decode operation for mixed batches and graph replay.
    """
    block_m, block_n, warps = config
    cached = k is None
    if cached:
        k = v = q  # Never dereferenced by the cached-K/V specialization.
    dim, heads = q.shape[2], q.shape[1]
    grid = (lengths.numel(), heads, triton.cdiv(query_len, block_m))
    return _fwd_kernel[grid](
        q,
        k,
        v,
        key_cache,
        value_cache,
        None,
        table,
        1.0 / (dim**0.5) if sm_scale is None else sm_scale,
        scale,
        scale if v_scale is None else v_scale,
        1.0,
        starts,
        lengths,
        key_cache.shape[4],
        out,
        table.stride(0),
        table.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        stride_k_cache_bs=key_cache.stride(0),
        stride_k_cache_h=key_cache.stride(1),
        stride_k_cache_d=key_cache.stride(2),
        stride_k_cache_bl=key_cache.stride(3),
        stride_k_cache_x=key_cache.stride(4),
        stride_v_cache_bs=value_cache.stride(0),
        stride_v_cache_h=value_cache.stride(1),
        stride_v_cache_d=value_cache.stride(2),
        stride_v_cache_bl=value_cache.stride(3),
        BLOCK_SIZE=32,
        PHYSICAL_BLOCK_SIZE=value_cache.shape[3],
        num_queries_per_kv=heads // key_cache.shape[1],
        IN_PRECISION=None,
        BLOCK_DMODEL=dim,
        BLOCK_DMODEL_PADDED=triton.next_power_of_2(dim),
        SLIDING_WINDOW=window,
        SKIP_DECODE=True,
        USE_FP8=False,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_unroll_cache=4,
        num_unroll_request=1,
        num_warps=warps,
        num_stages=1,
        USE_SINKS=False,
        CAUSAL=causal,
        KV_FROM_CACHE=cached,
    )


def gfx1151_qwen_prefill(
    query,
    key,
    value,
    output,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    sm_scale,
    config,
    *,
    causal=True,
    window=0,
):
    """BF16 prefix and mixed decode after the caller's capability checks."""
    launch_configured_prefill(
        query,
        key,
        value,
        output,
        key_cache,
        value_cache,
        block_table,
        query_start_loc,
        seq_lens,
        k_scale,
        max_query_len,
        causal,
        window,
        config,
        v_scale=v_scale,
        sm_scale=sm_scale,
    )
    # Suppress only the already-computed prefix launch. The original decode
    # operation filters by the real device query lengths, so it remains valid
    # when a captured batch replays with a mixture of prefill and decode rows.
    chunked_prefill_paged_decode(
        query=query,
        key=key,
        value=value,
        output=output,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        max_query_len=1,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=sm_scale,
        causal=causal,
        sliding_window=window,
    )
