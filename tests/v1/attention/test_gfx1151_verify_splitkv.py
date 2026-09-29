# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the gfx1151 GQA-packed split-KV verification attention.

Covers the contract (`splitkv_supports`), numerical agreement with an FP32
reference across batch/context/query-length combinations, ragged batches where
sequences carry different query lengths (the mixed single-token graph rows the
verification path has to tolerate), determinism of the partition combine, and
agreement with the shipped prefix tile.
"""

import pytest
import torch

from benchmarks.kernels.benchmark_gfx1151_qwen_attention import (
    attention_reference,
    make_attention_inputs,
)
from vllm.v1.attention.ops.gfx1151_verify_splitkv import (
    splitkv_supports,
    splitkv_verify_attention,
)

requires_gfx1151 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip), reason="requires ROCm GPU"
)


def _supported(qlen=8, ctx=1024, causal=True, window=0, sinks=None,
               dtype=torch.bfloat16, heads=24, kv_heads=4, dim=256):
    return splitkv_supports(
        heads, kv_heads, dim, qlen, ctx, causal, window, sinks, dtype
    )


def test_contract_accepts_target_verification_shape():
    assert _supported()
    assert _supported(qlen=4)
    assert _supported(qlen=8, ctx=4096)
    # Wide speculative blocks pack fewer GQA heads per program but stay legal.
    assert _supported(qlen=16)
    assert _supported(qlen=32)


def test_heads_per_prog_packing():
    from vllm.v1.attention.ops.gfx1151_verify_splitkv import pick_heads_per_prog

    assert pick_heads_per_prog(24, 4, 8) == 6   # all GQA heads: K/V read once
    assert pick_heads_per_prog(24, 4, 16) == 3  # 48 packed rows
    assert pick_heads_per_prog(24, 4, 32) == 2  # 64 packed rows
    assert pick_heads_per_prog(24, 4, 4) == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"qlen": 64}, id="qlen-above-32"),
        pytest.param({"causal": False}, id="non-causal"),
        pytest.param({"window": 2047}, id="sliding-window"),
        pytest.param({"sinks": object()}, id="attention-sinks"),
        pytest.param({"dtype": torch.float16}, id="fp16"),
        pytest.param({"dim": 128}, id="draft-head-dim"),
        pytest.param({"ctx": 4}, id="context-shorter-than-query"),
    ],
)
def test_contract_rejects(kwargs):
    assert not _supported(**kwargs)


@requires_gfx1151
@pytest.mark.parametrize(
    "batch,ctx,qlen",
    [
        (1, 640, 4),
        (1, 1024, 8),
        (1, 1400, 8),
        (1, 2048, 8),
        (2, 1152, 8),
        (3, 832, 8),
        (4, 1024, 8),
        (1, 1400, 16),
        (2, 1024, 16),
    ],
)
def test_matches_fp32_reference(batch, ctx, qlen):
    torch.manual_seed(0)
    q, k, v, kc, vc, table, lengths, starts, _, _ = make_attention_inputs(
        batch, ctx, 832, query_len=qlen
    )
    out = torch.empty_like(q)
    splitkv_verify_attention(
        q, kc, vc, table, lengths, starts, out,
        max_query_len=qlen, max_seq_len=ctx, sm_scale=256**-0.5,
    )
    torch.cuda.synchronize()
    ref = attention_reference(q, k, v, True, 0)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, rel
    assert torch.isfinite(out.float()).all()


@requires_gfx1151
def test_ragged_query_lengths():
    """Sequences with different query lengths in one batch, as graph rows mix."""
    torch.manual_seed(3)
    ctx = 1024
    q, k, v, kc, vc, table, lengths, starts, _, _ = make_attention_inputs(
        2, ctx, 832, query_len=8
    )
    # seq 0 keeps only its last 3 query rows, seq 1 keeps all 8.
    ragged_q = torch.cat([q[5:8], q[8:16]], dim=0).contiguous()
    ragged_starts = torch.tensor([0, 3, 11], dtype=torch.int32, device=q.device)
    out = torch.empty_like(ragged_q)
    splitkv_verify_attention(
        ragged_q, kc, vc, table, lengths, ragged_starts, out,
        max_query_len=8, max_seq_len=ctx, sm_scale=256**-0.5,
    )
    torch.cuda.synchronize()
    ref0 = attention_reference(q[5:8], k[0:1], v[0:1], True, 0)
    ref1 = attention_reference(q[8:16], k[1:2], v[1:2], True, 0)
    ref = torch.cat([ref0, ref1], dim=0)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, rel


@requires_gfx1151
def test_deterministic_across_repeats():
    torch.manual_seed(1)
    q, k, v, kc, vc, table, lengths, starts, _, _ = make_attention_inputs(
        1, 1400, 832, query_len=8
    )
    outs = []
    for _ in range(3):
        out = torch.empty_like(q)
        splitkv_verify_attention(
            q, kc, vc, table, lengths, starts, out,
            max_query_len=8, max_seq_len=1400, sm_scale=256**-0.5,
        )
        torch.cuda.synchronize()
        outs.append(out.clone())
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])


@requires_gfx1151
def test_agrees_with_shipped_prefix_tile():
    """Both paths approximate the same FP32 reference within bf16 tolerance."""
    from vllm.v1.attention.ops.gfx1151_qwen_prefill import launch_configured_prefill

    torch.manual_seed(5)
    ctx, qlen = 1400, 8
    q, k, v, kc, vc, table, lengths, starts, _, _ = make_attention_inputs(
        1, ctx, 832, query_len=qlen
    )
    out_split = torch.empty_like(q)
    splitkv_verify_attention(
        q, kc, vc, table, lengths, starts, out_split,
        max_query_len=qlen, max_seq_len=ctx, sm_scale=256**-0.5,
    )
    out_prefix = torch.empty_like(q)
    launch_configured_prefill(
        q, None, None, out_prefix, kc, vc, table, starts, lengths, 1.0, qlen,
        True, 0, (16, 128, 8), v_scale=1.0, sm_scale=256**-0.5,
    )
    torch.cuda.synchronize()
    rel = ((out_split.float() - out_prefix.float()).norm()
           / out_prefix.float().norm()).item()
    assert rel < 1e-2, rel
    assert torch.equal(out_split.argmax(-1), out_prefix.argmax(-1))
