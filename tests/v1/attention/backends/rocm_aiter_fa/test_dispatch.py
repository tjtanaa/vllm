# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every AiterFlashAttentionImpl.forward branch: kernel taken and output.

Dispatch table (rocm_aiter_fa.py, AiterFlashAttentionImpl.forward):

| Row | Phase   | Condition                         | Kernels                        |
| --- | ------- | --------------------------------- | ------------------------------ |
| P1  | prefill | always                            | varlen (causal from metadata)  |
| E1  | extend  | no sliding window                 | varlen suffix; per context     |
|     |         |                                   | chunk gather + varlen; merge   |
| E2  | extend  | sliding window                    | gather window + varlen         |
| D1  | decode  | q=1, packed cache, head >= 64     | paged_attention_v1             |
| D2  | decode  | q=1, sliding window, no           | paged_attention_v1 over the    |
|     |         | sinks/alibi                       | window-trimmed block table     |
| D3  | decode  | sliding window with alibi         | unified_attention              |
| D4  | decode  | sinks                             | unified_attention              |
| D5  | decode  | q>1 (spec decode), causal         | unified_attention              |
| D6  | decode  | non-causal                        | mha_v3 flash_attn_with_kvcache |
| D7  | decode  | shuffle layout                    | paged_attention_common         |
| D8  | decode  | head < 64                         | unified_attention              |
| S1  | any     | KV-sharing / update-bypass layer  | gather new K/V from the cache  |
|     |         | with prefill or extend rows       | before prefill/extend          |
| X1  | -       | attn_metadata is None (profiling) | none, output zeroed            |
| X2  | -       | output_scale given                | NotImplementedError            |
| X3  | decode  | shuffle + window / sinks / q>1    | assertion                      |
"""

from collections import Counter
from dataclasses import dataclass, field

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import (
    GATHER,
    MERGE,
    MHA_V3,
    PAGED_COMMON,
    PAGED_V1,
    UNIFIED,
    VARLEN,
    FaCase,
    blocks_for,
)
from tests.v1.attention.model_shapes import (
    MUSE_SPARK_SLIDING,
    QWEN3_8_27B,
    QWEN3_235B,
    AttnShape,
)
from vllm.v1.attention.backends.rocm_aiter_fa import _CP_TOKENS_PER_ITER_ROCM

SMALL_WINDOW = 64


@dataclass
class Row:
    id: str
    query_lens: list[int]
    seq_lens: list[int]
    kernels: dict[str, int]
    shape: AttnShape = QWEN3_235B
    case: dict = field(default_factory=dict)
    shuffle: bool = False
    causal: bool = True
    decode_threshold: int = 1


ROWS = [
    Row("P1-prefill", [33, 64], [33, 64], {VARLEN: 1}),
    Row("P1-prefill-noncausal", [33, 64], [33, 64], {VARLEN: 1}, causal=False),
    Row("E1-extend-one-chunk", [40, 3], [200, 19], {VARLEN: 2, GATHER: 1, MERGE: 1}),
    Row(
        "E1-extend-two-chunks",
        [16],
        [16 + _CP_TOKENS_PER_ITER_ROCM + 100],
        {VARLEN: 3, GATHER: 2, MERGE: 2},
    ),
    Row(
        "E2-extend-swa",
        [40, 3],
        [400, 19],
        {VARLEN: 1, GATHER: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW},
    ),
    Row("D1-decode", [1, 1, 1], [17, 300, 1025], {PAGED_V1: 1}),
    Row(
        "D2-decode-swa",
        [1, 1, 1],
        [17, 300, 1025],
        {PAGED_V1: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW},
    ),
    Row(
        "D3-decode-swa-alibi",
        [1, 1],
        [17, 300],
        {UNIFIED: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW, "use_alibi": True},
    ),
    Row("D4-decode-sinks", [1, 1], [17, 300], {UNIFIED: 1}, case={"use_sinks": True}),
    Row("D5-decode-spec", [4, 4, 2], [40, 300, 77], {UNIFIED: 1}, decode_threshold=4),
    Row(
        "D6-decode-noncausal",
        [4, 4],
        [40, 300],
        {MHA_V3: 1},
        causal=False,
        decode_threshold=4,
    ),
    Row(
        "D7-decode-shuffle", [1, 1, 1], [17, 300, 1025], {PAGED_COMMON: 1}, shuffle=True
    ),
    Row(
        "D8-decode-head32",
        [1, 1],
        [17, 300],
        {UNIFIED: 1},
        shape=AttnShape("synthetic", 8, 2, 32),
    ),
    Row(
        "mixed",
        [1, 1, 40, 3, 64],
        [100, 300, 200, 19, 64],
        {PAGED_V1: 1, VARLEN: 3, GATHER: 1, MERGE: 1},
    ),
    Row(
        "mixed-swa",
        [1, 1, 40, 64],
        [100, 300, 200, 64],
        {PAGED_V1: 1, VARLEN: 2, GATHER: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW},
    ),
    Row(
        "mixed-shuffle",
        [1, 1, 40, 64],
        [100, 300, 200, 64],
        {PAGED_COMMON: 1, VARLEN: 3, GATHER: 1, MERGE: 1},
        shuffle=True,
    ),
    Row(
        "S1-shared-prefill",
        [33, 64],
        [33, 64],
        {GATHER: 1, VARLEN: 1},
        case={"kv_sharing": True},
    ),
    Row(
        "S1-shared-mixed",
        [1, 40, 64],
        [100, 200, 64],
        {GATHER: 2, VARLEN: 3, MERGE: 1, PAGED_V1: 1},
        case={"kv_sharing": True},
    ),
    Row(
        "S1-shared-extend-swa",
        [40, 3],
        [400, 19],
        {GATHER: 2, VARLEN: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW, "kv_sharing": True},
    ),
    Row(
        "S1-shared-mixed-swa",
        [1, 40, 64],
        [100, 200, 64],
        {GATHER: 2, VARLEN: 2, PAGED_V1: 1},
        shape=MUSE_SPARK_SLIDING,
        case={"sliding_window": SMALL_WINDOW, "kv_sharing": True},
    ),
    Row(
        "S1-shared-mixed-shuffle",
        [1, 40, 64],
        [100, 200, 64],
        {GATHER: 2, VARLEN: 3, MERGE: 1, PAGED_COMMON: 1},
        shuffle=True,
        case={"kv_sharing": True},
    ),
]


def _run_row(row: Row, spy, monkeypatch, kv_cache_dtype: str = "auto"):
    from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import set_shuffle

    set_shuffle(monkeypatch, row.shuffle)
    case = FaCase(row.shape, kv_cache_dtype=kv_cache_dtype, **row.case).setup(
        blocks_for(row.seq_lens, 16)
    )
    case.builder.reorder_batch_threshold = row.decode_threshold
    batch = case.batch(row.query_lens, row.seq_lens, causal=row.causal)
    query = case.query(batch.num_tokens)
    metadata = case.build(batch)
    cache_before = case.kv_cache.view(torch.uint8).clone()
    spy.reset()
    output = case.run(batch, query, metadata)
    if case.kv_sharing:
        # A sharing layer reads the target's cache and must never write it.
        torch.testing.assert_close(
            case.kv_cache.view(torch.uint8), cache_before, atol=0, rtol=0
        )
    return case, batch, query, output


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r.id)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_dispatch_row(row, kv_cache_dtype, spy, monkeypatch, request):
    if row.id == "D6-decode-noncausal" and kv_cache_dtype == "fp8":
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                raises=ValueError,
                reason="non-causal decode passes a bf16 query with an fp8 KV "
                "cache to mha_v3 flash_attn_with_kvcache, which rejects mixed "
                "FP8/non-FP8 inputs",
            )
        )
    case, batch, query, output = _run_row(row, spy, monkeypatch, kv_cache_dtype)
    assert Counter(spy.names()) == Counter(row.kernels), spy.names()
    case.check(batch, query, output)


def test_decode_swa_uses_trimmed_table(spy, monkeypatch):
    row = next(r for r in ROWS if r.id == "D2-decode-swa")
    case, batch, _, _ = _run_row(row, spy, monkeypatch)
    md = case.build(batch).decode_metadata
    (args,) = spy.args_of(PAGED_V1)
    block_table, seq_lens, max_seq_len = args[6], args[8], args[9]
    assert block_table.data_ptr() == md.swa_block_table.data_ptr()
    assert seq_lens.data_ptr() == md.swa_seq_lens.data_ptr()
    assert max_seq_len == md.swa_max_seq_len < max(row.seq_lens)


def test_extend_swa_gathers_only_the_window(spy, monkeypatch):
    row = next(r for r in ROWS if r.id == "E2-extend-swa")
    _, batch, _, _ = _run_row(row, spy, monkeypatch)
    (gather,) = spy.kwargs_of(GATHER)
    expected = sum(
        min(s, q + SMALL_WINDOW) for q, s in zip(row.query_lens, row.seq_lens)
    )
    assert gather["total_tokens"] == expected


def test_profiling_run_zeroes_output_without_kernels(spy):
    case = FaCase(QWEN3_235B).setup(16)
    query = case.query(8)
    output = torch.full_like(query, 7.0)
    spy.reset()
    case.layer.impl.forward(case.layer, query, None, None, case.kv_cache, None, output)
    assert spy.names() == []
    assert torch.count_nonzero(output) == 0


def test_fused_output_quant_is_rejected():
    case = FaCase(QWEN3_235B).setup(16)
    batch = case.batch([4], [4])
    query = case.query(4)
    with pytest.raises(NotImplementedError, match="fused output quantization"):
        case.layer.impl.forward(
            case.layer,
            query,
            query[:, :4],
            query[:, :4],
            case.kv_cache,
            case.build(batch),
            torch.empty_like(query),
            output_scale=torch.ones(1, device=query.device),
        )


@pytest.mark.parametrize(
    "case_kwargs, query_lens, threshold",
    [
        ({"sliding_window": SMALL_WINDOW}, [1, 1], 1),
        ({"use_sinks": True}, [1, 1], 1),
        ({}, [4, 4], 4),
    ],
    ids=["swa", "sinks", "spec"],
)
def test_shuffle_rejects_unsupported_decode(
    monkeypatch, case_kwargs, query_lens, threshold
):
    from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import set_shuffle

    set_shuffle(monkeypatch, True)
    seq_lens = [300, 500]
    case = FaCase(MUSE_SPARK_SLIDING, **case_kwargs).setup(blocks_for(seq_lens, 16))
    case.builder.reorder_batch_threshold = threshold
    batch = case.batch(query_lens, seq_lens)
    with pytest.raises(AssertionError, match="Shuffle KV cache layout"):
        case.run(batch, case.query(batch.num_tokens))


def test_padded_tokens_are_not_written():
    """CUDA-graph padding: rows past num_actual_tokens stay untouched."""
    case = FaCase(QWEN3_8_27B).setup(64)
    batch = case.batch([1, 1, 1], [40, 90, 500], num_padded_tokens=8)
    query = case.query(8)
    output = case.run(batch, query)
    case.check(batch, query, output)
    assert torch.isnan(output[batch.num_tokens :]).all()
