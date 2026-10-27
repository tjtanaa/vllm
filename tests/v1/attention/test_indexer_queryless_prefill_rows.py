# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA indexer prefill chunks with rows that have no query tokens.

A request whose KV an async connector is still loading is scheduled with zero
new tokens but a non-zero seq_len (vllm-project/vllm#51486); FULL CUDA graph
padding adds rows with neither. Neither reads K, so the indexer must not
gather it, and no prefill chunk may hold only such rows (its logits would be
computed for an empty query slice).
"""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
)

if not (current_platform.is_cuda() or current_platform.is_rocm()):
    pytest.skip("DSA indexer metadata needs a GPU", allow_module_level=True)

BLOCK_SIZE = 64
MAX_MODEL_LEN = 16384


def _build(query_lens: list[int], seq_lens: list[int]):
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        max_model_len=MAX_MODEL_LEN,
        block_size=BLOCK_SIZE,
        hf_config_override={"index_topk": 256},
    )
    device = torch.device(current_platform.device_type)
    with set_current_vllm_config(vllm_config):
        cache = DeepseekV32IndexerCache(
            head_dim=128 + 4,
            dtype=torch.uint8,
            prefix="model.layers.0.self_attn.indexer.k_cache",
            cache_config=vllm_config.cache_config,
        )
        spec = DeepseekV32IndexerBackend.customize_spec(
            cache.get_kv_cache_spec(vllm_config)
        )
        builder = DeepseekV32IndexerMetadataBuilder(
            spec,
            [cache.prefix],
            vllm_config,
            device,
            block_table_width=MAX_MODEL_LEN // BLOCK_SIZE,
        )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        BLOCK_SIZE,
        device,
        arange_block_indices=True,
    )
    return builder.build(0, common)


def _chunk_fields(md):
    chunks = md.prefill.chunks
    return {
        "ks": torch.cat([c.cu_seqlen_ks for c in chunks]).tolist(),
        "ke": torch.cat([c.cu_seqlen_ke for c in chunks]).tolist(),
        "gathered": sum(c.total_seq_lens for c in chunks if not c.skip_kv_gather),
        "spans": [(c.token_start, c.token_end) for c in chunks],
    }


@pytest.mark.parametrize("tail_seq", [0, 3000], ids=["pad-seq0", "kv-load"])
@pytest.mark.parametrize(
    "query_lens, seq_lens, logits_mb",
    [
        ([100, 7], [100, 7], None),
        ([1, 64, 200], [900, 500, 1200], None),
        ([700, 300, 1000], [700, 300, 1000], 1),
    ],
    ids=["prefill", "mixed-extend", "sub-chunks"],
)
def test_queryless_prefill_rows_change_no_chunk(
    default_vllm_config, monkeypatch, query_lens, seq_lens, logits_mb, tail_seq
):
    """Trailing query-less rows leave the real rows' bounds and K gather
    unchanged, and every chunk keeps query tokens."""
    if logits_mb is not None:
        monkeypatch.setattr(envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", logits_mb)
    want = _chunk_fields(_build(query_lens, seq_lens))
    got = _chunk_fields(
        _build(query_lens + [0, 0], seq_lens + [tail_seq, tail_seq // 2])
    )
    assert got == want


def test_many_queryless_rows_form_no_chunk(default_vllm_config, monkeypatch):
    """Many loading requests whose contexts exceed the K-gather workspace on
    their own must not become chunks without query tokens."""
    query_lens, seq_lens = [128], [128]
    md = _build(query_lens + [0] * 8, seq_lens + [MAX_MODEL_LEN] * 8)
    for start, end in _chunk_fields(md)["spans"]:
        assert end > start, (start, end)
    assert _chunk_fields(md) == _chunk_fields(_build(query_lens, seq_lens))
