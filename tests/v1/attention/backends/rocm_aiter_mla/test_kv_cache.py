# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA latent cache writes and the reads that depend on them.

MLA caches one ``[kv_lora_rank | rope]`` row per token. The backend writes it
through ``do_kv_cache_update`` (``concat_and_cache_mla``); every read path
(asm/Gluon decode, the prefill context gather) must see the same bytes and
the same scale.
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_mla import mla_shapes as shapes
from tests.v1.attention.backends.rocm_aiter_mla.mla_harness import (
    DEVICE,
    FP8_K_SCALE,
    WRITE,
    MlaCase,
    blocks_for,
)
from vllm.platforms import current_platform

SHAPE = shapes.DSV3_TP8


def _decode_cache(case: MlaCase) -> torch.Tensor:
    """The cache rows as the dtype ``kv_cache_dtype`` stores, in float32."""
    rows = case.kv_cache.reshape(case.num_slots, -1)
    if not case.quantized:
        return rows.float()
    fp8 = current_platform.fp8_dtype()
    return rows.view(torch.uint8).view(fp8).float() * FP8_K_SCALE


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16", "fp8", "fp8_e4m3"])
@pytest.mark.parametrize("block_size", [1, 16, 64])
def test_write_round_trips_every_slot(kv_cache_dtype, block_size, spy):
    """The cache holds exactly the (fp8-representable) latent per slot."""
    case = MlaCase(SHAPE, kv_cache_dtype=kv_cache_dtype, block_size=block_size)
    case.setup(blocks_for([256], block_size))
    assert WRITE in spy.names()
    stored = _decode_cache(case)
    expected = case.latent_all.float()
    torch.testing.assert_close(stored, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_padded_slots_are_not_written(kv_cache_dtype):
    case = MlaCase(SHAPE, kv_cache_dtype=kv_cache_dtype)
    case.setup(blocks_for([64], 16))
    before = case.kv_cache.clone()
    n = 8
    width = SHAPE.kv_lora_rank + SHAPE.qk_rope_head_dim
    kv_c, k_pe = case.split_latent(
        torch.randn(n, width, dtype=case.dtype, device=DEVICE) * 100
    )
    slots = torch.full((n,), -1, dtype=torch.int64, device=DEVICE)
    case.layer.update_kv_cache(
        kv_c, k_pe, case.kv_cache, slots, None, kv_cache_dtype, case.layer._k_scale
    )
    assert torch.equal(case.kv_cache, before)


@pytest.mark.parametrize("phase", ["decode", "extend"])
def test_reads_apply_the_cache_scale(phase):
    """Negative control for the fp8 scale plumbing: reading an fp8 cache with
    the wrong k_scale must not pass the reference check."""
    case = MlaCase(SHAPE, kv_cache_dtype="fp8")
    seq_lens = [300, 500]
    case.setup(blocks_for(seq_lens, 16))
    query_lens = [1, 1] if phase == "decode" else [17, 33]
    batch = case.batch(query_lens, seq_lens)
    query = case.query(batch.num_tokens)
    metadata = case.build(batch)
    case.check(batch, query, case.run(batch, query, metadata), metadata)

    # Rewrite the cache at a different scale but keep reading with the old one.
    case.layer._k_scale.fill_(FP8_K_SCALE * 4)
    case._fill_cache()
    case.layer._k_scale.fill_(FP8_K_SCALE)
    out = case.run(batch, query, case.build(batch))
    with pytest.raises(AssertionError):
        case.check(batch, query, out, metadata)
