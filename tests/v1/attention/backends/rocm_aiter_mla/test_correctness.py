# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA output vs a float32 reference over per-rank model shapes.

Dimensions: per-rank MLA shape (head count picks the decode kernel; GLM-4.7
brings 192/256 nope/v dims) x phase (prefill / extend / decode / spec decode /
mixed) x tokens per request (1, 2048, 4096, 8192) x requests per batch (1, 16,
128) x block size (1, 16, 64) x KV dtype (auto, fp8). Pre-merge runs a
covering set where every value of each dimension meets every phase;
``-m slow_test`` runs the full product.

MLA splits a batch differently from dense backends: a request with context
and more new tokens than the decode threshold is a *prefill with chunked
context* (``forward_mha`` + context merge), and a prefill batch takes the fp8
asm prefill only when no prefill in it has context. ``prefill`` below has no
context, ``extend`` has, so both prefill kernels are covered.

Prefill and extend batches are capped at ``TOKEN_BUDGET`` new tokens, as the
scheduler's max_num_batched_tokens would cap them; requests beyond the budget
become decodes over the same context, which is what a running batch holds.
"""

import itertools

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_mla import mla_shapes as shapes
from tests.v1.attention.backends.rocm_aiter_mla.mla_harness import (
    MlaCase,
    blocks_for,
)
from tests.v1.attention.model_shapes import MLAShape

SHAPES = shapes.REGIME_SHAPES
PHASES = ["prefill", "extend", "decode", "spec", "mixed"]
TOKENS = [1, 2048, 4096, 8192]
BATCHES = [1, 16, 128]
BLOCK_SIZES = [1, 16, 64]
KV_DTYPES = ["auto", "fp8"]
TOKEN_BUDGET = 16384
EXTEND_CHUNK = 512
# MTP-3 verify block: the widest query the padded asm decode has a
# persistent kernel for.
SPEC_TOKENS = 3


def make_lens(
    phase: str, tokens: int, batch: int, spec_tokens: int = SPEC_TOKENS
) -> tuple[list[int], list[int]]:
    """(query_lens, seq_lens) in reorder order: decodes, then prefills."""
    if phase == "decode":
        return [1] * batch, [max(tokens, 2)] * batch
    if phase == "spec":
        q = spec_tokens + 1
        return [q] * batch, [max(tokens, q + 1)] * batch
    if phase == "prefill":
        n = max(1, min(batch, TOKEN_BUDGET // tokens))
        return [1] * (batch - n) + [tokens] * n, [tokens] * batch
    if phase == "extend":
        if tokens == 1:
            return [1] * batch, [2] * batch
        chunk = max(2, min(EXTEND_CHUNK, tokens - 1, TOKEN_BUDGET // batch))
        return [chunk] * batch, [tokens] * batch
    # mixed: half decodes, a quarter extends, the rest prefills (>= 1 each).
    n_dec = max(1, batch // 2) if batch > 1 else 0
    n_ext = max(1, batch // 4) if batch > 2 else 0
    n_pre = batch - n_dec - n_ext
    pre_len = max(2, min(tokens, TOKEN_BUDGET // (2 * max(n_pre, 1))))
    ext_q = max(2, min(EXTEND_CHUNK, tokens - 1))
    ctx = max(tokens, ext_q + 1)
    q = [1] * n_dec + [ext_q] * n_ext + [pre_len] * n_pre
    s = [ctx] * n_dec + [ctx] * n_ext + [pre_len] * n_pre
    return q, s


def run_case(
    shape: MLAShape,
    phase: str,
    tokens: int,
    batch: int,
    block_size: int,
    kv_cache_dtype: str,
    dtype: torch.dtype = torch.bfloat16,
    num_speculative_tokens: int | None = None,
):
    if num_speculative_tokens is None:
        num_speculative_tokens = SPEC_TOKENS if phase == "spec" else 0
    query_lens, seq_lens = make_lens(phase, tokens, batch, num_speculative_tokens)
    case = MlaCase(
        shape,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        max_model_len=max(16384, max(seq_lens)),
        max_num_batched_tokens=max(TOKEN_BUDGET, sum(query_lens)),
        num_speculative_tokens=num_speculative_tokens,
    ).setup(blocks_for(seq_lens, block_size))
    batch_ = case.batch(query_lens, seq_lens)
    query = case.query(batch_.num_tokens)
    metadata = case.build(batch_)
    if phase in ("decode", "spec"):
        assert metadata.num_prefills == 0
    case.check(batch_, query, case.run(batch_, query, metadata), metadata)


def _covering_set():
    """Each (tokens, batch) pair with each phase once; the shape, block size
    and KV dtype rotate so each of them appears across the set."""
    cases = []
    for i, (phase, tokens, batch) in enumerate(
        itertools.product(PHASES, TOKENS, BATCHES)
    ):
        shape = SHAPES[i % len(SHAPES)]
        block_size = BLOCK_SIZES[i % len(BLOCK_SIZES)]
        kv = KV_DTYPES[(i // len(BLOCK_SIZES)) % len(KV_DTYPES)]
        cases.append(
            pytest.param(
                shape,
                phase,
                tokens,
                batch,
                block_size,
                kv,
                id=f"{shape.id}-{phase}-t{tokens}-b{batch}-bs{block_size}-{kv}",
            )
        )
    return cases


@pytest.mark.parametrize(
    "shape, phase, tokens, batch, block_size, kv_cache_dtype", _covering_set()
)
def test_covering_set(shape, phase, tokens, batch, block_size, kv_cache_dtype):
    run_case(shape, phase, tokens, batch, block_size, kv_cache_dtype)


@pytest.mark.parametrize("shape", shapes.ALL_SHAPES, ids=lambda s: s.id)
@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
@pytest.mark.parametrize("phase", ["mixed", "spec", "prefill"])
def test_every_rank_shape(shape, kv_cache_dtype, phase):
    """Per-rank shapes of the deployed TP sizes of every dense MLA model."""
    run_case(shape, phase, 2048, 16, 16, kv_cache_dtype)


@pytest.mark.parametrize("phase", PHASES)
def test_float16_model(phase):
    """fp16 is declared for an fp8 cache only (the selector refuses fp16 with
    an unquantized cache: no fp16 decode kernel)."""
    run_case(shapes.DSV3_TP8, phase, 2048, 16, 16, "fp8", dtype=torch.float16)


@pytest.mark.parametrize("kv_cache_dtype", ["bfloat16", "fp8_e4m3"])
@pytest.mark.parametrize("phase", PHASES)
def test_other_declared_kv_dtypes(kv_cache_dtype, phase):
    run_case(shapes.DSV3_TP8, phase, 2048, 16, 16, kv_cache_dtype)


@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3, 5, 7])
@pytest.mark.parametrize(
    "shape", [shapes.DSV3_TP8, shapes.DSV3_TP16, shapes.KIMI_K3_TP8], ids=lambda s: s.id
)
@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
def test_spec_decode_query_lengths(shape, kv_cache_dtype, num_speculative_tokens):
    """Verify blocks of 2-8 tokens: qlen > 4 leaves the padded asm decode's
    persistent kernels (bf16 takes Gluon MTP or the non-persistent entry)."""
    run_case(
        shape,
        "spec",
        2048,
        16,
        16,
        kv_cache_dtype,
        num_speculative_tokens=num_speculative_tokens,
    )


@pytest.mark.parametrize(
    "seq_len",
    # Block (16/64), page-index expansion chunk (1024), kv granularity (16),
    # fp8 prefill tile (256) and kv granularity (128) edges.
    [15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025],
)
@pytest.mark.parametrize("phase", ["decode", "spec", "prefill", "extend"])
@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
def test_boundaries(phase, seq_len, kv_cache_dtype):
    batch = 3
    seq_lens = [seq_len - 1, seq_len, seq_len + 1]
    spec = 0
    if phase == "decode":
        query_lens = [1] * batch
    elif phase == "spec":
        spec = SPEC_TOKENS
        query_lens = [spec + 1] * batch
    elif phase == "prefill":
        query_lens = list(seq_lens)
    else:
        query_lens = [min(7, s - 1) for s in seq_lens]
    case = MlaCase(
        shapes.DSV3_TP8,
        kv_cache_dtype=kv_cache_dtype,
        num_speculative_tokens=spec,
    ).setup(blocks_for(seq_lens, 16))
    batch_ = case.batch(query_lens, seq_lens)
    query = case.query(batch_.num_tokens)
    metadata = case.build(batch_)
    case.check(batch_, query, case.run(batch_, query, metadata), metadata)


@pytest.mark.slow_test
@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.id)
def test_full_matrix(shape, phase, tokens, batch, block_size, kv_cache_dtype):
    run_case(shape, phase, tokens, batch, block_size, kv_cache_dtype)
