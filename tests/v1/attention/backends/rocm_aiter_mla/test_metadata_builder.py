# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AiterMLAMetadataBuilder: every field against an independent value.

Covers the decode/prefill split (MLA has no separate extend phase: a request
with context and more new tokens than the reorder threshold is a prefill with
chunked context), the flat per-token page view the asm and Gluon decodes
read, chunked-context planning, padded batches, the capture build, persistent
buffers and the no-GPU-sync guarantee of decode-only builds.
"""

from contextlib import contextmanager

import pytest
import torch

from tests.v1.attention.backends.harness import make_paged_batch
from tests.v1.attention.backends.rocm_aiter_mla import mla_shapes as shapes
from tests.v1.attention.backends.rocm_aiter_mla.mla_harness import (
    DEVICE,
    MlaCase,
    blocks_for,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata


@contextmanager
def no_gpu_sync():
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def _case(shape=shapes.DSV3_TP8, num_blocks=2048, **kwargs) -> MlaCase:
    return MlaCase(shape, **kwargs).setup(num_blocks)


def _flat_indices(case: MlaCase, batch, num_decodes: int) -> torch.Tensor:
    """Per-token cache slots of the decode requests, in request order."""
    return torch.cat([batch.slots[i] for i in range(num_decodes)]).to(torch.int32)


@pytest.mark.parametrize(
    "query_lens, seq_lens, spec, expected",
    [
        ([1, 1, 1], [17, 300, 4096], 0, (3, 0)),
        ([64, 7], [64, 7], 0, (0, 2)),
        ([40, 3], [200, 19], 0, (0, 2)),
        ([1, 1, 40, 64], [100, 300, 200, 64], 0, (2, 2)),
        # With MTP-3 the threshold is 4, but the decode kernels need a uniform
        # block: a 3-token row behind 4-token rows is a prefill with context.
        ([4, 4, 3, 40], [100, 300, 200, 64], 3, (2, 2)),
        ([3, 3, 40], [100, 300, 64], 3, (2, 1)),
        # A 5-token request is past the threshold, so it is a prefill.
        ([4, 4, 5], [100, 300, 200], 3, (2, 1)),
    ],
    ids=["decode", "prefill", "extend", "mixed", "spec", "spec-short", "spec+prefill"],
)
def test_phase_split(query_lens, seq_lens, spec, expected):
    case = _case(num_speculative_tokens=spec)
    assert case.builder.reorder_batch_threshold == spec + 1
    md = case.build(case.batch(query_lens, seq_lens))
    n_dec, n_pre = expected
    assert (md.num_decodes, md.num_prefills) == (n_dec, n_pre)
    assert md.num_decode_tokens == sum(query_lens[:n_dec])
    assert md.num_actual_tokens == sum(query_lens)
    assert (md.decode is None) == (n_dec == 0)
    assert (md.prefill is None) == (n_pre == 0)


@pytest.mark.parametrize("block_size", [1, 16, 64])
@pytest.mark.parametrize("spec", [0, 3])
def test_decode_fields_match_the_flat_page_view(block_size, spec):
    q = spec + 1
    seq_lens = [17, 300, 1025, 4096]
    case = _case(
        block_size=block_size,
        num_blocks=blocks_for(seq_lens, block_size),
        num_speculative_tokens=spec,
    )
    batch = case.batch([q] * 4, seq_lens)
    md = case.build(batch)
    d = md.decode
    assert md.num_decodes == 4
    seq = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    indptr = torch.zeros(5, dtype=torch.int32, device=DEVICE)
    indptr[1:] = seq.cumsum(0)
    assert torch.equal(d.paged_kv_indptr, indptr)
    total = int(indptr[-1])
    assert torch.equal(d.paged_kv_indices[:total], _flat_indices(case, batch, 4))
    assert torch.equal(d.paged_kv_last_page_len, torch.ones_like(seq))
    qo = torch.arange(0, 5 * q, q, dtype=torch.int32, device=DEVICE)
    assert torch.equal(d.qo_indptr, qo)
    assert d.max_qo_len == q
    assert torch.equal(d.seq_lens, seq)


def test_prefill_fields_are_rebased_past_the_decodes():
    query_lens, seq_lens = [1, 1, 40, 64], [100, 300, 200, 64]
    case = _case()
    batch = case.batch(query_lens, seq_lens)
    md = case.build(batch)
    p = md.prefill
    assert torch.equal(
        p.query_start_loc, torch.tensor([0, 40, 104], device=DEVICE, dtype=torch.int32)
    )
    assert torch.equal(p.block_table, batch.common.block_table_tensor[2:])
    assert p.max_query_len == 64
    assert torch.equal(p.query_lens_cpu, torch.tensor([40, 64], dtype=torch.int32))


@pytest.mark.parametrize(
    "query_lens, seq_lens, max_model_len",
    [
        ([64, 7], [64, 7], 16384),  # no context
        ([40, 3], [200, 19], 16384),  # context << one chunk
        ([32, 32, 32], [4032, 4032, 4032], 1024),  # several chunks
        ([8, 100, 3], [8000, 108, 20], 1024),  # uneven contexts
    ],
    ids=["none", "small", "chunks", "uneven"],
)
def test_context_chunking(query_lens, seq_lens, max_model_len):
    case = _case(
        max_model_len=max_model_len,
        max_num_seqs=16,
        num_blocks=blocks_for(seq_lens, 16),
    )
    md = case.build(case.batch(query_lens, seq_lens))
    chunked = md.prefill.chunked_context
    contexts = [s - q for q, s in zip(query_lens, seq_lens)]
    if not any(contexts):
        assert chunked is None
        return
    workspace = case.builder.chunked_prefill_workspace_size
    covered = [0] * len(contexts)
    for chunk in chunked.chunks:
        assert chunk.num_context_tokens <= workspace
        for i, n in enumerate(chunk.cu_seq_lens.diff().tolist()):
            covered[chunk.request_slice.start + i] += n
    assert covered == contexts
    expected_chunks = -(-sum(contexts) // workspace)
    assert len(chunked.chunks) >= expected_chunks


def test_padded_tokens_are_not_counted():
    """CUDA-graph padding: trailing slot_mapping -1 rows past the real
    tokens stay out of every phase."""
    case = _case()
    batch = case.batch([1, 1, 1], [100, 200, 300], num_padded_tokens=8)
    md = case.build(batch)
    assert md.num_actual_tokens == 3
    assert md.num_decode_tokens == 3


def _padded_uniform_batch(case: MlaCase, q: int, real: int, padded: int):
    """``real`` live verify requests then ``padded - real`` dummy requests
    with no tokens, as a full graph replays at its captured request count."""
    batch = case.batch([q] * real, [500 + 37 * i for i in range(real)])
    c = batch.common
    qsl = torch.cat(
        [c.query_start_loc_cpu, c.query_start_loc_cpu[-1:].repeat(padded - real)]
    )
    seq = torch.cat(
        [c.seq_lens_cpu_upper_bound, torch.zeros(padded - real, dtype=torch.int32)]
    )
    bt = torch.cat(
        [
            c.block_table_tensor,
            torch.zeros(
                padded - real,
                c.block_table_tensor.shape[1],
                dtype=torch.int32,
                device=DEVICE,
            ),
        ]
    )
    common = CommonAttentionMetadata(
        query_start_loc=qsl.to(DEVICE),
        query_start_loc_cpu=qsl,
        seq_lens=seq.to(DEVICE),
        seq_lens_cpu_upper_bound=seq,
        num_reqs=padded,
        num_actual_tokens=padded * q,
        max_query_len=q,
        max_seq_len=c.max_seq_len,
        block_table_tensor=bt,
        slot_mapping=torch.cat(
            [
                c.slot_mapping,
                torch.full(((padded - real) * q,), -1, device=DEVICE),
            ]
        ),
        causal=True,
    )
    return batch, common


def test_uniform_padded_verify_gets_dummy_rows():
    """Full graphs pad a verify batch with empty requests; the builder must
    give each the uniform qlen so the uniform-qlen kernel stays valid."""
    q, real, padded = 4, 3, 5
    case = _case(
        num_speculative_tokens=3, cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    batch, common = _padded_uniform_batch(case, q, real, padded)
    md = case.builder.build(0, common)
    d = md.decode
    assert md.num_decodes == padded and d.max_qo_len == q
    assert torch.equal(
        d.qo_indptr,
        torch.arange(0, (padded + 1) * q, q, dtype=torch.int32, device=DEVICE),
    )
    # Dummy requests read q pages so the kernel's causal tail is non-empty.
    assert d.seq_lens[real:].tolist() == [q] * (padded - real)
    assert d.seq_lens[:real].tolist() == batch.seq_lens


def test_uniform_padded_verify_output_is_correct():
    """The real rows of a padded verify batch match the reference."""
    q, real, padded = 4, 3, 5
    case = _case(
        num_speculative_tokens=3, cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    batch, common = _padded_uniform_batch(case, q, real, padded)
    md = case.builder.build(0, common)
    query = case.query(padded * q)
    out = case.run(batch, query, md)
    case.check(batch, query, out, md)


@pytest.mark.parametrize(
    "shape, spec, kv_cache_dtype",
    [
        (shapes.DSV3_TP8, 0, "auto"),
        (shapes.DSV3_TP8, 3, "fp8"),
        (shapes.DSV3_TP16, 0, "auto"),  # Gluon decode
        (shapes.KIMI_K3_TP8, 0, "auto"),  # padded asm
        (shapes.KIMI_K3_TP8, 3, "fp8"),
        (shapes.DSV3_TP16, 3, "auto"),  # Gluon verify
    ],
    ids=lambda v: getattr(v, "id", str(v)),
)
@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.NONE, CUDAGraphMode.FULL_AND_PIECEWISE],
    ids=lambda m: m.name,
)
def test_decode_only_build_does_not_sync(shape, spec, kv_cache_dtype, cudagraph_mode):
    case = _case(
        shape,
        num_speculative_tokens=spec,
        kv_cache_dtype=kv_cache_dtype,
        cudagraph_mode=cudagraph_mode,
    )
    batch = case.batch([spec + 1] * 3, [100, 300, 2000])
    case.build(batch)  # warm up lazy imports and JIT
    torch.cuda.synchronize()
    with no_gpu_sync():
        case.build(batch)


def test_capture_build_matches_regular_build():
    case = _case(
        num_speculative_tokens=3, cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    batch = case.batch([4] * 3, [100, 300, 2000])
    regular = case.build(batch)
    regular_fields = (
        regular.decode.paged_kv_indptr.clone(),
        regular.decode.qo_indptr.clone(),
        regular.decode.seq_lens.clone(),
    )
    capture = case.builder.build_for_cudagraph_capture(batch.common)
    assert capture.num_decodes == regular.num_decodes
    assert capture.decode.max_qo_len == regular.decode.max_qo_len
    for a, b in zip(
        regular_fields,
        (
            capture.decode.paged_kv_indptr,
            capture.decode.qo_indptr,
            capture.decode.seq_lens,
        ),
    ):
        assert torch.equal(a, b)


def test_full_graph_buffers_are_reused_across_builds():
    case = _case(
        num_speculative_tokens=3, cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    first = case.build(case.batch([4] * 3, [100, 300, 2000]))
    ptrs = {
        name: getattr(first.decode, name).data_ptr()
        for name in ("paged_kv_indptr", "qo_indptr", "seq_lens", "paged_kv_indices")
    }
    ptrs["work_meta_data"] = first.work_meta_data.data_ptr()
    second = case.build(case.batch([4] * 2, [4000, 17]))
    for name, ptr in ptrs.items():
        src = second if name == "work_meta_data" else second.decode
        assert getattr(src, name).data_ptr() == ptr, name


def test_full_graph_tails_are_flattened():
    """A graph captured at more requests than the replay reads past the live
    rows; the persistent indptrs must repeat their last value there, not keep
    the previous batch's entries."""
    case = _case(
        num_speculative_tokens=3, cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    case.build(case.batch([4] * 4, [1000, 1000, 1000, 1000]))
    md = case.build(case.batch([4] * 2, [100, 50]))
    full_indptr = case.builder.paged_kv_indptr
    full_qo = case.builder.qo_indptr
    assert (full_indptr[3:] == 150).all()
    assert (full_qo[3:] == 8).all()
    assert md.decode.paged_kv_indptr.numel() == 3


def test_non_causal_requires_a_marked_group():
    """A non-causal batch on a group whose layers did not ask for it is a
    configuration error, not a silent causal run."""
    case = _case(num_speculative_tokens=3)
    batch = make_paged_batch(
        [4] * 2, [100, 200], 16, DEVICE, num_blocks=2048, causal=False
    )
    with pytest.raises(ValueError, match="Non-causal multi-token MLA"):
        case.build(batch)


def test_non_causal_requires_a_uniform_block():
    case = _case(num_speculative_tokens=3, non_causal=True)
    batch = make_paged_batch(
        [4, 2], [100, 200], 16, DEVICE, num_blocks=2048, causal=False
    )
    with pytest.raises(ValueError, match="uniform query block"):
        case.build(batch)
