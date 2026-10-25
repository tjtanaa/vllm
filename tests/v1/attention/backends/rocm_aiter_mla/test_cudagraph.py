# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA decode under full CUDA graphs, at the layer level.

The builder declares ``UNIFORM_BATCH``: uniform decode and verify batches are
captured whole. A graph is captured once per batch shape and then replayed
with whatever the scheduler holds next, so every tensor the kernels read must
live in a persistent buffer the next build rewrites in place, and nothing a
replay depends on may be baked into the graph as a Python value. Each test
captures ``forward_impl`` on one batch and replays it on batches with other
sequence lengths and block tables, against the float32 reference. The runner
modes themselves (NONE / PIECEWISE / FULL_DECODE_ONLY / FULL_AND_PIECEWISE)
run end to end in tests/v1/e2e/attention/test_rocm_aiter_mla_runtime.py.
"""

import pytest
import torch

from tests.v1.attention.backends.harness import make_paged_batch
from tests.v1.attention.backends.rocm_aiter_mla import mla_shapes as shapes
from tests.v1.attention.backends.rocm_aiter_mla.mla_harness import (
    DEVICE,
    MlaCase,
)
from vllm.config.compilation import CUDAGraphMode

NUM_REQS = 4
CAPTURE_LENS = [700, 1500, 3000, 4100]
REPLAY_LENS = [
    [33, 900, 129, 5000],  # shorter and longer than at capture
    [9, 10, 11, 12],  # every request shorter than the shortest captured one
    [4096, 4096, 4096, 4096],
]


def _graph_case(shape, spec, kv_cache_dtype, monkeypatch, mode="auto"):
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", mode)
    return MlaCase(
        shape,
        kv_cache_dtype=kv_cache_dtype,
        num_speculative_tokens=spec,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    ).setup(2 * sum(-(-s // 16) for s in [5000] * NUM_REQS))


def _write_query_kv(case: MlaCase, batch) -> None:
    """KV writes run eagerly; the graph holds attention only."""
    kv_c, k_pe = case.split_latent(case.latent_all[batch.query_slots])
    case.layer.update_kv_cache(
        kv_c,
        k_pe,
        case.kv_cache,
        batch.common.slot_mapping,
        None,
        case.kv_cache_dtype,
        case.layer._k_scale,
    )


def _replays_match_eager(case: MlaCase, q: int) -> None:
    s = case.shape
    n = NUM_REQS * q
    batch = make_paged_batch(
        [q] * NUM_REQS, CAPTURE_LENS, 16, DEVICE, num_blocks=case.num_blocks, seed=1
    )
    md = case.build(batch)
    query = case.query(n)
    width = s.kv_lora_rank + s.qk_rope_head_dim
    latent = torch.zeros(n, width, dtype=case.dtype, device=DEVICE)
    kv_c, k_pe = case.split_latent(latent)
    output = torch.empty(n, s.num_heads * s.v_head_dim, dtype=case.dtype, device=DEVICE)
    _write_query_kv(case, batch)

    def forward():
        case.layer.forward_impl(query, kv_c, k_pe, case.kv_cache, md, output)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()  # warm up JIT and lazy allocations outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    graph.replay()
    torch.cuda.synchronize()
    case.check(batch, query, output.view(n, s.num_heads, s.v_head_dim), md)

    for i, seq_lens in enumerate(REPLAY_LENS):
        replay = make_paged_batch(
            [q] * NUM_REQS,
            seq_lens,
            16,
            DEVICE,
            num_blocks=case.num_blocks,
            seed=10 + i,
        )
        md_replay = case.build(replay)
        # Same captured shape: the Python-side routing must not change.
        assert md_replay.decode.max_qo_len == md.decode.max_qo_len
        assert md_replay.decode.use_gluon_decode == md.decode.use_gluon_decode
        assert md_replay.decode.use_gluon_verify == md.decode.use_gluon_verify
        _write_query_kv(case, replay)
        query.copy_(case.query(n))
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        case.check(replay, query, output.view(n, s.num_heads, s.v_head_dim), md_replay)


@pytest.mark.parametrize(
    "shape",
    [shapes.DSV3_TP8, shapes.DSV3_TP1, shapes.KIMI_K3_TP8, shapes.KIMI_K3_TP4],
    ids=lambda s: s.id,
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_decode_graph_replays_new_lengths(shape, kv_cache_dtype, monkeypatch):
    case = _graph_case(shape, 0, kv_cache_dtype, monkeypatch)
    _replays_match_eager(case, 1)


@pytest.mark.parametrize(
    "shape", [shapes.DSV3_TP16, shapes.KIMI_LINEAR_TP8], ids=lambda s: s.id
)
def test_gluon_decode_graph_replays_new_lengths(shape, monkeypatch):
    case = _graph_case(shape, 0, "auto", monkeypatch)
    _replays_match_eager(case, 1)


@pytest.mark.parametrize("spec", [1, 3])
@pytest.mark.parametrize(
    "shape", [shapes.DSV3_TP8, shapes.KIMI_K3_TP8], ids=lambda s: s.id
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_spec_decode_graph_replays(shape, spec, kv_cache_dtype, monkeypatch):
    case = _graph_case(shape, spec, kv_cache_dtype, monkeypatch)
    _replays_match_eager(case, spec + 1)


@pytest.mark.parametrize("spec", [1, 3, 5])
def test_gluon_verify_graph_replays(spec, monkeypatch):
    """Gluon's MTP entry takes a lower bound on the KV length it splits; a
    replay with shorter sequences than at capture must stay correct."""
    case = _graph_case(shapes.DSV3_TP16, spec, "auto", monkeypatch)
    _replays_match_eager(case, spec + 1)


def test_non_causal_block_graph_replays(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", "auto")
    case = MlaCase(
        shapes.DSV3_TP8,
        num_speculative_tokens=3,
        non_causal=True,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    ).setup(2 * sum(-(-s // 16) for s in [5000] * NUM_REQS))
    q = 4
    n = NUM_REQS * q
    s = case.shape
    batch = make_paged_batch(
        [q] * NUM_REQS,
        CAPTURE_LENS,
        16,
        DEVICE,
        num_blocks=case.num_blocks,
        causal=False,
        seed=1,
    )
    md = case.build(batch)
    query = case.query(n)
    kv_c, k_pe = case.split_latent(
        torch.zeros(
            n, s.kv_lora_rank + s.qk_rope_head_dim, dtype=case.dtype, device=DEVICE
        )
    )
    output = torch.empty(n, s.num_heads * s.v_head_dim, dtype=case.dtype, device=DEVICE)
    _write_query_kv(case, batch)
    case.layer.forward_impl(query, kv_c, k_pe, case.kv_cache, md, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.layer.forward_impl(query, kv_c, k_pe, case.kv_cache, md, output)
    replay = make_paged_batch(
        [q] * NUM_REQS,
        REPLAY_LENS[0],
        16,
        DEVICE,
        num_blocks=case.num_blocks,
        causal=False,
        seed=7,
    )
    md_replay = case.build(replay)
    _write_query_kv(case, replay)
    query.copy_(case.query(n))
    graph.replay()
    torch.cuda.synchronize()
    case.check(replay, query, output.view(n, s.num_heads, s.v_head_dim), md_replay)
