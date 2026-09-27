# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random
import time
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
)
from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd

pytestmark = pytest.mark.skip_global_cleanup

NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 64]
HEAD_SIZES = [24, 128]
DTYPES = [torch.float16]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]
SLIDING_WINDOW = [0, 16, 2048]
KV_CACHE_DTYPES = ["auto", "fp8", "fp8_e5m2"]

OPS = [chunked_prefill_paged_decode, context_attention_fwd]


@pytest.mark.parametrize(
    "length,page,bound,width,expected",
    [
        (769, 784, 0, 0, (769, 1)),
        (769, 784, 4096, 0, (4096, 6)),
        (769, 784, 4096, 8, (4096, 8)),
        (1024, 912, 4096, 0, (4096, 5)),
        (4096, 1056, 4096, 0, (4096, 4)),
        (4097, 1056, 0, 8, (4097, 8)),
    ],
)
def test_gfx1151_benchmark_context_bound_sizes_the_table(
    length, page, bound, width, expected
):
    from benchmarks.kernels.benchmark_gfx1151_qwen_attention import (
        resolve_context_layout,
    )

    assert resolve_context_layout(length, page, bound, width) == expected


@pytest.mark.parametrize(
    "length,page,bound,width",
    [
        (0, 784, 4096, 0),
        (-1, 784, 4096, 0),
        (769, 0, 4096, 0),
        (769, 784, -1, 0),
        (769, 784, 768, 0),
        (769, 784, 4096, 5),
        (769, 784, 4096, -1),
    ],
)
def test_gfx1151_benchmark_context_bound_rejects_before_gpu_allocation(
    monkeypatch, length, page, bound, width
):
    import benchmarks.kernels.benchmark_gfx1151_qwen_attention as bench

    def fail_allocation(*args, **kwargs):
        pytest.fail("Invalid bounds must be rejected before tensor allocation")

    monkeypatch.setattr(bench, "make_attention_inputs", fail_allocation)
    with pytest.raises(ValueError):
        bench.benchmark(
            1, length, page, 1, 1, context_bound=bound, block_table_width=width
        )


def test_gfx1151_benchmark_context_bound_preserves_actual_lengths_and_values():
    from benchmarks.kernels.benchmark_gfx1151_qwen_attention import (
        make_attention_inputs,
        resolve_context_layout,
    )

    inputs = []
    for bound in (769, 4096):
        _, width = resolve_context_layout(769, 784, bound)
        torch.manual_seed(42)
        inputs.append(
            make_attention_inputs(1, 769, 784, 1, block_table_width=width, device="cpu")
        )
    short, captured = inputs
    assert torch.equal(short[0], captured[0])  # Q and reference K/V are unchanged.
    assert torch.equal(short[1], captured[1])
    assert torch.equal(short[2], captured[2])
    assert captured[5].shape == (1, 6)
    assert torch.equal(short[5], captured[5][:, :1])
    assert torch.all(captured[5][:, 1:] == -1)
    assert torch.equal(short[6], captured[6])
    assert captured[6].tolist() == [769]


def test_gfx1151_benchmark_digest_uses_logical_values_not_storage_padding():
    from benchmarks.kernels.benchmark_gfx1151_qwen_attention import tensor_digest

    storage = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    view = storage[:, :4]
    digest = tensor_digest(view)
    assert digest == tensor_digest(view.clone())
    storage[:, 4:] = float("nan")
    assert digest == tensor_digest(view)
    assert digest != tensor_digest(view.float())
    assert digest != tensor_digest(view.contiguous().reshape(2, 6))
    view[0, 0] += 1
    assert digest != tensor_digest(view)


@pytest.mark.parametrize("query_len", [19, 37])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [0, 11])
def test_gfx1151_benchmark_reference_chunking(query_len, causal, window):
    """Reference chunk boundaries must preserve GQA and trailing-query masks."""
    from benchmarks.kernels.benchmark_gfx1151_qwen_attention import (
        attention_reference,
    )

    generator = torch.Generator().manual_seed(42)
    q = torch.randn(2 * query_len, 6, 8, generator=generator).bfloat16()
    k = torch.randn(2, 37, 2, 8, generator=generator).bfloat16()
    v = torch.randn(2, 37, 2, 8, generator=generator).bfloat16()
    actual = attention_reference(q, k, v, causal, window, chunk_size=7)
    positions = torch.arange(37 - query_len, 37)[:, None]
    keys = torch.arange(37)[None, :]
    mask = torch.ones(query_len, 37, dtype=torch.bool)
    if causal:
        mask &= keys <= positions
    if window:
        mask &= keys >= positions - window + 1
    expected = (
        F.scaled_dot_product_attention(
            q.double().view(2, query_len, 6, 8).transpose(1, 2),
            k.double().repeat_interleave(3, dim=2).transpose(1, 2),
            v.double().repeat_interleave(3, dim=2).transpose(1, 2),
            attn_mask=mask,
        )
        .transpose(1, 2)
        .reshape_as(q)
    )
    torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("query_len", [24, 37])
@pytest.mark.parametrize("draft", [False, True])
def test_gfx1151_benchmark_layout_preserves_values_and_page_guards(
    batch, query_len, draft
):
    """Storage-layout experiments must not change attention inputs or hide padding."""
    from benchmarks.kernels.benchmark_gfx1151_qwen_attention import (
        make_attention_inputs,
    )

    # Small CPU shapes cover both cached-prefix and full-current K/V. Keep the
    # actual projection row stride and offset rather than scaling them down.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        original = make_attention_inputs(batch, 37, 17, query_len, draft, device="cpu")
        torch.manual_seed(42)
        measured = make_attention_inputs(
            batch,
            37,
            17,
            query_len,
            draft,
            cache_page_padding=0,
            block_table_width=8,
            value_row_stride=0 if draft else 14336,
            value_storage_offset=0 if draft else 13312,
            device="cpu",
        )
    q, k, v, kc, vc, table, lengths, starts, new_k, new_v = measured
    for actual, expected in zip((q, k, v), original[:3]):
        assert torch.equal(actual, expected)
    assert torch.equal(table[:, :3], original[5])
    assert (table[:, 3:] == -1).all()
    assert table.stride() == (8, 1)
    assert torch.equal(table[:, :3].flatten().sort().values, torch.arange(batch * 3))
    assert lengths.tolist() == [37] * batch
    assert starts.tolist() == [b * query_len for b in range(batch + 1)]
    kv_heads, dim = k.shape[-2:]
    assert kc.stride(0) == vc.stride(0) == 2 * kv_heads * dim * 17
    assert original[3].stride(0) == kc.stride(0) + 128
    for inputs in (original, measured):
        cache_k, cache_v, pages = inputs[3:6]
        for b in range(batch):
            for page in range(3):
                physical = int(pages[b, page])
                first, count = page * 17, min(17, 37 - page * 17)
                actual_k = cache_k[physical, :, :, :count].permute(2, 0, 1, 3)
                assert torch.equal(
                    actual_k.reshape(count, kv_heads, dim), k[b, first : first + count]
                )
                assert torch.equal(
                    cache_v[physical, :, :, :count].permute(2, 0, 1),
                    v[b, first : first + count],
                )
                assert cache_k[physical, :, :, count:].isnan().all()
                assert cache_v[physical, :, :, count:].isnan().all()
        # Trailing page padding is never initialized with usable token values.
        storage = cache_k.as_strided(
            (batch * 3, cache_k.stride(0)), (cache_k.stride(0), 1)
        )
        assert storage[:, 2 * kv_heads * dim * 17 :].isnan().all()
    if draft:
        assert new_k is None and new_v is None
    else:
        assert torch.equal(
            new_k.view(batch, query_len, kv_heads, dim), k[:, -query_len:]
        )
        assert torch.equal(
            new_v.view(batch, query_len, kv_heads, dim), v[:, -query_len:]
        )
        assert new_v.stride() == (14336, 256, 1)
        assert new_v.storage_offset() == 13312
        storage = new_v.as_strided(
            (new_v.untyped_storage().nbytes() // new_v.element_size(),), (1,), 0
        )
        assert storage[:13312].isnan().all()
        rows = new_v.as_strided((batch * query_len - 1, 14336), (14336, 1))
        assert rows[:, 1024:].isnan().all()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("query_len", [8, 32])
def test_gfx1151_draft_dense_projection_preserves_cached_values(batch, query_len):
    """The dense draft timing baseline must read the same K/V as its cache."""
    from benchmarks.kernels.benchmark_gfx1151_qwen_attention import (
        make_attention_inputs,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        cached = make_attention_inputs(batch, 37, 17, query_len, True, device="cpu")
        torch.manual_seed(42)
        dense = make_attention_inputs(
            batch,
            37,
            17,
            query_len,
            True,
            value_row_stride=6144,
            value_storage_offset=5120,
            draft_dense_kv=True,
            device="cpu",
        )
    for actual, expected in zip(dense[:3], cached[:3]):
        assert torch.equal(actual, expected)
    new_k, new_v = dense[-2:]
    assert cached[-2:] == (None, None)
    assert new_k.stride() == (1024, 128, 1)
    assert new_v.stride() == (6144, 128, 1)
    assert new_v.storage_offset() == 5120
    assert torch.equal(new_k.view(batch, query_len, 8, 128), dense[1][:, -query_len:])
    assert torch.equal(new_v.view(batch, query_len, 8, 128), dense[2][:, -query_len:])


@pytest.mark.parametrize(
    "options",
    [
        {"batch": 0},
        {"query_len": 38},
        {"block_size": 0},
        {"cache_page_padding": -1},
        {"block_table_width": -1},
        {"block_table_width": 2},
        {"value_row_stride": -1},
        {"value_row_stride": 1023},
        {"value_storage_offset": -1},
        {"draft": True, "value_row_stride": 14336},
        {"draft": True, "value_storage_offset": 13312},
        {"draft_dense_kv": True},
    ],
)
def test_gfx1151_benchmark_rejects_invalid_layout_before_allocation(
    options, monkeypatch
):
    """Reject overlapping V rows or undersized page tables before touching a device."""
    from benchmarks.kernels.benchmark_gfx1151_qwen_attention import (
        make_attention_inputs,
    )

    def no_allocation(*args, **kwargs):
        raise AssertionError("Invalid layout reached device allocation")

    monkeypatch.setattr(torch, "full", no_allocation)
    kwargs = dict(batch=1, length=37, block_size=17, query_len=24, device="cpu")
    with pytest.raises(ValueError):
        make_attention_inputs(**(kwargs | options))


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("window", [0, 11])
@pytest.mark.parametrize("config", [(128, 64, 4), (32, 32, 8)])
@pytest.mark.parametrize("explicit_scales", [False, True])
def test_gfx1151_prefix_tuning_preserves_math_contract(
    monkeypatch, draft, window, config, explicit_scales
):
    """A tile experiment may change launch geometry, not strides or attention math."""
    from vllm.v1.attention.ops import gfx1151_qwen_prefill as tuning
    from vllm.v1.attention.ops import prefix_prefill

    launches = []

    class CaptureLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(
                    (grid(kwargs) if callable(grid) else grid, args, kwargs)
                )

            return launch

    recorder = CaptureLaunch()
    monkeypatch.setattr(prefix_prefill, "_fwd_kernel", recorder)
    monkeypatch.setattr(tuning, "_fwd_kernel", recorder)
    heads, kv_heads, dim = (32, 8, 128) if draft else (24, 4, 256)
    query_len = 129
    q = torch.empty(2 * query_len, heads, dim, dtype=torch.bfloat16)
    k = None if draft else torch.empty(2 * query_len, kv_heads, dim, dtype=q.dtype)
    v = None if draft else torch.empty_like(k)
    out = torch.empty_like(q)
    kc = torch.empty(16, kv_heads, dim // 8, 32, 8, dtype=q.dtype)
    vc = torch.empty(16, kv_heads, dim, 32, dtype=q.dtype)
    table = torch.arange(16, dtype=torch.int32).view(2, 8)
    starts = torch.tensor([0, query_len, 2 * query_len], dtype=torch.int32)
    lengths = torch.tensor([256, 256], dtype=torch.int32)
    scale = torch.ones(())
    v_scale = torch.tensor(2.0) if explicit_scales else scale
    sm_scale = 0.11 if explicit_scales else None
    prefix_prefill.context_attention_fwd(
        q,
        k,
        v,
        out,
        "auto",
        kc,
        vc,
        table,
        starts,
        lengths,
        256,
        query_len,
        scale,
        v_scale,
        sliding_window=window,
        sm_scale=sm_scale,
        skip_decode=True,
        causal=not draft,
    )
    tuning.launch_configured_prefill(
        q,
        k,
        v,
        out,
        kc,
        vc,
        table,
        starts,
        lengths,
        scale,
        query_len,
        not draft,
        window,
        config,
        v_scale=v_scale,
        sm_scale=sm_scale,
    )
    _, original_args, original_kwargs = launches[0]
    grid, candidate_args, candidate_kwargs = launches[1]
    assert len(original_args) == len(candidate_args)
    for original, candidate in zip(original_args, candidate_args):
        if isinstance(original, torch.Tensor):
            assert candidate is original
        else:
            assert candidate == original
    assert candidate_kwargs == {
        **original_kwargs,
        "BLOCK_M": config[0],
        "BLOCK_N": config[1],
        "num_warps": config[2],
    }
    assert grid == (2, heads, math.ceil(query_len / config[0]))


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("layout", ["padded", "qwen_projection"])
@pytest.mark.parametrize(
    "length,query_len,page,config,cached",
    [
        (911, 24, 912, (16, 64, 4), False),
        (912, 24, 912, (32, 64, 4), False),
        (913, 24, 912, (16, 64, 4), False),
        (1024, 24, 912, (16, 64, 4), False),
        (2048, 24, 912, (32, 64, 4), False),
        (4096, 24, 912, (32, 64, 4), False),
        (1024, 1024, 784, (64, 64, 8), False),
        (2048, 2048, 784, (64, 64, 8), False),
        (4096, 4096, 784, (64, 64, 8), False),
        (2048, 1024, 784, (64, 64, 4), False),
        (4096, 1024, 784, (64, 64, 8), False),
        (4096, 1024, 784, (64, 64, 4), False),
        (1360, 1263, 784, (64, 64, 8), False),
        (1407, 1355, 784, (64, 64, 8), False),
    ]
    + [
        pytest.param(
            length,
            query_len,
            832,
            (16, 64, 4),
            cached,
            id=f"verification-l{length}-q{query_len}-cached{cached}",
        )
        for length in (831, 832, 833, 1024, 2048, 4096)
        for query_len in (4, 8, 16)
        for cached in (False, True)
    ],
)
@torch.inference_mode()
def test_gfx1151_prefix_tiles_ragged_pages_and_dirty_graph(
    batch,
    length,
    query_len,
    page,
    config,
    cached,
    combined,
    layout,
    draft=False,
    causal=True,
    window=0,
    projection_table_width=None,
):
    """Tile changes must preserve prefix math and skipped/padded output rows.

    Cover the prefix-only launcher and its combined serving wrapper, including
    graph replay with changed query boundaries. The prefix-only cases include
    empty slots; combined cases retain the generic decode contract of nonempty
    requests. Qwen projection strides and the two non-tile-aligned bounds come
    from normal-memory model traces; contents and ragged lengths are synthetic.
    Verification cases compare dense and cache-only operands to the same dense
    baseline, including physical page boundaries. This does not establish
    model-level token parity.
    """
    if not current_platform.is_rocm():
        pytest.skip("gfx1151 prefix tile")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1151":
        pytest.skip("gfx1151 prefix tile")
    from benchmarks.kernels.benchmark_gfx1151_qwen_attention import (
        attention_reference,
    )
    from benchmarks.kernels.gfx1151_qwen_prefill_tuning import (
        launch_configured_prefill,
    )
    from vllm.v1.attention.ops.gfx1151_qwen_prefill import gfx1151_qwen_prefill

    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    heads, kv_heads, dim = (32, 8, 128) if draft else (24, 4, 256)
    query_lens = [
        (query_len, max(2, query_len - 13), 1, int(combined))[b % 4]
        for b in range(batch)
    ]
    seq_lens = [
        max(query_lens[b], length - b * 17) if query_lens[b] else 0
        for b in range(batch)
    ]
    offsets = [0]
    for count in query_lens:
        offsets.append(offsets[-1] + count)
    tokens = offsets[-1]
    projection_layout = layout == "qwen_projection"
    query_width = dim if projection_layout else dim * 2
    query = torch.randn(tokens + 7, heads, query_width, device=device, dtype=dtype)[
        ..., :dim
    ]
    keys = torch.randn(batch, length, kv_heads, dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    if projection_layout:
        new_k = torch.empty(tokens, kv_heads, dim, device=device, dtype=dtype)
        projection_heads, value_offset = (48, 40) if draft else (56, 52)
        new_v_storage = torch.empty(
            tokens, projection_heads * dim, device=device, dtype=dtype
        )
        new_v = new_v_storage[:, value_offset * dim :].view(tokens, kv_heads, dim)
        assert new_v.stride() == (projection_heads * dim, dim, 1)
        assert new_v.storage_offset() == value_offset * dim
    else:
        new_k_storage = torch.empty(
            tokens, kv_heads, dim * 2, device=device, dtype=dtype
        )
        new_v_storage = torch.empty_like(new_k_storage)
        new_k, new_v = new_k_storage[..., :dim], new_v_storage[..., :dim]
    pages = math.ceil(length / page)
    page_elements = kv_heads * dim * page
    storage = torch.full(
        (batch * pages, 2 * page_elements + (0 if projection_layout else 128)),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    kc = storage[:, :page_elements].view(batch * pages, kv_heads, dim // 8, page, 8)
    vc = storage[:, page_elements : 2 * page_elements].view(
        batch * pages, kv_heads, dim, page
    )
    table_width = (6 if draft else 8) if projection_layout else pages + 2
    if projection_layout and projection_table_width is not None:
        table_width = projection_table_width
    assert pages <= table_width
    table_storage = torch.full(
        (batch, table_width), -1, device=device, dtype=torch.int32
    )
    table = table_storage[:, :pages]
    table.copy_(torch.randperm(batch * pages, device=device).int().view(batch, pages))
    reference = torch.full_like(query, float("nan"), dtype=torch.float32)
    active = torch.zeros(tokens + 7, device=device, dtype=torch.bool)
    for b, (qlen, slen) in enumerate(zip(query_lens, seq_lens)):
        lo, hi = offsets[b : b + 2]
        if qlen:
            new_k[lo:hi] = keys[b, slen - qlen : slen]
            new_v[lo:hi] = values[b, slen - qlen : slen]
        for p in range(pages):
            first = p * page
            count = min(page, slen - first)
            if count <= 0:
                continue
            physical = int(table[b, p])
            kc[physical, :, :, :count, :] = (
                keys[b, first : first + count]
                .view(count, kv_heads, dim // 8, 8)
                .permute(1, 2, 0, 3)
            )
            vc[physical, :, :, :count] = values[b, first : first + count].permute(
                1, 2, 0
            )
        if qlen > 1 or (combined and qlen == 1):
            active[lo:hi] = True
            reference[lo:hi] = attention_reference(
                query[lo:hi],
                keys[b : b + 1, :slen],
                values[b : b + 1, :slen],
                causal,
                window,
            )
    starts = torch.tensor(offsets, device=device, dtype=torch.int32)
    lengths = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    scale = torch.ones((), device=device)
    # Distinct layer-scale tensors match model calls, even for BF16 caches.
    v_scale = torch.ones((), device=device) if draft else scale
    output = torch.full_like(query, float("nan"))
    baseline = torch.full_like(query, float("nan"))

    def run_baseline():
        kwargs = {} if combined else {"skip_decode": True}
        function = chunked_prefill_paged_decode if combined else context_attention_fwd
        function(
            query,
            new_k,
            new_v,
            baseline,
            "auto",
            kc,
            vc,
            table,
            starts,
            lengths,
            length,
            query_len,
            scale,
            v_scale,
            causal=causal,
            sliding_window=window,
            **kwargs,
        )

    run_baseline()

    def run():
        if combined:
            gfx1151_qwen_prefill(
                query,
                None if cached else new_k,
                None if cached else new_v,
                output,
                kc,
                vc,
                table,
                starts,
                lengths,
                length,
                query_len,
                scale,
                v_scale,
                dim**-0.5,
                config,
                causal=causal,
                window=window,
            )
            return
        launch_configured_prefill(
            query,
            None if cached else new_k,
            None if cached else new_v,
            output,
            kc,
            vc,
            table,
            starts,
            lengths,
            scale,
            query_len,
            causal,
            window,
            config,
            v_scale=v_scale,
        )

    def check():
        assert torch.isnan(output[~active]).all()
        assert torch.isnan(baseline[~active]).all()
        result = output[active].float()
        expected = reference[active]
        assert torch.isfinite(result).all()
        assert (result - expected).norm() / expected.norm() < 0.01
        assert (
            ~torch.isclose(result, expected, atol=0.01, rtol=0.01)
        ).float().mean() < 0.05
        torch.testing.assert_close(output[active], baseline[active], atol=0, rtol=0)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check()
    if combined and batch > 1:
        # A graph captured with a prefix in request 0 must also compute its
        # output when that request becomes a single-token decode on replay.
        query_lens[0] = 1
        offsets = [0]
        for qlen in query_lens:
            offsets.append(offsets[-1] + qlen)
        starts.copy_(torch.tensor(offsets, device=device, dtype=torch.int32))
        active.zero_()
        reference.fill_(float("nan"))
        for b, (qlen, slen) in enumerate(zip(query_lens, seq_lens)):
            lo, hi = offsets[b : b + 2]
            new_k[lo:hi] = keys[b, slen - qlen : slen]
            new_v[lo:hi] = values[b, slen - qlen : slen]
            active[lo:hi] = True
            reference[lo:hi] = attention_reference(
                query[lo:hi],
                keys[b : b + 1, :slen],
                values[b : b + 1, :slen],
                causal,
                window,
            )
        baseline.fill_(float("nan"))
        run_baseline()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("layout", ["padded", "qwen_projection"])
@pytest.mark.parametrize(
    "length,query_len,causal,window",
    [
        (4096, q, causal, window)
        for q in (4, 8, 16, 24, 32)
        for causal, window in ((True, 0), (False, 0), (False, 2047))
    ]
    + [(length, 8, False, 2047) for length in (831, 832, 833, 1024, 2046, 2048, 2049)]
    + [(4096, 8, True, 2047)],
)
def test_gfx1151_draft_prefix_preserves_dense_reference_and_mixed_replay(
    batch, combined, layout, length, query_len, causal, window
):
    """Cached draft tiles must equal dense reference across page/window edges.

    Reuse the existing allocation, FP32 reference, sentinel and dirty-graph
    checks, including prefix-to-M1 transitions. Keep the dense baseline so
    cache-only operands cannot accidentally weaken the comparison.
    """
    test_gfx1151_prefix_tiles_ragged_pages_and_dirty_graph(
        batch,
        length,
        query_len,
        832,
        (16, 64, 8),
        True,
        combined,
        layout,
        draft=True,
        causal=causal,
        window=window,
    )


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("layout", ["padded", "qwen_projection"])
@pytest.mark.parametrize("block_m", [16, 32])
@pytest.mark.parametrize("boundary", [-1, 0, 1, "4k"])
@pytest.mark.parametrize(
    "draft,query_len,page,table_width",
    [
        (True, 4, 800, 8),
        (True, 16, 864, 8),
        (True, 24, 912, 8),
        (True, 32, 944, 8),
        (False, 32, 944, 8),
    ],
)
def test_gfx1151_wider_live_pages_preserve_dense_reference_and_mixed_replay(
    batch, combined, layout, block_m, boundary, draft, query_len, page, table_width
):
    """Both candidate tiles must preserve masking on wider DFlash cache pages.

    Reuse the dense FP32/bitwise, sentinel and dirty mixed-M1 graph checks.
    Page geometry comes from the original BF16 draft's 3/15/23/31-token model
    runs, including the target M32 path and non-32-aligned physical pages.
    """
    length = 4096 if boundary == "4k" else page + boundary
    test_gfx1151_prefix_tiles_ragged_pages_and_dirty_graph(
        batch,
        length,
        query_len,
        page,
        (block_m, 64, 8 if draft else 4),
        True,
        combined,
        layout,
        draft=draft,
        causal=not draft,
        window=2047 if draft else 0,
        projection_table_width=table_width,
    )


def create_causal_attention_mask_for_sdpa(
    query_lens: list[int],
    seq_lens: list[int],
    sliding_window: int = 0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    total_queries = sum(query_lens)
    total_keys = sum(seq_lens)

    # Create a mask filled with -inf
    mask = torch.full(
        (total_queries, total_keys), float("-inf"), device=device, dtype=dtype
    )

    query_start = 0
    key_start = 0

    for query_len, seq_len in zip(query_lens, seq_lens):
        query_end = query_start + query_len
        key_end = key_start + seq_len
        q_indices = torch.arange(query_len, device=device)
        k_indices = torch.arange(seq_len, device=device)
        q_pos_in_seq = seq_len - query_len + q_indices

        valid_mask = k_indices[None, :] <= q_pos_in_seq[:, None]

        if sliding_window > 0:
            valid_mask &= k_indices[None, :] >= (
                q_pos_in_seq[:, None] - sliding_window + 1
            )

        mask[query_start:query_end, key_start:key_end][valid_mask] = 0.0

        query_start = query_end
        key_start = key_end

    return mask


def create_alibi_causal_mask(
    query_len: int,
    seq_len: int,
    alibi_slopes: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    query_pos = torch.arange(
        seq_len - query_len, seq_len, device=device, dtype=torch.float32
    )
    key_pos = torch.arange(seq_len, device=device, dtype=torch.float32)

    rel_pos = key_pos[None, :] - query_pos[:, None]

    # Apply ALiBi slopes: [num_heads, query_len, seq_len]
    alibi_bias = alibi_slopes[:, None, None] * rel_pos[None, :, :]
    alibi_bias = alibi_bias.to(dtype)

    # Apply causal mask: prevent attending to future positions
    # causal_mask[i, j] = True if key_pos[j] <= query_pos[i]
    causal_mask = key_pos[None, :] <= query_pos[:, None]
    alibi_bias = alibi_bias.masked_fill(~causal_mask[None, :, :], float("-inf"))

    # Add batch dimension: [1, num_heads, query_len, seq_len]
    # SDPA expects batch dimension even for single sequences
    return alibi_bias.unsqueeze(0)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
    block_size: int = 32,
) -> None:
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    if (
        current_platform.is_rocm()
        and op is chunked_prefill_paged_decode
        and kv_cache_dtype == "fp8_e5m2"
    ):
        pytest.skip("ROCm custom paged attention does not support fp8_e5m2 KV cache")

    set_random_seed(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.accelerator.set_device_index(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    # ensure one sequence in batch is a decode
    query_lens[-1] = 1

    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.int32)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens), dim=0).to(torch.int32)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1]), dim=0).to(
        torch.int32
    )
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    torch.accelerator.synchronize()
    start_time = time.time()
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time) * 1000:.2f} ms")

    scale = float(1.0 / (head_size**0.5))

    # Reshape for SDPA: (seq_len, num_heads, head_size) ->
    # (1, num_heads, seq_len, head_size)
    query_sdpa = query.view(num_tokens, num_kv_heads, num_queries_per_kv, head_size)
    query_sdpa = query_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, num_tokens, head_size
    )

    # Expand key and value for GQA/MQA to match query heads
    key_sdpa = key[:, :, None, :].expand(
        key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1]
    )
    key_sdpa = key_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )

    value_sdpa = value[:, :, None, :].expand(
        value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1]
    )
    value_sdpa = value_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )

    attn_mask = create_causal_attention_mask_for_sdpa(
        query_lens, seq_lens, sliding_window, device=device, dtype=dtype
    )

    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    torch.accelerator.synchronize()
    start_time = time.time()
    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"PyTorch SDPA Time: {(end_time - start_time) * 1000:.2f} ms")

    # Reshape output back to (num_tokens, num_heads, head_size)
    output_ref = output_ref.view(num_heads, num_tokens, head_size)
    output_ref = output_ref.permute(1, 0, 2).contiguous()
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-4
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_cached_kv(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    block_size: int = 32,
) -> None:
    # Exercises the KV_FROM_CACHE path of context_attention_fwd: the current
    # chunk K/V are not passed as dense tensors (k=v=None); they are read back
    # from the paged KV cache, as done by layers that re-attend an already
    # cached sequence with the query only (e.g. IQuest LoopCoder's
    # attn(q, None, None)). The whole sequence therefore lives in the cache and
    # the result must still match a dense causal SDPA reference.
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    set_random_seed(0)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    # ensure one sequence in batch is a decode
    query_lens[-1] = 1
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens), dim=0).to(torch.int32)
    max_input_len = MAX_SEQ_LEN
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1]), dim=0).to(
        torch.int32
    )
    # Unlike the dense test, write the WHOLE sequence (context + current chunk)
    # into the paged cache, since the current chunk is read back from the cache.
    for i in range(BS):
        cur = 0
        block_id = 0
        while cur < seq_lens[i]:
            start_loc = b_seq_start_loc[i] + cur
            if cur + block_size > seq_lens[i]:
                end_loc = b_seq_start_loc[i] + seq_lens[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur += block_size
            block_id += 1
    # transpose to the paged cache layouts the kernel expects:
    #   K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    #   V_cache[num_blocks, num_kv_heads, head_size, block_size]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Cached-K/V path: current-chunk k and v are None.
    context_attention_fwd(
        query,
        None,
        None,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=0,
    )
    torch.accelerator.synchronize()

    scale = float(1.0 / (head_size**0.5))

    query_sdpa = query.view(num_tokens, num_kv_heads, num_queries_per_kv, head_size)
    query_sdpa = query_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, num_tokens, head_size
    )
    key_sdpa = key[:, :, None, :].expand(
        key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1]
    )
    key_sdpa = key_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )
    value_sdpa = value[:, :, None, :].expand(
        value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1]
    )
    value_sdpa = value_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )

    attn_mask = create_causal_attention_mask_for_sdpa(
        query_lens, seq_lens, 0, device=device, dtype=dtype
    )
    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    output_ref = output_ref.view(num_heads, num_tokens, head_size)
    output_ref = output_ref.permute(1, 0, 2).contiguous()
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-4
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_cached_kv_block_table_boundary(device: str) -> None:
    # Boundary guard for the KV_FROM_CACHE block-table load. With an
    # exact-sized block table (row length == number of blocks for the
    # sequence) and a query that ends the sequence, the last K/V tile has
    # padded lanes whose absolute positions step past the sequence end. Those
    # lanes must not read a block-table entry past this batch's row.
    #
    # seq_len is a whole number of blocks, so bn_logical for a padded lane
    # lands exactly on num_blocks (one past the last valid row entry), and
    # query_len is not tile-aligned so the overshoot is actually exercised.
    # The result must still match a dense causal SDPA reference.
    set_random_seed(0)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    dtype = torch.float16
    kv_cache_dtype = "auto"
    num_heads = 4
    num_queries_per_kv = 1
    num_kv_heads = num_heads // num_queries_per_kv
    head_size = 32
    block_size = 32
    num_blocks = 4
    seq_len = num_blocks * block_size  # 128 == exact multiple of block_size
    query_len = 99  # < seq_len and not a multiple of the kernel tile

    query = torch.empty(query_len, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(query_len, num_heads, head_size, dtype=dtype)

    kv = torch.empty(seq_len, 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    # Exact-sized block table: row length == num_blocks (identity mapping).
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(1, num_blocks)
    b_seq_len = torch.tensor([seq_len], dtype=torch.int32)
    b_start_loc = torch.tensor([0, query_len], dtype=torch.int32)

    # Write the whole sequence (context + current chunk) into the paged cache.
    for cur in range(0, seq_len, block_size):
        block_id = cur // block_size
        end = min(cur + block_size, seq_len)
        start_slot = block_table[0, block_id] * block_size
        end_slot = start_slot + (end - cur)
        k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
            key[cur:end]
        )
        v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
            value[cur:end]
        )

    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Cached-K/V path: current-chunk k and v are None.
    context_attention_fwd(
        query,
        None,
        None,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        seq_len,
        query_len,
        k_scale,
        v_scale,
        sliding_window=0,
    )
    torch.accelerator.synchronize()

    scale = float(1.0 / (head_size**0.5))
    query_sdpa = query.view(query_len, num_kv_heads, num_queries_per_kv, head_size)
    query_sdpa = query_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, query_len, head_size
    )
    key_sdpa = key[:, :, None, :].expand(
        seq_len, num_kv_heads, num_queries_per_kv, head_size
    )
    key_sdpa = key_sdpa.permute(1, 2, 0, 3).reshape(1, num_heads, seq_len, head_size)
    value_sdpa = value[:, :, None, :].expand(
        seq_len, num_kv_heads, num_queries_per_kv, head_size
    )
    value_sdpa = value_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, seq_len, head_size
    )

    attn_mask = create_causal_attention_mask_for_sdpa(
        [query_len], [seq_len], 0, device=device, dtype=dtype
    )
    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    output_ref = output_ref.view(num_heads, query_len, head_size)
    output_ref = output_ref.permute(1, 0, 2).contiguous()
    torch.testing.assert_close(output, output_ref, atol=1e-4, rtol=0)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_cached_kv_alibi_unsupported(device: str) -> None:
    # The cached-K/V (k=None) path is not supported together with ALiBi; the
    # entry point must reject it up-front with a clear NotImplementedError
    # rather than launching the kernel with an unsupported combination.
    set_random_seed(0)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    num_heads = 4
    num_kv_heads = 4
    head_size = 16
    x = 8
    block_size = 16
    num_blocks = 4
    query_len = 8

    query = torch.empty(query_len, num_heads, head_size, dtype=torch.float16)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty_like(query)
    k_cache = torch.zeros(
        num_blocks, num_kv_heads, head_size // x, block_size, x, dtype=torch.float16
    )
    v_cache = torch.zeros(
        num_blocks, num_kv_heads, head_size, block_size, dtype=torch.float16
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(1, num_blocks)
    b_seq_len = torch.tensor([query_len], dtype=torch.int32)
    b_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    alibi_slopes = torch.ones(num_heads, dtype=torch.float32, device=device)

    with pytest.raises(NotImplementedError):
        context_attention_fwd(
            query,
            None,
            None,
            output,
            "auto",
            k_cache,
            v_cache,
            block_table,
            b_start_loc,
            b_seq_len,
            query_len,
            query_len,
            k_scale,
            v_scale,
            alibi_slopes=alibi_slopes,
        )


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_alibi(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
    block_size: int = 32,
) -> None:
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    if (
        current_platform.is_rocm()
        and op is chunked_prefill_paged_decode
        and kv_cache_dtype == "fp8_e5m2"
    ):
        pytest.skip("ROCm custom paged attention does not support fp8_e5m2 KV cache")

    set_random_seed(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.accelerator.set_device_index(device)

    def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
        # Fork from: vllm/vllm/model_executor/models/bloom.py#L44
        closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
        base = torch.tensor(
            2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != total_num_heads:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
                dtype=torch.float32,
            )
            num_remaining_heads = min(
                closest_power_of_2, total_num_heads - closest_power_of_2
            )
            extra_powers = torch.arange(
                start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
            )
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
        return slopes

    alibi_slopes = _get_alibi_slopes(num_heads).to(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)
    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.int32)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens), dim=0).to(torch.int32)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1]), dim=0).to(
        torch.int32
    )
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    torch.accelerator.synchronize()
    start_time = time.time()
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time) * 1000:.2f} ms")
    scale = float(1.0 / (head_size**0.5))

    # Prepare query, key, value for SDPA
    # Expand key and value for GQA/MQA to match query heads
    key_expanded = key[:, :, None, :].expand(
        key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1]
    )
    value_expanded = value[:, :, None, :].expand(
        value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1]
    )

    output_ref = torch.empty_like(output)

    torch.accelerator.synchronize()
    start_time = time.time()

    query_start = 0
    key_start = 0
    for i, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        query_end = query_start + query_len
        key_end = key_start + seq_len

        # Get query, key, value for this sequence
        q = query[query_start:query_end]  # [query_len, num_heads, head_size]
        k = key_expanded[
            key_start:key_end
        ]  # [seq_len, num_kv_heads, num_queries_per_kv, head_size]
        v = value_expanded[
            key_start:key_end
        ]  # [seq_len, num_kv_heads, num_queries_per_kv, head_size]

        # Reshape for SDPA: (batch=1, num_heads, seq_len, head_size)
        q_sdpa = q.view(query_len, num_kv_heads, num_queries_per_kv, head_size)
        q_sdpa = (
            q_sdpa.permute(1, 2, 0, 3)
            .reshape(1, num_heads, query_len, head_size)
            .contiguous()
        )

        k_sdpa = (
            k.permute(1, 2, 0, 3).reshape(1, num_heads, seq_len, head_size).contiguous()
        )
        v_sdpa = (
            v.permute(1, 2, 0, 3).reshape(1, num_heads, seq_len, head_size).contiguous()
        )

        # Create ALiBi causal mask for this sequence using utility function
        alibi_mask = create_alibi_causal_mask(
            query_len, seq_len, alibi_slopes, device, dtype
        )

        # Compute attention
        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=alibi_mask,
            dropout_p=0.0,
            scale=scale,
        )

        # Reshape output back to [query_len, num_heads, head_size]
        out = out.view(num_heads, query_len, head_size).permute(1, 0, 2)
        output_ref[query_start:query_end].copy_(out)

        query_start = query_end
        key_start = key_end

    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"PyTorch SDPA Time: {(end_time - start_time) * 1000:.2f} ms")
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-6
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


# These tests are optional to only run when explicitly invoked
#
# pytest -v -s --optional \
# tests/kernels/test_prefix_prefill.py::test_contexted_kv_attention_f32
#
# These tests are useful to test model dtype float32 on Turing devices.
# We skip them to not increase the time when running tests on CI
@pytest.mark.optional
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_f32(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
) -> None:
    test_contexted_kv_attention(
        num_heads,
        num_queries_per_kv,
        head_size,
        sliding_window,
        dtype,
        kv_cache_dtype,
        device,
        op,
    )


@pytest.mark.optional
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_alibi_f32(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
) -> None:
    test_contexted_kv_attention_alibi(
        num_heads, num_queries_per_kv, head_size, dtype, kv_cache_dtype, device, op
    )


# Hybrid mamba + full-attention models get a non-power-of-2 attention page.
NONSTANDARD_BLOCK_SIZE_SHAPES = [
    (64, 1, 128, 544),
    (8, 4, 256, 1040),
    (8, 4, 256, 1056),
]


@pytest.mark.parametrize(
    "num_heads,num_queries_per_kv,head_size,block_size", NONSTANDARD_BLOCK_SIZE_SHAPES
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_qwen3_nonstandard_block_size(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    device: str,
    op: Callable,
) -> None:
    """Non-power-of-2 pages must match, even when a tile straddles a page."""
    if not current_platform.is_rocm():
        pytest.skip("Non-power-of-2 block sizes are only exercised on ROCm CI.")

    test_contexted_kv_attention(
        num_heads=num_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        block_size=block_size,
        sliding_window=0,
        dtype=dtype,
        kv_cache_dtype="auto",
        device=device,
        op=op,
    )


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "length,query_len,head_dim,page,causal,window,with_sinks",
    [
        (1, 1, 256, 784, True, 0, False),
        (16, 4, 256, 16, True, 0, True),
        (255, 1, 256, 784, True, 0, False),
        (256, 1, 256, 784, True, 0, True),
        (257, 1, 256, 784, True, 0, True),
        (128, 8, 128, 64, False, 17, True),
        (256, 16, 128, 64, True, 31, False),
        (256, 32, 256, 64, True, 0, False),
        (257, 24, 128, 64, False, 31, True),
        (783, 1, 256, 784, True, 0, False),
        (784, 1, 256, 784, True, 0, False),
        (785, 1, 256, 784, True, 0, False),
        (1024, 1, 256, 784, True, 0, True),
        (2048, 1, 256, 784, True, 256, False),
        (4096, 1, 256, 784, True, 0, False),
        (1024, 4, 256, 832, True, 0, False),
        (1024, 8, 256, 832, True, 0, False),
        (2048, 16, 256, 864, True, 0, False),
        (2048, 24, 256, 912, True, 0, False),
        (4096, 32, 256, 1056, True, 0, False),
        (785, 4, 128, 832, False, 0, False),
        (1024, 8, 128, 832, False, 2047, False),
        (2048, 16, 128, 864, False, 2047, True),
        (4096, 24, 128, 912, False, 2047, False),
        (4096, 32, 128, 1056, True, 2047, False),
        (4096, 32, 128, 1056, False, 17, True),
        (4095, 24, 128, 912, False, 2047, True),
        (4097, 24, 128, 912, False, 2047, False),
        (2048, 1, 256, 784, True, 1, True),
        (4096, 16, 256, 864, True, 31, False),
        (1024, 8, 128, 832, False, 512, True),
    ],
)
@torch.inference_mode()
def test_gfx1151_qwen_partitioned_attention_masks_pages_and_graph(
    batch,
    length,
    query_len,
    head_dim,
    page,
    causal,
    window,
    with_sinks,
    ragged,
    *,
    context_bound=0,
):
    """Single- and multi-partition paths preserve masks and strided caches."""
    if not current_platform.is_rocm():
        pytest.skip("gfx1151 HIP kernel")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1151":
        pytest.skip("gfx1151 HIP kernel")
    from vllm import _custom_ops  # noqa: F401

    torch.manual_seed(42)
    max_seq_len = context_bound or length
    assert max_seq_len >= length
    device, dtype = "cuda", torch.bfloat16
    heads, kv_heads = (24, 4) if head_dim == 256 else (32, 8)
    query_lens = [
        max(1, query_len - b % 3) if ragged else query_len for b in range(batch)
    ]
    seq_lens = [
        max(query_lens[b], length - b * 13) if ragged else length for b in range(batch)
    ]
    if ragged and batch > 1:
        query_lens[-1] = seq_lens[-1] = 0  # A graph-padded request slot.
    offsets = [0]
    for size in query_lens:
        offsets.append(offsets[-1] + size)
    actual_tokens = offsets[-1]
    pages = (length + page - 1) // page
    page_elements = kv_heads * head_dim * page
    storage = torch.full(
        (batch * pages, page_elements * 2 + 128),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    kc = storage[:, :page_elements].view(
        batch * pages, kv_heads, head_dim // 8, page, 8
    )
    vc = storage[:, page_elements : page_elements * 2].view(
        batch * pages, kv_heads, head_dim, page
    )
    table = torch.randperm(batch * pages, device=device).int().view(batch, pages)
    if context_bound:
        padded_table = torch.full(
            (batch, (max_seq_len + page - 1) // page),
            -1,
            device=device,
            dtype=torch.int32,
        )
        padded_table[:, :pages] = table
        table = padded_table
    # A strided Q view, matching projection slices; output is contiguous.
    query = torch.randn(
        batch * query_len, heads, head_dim * 2, device=device, dtype=dtype
    )[..., :head_dim]
    keys = torch.randn(batch, length, kv_heads, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    for b in range(batch):
        for p in range(pages):
            lo = p * page
            count = min(page, seq_lens[b] - lo)
            if count <= 0:
                continue
            physical = int(table[b, p])
            kc[physical, :, :, :count, :] = (
                keys[b, lo : lo + count]
                .view(count, kv_heads, head_dim // 8, 8)
                .permute(1, 2, 0, 3)
            )
            vc[physical, :, :, :count] = values[b, lo : lo + count].permute(1, 2, 0)
    lengths = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    starts = torch.tensor(offsets, device=device, dtype=torch.int32)
    sinks = torch.randn(heads, device=device) if with_sinks else None
    output = torch.full(query.shape, float("nan"), device=device, dtype=dtype)
    workspace = torch.full(
        (batch * query_len, heads, (max_seq_len + 255) // 256, head_dim + 2),
        float("nan"),
        device=device,
        dtype=torch.float32,
    )
    reference = torch.empty_like(output[:actual_tokens], dtype=torch.float32)
    for b in range(batch):
        qlen, slen = query_lens[b], seq_lens[b]
        if not qlen:
            continue
        q = query[offsets[b] : offsets[b + 1]].float()
        k = keys[b, :slen].repeat_interleave(heads // kv_heads, dim=1).float()
        v = values[b, :slen].repeat_interleave(heads // kv_heads, dim=1).float()
        scores = torch.einsum("qhd,khd->hqk", q, k) * head_dim**-0.5
        qpos = torch.arange(slen - qlen, slen, device=device)[:, None]
        kpos = torch.arange(slen, device=device)[None, :]
        valid = torch.ones(qlen, slen, device=device, dtype=torch.bool)
        if causal:
            valid &= kpos <= qpos
        if window:
            valid &= kpos >= qpos - window + 1
        scores.masked_fill_(~valid[None, :, :], float("-inf"))
        if sinks is not None:
            scores = torch.cat([scores, sinks[:, None, None].expand(-1, qlen, 1)], -1)
        probs = scores.softmax(-1)[..., :slen]
        reference[offsets[b] : offsets[b + 1]] = torch.einsum("hqk,khd->qhd", probs, v)

    def run():
        torch.ops._rocm_C.gfx1151_qwen_paged_attention(
            query,
            kc,
            vc,
            table,
            lengths,
            starts,
            sinks,
            output,
            workspace,
            max_seq_len,
            query_len,
            head_dim**-0.5,
            window,
            causal,
        )

    def check():
        result = output[:actual_tokens].float()
        assert torch.isfinite(result).all()
        assert torch.isnan(output[actual_tokens:]).all()
        assert torch.isnan(workspace[actual_tokens:]).all()
        span = min(max_seq_len, window + query_len - 1) if window else max_seq_len
        active_parts = (span + 255) // 256
        if active_parts == 1:
            assert torch.isnan(workspace).all(), "Short path must bypass workspace"
        else:
            assert torch.isnan(workspace[:, :, active_parts:]).all(), (
                "Window-clipped dispatch must not write unused partitions"
            )
        relative_l2 = (result - reference).norm() / reference.norm()
        assert relative_l2 < 0.01
        errors = ~torch.isclose(result, reference, atol=0.01, rtol=0.01)
        assert errors.float().mean() < 0.05

    run()
    check()
    # The generic Triton decode kernel can fault for a zero-query padded slot
    # (reproduced without HIP at B=2, M=8, D=128, page=64, window=17).
    # Compare only real requests there; HIP still receives the padded metadata
    # and must preserve its NaN output/scratch guards against the FP32 reference.
    baseline_requests = batch - int(ragged and batch > 1)
    baseline = torch.empty_like(output)
    scale_tensor = torch.ones((), device=device)
    chunked_prefill_paged_decode(
        query=query,
        key=None,
        value=None,
        output=baseline,
        kv_cache_dtype="auto",
        key_cache=kc,
        value_cache=vc,
        block_table=table[:baseline_requests],
        query_start_loc=starts[: baseline_requests + 1],
        seq_lens=lengths[:baseline_requests],
        max_seq_len=max_seq_len,
        max_query_len=query_len,
        k_scale=scale_tensor,
        v_scale=scale_tensor,
        sliding_window=window,
        sinks=sinks,
        causal=causal,
    )
    assert torch.isfinite(baseline[:actual_tokens]).all()
    assert (
        output[:actual_tokens].float() - baseline[:actual_tokens].float()
    ).norm() / reference.norm() < 0.01
    online_decode = (
        head_dim == 256
        and query_len == 1
        and max_seq_len > 256
        and not window
        and not with_sinks
        and page & (page - 1) != 0
    )
    if online_decode:
        # Independent partition maxima and a different probability-sum tree
        # both passed the relative gate but changed real-model greedy tokens.
        torch.testing.assert_close(
            output[:actual_tokens], baseline[:actual_tokens], atol=0, rtol=0
        )
    graph = torch.cuda.CUDAGraph()
    if context_bound:
        # Capture with a shorter device sequence while retaining the host bound
        # and stable addresses. Replay must read updated lengths, not the values
        # observed during capture. Zero-query graph-padded requests stay empty.
        capture_lengths = torch.tensor(query_lens, device=device, dtype=torch.int32)
        replay_lengths = lengths.clone()
        lengths.copy_(capture_lengths)
    with torch.cuda.graph(graph):
        run()
    if context_bound:
        lengths.copy_(replay_lengths)
    output.fill_(float("nan"))
    workspace.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check()
    if online_decode:
        torch.testing.assert_close(
            output[:actual_tokens], baseline[:actual_tokens], atol=0, rtol=0
        )
    invalid_query = query.as_strided(query.shape, (query.stride(0), query.stride(1), 2))
    with pytest.raises(RuntimeError, match="unsupported query/output layout"):
        torch.ops._rocm_C.gfx1151_qwen_paged_attention(
            invalid_query,
            kc,
            vc,
            table,
            lengths,
            starts,
            sinks,
            output,
            workspace,
            max_seq_len,
            query_len,
            head_dim**-0.5,
            window,
            causal,
        )


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "length,query_len,head_dim,page,causal,window,with_sinks",
    [
        (1, 1, 256, 784, True, 0, False),
        (255, 1, 256, 784, True, 0, True),
        (256, 1, 256, 784, True, 0, False),
        (257, 1, 256, 784, True, 0, False),
        (702, 1, 256, 784, True, 0, False),
        (784, 1, 256, 784, True, 0, False),
        (785, 1, 256, 784, True, 0, False),
        (1023, 1, 256, 784, True, 0, False),
        (769, 4, 256, 832, True, 0, False),
        (769, 8, 256, 832, True, 0, False),
        (769, 16, 256, 864, True, 0, False),
        (769, 24, 256, 912, True, 0, False),
        (769, 32, 256, 1056, True, 0, False),
        (769, 4, 128, 832, False, 2047, False),
        (769, 8, 128, 832, False, 2047, False),
        (769, 16, 128, 864, False, 2047, True),
        (769, 24, 128, 912, False, 2047, False),
        (769, 32, 128, 1056, False, 2047, False),
    ],
)
def test_gfx1151_short_sequence_replay_under_4k_capture_bound(
    batch, length, query_len, head_dim, page, causal, window, with_sinks, ragged
):
    """The production capture bound can exceed the actual replay sequence."""
    test_gfx1151_qwen_partitioned_attention_masks_pages_and_graph(
        batch,
        length,
        query_len,
        head_dim,
        page,
        causal,
        window,
        with_sinks,
        ragged,
        context_bound=4096,
    )
