# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501  (the docstring tables keep one row per line)
"""ROCM_AITER_MLA persistent buffers and scratch: allocated only for the
configurations that read them, and reserved before the runner locks the
workspace after warmup.

Persistent builder state and the configurations that read it:

| Buffer | Allocated when | Read by |
|---|---|---|
| ``paged_kv_indices`` (max_num_seqs x max_model_len int32) | always | asm and Gluon decode (flat per-token page view). Sized per request, not by cache capacity: prefix sharing lets one cache block appear in many requests. |
| ``_mla_work_*`` / ``_mla_reduce_*`` | always | persistent asm decode schedule |
| ``paged_kv_indptr``, ``qo_indptr``, ``_graph_seq_lens`` | full CUDA graphs | captured decode replays |
| ``fp8_ps_*`` + workspace reservation | fp8 KV, bf16, qk/v 192/128, gfx950 | fp8 asm prefill |
| ``_varlen_*`` rows | adaptive verification | varlen verify gather/scatter |
| ``_dcp_verify_buffers``, ``_g_kv_indptr_buf`` | DCP verify under full graphs / cprr | DCP routes (multi-GPU suites) |

Per-forward allocations in the hot path (G2): the decode output ``o``
(``torch.empty``, one per call, the size of the result), a padded query copy
for head counts that are not a multiple of 16 (``get_mla_padded_q``), and the
fp8 prefill scratch, which comes from the workspace manager.
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_mla import mla_shapes as shapes
from tests.v1.attention.backends.rocm_aiter_mla.mla_harness import (
    DEVICE,
    FP8_PREFILL,
    LAYER,
    MlaCase,
    blocks_for,
    watch_case_kernels,
)
from vllm.config import set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.mla import rocm_aiter_mla
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

FULL = CUDAGraphMode.FULL_AND_PIECEWISE


@pytest.mark.parametrize(
    "kwargs, fp8_prefill, full_graph, varlen",
    [
        ({}, False, False, False),
        ({"kv_cache_dtype": "fp8"}, True, False, False),
        ({"kv_cache_dtype": "fp8", "dtype": torch.float16}, False, False, False),
        ({"cudagraph_mode": FULL}, False, True, False),
        (
            {"num_speculative_tokens": 3, "adaptive_verification": True},
            False,
            False,
            True,
        ),
    ],
    ids=["bf16", "fp8", "fp8-fp16", "full-graph", "adaptive"],
)
@pytest.mark.parametrize(
    "shape", [shapes.DSV3_TP8, shapes.GLM47_FLASH_TP1], ids=lambda s: s.id
)
def test_persistent_buffers_only_for_configs_that_read_them(
    shape, kwargs, fp8_prefill, full_graph, varlen
):
    case = MlaCase(shape, **kwargs).setup(256)
    b = case.builder
    fp8_prefill = (
        fp8_prefill
        and rocm_aiter_mla._fp8_mla_prefill_supported()
        and rocm_aiter_mla._fp8_mla_prefill_dims_supported(
            shape.qk_nope_head_dim + shape.qk_rope_head_dim, shape.v_head_dim
        )
    )
    assert b._fp8_prefill_enabled == fp8_prefill
    # The builder owns the decision (it alone knows the model dtype); the impl
    # only takes the asm path when the builder attached its metadata.
    if fp8_prefill:
        assert case.impl._fp8_prefill_enabled
    assert hasattr(b, "fp8_ps_work_info") == fp8_prefill
    assert (b._graph_seq_lens is not None) == full_graph
    assert hasattr(b, "qo_indptr") == full_graph
    assert hasattr(b, "_varlen_q_rows") == varlen
    # Single-GPU configs never allocate the DCP verify state.
    assert b._dcp_verify_buffers is None
    assert b._g_kv_indptr_buf is None
    # One page per token of every request; adaptive verification pads each
    # request's page list by its draft count.
    pages_per_req = case.max_model_len + (case.num_speculative_tokens if varlen else 0)
    assert b.paged_kv_indices.numel() == case.max_num_seqs * pages_per_req


@pytest.mark.parametrize(
    "shape", [shapes.DSV3_TP8, shapes.KIMI_K3_TP8], ids=lambda s: s.id
)
def test_fp8_prefill_scratch_is_reserved_before_the_lock(spy, fp8_asm_prefill, shape):
    """The runner locks the workspace after warmup; the largest prefill the
    scheduler can hand the layer must then fit the builder's reservation."""
    budget = 8192
    seq_lens = [budget // 4] * 4
    case = MlaCase(
        shape,
        kv_cache_dtype="fp8",
        max_num_seqs=16,
        max_model_len=budget,
        max_num_batched_tokens=budget,
    ).setup(blocks_for(seq_lens, 16))
    watch_case_kernels(spy, case)
    lock_workspace()
    try:
        batch = case.batch(seq_lens, seq_lens)
        query = case.query(batch.num_tokens)
        metadata = case.build(batch)
        case.check(batch, query, case.run(batch, query, metadata), metadata)
    finally:
        unlock_workspace()
    assert FP8_PREFILL in spy.names()


def test_builder_persistent_memory_budget():
    """DeepSeek-V3 at TP8 with a 16k context and 256 sequences under full
    graphs: the flat page index (16 MiB), the decode schedule and the graph
    buffers stay under 64 MiB besides the chunked-prefill workspace."""
    case = MlaCase(shapes.DSV3_TP8, cudagraph_mode=FULL).setup(16)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with set_current_vllm_config(case.config):
        builder = AiterMLAMetadataBuilder(case.spec, [LAYER], case.config, DEVICE)
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() - before
    workspace = builder.chunked_prefill_workspace
    used -= workspace.numel() * workspace.element_size()
    assert used <= 64 * 2**20, f"{used / 2**20:.1f} MiB"
