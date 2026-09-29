# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for attention backend selectors."""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.gfx1151_qwen_attn import _verification_tile
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

# Measured verification tile for width 4/8/16 (see _verification_tile docstring).
QWEN_VERIFY_TILE = _verification_tile()

# ROCm-specific attention backend selection tests
pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


@pytest.fixture
def gfx1151_config(monkeypatch):
    from vllm.v1.attention.backends import gfx1151_qwen_attn as backend

    monkeypatch.setenv("VLLM_GFX1151_QWEN_ATTENTION", "1")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx1151"),
    )
    monkeypatch.setattr(
        torch.ops._rocm_C, "gfx1151_qwen_paged_attention", MagicMock(), raising=False
    )
    hf = SimpleNamespace(
        model_type="qwen3_5_text",
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, hf_text_config=hf, max_model_len=4096
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        attention_config=SimpleNamespace(backend=None, backend_per_kind={}),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        speculative_config=None,
        kv_transfer_config=None,
    )
    monkeypatch.setattr(backend, "get_current_vllm_config", lambda: config)
    return config


@pytest.mark.parametrize(
    "unsupported",
    [
        "none",
        "disabled",
        "arch",
        "fp16",
        "fp8_cache",
        "tp",
        "pp",
        "pcp",
        "dcp",
        "model",
        "heads",
        "encoder",
        "override",
        "connector",
        "missing_op",
    ],
)
def test_gfx1151_model_selection_falls_back(gfx1151_config, monkeypatch, unsupported):
    from vllm.model_executor.models.gfx1151_qwen3_5_attention import (
        Gfx1151Qwen3_5Attention,
        qwen_attention_cls,
    )
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    config = gfx1151_config
    hf = config.model_config.hf_text_config
    if unsupported == "disabled":
        monkeypatch.setenv("VLLM_GFX1151_QWEN_ATTENTION", "0")
    elif unsupported == "arch":
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda device: SimpleNamespace(gcnArchName="gfx1100"),
        )
    elif unsupported == "fp16":
        config.model_config.dtype = torch.float16
    elif unsupported == "fp8_cache":
        config.cache_config.cache_dtype = "fp8"
    elif unsupported in ("tp", "pp", "pcp", "dcp"):
        field = dict(
            tp="tensor_parallel_size",
            pp="pipeline_parallel_size",
            pcp="prefill_context_parallel_size",
            dcp="decode_context_parallel_size",
        )
        setattr(config.parallel_config, field[unsupported], 2)
    elif unsupported == "model":
        hf.model_type = "llama"
    elif unsupported == "heads":
        hf.num_key_value_heads = 8
    elif unsupported == "encoder":
        hf.is_causal = False
    elif unsupported == "override":
        config.attention_config.backend = AttentionBackendEnum.ROCM_ATTN
    elif unsupported == "connector":
        config.kv_transfer_config = object()
    elif unsupported == "missing_op":
        monkeypatch.setattr(torch.ops, "_rocm_C", SimpleNamespace())
    expected = Gfx1151Qwen3_5Attention if unsupported == "none" else Qwen3NextAttention
    assert qwen_attention_cls(config, hf) is expected


def test_gfx1151_draft_selection_requires_dflash2_geometry(gfx1151_config):
    from vllm.model_executor.models.gfx1151_qwen3_5_attention import (
        Gfx1151DFlash2Attention,
        qwen_attention_cls,
    )
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Attention

    hf = SimpleNamespace(
        architectures=["DFlash2DraftModel"],
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    assert qwen_attention_cls(gfx1151_config, hf, draft=True) is Gfx1151DFlash2Attention
    hf.architectures = ["DFlashDraftModel"]
    assert qwen_attention_cls(gfx1151_config, hf, draft=True) is DFlashQwen3Attention


@pytest.mark.parametrize(
    "unsupported",
    [
        "none",
        "disabled",
        "master_disabled",
        "geometry",
        "activation",
        "override",
        "missing_op",
    ],
)
def test_gfx1151_gdn_model_selection(gfx1151_config, monkeypatch, unsupported):
    from vllm.model_executor.models import gfx1151_qwen3_5_attention as model

    monkeypatch.setenv("VLLM_GFX1151_QWEN_GDN", "1")
    monkeypatch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    monkeypatch.setattr(
        torch.ops._rocm_C, "gfx1151_qwen_gdn_decode_core", MagicMock(), raising=False
    )
    config = gfx1151_config.model_config.hf_text_config
    config.hidden_act = "silu"
    for name, value in (
        ("linear_num_key_heads", 16),
        ("linear_num_value_heads", 48),
        ("linear_key_head_dim", 128),
        ("linear_value_head_dim", 128),
        ("linear_conv_kernel_dim", 4),
    ):
        setattr(config, name, value)
    if unsupported == "disabled":
        monkeypatch.setenv("VLLM_GFX1151_QWEN_GDN", "0")
    elif unsupported == "master_disabled":
        monkeypatch.setenv("VLLM_GFX1151_QWEN_ATTENTION", "0")
    elif unsupported == "geometry":
        config.linear_num_value_heads = 32
    elif unsupported == "activation":
        config.hidden_act = "relu"
    elif unsupported == "override":
        monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    elif unsupported == "missing_op":
        monkeypatch.setattr(
            torch.ops, "_rocm_C", SimpleNamespace(gfx1151_qwen_paged_attention=object())
        )
    expected = (
        model.Gfx1151QwenGatedDeltaNetAttention
        if unsupported == "none"
        else model.QwenGatedDeltaNetAttention
    )
    assert model.qwen_gdn_cls(gfx1151_config, config) is expected


@pytest.mark.parametrize(
    "case",
    [
        "decode",
        "spec",
        "spec8",
        "spec16",
        "spec24",
        "spec32",
        "sd",
        "padding",
        "disabled",
        "prefill",
        "mixed",
        "missing_metadata",
        "missing_indices",
        "missing_accepted",
        "fp16",
        "state_dtype",
        "conv_dtype",
        "conv_capacity",
        "conv_stride",
        "a_stride",
        "metadata_dtype",
        "large_batch",
        "output_stride",
        "dt_bias_dtype",
    ],
)
def test_gfx1151_gdn_dispatch_contract(gfx1151_config, monkeypatch, case):
    """Exercise real dispatch/stride guards without launching GPU work."""
    from vllm.model_executor.models import gfx1151_qwen3_5_attention as model
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    spec = case.startswith("spec") or case in ("mixed", "missing_accepted")
    width = (
        int(case[4:]) if case.startswith("spec") and case[4:] else (4 if spec else 1)
    )
    batch = 9 if case == "large_batch" else 2
    tokens = batch * width + (2 if case == "padding" else 0)
    layer = object.__new__(model.Gfx1151QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer._gfx1151_gdn_enabled = case != "disabled"
    layer._gfx1151_accepted = torch.ones(8, dtype=torch.int32)
    layer.prefix = "model.layers.0.linear_attn"
    layer.head_k_dim = 128
    layer.activation = "silu"
    layer.A_log = torch.empty(48, dtype=torch.float32)
    layer.dt_bias = torch.empty(48, dtype=torch.bfloat16)
    layer.conv1d = SimpleNamespace(
        weight=torch.empty(10240, 1, 4, dtype=torch.bfloat16), bias=None
    )
    conv = torch.empty(2, 10240, 34, dtype=torch.bfloat16)
    state = torch.empty(2, 48, 128, 128, dtype=torch.float32)
    packed = torch.empty(tokens, 16384, dtype=torch.bfloat16)
    mixed = packed[:, :10240]
    b, a = torch.empty(tokens, 96, dtype=torch.bfloat16).chunk(2, -1)
    output = torch.empty(tokens, 48, 128, dtype=torch.bfloat16)
    indices = torch.ones(batch, width, dtype=torch.int32)
    starts = torch.arange(batch + 1, dtype=torch.int32) * width
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0 if spec else batch,
        num_decode_tokens=0 if spec else batch,
        num_spec_decodes=batch if spec else 0,
        num_spec_decode_tokens=tokens if spec else 0,
        num_actual_tokens=tokens,
        spec_sequence_masks=torch.ones(batch, dtype=torch.bool) if spec else None,
        spec_state_indices_tensor=indices if spec else None,
        spec_query_start_loc=starts if spec else None,
        non_spec_state_indices_tensor=indices[:, 0] if not spec else None,
        non_spec_query_start_loc=starts if not spec else None,
        num_accepted_tokens=torch.ones(batch, dtype=torch.int32) if spec else None,
    )
    if case == "sd":
        conv = torch.empty(2, 34, 10240, dtype=torch.bfloat16)
    elif case == "prefill":
        metadata.num_prefills = 1
    elif case == "mixed":
        metadata.num_decodes = 1
    elif case == "missing_metadata":
        metadata = None
    elif case == "missing_indices":
        metadata.non_spec_state_indices_tensor = None
    elif case == "missing_accepted":
        metadata.num_accepted_tokens = None
    elif case == "fp16":
        mixed = mixed.half()
    elif case == "state_dtype":
        state = state.half()
    elif case == "conv_dtype":
        conv = conv.float()
    elif case == "conv_capacity":
        conv = conv[..., :2]
    elif case == "conv_stride":
        conv = torch.empty(2, 10240, 68, dtype=torch.bfloat16)[..., ::2]
    elif case == "a_stride":
        a = torch.empty(tokens, 96, dtype=torch.bfloat16)[:, ::2]
    elif case == "metadata_dtype":
        metadata.non_spec_query_start_loc = starts.long()
    elif case == "output_stride":
        output = torch.empty(tokens, 48, 256, dtype=torch.bfloat16)[..., ::2]
    elif case == "dt_bias_dtype":
        layer.dt_bias = layer.dt_bias.half()
    layer.kv_cache = (conv, state)
    monkeypatch.setattr(model, "is_conv_state_dim_first", lambda: case != "sd")
    monkeypatch.setattr(
        model,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
    )
    launch = MagicMock()
    monkeypatch.setattr(model.ops, "gfx1151_qwen_gdn_decode_core", launch)
    with (
        patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ),
        patch.object(
            torch.Tensor, "item", side_effect=AssertionError("device synchronization")
        ),
        patch.object(torch, "ones", side_effect=AssertionError("forward allocation")),
        patch.object(model.QwenGatedDeltaNetAttention, "_forward_core") as fallback,
    ):
        layer._forward_core(mixed, b, a, output)
    if case in ("decode", "sd", "padding"):
        fallback.assert_not_called()
        launch.assert_called_once()
        args = launch.call_args.args
        assert args[0].shape == (tokens, 10240)
        assert args[5].shape == (batch, width)
        assert args[6].shape == (batch + 1,)
        assert args[7].shape == (batch,)
        assert args[11].shape == (2, 10240, 34)
        assert args[12].shape == (10240, 4)
        assert args[9].data_ptr() == output.data_ptr()
        if not spec:
            assert args[7].data_ptr() == layer._gfx1151_accepted.data_ptr()
    else:
        # Even otherwise valid speculative metadata must retain Triton: the
        # numerically correct fused HIP verification kernel is not faster yet.
        launch.assert_not_called()
        fallback.assert_called_once_with(mixed, b, a, output)


@pytest.mark.parametrize(
    "conv_dtype,state_dtype,interleaved,enabled",
    [
        (torch.bfloat16, torch.float32, False, True),
        (torch.bfloat16, torch.bfloat16, False, True),
        (torch.float32, torch.float32, False, False),
        (torch.bfloat16, torch.float16, False, False),
        (torch.bfloat16, torch.float32, True, False),
    ],
)
def test_gfx1151_gdn_initialization_preserves_fallback(
    monkeypatch, conv_dtype, state_dtype, interleaved, enabled
):
    """Keep the original forward/norm mode and allocate metadata only once."""
    from vllm.model_executor.models import gfx1151_qwen3_5_attention as model

    def initialize(layer, **kwargs):
        torch.nn.Module.__init__(layer)
        layer.gqa_interleaved_layout = interleaved
        layer.norm = SimpleNamespace(weight=torch.empty(128, dtype=torch.bfloat16))
        layer.enable_fused_gdn_decode = False
        layer.gdn_decode_kernel = "triton"

    monkeypatch.setattr(model.QwenGatedDeltaNetAttention, "__init__", initialize)
    monkeypatch.setattr(
        model.QwenGatedDeltaNetAttention,
        "get_state_dtype",
        lambda self: (conv_dtype, state_dtype),
    )
    monkeypatch.setattr(
        model, "qwen_gdn_cls", lambda *args: model.Gfx1151QwenGatedDeltaNetAttention
    )
    layer = model.Gfx1151QwenGatedDeltaNetAttention(None, None)
    assert layer._gfx1151_gdn_enabled is enabled
    assert layer.enable_fused_gdn_decode is False
    assert layer.gdn_decode_kernel == "triton"
    if enabled:
        assert torch.equal(layer._gfx1151_accepted, torch.ones(8, dtype=torch.int32))
        assert "_gfx1151_accepted" not in layer.state_dict()
    with (
        patch.object(model.QwenGatedDeltaNetAttention, "forward_cuda") as packed,
        patch.object(model.QwenGatedDeltaNetAttention, "forward_hip") as original,
    ):
        hidden = torch.empty(1, 1)
        result = layer.forward_hip(hidden)
    if enabled:
        packed.assert_called_once_with(hidden)
        original.assert_not_called()
        assert result is packed.return_value
    else:
        original.assert_called_once_with(hidden)
        packed.assert_not_called()
        assert result is original.return_value


def test_gfx1151_gdn_profile_retains_prefill_warmup(monkeypatch):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as base
    from vllm.model_executor.models import gfx1151_qwen3_5_attention as model

    layer = object.__new__(model.Gfx1151QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.linear_attn"
    layer._gfx1151_gdn_enabled = True
    layer._warmup_prefill_kernels = MagicMock()
    for module in (model, base):
        monkeypatch.setattr(
            module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
        )
    mixed, output = torch.empty(1, 10240), torch.zeros(1, 48, 128)
    ba = torch.empty(1, 48)
    layer._forward_core(mixed, ba, ba, output)
    layer._warmup_prefill_kernels.assert_called_once_with(mixed, 0)
    assert not torch.count_nonzero(output)


@pytest.mark.parametrize(
    "case",
    [
        "supported",
        "draft",
        "draft_16",
        "draft_24",
        "draft_32",
        "target_24",
        "large_query",
        "short",
        "long",
        "fp16",
        "q_stride",
        "cache_stride",
        "output_stride",
        "metadata",
        "alibi",
        "quant_output",
        "sinks_dtype",
    ],
)
def test_gfx1151_forward_dispatch_preserves_fallback(gfx1151_config, monkeypatch, case):
    """CPU layout/dispatch test: only device detection and GPU launch are mocked."""
    from vllm.v1.attention.backends import gfx1151_qwen_attn as backend

    draft = case == "draft" or case.startswith("draft_")
    heads, kv_heads, dim = (32, 8, 128) if draft else (24, 4, 256)
    qlen = int(case.split("_")[1]) if case.startswith("draft_") else (8 if draft else 1)
    if case == "target_24":
        qlen = 24
    if case == "large_query":
        qlen = 64
    impl = backend.Gfx1151QwenAttentionImpl(
        heads, dim, dim**-0.5, kv_heads, None, None, "auto"
    )
    query = torch.empty(qlen, heads, dim, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    cache = torch.empty(2, 2, 912, kv_heads * dim, dtype=torch.bfloat16)
    metadata = backend.Gfx1151QwenAttentionMetadata(
        num_actual_tokens=qlen,
        max_query_len=qlen,
        query_start_loc=torch.tensor([0, qlen], dtype=torch.int32),
        max_seq_len=1024,
        seq_lens=torch.tensor([1024], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.zeros(qlen, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        num_partitions=4,
    )
    output_scale = None
    if case == "short":
        metadata.max_seq_len = 512
    elif case == "long":
        metadata.max_seq_len = 8192
    elif case == "fp16":
        query = query.half()
    elif case == "q_stride":
        query = torch.empty(qlen, heads, dim * 2, dtype=query.dtype)[..., ::2]
    elif case == "cache_stride":
        cache = cache[..., ::2]
    elif case == "output_stride":
        output = torch.empty(qlen, heads, dim * 2, dtype=query.dtype)[..., ::2]
    elif case == "metadata":
        metadata.query_start_loc = metadata.query_start_loc.long()
    elif case == "alibi":
        impl.alibi_slopes = torch.ones(heads)
    elif case == "quant_output":
        output_scale = torch.ones(())
    elif case == "sinks_dtype":
        impl.sinks = torch.zeros(heads, dtype=torch.bfloat16)
    manager = MagicMock()
    manager.get_simultaneous.side_effect = lambda spec: [
        torch.empty(*spec[0], dtype=spec[1])
    ]
    monkeypatch.setattr(backend, "is_workspace_manager_initialized", lambda: True)
    monkeypatch.setattr(backend, "current_workspace_manager", lambda: manager)
    launch = MagicMock()
    monkeypatch.setattr(backend.ops, "gfx1151_qwen_paged_attention", launch)
    with (
        patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ),
        patch.object(
            backend.RocmAttentionImpl, "forward", return_value=output
        ) as fallback,
    ):
        result = impl.forward(
            None, query, None, None, cache, metadata, output, output_scale
        )
    assert result is output
    if case == "supported":
        fallback.assert_not_called()
        launch.assert_called_once()
        manager.get_simultaneous.assert_called_once()
        args = launch.call_args.args
        assert args[1].stride() == (2 * 912 * kv_heads * dim, 912 * dim, 912 * 8, 8, 1)
        assert args[2].stride() == (2 * 912 * kv_heads * dim, 912 * dim, 912, 1)
        assert args[8].shape == (qlen, heads, 4, dim + 2)
    else:
        fallback.assert_called_once()
        launch.assert_not_called()
        manager.get_simultaneous.assert_not_called()


@pytest.mark.parametrize(
    "case,batch,qlen,expected",
    [
        ("supported", 1, 24, (16, 64, 4)),
        ("supported", 2, 24, (32, 64, 4)),
        ("supported", 4, 24, (32, 64, 4)),
        ("supported", 8, 24, (32, 64, 4)),
        ("supported", 1, 1024, (64, 64, 8)),
        ("supported", 8, 1024, (64, 64, 8)),
        ("supported", 1, 4096, (64, 64, 8)),
        ("short_query", 1, 256, None),
        ("large_batch", 9, 24, None),
        ("long_context", 1, 24, None),
        ("disabled", 1, 24, None),
        ("noncausal", 1, 24, None),
        ("sliding", 1, 24, None),
        ("sinks", 1, 24, None),
        ("missing_key", 1, 24, None),
        ("missing_value", 1, 24, None),
        ("key_dtype", 1, 24, None),
        ("value_stride", 1, 24, None),
        ("short_key", 1, 24, None),
        ("query_dtype", 1, 24, None),
        ("output_dtype", 1, 24, None),
        ("quant_output", 1, 24, None),
        ("table_dtype", 1, 24, None),
        ("empty", 1, 24, None),
        ("cascade", 1, 24, None),
        ("missing_key", 2, 8, QWEN_VERIFY_TILE),
        ("missing_value", 2, 8, QWEN_VERIFY_TILE),
        ("noncausal", 2, 8, None),
        ("sliding", 2, 8, None),
        ("sinks", 2, 8, None),
    ]
    + [("supported", b, q, QWEN_VERIFY_TILE) for b in (1, 2, 4, 8) for q in (4, 8, 16)]
    + [("supported", b, 32, (16 if b == 1 else 32, 64, 4)) for b in (1, 2, 4, 8)]
    + [("missing_key", 1, 32, (16, 64, 4)), ("missing_value", 2, 32, (32, 64, 4))]
    + [
        ("draft_supported", b, q, (16 if q <= 16 else 32, 64, 8))
        for b in (1, 2, 4, 8)
        for q in (4, 8, 16, 24, 32)
    ]
    + [("draft_missing_key", 1, 24, (32, 64, 8))]
    + [("draft_missing_value", 1, 32, (32, 64, 8))]
    + [
        ("draft_" + case, 1, 8, None)
        for case in (
            "causal",
            "full_window",
            "other_window",
            "sinks",
            "disabled",
            "long_context",
            "query_dtype",
            "output_dtype",
            "quant_output",
            "table_dtype",
            "empty",
            "cascade",
        )
    ]
    + [("draft_supported", 1, 64, None), ("draft_supported", 9, 8, None)],
)
def test_gfx1151_prefix_dispatch_contract(
    gfx1151_config, monkeypatch, case, batch, qlen, expected
):
    """Select measured tiles only for supported contracts, without HIP scratch."""
    from vllm.v1.attention.backends import gfx1151_qwen_attn as backend

    draft = case.startswith("draft_")
    case = case.removeprefix("draft_")
    heads, kv_heads, dim = (32, 8, 128) if draft else (24, 4, 256)
    impl = backend.Gfx1151QwenAttentionImpl(
        heads, dim, dim**-0.5, kv_heads, None, None, "auto"
    )
    if draft:
        impl.sliding_window = (2047, 0)
    tokens, page, length = batch * qlen, 912, 4096
    query = torch.empty(tokens, heads, dim, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    key = torch.empty(tokens, kv_heads, dim * 2, dtype=query.dtype)[..., :dim]
    value = torch.empty_like(key)
    cache = torch.empty(5, 2, page, 1024, dtype=query.dtype)
    metadata = backend.Gfx1151QwenAttentionMetadata(
        num_actual_tokens=tokens,
        max_query_len=qlen,
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32) * qlen,
        max_seq_len=length,
        seq_lens=torch.full((batch,), length, dtype=torch.int32),
        block_table=torch.arange(5, dtype=torch.int32).repeat(batch, 1),
        slot_mapping=torch.zeros(tokens, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        num_partitions=16,
        causal=not draft,
    )
    output_scale = None
    if case == "long_context":
        metadata.max_seq_len = 8192
        metadata.num_partitions = 32
    elif case == "disabled":
        impl._enabled = False
    elif case == "noncausal":
        metadata.causal = False
    elif case == "sliding":
        impl.sliding_window = (2047, 0)
    elif case == "sinks":
        impl.sinks = torch.zeros(heads)
    elif case == "causal":
        metadata.causal = True
    elif case == "full_window":
        impl.sliding_window = (-1, -1)
    elif case == "other_window":
        impl.sliding_window = (1023, 0)
    elif case == "missing_key":
        key = None
    elif case == "missing_value":
        value = None
    elif case == "key_dtype":
        key = key.half()
    elif case == "value_stride":
        value = torch.empty(tokens, 4, 512, dtype=query.dtype)[..., ::2]
    elif case == "short_key":
        key = key[:-1]
    elif case == "query_dtype":
        query = query.half()
    elif case == "output_dtype":
        output = output.half()
    elif case == "quant_output":
        output_scale = torch.ones(())
    elif case == "table_dtype":
        metadata.block_table = metadata.block_table.long()
    elif case == "empty":
        metadata.num_actual_tokens = 0
    elif case == "cascade":
        metadata.use_cascade = True
    layer = SimpleNamespace(_k_scale=torch.ones(()), _v_scale=torch.ones(()))
    launch, hip, workspace = MagicMock(), MagicMock(), MagicMock()
    splitkv = MagicMock()
    monkeypatch.setattr(backend, "gfx1151_qwen_prefill", launch)
    monkeypatch.setattr(backend.ops, "gfx1151_qwen_paged_attention", hip)
    monkeypatch.setattr(backend, "splitkv_verify_attention", splitkv)
    monkeypatch.setattr(backend, "is_workspace_manager_initialized", lambda: False)
    monkeypatch.setattr(backend, "current_workspace_manager", workspace)
    with (
        patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ),
        patch.object(
            backend.RocmAttentionImpl, "forward", return_value=output
        ) as fallback,
    ):
        result = impl.forward(
            layer, query, key, value, cache, metadata, output, output_scale
        )
    assert result is output
    hip.assert_not_called()
    workspace.assert_not_called()
    # Mirror the backend's split-KV gate: the GQA-packed kernel is tried before
    # the prefix tile and returns early, so those cases never reach `launch`.
    splitkv_expected = (
        expected is not None
        and not draft
        and qlen in (4, 8, 16, 32)
        and batch <= 2
        and backend.envs.VLLM_GFX1151_QWEN_VERIFY_SPLITKV
        and backend.splitkv_supports(
            heads,
            kv_heads,
            dim,
            qlen,
            metadata.max_seq_len,
            metadata.causal,
            max(0, impl.sliding_window[0]),
            impl.sinks,
            query.dtype,
        )
    )
    if splitkv_expected:
        fallback.assert_not_called()
        launch.assert_not_called()
        splitkv.assert_called_once()
        assert splitkv.call_args.kwargs == dict(
            max_query_len=qlen,
            max_seq_len=metadata.max_seq_len,
            sm_scale=impl.scale,
        )
        args = splitkv.call_args.args
        assert args[0].data_ptr() == query.data_ptr()
        assert args[0].shape == (tokens, heads, dim)
        assert args[3] is metadata.block_table
        assert args[4] is metadata.seq_lens
        assert args[5] is metadata.query_start_loc
        assert args[6].data_ptr() == output.data_ptr()
    elif expected is None:
        fallback.assert_called_once()
        launch.assert_not_called()
        splitkv.assert_not_called()
    else:
        fallback.assert_not_called()
        launch.assert_called_once()
        splitkv.assert_not_called()
        args = launch.call_args.args
        assert args[-1] == expected
        assert args[0].data_ptr() == query.data_ptr()
        if draft or qlen in (4, 8, 16, 32):
            assert args[1] is None and args[2] is None
        else:
            assert args[1].stride() == key.stride()
            assert args[2].data_ptr() == value.data_ptr()
        assert args[6] is metadata.block_table
        assert args[7] is metadata.query_start_loc
        assert args[8] is metadata.seq_lens
        assert args[11] is layer._k_scale and args[12] is layer._v_scale
        assert args[13] == impl.scale
        assert launch.call_args.kwargs == dict(
            causal=not draft, window=2047 if draft else 0
        )


@pytest.mark.parametrize("override", [None, 4096, 32768])
@pytest.mark.parametrize("max_model_len", [4096, 8192])
def test_gfx1151_max_context_cap_and_scratch(
    gfx1151_config, monkeypatch, max_model_len, override
):
    """The context cap gates the tuned paths and sizes the reserved scratch.

    CUDA-graph capture runs with max_seq_len = max_model_len, so a cap below
    --max-model-len silently disables every tuned gfx1151 attention path.
    VLLM_GFX1151_QWEN_MAX_CONTEXT raises it; it must never lower the delivered
    4096, and the reservation must scale with min(max_model_len, cap) so that
    raising the cap is a no-op for a 4096-context run - the configuration every
    published gfx1151 number was measured with.
    """
    from vllm.v1.attention.backends import gfx1151_qwen_attn as backend

    if override is None:
        monkeypatch.delenv("VLLM_GFX1151_QWEN_MAX_CONTEXT", raising=False)
    else:
        monkeypatch.setenv("VLLM_GFX1151_QWEN_MAX_CONTEXT", str(override))
    gfx1151_config.model_config.max_model_len = max_model_len
    gfx1151_config.scheduler_config.max_num_seqs = 8
    impl = backend.Gfx1151QwenAttentionImpl(24, 256, 256**-0.5, 4, None, None, "auto")

    cap = max(backend.MAX_CONTEXT, override or 0)
    assert impl._max_context == cap
    partitions = (
        min(max_model_len, cap) + backend.PARTITION_SIZE - 1
    ) // backend.PARTITION_SIZE
    assert impl._workspace_shape[2] == partitions
    assert impl._workspace_shape[0] == min(8, backend.MAX_BATCH) * backend.MAX_QUERY
    if max_model_len <= backend.MAX_CONTEXT:
        # Unchanged by the override: the measured configuration stays identical.
        assert partitions == max_model_len // backend.PARTITION_SIZE
    else:
        assert partitions == (
            (max_model_len + backend.PARTITION_SIZE - 1) // backend.PARTITION_SIZE
            if override and override >= max_model_len
            else backend.MAX_CONTEXT // backend.PARTITION_SIZE
        )


def test_gfx1151_profile_reserves_scratch_before_capture(gfx1151_config, monkeypatch):
    from vllm.v1.attention.backends import gfx1151_qwen_attn as backend

    impl = backend.Gfx1151QwenAttentionImpl(24, 256, 256**-0.5, 4, None, None, "auto")
    manager = MagicMock()
    monkeypatch.setattr(backend, "is_workspace_manager_initialized", lambda: True)
    monkeypatch.setattr(backend, "current_workspace_manager", lambda: manager)
    output = torch.ones(1, 24, 256)
    impl.forward(None, output, None, None, torch.empty(0), None, output)
    manager.get_simultaneous.assert_called_once_with(
        ((256, 24, 16, 258), torch.float32)
    )
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("max_model_len", [4096, 8192])
@pytest.mark.parametrize("draft", [False, True])
def test_gfx1151_capture_dispatch_and_locked_workspace(
    gfx1151_config, monkeypatch, max_model_len, draft
):
    """Capture-time dispatch must fit reserved scratch without growing it.

    Use the actual CPU workspace manager and metadata builder. Device detection
    and native launches are mocked; this is not a GPU graph/numerics test.
    """
    from vllm.v1.attention.backends import gfx1151_qwen_attn as backend
    from vllm.v1.worker.workspace import WorkspaceManager

    gfx1151_config.model_config.max_model_len = max_model_len
    gfx1151_config.scheduler_config.max_num_seqs = 8
    heads, kv_heads, dim = (32, 8, 128) if draft else (24, 4, 256)
    impl = backend.Gfx1151QwenAttentionImpl(
        heads, dim, dim**-0.5, kv_heads, None, None, "auto"
    )
    if draft:
        impl.sliding_window = (2047, 0)
    manager = WorkspaceManager(torch.device("cpu"))
    empty_cache = MagicMock()
    monkeypatch.setattr(torch.accelerator, "empty_cache", empty_cache)
    monkeypatch.setattr(backend, "is_workspace_manager_initialized", lambda: True)
    monkeypatch.setattr(backend, "current_workspace_manager", lambda: manager)
    profile_output = torch.ones(1, heads, dim, dtype=torch.bfloat16)
    impl.forward(None, profile_output, None, None, torch.empty(0), None, profile_output)
    empty_cache.assert_called_once()
    manager.lock()
    (reserved,) = manager.get_simultaneous((impl._workspace_shape, torch.float32))
    pointer = reserved.data_ptr()
    launch = MagicMock()
    prefix_launch = MagicMock()
    splitkv_launch = MagicMock()
    monkeypatch.setattr(backend.ops, "gfx1151_qwen_paged_attention", launch)
    monkeypatch.setattr(backend, "gfx1151_qwen_prefill", prefix_launch)
    monkeypatch.setattr(backend, "splitkv_verify_attention", splitkv_launch)
    builder = object.__new__(backend.Gfx1151QwenAttentionMetadataBuilder)
    builder.device = torch.device("cpu")

    # Capture uses a static max_seq_len, not the short dummy device lengths.
    # Eager calls subsequently obtain fresh metadata with the actual bound.
    widths = (4, 8, 16, 24, 32) if draft else (1, 4, 8, 16, 24, 32)
    cases = [(True, max_model_len, 8, width) for width in widths]
    cases += [
        (False, length, batch, width)
        for length in (1024, 2048, 3072, 4096)
        for batch in (1, 2, 4, 8)
        for width in widths
    ]
    for capture, length, batch, width in cases:
        tokens = batch * width
        page = 912
        num_pages = (length + page - 1) // page
        cache = torch.empty(num_pages, 2, page, kv_heads * dim, dtype=torch.bfloat16)
        query = torch.empty(tokens, heads, dim, dtype=torch.bfloat16)
        output = torch.empty_like(query)
        key = None if draft else torch.empty(tokens, kv_heads, dim, dtype=query.dtype)
        value = None if draft else torch.empty_like(key)
        layer = SimpleNamespace(_k_scale=torch.ones(()), _v_scale=torch.ones(()))
        common = SimpleNamespace(
            num_actual_tokens=tokens,
            max_query_len=width,
            max_seq_len=length,
            query_start_loc=torch.arange(batch + 1, dtype=torch.int32) * width,
            seq_lens=torch.full((batch,), length, dtype=torch.int32),
            block_table_tensor=torch.arange(num_pages, dtype=torch.int32).repeat(
                batch, 1
            ),
            slot_mapping=torch.arange(tokens),
            causal=not draft,
        )
        metadata = (
            builder.build_for_cudagraph_capture(common)
            if capture
            else builder.build(0, common)
        )
        launch.reset_mock()
        prefix_launch.reset_mock()
        splitkv_launch.reset_mock()
        with (
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            ),
            patch.object(
                backend.RocmAttentionImpl, "forward", return_value=output
            ) as fallback,
        ):
            impl.forward(layer, query, key, value, cache, metadata, output)
        # Mirror the backend's split-KV gate so the expected dispatch is derived
        # from the same conditions rather than restated as a literal.
        splitkv_expected = (
            not draft
            and width in (4, 8, 16, 32)
            and batch <= 2
            and backend.envs.VLLM_GFX1151_QWEN_VERIFY_SPLITKV
            and backend.splitkv_supports(
                heads,
                kv_heads,
                dim,
                width,
                length,
                not draft,
                max(0, impl.sliding_window[0]),
                impl.sinks,
                query.dtype,
            )
        )
        if length > impl._max_context:
            fallback.assert_called_once()
            launch.assert_not_called()
            prefix_launch.assert_not_called()
            splitkv_launch.assert_not_called()
        elif splitkv_expected:
            # GQA-packed split-KV verification is tried before the prefix tile
            # and returns early, so neither other launcher may fire.
            fallback.assert_not_called()
            launch.assert_not_called()
            prefix_launch.assert_not_called()
            splitkv_launch.assert_called_once()
            assert splitkv_launch.call_args.kwargs == dict(
                max_query_len=width, max_seq_len=length, sm_scale=impl.scale
            )
            args = splitkv_launch.call_args.args
            assert args[0].shape == (tokens, heads, dim)
            assert args[6].shape == (tokens, heads, dim)
            assert args[4].shape == (batch,)
        elif draft or width in (4, 8, 16, 24, 32):
            fallback.assert_not_called()
            launch.assert_not_called()
            splitkv_launch.assert_not_called()
            prefix_launch.assert_called_once()
            args = prefix_launch.call_args.args
            assert args[9:11] == (length, width)
            if draft:
                expected = (16 if width <= 16 else 32, 64, 8)
            elif width <= 16:
                expected = QWEN_VERIFY_TILE
            else:
                expected = (16 if batch == 1 else 32, 64, 4)
            assert args[-1] == expected
            assert prefix_launch.call_args.kwargs == dict(
                causal=not draft, window=2047 if draft else 0
            )
            if draft or width != 24:
                assert args[1] is None and args[2] is None
        else:
            fallback.assert_not_called()
            launch.assert_called_once()
            prefix_launch.assert_not_called()
            splitkv_launch.assert_not_called()
            workspace = launch.call_args.args[8]
            assert workspace.data_ptr() == pointer
            assert workspace.is_contiguous()
            assert workspace.shape == (tokens, heads, (length + 255) // 256, dim + 2)
            assert launch.call_args.args[-1] is (not draft)
        empty_cache.assert_called_once()  # No allocation growth after locking.


def test_gfx1151_metadata_preserves_paged_and_noncausal_contract():
    from vllm.v1.attention.backends import gfx1151_qwen_attn as backend

    common = SimpleNamespace(
        num_actual_tokens=8,
        max_query_len=8,
        max_seq_len=1057,
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        seq_lens=torch.tensor([1057], dtype=torch.int32),
        block_table_tensor=torch.tensor([[7, 2]], dtype=torch.int32),
        slot_mapping=torch.arange(8),
        causal=False,
    )
    builder = object.__new__(backend.Gfx1151QwenAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    metadata = builder.build(0, common)
    assert isinstance(metadata, backend.Gfx1151QwenAttentionMetadata)
    assert metadata.num_partitions == 5
    assert metadata.causal is False
    assert metadata.query_start_loc is common.query_start_loc
    assert metadata.seq_lens is common.seq_lens
    assert metadata.block_table is common.block_table_tensor
    assert metadata.slot_mapping is common.slot_mapping
    captured = builder.build_for_cudagraph_capture(common)
    assert captured.num_partitions == 5
    assert captured.max_seq_len == 1057
    assert torch.equal(captured.seq_lens, torch.tensor([1], dtype=torch.int32))
    assert not torch.count_nonzero(captured.query_start_loc)


def test_gfx1151_explicit_backend_rejects_wrong_device(gfx1151_config, monkeypatch):
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.gfx1151_qwen_attn import Gfx1151QwenAttentionBackend

    backend = AttentionBackendEnum.GFX1151_QWEN_ATTN.get_class()
    assert backend is Gfx1151QwenAttentionBackend
    assert backend.supports_compute_capability(DeviceCapability(11, 5))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx1100"),
    )
    assert not backend.supports_compute_capability(DeviceCapability(11, 0))


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig for testing."""
    config = MagicMock()
    config.model_config.dtype = torch.float16
    config.model_config.hf_config.architectures = ["LlamaForCausalLM"]
    config.cache_config.block_size = 16
    return config


@pytest.fixture
def mock_get_cdna_version():
    """Mock cdna version arch detection to return True."""
    with patch("vllm.platforms.rocm.get_cdna_version", return_value=3):
        yield


def test_aiter_unified_attention_uses_dedicated_metadata_builder():
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
        RocmAiterUnifiedAttentionMetadataBuilder,
    )
    from vllm.v1.attention.backends.rocm_attn import (
        RocmAttentionBackend,
        RocmAttentionMetadataBuilder,
    )

    assert RocmAttentionBackend.get_builder_cls() is RocmAttentionMetadataBuilder
    assert (
        RocmAiterUnifiedAttentionBackend.get_builder_cls()
        is RocmAiterUnifiedAttentionMetadataBuilder
    )


def test_aiter_unified_attention_capture_preserves_query_start_locations():
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionMetadataBuilder,
    )

    builder = object.__new__(RocmAiterUnifiedAttentionMetadataBuilder)
    metadata = MagicMock()
    metadata.seq_lens = torch.tensor([1048576, 524288], dtype=torch.int32)
    builder.build = MagicMock(return_value=metadata)
    common = MagicMock()
    expected_query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    common.query_start_loc = expected_query_start_loc.clone()
    metadata.query_start_loc = common.query_start_loc

    actual = builder.build_for_cudagraph_capture(common)

    builder.build.assert_called_once_with(0, common)
    assert actual is metadata
    assert torch.equal(actual.seq_lens, torch.ones_like(actual.seq_lens))
    assert actual.query_start_loc is common.query_start_loc
    assert torch.equal(actual.query_start_loc, expected_query_start_loc)


@pytest.mark.parametrize("use_dcp", [False, True])
@pytest.mark.parametrize(
    "env_vars, selected_backend, expected_backend_path",
    [
        # Test Case: Explicit FLEX_ATTENTION backend
        (
            {},
            "FLEX_ATTENTION",
            AttentionBackendEnum.FLEX_ATTENTION.get_path(),
        ),
        # Test Case 1: Default (no env vars, no explicit backend)
        (
            {},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 2: Explicit TRITON_ATTN backend
        (
            {},
            "TRITON_ATTN",
            AttentionBackendEnum.TRITON_ATTN.get_path(),
        ),
        # Test Case 3: Explicit ROCM_ATTN backend
        (
            {},
            "ROCM_ATTN",
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 4: Explicit ROCM_AITER_FA backend
        (
            {},
            "ROCM_AITER_FA",
            AttentionBackendEnum.ROCM_AITER_FA.get_path(),
        ),
        # Test Case 5: Explicit ROCM_AITER_UNIFIED_ATTN backend
        (
            {},
            "ROCM_AITER_UNIFIED_ATTN",
            AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN.get_path(),
        ),
        # Test Case 6: VLLM_ROCM_USE_AITER=1
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 7: VLLM_ROCM_USE_AITER=1 + explicit TRITON_ATTN
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            "TRITON_ATTN",
            AttentionBackendEnum.TRITON_ATTN.get_path(),
        ),
        # Test Case 8: VLLM_ROCM_USE_AITER=1 + VLLM_ROCM_USE_AITER_MHA=0
        (
            {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MHA": "0"},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 9: VLLM_ROCM_USE_AITER=1 + explicit ROCM_ATTN
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            "ROCM_ATTN",
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
    ],
)
def test_standard_attention_backend_selection(
    env_vars,
    selected_backend,
    expected_backend_path,
    use_dcp,
    mock_vllm_config,
    mock_get_cdna_version,
    monkeypatch,
):
    """Standard ROCm backends remain selectable without DCP and reject DCP."""
    # Set environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    # Import after setting env vars to ensure they're picked up
    # Reload envs to pick up new environment variables
    import importlib

    import vllm.envs as envs

    importlib.reload(envs)

    # Convert string backend to enum if provided
    backend_enum = None
    if selected_backend:
        backend_enum = getattr(AttentionBackendEnum, selected_backend)

    # Get the backend class path
    from vllm.platforms.rocm import RocmPlatform

    attn_selector_config = AttentionSelectorConfig(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_dcp=use_dcp,
    )

    if use_dcp:
        with pytest.raises(ValueError, match="DCP not supported"):
            RocmPlatform.get_attn_backend_cls(backend_enum, attn_selector_config)
        return

    backend_path = RocmPlatform.get_attn_backend_cls(
        selected_backend=backend_enum, attn_selector_config=attn_selector_config
    )

    assert backend_path == expected_backend_path


@pytest.mark.parametrize("use_dcp", [False, True])
@pytest.mark.parametrize(
    "env_vars, selected_backend, block_size, expected_backend_path, should_raise",
    [
        # Test Case 1: TRITON_MLA with block_size != 1
        (
            {},
            "TRITON_MLA",
            16,
            AttentionBackendEnum.TRITON_MLA.get_path(),
            False,
        ),
        # Test Case 2: TRITON_MLA with block_size == 1 (should raise)
        (
            {},
            "TRITON_MLA",
            1,
            None,
            True,
        ),
        # Test Case 3: ROCM_AITER_MLA with block_size == 1
        (
            {},
            "ROCM_AITER_MLA",
            1,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 4: ROCM_AITER_MLA with block_size != 1 (should raise)
        (
            {},
            "ROCM_AITER_MLA",
            16,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 5: VLLM_ROCM_USE_AITER=1 with block_size == 1
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            1,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 6: VLLM_ROCM_USE_AITER=1 with block_size == 16
        # (should use ROCM_AITER_MLA now, as it supports block_size 16)
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            16,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 7: VLLM_ROCM_USE_AITER=1 + explicit TRITON_MLA
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            "TRITON_MLA",
            16,
            AttentionBackendEnum.TRITON_MLA.get_path(),
            False,
        ),
        # Test Case 8: Explicit ROCM_AITER_TRITON_MLA
        (
            {},
            "ROCM_AITER_TRITON_MLA",
            16,
            AttentionBackendEnum.ROCM_AITER_TRITON_MLA.get_path(),
            False,
        ),
    ],
)
def test_mla_backend_selection(
    env_vars,
    selected_backend,
    block_size,
    expected_backend_path,
    should_raise,
    use_dcp,
    mock_vllm_config,
    monkeypatch,
):
    """Dense MLA remains selectable with DCP for valid block sizes."""
    # Set environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    # Import after setting env vars
    # Reload envs
    import importlib

    import vllm.envs as envs

    importlib.reload(envs)

    # Mock is_aiter_mla_enabled based on env vars and block_size
    aiter_enabled = env_vars.get("VLLM_ROCM_USE_AITER") == "1"

    mock_rocm_ops = MagicMock()
    mock_rocm_ops.is_mla_enabled.return_value = aiter_enabled
    mock_aiter_module = MagicMock()
    mock_aiter_module.rocm_aiter_ops = mock_rocm_ops

    with patch.dict("sys.modules", {"vllm._aiter_ops": mock_aiter_module}):
        # Convert string backend to enum if provided
        backend_enum = None
        if selected_backend:
            backend_enum = getattr(AttentionBackendEnum, selected_backend)

        from vllm.platforms.rocm import RocmPlatform

        if should_raise:
            with pytest.raises(ValueError):
                attn_selector_config = AttentionSelectorConfig(
                    head_size=128,
                    dtype=torch.float16,
                    kv_cache_dtype="auto",
                    block_size=block_size,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=False,
                    use_dcp=use_dcp,
                )
                attn_selector_config = AttentionSelectorConfig(
                    head_size=128,
                    dtype=torch.float16,
                    kv_cache_dtype="auto",
                    block_size=block_size,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=False,
                    use_dcp=use_dcp,
                )
                backend_path = RocmPlatform.get_attn_backend_cls(
                    selected_backend=backend_enum,
                    attn_selector_config=attn_selector_config,
                )

        else:
            attn_selector_config = AttentionSelectorConfig(
                head_size=128,
                dtype=torch.float16,
                kv_cache_dtype="auto",
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
                use_dcp=use_dcp,
            )

            backend_path = RocmPlatform.get_attn_backend_cls(
                selected_backend=backend_enum, attn_selector_config=attn_selector_config
            )

            assert backend_path == expected_backend_path


@pytest.mark.parametrize("use_dcp", [False, True])
@pytest.mark.parametrize(
    "selected_backend", [None, AttentionBackendEnum.ROCM_AITER_MLA_SPARSE]
)
def test_sparse_mla_backend_rejects_dcp(selected_backend, use_dcp):
    """Sparse MLA remains selectable without DCP and fails early with DCP."""
    from vllm.platforms.rocm import RocmPlatform

    selector_config = AttentionSelectorConfig(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=True,
        use_sparse=True,
        use_dcp=use_dcp,
    )
    if use_dcp:
        with pytest.raises(ValueError, match="DCP not supported"):
            RocmPlatform.get_attn_backend_cls(selected_backend, selector_config)
    else:
        assert RocmPlatform.get_attn_backend_cls(selected_backend, selector_config) == (
            AttentionBackendEnum.ROCM_AITER_MLA_SPARSE.get_path()
        )


@pytest.mark.parametrize("use_dcp", [False, True])
@pytest.mark.parametrize(
    "selected_backend, head_size, kv_cache_dtype",
    [
        (AttentionBackendEnum.TRITON_ATTN_DIFFKV, 192, "bfloat16"),
        # A 128-dimensional head uses 118 packed bytes, or 59 fp16 elements.
        (AttentionBackendEnum.TURBOQUANT, 59, "turboquant_k3v4_nc"),
    ],
)
def test_specialized_attention_backends_reject_dcp(
    selected_backend, head_size, kv_cache_dtype, use_dcp
):
    """Valid DiffKV and compressed-cache configurations must reject DCP."""
    from vllm.platforms.rocm import RocmPlatform

    selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=16,
        use_dcp=use_dcp,
    )
    if use_dcp:
        with pytest.raises(ValueError, match="DCP not supported"):
            RocmPlatform.get_attn_backend_cls(selected_backend, selector_config)
    else:
        assert RocmPlatform.get_attn_backend_cls(selected_backend, selector_config) == (
            selected_backend.get_path()
        )


def test_aiter_fa_requires_mi3xx(mock_vllm_config):
    """Test that ROCM_AITER_FA requires CDNA3+ architecture."""
    from vllm.platforms.rocm import RocmPlatform

    # Mock cdna version to return 1 (used by supports_compute_capability)
    with (
        patch("vllm.platforms.rocm.get_cdna_version", return_value=1),
        pytest.raises(
            ValueError,
            match="compute capability not supported",
        ),
    ):
        attn_selector_config = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        )

        RocmPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.ROCM_AITER_FA,
            attn_selector_config=attn_selector_config,
        )


def test_sparse_not_supported(mock_vllm_config):
    """Test that sparse MLA without use_mla flag raises an error."""
    from vllm.platforms.rocm import RocmPlatform

    with pytest.raises(
        ValueError,
        match="No valid attention backend found",
    ):
        attn_selector_config = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=True,
        )

        RocmPlatform.get_attn_backend_cls(
            selected_backend=None, attn_selector_config=attn_selector_config
        )


def _kv_connector_selector_config() -> AttentionSelectorConfig:
    return AttentionSelectorConfig(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_kv_connector=True,
    )


def test_unified_attn_declares_kv_connector_support():
    """ROCM_AITER_UNIFIED_ATTN opts into KV connectors and ROCM_ATTN does not."""
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend

    assert RocmAiterUnifiedAttentionBackend.supports_kv_connector() is True
    assert RocmAttentionBackend.supports_kv_connector() is False


def test_unified_attn_supports_kv_connector(mock_vllm_config, mock_get_cdna_version):
    """ROCM_AITER_UNIFIED_ATTN can be selected with KV connectors."""
    from vllm.platforms.rocm import RocmPlatform

    backend_path = RocmPlatform.get_attn_backend_cls(
        selected_backend=AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
        attn_selector_config=_kv_connector_selector_config(),
    )

    assert backend_path == AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN.get_path()


def test_rocm_attn_rejects_kv_connector(mock_vllm_config, mock_get_cdna_version):
    """Selecting ROCM_ATTN with a KV connector is illegal."""
    from vllm.platforms.rocm import RocmPlatform

    attn_selector_config = _kv_connector_selector_config()

    with pytest.raises(ValueError, match="KV connector not supported"):
        RocmPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.ROCM_ATTN,
            attn_selector_config=attn_selector_config,
        )


@pytest.mark.parametrize(
    "aiter_found, expected_backend",
    [
        (True, AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN),
        (False, AttentionBackendEnum.TRITON_ATTN),
    ],
)
def test_auto_selection_for_kv_connector(
    aiter_found, expected_backend, mock_vllm_config, mock_get_cdna_version
):
    """Auto-selection with a KV connector and AITER enabled resolves to unified attn,
    and to triton attn if AITER not enabled."""
    from vllm.platforms.rocm import RocmPlatform

    with patch(
        "vllm._aiter_ops.is_aiter_found_and_supported", return_value=aiter_found
    ):
        backend_path = RocmPlatform.get_attn_backend_cls(
            selected_backend=None,
            attn_selector_config=_kv_connector_selector_config(),
        )

    assert backend_path == expected_backend.get_path()


def test_unified_attn_prefers_block_contiguous_layout():
    """Unified attn prefers a block-first KV layout, hence ok with kv connectors."""
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend

    unified_preferred = RocmAiterUnifiedAttentionBackend.supported_kv_cache_layouts()[0]
    rocm_attn_preferred = RocmAttentionBackend.supported_kv_cache_layouts()[0]

    assert unified_preferred.is_block_contiguous is True
    assert rocm_attn_preferred.is_block_contiguous is False


def test_unified_attn_drops_lhbnc_with_kv_connector():
    """Connectors move a block as one contiguous byte range, which LHBNC breaks."""
    from vllm.config import KVTransferConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )
    from vllm.v1.kv_cache_interface import KVCacheLayout

    assert KVCacheLayout.LHBNC in (
        RocmAiterUnifiedAttentionBackend.supported_kv_cache_layouts()
    )

    config = VllmConfig(
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector", kv_role="kv_both"
        )
    )
    with set_current_vllm_config(config):
        layouts = RocmAiterUnifiedAttentionBackend.supported_kv_cache_layouts()

    assert layouts
    assert all(layout.is_block_compact for layout in layouts)
