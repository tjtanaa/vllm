# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA capability contract: what the selector accepts and rejects.

Each declaration is checked both ways: declared values pass the selector (and
the forward suites run them), undeclared values are refused at selection time
with a reason, instead of failing later inside a kernel.
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_mla import mla_shapes as shapes
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms.rocm import RocmPlatform, on_gfx950
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLABackend,
    AiterMLAHelper,
    AiterMLAImpl,
    AiterMLAMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

BACKEND = AiterMLABackend
# kv_lora_rank + qk_rope_head_dim for every dense MLA model in mla_shapes, and
# the NoPE latent (GLM-5.3-Flash style, rope 0).
LATENT_SIZES = [576, 512]


def _select(**overrides) -> str:
    config = AttentionSelectorConfig(
        **{
            "head_size": 576,
            "dtype": torch.bfloat16,
            "kv_cache_dtype": "auto",
            "block_size": 16,
            "use_mla": True,
            **overrides,
        }
    )
    return RocmPlatform.get_attn_backend_cls(
        selected_backend=AttentionBackendEnum.ROCM_AITER_MLA,
        attn_selector_config=config,
    )


def test_declarations_match_flagship_model_needs():
    """Pin the declared surface; changing it must update these suites."""
    assert BACKEND.is_mla()
    assert not BACKEND.is_sparse()
    assert BACKEND.get_supported_head_sizes() == []
    assert BACKEND.supported_dtypes == [torch.float16, torch.bfloat16]
    assert set(BACKEND.supported_kv_cache_dtypes) == {
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    }
    (block_size,) = BACKEND.get_supported_kernel_block_sizes()
    assert block_size.base == 1
    assert not BACKEND.supports_sink()
    assert not BACKEND.supports_sliding_window()
    assert not BACKEND.supports_mm_prefix()
    assert BACKEND.supports_non_causal() == on_gfx950()
    assert BACKEND.supports_dcp()
    assert BACKEND.supports_pcp()
    assert BACKEND.supports_non_causal_dcp()
    assert (
        AiterMLAMetadataBuilder.get_cudagraph_support(None, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )


@pytest.mark.parametrize("head_size", LATENT_SIZES)
@pytest.mark.parametrize(
    "dtype, kv_cache_dtype",
    [
        (torch.bfloat16, "auto"),
        (torch.bfloat16, "bfloat16"),
        (torch.bfloat16, "fp8"),
        (torch.bfloat16, "fp8_e4m3"),
        # fp16 decodes on fp8 queries only.
        (torch.float16, "fp8"),
        (torch.float16, "fp8_e4m3"),
    ],
    ids=lambda v: str(v).replace("torch.", ""),
)
@pytest.mark.parametrize("block_size", [1, 16, 64, 128])
def test_selector_accepts_declared_combinations(
    head_size, dtype, kv_cache_dtype, block_size
):
    path = _select(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
    )
    assert path == AttentionBackendEnum.ROCM_AITER_MLA.get_path()


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_dcp": True},
        {"use_pcp": True},
        {"use_kv_connector": True},
        {"use_adaptive_verification": True},
    ],
    ids=lambda o: next(iter(o)),
)
def test_selector_accepts_declared_features(overrides):
    assert _select(**overrides) == AttentionBackendEnum.ROCM_AITER_MLA.get_path()


@pytest.mark.skipif(not on_gfx950(), reason="non-causal kernels are gfx950 only")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_selector_accepts_non_causal_bf16(kv_cache_dtype):
    """DSpark drafts run a non-causal block through this backend."""
    assert _select(use_non_causal=True, kv_cache_dtype=kv_cache_dtype) == (
        AttentionBackendEnum.ROCM_AITER_MLA.get_path()
    )


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"dtype": torch.float32}, "dtype"),
        ({"kv_cache_dtype": "nvfp4"}, "kv_cache_dtype"),
        # ROCm concat_and_cache_mla cannot write it.
        ({"kv_cache_dtype": "fp8_e5m2"}, "kv_cache_dtype"),
        ({"kv_cache_dtype": "fp8_ds_mla"}, "kv_cache_dtype"),
        ({"kv_cache_dtype": "int8_per_token_head"}, "kv_cache_dtype"),
        ({"use_mla": False}, "non-MLA"),
        ({"use_sparse": True}, "sparse"),
        ({"has_sink": True}, "sinks"),
        ({"has_sliding_window": True}, "sliding window"),
        ({"use_mm_prefix": True}, "mm_prefix|multimodal"),
        ({"attn_type": AttentionType.ENCODER_DECODER}, "attention type"),
        ({"attn_type": AttentionType.ENCODER}, "attention type"),
        ({"attn_type": AttentionType.ENCODER_ONLY}, "attention type"),
        ({"use_batch_invariant": True}, "batch invariance"),
        # Neither the asm decode nor Gluon takes an fp16 query.
        ({"dtype": torch.float16}, "fp16 MLA decode needs an fp8 KV cache"),
        (
            {"dtype": torch.float16, "use_non_causal": True},
            "fp16 MLA decode needs an fp8 KV cache",
        ),
        ({"kv_cache_dtype": "float16"}, "kv_cache_dtype"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_selector_rejects_undeclared(overrides, reason):
    with pytest.raises(ValueError, match=reason):
        _select(**overrides)


@pytest.mark.skipif(on_gfx950(), reason="gfx950 ships non-causal kernels")
def test_selector_rejects_non_causal_without_kernels():
    with pytest.raises(ValueError, match="non-causal"):
        _select(use_non_causal=True)


@pytest.mark.parametrize(
    "aiter, mla, first",
    [
        ("1", "1", AttentionBackendEnum.ROCM_AITER_MLA),
        ("1", "0", AttentionBackendEnum.TRITON_MLA),
        ("0", "1", AttentionBackendEnum.TRITON_MLA),
    ],
)
def test_env_gates_backend_priority(monkeypatch, aiter, mla, first):
    from vllm.platforms.rocm import _get_backend_priorities

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", aiter)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MLA", mla)
    rocm_aiter_ops.refresh_env_variables()
    try:
        priorities = _get_backend_priorities(use_mla=True, use_sparse=False)
        assert priorities[0] == first
        assert AttentionBackendEnum.TRITON_MLA in priorities
        # Sparse MLA never falls back onto the dense backend.
        assert _get_backend_priorities(use_mla=True, use_sparse=True) == [
            AttentionBackendEnum.ROCM_AITER_MLA_SPARSE
        ]
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize("shape", shapes.ALL_SHAPES, ids=lambda s: s.id)
def test_every_deployed_head_count_is_valid(shape):
    """Every per-rank count a real model reaches has a decode kernel."""
    assert AiterMLAHelper.is_valid_num_heads(shape.num_heads)
    kernel_heads = AiterMLAHelper.get_actual_mla_num_heads(shape.num_heads)
    assert kernel_heads % 16 == 0 or kernel_heads == 24
    assert kernel_heads >= shape.num_heads


@pytest.mark.parametrize("num_heads", [0, 136, 200])
def test_impl_rejects_head_counts_without_a_kernel(num_heads):
    """Above 128 heads only multiples of 16 have an asm decode kernel."""
    with pytest.raises(AssertionError, match="multiple of 16"):
        AiterMLAHelper.check_num_heads_validity(num_heads)


@pytest.mark.parametrize(
    "feature",
    [
        {"alibi_slopes": [1.0] * 16},
        {"sliding_window": 128},
        {"logits_soft_cap": 30.0},
    ],
    ids=lambda f: next(iter(f)),
)
def test_impl_rejects_unsupported_features(feature, default_vllm_config):
    kwargs = dict(
        num_heads=16,
        head_size=576,
        scale=192**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        kv_b_proj=None,
    )
    kwargs.update(feature)
    with pytest.raises(NotImplementedError, match="Aiter MLA does not support"):
        AiterMLAImpl(**kwargs)


def test_impl_rejects_kv_sharing(default_vllm_config):
    with pytest.raises(NotImplementedError, match="KV sharing"):
        AiterMLAImpl(
            16,
            576,
            192**-0.5,
            1,
            None,
            None,
            "auto",
            None,
            AttentionType.DECODER,
            "model.layers.0.self_attn.attn",
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            qk_head_dim=192,
            v_head_dim=128,
            kv_b_proj=None,
        )


def test_impl_claims_context_parallelism_with_lse():
    """DCP merges partial outputs by LSE, so the impl must return natural-log
    LSE from decode; PCP is inherited from the MLA base."""
    assert AiterMLAImpl.supports_dcp
    assert AiterMLAImpl.supports_pcp
    assert AiterMLAImpl.can_return_lse_for_decode
    assert AiterMLAImpl.lse_base_on_e
