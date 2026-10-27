# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When the Kimi-K3 AMD MLA wrapper takes its fused decode Q-prep path."""

from types import SimpleNamespace

import pytest

from vllm._aiter_ops import rocm_aiter_ops
from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper


def _wrapper(backend_name: str) -> SimpleNamespace:
    """The attributes _fused_qk_prep_supported reads, for a fused-eligible
    Kimi-K3 rank (fp8 cache, kv_lora_rank 512, rope 64, 16 heads)."""
    attn = SimpleNamespace(
        kv_cache_dtype="fp8",
        attn_backend=SimpleNamespace(get_name=lambda: backend_name),
        impl=SimpleNamespace(dcp_world_size=1),
        use_pcp=False,
        hisparse_cache=None,
        q_pad_num_heads=None,
        is_aiter_triton_fp4_bmm_enabled=False,
        is_aiter_triton_fp8_bmm_enabled=False,
    )
    return SimpleNamespace(
        mla_attn=attn, kv_lora_rank=512, qk_rope_head_dim=64, num_heads=16
    )


@pytest.mark.parametrize(
    ("backend_name", "fused"), [("ROCM_AITER_MLA", True), ("TRITON_MLA", False)]
)
def test_fused_qk_prep_only_for_rocm_aiter_mla(monkeypatch, backend_name, fused):
    """The fused path pads the query to AITER's head counts and expects a bf16
    decode output. With VLLM_ROCM_USE_AITER_MLA on but the layer pinned to
    TRITON_MLA (an fp8 cache returns fp8), it must not be taken: it crashed in
    the W_UV baddbmm on the fp8 output."""
    monkeypatch.setattr(rocm_aiter_ops, "is_mla_enabled", lambda: True)
    supported = KimiK3MultiHeadLatentAttentionWrapper._fused_qk_prep_supported(
        _wrapper(backend_name)
    )
    assert supported is fused
