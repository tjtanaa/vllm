# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA decode-only metadata builds must not sync the GPU.

The builder runs between decode steps on the critical path. A small-head
(< 16 heads) bf16 verify block takes Gluon's MTP decode, which needs a lower
bound on the KV lengths; reading it with ``.min().item()`` on device lengths
synced the GPU every step. The test builds real decode metadata with the
production builder under ``torch.cuda.set_sync_debug_mode("error")``.
"""

import types
from contextlib import contextmanager

import pytest
import torch

from vllm._aiter_ops import is_aiter_found
from vllm.platforms import current_platform


def _on_gfx950() -> bool:
    if not (current_platform.is_rocm() and is_aiter_found()):
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="Gluon MLA decode is gfx950-only"
)

BLOCK_SIZE = 16


@contextmanager
def _no_gpu_sync():
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def _builder(num_heads: int, num_speculative_tokens: int, max_model_len: int):
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import SpeculativeConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.attention.mla_attention import get_mla_dims
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda:0")
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-R1",
        max_model_len=max_model_len,
        num_gpu_blocks=1024,
        block_size=BLOCK_SIZE,
        max_num_seqs=16,
        hf_config_override={"num_attention_heads": num_heads},
    )
    # The builder reads the head count from the architecture config.
    vllm_config.model_config.model_arch_config.total_num_attention_heads = num_heads
    if num_speculative_tokens:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=num_speculative_tokens
        )
    spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    mla_dims = get_mla_dims(vllm_config.model_config)
    layer_name = "placeholder"
    vllm_config.compilation_config.static_forward_context[layer_name] = (
        types.SimpleNamespace(
            prefill_backend=torch.empty((1,)),
            q_lora_rank=mla_dims.q_lora_rank,
            kv_lora_rank=mla_dims.kv_lora_rank,
            qk_nope_head_dim=mla_dims.qk_nope_head_dim,
            qk_rope_head_dim=mla_dims.qk_rope_head_dim,
            v_head_dim=mla_dims.v_head_dim,
            non_causal_multi_token_decode=False,
        )
    )
    init_workspace_manager(device)
    builder_cls = AttentionBackendEnum.ROCM_AITER_MLA.get_class().get_builder_cls()
    with set_current_vllm_config(vllm_config):
        builder = builder_cls(spec, [layer_name], vllm_config, device)
    return builder, vllm_config, device


@pytest.mark.parametrize("num_heads", [8, 4])
@pytest.mark.parametrize("num_speculative_tokens", [0, 1, 3])
def test_decode_only_build_does_not_sync(num_heads, num_speculative_tokens):
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.config.vllm import set_current_vllm_config

    seq_lens = [100, 37, 600, 1500]
    builder, vllm_config, device = _builder(
        num_heads, num_speculative_tokens, max_model_len=2048
    )
    qlen = num_speculative_tokens + 1
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=[qlen] * len(seq_lens)),
        BLOCK_SIZE,
        device,
    )
    torch.accelerator.synchronize()
    with set_current_vllm_config(vllm_config), _no_gpu_sync():
        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
    decode = metadata.decode
    assert decode is not None and metadata.num_prefills == 0
    assert builder._decode_num_heads == num_heads
    if qlen > 1:
        # Small-head bf16 verify is the Gluon MTP route, which needs the bound.
        assert decode.use_gluon_verify
        # Async scheduling's host lengths assume every draft was accepted.
        slack = qlen - 1 if vllm_config.scheduler_config.async_scheduling else 0
        assert decode.min_kv_seq_len == min(seq_lens) - slack
    else:
        assert decode.use_gluon_decode
