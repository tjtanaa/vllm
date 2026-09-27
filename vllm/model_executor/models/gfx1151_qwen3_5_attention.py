# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-local gfx1151 attention selection; generic model paths stay unchanged."""

import os

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.gfx1151_qwen_attn import (
    Gfx1151QwenAttentionBackend,
    supports_gfx1151_qwen_config,
)

from .qwen3_dflash import DFlashQwen3Attention
from .qwen3_next import Qwen3NextAttention

logger = init_logger(__name__)


class Gfx1151Qwen3_5Attention(Qwen3NextAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, attn_backend=Gfx1151QwenAttentionBackend, **kwargs)


class Gfx1151DFlash2Attention(DFlashQwen3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, attn_backend=Gfx1151QwenAttentionBackend, **kwargs)


class Gfx1151QwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """Fused decode core, preserving original normalization and prefill."""

    def __init__(self, config, vllm_config, **kwargs):
        super().__init__(config=config, vllm_config=vllm_config, **kwargs)
        conv_dtype, state_dtype = self.get_state_dtype()
        self._gfx1151_gdn_enabled = (
            qwen_gdn_cls(vllm_config, config) is type(self)
            and not self.gqa_interleaved_layout
            and conv_dtype == torch.bfloat16
            and state_dtype in (torch.float32, torch.bfloat16)
        )
        if self._gfx1151_gdn_enabled:
            # Keep RMSNorm outside the core op so compilation and prefill use
            # exactly the original norm path. Fused-norm mode also changes
            # prefill to FLA normalization and can alter greedy generation.
            self.register_buffer(
                "_gfx1151_accepted",
                torch.ones(8, dtype=torch.int32, device=self.norm.weight.device),
                persistent=False,
            )

    def forward_hip(self, hidden_states):
        if self._gfx1151_gdn_enabled:
            return super().forward_cuda(hidden_states)
        return super().forward_hip(hidden_states)

    def _gfx1151_decode_metadata(self, metadata):
        if not self._gfx1151_gdn_enabled or not isinstance(
            metadata, GDNAttentionMetadata
        ):
            return None
        # Fused verification still regresses against Triton. Keep it on the
        # fallback until a repeatable multi-token winner is measured.
        if (
            metadata.num_prefills
            or metadata.num_spec_decodes
            or not 1 <= metadata.num_actual_tokens <= 256
        ):
            return None
        requests = metadata.num_decodes
        indices = metadata.non_spec_state_indices_tensor
        starts = metadata.non_spec_query_start_loc
        accepted = self._gfx1151_accepted
        if (
            metadata.spec_sequence_masks is not None
            or metadata.num_decode_tokens != requests
            or indices is None
            or indices.ndim != 1
        ):
            return None
        indices = indices.unsqueeze(1)
        if (
            not 1 <= requests <= 8
            or indices.size(0) < requests
            or starts is None
            or starts.ndim != 1
            or starts.numel() < requests + 1
            or accepted is None
            or accepted.ndim != 1
            or accepted.numel() < requests
        ):
            return None
        return indices[:requests], starts[: requests + 1], accepted[:requests]

    def _gfx1151_inputs_supported(self, mixed, a, b, output, conv, metadata):
        indices, starts, accepted = metadata
        state, weight = self.kv_cache[1], self.conv1d.weight
        device, tokens = mixed.device, mixed.size(0)
        tensors = (
            mixed,
            a,
            b,
            output,
            conv,
            state,
            weight,
            self.A_log,
            self.dt_bias,
            indices,
            starts,
            accepted,
        )
        if not mixed.is_cuda or any(t.device != device for t in tensors):
            return False
        if any(t.dtype != torch.bfloat16 for t in (mixed, a, b, output, conv, weight)):
            return False
        if (
            self.activation not in ("silu", "swish")
            or mixed.ndim != 2
            or mixed.shape[1] != 10240
            or mixed.stride(1) != 1
            or mixed.stride(0) < 10240
            or any(
                t.shape != (tokens, 48) or t.stride(1) != 1 or t.stride(0) < 48
                for t in (a, b)
            )
            or any(
                t.shape != (tokens, 48, 128)
                or t.stride(2) != 1
                or t.stride(1) != 128
                or t.stride(0) < 6144
                for t in (output,)
            )
            or not output.is_contiguous()
            or state.ndim != 4
            or state.size(0) < 1
            or state.shape[1:] != (48, 128, 128)
            or state.dtype not in (torch.float32, torch.bfloat16)
            or state.stride()[1:] != (16384, 128, 1)
            or state.stride(0) < 786432
            or weight.shape != (10240, 1, 4)
            or weight.stride(2) != 1
            or weight.stride(0) < 4
            or conv.ndim != 3
            or conv.size(0) < 1
            or conv.size(1) != 10240
            or conv.size(2) < indices.size(1) + 2
        ):
            return False
        ds = (
            conv.stride(2) == 1
            and conv.stride(1) >= conv.size(2)
            and conv.stride(0) >= 10240 * conv.stride(1)
        )
        sd = (
            conv.stride(1) == 1
            and conv.stride(2) >= 10240
            and conv.stride(0) >= conv.size(2) * conv.stride(2)
        )
        if not (ds or sd):
            return False
        if any(
            t.dtype != torch.int32 or not t.is_contiguous()
            for t in (indices, starts, accepted)
        ):
            return False
        for tensor, size, dtypes in (
            (self.A_log, 48, (torch.float32,)),
            (self.dt_bias, 48, (torch.float32, torch.bfloat16)),
        ):
            if (
                tensor.shape != (size,)
                or not tensor.is_contiguous()
                or tensor.dtype not in dtypes
            ):
                return False
        bias = self.conv1d.bias
        return bias is None or (
            bias.device == device
            and bias.dtype == torch.bfloat16
            and bias.shape == (10240,)
            and bias.is_contiguous()
        )

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        raw_metadata = get_forward_context().attn_metadata
        metadata = (
            raw_metadata.get(self.prefix) if isinstance(raw_metadata, dict) else None
        )
        decode = self._gfx1151_decode_metadata(metadata)
        if decode is not None:
            n = metadata.num_actual_tokens
            conv = self.kv_cache[0]
            if not is_conv_state_dim_first():
                conv = conv.transpose(-1, -2)
            mixed, a_, b_ = mixed_qkv[:n], a[:n], b[:n]
            output = core_attn_out[:n]
            if mixed.size(0) == n and self._gfx1151_inputs_supported(
                mixed, a_, b_, output, conv, decode
            ):
                ops.gfx1151_qwen_gdn_decode_core(
                    mixed,
                    a_,
                    b_,
                    self.A_log,
                    self.dt_bias,
                    *decode,
                    self.kv_cache[1],
                    output,
                    self.head_k_dim**-0.5,
                    conv,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                )
                logger.info_once(
                    "gfx1151 Qwen fused HIP GDN core engaged: M=%d",
                    decode[0].size(1),
                )
                return
        super()._forward_core(mixed_qkv, b, a, core_attn_out)


def qwen_gdn_cls(vllm_config, config):
    supported = (
        envs.VLLM_GFX1151_QWEN_GDN
        and "VLLM_GDN_DECODE_KERNEL" not in os.environ
        and supports_gfx1151_qwen_config(vllm_config, config)
        and getattr(config, "model_type", None) == "qwen3_5_text"
        and getattr(config, "hidden_act", None) in ("silu", "swish")
        and tuple(
            getattr(config, key, None)
            for key in (
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
            )
        )
        == (16, 48, 128, 128, 4)
        and hasattr(torch.ops._rocm_C, "gfx1151_qwen_gdn_decode_core")
    )
    return (
        Gfx1151QwenGatedDeltaNetAttention if supported else QwenGatedDeltaNetAttention
    )


def qwen_attention_cls(vllm_config, config, *, draft=False):
    if supports_gfx1151_qwen_config(vllm_config, config):
        return Gfx1151DFlash2Attention if draft else Gfx1151Qwen3_5Attention
    return DFlashQwen3Attention if draft else Qwen3NextAttention
