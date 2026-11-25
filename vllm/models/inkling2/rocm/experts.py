# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on ROCm: the `FlatExperts` backends of
`common.moe.flat_experts_class`, for GPUs without the CDNA4 block-scaled
matrix cores (e.g. gfx1151).

NVFP4 runs the vendor-neutral NVFP4 emulation MoE Triton kernel on bf16
activations (W4A16), as on XPU (`xpu.experts.XpuNvfp4Experts`).

MXFP8 is dequantized to bf16 once after loading and runs the bf16 fused MoE
Triton kernel (W16A16, where the checkpoint's recipe quantizes the expert
activations to MXFP8): the weights take twice the memory of the checkpoint
layout. Both keep the checkpoint's interleaved `w13` rows, consumed by
`swiglu`, and scale the SiTU output by the routing weight before the down
projection, as on Hopper.
"""

from collections.abc import Callable
from functools import partial

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.models.inkling2.common.moe import FlatExperts, QuantFormat
from vllm.models.inkling2.common.ops.moe import swiglu
from vllm.models.inkling2.xpu.experts import (
    BATCH_INVARIANT_CONFIG_TOKENS,
    XpuNvfp4Experts,
)
from vllm.triton_utils import tl

MXFP8_BLOCK = 32


def dequantize_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """bf16 `[E, N, K]` of e4m3 `weight` `[E, N, K]` with e8m0 block scales
    `scale` `[E, N, K / 32]` (uint8). One expert is converted at a time, so
    this step needs one expert's fp32 copy of extra memory."""
    out = torch.empty(weight.shape, dtype=torch.bfloat16, device=weight.device)
    for e in range(weight.shape[0]):
        w = weight[e].float().unflatten(-1, (-1, MXFP8_BLOCK))
        s = torch.exp2(scale[e].float() - 127.0).unsqueeze(-1)
        out[e].copy_((w * s).flatten(-2))
    return out


class RocmMxfp8Experts:
    """MXFP8 experts dequantized to bf16 on the bf16 fused MoE Triton kernel
    (`invoke_fused_moe_triton_kernel`): `w13` GEMM, `swiglu` (routing weight
    before the down projection) and `w2` GEMM over the flat rows.

    Args:
        weights: Per projection (`w13`, `w2`), the checkpoint weights
            `[E, N, K]` (e4m3), e8m0 block scales `[E, N, K / 32]` (uint8)
            and no global scale.
        shapes: `(N, K)` of each projection.
        nvfp4: Must be False.
        topk, gate_cap, up_cap: Unused (passed per call).
    """

    def __init__(
        self,
        weights,
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        *,
        topk: int,
        gate_cap: float,
        up_cap: float,
    ) -> None:
        assert not nvfp4
        self.weights = {}
        for name, (n, k) in zip(("w13", "w2"), shapes):
            weight, scale, _ = weights[name]
            assert weight.shape[1:] == (n, k) and scale.shape[1:] == (n, k // 32)
            assert weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.uint8
            self.weights[name] = dequantize_mxfp8(weight, scale)
        self.num_experts = self.weights["w13"].shape[0]

    def _gemm(self, name, a, top_k, alignment, config):
        """bf16 `[rows, N]`: row `r` of the aligned rows is
        `a[r // top_k] @ W[e].T` for its expert `e`; the rows the alignment
        does not list are left unset."""
        weight = self.weights[name]
        sorted_ids, expert_ids, num_post_padded = alignment
        out = torch.empty(
            (a.shape[0], top_k, weight.shape[1]), dtype=torch.bfloat16, device=a.device
        )
        invoke_fused_moe_triton_kernel(
            a,
            weight,
            out,
            None,
            None,
            None,
            sorted_ids,
            expert_ids,
            num_post_padded,
            False,
            top_k,
            config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
        )
        return out.view(-1, weight.shape[1])

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor:
        """bf16 `[T * topk, H]` down-projection rows of the local slots of
        `ids` (-1: not local); the other rows are left unset."""
        tokens, topk = ids.shape
        w13, w2 = self.weights["w13"], self.weights["w2"]
        if tokens == 0:
            return x.new_empty((0, w2.shape[1]))
        config_tokens = (
            BATCH_INVARIANT_CONFIG_TOKENS if envs.VLLM_BATCH_INVARIANT else tokens
        )
        config = try_get_optimal_moe_config(
            w13.size(), w2.size(), topk, None, config_tokens
        )
        alignment = moe_align_block_size(
            ids, config["BLOCK_SIZE_M"], self.num_experts, ignore_invalid_experts=True
        )[:3]
        gate_up = self._gemm("w13", x, topk, alignment, config)
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return self._gemm("w2", activated, 1, alignment, config)


def rocm_flat_experts(
    quant_format: QuantFormat, topk: int, gate_cap: float, up_cap: float
) -> Callable[..., FlatExperts]:
    """The ROCm `FlatExperts` constructor of `quant_format` experts."""
    cls: type = RocmMxfp8Experts if quant_format == "mxfp8" else XpuNvfp4Experts
    return partial(cls, topk=topk, gate_cap=gate_cap, up_cap=up_cap)
