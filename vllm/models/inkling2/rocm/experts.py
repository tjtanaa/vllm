# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on ROCm: the `FlatExperts` backends of
`common.moe.flat_experts_class`, for GPUs without the CDNA4 block-scaled
matrix cores (e.g. gfx1151).

NVFP4 runs on bf16 activations (W4A16): batches of up to
`NVFP4_GEMV_MAX_TOKENS` tokens (decode and DSpark verify) on a
matrix-vector kernel that streams each slot's expert weights once
(`_nvfp4_gemv_kernel`), larger ones on the vendor-neutral NVFP4 emulation
MoE Triton kernel as on XPU (`xpu.experts.XpuNvfp4Experts`), with launch
configs measured on gfx1151 (`nvfp4_config`).

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
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
)
from vllm.models.inkling2.common.moe import FlatExperts, QuantFormat
from vllm.models.inkling2.common.ops.moe import swiglu
from vllm.models.inkling2.xpu.experts import (
    BATCH_INVARIANT_CONFIG_TOKENS,
    XpuNvfp4Experts,
)
from vllm.triton_utils import tl, triton

MXFP8_BLOCK = 32
# Batches of at most this many tokens take the matrix-vector kernel (1.5x
# faster than the emulation kernel on gfx1151 for one token). It reads the
# weights once per slot, so on DSpark verify batches, whose tokens share
# most experts, the emulation kernel (once per expert) is faster.
NVFP4_GEMV_MAX_TOKENS = 1


@triton.jit
def _nvfp4_gemv_kernel(
    A,
    IDS,
    W,
    S,
    G,
    Out,
    N: tl.constexpr,
    K: tl.constexpr,
    A_DIV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KP: tl.constexpr,
):
    """Row `r` of `Out` `[R, N]` (bf16) is `A[r // A_DIV] @ W[IDS[r]].T`:
    e2m1 weights `W` `[E, N, K / 2]` (low nibble: even K index) with e4m3
    block scales `S` `[E, N, K / 16]` and fp32 global scales `G` `[E]`.
    Rows with `IDS` -1 are left unset. fp32 products and sums, the global
    scale on the sum, as the emulation kernel."""
    r = tl.program_id(0)
    expert = tl.load(IDS + r)
    if expert < 0:
        return
    expert = expert.to(tl.int64)
    KP: tl.constexpr = K // 2
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    a_row = A + (r // A_DIV).to(tl.int64) * K
    w_rows = W + expert * N * KP + n[:, None].to(tl.int64) * KP
    s_rows = S + expert * N * (K // 16) + n[:, None].to(tl.int64) * (K // 16)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for kp0 in range(0, KP, BLOCK_KP):
        kp = kp0 + tl.arange(0, BLOCK_KP)
        raw = tl.load(w_rows + kp[None, :])
        scale = tl.load(s_rows + kp[None, :] // 8)
        scale = tl.cast(scale, tl.float8e4nv, bitcast=True).to(tl.float32)
        a_even = tl.load(a_row + 2 * kp).to(tl.float32)
        a_odd = tl.load(a_row + 2 * kp + 1).to(tl.float32)
        w_even = _e2m1_inline(raw & 0x0F)
        w_odd = _e2m1_inline((raw >> 4) & 0x0F)
        acc += tl.sum(scale * (w_even * a_even[None, :] + w_odd * a_odd[None, :]), 1)
    acc = acc * tl.load(G + expert)
    tl.store(Out + r.to(tl.int64) * N + n, acc.to(tl.bfloat16))


def nvfp4_gemv(
    a: torch.Tensor,
    ids: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    a_div: int,
) -> torch.Tensor:
    """bf16 `[R, N]` rows `a[r // a_div] @ W[ids[r]].T` of the `R` flat slots
    `ids` (-1: left unset)."""
    num_experts, n, kp = weight.shape
    block_n, block_kp = 16, 64
    assert n % block_n == 0 and kp % block_kp == 0 and a.is_contiguous()
    out = torch.empty((ids.shape[0], n), dtype=torch.bfloat16, device=a.device)
    _nvfp4_gemv_kernel[(ids.shape[0], n // block_n)](
        a,
        ids,
        weight,
        scale.view(torch.uint8),
        global_scale,
        out,
        N=n,
        K=2 * kp,
        A_DIV=a_div,
        BLOCK_N=block_n,
        BLOCK_KP=block_kp,
        num_warps=2,
    )
    return out


def nvfp4_config(num_tokens: int) -> tuple[dict, bool] | None:
    """The emulation kernel's launch config and split-dots flag for a batch
    of `num_tokens` tokens (8 slots over 128 experts at Inkling 2 Mini's
    shapes, best of a sweep on gfx1151), or None for the XPU defaults."""
    if num_tokens < 256:
        return None
    if num_tokens < 1024:
        block_m, num_stages = 32, 1
    elif num_tokens < 2048:
        block_m, num_stages = 64, 2
    else:
        block_m, num_stages = 128, 1
    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": num_stages,
    }
    return config, False


class RocmNvfp4Experts(XpuNvfp4Experts):
    """`XpuNvfp4Experts` with the matrix-vector kernel for small batches and
    the gfx1151 launch configs of the emulation kernel."""

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
        tokens, topk = ids.shape
        if envs.VLLM_BATCH_INVARIANT or tokens == 0:
            return super().__call__(
                x, ids, weights, src2dst, num_global_experts, gate_cap, up_cap
            )
        if tokens <= NVFP4_GEMV_MAX_TOKENS:
            flat = ids.view(-1)
            gate_up = nvfp4_gemv(x, flat, *self.weights["w13"], topk)
            activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
            del gate_up
            return nvfp4_gemv(activated, flat, *self.weights["w2"], 1)
        tuned = nvfp4_config(tokens)
        if tuned is None:
            return super().__call__(
                x, ids, weights, src2dst, num_global_experts, gate_cap, up_cap
            )
        config, split_dots = tuned
        alignment = moe_align_block_size(
            ids, config["BLOCK_SIZE_M"], self.num_experts, ignore_invalid_experts=True
        )[:3]
        gate_up = self._gemm("w13", x, topk, alignment, config, split_dots)
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return self._gemm("w2", activated, 1, alignment, config, split_dots)


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
    cls: type = RocmMxfp8Experts if quant_format == "mxfp8" else RocmNvfp4Experts
    return partial(cls, topk=topk, gate_cap=gate_cap, up_cap=up_cap)
