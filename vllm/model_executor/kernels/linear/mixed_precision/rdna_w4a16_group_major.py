# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in gfx1151 W4A16 group-major cache for small Qwen verification batches.

Original ExLlama weights remain available to the HIP skinny and generic paths.
The derived cache adds approximately 11.68 GiB for the full 27B target. Enable
VLLM_GFX1151_W4_GROUP_MAJOR before loading; changing it requires a restart.
"""

import torch

import vllm.envs as envs
from vllm.config.vllm import get_current_vllm_config_or_none
from vllm.model_executor.kernels.linear.mixed_precision import (
    rdna_hybrid_w4a16 as original,
)
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

SHAPES = {(5120, 17408), (34816, 5120), (16384, 5120), (14336, 5120), (5120, 6144)}
Q_BUFFER = "_gfx1151_group_major_weights"
S_BUFFER = "_gfx1151_group_major_scales"


@triton.jit
def group_major_w4a16_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_MAJOR: tl.constexpr = False,
):
    tl.static_assert(not GROUP_MAJOR or (SPLIT_K == 1 and BLOCK_K == 128))
    pid_m, pid_n, partition = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    shifts = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    shifts = tl.reshape(tl.broadcast_to(shifts[None, :], (BLOCK_K // 8, 8)), (BLOCK_K,))
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_N, BLOCK_K))
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    blocks_per_partition: tl.constexpr = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K)
    for block in range(blocks_per_partition):
        start = (partition * blocks_per_partition + block) * BLOCK_K
        kk = start + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + rows[:, None] * K + kk[None, :],
            mask=(rows[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        packed_k = start // 8 + tl.arange(0, BLOCK_K // 8)
        if GROUP_MAJOR:
            packed_offsets = (
                (start // 128) * N * 16
                + columns[:, None] * 16
                + tl.arange(0, 16)[None, :]
            )
        else:
            packed_offsets = columns[:, None] * (K // 8) + packed_k[None, :]
        words = tl.load(
            b_ptr + packed_offsets,
            mask=(columns[:, None] < N) & (packed_k[None, :] < K // 8),
            other=0,
        )
        values = tl.interleave(words, words)
        values = tl.interleave(values, values)
        values = tl.interleave(values, values)
        codes = (values >> shifts) & 15
        if GROUP_MAJOR:
            scale_offsets = (start // 128) * N + columns
        else:
            scale_offsets = columns * (K // 128) + start // 128
        scales = tl.load(
            scales_ptr + scale_offsets,
            mask=(columns < N) & (start < K),
            other=0.0,
        )
        weight = (codes - 8).to(scales.dtype) * scales[:, None]
        accumulator += tl.dot(a, tl.trans(weight), out_dtype=tl.float32)
    tl.store(
        partial_ptr + partition * M * N + rows[:, None] * N + columns[None, :],
        accumulator,
        mask=(rows[:, None] < M) & (columns[None, :] < N),
    )


def eligible_weights(w_q, w_s, w_zp, group_size):
    if not original._on_gfx1151() or group_size != 128 or w_zp is not None:
        return False
    if w_q.ndim != 2 or w_s.ndim != 2:
        return False
    n, packed_k = w_q.shape
    k = packed_k * 2
    return (
        (n, k) in SHAPES
        and w_s.shape == (n, k // 128)
        and w_q.dtype == torch.int8
        and w_s.dtype == torch.bfloat16
        and w_q.is_cuda
        and w_s.device == w_q.device
        and w_q.is_contiguous()
        and w_s.is_contiguous()
        and not w_q.requires_grad
        and not w_s.requires_grad
        and w_q.data_ptr() % 16 == 0
        and w_s.data_ptr() % 16 == 0
    )


def group_major_operands(weights, scales):
    """Losslessly repack [N,K/8] words and [N,K/128] scales by K group."""
    if weights.ndim != 2 or scales.ndim != 2:
        raise ValueError("Expected two matrices")
    n, words = weights.shape
    if n <= 0 or words <= 0 or words % 16 or scales.shape != (n, words // 16):
        raise ValueError("Expected complete 128-element quantization groups")
    if not weights.is_contiguous() or not scales.is_contiguous():
        raise ValueError("Expected contiguous source operands")
    packed = weights.reshape(n, words // 16, 16).permute(1, 0, 2).contiguous()
    return packed, scales.t().contiguous()


def prepare_layer(layer, w_q, w_s, w_zp, group_size):
    """Prepare derived buffers, refreshing captured storage during reload."""
    info = get_layerwise_info(layer)
    old_buffers = (
        info.kernel_tensors[1] if info.kernel_tensors is not None else layer._buffers
    )
    old_q, old_s = old_buffers.get(Q_BUFFER), old_buffers.get(S_BUFFER)
    has_cache = old_q is not None or old_s is not None
    config = get_current_vllm_config_or_none()
    offloaded = config is not None and (
        config.offload_config.uva.cpu_offload_gb > 0
        or config.offload_config.prefetch.offload_group_size > 0
    )
    eligible = (
        envs.VLLM_GFX1151_W4_GROUP_MAJOR
        and not offloaded
        and eligible_weights(w_q, w_s, w_zp, group_size)
    )
    if not eligible:
        if has_cache:
            raise RuntimeError("Cannot invalidate a captured group-major weight cache")
        return False
    # Reload restores the original buffer set. Do not introduce a new graph
    # input when the original layer did not have this cache.
    if info.kernel_tensors is not None and not has_cache:
        return False

    words = w_q.view(torch.int32)
    grouped_q, grouped_s = group_major_operands(words, w_s)
    if not torch.equal(grouped_q.permute(1, 0, 2).reshape_as(words), words):
        raise RuntimeError("Group-major packed-weight round trip failed")
    if not torch.equal(grouped_s.t(), w_s):
        raise RuntimeError("Group-major scale round trip failed")
    if has_cache:
        for old, new in ((old_q, grouped_q), (old_s, grouped_s)):
            if (
                old is None
                or old.shape != new.shape
                or old.dtype != new.dtype
                or old.device != new.device
                or not old.is_contiguous()
                or old.requires_grad
                or old.data_ptr() % 16
            ):
                raise RuntimeError("Incompatible captured group-major weight cache")
        # The generic reloader skips derived non-persistent buffers. Update
        # their original storage explicitly before it restores graph references.
        old_q.copy_(grouped_q)
        old_s.copy_(grouped_s)
        grouped_q, grouped_s = old_q, old_s
    layer.register_buffer(Q_BUFFER, grouped_q, persistent=False)
    layer.register_buffer(S_BUFFER, grouped_s, persistent=False)
    return True


def can_use(x, w_q, w_s, w_zp, bias, group_size, grouped_q, grouped_s):
    if not envs.VLLM_GFX1151_W4_GROUP_MAJOR:
        return False
    if bias is not None or not eligible_weights(w_q, w_s, w_zp, group_size):
        return False
    if x.ndim != 2 or grouped_q is None or grouped_s is None:
        return False
    m, k = x.shape
    n = w_q.shape[0]
    if not 2 <= m <= 64 or w_q.shape[1] * 2 != k:
        return False
    if m <= original.MAX_SKINNY_BATCH_SIZE and k * m <= original.LDS_CAPACITY_ELEMENTS:
        return False
    if original.gfx1151_lds_tile_eligible(
        x, w_q, w_s, w_zp, bias, group_size
    ) is not None:
        # The K-tiled LDS decode kernel covers this batch and reads the
        # original ExLlama weights, so the derived cache is not needed here.
        return False
    return (
        x.dtype == torch.bfloat16
        and x.device == w_q.device
        and x.is_contiguous()
        and not x.requires_grad
        and grouped_q.shape == (k // 128, n, 16)
        and grouped_q.dtype == torch.int32
        and grouped_s.shape == (k // 128, n)
        and grouped_s.dtype == torch.bfloat16
        and all(
            t.device == x.device
            and t.is_contiguous()
            and not t.requires_grad
            and t.data_ptr() % 16 == 0
            for t in (x, grouped_q, grouped_s)
        )
    )


def apply(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    grouped_q: torch.Tensor,
    grouped_s: torch.Tensor,
) -> torch.Tensor:
    if not can_use(x, w_q, w_s, w_zp, bias, group_size, grouped_q, grouped_s):
        return original._rdna_hybrid_w4a16_apply_impl(
            x, w_q, w_s, w_zp, bias, cu_count, group_size
        )
    m, k = x.shape
    n = w_q.shape[0]
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    group_major_w4a16_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32), 1)](
        x,
        grouped_q,
        grouped_s,
        output,
        m,
        n,
        k,
        1,
        32,
        32,
        128,
        GROUP_MAJOR=True,
        num_warps=4,
        num_stages=2,
    )
    return output


def fake(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    grouped_q: torch.Tensor,
    grouped_s: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((x.shape[0], w_q.shape[0]), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="gfx1151_w4_group_major",
    op_func=apply,
    mutates_args=[],
    fake_impl=fake,
)
